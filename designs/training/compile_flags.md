# Training — compile flags

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was
split by topic (it was 8,219 lines / 676 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.**

---

## Compiled CPU opponents — the trainer-side path was DELETED (deletion pass U3); the offline compile and its gates remain

> **What is true now.** The compile-opponents flag (and its `-preload` / `-strict` variants and generated `no-` forms), the forkserver preload (`compile_preload.py`), the parent cache prewarm (`compile_prewarm.py`) and the cross-process revert QUORUM are DELETED: under the Rust env core every policy opponent forwards through the T2 inference service, so the trainer compiles no opponent, and `SnapshotPool` no longer takes a compile argument. **`--compile-trainer` is the only compile flag** (the GPU/learner half, below). What survives is `agents.model.compile_opponents.maybe_compile_extractor(model, enabled, label, hide_cuda)` — ONE frozen model's extractor, no `strict` argument, no quorum — used by the OFFLINE readers (the prober's counterfactual view, the snapshot ladder, the eval worker, the cf producer, the search-dividend perf probe), with its decision-level parity gate (`opponent_parity.py`) and the measured speed-up floor. Everything from here down to "The recurring promotion cost" is the RECORDED design and dated measurements of the original opponent-compile path, kept as history; where a sentence names a deleted flag, module or quorum, it describes what was believed and built then.

**`--compile-opponents`** `torch.compile`s each frozen OPPONENT's feature extractor in the env workers
(pool / stable / exploiter loads, via `agents.model.snapshot.maybe_compile_extractor`). It is a
**runtime PERF knob** — never versioned, never in `check_compatible`.

🚨 **DEFAULT ON since 2026-08-17 (owner decision: the compile flags are FALLBACKS, not opt-ins).**
`--no-compile-opponents` is the way back to eager, and it takes `--compile-opponents-preload` with it
(the preload FOLLOWS this flag, so one flag turns the whole path off). `--compile-opponents-strict`
stays opt-in — default-ON is about the compile, not about the failure mode, and warn-and-fall-back
IS the fallback the default wants. **"Not inherited on resume" now cuts the other way**: a flagless
resume gets the compile ON, so it is the opt-out you must re-pass, not the flag. Pinned by
`src/main/compile_defaults_test.py` (defaults + opt-outs by value) and
`src/main/launcher/compile_flag_forwarding_test.py` (the launcher forwards all of them and owns no
default of its own).

**Why it works now when it didn't in June.** The 2026-06-30 attempt compiled only
`DamageOperator.forward` inside `policy.get_distribution` and measured **0.70× (slower)** — dynamo
overhead around a graph still running ~10k eager dispatches. Compiling the WHOLE extractor gives one
fused graph: `torch._dynamo.explain` reports **0 graph breaks / 1 graph**, and B=1 CPU
`get_distribution` goes **4.84 → 0.91 ms (5.3×)** on a real checkpoint, logits within 9.5e-7 of eager
with **0/16 argmax flips**.

**`suppress_errors` is GONE — the crash it hid was ONE op (2026-08-03).** The helper used to set
`torch._dynamo.config.suppress_errors = True` globally, because the expected-latent-defender read
(`BeliefHead.species_posterior`, then reached via `--threat-unrevealed-outgoing`) crashed the
Inductor CPU backend (`AssertionError: buf307`). That made the LITERAL production config
compile only PARTIALLY — dynamo falling back to eager per FRAME for the failing region, measured
6.48 → 1.78 ms = 3.6× — and made every OTHER backend failure in the process silent too.
`tmp/inductor_crash_repro.py` narrowed it to a single op: the softmax over species logits in the
expected-latent-defender read, which lowers to a `[B,6,n_species]` numerator + a `[B,6,1]` denominator
that the CPU scheduler asserts on while fusing. `BeliefHead.species_posterior` now spells the identical
math as `log_softmax(...).exp()`, which lowers cleanly. **The literal production arch now compiles
WHOLE with suppression OFF: 6.371 → 0.976 ms = 6.53×**, 1 graph / 0 breaks, max|Δ| vs eager 5.07e-07 —
nearly double the per-forward win, and backend failures are loud again. `tmp/softmax_variant_probe.py`
records that `.contiguous()`, `.clone()`, a 2-D reshape and a hand-rolled `exp / sum` **all still
FAIL**, so the spelling is load-bearing; it is pinned by `extractor_compiles_test.py` (default-ON,
a real compile — `GEN3AI_SKIP_COMPILE_TESTS=1` opts out; verified to FAIL if the old spelling returns).

**Measured end-to-end on the LITERAL production arch (`tmp/literal_arch_ab.sh`, 2026-08-03):**
marginal FPS **406.5 -> 541.8 = +33.3%** at `--n-envs 48`, 4 samples per arm, **ranges disjoint**
(off max 417 < on min 512), 48/48 workers compiled, 0 reverts. This arm is the one the earlier A/B
could not run: the `species_posterior` softmax used to crash Inductor, so that measurement had to
drop the expected-latent read plus the between-layers refine loop.

**READ THE TWO NUMBERS TOGETHER — the per-forward win has SATURATED.** Fixing the softmax doubled the
per-forward speedup (3.6x -> 6.53x), but end-to-end moved only 31.0% -> 33.3% (and those are
different arches, so even that 2.3pt is generous). Amdahl: the opponent forward is no longer the
rollout bottleneck. Whatever is left — obs build, protocol parse, bridge wait, the PPO update — now
dominates, so further compiler work on this path is spent effort. The next throughput lever has to
come from a different stage.

**Prior measurement, reduced arch** at the production `--n-envs 48` shape with
`--async-rollout --grad-checkpointing --self-play --self-play-use-cpu` against a seeded pool, marginal
FPS **498 → 653 = +31.0%**, 6 samples per arm, **ranges disjoint** (off max 512 < on min 614). It is
the first throughput lever here that the `SubprocVecEnv` barrier does NOT absorb — the win is *larger*
at 48 envs than at 8 (+26.6%). Adversarial checks: the compiled path is the real `pool:snapshot_*`
opponent, and `ep_len_mean` is unchanged (47.4 vs 45.9), so it is not an artifact of shorter battles.

Three properties make it cheap: the compile is keyed on the CODE OBJECT (a second extractor instance
in the same process compiles in **0.00 s**, so pool promotions are free), parameters are graph INPUTS
(a different checkpoint's `load_state_dict` does NOT recompile), and an on-disk
`TORCHINDUCTOR_CACHE_DIR` shared by the RUN's process tree turns each worker's cold codegen into a
cache hit. Since K3 that dir is the run's own `<run>/compile_cache/inductor`, never a box-wide one
("The HERMETIC per-run compile cache" below).

Four guards, each protecting against a failure that actually happened while building it:
- **CUDA-context OOM.** Compiling even a CPU model in a CUDA-visible process initialises CUDA and takes
  ~252 MiB of card; ×48 workers is the June OOM. The helper sets `CUDA_VISIBLE_DEVICES=""` — but only
  when the caller passes **`hide_cuda=True`**. That used to be INFERRED from
  `torch.cuda.is_initialized()` as a proxy for "am I an env worker", which was correct only by accident
  of the call sites: the first main-process caller would have silently blinded the learner's GPU. It is
  now the caller's explicit declaration (all three training sites are env workers → `True`), and a
  caller that declares `hide_cuda=True` in a process that already holds a context is REFUSED rather
  than quietly compiled. Verified live: 48 compiled workers, exactly ONE context (the learner).
- **A compile that LOSES.** June measured 0.70× (dynamo overhead > fusion win on a fragmented graph),
  so the helper **times eager vs compiled at load and REVERTS** below a 1.05× floor. This used to be
  load-bearing because `suppress_errors` made a failed compile silent; with suppression gone a failure
  raises and is caught, and this is now a second line of defence against a merely-fragmented graph.
  **The floor value is unchanged; the MEASUREMENT under it was rebuilt 2026-08-24 — see below.**
- **A LATE failure.** `torch.compile` guards on input properties, so an unseen shape can trigger a
  fresh trace at CALL time, long after load. `_eager_fallback_on_error` wraps the compiled callable so
  that degrades THIS opponent to eager (and says so) instead of killing a 3-hour run. This is the
  scoped replacement for global `suppress_errors`: same never-crash property, one model, and loud.
- **Resume safety.** It patches the BOUND `fe.forward`, never the module — `torch.compile(module)`
  would prefix every state_dict key with `_orig_mod.`.

**Per-worker startup cost, measured (`tmp/compile_spawn_cost.py`, 16 workers, 16-core box).** Wall
clock until all workers are ready: **private cache per worker 163.4 s / cold shared cache 59.6 s /
warm shared cache 30.1 s.** So `TORCHINDUCTOR_CACHE_DIR` is not a nicety — without it the startup cost
nearly triples. The residual ~30 s is dynamo tracing + guard construction, which the on-disk cache
cannot remove, and it is paid once per launcher restart (every 3 h).

**Warm the Inductor cache in the parent — `agents.model.compile_prewarm` (BUILT).** Each env worker
compiles its own frozen opponent. `train_rl_agent` calls `prewarm_extractor_compile(...)` before the
vec env exists, so the workers hit a WARM shared on-disk cache instead of racing on a cold one:
**59.6 s -> 30.1 s** wall for 16 workers (`tmp/compile_spawn_cost.py`; a private cache per worker is
163.4 s, so `TORCHINDUCTOR_CACHE_DIR` is load-bearing). It builds the extractor from
`build_extractor_arch_kwargs(args)` — the same table the real model uses — so the cached codegen is
keyed to the graph the workers actually run; weights are graph INPUTS, not baked constants, so a
fresh random extractor warms the cache for every opponent checkpoint.

**THE FORKSERVER PRELOAD WORKS NOW (`--compile-opponents-preload`, `gen3_forkserver_preload_v1`,
2026-08-16) — and the fix was one level deeper than the plan.** SB3's `SubprocVecEnv` uses
`mp.get_context("forkserver")`, and a forkserver child inherits memory copy-on-write, so
`agents.model.compile_preload` (armed via `set_forkserver_preload`) compiles the extractor ONCE in
the forkserver and every worker inherits the traced graph (~0.12 s vs ~30 s per worker). The 2026-08
attempt at exactly this **wedged a real 48-env run** — 2 workers forked instead of 48, parent blocked
in `unix_stream_data_wait`, box at 0.2 load, no error anywhere — because `fork()` copies every mutex
but only the calling thread, and importing the extractor started poke-env's GLOBAL asyncio loop
thread: any `poke_env.x` import executed the eager package `__init__` → `player` → `ps_client` →
`concurrency`. The planned fix was a ~12-file model-layer refactor; the shipped fix is at the ROOT
instead — **`poke_env/__init__.py`, `poke_env/player/__init__.py` and `poke_env/battle/__init__.py`
are LAZY (PEP 562)**, so the enum/data/battle subtrees the extractor needs are thread-free, the
public surface is unchanged, and the loop thread starts exactly when a player/client module is
imported (what every training-side consumer does anyway). The laziness also dissolved an
order-dependent `battle ↔ player.battle_order` circular import the eager inits had been masking.

Three guards, all loud:
- `compile_prewarm.extractor_import_is_fork_safe()` is the executable invariant (import ⇒
  single-threaded), pinned by `compile_prewarm_test.py` — if the lazy init regresses, the suite
  fails before any run arms the preload.
- The preload pins `torch._inductor.config.compile_threads = 1` (the codegen pool never exists) and
  calls `shutdown_compile_workers()` anyway.
- After its compile the preload asserts `threading.active_count() == 1` and **RAISES otherwise**,
  killing the forkserver bootstrap so `SubprocVecEnv` construction fails with a traceback in the
  parent — the silent wedge is unrepresentable, not just unlikely.

Proven live 2026-08-16: a real 4-worker `SubprocVecEnv` CPU run with the preload armed compiled once
(41 s), forked all workers, trained to completion. When armed it REPLACES the in-trainer cache
prewarm (the forkserver compile populates the same on-disk cache, which the Popen'd eval workers
still hit). Honest sizing unchanged: all-workers-ready improves ~30 s → ~20 s at 16 workers (maybe
~75 s → ~25 s at 48), ~50 s per 3 h restart — the reason to have it is that the architecture now
permits it and the guard structure makes it safe, not throughput.

**DEFAULT since 2026-08-17: it FOLLOWS `--compile-opponents`** (tri-state — `None` = unset ⇒ follow),
so both ship on and `--no-compile-opponents` turns the pair off in one flag;
`--no-compile-opponents-preload` keeps the per-worker compile and reverts to the cache prewarm. The
"requires `--compile-opponents`" error now fires only on an EXPLICIT preload beside an explicitly-off
opponent compile — erroring on the pairing the DEFAULTS produce would have made
`--no-compile-opponents` itself a usage error, which is the regression
`compile_defaults_test.py::test_no_compile_opponents_alone_is_not_a_usage_error` exists to catch.

**What justifies defaulting the thing whose predecessor hung a run**: the predecessor's cause is
fixed at the ROOT (lazy `poke_env` inits ⇒ a thread-free extractor import, pinned by
`compile_prewarm_test.py`), and the failure MODE is inverted — a preload that cannot prove
single-threadedness RAISES during forkserver bootstrap, so `SubprocVecEnv` construction dies with a
traceback in the parent instead of wedging 2 of 48 workers in silence. A loud startup failure with a
one-flag opt-out is a defensible default; a silent 13-hour stall would not have been.
**The 48-worker FORK STORM is now measured** (2026-08-17, `tmp/preload_fork_probe.py`, CPU, beside a
live run): arm the preload, then `forkserver` `Pool(48)` — **48/48 workers forked, 41 distinct pids
took a task, 48/48 reported the compiled graph present in inherited memory**, 19.7 s wall after an
11 s preload compile. That is the exact mechanism that wedged at 2-of-48 before, so the count-specific
fear is addressed directly rather than by extrapolation from the 4-worker run.
**⚠️ What is STILL untested is the full 48-env TRAINING composition** — real `Gen3Env` workers,
bridge children, a mid-run pool promotion — not the fork itself. If it ever refuses, the message
names the surviving thread and `--no-compile-opponents-preload` is the immediate way past it.

**Its fail-loud path was also observed, by accident.** A malformed `GEN3AI_PRELOAD_ARCH` during that
probe made the preload's extractor construction raise: the child's traceback printed in full and the
parent died on `EOFError: unexpected EOF` out of `forkserver.read_signed`. Loud, immediate, no wedge
— but note the PARENT-side exception is not self-describing, so **the diagnosis is in the child's
stderr**, which under the launcher lands in `launcher_child.log`.

**Failure is LOUD (`--compile-opponents-strict`).** Falling back to eager is a ~6.5× regression on the
opponent forward that is otherwise invisible — the run just produces fewer steps/hour forever and
looks healthy. Every failure path (`DISABLED`, `REVERTED`, mid-run `FELL BACK`, mis-declared
`hide_cuda`) goes through `_compile_warn`: stderr **and** the launcher event stream, so it surfaces in
the TUI. `--compile-opponents-strict` promotes them to a `CompileExtractorError` for anyone who would
rather fail at startup than find it in the FPS graph a day later — **but a below-floor TIMING verdict
is promoted only on a quorum**, for the reason immediately below.

### The opponents' decision-level parity check (`gen3_opponent_compile_parity_v1`, 2026-09-29)

Until 2026-09-29 a compiled opponent was checked for SPEED only, on an all-zero observation. Nothing
compared its decisions with eager, so a CPU Inductor miscompile would have handed the learner a wrong
self-play / eval / teacher opponent with every gate green. `maybe_compile_extractor` now runs
`agents.model.opponent_parity.check_opponent_parity` after forcing the compile and before installing
it. The check compares eager with the compiled callable on features, MASKED legal log-probs and V,
at the learner gate's bars (`compile_trainer._FP32_TOL` via `decision_verdicts`). It uses 16 rows of
the committed real-obs fixture, each fed at **B=1** with the float `action_mask` and every declared
extra key. That is the warm-up's exact key set, dtypes and shape, so the check judges the graph that
plays and triggers no recompile. **On fresh weights** both arms also run on the seeded in-place
perturbation (`parity_probe.perturbed_parameters`), judged with the vacuity guard ON, exactly as the
learner gate does.

- **It RAISES the learner gate's typed error** (`CompileTrainerError`, or `VacuousCompileParityError`)
  whatever `--compile-opponents-strict` says, after uninstalling the compile. A timing miss costs
  throughput. A parity miss is a WRONG opponent (GIGO), and a CPU miscompile is systemic, since the
  same code object runs in every worker. Any other exception from the check (a backend crash on the
  real rows) takes the usual warn / fall-back / strict path. A stand-in policy without the Gen3 heads
  is refused rather than installed unvalidated.
- **Cost, measured** (CPU, one thread, torch 2.5.1, 2026-09-29): **~0.28 s on trained weights**
  (`ai_v14_01_base/final_model.zip`: features 1.5e-05, legal log-prob 4.3e-06, V 2.4e-07, all PASS)
  and **~0.55 s on fresh ones**, where the perturbed pass doubles it. For comparison, the first
  compile of a process took 106 s with a cold cache and ~6 s warm, and a reused-compile load took
  ~0.3 s.
- **Cadence: once per DISTINCT WEIGHTS per process, never per game.** A consumer that
  loads a model per game or per iteration would re-pay it each time, so a PASS
  is cached process-locally under a blake2b fingerprint of the policy's `state_dict` bytes plus
  every submodule's train/eval mode (`weights_fingerprint`, ~18 ms for the 7.0M-parameter policy).
  The compiled callable is shared per process by dynamo's code-object cache and the parameters are
  graph inputs, so the same weights through the same graph cannot give a different verdict. A
  changed weight is a new key and is checked again. A failure is never cached. The PASS line (with
  its ms, or `[cached: …]`) is printed on every load.
- **Pinned by** `agents/model/opponent_parity_test.py`, on the real production policy with
  `torch.compile` stubbed. A pointer-only miscompile (move cells read only by the pointer head) is
  refused on fresh and on informative weights, under strict and under non-strict. With
  `compile_opponents.py` reverted to its pre-check version the same test FAILS (`DID NOT RAISE`), and
  `test_REVERT_…` pins that the path installs the miscompile once the seam is neutralised.
  `compile_extractor_test` stubs the seam, because its stand-ins are 8 wide and have no heads.

### 🧯 The floor's MEASUREMENT was broken, and the fix is under the gate, not on it (2026-08-24)

**The gate killed three production launches on timing noise, and it was uninformative in BOTH
directions.** Measured on the same checkpoint, the same box, `--n-envs 48`: the **eager arm alone
spread 7.7×** (14.94–115.71 ms) and the compiled arm 2.08–17.90 ms. The old gate compared ONE eager
aggregate to ONE compiled aggregate, so the verdict was decided by which end of each spread the two
arms landed on — the same checkpoint that scored **0.78× (FATAL under strict)** scored **6.3× median
across 48/48 workers** minutes later, and one failure landed at **exactly 1.05×**, the boundary tell
that should have ended the debugging. The false-PASS direction is equally live: a cold-measured eager
arm lets a genuinely broken compile read 29×, so "0 workers below the floor" was never evidence of
health.

⚠️ **Two beliefs from that week are RETRACTED. Do not re-derive a plan from either.** "~half the
workers land under the floor for a frozen fork of the current net, so compiling this class buys ~5%"
was **one noisy pair**, not an opponent-class fact — the target class was never the problem. And the
error text's "the graph is probably fragmented" asserted a cause **a ratio of two timings cannot
distinguish from a busy box**; it sent three separate investigations after the wrong thing. Dropping
the floor to 0.7× was proposed and **rejected**: widening a broken instrument buys a confidently
wrong answer in the other direction.

**What shipped instead (all four composed — `agents/model/compile_opponents.py`; the quorum and strict-verdict rows were DELETED in U3, the median-of-alternated-arms measurement and the floor stay):**

| | old | now |
|---|---|---|
| aggregation | one min-of-12 per arm | **median of 5 samples**, each a min-of-4 |
| ordering | eager block, then compiled block | **alternated** sample-by-sample, round order flipping |
| warm-up | 3 calls *inside* each arm's own timing, eager measured cold-first | **both arms warmed identically before EITHER is timed** (`_warm_arm`) |
| strict verdict | any one worker below the floor is fatal | **quorum**: warn always; fatal only if **>25%** of the reporting compiles reverted, and never on the first 4 |
| message | asserts a cause | prints **both arms' full sample series, both medians, the ratio, the floor, and the running quorum** |

Alternation is the load-bearing one, and it is why this is a *measurement* fix rather than a
tolerance: this box normally carries a trainer and 47 sibling workers each running a ~30 s compile,
so the regime **drifts across the measurement window**. Back-to-back arms charge that drift entirely
to whichever arm ran during it; interleaving charges it to both.

**The quorum is cross-process and its shape is a deliberate compromise.** `arm_compile_quorum(run_dir)`
is called once in `train_rl_agent` before the vec env exists; it clears and publishes
`<run_dir>/.compile_quorum` in `GEN3AI_COMPILE_QUORUM_DIR`, which every `SubprocVecEnv` worker,
forkserver child and Popen'd eval worker inherits. Each verdict is one empty file (`<pid>-<ns>.ok` /
`.revert`) — create-and-count, no lock, no server. ⚠️ **It is a PREFIX estimate, stated rather than
hidden**: a worker sees only the verdicts written before it looked, so an isolated bad reading can
never be fatal (1 of a growing denominator, and the first 4 decide nothing) while a systemic failure
trips as soon as enough workers agree. Within a restart window the tally also spans the whole process
tree, so a healthy startup dilutes a later mid-run regression. A real barrier across 48 spawned
workers is cross-process plumbing this perf knob does not justify. A **compile that ERRORS** is
unaffected — that is a fact, not a reading, and stays fatal in its own process immediately.

**One honest residual, pinned by a test rather than glossed** (`test_the_residual_drift_BIAS_is_bounded_but_real`):
alternation cancels most of a drifting regime but not all of it — each arm's five samples sit at
slightly different moments, leaving a bias of order `drift^0.1` (~1.5× under a hostile 64× drift).
That is enough to carry a *marginally* losing 0.70× compile up to the floor in ~2% of draws (98%
still revert, median reading 0.72×) — which is precisely why a below-floor reading is the quorum's
business and not one worker's. A compile losing by a real margin (0.40×) survives no drift the
regime produces (max reading 0.62×).

The warm-up obs now also carries **`action_mask` as float32** alongside `observation`: dynamo guards
on a dict's KEY SET and on dtype as hard as it guards on shape, and every real opponent call arrives
through `policy.get_distribution` with both keys. Warming with one key left the first LIVE decision
to re-trace the whole extractor — **19.5 s against a 3.8 ms steady state**, measured in the cf
producer (53870dd).

**Verification.** `compile_extractor_test.py::TestTheProductionRegime` feeds the recorded regime
through both decision logics via a drift model that reproduces **both** recorded extremes without
being fitted to them (0.77× and 51× against the real 0.78× and 47.8×): the old logic's verdict flips
run-to-run, the new one does not, and the ratio spread collapses by >10×. Revert-verified — restoring
the back-to-back design fails 7 tests, and making the below-floor verdict per-worker-fatal fails 6.
Measured live on this box (nice'd, beside the live run, real `ai_v9_34_tick1_0824` checkpoint, 3
repetitions): **7.52× median, range 7.45–7.56×, spread 1.015×**, eager ~14.9 ms / compiled ~2.0 ms.
Note the old design also reads stably *there* — a single warm process is not the regime that breaks
it, which is exactly why the synthetic-regime test exists.

**Caught at CODE time — and for all FOUR compile targets, not just this one.**
`src/agents/model/extractor_compiles_test.py` owns the device x grad matrix, because Inductor's CPU
backend emits C++ and its CUDA backend emits **Triton** — different lowering paths with different
bugs, so a green CPU-forward test is not evidence about any other cell:

| | forward | forward + backward |
|---|---|---|
| **CPU** | ✅ the frozen self-play OPPONENT (this section) | ❌ **does not lower** — but only in ONE of Inductor's three C++ store kernels: `CppKernel`/`CppVecKernel` both emit `atomic_add`, while `CppTile2DKernel` (the transposed variant, chosen by index LAYOUT) carries `assert mode is None`. CONFIG-CONDITIONAL too: the scatter is a gather's backward, so `--belief-grad-mode label_only` (stop-grad belief publication) deletes it and the backward then compiles (bisected 2026-08-15) |
| **CUDA** | ✅ eval / inference on the card | ✅ the TRAINER's step — **155.1 → 88.5 ms** fwd+bwd at batch 4096 (**1.75x**); provenance in the measurement table below. **SHIPPED as `--compile-trainer`, and it DEFAULTS ON when the resolved device is cuda** |

The ❌ cell is a **limitation PIN** (`test_cpu_backward_still_does_not_compile`) and it FAILS IF THE
LIMITATION LIFTS — three things assume it holds, starting with `maybe_compile_extractor` routing
every grad-enabled call to eager. It matches the TRACEBACK, not the message: torch raises a bare
unannotated `AssertionError` whose `str()` is empty, so `str(exc)` has nothing to match on.

Each compile cell runs **by default** (~10 s each on a warm cache; `GEN3AI_SKIP_COMPILE_TESTS=1`
opts out), so "the model stopped compiling" fails the suite instead of silently costing throughput.

⚠️ **The CUDA cells SKIP under a normal `pytest` run** — the root `conftest.py` hard-sets
`CUDA_VISIBLE_DEVICES=""` for the whole suite so a stray `device="auto"` can never steal VRAM from
a live training run. **You cannot compile FOR cuda ON the cpu** (measured 2026-08-14, torch 2.5.1 /
triton 3.1.0: with the device hidden, an Inductor cuda compile dies `RuntimeError: No CUDA GPUs are
available` — the backend queries live device properties, so codegen is not a blind AOT
source→PTX step; and a `FakeTensorMode` trace only exercises **dynamo**, which is device-agnostic
anyway and never reaches the backend where the device-specific bugs live). So the CUDA cells need
the real card:

```bash
GEN3AI_TEST_ALLOW_GPU=1 pytest src/agents/model/extractor_compiles_test.py -q   # 8 passed
```

Even unhidden they refuse to run when the card is BUSY (a free-VRAM floor read via `nvidia-smi`, so
the *check* creates no CUDA context either) — a compile test must never be what OOMs a 20-hour run.
Every skip NAMES the cause and the knob rather than saying "no CUDA device", because a silent skip
on a box that is always training would turn the gate into a no-op that still reads green.

### The recurring promotion cost — measured, and it is SMALL (~2.7%)

> *History: measured on the trainer-side opponent compile and its preload, both deleted in U3.*

Everything above sizes **startup**. There is a second bill during the run, when a self-play
promotion makes env workers compile the new opponent. Measured on gen-14, 2026-08-17
(`designs/research_state/measurements/gen14_pool_refresh_compile_cost.json`, n=2 events):

| event | excess over the 138.6 s baseline | compiles | path |
|---|---|---|---|
| iteration 22 | **+1095 s** | 48 | all *timed* — each process's FIRST compile |
| iteration 42 | **+77 s** | 27 of 48 | all *"reused this process's validated compile"* |

**Read the second row, not the first.** Iteration 22 is not a promotion in the steady-state sense —
it is where **self-play first activates**: the pool is seeded from empty, so all 48 workers at once
load a 41 MB checkpoint *and* pay their process's first compile (the `revalidate` branch, which also
times eager-vs-compiled). It happens once per run. The recurring cost is **+77 s per promotion ≈
2.7% of wall-clock ≈ 16 min over a 25M run**, and `--compile-opponents` is net **+40%**.

**The caches work.** The Inductor cache (then box-wide; the run's own since K3) is HIT at a promotion (13 files written, vs 6600+ at
run startup), `SnapshotPool._model_cache` keeps one compile per worker per snapshot, and
`_COMPILE_VALIDATED` puts every compile after a process's first on the cheap path. Nothing here
needs fixing.

**The one-time event IS addressable, and the flag for it is now ON BY DEFAULT — `--compile-opponents-preload`.**
The +1095 s is 48 workers each paying their process's FIRST compile, and fork-inheritance is exactly
the thing that removes it: the preload compiles once in the forkserver and workers inherit the traced
graph copy-on-write (**0.12 s per worker vs ~30 s**). Note the on-disk Inductor cache and the fork
inheritance fix DIFFERENT halves — the disk cache removes codegen, the fork removes per-process
dynamo tracing and guard construction, which is the half that was left.

Why the cost landed at iteration 22 rather than at worker startup: **the pool is empty until the
first promotion**, so workers have nothing to compile when they fork, and their first compile is
deferred to the moment self-play activates.

Two limits, unchanged by the default flip — expect a SHRUNK event, not a gone one:
- It SHRINKS the event, it does not remove it — those 48 workers also each load a 41 MB checkpoint
  (`load_model_snapshot` → deserialize → build policy), which no compile flag touches.
- The 0.12 s figure is a standalone probe of STARTUP compiles, and the flag's live proof is a
  **4-worker** run. A snapshot extractor compiled 2M steps AFTER the fork should still hit the
  inherited dynamo state (same `forward` code object, same shapes) but that case is not directly
  measured. And the hang this flag's predecessor caused was specifically at **48 workers** — it is
  fail-loud now (it RAISES rather than wedging), so the risk is a loud crash at construction, not a
  silent 13 h stall, but **48 envs is still untested for the fixed version**, and defaulting it on
  is what schedules that test for the next fresh launch. **Do not retrofit it onto a LIVE run**
  (a launcher-pinned worktree keeps its own code, so a live run does not pick this up); if a fresh
  48-env launch refuses at construction, the message names the surviving thread and
  `--no-compile-opponents-preload` is the immediate way past it.

⚠️ **A `[SELFPLAY EVAL] … [Ns]` line beside a slow iteration is NOT its cause — eval is genuinely
non-blocking.** gen-13 ran an **1865 s** eval cycle inside a **395 s** iteration. Attributing
iteration cost to an overlapping eval (or vice versa) is a window coincidence; separate them by the
compile path (`timed` vs `reused`), which is what actually distinguishes the expensive event.
## The HERMETIC per-run compile cache (K3, `gen3_hermetic_compile_cache_v1`, 2026-09-30)

**The owner (2026-09-29): "always initialize from a fresh setup — who knows how many silent errors a
shared cache allows."** Until K3 the learner's Inductor/Triton caches were UNMANAGED: one box-wide
`/tmp/torchinductor_goodlad` (12 GB on tmpfs, i.e. RAM, on 2026-09-30) used by every run, pin, torch
env, benchmark and test, and the opponents' CPU compile in a second box-wide dir
(`/tmp/gen3ai_inductor_cache`, 142 MB). The K1b fault (below) proved a cache KEY can omit a setting that
changes the artifact, so the rule does not trust the key. `agents.model.compile_cache` is the one
module; every compile site calls it.

| process | its compile cache | reused when |
|---|---|---|
| a TRAINING RUN whose code checkout is CLEAN (since 2026-10-10, the default — "The SHARED compile cache" below) | the SHARED entry for its key, `~/.cache/gen3ai/compile_shared/<key>/{inductor,triton}`; T2's AOT packages in `<run>/compile_cache/t2_aot/` | by ANY run with the same key (K3's stamp + the platform + the torch env + the run's compile surface) |
| a TRAINING RUN on a DIRTY tree, an unnameable commit, or opted out (`GEN3AI_SHARED_COMPILE_CACHE=0`) (the trainer, its env core, eval subprocess, T2 service — every child inherits) | `<run>/compile_cache/{inductor,triton,t2_aot}` — `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `GEN3AI_COMPILE_CACHE_DIR` | only by the run's OWN restart (`--model` = a checkpoint this run wrote, `fork_lr.is_same_run_checkpoint`) AND a matching `stamp.json`; a fresh launch or a fork WIPES it to EMPTY |
| a pytest process (controller and each xdist worker) | a fresh `gen3ai_pytest_compile_<pid>_*` dir under the SCRATCH root (root `conftest.py`) | within that process's session only; deleted at unconfigure |
| anything else that compiles (a benchmark, the prober, an offline meter, a standalone eval) | a fresh private `gen3ai_compile_<pid>_*` dir under the SCRATCH root (`ensure_hermetic_cache`) | never; deleted when its creator exits |

**The SCRATCH root is the REAL DISK: `$GEN3AI_SCRATCH`, else `~/.cache/gen3ai/tmp` — never `/tmp`**
(`utils.scratch.scratch_root` — the ONE helper, which the root `conftest.py` also loads, by path, for pytest's own temp root — and which REFUSES a tmpfs/ramfs root). `/tmp` on this box is tmpfs: RAM
(45 GB, shared with every process), and a 1,048,576-inode cap. A compile cache is thousands of files
and up to 1.5 GB (torch 2.8). The benchmark's cold/warm roots and Lane E's replay-child cache live
there too.

**Does the disk cost startup time vs tmpfs? No detectable difference (measured 2026-09-30, the owner's
question).** `compile_cache_benchmark --root-bases ~/.cache/gen3ai/tmp,/dev/shm` (ext4 NVMe vs tmpfs;
the explicit benchmark override — the production path refuses tmpfs), torch 2.5.1, one fresh root per
base, COLD on each then two WARM reads INTERLEAVED, load 2.5–17 (`m5_k3/storage_ssd_vs_shm_stable_cuda.json`):

| part | COLD ext4 / tmpfs (s) | WARM ext4 / tmpfs (s) | 2nd WARM ext4 / tmpfs (s) |
|---|---|---|---|
| learner (gate + prewarm) | 810 / 770 | 254 / 247 | **196 / 233** |
| T2 startup | **240 / 273** | 78 / 75 | 90 / 84 |

The sign flips between reps and every delta (4–40 s) is inside the rep-to-rep spread of the same arm
(learner warm 196–254 s on ext4 alone) and tracks the load average, not the medium. As expected:
cold startup is CPU-bound codegen + the C++/Triton compilers; warm reads of just-written files are
served by the page cache; and steady-state training never reads the on-disk cache at all (compiled
kernels live in process memory). n = 1 round per part: a difference below ~10% is NOT excluded.

- **THE STAMP** (`<run>/compile_cache/stamp.json`): the code's commit (`utils.git.get_git_hash`, i.e. a
  pinned child's PIN) + whether `src/` is dirty + `torch.__version__` + the interpreter +
  `compile_config.config_row_hash()` (SHA256 of the `COMPILE_CONFIG` row — data only, in its own module so
  stamping a cache never imports the sentinel; P10-D). Any field that differs
  WIPES; a DIRTY or unnamed tree NEVER reuses (a sha cannot name it); an unreadable stamp WIPES. The
  decision is printed at startup: `🧊 [CompileCache] CREATED | WIPED | REUSED <root> — <reason>`.
- **WHERE IT IS DECLARED.** `main.train.lifecycle._declare_compile_cache`, called in
  `train_rl_agent.main` the moment `model_dir` exists — before the vec env, T2 and `build_and_train`. A shared value already in force at that
  point (torch exports its default on a process's FIRST compile) means something compiled first: a
  typed `CompileCacheError` → `FATAL_CONFIG`.
- **A torch default is never honoured.** `ensure_hermetic_cache` treats `/tmp/torchinductor_<user>`
  and the retired `/tmp/gen3ai_inductor_cache` as undeclared. T2's AOT packages now go under the root
  in force (`compile_cache.t2_aot_dir`), not a stray `/tmp/t2_aot_*` that outlived its process.
- **The parity gates are unchanged and still run at EVERY start** — the real-obs learner gate, T2's
  startup gate and the opponents' decision-level check verify OUTPUTS whatever was loaded; the cache
  decides only what may be loaded.
- **Pinned by** `agents/model/compile_cache_test.py` (each test fails on a revert of the rule it names:
  fresh/fork wipe, restart reuse, every stamp field, dirty never reuses, a real `torch.compile` lands
  in the run's dir, a private dir deleted at exit, shared values refused, the trainer's call precedes
  every compile/spawn site, every production `torch.compile` / `aoti_compile_and_package` module
  calls the cache helper, no string literal names a box-wide dir) and `compile_extractor_test`.
- **A crash loop on a corrupt cache** (not observed): delete `<run>/compile_cache/`; the next start is
  cold and correct.
- **Temp caches never outlive their owner (`/tmp` is tmpfs: RAM AND a 1M-inode cap — 2026-09-30 the
  box ran OUT OF INODES, 1,047,791 / 1,048,576, and a routine gate died with 40 `could not create
  numbered dir` errors; the box-wide Inductor dir alone held 395k files).** Every temp cache is named
  `gen3ai_compile_<pid>_*` / `gen3ai_pytest_compile_<pid>_*` under the scratch root; its creator deletes it (`atexit`;
  pytest at `pytest_unconfigure`, which runs after a FAILING session too, with `atexit` as backstop),
  and every new declaration SWEEPS the ones whose creator PID is dead — so a SIGKILL leaks only until
  the next compile anywhere on the box. Lane E's replay child now gets a `TemporaryDirectory` cache
  (it leaked two `mkdtemp`s per gate run, ~2 GB by 2026-09-30). Pinned by
  `compile_cache_test::test_a_pytest_session_removes_its_cache_even_when_it_fails` (a real child
  session, passing and failing) and `::test_a_SIGKILLED_owner_s_cache_is_swept_by_the_next_declaration`.
- 🚨 **A COMPILE-PATH CHANGE RUNS THE COMPILE FILES' `slow` TIER UNDER TORCH 2.8 BEFORE SHIPPING**
  (orchestrator, 2026-09-30; **the BOTH-torches half RETIRED the same day** — owner: torch 2.8 is the
  default, new code targets 2.8 only, 2.5.1 is a legacy-resume env whose version-keyed branches wait
  for the deletion pass — so `gen3ai_stable` is no longer a required run; the text below records the
  rule as it was set). K3 made the first CPU codegen of the production extractor in each pytest
  process COLD (70–90 s), so the codegen-numerics tests of `extractor_compiles_test`
  (`…_without_suppression`, `test_cpu_backward_still_does_not_compile`) are `slow`; the one-graph
  `explain` tests stay routine. So: `python3 -m pytest src/agents/model/extractor_compiles_test.py
  src/agents/model/compile_control_test.py
  src/agents/inference/service/service_cuda_test.py -m slow` under `gen3ai_torch28` (until
  2026-09-30 also `gen3ai_stable`; the CUDA file under `scripts/ops/gpu_lock.sh`); the rows land in
  `designs/ops/slow_tier_status.json`, which the routine gate reads. Production's guard is unchanged:
  the real-obs parity gate runs at every start. **First finding of that rule (2026-09-30): on torch
  2.8.0+cu126 the CPU forward+BACKWARD of the production extractor now LOWERS** (the 2.5.1
  `atomic_add` scatter refusal is gone); `test_cpu_backward_still_does_not_compile` is now keyed per
  torch (`_CPU_BACKWARD_LOWERS`). `maybe_compile_extractor`'s eager route for grad-enabled calls
  stays load-bearing while 2.5.1 is supported; on 2.8 the compiled opponent artifact is no longer
  inference-only by necessity (UNVERIFIED numerically — the pin checks that it lowers and runs).

**COST — cold vs warm, measured (`src/agents/model/compile_cache_benchmark.py`, 2026-09-30, RTX 3080 Ti,
`gen3ai_stable` torch 2.5.1, a perturbed fresh production-arch policy, each number a FRESH process;
COLD = the first process on an empty root = a fresh launch, WARM = the next process on it = the run's
own restart; load average ~7–9 on 16 cores from peer agents' test gates, so absolute numbers are
contended — REPORTED, not rescaled):**

| part (a fresh process each) | COLD, round 0 / 1 (s) | WARM, round 0 / 1 (s) | 2nd WARM, round 0 / 1 (s) | cache |
|---|---|---|---|---|
| **learner** — the real-obs parity gate + the sentinel's prewarm (n_envs 48, micro-batch 2048, half-batch) | **681 / 746** (gate 242 / 252 + prewarm 440 / 494) | **229 / 251** (gate 63 / 72 + prewarm 165 / 179) | 188 / 262 | 466 MB |
| **T2** — `InferenceService.startup()`, 1 trainee + 20 pool slots, buckets (8, 48), 8 lanes, `graph` | **202 / 252** | **57 / 79** | 56 / 81 | 116 MB |

Load average at each phase's start: round 0 1.3–4.5, round 1 2.2–10.9 (the round-1 warm reads are the
contended ones); no other GPU process. Raw rows: `designs/research_state/measurements/m5_k3/`. **What
a restart saves: ~430–500 s of learner startup and ~145–175 s of T2 startup — WARM ≈ 0.3× COLD for
both.** T2's cold read sits at the low end of Lane G's 205–430 s with fresh caches (**UNVERIFIED** why:
Lane G's slot/bucket shapes and box load differ and are not re-measured here). torch 2.8 (`gen3ai_torch28`): **learner 394 s cold → 117 / 134 s warm (cache 1.5 GB — 3.2× 2.5.1's); T2 172 s cold → 32 / 33 s warm (382 MB)** (one round, 2026-09-30, same shapes, load not re-recorded per phase; `m5_k3/cold_warm_torch28_cuda.json`). A 2.8 run's cache is ~2 GB on disk.

The warm residual is dynamo tracing + guard construction + CUDA-graph capture + the parity gates,
which no on-disk cache removes. Per-run disk: ~0.6 GB (learner 466 MB + T2 116 MB + opponents 13 MB) on the run's own disk, not tmpfs — `models/` retention may delete `compile_cache/` freely (it is regenerable; the next start is cold). The CPU opponents' compile in one env worker
(`maybe_compile_extractor`, CUDA hidden, same load): **93.5 s cold → 26.6 s warm** (13 MB). (Until U3 the parent prewarm / forkserver preload paid it once at a fresh launch; neither exists now.) What K3 changes is only what a FRESH launch pays: the old box-wide cache was usually warm for a
new run of an unchanged graph; the run's own cache is always cold at a fresh launch, by design.

### The SHARED compile cache, by KEY (`gen3_shared_compile_cache_v1`, T25, 2026-10-10)

**The owner (2026-10-10), verbatim: "Shared cache is fine as long as we check the git hash and the
workspace is clean."** This relaxes the 2026-09-29 "always initialize from a fresh setup" rule above
for TRAINING RUNS only, under exactly those two conditions; K3's per-run cache stays the fallback, and
the pytest / private temp caches are unchanged. Module: `agents.model.compile_cache_shared`, called
from `compile_cache.prepare_run_cache` (the trainer's `_declare_compile_cache`, unchanged position).
Pinned by `src/agents/model/compile_cache_shared_test.py`; the root `conftest.py` sets
`GEN3AI_SHARED_COMPILE_CACHE=0` for every test so no test touches the operator's shared root.

**THE KEY** — the entry dir is the first 32 hex of the SHA256 of these fields; the FULL fields are
written into the entry's `stamp.json` at creation and compared on every open, so a mismatch is a
MISS (the per-run cache), never a hit:

| field | why it is in the key |
|---|---|
| K3's per-run stamp, verbatim (code commit, `code_dirty`, torch, the interpreter, `config_row_hash()`) | every component of the per-run stamp is kept; the CONFIG ROW is the K1b protection — the settings the adapter pins that Inductor's own key omitted |
| the commit again, `git rev-parse HEAD` in the checkout the code was IMPORTED from (`utils.paths.repo_root()`), which must agree with `$LAUNCHER_GIT_HASH` when set | a PINNED child imports its pin's worktree (the launcher's `PYTHONPATH`) while standing in main, so this is the PIN commit; the launcher already passes the pin in `$LAUNCHER_GIT_HASH` (no launcher change), and a disagreement does not share |
| python's full version, triton's version, torch's CUDA version, each GPU's name + compute capability + driver (`nvidia-smi`, no CUDA context), `$CUDA_VISIBLE_DEVICES` | the artifacts are kernels for one toolchain and one card; an unreadable GPU does not share |
| every `TORCHINDUCTOR_*` / `TORCHDYNAMO_*` / `TORCH_COMPILE*` / `TRITON_*` / `PYTORCH_*` / `TORCH_CUDA_ARCH_LIST` / `INDUCTOR_*` variable (the two cache-dir variables excluded) | an environment knob can change codegen without changing the graph — the K1b failure class |
| the run's compile surface: `extractor_arch.arch_toggles_from_args(args)` (every version-checked architecture toggle) + `COMPILE_ARG_KEYS` (`compile_trainer`, `device`, `device_batch`, `batch_size`, `grad_checkpointing`, `learner_cudagraphs`, `t2_backend`, `t2_buckets`, `t2_lanes`, `t2_opponent_bucket_cap`) | the model config and the flags that shape a compile; a new trainer flag whose name says compile / precision / device / batch / T2 FAILS `test_every_compile_flag_is_keyed_or_declared_irrelevant` until it is keyed or declared irrelevant. The compile REGIONS are code, named by the commit |

Why this keeps the K1b protection: K1b was a GLOBAL compile setting (`donated_buffer`) that changed the
artifact without changing Inductor's key. Global compile settings come from four places — the code
(keyed by the commit, and a dirty tree never shares), the adapter's config row (keyed), environment
variables (keyed), and argv flags (the compile surface, keyed). No trainer flag sets a global torch
compile config today (grep 2026-10-10: only `compile_control` / `compile_config`, the T2 engine's local
`dynamo.config.patch`, and AOT's local `inductor_config.patch` touch one — all code). Everything else
reaches an artifact through the traced graph, which Inductor's and AOTAutograd's own keys cover.

**THE CLEAN RULE** (`workspace_state`, run in the CHILD's checkout — `repo_root()` of the imported
code, i.e. a pinned child's own worktree, never the launcher's cwd):

- **No tracked change anywhere** (staged or not; `git status --porcelain --untracked-files=no
  --ignore-submodules=all`) and **no untracked file under `src/`** (`--untracked-files=all -- src`).
- **Untracked files OUTSIDE `src/` do not matter**: the trainer and every process it spawns import
  from `src/` alone (`PYTHONPATH=<checkout>/src`), so nothing outside it can enter a compiled graph.
  `data/` is read from MAIN's cwd even by a pinned child, and whatever data reaches the model reaches
  it as tensor VALUES or shapes — graph inputs, which Inductor's own key covers.
- **No IGNORED or untracked IMPORTABLE file under `src/`** (`untracked_importables`: every
  `.py` / `.pyc` / `.pyo` / `.pyd` / `.so` / `.pth` / `.pyx` / `.pxd` there that git does not track).
  The owner, 2026-10-10: the shared cache is fine "as long as we're confident it comes from a hash with
  no changes". `git status` never reports an IGNORED file, so before this rule a gitignored `src/foo.py`
  passed as clean. Two directory names are skipped, each for a stated reason:
  - `__pycache__`: CPython writes it from the TRACKED source and validates each `.pyc` against that
    source's mtime + size before using it. A sourceless `.pyc` OUTSIDE `__pycache__` IS counted.
  - a Cargo `target/` (`src/rust_*/target/`): it is built by cargo from the tracked crate sources, and it
    feeds observations, never code, to a torch compile. This is the submodule's reason, below.

  `test_an_IGNORED_importable_file_under_src_is_DIRT` pins both halves. On 2026-10-10 the live launcher
  pin worktree (`/tmp/launcher-95d014fa-*`) held no such file outside those two, so pinned runs stay
  shareable.
- **The `deps/pokemon-showdown` submodule is EXCLUDED** (`--ignore-submodules=all`). No torch compile
  reads it: it is the JS simulator, training runs on the Rust core, and nothing it holds reaches a
  compiled graph except as values. And the launcher's pinned worktree REPLACES it with a symlink to
  the launching checkout's submodule (`main/launcher/worktree.py`) — so it is never "at its recorded
  commit" there, and a plain `git status --porcelain` there EXITS 128 ("expected submodule path … not
  to be a symbolic link", measured 2026-10-10 in the test's real repo). Requiring a clean submodule
  would have made every pinned run unshareable.
- A dirty tree, an unknown commit, a `$LAUNCHER_GIT_HASH` disagreement, an unreadable GPU, a missing
  compile surface or the opt-out falls back to K3's per-run cache with ONE line:
  `🧊 [CompileCache] NOT SHARED — <why>; this run compiles into its own hermetic cache`.

**SAFETY.**

- **Entry creation is atomic**: a `.new-<key>-<pid>-*` dir gets its lock file, stamp and subdirs, then
  ONE `rename` publishes it; a concurrent creator that loses the rename uses the winner's entry. An
  entry is never visible without its complete stamp.
- **The writers inside an entry are torch's and triton's — READ in torch 2.8.0 / triton 3.4.0, not
  assumed**: Inductor's FX-graph, AOTAutograd, autotune, multi-kernel and generated-source writes go
  through `codecache.write_atomic` (a `.<pid>.<thread>.tmp` file + `rename`); Triton's
  `FileCacheManager.put` writes into a `tmp.pid_<pid>_<uuid>` dir + `os.replace`, and a kernel's
  group file last. Inductor's C++ builds (CPU kernels, AOTInductor) run under a cross-process
  `FileLock` in the cache's own `locks/` dir — SERIALISED against a concurrent writer — but the linker
  writes the `.so` in place, so **a process SIGKILLed mid-link can leave a truncated library**; the
  next process FAILS TO LOAD it (loud, never a silent wrong kernel). Recovery: delete that entry, or
  start with `GEN3AI_SHARED_COMPILE_CACHE=0`. `test_torchs_own_cache_writes_are_atomic_renames` pins
  the temp-file-plus-rename pattern, so a torch upgrade that drops it fails a test.
- **Live use is a LEASE**: every process that compiles into an entry holds a SHARED `flock` on its
  `.inuse.lock` for its lifetime — the trainer from `prepare_run_cache`, every child from
  `ensure_hermetic_cache` — released by the kernel when the process dies, SIGKILL included.
- **Pruning** runs at every shared start, after this process holds its own lease: least-recently-used
  first (`.last_used`'s mtime, touched on every open) until the root is under its cap
  (`GEN3AI_COMPILE_SHARED_CAP_GB`, default 20 GiB ≈ 8 entries at K3's measured 2.3–2.4 GB per run).
  An entry is removed only under an EXCLUSIVE non-blocking `flock`, which no lease holder lets anyone
  take, and only by renaming it to a `.trash-*` dir first — a process that opened its lock file just
  before sees the inode vanish from the key path and retries on a fresh entry. One pruner at a time
  (`.prune.lock`); dead creators' `.new-*` dirs and leftover `.trash-*` dirs are swept. An over-cap
  entry that is in use stays and is reported (`… over-cap in use`).
- **The gates still run on a hit — a hit only skips compile time.** R1's startup parity gate, the
  prewarm, the lock and the update-10 canary (`arm_compile_sentinel`) and T2's startup parity gate at
  every slot × bucket (`InferenceService._startup`) run exactly as on a cold start; pinned at the CALL
  level under a real shared HIT (`test_the_learner_startup_gate_and_the_canary_are_ARMED_on_a_hit`,
  `test_T2s_startup_parity_gate_runs_every_slot_x_bucket_on_a_hit`) and statically (no gate module
  imports the cache decision).

**Where things land.** The run's own `<run>/compile_cache/` keeps `stamp.json` (`"mode": "shared"`, the
entry, the key, the K3 stamp) and `t2_aot/` (T2's packages are `mkdtemp`-unique, never reused, so they
stay out of the shared entry: `$GEN3AI_T2_AOT_DIR`). A same-run restart whose previous start was shared
and whose tree is now dirty WIPES the run-local dir and starts the per-run cache cold. The startup
line: `🧊 [CompileCache] SHARED-HIT | SHARED-NEW <entry> — clean tree, key <12 hex>; … ; <x> / 20 GiB
after pruning N entries`. Opt-outs and knobs: `GEN3AI_SHARED_COMPILE_CACHE` (`0` / `off` / `false` /
`no` = off; unset / `1` / `on` = on; anything else = off, said in the line), `GEN3AI_COMPILE_SHARED_DIR`
(the root; tmpfs refused), `GEN3AI_COMPILE_SHARED_CAP_GB`.

**What it saves (ESTIMATE, from measured parts; the read below is NOT yet run).** `--arch endstate`'s
learner compile + gate + prewarm is 277.5 s cold (`perf_phase_2026-10-10/README.md`, F-PA-4) plus T2's
build; K3's torch-2.8 read put WARM at ~0.3× COLD for both the learner (394 → 117–134 s) and T2 (172 →
32–33 s). Shared, every fresh launch and fork of an unchanged commit + config starts warm, not only a
run's own restart.

**The measurement (GPU; the perf agent's next lease window — commands only, not run here).** A cold vs
warm-shared startup on `--arch endstate`, at a CLEAN commit that contains this change, through the
launcher's normal pinned path (which also proves the symlinked-submodule rule live). Each launch is a
fresh one-update run; read the learner's `… — reset + prewarm took <N>s` line and the `🦀 [RUST ENV] T2
up in <N>s` line from each child log, and the `🧊 [CompileCache]` line (must say `SHARED-NEW`, then
`SHARED-HIT`, then `NOT SHARED` for the control).

```bash
cd /home/goodlad/dev/gen3ai                     # the MAIN checkout after landing (clean), or a clean worktree at the commit
export PYTHONPATH=$PWD/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export GEN3AI_COMPILE_SHARED_DIR=$HOME/.cache/gen3ai/tmp/t25_cc_$(date +%Y%m%d_%H%M)   # an EMPTY root of its own
eval "$(scripts/ops/gpu_lease.sh acquire --owner perf_T25 --max-hours 2)"   # the orchestrator's grant; exports GEN3AI_GPU_LEASE_TOKEN
L="$PY -m main.launcher --restart-interval-hours 0 --device cuda --arch endstate --steps 98304 --snapshot-ladder-games 0"
$L --dry-run --run-name rb_t25_cc_cold          # validate first (never launch-and-kill)
# 1. COLD-SHARED: the first process on the empty root (expect SHARED-NEW)
timeout 3000 scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 40 $L --run-name rb_t25_cc_cold
# 2. WARM-SHARED: a DIFFERENT fresh run, same commit + config (expect SHARED-HIT on the same entry)
timeout 3000 scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 40 $L --run-name rb_t25_cc_warm
# 3. CONTROL: sharing off (expect "NOT SHARED … opted out": a cold per-run cache, today's fresh-launch cost)
GEN3AI_SHARED_COMPILE_CACHE=0 timeout 3000 scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 40 $L --run-name rb_t25_cc_control
scripts/ops/gpu_lease.sh release
for r in cold warm control; do echo "== $r"; grep -hE "CompileCache\]|reset \+ prewarm took|\] T2 (up|SHARED) in" models/rb_t25_cc_$r/launcher_child.log; done
du -sh "$GEN3AI_COMPILE_SHARED_DIR"/*/
```

The desktop must not hold the GPU (T23) and the disk preflight must pass, exactly as for any CUDA
launch. The three runs are throwaway (delete their run dirs after the read; the shared root too). The
verdict is descriptive (n = 1 per arm, contention REPORTED): warm-shared learner and T2 seconds beside
cold and the control, plus the entry's size.

## FP32 matmul precision — the ONLY precision (TF32 retired; `--matmul-precision` DELETED)

Every fp32 matmul in the trainer process runs at PyTorch's own default,
`torch.get_float32_matmul_precision() == "highest"` (full FP32, no TF32). Nothing in the trainer sets
it and there is no flag. TF32 was retired by the owner (2026-10-01; deletion pass K2): its ~1e-2
rounding cannot be checked deterministically (the K9(b) exclusion sweep), it had cost a startup crash
(`ai_v14_04_lbat_t32` died on the fp32 compile gate), a second gate graph and sentinel headroom, and
the T32b read found it futile (+2.48 % GPU time per step, ledger 2026-10-01). What stays:

- **The provenance stamp.** `metadata.json` still records `matmul_precision`, read back from torch
  (always `highest` now; `main.train.run_io._matmul_precision`).
- **Every parity gate refuses any other precision.** The region gate (`compile_regions.gate_regions`),
  K9(b) (`consistency.behaviour_gate`, `UndeclaredPrecision`) and T2's parity judge each call
  `parity_probe.unmeasured_precision()` and raise their own typed error naming the precision, so a
  process someone set to `high` or `medium` is refused instead of judged at bars measured elsewhere.
  The inference service's freeze check (a precision change after the freeze is a lifecycle violation)
  and `torch_state_guard` (a test that leaks the precision or `allow_tf32` FAILS) are unchanged.
- **A run that recorded TF32 resumes pinned or not at all.** A resume or fork whose `metadata.json`
  recorded `matmul_precision != "highest"` (one run on record, `ai_v14_04b_lbat_t32`) is refused
  `FATAL_CONFIG` naming `--matmul-precision` and the pin
  (`model_version.retired_levers.RETIRED`, `LAST_COMMIT_K2`: any commit at or before it still has the
  flag). A recorded argv that types the flag fails argparse on HEAD and must run pinned
  (`designs/deleted_flags.md`); `learner_benchmark.build_trainer_argv` drops flags the trainer's parser
  no longer knows, so a re-pointed recorded command still runs.

## Compiled GPU trainer (`--compile-trainer`, DEFAULT ON for cuda)

`torch.compile`s the LEARNER — the CUDA forward **and backward** the PPO step runs. The other half of
the pair above, and the larger of the two.

> **WHAT SHIPS NOW (2026-10-02, deletion pass K1; the rollout region R0 deleted 2026-10-03, P10-E).**
> The learner compiles ONLY as its DECLARED REGION — R1 the micro-step (K8 below) — installed, gated,
> prewarmed and locked by `compile_trainer.arm_compile_sentinel`; the trainer's earlier compile step
> (`preflight_compile_trainer`) compiles nothing and only refuses a learner the regions cannot serve.
> HEAD runs **torch >= 2.8 only** (`utils/torch_floor.py`: the trainer exits `FATAL_CONFIG` on an
> older torch). The torch-2.5.1 extractor-only compile, its startup parity gate (speed refusal, probe
> loss, coverage guard), the CUDA trunk split and `compile_control`'s 2.5.1 rows were DELETED; a run
> trained on 2.5.1 resumes PINNED to its own commit, which still carries them. The subsections below
> that describe the extractor-only compile are its HISTORY — measured facts with their provenance,
> not current mechanism — except where they say they still apply.

🚨 **DEFAULT since 2026-08-17, and it is the one default that could NOT be a flat `True`.** This
flag REFUSES a non-cuda device (the first row of the refusal table below), so `default=True` would
convert every working CPU invocation — the `--debug` smoke, a laptop, CI — into a `FATAL_CONFIG`
exit. The default is therefore **AUTO**, resolved by `train_rl_agent.resolve_compile_trainer_default`
(pure, injectable, unit-tested without a card):

| resolved device | `--debug` | default |
|---|---|---|
| `cuda` / `cuda:N` | no | **ON** |
| `auto` on a box with a card | no | **ON** |
| `auto` with no card, `cpu`, anything else explicit | no | OFF |
| any device, including an explicit `--device cuda` | **yes** | **OFF** |

`--debug` is excluded outright because a smoke exists to prove the pipeline in ~1 minute and a
multi-minute Inductor compile (plus a CUDA context taken from whatever run owns the card) defeats
that. **The REFUSAL is unchanged**: an explicit `--compile-trainer --device cpu` still exits
`FATAL_CONFIG` with the same message. `--no-compile-trainer` is the opt-out.

**⚠️ The device is only HALF the auto default, and the other half is easy to miss.**
`check_shape_stability` (below) refuses an UPDATE (the real size: `--rollout-target-samples`, else
`n_steps * n_envs`) that does not divide by `--batch-size` — correct for someone who ASKED for the compile, and fatal for a DEFAULT,
because it would convert a class of command that works today into a startup `FATAL_CONFIG`. (It also refused the async-rollout collector until that collector was deleted, U3.)
So `resolve_compile_trainer_auto` runs those same checks and, on a refusal, **leaves the default OFF
and says why** rather than refusing to launch:

```
⚡ --compile-trainer would be ON by default here, but this config cannot take it — leaving it
   OFF rather than refusing to launch. Reason: … (pass --compile-trainer explicitly to make
   this a hard error instead.)
```

The rule, and it generalises to any future default: **a default yields to the config the user typed
and announces it; an explicit flag refuses.** Pinned by `src/main/compile_defaults_test.py`
(`test_auto_yields_to_a_rollout_that_does_not_divide_the_batch`, and
`test_an_explicit_flag_never_reaches_the_auto_path`, which holds the refusal in place).

**Measured** (2026-08-14, v76 `gen3_ctx_dedup_v1`, RTX 3080 Ti, the real
`MaskablePPO -> ActorCriticPolicy._build()` path, gen-9's own `cli_args`: batch 4096, PopArt on (PopArt has since been deleted);
`policy.evaluate_actions` fwd+bwd, arms interleaved, 3 pairs, idle box):

| scope | eager | compiled | speedup |
|---|---|---|---|
| extractor only (**what ships**) | 155.1 ms | 88.5 ms | **1.753x** |
| whole `evaluate_actions` | 155.5 ms | 88.5 ms | 1.757x |

> **These are THE numbers to quote for this result.** `extractor_compiles_test.py`'s docstring
> carries a *second*, independently-measured pair for the same lever — **150.85 → 86.21 ms, also
> 1.75x** — taken in a different session as that test's own in-situ check. Two sessions, two pairs,
> one ratio; they corroborate rather than conflict, but only this row set carries the full
> provenance above, so a doc quoting `150.85` is quoting the test, not this benchmark.

**End-to-end FPS: ~+62%** — but read the derivation before quoting it. It is `1.75x` applied to a
**~89% train share**, and that share is an **EXTRAPOLATION, not a measurement**: the 89% is
projected to production's 10 epochs from a *measured* **61%** at `n_envs=8, n_steps=128, 2 epochs`
(the gen7/gen8 regression investigation). The 2026-08-23 idle-box re-baseline
(`designs/research_state/measurements/post_paydown_baselines_2026-08-23.json`) re-measured
`obs_build`, `trainer_turn` and both bridge benchmarks — it did **not** measure the train share, so
there is no fresher figure to substitute and this one has not been re-derived since. **UNVERIFIED:**
the ~89% at production `n_envs=48 / n_steps=2048 / 10 epochs`. The 1.75x itself is measured; the
end-to-end number inherits the extrapolation's uncertainty.

**We compiled the EXTRACTOR (until K8), and the second row is
why**: the two scopes measure the same to within 0.004x — the mlp_extractor, the pointer head and
the value head contribute nothing — so the whole-policy scope buys nothing for strictly more graph
(and more surface for SB3's distribution objects and the mask path to break on). Same win, smaller
blast radius. Also confirmed: rollout 2048x48 / batch 4096 = **exactly 24 minibatches, no
remainder**, so one graph and no per-epoch recompile.

**FAIL-LOUD BY DESIGN, and deliberately asymmetric with the offline opponent compile.** The opponent path
warns and falls back to eager because it prints a
`[CompileExtractor]` line either way. Here there is nothing to notice: a silent fallback trains
perfectly correctly and just produces ~38% fewer steps/hour forever. So every failure is fatal
(`CompileTrainerError` -> `TrainExitCode.FATAL_CONFIG`, so the launcher gives up instead of
restart-looping), and the flag has no `strict` variant because there is nothing to opt into. The
refusals, each guarding an otherwise-invisible outcome:

| refusal | where | why |
|---|---|---|
| `--device cpu` | `preflight_compile_trainer` | the compiled learner is gated and measured on CUDA only. (On torch 2.5.1 the CPU BACKWARD did not even lower — `CppTile2DKernel.store` asserted on the `atomic_add` mode; on 2.8 it lowers, per `extractor_compiles_test::test_cpu_backward_still_does_not_compile`'s per-torch pin, but no CPU compiled learner has ever been gated) |
| a learner without the micro-step (`_micro_static`) | `preflight_compile_trainer`, `arm_compile_sentinel` | R1 cannot be built, and there is no smaller fallback compile (`compile_trainer_test::test_a_learner_without_the_micro_step_is_REFUSED_not_compiled_another_way`) |
| a graph break inside a region, a compile error | the region gate | `fullgraph=True`; bisect the op — the whole "torch cannot compile our model" story was ONE op (see `src/agents/model/CLAUDE.md`, the `species_posterior` precedent) |
| a region disagrees with eager on REAL rows | the region gate (K8 below) | a faster wrong model is not a win |
| an undeclared signature, a cache-limit hit, a region run eager | the sentinel | a silent stall or a silent ~2x slowdown |

### 🚨 The single-graph CUDA miscompile, and the trunk split (`gen3_inductor_trunk_split_v1`, 2026-09-28) — HISTORY

> **The split was DELETED on 2026-10-02 (deletion pass K1)** with HEAD's torch-2.5.1 support: it was
> OFF on torch 2.8 (Lane K1 below), and a 2.5.1 run resumes pinned to its own commit, which keeps it.
> This section is why 2.5.1 must never run HEAD code (`utils/torch_floor.py` refuses it).

**Measured** (torch 2.5.1+cu121, RTX 3080 Ti, fp32 `highest`, `ai_v14_01_base` weights on 3,840 REAL
rows from its own `eval_traces`): with the whole extractor as ONE CUDA Inductor graph, the compiled
learner disagreed with eager by up to **7.65 on pi_features** (scale ~16). At the decision level,
**argmax agreement was 70.9%**, masked-policy TV had median 0.12 and p99 0.63, win-prob |dV| had
median 0.024 and max 0.33, and in the TRAIN graph the extractor gradient's **cosine to eager was
0.778**. dynamo `backend="eager"` and `"aot_eager"` matched exactly, and the CPU Inductor compile
(the CPU opponent compile: eval workers, sentinels, pool, stable and exploiter opponents, so every eval
trace) matched to 3e-5, the batch-shape noise floor. **Scope: the GPU LEARNER only**, since
`--compile-trainer` became default-on for cuda in `28eaef29` (2026-08-17). A run's rollouts, its PPO
ratio and its gradients were computed on the diverged function, while its eval, opponents and
traces used the correct one. The old startup probe used ALL-ZERO obs, and those agreed to 4.8e-7:
zero obs exercise no masking, no top-K seat selection and no edge family.

**The fix is a deliberate graph break** in `TeamTransformer.forward`, placed right after the attention
bias and its edge families are built. It fires only when compiling on CUDA
(`_CUDA_TRUNK_SPLIT and attn_bias.is_cuda and torch.compiler.is_compiling()`), so eager is
bit-identical and the CPU compile keeps its one graph (`extractor_compiles_test`'s one-graph pins
are CPU). After the split, on the same 3,840 rows: argmax agreement 1.0000, TV p99 1.5e-6, |dV| max
1.0e-6, legal logits max 2.7e-5, gradient cosine 1.000000 (rel err 4.7e-7). The effect depends on
WHERE the break sits:

| break at | parity |
|---|---|
| after the edge families / the transformer's start / the transformer's end | restored (1.2e-5) |
| after `unpack`, after the entity seats, before the projection | still 7.65 |

**No Inductor knob fixes it.** Tried: math-only SDPA, `pattern_matcher=False`,
`allow_buffer_reuse=False`, `inplace_buffers=False`, `comprehensive_padding=False` and
`shape_padding=False`. **UNVERIFIED — the root-cause op is not named.** The accuracy minifier's
dumped 9,805-node AOT graph does not reproduce the failure standalone.

**Speed cost of the split, measured under the live L95 arm's contention.** Compiled fwd+bwd
minimum-of-15 at batch 1024: 46.3/46.6 ms with the split vs 45.7/45.8 ms without (+1.5%). Medians
48.4-50.5 vs 46.4-47.9 ms (≤ +4%). Batch 512 is noise-dominated. Batch 2048 could not be measured
beside the live run's VRAM. The split also lowers peak memory: 1.7 vs 2.2 GiB at batch 1024.

### Lane K1 — torch 2.8 (`gen3ai_torch28`), THE DEFAULT; the split OFF there (2026-09-28; default 2026-09-30)

🚨 **torch 2.8 is the DEFAULT interpreter for every new run, gate and tool (owner, 2026-09-30: "go
ahead with moving to 2.8"); torch 2.5.1 is LEGACY.** `gen3ai_torch28` (torch **2.8.0+cu126**,
torchvision 0.23.0+cu126, torchaudio 2.8.0+cu126, triton 3.4.0, sympy 1.14.0, the cu126 `nvidia-*`
wheels and the CUDA 12.6 headers; `environment_torch28.yml`) is the env `scripts/bootstrap.sh`
manages, and every gate command, `scripts/land.sh`, `scripts/ops/_common.sh` and the docs name it.
`gen3ai_stable` (torch 2.5.1+cu121, `environment.yml`) is kept FROZEN — bootstrap no longer reads
`environment.yml` or touches the env — only so an old run resumes on the torch it trained on.
**New code targets 2.8 only**; the rule that a compile change also runs its tests under 2.5.1 is
RETIRED. The 2.5.1-keyed branches (the split, `compile_control`'s 2.5.1 rows, the 2.5.1
learner-golden entry, the extractor-only compile, `aot.py`'s < 2.8 refusal) were DELETED by the
deletion pass on 2026-10-02 (K1, `designs/ops/deletion_pass_manifest.md` R7/R8), behind ONE HEAD-side
refusal: `src/utils/torch_floor.py`, called first thing in the trainer's `main()`, exits
`FATAL_CONFIG` on torch < 2.8 (so a 2.5.1 checkpoint under `--no-pin`, `--sync-to-main` or a bare
`train_rl_agent.py --model` stops with the reason instead of running HEAD code that would miscompile).

**torch 2.8.0+cu128 (2026-10-12, the cloud-GPU benchmark).** A Blackwell card (RTX 5090, sm_120) needs
the CUDA 12.8 build — the +cu126 wheels' arch list stops at sm_90. The compile sentinel keys on the
EXACT version string, so it first REFUSED `2.8.0+cu128` (`CompileSentinelError`, the rented box's first
GPU sanity). On that box: all 20 hashed torch internals equal the +cu126 row, both compile-config keys
exist, the compile contract tests pass (68 / 69; the red one used `2.8.0+cu128` as its unknown example),
and the end-state learner compiled and passed the R1 parity gate. So `compile_control` / `compile_config`
carry a +cu128 row that COPIES the +cu126 one. The shared env stays +cu126; the remote installs +cu128
(`tools/cloud/vast/requirements.py`). Runbook: `designs/ops/cloud_gpu_benchmark.md`.

**Which torch a run uses is RECORDED and ENFORCED.** Every save writes `metadata.json`'s
`torch_version` (and a `torch` key per `pin_history` span); a run without one predates the record
and trained on 2.5.1. The launcher (`src/main/launcher/torch_runtime.py`) resumes or forks a run
under its recorded torch — it SELECTS `gen3ai_stable` for a 2.5.1 run and refuses a mismatch
`FATAL_CONFIG` unless `--allow-torch-switch`; `--dry-run` prints the interpreter, its torch and the
run's record. A bare `train_rl_agent.py --model` is not checked by the launcher; the torch floor
above refuses it on 2.5.1. (Until 2026-10-02 the split was keyed on the torch version — OFF on
2.8.0+cu126, ON elsewhere — so a legacy resume kept the split it trained with; a pinned resume still
does, from its own commit.)

**The 2.8 sweep before the flip (2026-09-30 / 10-01, `gen3ai_torch28`):** routine gate 12,690 passed —
its only 2.8-specific red was the mypy gate (torch 2.8's stub types `nn.Module.__getattr__` as
`Tensor | Module`; 111 errors fixed by annotations / casts, clean on both torches); slow tier (CPU)
67 passed / 0 failed; the GPU compile files (`extractor_compiles`, `compile_control`, `compile_trainer`,
`compiled_train_probes`, `compile_defaults`) pass under the GPU lock. Four GPU-only gate tests had driven
`compile_trainer_extractor` with a heads-less stand-in policy, which the grad-COVERAGE guard refuses on
BOTH torches (15–17 of 232 zero; the real fresh production policy on the perturbed rung: 1 of 254,
`c5_map`) — they now build the real policy (`build_fresh_model`).

**Measured, split OFF on 2.8** (RTX 3080 Ti beside a live training run, `ai_v14_06_lbat_ctrl_fix`
`final_model.zip`):

| check | fp32 (`highest`) |
|---|---|
| the real startup gate (64 fixture rows) — eval readout | PASS: features 2.4e-5, legal log-prob 9.5e-6, V 4.2e-7 |
| the real startup gate — TRAIN graph gradient cosine | 1.000000 |
| 3,840 rows of its own `eval_traces` (steps 78M/80M/82M) — argmax agreement | 1.0000 |
| same — masked-policy TV p99 / max | 1.6e-6 / 2.7e-6 |
| same — win-prob \|dV\| max | 1.4e-6 |
| train graph, 64 rows, loss = mean pi² + mean vf² — gradient cosine | 1.000000 (rel 7.3e-7) |

(`compile_trainer_test::test_REVERTING_the_trunk_split_FAILS_the_real_obs_gate`
asserted, per torch, 2.5.1 ⇒ split ON and unsplit FAILS, 2.8 ⇒ split OFF and unsplit PASSES; it was
deleted with the split, K1 2026-10-02.)

**Graph count** (`torch._dynamo.explain`, the extractor's `forward`, 32 rows, CPU): **1 graph /
0 breaks on both torches, at every `--log-level`** (2026-09-30: 7,062 ops on 2.5.1, 6,870 on 2.8,
production config). The ObservationDebugger that used to attach at `--log-level periodic` — and
whose rate-limited logger's `time.time()` (`utils/logging/rate_limiter.py:14`) was the source of
BOTH of the 3-graphs / 2-breaks the periodic build read — was removed that day. On CUDA the 2.5.1 split adds one; on 2.8 the CUDA explain is also **1 graph / 0 breaks** (6,870 ops) — the unsplit graph.

🚨 **`explain`'s "0 breaks" is NOT the `fullgraph=True` verdict on 2.5.1** (measured 2026-09-30, the K8
inventory, `designs/research_state/measurements/k8_inventory/`). The test is `torch.compile(fe.forward,
fullgraph=True)` on the production surface with the committed real-obs rows, CPU:
- **2.5.1 REFUSES in every mode** (train/grad, train/no-grad, eval/no-grad, and through
  `Module.__call__`, `extract_features` and the rollout's `policy(obs, action_masks=…)` alike) at
  `forward_guard.py:57`, `forward_guard_for`'s `WeakKeyDictionary.get` ["inline in skipfiles"]. The
  break sits at the frame's FIRST instruction, so it ends no graph, and `explain`, which counts breaks
  through the graphs they end, reports none. The inventory's capture sees it (4 events in a first
  update).
- **2.8 traces the lookup and compiles `fullgraph=True` in all three modes.**

So the extractor meets K8's per-region rule (ONE declared `fullgraph=True` region; K8 is DECLARED
COMPILE REGIONS, owner 2026-09-30, `designs/endstate/program_rust_core.md` Lane K) on **2.8 only**; on
2.5.1 the guard lookup must leave the traced region first (the inventory's fix 5a). One more extractor
break is reproducible in the trainer process: the rollout-shaped eval/no-grad forward broke at
`encoders.py:287` (`move_self_attn`, the native `nn.MultiheadAttention` fast path) on both torches with
C's checkpoint and real rows. A standalone production-surface eval compile does NOT reproduce it —
**UNVERIFIED** why; the inventory's fix 5b (fast path off in the compiled eval region) removes it
either way.

**Speed A/B (2026-09-29, idle RTX 3080 Ti, the same saved rollout buffer, `learner_benchmark`,
K=5):** 2.8-unsplit is 1–2% FASTER than 2.5.1-split — baseline 57.21 s vs 58.45 s per update,
diag_skipped 50.47 vs 51.27, epochs_half 33.08 vs 32.97 — almost all of it in the
backward (27.31 vs 28.33 s); the loss agrees to 5 decimals. Within ~2× the run-to-run spread, so a
small effect at most. Table and reports: `designs/research_state/measurements/m5_k1/`.

### CUDA-graph capturability of the compiled extractor (M5 T2, 2026-09-29)

The compiled decision forward is captured as a `torch.cuda.CUDAGraph` by the T2 inference service
(`src/agents/inference/service/`), which requires a graph with NO host synchronisation. Measured
with `torch.cuda.set_sync_debug_mode` on torch 2.5.1: the compiled forward had exactly ONE —
`team_transformer._event_reference_cells` built its `[1,1,12]` side-sign constant on the CPU and
`.to(device)`-copied it every call. That copy was also why Inductor's own `mode="reduce-overhead"`
skipped cudagraphs ("cpu device (cat_100)") and ran at 2.1 s/call. It is now built on the device
(same values; eager bit-identical). The EAGER forward still makes 24 host syncs (Python-scalar
`index_put`s and host-built constants across `extractor_ctx`, `damage_op*`, `team_transformer`), so
eager itself cannot be captured; only the compiled graph is. Detail and the AOT verdict:
`designs/endstate/program_rust_core.md` §2, "T2 DESIGN".

### Lane K1b — two backwards through one compiled graph (`gen3_donated_buffer_off_v1`, 2026-09-29)

K1's parity checks never put TWO backwards through one compiled graph; the first real update does —
`grad_balance._flat_grads` and the per-term noise-scale probe call `autograd.grad(retain_graph=True)`
on the compiled train graph before `loss.backward()` (K2 runs every probe on a process's first
update). On torch 2.8 that failed twice, for two reasons:

1. **`torch._functorch.config.donated_buffer` defaults True from torch 2.6** (2.5.1: False). A
   donating backward reuses the forward's saved activations in place, legal only for single-use
   backwards. The donated set is fixed at compile time and cleared only if the graph's FIRST backward
   retains it; the sentinel's prewarm does a plain `.backward()` first, so the first update's
   grad-balance probe raised *"This backward function was compiled with non-empty donated buffers"*.
2. **The Inductor FX-graph cache key is blind to donation** (`FxGraphHashDetails` carries no donated
   indices). With donation OFF, the same graph was still SERVED the donating backward the crashed run
   had cached in `/tmp/torchinductor_goodlad`: the probe's first backward overwrote saved tensor
   `[2048, 128]` (version 1 → 2) and the second raised *"modified by an inplace operation"*; with a
   fresh `TORCHINDUCTOR_CACHE_DIR` the same run completed. Here a version counter caught it; a
   donating Triton kernel that overwrites a saved activation without an ATen op would not bump one —
   **UNVERIFIED** whether any graph of ours does that, which is why the tag is not optional.

**The fix is one row in `compile_control._COMPILE_CONFIG`**, applied at `install()` (before the
gate's first compile) and restored at `uninstall()`: `donated_buffer = False` on both torches, and on
2.8 `torch.compiler.config.cache_key_tag = "gen3_donated_buffer_off_v1"` — a public knob that IS in
the Inductor and AOTAutograd cache keys, so our graphs can never be served an artifact compiled under
the default config by any other process. The readers (`aot_dispatch_autograd`,
`AOTDispatchAutograd.post_compile`, `FxGraphHashDetails.__init__`) are in the hashed drift row.
Restructuring the calls instead was rejected: the outcome depends on which backward reaches each
graph first (prewarm order, K2 cadence), and it would not touch the
cache hazard. Donation OFF runs the same kernels on the same inputs — `compile_control_test` pins the
gradient bit-for-bit on vs off — and costs nothing measurable: production extractor fwd+bwd on 2.8,
batch 64 ~10.1 ms and batch 2048 ~74.8 ms either way, peak memory +0.3%
(`designs/research_state/measurements/m5_k1/`).

**Tests.** `compile_control_test` (routine): install pins the row and uninstall restores it; a
donating backward REFUSES a retain-graph probe and the pinned row accepts it; the pin changes no
gradient; the cache key carries the tag and not donation. (`agents/training/compiled_train_probes_test.py`,
the `slow` end-to-end — the extractor-only pipeline plus ONE full `train()` with every first-update
probe, revert `donated_buffer` ⇒ the probe raised — was DELETED with that pipeline, K1 2026-10-02.
⚠️ No test now drives a full first `train()` through the COMPILED REGIONS with the donating default
forced; the routine `compile_regions_test` runs two real updates through R1 on CPU with the pin in
force, which is not the same revert-must-fail.)

**Batch 1 never reached a compiled learner graph (`gen3_batch1_eager_v1`, 2026-09-30, Lane K) — HISTORY;
the route and the region it guarded are DELETED (P10-E, 2026-10-03).**
The K1 finding: on torch 2.8.0+cu126 a **batch-1 CUDA eval/no-grad** graph of the production extractor
fails to LOWER — Triton `CompilationError` (`'constexpr_type' object has no attribute 'is_block'` on a
fully-constant `tl.broadcast_to` index); batch 2 and 4 compile. Region R0's dispatcher
(`compile_regions._make_r0`) ran a batch in `compile_trainer.EAGER_BATCHES` (= {1}) through the EAGER
rollout core and every other batch through the compiled one. R0 itself is gone (K8 below: the Rust
collector serves rollouts through T2, so the learner process never called it), and with it
`EAGER_BATCHES`, the dispatcher and `install_rollout_region`: the learner's `policy.forward` /
`rollout_core` are eager at EVERY batch (`predict`-class callers), so there is no batch-1 route to
declare and no batch-1 CUDA graph to lower. Measured then (RTX 3080 Ti, torch 2.8.0+cu126): the batch-1
call equalled eager bit-for-bit, and reverting the routing raised the `CompilationError`; the 2026-09-30
smoke (`~/gen3ai_archive/k6_k8/smoke28_b1.log`) ran gate → prewarm → lock → two iterations with 0 compiles
after the lock. `compile_regions_test` now pins that an eager rollout forward at every batch (4 and 1)
takes no region route and compiles nothing after the lock.

### The startup parity gate (`gen3_compile_parity_real_obs_v1`)

> **Since 2026-10-02 the gate described here — the extractor-only gate — is DELETED** with the
> extractor-only compile (K1). What still applies, now in the REGION gate (`compile_regions.gate_regions`,
> K8 below) and the canary: the committed real-obs fixture, the bars and verdicts
> (`compile_trainer.decision_verdicts` / `train_verdict`, the per-parameter rule), the fail-closed
> vacuity guard, the fresh-weights perturbation and its LADDER. What went with the gate: its probe
> loss (`gate_loss`) and the zero-gradient COVERAGE guard (R1 is judged on the production
> micro-step's own loss) and the speed refusal. TF32 was retired by deletion pass K2: the gate's
> TF32 rule (against an fp32 eager reference), its "same graph at fp32" second arm and the
> `--matmul-precision high` flag it served are gone, and fp32 `highest` is the only precision. The
> measurements below are that gate's, with their provenance.

The extractor-only gate validated on **REAL observation rows, never zeros**. The rows are a
committed fixture, `src/agents/model/compile_parity_obs.npz`: 64 decision rows (committed from 8 reproducible
poke-env bridge battles; since P6 of the poke-env retirement the regenerator plays 8 seeded Rust-core battles,
`utils.rust_env.fixture_battles`), made by `python -m agents.model.compile_parity_fixture --write`. It can be
regenerated on any box for any layout. A missing fixture, or one of the wrong width, **REFUSES**
(`FATAL_CONFIG`) rather than falling back to zeros, and
`compile_trainer_test::test_the_committed_real_obs_fixture_matches_the_live_layout` fails the
routine gate the moment the layout changes. It checks four things, eager vs compiled:

| quantity | fp32 (`highest`) bar | healthy (after the split) | the miscompile |
|---|---|---|---|
| (pi ‖ vf) features, no-grad | 1e-4 (unchanged) | 2.4e-05 | 10.7 |
| MASKED legal log-probs (what the rollout samples and PPO's ratio reads) | 1e-3 | 1.2e-05 | legal logits off by up to 7.39 (3,840 rows) |
| V (the win-prob sigmoid under the win-prob critic (the only critic)) | 1e-4 | 5.4e-07 | 0.33 (3,840 rows) |
| TRAIN graph: fwd features + gradient cosine on the gate's own loss (since `gen3_gate_grad_coverage_v1`: the probe loss over the whole policy, plus the coverage guard and the per-parameter rule below) | 1e-4 · cos ≥ 0.9999 | 1.9e-05 · 1.000000 | 7.65 · 0.778 |

The healthy column is the gate's own line on `ai_v14_01_base/final_model.zip`.
Measured through the real gate on torch 2.5.1: with the split OFF it FAILED at 10.7.

**On FRESH weights the gate ALSO runs on a seeded perturbation (`gen3_fresh_parity_probe_v1`).** A
fresh launch (`--arch production`, no `--model`) runs the gate on the freshly initialised policy, and
the pointer head's three scorers are zero-init, so every legal log-prob on a row is `-log(n_legal)`
whatever the extractor computed. On those weights the legal-log-prob check cannot fail, and the
features and gradient checks are blind to what feeds only the pointer head. On the fresh production
policy, 52 of 232 extractor parameters got zero gradient from the gate's loss at the time (the
train-graph coverage fix below closes that). **Measured** (M5 T2,
2026-09-29): an AOT miscompile read max|dlogp| 0.0 on a fresh policy and 0.68 on a real one. So:

- **Detection.** The gate reads the eager arm's decision readout. It counts the weights as fresh when
  any quantity is not above its own bar (`agents.model.parity_probe.spread`). For log-probs that is
  the within-row spread over the legal entries; for everything else it is the spread across rows.
  On a fresh production policy the log-prob spread is exactly 0.0.
- **The perturbed pass.** Both arms run again on every policy parameter plus seeded Gaussian noise
  (seed `20260929`, scale 0.05), with the SAME installed compiled graph. Parameters are graph
  inputs, so the change neither recompiles nor escapes the graph. The noise comes from a private CPU
  generator, so training's RNG stream is untouched. The parameters are restored bit-exactly and
  re-checked; a mismatch raises. Its verdicts are prefixed `[fresh weights, seeded perturbation
  scale=X]` in the PASS line, and a `[CompileTrainer] FRESH weights (…)` line says it ran and at
  which scale.
- **The LADDER (`gen3_parity_perturb_ladder_v1`, 2026-09-30).** The scale is not fixed. The gate
  walks the DECLARED ladder `parity_probe.PERTURB_LADDER` — (scale, seed offset) rungs, 0.05 then
  0.1, 8 seeds each, the scale capped at `PERTURB_MAX_SCALE` = 0.1 — and judges at the FIRST rung
  whose EAGER arm is informative on every rule. It tests that by running
  the rules on the eager arm against itself, where every delta is 0, so only a vacuity guard can
  fire. No informative rung ⇒ `VacuousCompileParityError` (`FATAL_CONFIG`) naming each rung.
  - Why: a COLLAPSED win-prob critic stays vacuous on V at the fresh-weights rung. The pre-fix gate
    would have refused a resume onto such a checkpoint (`VacuousCompileParityError`, `FATAL_CONFIG`).
  - Why capped, with seeds instead of larger scales: the bars are absolute, and the perturbed
    network's compiled-vs-eager noise grows steeply with the scale. Measured on the inference
    decision (RTX 3080 Ti, fp32): |Δ log π| ≤ 4.5e-6 at 0.1, up to 1.7e-3 at 0.2, 0.66 at 0.5. A
    critic saturated beyond what 0.1 moves (win logit ≈ −12) is refused.
  - A fresh policy's rung is 0.05, so every fresh launch is unchanged.
  - ⚠️ **UNVERIFIED on CUDA at the 0.1 rungs:** whether this gate's per-parameter GRADIENT bar
    (1e-3, measured at 0.05 on fresh weights) holds at 0.1 with a real `torch.compile`. The forward
    decision's noise at 0.1 is measured (above); the backward's is not.
  - The offline opponent compile's gate (`opponent_parity.py`) and the T2 inference service walk the
    same ladder.
  - **One precision.** fp32 `highest` is the only precision, so the ladder's scale cap is the single
    `PERTURB_MAX_SCALE` = 0.1. `parity_probe` also holds `LOGPROB_BAR` (1e-3, where `_FP32_TOL`'s
    legal log-prob bar is read from) and T2's greedy near-tie band `NEAR_TIE_BAND` (2x that bar); a
    process at any other precision is refused (`unmeasured_precision`).
- **Fail-closed.** `decision_verdicts` / `train_verdict` refuse a vacuous comparison
  (`VacuousCompileParityError`, a `CompileTrainerError`, so it is `FATAL_CONFIG`). Only the gate's
  own real-weights pass waives the check with `allow_vacuous=True`, and only after the perturbed
  pass has passed with the check ON.

**Measured end to end** (RTX 3080 Ti, torch 2.5.1, fresh production policy, 64 fixture rows,
2026-09-29). The real pass read legal log-prob max|Δ| **0.00e+00**, which is vacuous. The perturbed
pass read 5.96e-07 (features 5.25e-06, V 4.77e-07, grad cosine 1.000000), and the gate PASSED. With
the trunk split forced OFF (the known torch-2.5.1 miscompile) the gate REFUSED on the perturbed
arm's features at 3.61. In both cases the weights and the CPU/CUDA RNG states were bit-identical
afterwards. (`parity_probe_test` pinned that wiring on CPU until the gate's deletion; it now pins the
premises and the guard, and `compile_regions_test` the region gate's fresh-weights pass.)

The inference service's parity gate (`agents.inference.service`) does the same per slot. A
`VacuousParity` (a `ParityFailure`) triggers a re-run of the slot on the perturbed weights, in place
in the group's stacked storage, which the CUDA graphs read, up the same ladder
(`ServiceSpec.perturb_ladder`). The real weights then run with the check waived. Each verdict
records its path (`ParityReport.path`). A refusal exits the trainer `FATAL_CONFIG`
(`main.exit_codes`), because a restart would replay it.

**The train graph's COVERAGE (`gen3_gate_grad_coverage_v1`, 2026-09-29).** Until this change the
train-graph check backpropagated `mean pi² + mean vf²`, a loss over the extractor's two feature
outputs only. The pointer head reads the extractor's STASH (`stash.pointer_inputs`: move seats, team
tokens, per-action move/switch cells), and the aux heads read further stash fields. Every parameter
that feeds only those paths got a ZERO gradient from the gate's loss. On the perturbed production
policy (CPU, 64 fixture rows) that was **43 of 232 extractor parameters (65 of 254 policy
parameters)**, so a backward-only miscompile there passed on any weights. Three changes:

- **The probe loss** (`agents.model.compile_gate_probe.gate_loss`) is the old features term plus
  what the production loss reads: the MASKED legal log-probs through the real pointer head, `V`
  (`policy._critic_value`), and every graph-carrying stash tensor (the scalar `value_net` it also read is
  DELETED with the SB3 value tower, version break part 2, audit F1). Each term
  gets fixed, deterministic, sign-varying weights (no RNG). The gradient is read over the WHOLE
  policy (extractor and heads). It leaves **1 of 254** parameters with zero gradient (0.39%, the same
  on CUDA, fresh-perturbed and trained): `edge_bias.c5_map.weight`, the Baton-Pass receiver edge.
  Its input cells are all zero because no fixture row has a Baton-Pass seat.
- **The coverage guard** (`coverage_verdict`) refuses the launch (`VacuousCompileParityError`) when
  more than **2%** of the policy's parameters (5 of 254) get zero gradient from the probe loss, and it
  names them. The bar sits 5x above the healthy count and 13x below the old loss. It applies to the
  informative pass only: on unperturbed fresh weights the zero-init heads zero 35 of 254 by
  construction.
- **The per-parameter rule.** A GLOBAL cosine is dominated by the largest gradients. On CPU, dropping
  the whole move-cell gradient left it at 0.9999998 (float64), and removing the switch-cell
  gradient left it at 0.9999989. The cosine is now computed in float64: in fp32 it read 1.0005 over
  the 7.0M-entry gradient. So every parameter whose eager gradient norm is above 1e-3 x the largest
  is also held to a relative error `||c_p − e_p|| / ||e_p||`, with **two bars**:

| weights | per-param bar | healthy max (RTX 3080 Ti, torch 2.5.1, fp32) | backward defects |
|---|---|---|---|
| FRESH, seeded-perturbation pass (every fresh launch) | 1e-3 | 7.3e-06 over 6 perturbation seeds | move cells ×0.9 → 1.01e-02 REFUSED; dropped → 1.01e-01 REFUSED |
| REAL (trained) weights | 0.2 | 9.3e-04 on `ai_v14_01_base/final` (and isolated 4.9e-02 alpha seat-scorer outliers on 2 of 6 perturbed trained states) | move cells dropped → 1.00 REFUSED; team tokens ×0.5 → cosine 0.9928 REFUSED; **move cells ×0.9 → 1.0e-01 PASSES** (below the resolution on trained weights) |

  Measured end to end, the gate PASSES correct compiles on fresh and trained weights: fresh
  per-param max 6.6e-06, trained 9.3e-04, coverage 1/254. It REFUSES every defect in the table
  except the ×0.9 case on trained weights. (The probe loss and the coverage guard were deleted with
  the gate; `compile_gate_probe_test` now pins the per-parameter rule alone.)

**Cost.** The gate adds a readout and a train step per arm. dynamo's `cache_size_limit` (8) holds with
room to spare: measured 4 entries per code object after the gate plus alternating rollout (48,
no-grad) and train (512, grad) shapes, 0 cache-limit hits.

**Two MORE refusals, decided at startup, and the reasoning behind them is counter-intuitive enough
to be worth stating.** Recompiles here are NORMAL: `share_features_extractor=True` means one
extractor serves both paths, so `fe.forward` is called at batch=`n_envs` during rollout and
batch=`batch_size` during train, alternating forever. **Measured** (2026-08-14): alternating two
shapes converges after ~6 calls to a fixed 17 graphs and then never recompiles again (steady state
8.8 ms at batch 48 / 74 ms at 512). So `torch._dynamo.config.error_on_recompile = True` would crash
a perfectly healthy run on its second call, and `automatic_dynamic_shapes` is what makes the
two-shape case work rather than being the hazard.

The actual hazard is dynamo's **`cache_size_limit` (8)**: exceed it for one code object and dynamo
falls back to **eager SILENTLY** — precisely the invisible ~1.75x regression this flag exists to
prevent. One config gets there, decidable before training starts, and is fatal
(`check_shape_stability`, pure and unit-tested; its async-rollout rule left with the async collector,
U3; it judges the REAL update size since P10-E, F9):

| refused | why |
|---|---|
| the UPDATE size (`--rollout-target-samples` under the complete-game trigger, `n_steps * n_envs` when it is 0 — `rust_rollout.trigger.update_rows`, `main.train.compile_flags.update_rows_for`) not divisible by `batch_size` | the remainder micro-batch is a SECOND shape the one compiled R1 graph does not declare, replayed every epoch. The error names a concrete divisor to use instead rather than leaving you to do arithmetic |

The check used to take `n_steps * n_envs`, which `--rollout-target-samples` overrides (the production
recipe sets the target independently): a valid run — target divides, the product does not — had
`--compile-trainer` QUIETLY switched off by the auto default, and an invalid one — the product
divides, the target does not — passed this and died at the first buffer build. `_maybe_compile_trainer`
reads the MODEL's own `n_steps` / `batch_size` / `n_envs` when it has them (a resume restores the
checkpoint's; the argv's are inert), with the argv's trigger and target. It is decided at startup and
backed by two runtime refusals — `RustCollector._ensure_buffer` (every buffer build, both triggers) and
R1's dispatcher (one declared row count) — so a ragged micro-batch can neither reach the compiled
graph nor run eager (`gen3_r1_no_ragged_v1`).

Production is safe by arithmetic — the 98,304-row update (`rollout_target_samples` 98,304) = 48 x 2048
exactly, so exactly one micro-batch shape — but that was luck of the config until these guards existed.

⚠️ **The validation runs at a SMALL batch on a ZERO observation, and both halves are load-bearing.**

*Why not the train batch.* Because **validating at `batch_size` needs MORE GPU memory than training
does** — validation runs the arm eager AND compiled in one process with Inductor's workspace on top,
where training only ever needs one of them. At batch 4096 that exceeds the card. This was learned by
shipping it: it took down a gen-10 launch that had been running at 935 fps, first as a mystifying
`CUDA error: invalid configuration argument` and then, once the obs was valid enough to get further,
as the plain `OutOfMemoryError` underneath. The small batch is not a shortcut around an unexplained
bug; it is the only shape the check can afford. The honesty problem it was meant to solve — a
batch-64 ratio reading as if it were the production figure — is fixed by NAMING the shape in the log
line instead.

*Why zeros and not `torch.rand`.* A random float vector is **not a valid observation**, so it can drive the forward down branches no real battle
reaches. All-zero is the canonical "nothing known" state (every categorical id 0, every flag clear),
structurally legal, and it is what `snapshot._zero_obs` has always used on the opponent path. The
trainer path briefly diverged to `rand` for no reason and that is what disguised the OOM as a CUDA
config error.

*Where per-shape correctness was checked.* `torch.compile` compiles lazily PER SHAPE, so the graphs
production trained with were never the one that gate compiled; `compile_trainer_test::test_every_production_shape_agrees_with_eager`
asserted compiled == eager at each shape on a free GPU (deleted with the extractor-only compile —
the region gate judges the declared signatures themselves). Measured once against the live
gen-10 config on REAL observations off the rust bridge: **batch 48 -> 9.5e-07, batch 64 -> 7.2e-07,
batch 4096 -> 3.6e-06**, against a value scale of 2.111 — float32 rounding, not a wrong kernel.

**With the default ON, EVERY plain cuda run compiles the learner with nobody having typed a
flag**, so it is announced at startup when the auto default resolves to on
(`⚡ --compile-trainer ON by default (device=cuda)`, naming `--no-compile-trainer`), not
conditional on a launcher being attached.

**Mechanics.** Never compiles the module: `torch.compile(module)` returns an `OptimizedModule` and
prefixes every `state_dict` key with `_orig_mod.`, which would land in every checkpoint of the run and
make them unloadable by anything else. The regions compile FUNCTIONS over the module and are stored
off it (`model._compiled_micro_step`; the weak `policy._ROLLOUT_REGIONS` registry of the deleted R0 is gone too). **Runtime perf knob**: never versioned, never in `check_compatible`, NOT
inherited on resume — but with the AUTO default that means a flagless cuda resume gets it ON, so it
is `--no-compile-trainer` you re-pass each launch, not the flag.

Tests: `agents/model/compile_trainer_test.py` (the verdicts and the preflight are pure functions so
every refusal is testable without a GPU — a contract that needs a free card is a contract that gets
checked rarely), `agents/model/compile_regions_test.py` (the regions), and the extractor's own
compilability in `agents/model/extractor_compiles_test.py`.

### The compile sentinel (`gen3_compile_sentinel_v1`, 2026-09-28)

The owner's question: *"Is there no way to force it to raise or reject?"* — yes (first on torch
2.5.1; now torch 2.8 only), behind ONE adapter: **`src/agents/model/compile_control.py` is the only
runtime module that touches `torch._dynamo`**. It is also the only module that sets `torch._functorch.config` — the
per-version `_FUNCTORCH_CONFIG` row (`donated_buffer = False`, Lane K1b below), applied at `install()`.
Two failures were silent before it:

* **CACHE-LIMIT FALLBACK.** Dynamo keeps at most `cache_size_limit` (8) entries per CODE OBJECT per
  ID-matched `self`. The ninth is not compiled: `convert_frame._compile` logs ONE warning
  (`torch._dynamo hit config.cache_size_limit (8)`), raises a SOFT failure that
  `ConvertFrame.__call__` swallows, and that frame runs EAGER from then on — ~1.75x slower, no error.
* **LATE RECOMPILES.** A stall each, spending the cache toward the fallback.

**THE PHASES** (`compile_control.control()`, one per process):

| phase | where | what |
|---|---|---|
| 1 `reset()` | `_arm_compile_sentinel`, after `_apply_grad_checkpointing` | `torch._dynamo.reset()` — drops any graph compiled before the region is installed; the log line names how many cache entries it dropped (it read a stale "dropped 0" from a `gate()` only tests called until P10-E, which deleted `gate()` and its `entries_after_gate`) |
| (the startup gate) | `compile_regions.gate_regions`, after the install | R1 compiled vs eager on real rows; compiles freely (the cache-limit detector already listens, a hit here is fatal too); its graph is the one the prewarm then hits |
| 2 `prewarm(calls)` | same, after the region's install + gate | runs every DECLARED signature NOW (`compile_regions.prewarm_calls`: R1's), RNG-neutral (`fork_rng`), gradients zeroed |
| 3 `lock(where)` | **the end of startup** — `arm_compile_sentinel`, right after the prewarm, BEFORE the first real iteration (K6, 2026-09-30; was: the end of the first `train()`, which absorbed whatever iteration 1 compiled — see below) | the `fail_on_recompile` stance (torch 2.8; 2.5.1's `error_on_recompile` mode was deleted 2026-10-02) + a compile-START callback counting every frame compile. `🧊 [COMPILE LOCK] …` states the per-code-object entry count and the headroom to the limit. A rejection is a typed FATAL that NAMES the failing guard(s) (`UNDECLARED SIGNATURE — the failing guard(s): …`): 2.5.1's `RecompileError` lists them; on 2.8 the stance's rejection says nothing, so the sentinel replays the call ONCE under `error_on_recompile` to read them |
| — `release(why)` | when `learn()` returns or raises | the lock's scope is TRAINING: a forward after `learn()` on the compiled model is a new signature by design (the trainer's former in-process FINAL EVALUATION was the measured case, now deleted). MEASURED on the real trainer (the end-to-end smoke below): a still-locked sentinel broke it with `RecompileError`. Prints `🧊 [COMPILE LOCK] released …` with the post-lock compile count (must be 0) |
| 4 `stats()` | every update | TB `compile/graphs_total`, `compile/recompiles_after_lock` (**must stay 0**), `compile/cache_limit_hits`, `compile/max_cache_entries_per_code`, `compile/locked`, `compile/regression_flag`, `compile/train_ms_vs_lock_baseline` |

The checks run at every rollout end and every update end. `CompileControl.attach` registers the
`compile_sentinel` owner on the loop's DECLARED HOOK TABLE (`agents/training/loop_hooks.py`,
`gen3_declared_loop_hooks_v1`), at the `collect`, `update` and `learn` points, inner to the freeze
guard. The update hook runs the canary inside its guard and then `record` (which TAKES the per-update
region-call window). A duck-typed model with no table gets the same bodies as instance-attribute
wrappers. `_compile_control` and the table are in `_excluded_save_params`, so no checkpoint carries a
live sentinel. A `RecompileError` raised at a call site, a cache-limit hit, or any
compile that STARTS after the lock is `CompileSentinelError` → **`[CompileSentinel] FATAL …` and
`os._exit(FATAL_CONFIG)`** — an exception inside `learn()` would reach `model_build`'s generic
`except` and become a restartable CRASH, and a restart would replay it. The launcher also matches
`[CompileSentinel] FATAL` / `[CompileTrainer] FATAL` in `_FATAL_CONFIG_SIGNATURES`. There is no
checkpoint save on this exit (the last periodic checkpoint stands); the weights are not wrong, the
throughput is.

**Why the reset (phase 1) — earlier graphs spent production's slots** (the original measurement below was of a gate that ran BEFORE the reset; the reset now runs before the region is installed).** The cache is keyed per
code object (`Gen3FeaturesExtractor.forward.__code__`, `TeamTransformer.forward.__code__`, the
trunk split's resume frames …), shared by every instance and every `torch.compile` wrapper of that
code — so a separate compiled callable for the gate would NOT have isolated it in 2.5.1
(`compile_control_test::test_contract_cache_is_per_code_object_shared_across_compiled_wrappers`).
The gate compiles train/grad + train/no-grad at batch 64; production rolls out in EVAL mode at `n_envs`, so **none of the gate's entries is a
production signature** (entries at lock = gate's + production's, exactly additive below). MEASURED
2026-09-28 on the real `ai_v14_01_base/final_model.zip` policy (RTX 3080 Ti, torch 2.5.1+cu121,
rollout batch 48, train batch 512 as the dynamic-train proxy — the train graph is dynamic in batch
either way, so 2048 lands on the same entry; `~/.claude/jobs/compile_sentinel_2026-09-28/`):

| precision | gate entries / code obj | at lock WITHOUT reset | at lock WITH reset (ships) |
|---|---|---|---|
| fp32 `highest` | 2 | max **5** (headroom 3), 13 code objects, 39 entries | max **3** (headroom 5), 9 code objects, 23 entries |

**Startup cost of the reset: ~0.** Run concurrently under identical conditions (a TF32 arm, the one
that sat one signature from the limit without it — TF32 is since retired), reset+prewarm took 220.8 s
vs 223.0 s for the prewarm alone on top of the gate's cache: the production graphs are compiled
either way (before `learn()` now, inside iteration 1 before), and the reset re-compiles nothing
production would have reused. Absolute prewarm times on this box ranged 192–359 s across the four
measurements (first-ever compile of a graph vs Inductor's on-disk FX-graph-cache hits, a live
trainer's CPU load, 3 concurrent measurements) — **UNVERIFIED on a quiet box**; the ledger figure
for production is the first update after a restart at 147 s vs 57 s steady (Lane K3).

**The late-shape table** — every learner-process caller of the compiled forward (read from the code,
2026-09-28) and how each is handled. Dynamo guards batch (dim 0 goes dynamic at the 2nd size, but 1
is ALWAYS specialized), grad mode, `module.training`, the obs DICT KEY SET, dtype/device, the TF32
flag and read attributes (`grad_checkpointing`):

| caller | signature | handled by |
|---|---|---|
| rollout `policy(obs)`, end-of-rollout `predict_values`, frozen-φ / PBRS (the learner-process forwards; the Rust collector's rollout is T2's own `DecisionModule`) | eval · no-grad · any batch | EAGER at every batch since P10-E (no compiled rollout region R0: it was never called); takes no region route and compiles nothing |
| `evaluate_actions` | train · grad · `batch_size` | prewarm |
| `rank_probe` (first minibatch of every update) | train · no-grad · `batch_size` | prewarm — HOOK-FREE since K6 (`gen3_rank_probe_stash_v1`): it reads `last_trunk_tokens` / `last_value_cls`. Its old forward hooks were a guard (`len(_forward_hooks) != 0`) — THE iteration-1 signature the `8fc297a2` lock absorbed (found 2026-09-30 on the real trainer with the lock moved before iteration 1); on 2.8 the hooks were instead silently SKIPPED inside the compiled frame, so `rank/trunk_*` and `rank/value_cls_*` vanished on every compiled 2.8 run |
| capacity half-batch cosine (every 50 minibatches) | train · grad · other sizes ≥ 2 | prewarm at `batch_size // 2` makes the train graph dynamic |
| truncated-episode `predict_values` (a shaped-critic-era caller; the shaped critic is no longer trainable) | eval · no-grad · **1** | EAGER (no compiled rollout region) |
| cf terms, capacity velocity, win-prob episode-start read | — | already eager (`type(fe).forward(fe, …)`) |
| eval, snapshot ladder, the opponents | — | other processes (fresh dynamo per process) |
| the FINAL evaluation after `learn()` (in-process; DELETED, P6) | eval · no-grad · 1, a smaller obs key set | after `release()` anyway |

A launcher restart is a fresh process (a fresh dynamo cache, a new gate, a new lock; the ON-DISK
Inductor/Triton cache is the run's own and is reused only under a matching stamp — K3 below).

**END-TO-END on the real trainer** (2026-09-28, the T32b argv at TF32 made tiny — `n_envs` 2, a fork
of `ai_v14_01_base` into `~/gen3ai_archive/compile_sentinel_smoke/`, 3 iterations): gate PASS →
`reset` dropped the gate's 36 entries → prewarm (4 signatures, max 3 entries per code object, 229 s
incl. the reset) → `🧊 [COMPILE LOCK] after the first rollout + update: max 4 cache entries per code
object (limit 8, headroom 4)` — iteration 1 added one signature the prewarm did not (identified
2026-09-30, K6: the rank probe's forward hooks — now hook-free, and the lock moved before iteration 1)
→ iterations 2-3 clean (`train_ms`
12.8 s → 2.8 s), checkpoint saves after the lock fine, `Training complete`. It then exposed the
final-eval defect `release()` fixes (the final eval is deleted; `release()` stays); **the fix is unit-tested, not re-run on the card** (the card
was handed back to a live benchmark). Nothing toggles TF32 /
matmul precision / autocast after startup.

**THE BEHAVIOURAL BACKSTOP, independent of dynamo:** the baseline is the median `train/train_ms` of
the first 5 updates AFTER the first update of the process (which runs every diagnostic probe —
K2's cadence), then FROZEN; `train_ms > 1.4x` it for 3 consecutive updates prints
`⚠️ [COMPILE REGRESSION?] …` once and sets `compile/regression_flag` = 1. WARN, never fatal —
contention (an eval burst, a peer job) has the same signature.

**THE TORCH GUARDRAILS.** (1) An exact version table (`_SUPPORTED`, `2.8.0+cu126` → the
`stance` mode — the only row since the 2.5.1 row was deleted, 2026-10-02); an unknown torch REFUSES. (2) **A source-hash drift tripwire** (the SB3
`instrumented_ppo._verify_upstream_unchanged` pattern): `_SOURCE_HASHES` = {torch version →
{qualname → SHA256 of `inspect.getsource`}} over every internal the adapter rests on
(`cache_size.compute_cache_size` / `is_recompilation` / `exceeds_cache_size_limit`,
`guards.get_and_maybe_log_recompilation_reason`, `convert_frame._compile` (emits the warning),
`ConvertFrame.__call__` (the soft-fail swallow), `ConvertFrameAssert.__call__` + `Tracker` +
`eval_frame._debug_get_cache_entry_list` (the per-code read), `callback.CompilationCallbackHandler`,
`OutputGraph.compile_and_call_fx_graph` (the graph counter), `torch._dynamo.reset`,
`torch.compiler.reset`). Verified WHERE A LEARNER COMPILES — `compile_control.require_supported_torch` (the
version table + this tripwire) at the compile preflight and at every `CompileControl` construction, a
failure the startup path's FATAL_CONFIG with the reason — and in the routine tier; NEVER at import (P10-D,
2026-10-03: the import-time check ran in EVERY trainer through the cache stamp and the T2 service's graph
counter, so a non-compile run died uncaught on any torch but the recorded one; `compile_sentinel_scope_test`); a mismatch raises naming each drifted qualname with both hashes. Re-record with
`python -m agents.model.compile_control --record` AFTER re-reading the functions and re-running
(3) the behavioural CONTRACT tests in `compile_control_test.py` (the warning's text and logger, the
raw `RecompileError`, the start callback, the counter / entry list / reset, the per-code-object
cache). Hash = the code changed; contract = the behaviour changed; both are kept.

**Per torch version** (`_SUPPORTED`, `_SOURCE_HASHES` — one row each; the contract tests run on
the installed torch and assert its own semantics): **2.8.0+cu126 → `stance`** (20 hashed internals,
`len(_SOURCE_HASHES["2.8.0+cu126"])`; the 2.5.1+cu121 → `error_on_recompile` row, 13 internals, was
DELETED with HEAD's 2.5.1 support, K1 2026-10-02 — a 2.5.1 run resumes pinned to its own commit): the lock is `torch.compiler.set_stance(
"fail_on_recompile")`, which raises at the call site on ANY cache miss of a compiled callable — a
recompile AND the first call of a never-compiled one — compiling nothing; `release`/`uninstall` set
the stance back to `"default"` (nothing sets `error_on_recompile` any more: the lock and `release`
no longer touch it, and its only user is the rejection DIAGNOSIS replay, a context-managed `patch`).
**What moved in 2.8** (re-read at the source): `exceeds_cache_size_limit`
→ `exceeds_recompile_limit`, and `config.cache_size_limit` is now an ALIAS of `config.recompile_limit`
(the warning reads `hit config.recompile_limit (8)`; the soft failure is `RecompileLimitExceeded`,
still swallowed into eager); `get_and_maybe_log_recompilation_reason` → `..._reasons`; the
compile-START callback takes a `CallbackArgs`, fires for LAZY_BACKWARD / TRITON_AUTOTUNING /
CUDAGRAPH_RECORDING too, and now runs AFTER the recompile check — so it no longer sees a REJECTED
recompile, and a caller's `except Exception` could have hidden one. **`CompileControl.wrap_compiled`**
closes that: the compiled region is installed through it and records any lock
rejection (sticky, `rejected_after_lock`) before it propagates. The stance's rejection is a plain
`RuntimeError` (text `_STANCE_REJECT_TEXT`), which `find_recompile_error` matches. Measured on the
real extractor on 2.8 (fp32, prewarm B=48/256/128): lock at 3 entries on 1 code object (headroom
5 of 8), steady iterations 0 compiles / 0 rejections, an undeclared batch-1 eval signature after the
lock → the typed FATAL. **Not done in K1:** `fullgraph=True` and 2.8's
`config.fail_on_recompile_limit_hit` (a HARD cache-limit failure) — both belong to K6.

**Tests** (`src/agents/model/compile_control_test.py`; CPU with dynamo's `eager` backend — the
cache, guards, limit and recompile path are dynamo's, independent of the backend): >8 guard
signatures trip the detector → FATAL (also before the lock); after the lock a new shape raises the
typed FATAL at the call site; a SWALLOWED `RecompileError` and a late FIRST compile are still
caught; the healthy two-shape alternation locks and runs clean; **the sentinel changes no numerics**
(outputs + gradients bit-identical on vs off); `attach` locks after the first update and exits
`FATAL_CONFIG`; the version refusal and the hash tripwire; the contract tests;
(the CUDA lock-headroom test went with the extractor-only compile, K1).

### FUNCTIONAL MASKING — the K8 prerequisite (`gen3_functional_masking_v1`, 2026-09-30)

sb3's `MaskableCategorical.apply_masking` pops its cached `probs` out of `__dict__` and re-runs
`Categorical.__init__` — an object mutation dynamo cannot trace (17–19 graph breaks per update in the K8
inventory) — and every `torch.distributions` object validates its logits and samples with host reads.
`agents/model/masked_categorical.py` is the same arithmetic as plain tensor ops, op for op (the
unmasked path's double normalisation, masking from the single-normalised `_original_logits`, the
`-1e8` floor, the masked entropy's `where`, the same `multinomial` call for a sample), and the
policy's two hot paths use it: `forward` (the rollout) and `evaluate_actions` (the learner; the
ride-along readers get a `MaskedPi` answering `.distribution.logits` / `.probs`).
`get_distribution` keeps sb3's object for its callers (`predict`). Pinned BIT-IDENTICAL
to sb3 (`masked_categorical_test`, both torches); the K9 learner golden is unchanged.

### K8 — DECLARED COMPILE REGIONS (`gen3_declared_regions_v1`, torch 2.8, 2026-10-01)

On torch 2.8 `--compile-trainer` no longer compiles the extractor alone: `arm_compile_sentinel`
installs the learner's DECLARED REGION (`agents/model/compile_regions.py`, the table `REGIONS`),
compiled `fullgraph=True, dynamic=False`, so a graph break INSIDE a region is a startup
`CompileTrainerError` naming dynamo's reason (FATAL_CONFIG), never a silent split:

| region | callable | declared signature | why the boundary is here |
|---|---|---|---|
| R0 `rollout_forward` — **DELETED 2026-10-03 (P10-E, owner-approved; see below)** | was `policy.rollout_core` | was eval / no-grad / `n_envs` | the Rust collector serves every rollout forward through T2, so the learner process never called it |
| R1 `learner_micro_step` | `instrumented_ppo.micro_step.micro_step` — `evaluate_actions` + fold steps 1-3a; its backward is AOTAutograd's from the same graph | train / grad / `batch_size` — ONE row count; a ragged micro-batch is REFUSED by the dispatcher (`gen3_r1_no_ragged_v1`) | the accumulation group and the optimizer step are per GROUP; the host read follows it |
| R2 rank probe | NO forward: reads R1's own stashes on the first micro-batch, spectra on the device (`rank_metrics.rank_probe_from_stash`) | — | — |
| R3 optimizer step | `clip_grad_norm_` + AdamW + `zero_grad`, EAGER by decision | — | 0.04 s per update; the KL->LR controller moves lr every update, so a compiled step keyed on it is a new signature per update |
| *eager, declared* | the batch build (micro-batches staged to the device by a prefetch thread, or one resident device copy per update — `--device-batch`, `instrumented_ppo/device_batches.py`), label alignment, the per-micro packed read (`micro_step.pack`; deferred to the optimizer step's one transfer, `instrumented_ppo/host_reads.py`, `gen3_batched_host_reads_v1`), the fold's eager tail, every cadence diagnostic, logging | — | host-side or data-dependent by nature |

The sequence: reset → `install` (removes any instance-level compiled extractor forward, so every
other caller of the extractor runs eager) → `gate_regions` holds R1 to eager on REAL rows (R1
ALWAYS on the K9 golden's labelled buffer — `gen3_r1_golden_rows_always_v1`, P10-C: a key the run
declares that the golden lacks takes its DECLARED placeholder (`compile_regions.fill_value`: the label
inventory's `host_const` value — the fork arm's `fork_pg_m` = 1.0, a multiplier on the policy term — or
the extra-obs-keys registry's zeros), a golden key the run does not declare is dropped (its term is off
in that run), and anything else — a shape mismatch, an undeclared key, an unreadable buffer — is
FATAL_CONFIG naming it; the old zero-label fallback blanked the critic / intent / belief losses, so
every `--fork-fraction > 0` run judged 176 parameters where the golden rows judge 219, measured CPU
B = 256 on the golden learner. When a key was filled, R1's judged parameter set is held to the golden
rows' own on the same weights (one extra eager micro-step without the key, the static resolved from a
buffer without it; `gen3_r1_judged_set_v1`): a parameter the golden rows judge CLEARLY (gradient-norm
ratio > `JUDGED_SET_BAND` = 2x the floor) that the run's rows do not judge is FATAL_CONFIG — the band
keeps a parameter a hair from the floor (the smallest judged one reads 1.01e-3) from ever deciding it.
The gate line reads `R1 judged set >= the golden rows' own: all N parameters …`. The loss, the gradient
over every policy parameter at the cosine and per-parameter bars, the per-parameter bar chosen by weight regime
(`train_verdict` takes NO default bar — a caller that names none is refused, P10-E; the retired extractor gate's TF32-era 1e-3 / 0.2 constants are deleted), on a seeded perturbation too when
the weights are fresh; at fp32 `highest`, the only precision, a process at any other is refused)
→ prewarm exactly the declared signature → LOCK (K6) → the canary at update 10 and every 100 updates (R1's loss and gradient on the golden rows, EVERY canary; confirmed in the same update before it FATALs — `designs/training/learner_lifecycle.md`)
(`compile_canary._regions`). The region is the ONLY compiled learner surface: on torch 2.5.1
`forward_guard`'s weakref lookup breaks `fullgraph=True`, and HEAD no longer runs 2.5.1 at all (the
extractor-only compile kept for it was deleted 2026-10-02, K1; `utils/torch_floor.py`).

**R0 (the compiled rollout region) was DELETED (P10-E, 2026-10-03; owner approved the change to the
declared region table).** On the only env core the Rust collector serves the trainee's rollout forward
through the T2 inference service (`agents/inference/service`, its own `DecisionModule`, gated on every
served bucket at load), so the learner process never called R0 — a production run's per-update route
counts were exactly R1's micro-batches × epochs (480 = 48 × 10 at N = 256, `lifecycle/compiled_region_calls`),
R0 contributing none. It was nevertheless compiled, gated, prewarmed and canaried at every launch, so
its canary verdict checked a graph nothing ran (a vacuous check, standing rule 8) and every launch paid
its compile. Deleted with it: `install_rollout_region` (its only other caller, the M5 throughput harness,
went with U3), the compiled `policy.forward` → region dispatch and the weak `policy._ROLLOUT_REGIONS`
registry, the batch-1 eager route (`EAGER_BATCHES`, `gen3_batch1_eager_v1`), R0's gate arm (the
decision-readout verdicts and the perturbation-ladder climb for a collapsed critic), its prewarm call and
inventory row, the canary's decision-readout arm (and its `compile/canary_max_abs_*` scalars, and the
decision-only canary: `GRAD_EVERY`, `compile/canary_grad_checked`), `region_calls`' R0 rows, and their
tests. KEPT, eager: `Gen3DualHeadMaskablePolicy.forward` and `rollout_core` — callers: the toy-collector
test kit (`rust_rollout/testkit.py`), `masked_categorical_test` (the bit-identical pin against sb3's
distribution object), the compile inventory's `fullgraph_check` / worker (a whole-step trace over the
eager forward), and `rollout_core`'s second caller, `compile_regions.weights_regime` (the freshness read
that picks R1's per-parameter bar); `predict` / `get_distribution` (`play.py`, the prober, the eval
workers) never went through `forward`. The T2 service builds its own `DecisionModule` and calls neither.
**MEASURED (P10-E, 2026-10-03; RTX 3080 Ti, torch 2.8.0+cu126, fp32 `highest`, two real
`--arch production` launches under `gpu_lock` with `--no-pin`, a scratch archive on disk, a COLD compile
cache each, N = 256, B = 2048, the 98,304-row update; the control is `900de5e2`'s code, the arm is the
deletion):**

| | with R0 (control) | R0 deleted |
|---|---|---|
| child attached → first TB event file / `model_config.json` (the compile lock is just before it) | 410 s (08:13:52 → 08:20:42) | 345 s (08:51:13 → 08:56:58) — **−65 s, −15.9 %** |
| the sentinel's own `reset + prewarm took` (it includes the startup gate) | 171.0 s | 117.1 s — **−53.9 s, −31.5 %** |
| graphs in the process / code objects holding entries at the lock | 5 / 2 | 4 / 1 |
| `lifecycle/compiled_region_calls` per update (updates 2-27) | 480 (= 48 micro-batches × 10 epochs); the canary update 482 | 480; the canary update 481 |
| the first update's count | 528 | 528 (the same 48 more R1 calls — a first-update-only pass, cause not traced) |
| `lifecycle/eager_fallback_calls`, `compile/recompiles_after_lock` | 0, 0 | 0, 0 |
| steady `update_wall_s` | 40.3-40.5 s | 40.3-40.7 s (unchanged: R0 was never in the update) |
| the canary at update 10 | pass, 0.41 s (R0 + R1) | pass, 0.37 s (R1: `R1 [trained weights]`, 199 parameters judged, max per-parameter error 4.25e-05) |

R0 contributed ZERO of the 480 region calls (the arm's counts are the same, with nothing deleted from
R1's path), which is the finding's proof: it was compiled, gated, prewarmed and canaried for nothing.
The saving is the compile + gate + prewarm time of one region, paid once per launch AND per interval /
crash restart (a restart compiles into the run's own warm cache, so its saving is smaller). The new
reset line also named, for the first time, what the reset drops at a production launch: `dropped 3
cache entries compiled before the region was installed [decide (engine.py:41) x3]` — the T2 inference
service's `decide` (`agents/inference/service/engine.py`), compiled when T2 builds, before the sentinel
runs; the line names the code objects it drops (read on the final code's own short launch, 114.6 s
`reset + prewarm`). Harmless today: T2's served graphs are CAPTURED CUDA graphs, replayed without dynamo, so
a dropped dynamo entry is never looked up again; a T2 call that ever reached dynamo after the lock would
be an undeclared signature (a typed FATAL), not a silent one.

A consequence: a resume onto a checkpoint whose win-prob head is saturated (a collapsed critic) is no
longer refused by an R0 "vacuous on every perturbation rung" verdict — R1's verdict is the loss and the
gradients, not V's spread (pinned by `compile_regions_test::test_a_COLLAPSED_critic_is_judged_by_R1_like_any_other_weights`).

**One startup gate per region** (`gen3_one_gate_per_region_v1`, 2026-10-01). The trainer runs no
extractor-only gate: `preflight_compile_trainer` compiles nothing, and the sentinel installs and gates
the regions (on 2026-10-01 behind `regions_follow` / `regions_requested`, both deleted with the
2.5.1 path on 2026-10-02). That gate compiled the extractor alone and judged it, but the regions
uninstall that compile, so it was judging a graph production never runs. It cost about 2 of the ~5
startup minutes (the R1 bar fix's restart proof, 2026-10-01).

The checks that remain on 2.8, each with its measured bar and the fault it catches:

| check | bar | catches |
|---|---|---|
| the region's startup gate | R1: loss rel ≤ 1e-4, gradient cosine ≥ 0.9999, the regime-selected per-parameter bar. Fresh weights also get a perturbed pass | a miscompile at t = 0 |
| the in-run canary | update 10, then every 100 updates, R1's loss and gradient on the golden rows, confirmed before it FATALs | a graph that changed by t = N |
| K9(b), the behaviour check | the learner's log π against the rollout's stored log-prob | stale weights, an eval/train-mode difference, a rollout/learner obs mismatch |
| the no-silent-eager guards | see above | a partly uncompiled learner |
| the learner golden (test time) | — | a change in what the update computes |

The extractor gate's speed refusal (`check_speedup`, measured at batch 64; deleted 2026-10-02) has
no counterpart at startup. The update-wall drift warning, the performance-shape tests and the milestone update benchmark cover speed (below).

**Measured** (2026-09-30/10-01): R1 on the production surface traces as ONE graph on 2.8 (CPU,
`aot_eager`, compiled loss bit-equal to eager); `compile_regions_test` (routine, CPU, dynamo `eager`
backend): graphs == regions x signatures (2 when R0 existed, 1 now), one cache entry per region code object, 0 compiles and
0 rejections after the lock across two real updates and an eager rollout forward (at batch 4 and 1); a host read planted inside R1 is the typed startup FATAL. On CUDA (Inductor, fp32, the
golden learner): the gate passes, 2 graphs, 0 compiles after the lock; NON-VACUOUS by an element-wise
read (`~/gen3ai_archive/k6_k8/r1_nonvacuous.log`): dynamo captured one graph in the compiled arm and
none in the eager arm; compiled vs eager loss |Δ| 6.0e-8 (B=16) / 1.2e-7 (B=2048), gradients differ
in 2.06M of 3.07M elements by at most 3.6e-7, against an eager-vs-eager floor of 7.5e-9 — the gate's
printed "loss rel 0.00e+00" on one earlier run was a one-scalar coincidence, not a comparison of a
function with itself.

**ACCEPTED (2026-10-01; `designs/research_state/measurements/k6_k8/acceptance/`):** the compile
inventory's time stage, torch 2.8 + TF32, arm C on the learner benchmark's pinned buffer, interleaved
A (pre-regions) / B (K8), n = 2 each — one production update **46.7 s → 36.3 s (0.778x, bar <= 0.80)**;
the compiled share of the update's wall (profiled, diag-skipped) **63.4% → 90.0% (bar >= 80%)**, of
kernel time 85.2% → 98.1%; host scalar reads 66,784 → 9,942 per update; GPU idle in the profiled
epochs 2.4 → 0.57 s; 0 compiles after the lock. A's spread (45.3 / 48.0 s) is wider than the margin
under the bar, so the verdict is "met" at n = 2, not "met with room".

**No silent fall-back to eager** (`gen3_no_silent_eager_v1`, 2026-10-01; owner: "I just most care
that we don't have silent performance regressions from it not being fully compiled").

The defences that run in every run are the PRIMARY ones.
- **Every region call goes through its dispatcher** (`compile_regions.install`), which takes the
  COMPILED route. There is no declared eager route any more (R1's ragged micro-batch is refused;
  R0 and its batch-1 route are deleted); a future one counts under `<region>_eager_<why>`
  (`agents/model/region_calls.py`).
- **A compiled route that silently ran eager is a typed FATAL.** Each region's body calls
  `note_eager_body`, which is a no-op inside a dynamo trace, and a compiled graph never re-enters the
  Python body. So if the body runs during a compiled-route call, the region ran EAGER. That covers
  dynamo disabled, a swallowed compile error, a skipped frame, and the `force_eager` stance.
- **A ragged micro-batch is a typed FATAL at R1's dispatcher** (`gen3_r1_no_ragged_v1`; it used to have a
  declared size of one per epoch and run eager, a route the collector's refusal made unreachable).
- **The lock refuses the three switches** that make dynamo run eager without an error. With the lock
  held, each of these is a sentinel violation (`CompileControl.violation`):
  - `torch._dynamo.config.disable`;
  - `suppress_errors`;
  - a stance other than `fail_on_recompile`.
- **The run asserts its compiled inventory** after the prewarm: one cache entry on R1's code object
  (`assert_inventory`). On 2.8 a policy without the micro-step says
  `⚠️ [CompileRegions] NOT installed` instead of compiling less silently.
- **Every update records** `lifecycle/compiled_region_calls`, `lifecycle/eager_fallback_calls`,
  `lifecycle/eager_share` and `lifecycle/update_wall_s`.
- **The update-wall drift warning** (`TrainMsWatch`) WARNs when `train_ms` exceeds 1.4x the run's own
  median of its first 5 post-lock updates for 3 consecutive updates. It is never fatal, because box
  contention looks the same.

The routes the audit closed or declared:

| route | where it leads now |
|---|---|
| cache-size-limit fallback | a typed FATAL (existing) |
| guard failure after the lock | a typed FATAL (existing) |
| `torch.compiler.disable` or a graph break inside a region | refused at startup, because the regions compile `fullgraph=True` |
| a region skipped by a flag | the loud line above |
| a ragged micro-batch | refused by R1's dispatcher (a typed FATAL) |
| batch 1 | no route: R0 is deleted; the learner's rollout forward is eager at every batch |
| dynamo disabled, swallowed errors, `force_eager` | a typed FATAL |

The CPU opponents' eager fallback is in the env workers, not the learner, and it already says so.

**Performance is checked by STRUCTURE plus a milestone benchmark — there is no wall-clock gate** (owner, 2026-10-02).
`compiled_perf_guard_test` (one production update held to a banked time, a compiled share >= 0.80) was RETIRED: a timing
test is noisy, needs an idle box and goes stale, and it went blind the day K2 retired TF32, its only baseline. What
replaces it, in every routine gate: the deterministic tests of the properties that make the update fast — the inventory
== declaration, 0 compiles after the lock, the no-silent-eager FATALs, **every micro-batch of a real update on R1's
compiled route (route counts)**, **a pinned host-scalar-read count per update**, **attention on the fused SDPA kernel, not
MATH**, the micro-batch never re-copied whole per micro-step, no CUDA stream / optimizer built after startup (the audit,
each test and what it catches: `designs/ops/testing.md` "Performance-shape tests"; the new ones are
`src/agents/model/update_performance_shape_test.py`). The run-time defences stay primary.

At milestones (after a torch upgrade, before a baseline run, on suspicion) run the **update benchmark**:
`python -m main.compile_inventory run --stage time --device cuda --keep-prewarm --unbracketed` under `gpu_lock` (≈ 6.5 min).
Lane C (2026-10-02) made it work again: since `gen3_update_fit_v1` the trainer runs one real `train()` at STARTUP (the CUDA
fit check's dry update, on a 4,096-row fixture at the workers' `--n-envs 2`), and the benchmark tools, which replaced
`train` for the whole process, took THAT as their measurement and exited before the pinned buffer was restored
(`learner_benchmark.learn_loop_only`; K2's un-banked fp32 baseline). Its read that day, fp32 'highest', K8 regions,
pinned 98,304 rows: **40.17 s, n = 5, spread 0.31%, compiled share 0.892** — and a planted regression (R1 on its eager
route, uncounted) read 116.5 s with share 0.000, which the route-count test above fails deterministically
(`designs/research_state/measurements/k6_k8/update_benchmark_fp32/`; `acceptance/perf_baseline.json` is kept as that
history, no test reads it). A read, not a baseline: it goes stale.

**R0's mask input is ONE declared dtype** (`gen3_r0_mask_dtype_v1`, 2026-10-01). The R0 dispatcher
turns the action masks into a bool tensor on the learner's device before the compiled core. The env
workers hand the rollout int8 numpy masks; the gate and the prewarm passed numpy bool, and dynamo
guards a numpy input's dtype. So from K8's regions landing (`f3878594`) until this fix, the first
rollout of every real torch-2.8 `--compile-trainer` launch on the python env core was an UNDECLARED
signature, and the sentinel FATAL'd at rollout end (`___from_numpy(action_masks) dtype mismatch.
expected Bool, actual Char`). The launcher restart proof found it. The acceptance read never
rolled out (it times `train()` on a pinned buffer), so it missed it. `masked_categorical` already
read every mask form as `as_tensor(m, dtype=bool)`, so no number changes.
`compile_regions_test`'s inventory test sends int8, bool and float masks after the lock.

**R1's per-parameter gradient bar is chosen by WEIGHT REGIME** (`gen3_r1_param_bar_by_regime_v1`,
2026-10-01; `compile_regions.R1_PARAM_BAR`, `weights_regime`). The gate first had the extractor
gate's FRESH bar, 1e-3, which was measured on that gate's 64-row probe loss. On R1 (the real PPO
micro-step at the production micro-batch, fp32 'highest') that bar failed every healthy start: an
fp32 resume of arm C read 2.47e-3, and a fresh launch at B = 2048 read 1.0e-2 to 2.5e-2.
**The sizing study's resume FATAL was noise, not a miscompile.** A matched-noise control
(`designs/research_state/measurements/k6_k8/r1_noise/`) compared each gradient with a float64
eager reference. On C's weights, CUDA EAGER is the inaccurate side (2.47e-3 vs fp64 on
`history_events.itemtr_emb`); the compiled gradient is within 1.7e-5. On fresh weights the
compiled error (2.5e-2 on `belief_head.species_head`) sits inside CPU eager's own (3.1e-2).

How the bars are set:
- `weights_regime` reads "fresh" when the legal log-probs on the fixture are constant within every
  row (the zero-init pointer head). Every resume, fork, restart and canary reads "trained".
- Each bar = 4x the largest healthy gate reading in its regime: fresh 0.100, trained 9.9e-3.
- A 10% backward error on a path reads about 0.1. The trained bar refuses it with 10x margin.
- The fresh bar would not refuse it, so a fresh launch ALSO runs R1 on the declared ladder's first
  seeded perturbation, judged at the trained bar. Perturbed-fresh weights read at most 2.3e-5.
- The in-run canary's R1 check uses the same selection. It had the same 1e-3 bar and would have
  FATAL'd every fp32 run at update 100.
- **An UNMOVED parameter is judged by its OWN regime** (`gen3_r1_unmoved_param_v1` +
  `gen3_r1_unmoved_init_v1`, 2026-10-04; `compile_regions.unmoved_parameters`, `r1_rungs`).
  `weights_regime` classifies the MODEL, but conditioning belongs to each parameter. A judged
  parameter is UNMOVED iff every element is BIT-IDENTICAL to its value at the run's fresh build, or
  every element is exactly 0.0. Both tests are categorical (sha256 equality with the init record;
  `== 0.0`), with no tolerance, so no input sits near the rule's boundary. An unmoved parameter is
  judged at the FRESH bar on the live weights. R1 then runs again with ONLY the unmoved parameters
  moved off their init (name-keyed seeded noise added, the ladder's first rung, restored bit-exactly),
  and EVERY parameter is judged at the TRAINED bar on that rung, so a miscompile on their paths is
  still refused tightly. The startup gate (a resume) and the canary share the rungs; the live rung's
  rule line names the rule in force (`[unmoved rule: …]`). The canary's TB scalar
  `compile/canary_unmoved_params` counts them.
  - **The INIT RECORD.** `compile_regions.record_param_init` (a `@startup_builder`) runs once, inside
    the trainer's fresh construction (`model_build.construct_fresh_learner`), before anything trains.
    It stores `{policy parameter name: sha256(dtype, shape, bytes)}` as the model attribute
    `param_init_sha256`. That is plain JSON data, so SB3's `save` writes it into every checkpoint and
    `load` restores it. Cost: 387 entries on the X5 blob arm with the X26 heads, about 87 KB of
    Python strings on the CPU and 46 KB in each checkpoint. Nothing on the GPU is kept; a check copies
    one parameter at a time to the host to hash it (the largest is 2.8 MB, 24 MB in all).
  - **The RESUME rule.** The record travels with the WEIGHTS: a restart, resume or fork reads the
    lineage's own fresh-build record from its checkpoint, and nothing ever recomputes it after a load
    (a rebuild at the same seed is not trusted to be the run's init). A checkpoint with NO record
    (saved before `gen3_r1_unmoved_init_v1`, or built outside the trainer's fresh construction, e.g.
    the K9 golden or `main.fresh_checkpoint`) gets the zero rule alone, which is 8b8fbac0's behaviour.
    The zero rule stays even with a record: an all-zero parameter has the fresh conditioning whatever
    its history.
  - **What the init rule adds.** In `rb_x5ab_oracle_sp_s1001`'s FATAL checkpoint (update 10, 987,904
    steps), 17 of 387 parameters are bit-identical to the fresh build at seed 1001. Ten are the SB3
    value tower that the win-prob critic never read (DELETED since the version break's part 2, audit F1; dead in every arm: the blob seed at 15.0M has
    the same ten, and R1 gives them no gradient, so they are never judged). Seven are the oracle's:
    `belief_head.{species_head, moves_head}.{weight, bias}`, `belief_head.norm.{weight, bias}` and
    `belief_slots.unknown_slot_emb`. The zero rule saw four of the seven. The init rule adds the
    ortho-init `moves_head.weight`, the LayerNorm weight (1.0) and `unknown_slot_emb`
    (`measurements/oracle_canary_2026-10-04/unmoved_init_species_fatal.json`). On the CPU, eager fp32
    against float64 on the live weights reads 9.4e-4 on `moves_head.weight` and 6.4e-4 on
    `unknown_slot_emb`, under the 9.88e-3 trained bar but over 1,000x a trained parameter's 4e-7.
    On the init rule's unmoved rung each reads at most 6.5e-7, and the worst of 199 parameters reads
    1.7e-5 (`unmoved_init_rungs_cpu_B2048.jsonl`).
  The case behind it: under `--oracle-reveal` the blob arm's species belief labels are all PAD, so
  the zero-init `belief_head.species_head` gets no gradient in training, yet the golden rows still
  supervise it. Under the old rule it FATAL'd `rb_x5ab_oracle_sp_s1001` at update 10 (1.40e-2 against
  the 9.88e-3 trained bar). That was fp32 NOISE, not a miscompile. CPU EAGER fp32 errs 1.51e-2
  against float64 on the zero head, all of it in 5 species columns of dL/dlogits, while the CPU
  compiled arm errs 4.4e-7. Moving the head off zero takes CPU eager to 4.1e-7.
  `designs/research_state/measurements/oracle_canary_2026-10-04/`.
  The LIMITS: the rule is per PARAMETER, so a parameter only PART of which never moves (an
  embedding's never-seen rows, a Linear's columns on an always-zero input) is judged at the model's
  regime. A run resumed from a checkpoint with no record does not see a dead non-zero-init
  parameter. Neither has tripped.
- **Every comparison PROVES its two arms independent** (`gen3_gate_independent_arms_v1`, 2026-10-05,
  F-XC-4; `compile_regions._r1_pair`, every rung of the gate and the canary). `region_calls`'
  counters must show the compiled arm dispatching the compiled route ONCE while running R1's Python
  body ZERO times, and the eager arm the reverse. Anything else is a typed refusal before any verdict: a gate whose two sides
  share their arithmetic (the compiled slot holding an eager wrapper, an "eager" reference that
  dispatches the graph) would read 0 everywhere and PASS. Counters, not identities: they record what
  each arm EXECUTED.
- **A non-finite gradient is NAMED, never read as a cosine** (`gen3_gate_nonfinite_named_v1`).
  `train_verdict` refuses a NaN / inf gradient on either arm FIRST (`NonFiniteGateArmError`, naming
  the parameters, compiled arm first), and `_cos` returns NaN, not 0.0, when either side is non-finite.
  Before this, a NaN norm fell through `na > 0` and the gate FATAL'd "cosine 0.000000 < 0.9999".
- **What the F-XC-4 readings were** (`designs/research_state/measurements/x5_fxc4_compile_gate_2026-10-05/`,
  CUDA, HEAD `e8008c2d`). Every `--belief-tokens fixed_mass` gate read loss rel `0.00e+00`, and one
  launch of four at `889add9d` FATAL'd at cosine 0. Neither meant the gate compared a thing with
  itself: the arms are distinct callables with distinct outputs, and R1 under fixed_mass IS one
  compiled graph (dynamo `unique_graphs` 1, no graph break, the inventory's one entry, the eager-body
  guard silent). The cosine 0 was a **NON-FINITE compiled backward**: 41 parameters NaN (the
  `PokemonEncoder`, the five embeddings and the belief heads that reinject into it), eager finite, on
  every rep of every unmodified compile at HEAD (8 of 8 processes). `aot_eager` and
  `aot_eager_decomp_partition` are finite; Inductor is not. `CompilerBisector` names the joint-graph
  pass `remove_noop_ops`: with none of its 5,809 removals the gradient is finite, with the first 1,973
  it is finite, and the 1,974th (an `aten.alias` of `DamageOperator._p_outspeed`'s sigmoid in
  `_outgoing_attacker_matrix`, the D2 edge family) makes it NaN — yet keeping every alias is still NaN,
  so the trigger is fusion-dependent, not one node. A torch 2.8 Inductor defect on the fixed_mass graph;
  blob's graph does not trip it (the blob seeds' gates and canaries pass on CUDA). The bit-equal loss is the forward's own arithmetic (one fp32 scalar;
  1 ulp apart in other runs), and the gate line's "train features" for R1 IS that loss, so it was one
  observation, not two.
- **The mechanism, and the fixed_mass-only fix** (`gen3_fm_index_max_v1`, 2026-10-05,
  `designs/research_state/measurements/x5_fxc4_nanfix_2026-10-05/`). A per-kernel NaN scan of the
  generated code (a host count of NaN in every tensor argument before and after every Triton kernel;
  codegen-only, so fusion is the real compile's) names the FIRST NaN-writing kernel: a backward
  reduction kernel that RECOMPUTES fixed_mass's 400-wide incoming damage sweep (the full candidate
  axis, which only fixed_mass prices; blob's is the top-6) from finite inputs and takes the
  backward of its ten channel maxima. `amax`'s backward is `grad · (x == amax) / Σ(x == amax)`. The
  partitioner recomputed `x` in the backward kernel, and Triton contracted its `a·b + c` chains into
  FMAs differently there than in the forward kernel that took the max (a different fusion), so on some
  rows NO element equalled the saved max: tie count 0, gradient 0/0 = NaN. With Triton's FMA
  contraction off (`TRITON_DEFAULT_FP_FUSION=0`, a diagnostic only) the same graph is finite — that is
  the mechanism; `remove_noop_ops` and the `_p_outspeed` alias only moved which nodes were recomputed.
  **The fix:** under fixed_mass those ten maxima are selected BY INDEX (`damage_op.max_by_index`, a
  gather at the detached argmax): the VALUE is bit-identical to `amax`, and the backward is a scatter
  at the saved index — no float equality left to break. The blob arm keeps `amax`, and its R1 and T2
  compiled code is byte-identical (hashed before / after in the measurement). The eager change is a
  TIE rule: the whole gradient goes to the first maximal element where `amax` split it evenly; the
  fixed_mass K9 golden did not move. On CUDA the fixed_mass startup gate now PASSES in 4 of 4 fresh
  processes (per-parameter rel err ≤ 1.4e-5 against bars ≥ 0.0099; cosine ≥ 1 − 4e-10), and a compiled
  regime-A launch passes its gate, the update-10 canary and 11 logger dumps.
  ⚠️ **The class is CLOSED on the gradient path since the version break's part 2 (audit F6a):**
  `max_by_index` moved to the leaf module `agents/model/index_max.py` and is THE spelling of every
  gradient-path value-reduction max in the forward, in every configuration (the blob arm is deleted);
  `amax` / `amin` remain only off any gradient path (`designs/model/op_contracts.md` "The op's MAXIMA").
  A NEW max over candidates on a gradient path uses it; a NaN FATAL of this shape elsewhere (a reduction
  this rule missed) gets the same cure, never a looser bar.

`compile_regions_trained_cuda_test` (GPU tier) runs real TRAINED weights through the real gate — arm C's until
2026-10-15, when that pre-era checkpoint stopped loading (23 size mismatches after the end-state adoption); now the
`production` baseline, read by name through `agents.training.baselines`, at fp32 and under the bf16 rule, with a
CPU test that loads it STRICT into the learner golden so a stale fixture is caught without a GPU. The
fp64-REFERENCED form of the gate (compiled no worse than k x eager's own error vs float64,
self-calibrating) was weighed and deferred to `designs/ops/TASK_BACKLOG.md` T16. One fp64 R1 pass at
B = 2048 costs ~11 s on the CPU, so time is not the obstacle. Two other things are:
- **The model is an fp32 program.** It casts with `.float()` in ~300 places and hard-codes float32
  in others. The measurement's fp64 arm needed two process-global monkeypatches (sb3's
  `preprocess_obs`, `torch.Tensor.float`), which a production gate cannot carry.
- **CUDA eager alone is no envelope.** On `ai_v14_05_lbat_l95`'s final the compiled error is 1.3e-3
  vs fp64 where CUDA eager's is 1.0e-5 but CPU eager's is also 1.3e-3. Those are two correct fp32
  implementations 100x apart, so the rule needs CPU eager in its envelope as well.

**R1's DECLARATION (`gen3_r1_declared_levers_v1`).** R1's signature is more than the batch's shapes:
dynamo specialises on its static flags (`MicroStatic`) and guards the observation dict's KEYS. Both
are DECLARED at startup from the resolved config and held to that declaration every update. R1 takes
NO per-update tensor dict: every value it multiplies by is a static flag or coefficient.

| lever | flag | in the batch | how it is declared |
|---|---|---|---|
| fork mask | `--fork-fraction > 0` | `fork_pg_m` (obs key) | the key is in the env's obs space at startup; injected rows make a RAGGED last micro-batch, which runs eager by declaration |
| label keys | `--move-belief-mode` | `known_moves` | in the obs space at startup (they ride the dict; the forward reads none) |

`compile_regions.install` records the declaration (`R1Declaration`: static flags, obs keys);
`check_r1_declared(model, st)` runs at every compiled `train()` and raises a typed
`CompileSentinelError` (FATAL_CONFIG via `compile_control.attach`) naming the field or key that
moved — the dynamo sentinel's guard dump stays the backstop. The gate, the prewarm and the canary
build R1's arguments through the same `_r1_static`. `r1_declared_levers_test` (routine, CPU, dynamo
`eager`): per inventory row, the learner built from that config, the real startup lifecycle, two real
updates — 0 compiles after the lock; a second test holds that an undeclared lever or key after the
lock is the sentinel's UNDECLARED SIGNATURE FATAL. (A row-weight lever whose weights were None on a
one-class rollout once flipped the signature on DATA; that lever, the strata weight, was deleted in
the P11c pass, and with it the per-update `var` dict.)

### Every non-training model can use it

`maybe_compile_extractor` is safe to apply to ANY frozen model, because the wrapper routes
**grad-enabled calls to eager**. That matters: the compiled artifact is inference-only (under
`requires_grad` dynamo hands the graph to AOTAutograd, whose CPU backward codegen fails on this
model's scatter/`index_add` — the documented reason the June `--compile-damage-op` integration was
inference-only), and the prober backprops through this same extractor for gradient saliency.

| consumer | what is compiled | gate |
|---|---|---|
| training env workers | — (none since U3: policy opponents forward through the T2 inference service) | — |
| `eval_worker` — **DELETED** (poke-env retirement P6 slice 6c; every eval cycle plays on the Rust eval core, whose policies T2 serves) | — | — |
| `snapshot_ladder` | both frozen ladder players | **default ON** — offline tool, nothing races it |
| prober (`session._load`) | the no-grad replay / rollout models | `--compile` (off by default) |
| `play.py` | nothing — the websocket/LADDER client loads a plain `MaskablePPO` and stays eager: one process, one battle at a time, and eager already measures **18 ms/decision** against a 150 s ladder timer, so a compile would buy latency nobody is waiting on | n/a |

**HISTORY — the Python eval worker (deleted in P6 slice 6c), kept for the record of how its compile was verified.**
Eval workers are fresh `Popen` processes, so they hit the shared on-disk Inductor cache the trainer
already warmed rather than inheriting anything; one worker plays hundreds of games, so the compile
repays many times over.

**Verified end to end**, not just wired: `python src/agents/training/eval_sharding_fuzz_test.py 4 2
--compile --neural-opponent` drives the REAL `eval_worker._run` over the bridge and logs
`eval-trainee: ON — 3.33 -> 0.67 ms (5.0x)` and `eval-opp:final_model.zip: ON`, with every exactness
assertion unchanged (units played + pooled exactly, full coverage, claim-exactly-once across two
workers) — the compile is value-preserving, which is why running the same fuzz both ways is the test.

⚠️ **`--debug-eval` does NOT exercise this.** Its final win-rate eval runs IN-PROCESS; it never
spawns an `eval_worker`, so it shows zero compile lines and proves nothing about this path.
⚠️ **A bots-only plan does not exercise the OPPONENT half.** Scripted bots have no extractor, so
`_get_opponent_model` never runs — `--neural-opponent` adds a FIXED (frozen neural) opponent, which
is the only kind that reaches it. That gap is why the opponent path went unverified at first, and
`src/main/eval_worker_compile_test.py` pinned the wiring in the fast unit suite (deleted with the worker).

**Validation is paid once per process** (`_COMPILE_VALIDATED`), and the reuse path STILL LOGS
(`ON (reused this process's validated compile)`) — it used to return silently, which made the eval
opponent's compile look like it had never run and cost a round of doubt. A success you cannot see in
the log is a success you will not trust. The eager-vs-compiled timing answers
"does this extractor's code object compile to something faster?", and `torch.compile` keys on exactly
that code object — so a second model in the same process cannot get a different answer. Consumers that
load models in a LOOP (an opponent rebuilt every iteration) would
otherwise re-pay ~15 eager forwards each time. Deliberately process-local: a fresh process
re-validates, since that is where a cold cache or a failing backend would actually show up.

The prober flag is OFF by default because a one-off `summary`/`list` never amortizes a ~10-20 s
compile; turn it on for the search-shaped commands (`better-line`, `falsify`, `falsify-scan`,
`replay-counterfactual`, `lookahead`), which do thousands of no-grad rollout forwards.

**BLAS thread pinning (not optional).** Each worker runs a full CPU opponent forward; at the library
default of one thread per core, N workers spawn N×cores competing threads. Measured on a 16-core box
with 8 neural-opponent envs: **6 fps at load average 110, vs 231 fps pinned** — a ~38× cliff.
`launcher/child.py` has always exported `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`, so production under
the launcher was never affected, but `python src/main/train_rl_agent.py …` (a documented entry point)
had no protection. `train_rl_agent` now sets them at import (`setdefault`, before torch is imported —
BLAS reads them at init), and each env worker additionally pins `torch.set_num_threads(1)` so an
explicit learner-side override can't silently un-pin the workers. Pinned by
`src/main/thread_pinning_test.py`; the compile guards by `src/agents/model/compile_extractor_test.py`
(incl. a regression that the global `suppress_errors` never comes back) and the uncompilable-op
regression by `src/agents/model/extractor_compiles_test.py`.

`tmp/production_cmd.py` reconstructs a runnable command from any run's `metadata.json` `cli_args`,
diffing against the live parser's defaults and REPORTING (never silently dropping) flags the tree no
longer has — that is how the production shape above was recovered.

## T25 perf levers on R1 (2026-10-10)

The perf phase (`designs/research_state/measurements/perf_phase_2026-10-10/PLAN.md`) adds the switches below and lever C's flag
(after them). Each is OFF by default. Off, the production and end-state compiles are byte-for-byte unchanged.

- **R1's Inductor option presets** (`compile_regions.R1_INDUCTOR_PRESETS`, `gen3_r1_inductor_presets_v1`):
  - the presets: `default` (no options), `coordesc` (`coordinate_descent_tuning`), `combo` (`combo_kernels`) and
    `cudagraphs` (`triton.cudagraphs`, i.e. cudagraph trees over R1's forward + backward). Presets join with `+`.
  - set `model._r1_inductor_preset` BEFORE `compile_regions.install`, which passes them as
    `torch.compile(..., options=)`: R1 only, never T2's graphs, never a global config.
  - `compile_regions_presets_test.py` pins this: the default passes NO `options`, every key is a real Inductor
    config of this torch, and an unknown name refuses.
  - They are MEASUREMENT switches with no CLI flag.
- **The trunk's compute precision: `--trunk-precision {fp32,bf16}`** (`gen3_trunk_precision_flag_v1`, config v155;
  `agents/model/trunk_precision.py`; **ADOPTED 2026-10-14**: the bf16 strength A/B PASSED, so `bf16` is
  `recipe.fresh.trunk_precision`, the production recipe, and `fp32` is the explicit opt-out).
  - **Adoption (2026-10-14).** The bf16 strength A/B (`designs/endstate/design_bf16_strength_ab.md`; 8 seeds × 15M per arm,
    the 8 × 8 mirrored cross, the closing test's coded rule): Δ̂ +1.06 pp, 90 % interval [−1.19, +3.31], PASS; `train_ms`
    54.3 s → 42.1 s (−22.5 %). The flag's default is still `fp32` — an UNSET `--trunk-precision` resolves to `fp32`
    — but `--arch production` writes `bf16` (the recipe applier) and the recipe surface REFUSES a fresh argv that
    leaves it untyped. A flagless fork or restart INHERITS its parent's recorded value (an fp32 parent's fork stays fp32
    unless it types `--trunk-precision bf16`; the A/B did not measure a mid-lineage precision switch). The K9 learner
    golden is the production (bf16) update and pins the fp32 opt-out beside it.
  - **The switch.** `bf16` class-swaps the LEARNER's live trunk rounds (`BiasedEncoderLayer`, `IdentityInitRound`)
    to subclasses whose forward runs under bf16 autocast and returns fp32. `apply_run_precision`, from
    `apply_training_hparams`, does it on both build paths, before grad checkpointing and the R1 install.
  - **Decisions stay fp32.** The heads, the pointer head, the masked log-softmax, the value, the losses and the
    AdamW state all run in fp32.
  - **Recorded.** It is a `ModelVersion` field, `_resolve`-inherited by a flagless resume. A pre-v155 config
    migrates to fp32. `check_compatible` never compares it. It is in the shared compile cache's
    `COMPILE_ARG_KEYS`.
  - **Design (b): bf16 lives in R1 only.** T2, eval and every snapshot build from a state dict, so they get base
    classes. The K9(b) probe's forward and its violation scan run under `trunk_precision.fp32_scope`, so K9(b)
    compares fp32 with fp32 at its unchanged bars. `trunk_precision_flag_test` pins it: bit-identical probe
    log-probs on a bf16 learner, revert-checked.
  - **THE bf16 PARITY RULE** (`gen3_r1_bf16_parity_v1`, `compile_regions._r1_verdict_bf16`). It applies when the
    policy's rounds are bf16, at the startup gate and at every canary, through the same rungs as fp32. Three arms
    run on the same weights and rows: compiled bf16, eager bf16, and the fp32 EAGER REFERENCE (`_r1_reference`:
    R1's eager body under `fp32_scope`, proved eager by the region counters). Two conditions are checked:
    1. **DRIFT:** eager bf16 vs the reference, loss rel and global gradient rel-L2, each ≤ `BF16_DRIFT_BARS`.
    2. **ENVELOPE:** compiled bf16 vs the reference ≤ K × max(eager bf16 vs the reference, a floor), for the loss,
       the global gradient and EVERY judged parameter (`BF16_ENVELOPE_K`). The floor is the fp32 bar of that
       quantity, except the scalar loss's, which is bf16's unit roundoff 2^-9 (`BF16_LOSS_FLOOR`).
    - With `--learner-cudagraphs on`, the graphed route's accumulation rung is judged by the same rule, its fp32
      reference arm accumulated eager under `fp32_scope`.
  - **The bars** are R1_BAR_K × the healthy GPU maxima of window D's 20 reads, derived by
    `scripts/derive_bf16_bars.py` into `results/bf16_parity_bars.json`. `compile_regions_bf16_test` re-derives them
    and fails on a differing constant.
    - DRIFT: loss 0.014, gradient rel-L2 0.68.
    - ENVELOPE K: loss 5.4, gradient 6.4, per parameter 12.
    - A `None` table refuses a bf16 R1 ("no measured bars").
  - ⚠️ **The bf16 gate's resolution — a KNOWN LIMIT, measured** (`gen3_r1_bf16_resolution_v1`, 2026-10-15). A
    parameter's backward miscompile is refused only above K = 12 × max(that parameter's own bf16 error, the fp32
    bar); bf16 moves the micro-batch gradient 1–18 % rel-L2 from fp32's, so the refusal threshold is coarse:
    - window D's 20 GPU reads (B = 2,048; `results/bf16_parity_bars.json` "teeth"): on perturbed / bf16-trained
      weights (18 reads) the MEDIAN parameter is refused above 0.22–1.29 relative error, the p90 above 0.53–3.07, the
      worst above 1.27–10.2; on FRESH weights the live rung's floor makes it 1.20 for every parameter, so on a fresh
      launch the seeded-perturbation rung is the one that judges.
    - So a 10 % backward error is NOT refused (the 2026-10-15 CUDA run: a planted 10 % on
      `pokemon_encoder.move_network.0.weight` passed). A DROPPED (error 1.0) or sign-flipped (2.0) gradient is
      refused on MOST parameters, not all: one passes on a parameter whose own bf16 error is above 1/12 ≈ 8.3 %
      (resp. 1/6). A NaN is always named (`NonFiniteGateArmError`, unchanged).
    - The resolution is COMPUTED exactly from the gate's arms (the larger root of |C_p − R_p + m|C_p| ŝ| = T per
      check, the minimum over the checks and the rungs; a one-off CPU read split pass / refuse at 0.97 / 1.03 m* on
      all four representative parameters) and CONFIRMED by plants at 0.5 m* (passes) and 2 m* (refused):
      `compile_regions_resolution_test.py` (CPU mechanics; `test_a_10pc_…` pins the limit, so a sharper gate fails
      it and updates this bullet). The CUDA numbers at HEAD — the K9 golden's fresh weights and the `production`
      baseline's trained weights, four representative parameters — come from
      `scripts/ops/verify_compile_gate_cuda.sh` (`summary.json`, "resolution"). **First GPU run 2026-10-15 at `87e493c3`: PASS 4/4**; its
      `summary.json` is committed at `designs/research_state/measurements/x26_exit_check_2026-10-15/gpu_t36_87e493c3/`.
    - The fp32 gate's teeth are unchanged and still pinned: a 10 % plant is FATAL on an fp32 learner
      (`compile_regions_independence_test.py` on CPU, `compile_regions_fixed_mass_cuda_test.py` on CUDA — both
      build `trunk_precision="fp32"` explicitly since the bf16 adoption).
  - 🚨 **A FALSE-FATAL mode, found 2026-10-15** (`rb_toy_ent_s4001`): the rule's per-parameter statistic crosses the
    PPO CLIP BOUNDARY of any fixture row whose ratio sits within rounding of it, so two healthy arms can read 30× apart
    (CPU: 18 % of healthy pairs above K at that FATAL's weights). Diagnosis, the GPU read and the fixture fix
    (T36, BUILT 2026-10-15; CONFIRMED on CUDA at `87e493c3`): "The bf16 canary FATAL on trained weights" below.
  - **Decision record — the bf16 gate's reference.**

    | Date | Decision | Why | Evidence |
    |---|---|---|---|
    | 2026-10-15 | **Keep the fp32 eager reference; do NOT add a same-precision (compiled bf16 vs EAGER bf16) miscompile rule.** The resolution above stands as a documented limit. | A miscompile is a compiler defect, so judging compiled bf16 against eager bf16 looked like it would leave only fusion / reassociation rounding and tighten the teeth. It does not: under autocast Inductor's fused kernels keep fp32 intermediates where eager rounds every op's output to bf16, so the two bf16 arms ALSO differ by bf16 rounding. Derived exactly as the envelope is (R1_BAR_K × the healthy max of d(C, E_bf16)_p / max(d(E_bf16, ref)_p, floor)), its K is 4 × 2.96 = 11.8 — the envelope's 12, the same teeth; as one absolute bar per regime (the fp32 rule's form) it is 4 × max d(C, E_bf16)_p = 0.29 fresh / 1.6 trained, coarser than the envelope's median teeth. "Clearly better" was taken as ≥ 2× sharper (K ≤ 6). The check would cost nothing at launch (both bf16 arms are already computed), but a rule with the same teeth is a second bar to maintain for no gain. | Window D, `results/window_D/E_bf16_envelope.json` (`Cbf_vs_Ebf`, `Ebf_vs_E32`): per-parameter d(C, E_bf16) median 0.9–6.2 % vs d(E_bf16, ref) 1.2–10.7 % by read, ratio median 0.55, max 2.96. Pinned by `compile_regions_resolution_test::test_a_SAME_PRECISION_rule_would_not_be_sharper_on_the_committed_GPU_reads`; re-read at HEAD by the verification script's `same_precision` block per rung. |

    Levers NOT taken here, each a separate GPU-measured decision: (a) Inductor's `emulate_precision_casts` (round
    intermediates where eager does) would make compiled-vs-eager bf16 a reassociation-only difference — a
    same-precision rule could then be tight, but it changes the production graph and its speed; (b) a SHADOW fp32
    compile of R1 judged by the fp32 rule at launch would catch the precision-independent miscompile classes (a wrong
    index, a dropped path, a NaN backward) at fp32's teeth, for a second compile per launch and restart — but it
    judges a graph that does not train.
  - **Pinned by** `trunk_precision_test.py` (the region itself), `trunk_precision_flag_test.py` (the flag surface,
    `fp32_scope`, the K9(b) probe), `compile_regions_bf16_test.py` (the rule: refused without bars, a healthy
    pass, a planted compiled error, a planted drift, a mis-wired reference, an fp32 learner untouched),
    `compile_regions_resolution_test.py` (the resolution, above) and `compile_regions_clip_margin_test.py` +
    `compile_regions_clip_margin_fatal_test.py` (the fixture off the PPO clip boundary, T36, below).

A preset gets its CLI flag, flag-registry row and recipe surface in its ADOPTION commit, after its paired GPU benchmark
and the R1 gate and canary are green. `--trunk-precision` had its flag already, because its parity rule needs a run
path to be gated on; it entered the recipe surface on the PASS of PLAN.md §2d (2026-10-14, above). The benchmark that sets the switches
is the measurement's `scripts/update_bench.py`. ⚠️ The `cudagraphs` PRESET alone graphs R1's ONE compile and dies on
the first (diagnostic) update (F-PB-2); lever C's route is `--learner-cudagraphs` (below), and `install` refuses the
preset combined with it.

### 🚨 The bf16 canary FATAL on trained weights (`rb_toy_ent_s4001`, 2026-10-15) — diagnosed on CPU; the FIXTURE FIXED (T36), CONFIRMED on CUDA at `87e493c3`

**The event.** The toy control `rb_toy_ent_s4001` (`--arch production` + 44 ablations: entity tokens and attention, no
damage op, no move resolution; bf16 trunk; `--compile-trainer`; pinned `3899277a`) ended FATAL at 72,658,984 steps
(update 700 of its second process). Verbatim: the compiled TRAIN graph's gradient is farther from the fp32 reference
than 12 × eager bf16's own error on 3 parameters — `pointer_head.move_proj.weight` compiled 4.03e-01 vs eager 2.82e-02,
`move_proj.bias` 3.26e-01 vs 2.59e-02, `move_score.weight` 3.03e-01 vs 2.41e-02 — CONFIRMED on the same rows and on
slice 1 in the same update. The six canaries before it passed (worst ratio 1.10 / 1.11 / 2.88 / 1.90 / 1.70 / 1.27 /
1.92); training looked healthy (bots win rate 0.87–0.90 flat; K9(b) excluded share 0, max |Δ log π| 2.2e-5).

**THE FINDING (CPU, the run's own checkpoints loaded STRICT at this checkout; `python -m main.ops.diag_bf16_canary cpu`):
the canary's per-parameter statistic contains a DECISION BOUNDARY — the PPO clip.** The canary judges R1 on the K9
golden's 64 committed rows (tiled to B = 2,048), whose old log-probs come from a near-uniform policy (−2.47 … 0). On
trained weights those rows' PPO ratio ρ = exp(log π − old) spreads over 0.002 … 6.9. A row whose ρ sits within
rounding of the clip boundary that SWITCHES ITS gradient (A > 0 at 1 + 0.15, A < 0 at 1 − 0.15; the other boundary
never switches it) carries its whole policy gradient in one arm and none in the other.

- **At the canary's own weights** (`final_model_canary_fatal.zip`): rows 4 (A < 0, ρ_fp32 0.8462 against 0.85: a
  margin of 0.0045 in log-prob) and 35 (A > 0, ρ 1.1581 against 1.15: 0.0070) sit on their boundaries; eager bf16
  moves a legal log-prob by up to 0.082 on these rows.
- **The null** (32 seeded multiplicative weight dithers, σ = 2^-9 relative, each read against its OWN fp32
  reference): eager bf16's error on `move_proj.weight` is BIMODAL. On the 17 seeds where no row's clip branch differs
  from fp32's: median 0.020, max 0.030. On the 15 where one does: median 0.294 (0.057–0.42). With the branch HELD to
  fp32's decisions (`implied_dz(..., active=)`), the error never exceeds 0.020 on any seed. Two HEALTHY eager arms
  read a canary ratio above 12 in **18 %** of 992 seed pairs (max 31×).
- **The run's CUDA numbers sit exactly on the two modes**: eager 2.82e-2 (the no-flip mode; CPU eager reads 2.83e-2
  on the same weights), compiled 4.03e-1 (the flip mode; CPU dithers in it read 0.39–0.42).
- **Eager bf16 flips on its own, undithered**: at 66M it reads 0.29 / 0.34 (slices 0 / 1) from fp32 on
  `move_proj.weight`, and ONE row (43: A > 0, ρ 1.1557 against 1.15) carries 0.288 of it.
- **Why the pointer head**: only the policy and entropy terms reach it, and its gradient is carried by a few rows
  (κ = Σ_rows |g_row| / |Σ_rows g_row| 1.8–3.2 at every checkpoint read; at the FATAL's weights the top row holds 30 % of the absolute mass), so
  one row switching moves it by tens of percent. The trunk's gradients also carry the value / win-prob losses, so the
  same row is a small share there, and the global gradient check passed.
- **Why the confirmation could not catch it**: slice 1 (rows 32–63) is a SUBSET of slice 0 (all 64), and the
  same-rows re-run is the same deterministic arithmetic; row 35 is in both. (Since T36 the confirmation is the two
  DISJOINT halves, below.)
- **Not "trained weights" as such — a per-STATE lottery**: at 15M a row sat 0.0014 from its boundary and 23 % of
  healthy pairs read above 12. The CENSUS of every 1M checkpoint 1M–62M (all before the last passing canary, 62.83M;
  B = 64 untiled, so the clip decisions are exact): a row sits within 0.01 of its switching boundary at 14 of 62, and
  eager bf16 ITSELF flips one against fp32 at 4 (8M 0.36, 13M 0.19, 52M 0.44, 61M 0.13; three also on slice 1).
  eager bf16's largest |Δ log π| per checkpoint has a median of 0.063 and a max of 0.161. A false FATAL needs a boundary
  row, the COMPILED arm across it while eager is not, and the row in rows 32–63 (or two consecutive disagreements);
  s4001 drew it at its eighth scheduled canary. A 15M run (the bf16 A/B's length) meets about two.
- **The `production` baseline's CUDA PASS is consistent with it** (`verify_compile_gate_cuda.sh` @`f0cf6da5`:
  `rb_es_end2_s2005` final, 15M, fp32-trained, read under the bf16 rule; same-precision ratio 1.55): on CPU that state's
  nearest row is 0.074 (log-prob) from its switching boundary against eager bf16's largest |Δ log π| of 0.055, so no
  dither flips a branch (0 of 8), eager's error is unimodal (0.0077–0.011) and no healthy pair reads above K (max
  1.1). It is one draw on a state WITHOUT a boundary row, not evidence that production states lack them: the
  production rows' ratios are as off-policy as the toy's (ρ 0.002–5.2), and the same clip sits in the same loss.
  **The PRODUCTION-surface census** (the bf16 A/B's eight end-state seeds `rb_bf_eb_s2001`–`s2008` at 15M, and s2001 at 5M / 10M; CPU, B = 64 untiled): a row sits within 0.01 of its switching boundary at 2 of 10 states (s2001 5M 0.0061, s2005 15M 0.0095) and within 0.02 at 3. Eager bf16 flips one at 1 (s2005 15M). That row's |A| is 0.011, so the head moved < 1 %: a flip's SIZE scales with the row's |A|·ρ, and s4001's rows had |A| 0.6–0.85. So X26 (the production surface, thousands of updates, a startup gate at every restart) CAN reach the same event — it is a per-state lottery on any surface whose fixture rows are off-policy, not a regime of the toy's pointer head.
- **The startup gate is SINGLE-SHOT**: `gate_regions` judges the same rows at every launch and every 6-hourly restart
  with no confirmation step, and its refusal is FATAL_CONFIG. A boundary row at a restart ends the run there.
- **The KEY TENSION, resolved**: Inductor's fp32 intermediates keep compiled bf16's log-probs as close to fp32's as
  eager's in the CONTINUOUS part (the frozen-branch error); a 0/1 switch needs only one arm's rounding to fall across a
  margin of 0.005.
- **The growth curve** (eager bf16 vs fp32, `move_proj.weight`, slice 0, CPU): 5M 0.010 · 15M 0.012 · 30M 0.016 ·
  45M 0.020 · 60M 0.019 · 62–63M 0.037–0.041 · **66M 0.29 (a flip)** · 67–72M 0.021–0.031 · FATAL 0.028. The
  move_proj pre-tanh magnitude grows (p99 3.9 → 11; |x| > 3 on 5 % → 57 % of units) and the continuous error with it
  (~3×), nowhere near the 12× bar on its own.

**Hypotheses, ranked, each with the read that kills it** (the GPU script reads every one; `VERDICT.md` is the form):

| rank | hypothesis | CPU evidence | killed / confirmed by (GPU, at the FATAL's weights) |
|---|---|---|---|
| 1 | **(iv) the FIXTURE**: the PPO clip on 64 off-policy rows is a decision boundary in the statistic (standing rule 8) | SUPPORTED: the bimodal null, the flip attribution, eager's own flip at 66M, the two boundary rows at the FATAL | KILLED if Cbf flips NO row's branch against E32 (`implied.Cbf.branch_flips`) or its FROZEN-branch error stays far above Ebf's (`frozen_branch_vs_ref`); CONFIRMED if the RELOCATED fixture (`relocated`) passes and its null reads P(ratio > K) ≈ 0 |
| 2 | **(iii) the yardstick is unusually small** | NOT SUPPORTED in its stated form: 2.8 % is eager's typical no-flip value (dither median 2.0 %, max 3.0 %); it is one draw from a BIMODAL distribution — (iv)'s mechanism, not a calibration defect | the same reads; `null.errors.Ebf` on CUDA |
| 3 | **(ii) bf16 numerics in the compiled graph** (a fused reduction or softmax in bf16, overflow) | UNLIKELY by construction: the pointer head, the masked log-softmax and the losses run OUTSIDE the bf16 region (design (b)); compiled and eager bf16 differ only in WHERE the trunk rounds | Cbf's frozen-branch error and `implied.Cbf.self` (its backward against its own forward) vs Ebf's; `Cbf_emul` (rounds where eager does), `Cbf_aot` (no Inductor codegen) and `Cbf_pheager` vs Ebf |
| 4 | **(i) a real Inductor miscompile** with trained weights | the least likely: a miscompile is deterministic, would read in the no-flip mode too, and the six earlier canaries read 1.1–2.9 | `fp32_rule` (C32 vs E32 at the fp32 bar: `over_bar` empty kills it); `Cbf_pheager` |

**THE GPU DIAGNOSTIC** — `scripts/ops/diag_bf16_canary.sh` (+ `src/main/ops/diag_bf16_canary.py`), run by the
lease holder: `GEN3AI_GPU_LEASE_TOKEN=<token> scripts/ops/diag_bf16_canary.sh rb_toy_ent_s4001` (`--plan` prints the
command). One process under `gpu_lock.sh` + `mem_cap.sh` (16 GB), a 20-minute wall, a 14-minute budget for STARTING
checkpoints (estimate: four Inductor compiles ~6 min, then ~1 min per checkpoint plus ~1–2 min per 32-seed null;
UNVERIFIED until the first window). Order: `fatal, 72M, 70M, 66M, 60M, 45M, 30M, 15M` (the null at 32 seeds on the
first two, 8 after; the checkpoints past 62.83M trained after the last passing canary, so they are read as SUSPECT —
the s4001 census below is all before it). Arms: E32, Ebf, Cbf (the production `compile_regions.install`), C32 (a
second learner at fp32), Cbf_emul (`emulate_precision_casts`), Cbf_aot (`aot_eager`), Cbf_pheager (the pointer head
eager behind a graph break); a LOCALISATION arm (C32, Cbf_emul, Cbf_aot, Cbf_pheager) that cannot compile or run is
recorded (`failed_arms`) and dropped, never the window's end.
Per checkpoint and slice: every pairwise per-parameter distance over the pointer head, the canary's statistic and
verdict replayed through `_r1_verdict_bf16` itself, the canary's own `_check` (the reproduction, verbatim), the fp32
rule on C32, the forward agreement, the implied-gradient read (self / vs reference / frozen branch / branch flips),
the per-row decomposition (κ, the rows carrying each arm's error), the switching margins, activations and weight
norms, the RELOCATED fixture's reads, and the dithered null with its flip attribution. Output
`~/.cache/gen3ai/diag_bf16_canary/<sha>/`: `gpu_<label>.json` as each finishes, `VERDICT.md`, `diag.log`. The CPU
reads that underlie this section: `python -m main.ops.diag_bf16_canary cpu --ckpts … --out D` (~10 s per checkpoint,
~2–4 min per 32-seed null on three cores).

**Decision options for X26** (cost → verdict):

| option | what it costs | verdict |
|---|---|---|
| fp32 for the pointer head only | — | **MOOT**: the head already runs fp32 (design (b)); the flipping row's log-prob moves because of the bf16 TRUNK upstream |
| Inductor `emulate_precision_casts` | a slower graph, its own benchmark | **does not fix it**: eager bf16 itself flips (66M) |
| recalibrate the bar on trained states | — | **NO**: (iii) is not shown; a bimodal statistic has no bar with teeth (healthy flips read up to 31×) |
| fp32 trunk for X26 | `train_ms` +29 % (42.1 → 54.3 s); leaves the adopted, A/B-passed recipe | a working STOPGAP: fp32 arms differ by ~1e-6 in log-prob, so flips are ~10^4× rarer — but the same latent boundary sits in the fp32 rule |
| an eager learner | the whole compiled speedup (not measured for this surface; UNVERIFIED) | NO: it removes the canary's subject, not the defect |
| **fix the canary's FIXTURE (standing rule 8)** | ~0.5–1 agent-day + this GPU window to validate + a `verify_compile_gate_cuda.sh` re-run | **BUILT (T36, below)**; CONFIRMED on CUDA at `87e493c3` |

**THE FIX (T36, `gen3_r1_clip_margin_v1`, BUILT 2026-10-15; CONFIRMED on CUDA at `87e493c3`, see "The CUDA confirmation" below).** Standing rule 8 —
*exclude inputs within a rounding error of a decision boundary rather than tolerating them by chance* — applied to the
gate's own fixture. Three parts, in `compile_regions` / `compile_canary`, for the STARTUP GATE and the CANARY alike
(they share `r1_rungs` → `_r1_judge`), under the bf16 rule and the fp32 rule alike (the fp32 rule has the same
boundary at ~10⁴ smaller odds: its arms differ by ~1e-6 in log-prob):

1. **Relocate the boundary rows, on every rung's own weights** (`relocate_clip_boundary` → `clip_relocation`). Before
   the compiled / eager / reference arms run, one fp32 eager forward of R1 (`fp32_scope`; a no-op on an fp32 trunk)
   reads every row's log-prob and normalised advantage; a row whose reference log-ratio is within `R1_CLIP_MARGIN` of
   (or AT) the bound that switches ITS gradient (A > 0: log 1.15; A < 0: log 0.85; A = 0 rows carry no policy
   gradient and are left alone) gets its OLD log-prob moved so the reference sits 2 × the margin inside the ACTIVE
   side. **Moved, not excluded**: the batch keeps its declared shape (no new signature) and every row; excluding a
   row would change the batch's advantage normalisation and drop its whole policy gradient from the check, while a
   moved row's gradient is judged IN FULL by every arm; a row far from its bound keeps the golden's label.
   Deterministic: a function of the reference's own forward on the same rows. The threshold itself is no verdict
   boundary: a row just inside it is moved to 2M, one just outside stays at > M, and either way an arm shift ≤ M
   switches nothing. Every rung relocates on ITS weights — the live rung, the fresh launch's seeded perturbation,
   the unmoved parameters' perturbation, the graphed route's accumulation rung (both micro-batches) and the
   judged-set rule's golden-alone reference.
2. **The margin's precondition, CHECKED** (`_clip_margin_rule`, on every rung): every judged arm's largest |Δ log π|
   from the reference is ≤ the margin, and — its consequence, checked rather than assumed — no arm switches any row's
   clip branch. A COMPILED arm that breaks it is refused BY NAME as a forward this far from the reference (≥ 4× the
   healthy maximum), never read through a gradient ratio; a YARDSTICK arm (eager, the fp32 reference) that breaks it is
   a recorded event and the rung re-runs at M′ ("The yardstick is not the compiled arm", below). Every R1 rule line prints `clip margin: <n>/<B> boundary rows
   relocated (M …; nearest unmoved …); max |dlogp| compiled …, eager …, fp32 reference … <= M, 0 clip-branch
   switches`, and every canary record carries it (`watch.*.clip_margin`).
   **The margin**: `R1_CLIP_MARGIN = R1_BAR_K × R1_LOGPROB_SHIFT_MAX = 4 × 0.18 = 0.72` (log-prob). 0.18 is the largest
   |Δ log π| of a judged arm measured so far, rounded UP to 2 significant digits as the bf16 bars are: eager bf16 vs
   eager fp32, the s4001 census max 0.161 (62 checkpoints), the production census max 0.055 (10 states), the
   32-seed dithered null at the FATAL's weights 0.173. **PROVISIONAL — CPU and eager only**: the COMPILED bf16 arm's
   shift on CUDA is the GPU window's `forward_vs_E32.Cbf.log_prob_max_abs` (and every gate line); if it exceeds 0.18,
   `R1_LOGPROB_SHIFT_MAX` rises with it (on a LIVE run, whose pin cannot move, `--canary-shift-max`: below).
3. **The canary's confirmation on two DISJOINT halves** (`compile_trainer.FIXTURE_HALVES`: slice 2 = rows 0–31, slice
   1 = rows 32–63). The halves share no row, so no single row can confirm a disagreement: independence by
   construction. The scheduled check (and the startup gate) keep ALL 64 rows — a half can leave a parameter below
   the per-parameter floor (the moves head on slice 1, `compile_regions_unmoved_init_test`) — and a miscompile only
   one half judges is FATAL at the next scheduled canary (two consecutive disagreements). The old "same rows again"
   re-run is gone: it re-ran the same deterministic arithmetic.

**The evidence (CPU, the FATAL's weights loaded strict at this checkout, B = 64 untiled, on the rows the FATAL was
judged on — the K9 buffer at the run's pin, committed as `src/agents/model/testdata/k9_golden_buffer_rb_toy_ent_s4001_pin.npz`,
because legacy deletion U3 rebuilt the committed golden since; `compile_regions_clip_margin_fatal_test`).** 32 seeded dithers, each against its own fp32 reference, two healthy
eager bf16 arms per ordered pair: on the RAW rows 212 of 992 pairs read the canary's ratio above K = 12 on a
pointer-head parameter (21 %, max 41.5×; 15 of 32 seeds switch a row's clip branch); on the RELOCATED rows **0 of 992**
(max 3.65× — the order of window D's healthy max 2.94), every dither passing the margin rule (largest eager shift
0.173 against M = 0.72). 21–22 of the 64 rows move, the FATAL's two boundary rows (4 and 35) among them. A planted
10 % backward error on `pointer_head.move_proj.weight` is still refused at the fp32 rule on these weights' relocated
rows (and on the K9 learner's, `compile_regions_clip_margin_test`); the bf16 resolution test re-pins nothing (it now
relocates per rung as `_r1_judge` does, so its resolution is the gate's). The diagnostic's CPU re-read at this
checkout (B = 2,048, production's relocation): at the FATAL's weights 22 of 64 rows move, eager switches none, the
32-dither null on every judged head parameter 0 over K (max 3.67×; raw up to 20 %); at 66M eager's own flip (0.29 /
0.34) reads 0.033 relocated and the null 14 % → 0; at 15M 21 % → 0. Eager's largest |Δ log π| there 0.082 / 0.107 /
0.043. ⚠️ The diagnostic's null had read the never-judged `struggle_score.*` (~1e-9 of the largest gradient) as a
"ratio"; it now reads only the parameters each seed's reference judges, as the canary does.

**What the GPU window must show** (the training session's window, at this commit, under the lease):
`scripts/ops/verify_compile_gate_cuda.sh` — all four CUDA tests PASS on the relocated rows (the fp32 gate's 10 %
plant FATAL, the bf16 rule's pass / NaN / resolution confirmed at 0.5× / 2×), `summary.json` verdict PASS; and
`scripts/ops/diag_bf16_canary.sh rb_toy_ent_s4001` (its `--buffer auto` reads the FATAL's own rows, the committed
pinned copy, since U3 rebuilt the golden) — at the FATAL's weights `canary_replay` (the canary's own `_check`,
now relocating) PASSES; `relocated.branch_flips.Cbf` = 0 and the relocated null reads P(ratio > K) = 0 for the Ebf
pairs AND `Cbf_vs_Ebf`; `forward_vs_E32.Cbf.log_prob_max_abs` ≤ `R1_LOGPROB_SHIFT_MAX` (0.18) at every checkpoint
read; `fp32_rule.over_bar` empty. If Cbf's shift exceeds 0.18, raise the constant (the margin is 4× it) and re-run; if
Cbf is off the reference with NO branch flip, or C32 is off at the fp32 bar, the fixture was not the whole story —
X26 takes the fp32 trunk and the miscompile is hunted with the localisation arms. **Order: T36 (built) → the GPU
window → X26 on the production bf16 compiled learner.**

**The CUDA confirmation (2026-10-15, `87e493c3`, the Training session's window).**
- `verify_compile_gate_cuda.sh`: PASS 4/4.
- `diag_bf16_canary.sh rb_toy_ent_s4001` (8 checkpoints, 15M to the FATAL) met every criterion of the paragraph above:
  - the canary's replay passes at the FATAL's weights (relocated ratio 0.655);
  - `relocated.branch_flips` is 0 for every arm at every checkpoint;
  - the relocated null reads P(ratio > K) = 0 for the Ebf pairs and for Cbf;
  - `fp32_rule.over_bar` is empty, and C32 sits within 4e-6 of E32;
  - Cbf's `forward_vs_E32.Cbf.log_prob_max_abs` peaks at **0.110** (FATAL) ≤ 0.18, so `R1_LOGPROB_SHIFT_MAX` stays.
- The UNRELOCATED canary would have refused at 15M too (ratio 19.5, 32 flips): the same boundary class, at another
  checkpoint.
- **The shift GROWS with training:** taken action 0.050 (15M) → 0.107 (70M) → 0.110 (FATAL). Eager bf16 grows alike, so
  the growth is the weights', not the compiler's.
- X26 watches it in-run: `canary_verdicts.jsonl` `.watch.scheduled.clip_margin.max_abs_dlogp.compiled`, with bands and
  actions in `designs/endstate/design_x26_readiness.md` §4.2.
- The per-checkpoint table is `designs/research_state/measurements/x26_exit_check_2026-10-15/canary_shift_87e493c3.json`.
  The diagnostic's own `VERDICT.md` checklist was left unfilled; the criteria above were re-read from its `gpu_*.json`.

**The watch-list record** (`gen3_canary_watch_v1`, this change): every canary record now carries the pointer head's
per-check ratio, compiled distance and eager yardstick (`designs/training/learner_lifecycle.md`), so the next run's
growth curve is recoverable; s4001's six passing canaries logged only their worst parameter.

### The check-only margin override (`--canary-shift-max`, `gen3_canary_shift_max_v1`, config v158, T37, 2026-10-15)

**Why.** The clip margin of the startup gate and of the canary is `R1_CLIP_MARGIN` = `R1_BAR_K` 4 × `R1_LOGPROB_SHIFT_MAX`
0.18 = 0.72 log-prob (T36, above). The bf16 HEADROOM WATCH (`design_x26_readiness.md` §4.2) raises 0.18 when the compiled
bf16 arm's taken-action shift passes it. A constant in code reaches a run only through its code pin, and **the launcher
REFUSES to move a live run's pin at a restart** (`worktree.resolve_pin`; readiness F28), while `--sync-to-main` would walk
X26 onto all of HEAD. So the delivery is a flag a same-run RESUME may change.

**What it is.** `--canary-shift-max V` (argparse default None; `type` = `compile_regions.resolve_shift_max`):
- **Check-only.** It is the margin's INPUT and nothing else: `compile_regions.clip_margin_of(model)` = `R1_BAR_K` × the
  shift, read by `relocate_clip_boundary` (where the gate moves its fixture rows) and `_clip_margin_rule` (where it refuses
  an arm). No loss, no optimizer, no seed, no bar of the parity rule (loss 5.4 / gradient 6.4 / per-parameter 12) reads it.
  **Proven**: `main/train/canary_shift_max_test.py::test_one_update_is_BIT_IDENTICAL_with_the_flag_at_two_values` runs
  the K9 golden's one update with the flag at None, 0.72 and 25.0 and compares every parameter's bytes, the per-group
  hashes and every pinned loss scalar; `…::test_the_fingerprint_has_teeth_…` shows the same harness moves when a real
  hyperparameter does.
- **Raise-only.** A value below the code's `R1_LOGPROB_SHIFT_MAX` (0.18), or a non-number / NaN / ±inf, is REFUSED, typed
  (`CanaryShiftMaxError`; an argparse error at parse, a `parser.error` in `resolve_config` for an INHERITED value, and the
  gate itself refuses a model carrying one): 0.18 is the largest healthy shift measured, so a lower margin has no measured
  basis and would make the gate refuse arms the measurements call healthy.
- **A typed value needs the gate it moves.** `combination_checks.canary_shift_max_needs_the_compiled_cuda_learner` refuses
  a TYPED value on `--no-compile-trainer` / `--device cpu` / `--debug` (silently inert otherwise, like `--learner-cudagraphs`);
  an INHERITED value is carried, never refused (a CPU smoke of an X26 fork still launches).
- **Recorded and inherited like `--trunk-precision`** (config v158, `ModelVersion.canary_shift_max`, Optional[float], default
  None = the code's constant; a pre-v158 config migrates to None). `resolve_config`'s `_resolve` gives: a TYPED value wins and is what
  the next save records; an untyped launcher restart, flagless resume or FORK inherits the recorded one; a fresh run is None.
  The resolved value is also in `metadata.json:cli_args`. Never compared by `check_compatible` (no weight shape, no served
  forward reads it). The model attribute `_canary_shift_max` is set by `apply_training_hparams` on both build paths and is
  never pickled (`_excluded_save_params`).
- **Printed.** The launch prints `🐤 [CANARY SHIFT MAX] V …` when a value is set; every R1 rule line prints `clip margin: n/B
  boundary rows relocated (M 2.000 = 4 x shift_max 0.5 [--canary-shift-max]; nearest unmoved …)` (`[default]` when
  untouched); every canary record in `canary_verdicts.jsonl` (pass, unconfirmed, fatal) carries `canary_shift_max`
  {value, source, margin}, and `watch.*.clip_margin` carries `shift_max` / `shift_max_source` beside the margin. The
  margin rule's refusal names the effective value and says how to raise it.

**The headroom watch's RED action uses it** (`design_x26_readiness.md` §4.2): when the compiled bf16 shift passes 0.18 but
stays under the margin, **stop the run at its next planned restart and resume the SAME run** — `--model <its latest
checkpoint>` at the SAME pin (verified with `--dry-run`; never `--sync-to-main`) — with `--canary-shift-max <v>`, v the
largest `compiled` value read rounded UP to 2 significant digits with ≥ 1.25× headroom. The flag must exist in the run's
pinned tree: it does for any pin at or after the T37 commit (X26's P_X26 is after it), and a pinned parser that lacks it
refuses the argv (`pinned_argv.pinned_parser_check`: `✗ NOT IN PINNED TREE`). The raise survives the launcher's later 6-hourly
restarts through the recorded field. It does not replace the diag (`scripts/ops/diag_bf16_canary.sh <run>`), which says
whether the shift is shared by eager bf16 and C32 is still ≈ 0; it replaces the FORK that a code-only raise would have forced.

### The yardstick is not the compiled arm (`gen3_r1_yardstick_margin_v1`, 2026-10-19)

**The defect** (the X34 u1700 UNCONFIRMED; `../research_state/measurements/x34_canary_u1700_2026-10-19/README.md`).
`_clip_margin_rule` refused ANY arm past the margin, and the canary logged every refusal as "compiled DISAGREES with
eager" on its confirm / FATAL ladder. At X34 update 1700 (pin `ae2d0901`, bf16 trunk) the arm that broke M = 0.72 was
EAGER bf16 at 0.856, the canary's own yardstick; compiled read 0.52 with 0 clip switches, and the gradient rule's worst
ratio was 3.58 against 12. The read: a bf16 CONDITIONING tail, not a compile bug. The fp32 function alone moves a
fixture row's log-prob by up to 0.54 under a half-ulp weight dither (CPU, 267M), both bf16 arms move together across
X34's history, and compiled vs eager at the SAME precision reads 0.017–0.028 on CPU Inductor, 4–6× tighter than either
arm's distance from fp32. One more tail on a passing compiler would have been a FATAL.

**The rule now** (`compile_regions.YARDSTICK_ARMS` = eager, fp32 reference; every other arm, "compiled" or "graphed", is
the judged one):
1. **A yardstick out of margin is an EVENT, not a refusal.** It is recorded (`watch.*.yardstick_out_of_margin`: the arms
   past M with shift, batch row and fixture row; M; M′; the first read's shift per arm, compiled included; and a
   `📏 [CompileCanary]` log line) and never counts toward confirm / FATAL (`compile/canary_yardstick_events` counts it).
2. **The rung re-runs ONCE at M′ = max(M, `YARDSTICK_HEADROOM` 1.25 × the largest YARDSTICK shift)** and is judged
   there: a new relocation at M′, new arms, the full verdict (the first read's verdict is discarded, since its gradient
   may carry the 0/1 switch the margin exists to exclude). **The compiled arm's shift never sets M′**, so a miscompile
   cannot widen its own bar, and a compiled arm past M with every yardstick inside is refused exactly as before. The
   startup gate's rungs (`r1_rungs`, the graphed route's accumulation rung) take the same path.
3. **Deterministic** (rule 8). The arms' forwards do not read the old log-probs, so the re-run reads the same shifts
   (CUDA arms are run-to-run bit-identical, window D), and the yardstick sits at 1/1.25 of M′. A yardstick STILL past M′
   did not reproduce its own forward: `YardstickRefusal`, which still refuses and still counts on the ladder.
4. **The log names the failing arm.** "compiled DISAGREES with eager" only when the compiled arm failed; a
   `YardstickRefusal` (the case above, and the bf16 DRIFT rule, eager bf16 vs fp32, now typed as one) is worded "the
   canary's YARDSTICK refused (eager / the fp32 reference, NOT the compiled arm)", in the warning and in the FATAL.
5. **The instrument gap closed.** Every record names each arm's worst row (`clip_margin.worst_row`, and
   `worst_fixture_row`: the committed fixture row it tiles, via `R1Batch.fixture_rows`) and carries `same_precision`:
   max |log π_compiled − log π_eager| and its row, **RECORDED with no bar** until a CUDA census reads it. An UNCONFIRMED
   or FATAL verdict writes the policy `state_dict` beside its record (`policy_pt`, `canary_u<update>_<verdict>_policy.pt`),
   at most `compile_canary.DUMP_WEIGHTS_MAX` = 3 per process (a module counter no checkpoint carries, like K9(b)'s T61
   cap). At u1700 neither the row nor the weights could be recovered.

**What it does at u1700.** Eager's 0.856 is an event; the rung re-runs at M′ = 1.07; both arms sit inside (0.52, 0.86)
and no row switches; the gradient rule judges the re-relocated rows. Expected verdict PASS, and nothing carries to u1800.
The gradient rule's ratio on the re-relocated rows is **UNVERIFIED** (the u1700 weights are gone); the planted replay
(`canary_yardstick_test`) and the real learner's (`compile_canary_test`, slow) read PASS with one recorded event.

**What it does not cover.** **UNVERIFIED:** the 1.25 headroom and the same-precision gap's healthy range have no CUDA
census behind them. A pin before this change (X34 at `ae2d0901`) keeps the old rule; its only remedy for a yardstick tail
is the `--canary-shift-max` raise on a resume (below, and `TRAINING_RUN_SOP.md` §3 rule 5).

### bf16 drift on a sharpening policy: the loss is read on the golden's own rows (`gen3_r1_loss_on_golden_rows_v1`, 2026-10-17)

**The event.** X26 (`rb_x26_s2601`, pinned `9c033489`) read the startup gate's bf16 DRIFT on the loss at about 1e-4 at
every 6-hourly restart until 2026-10-17 03:35. The 121.14M restart then read **6.68e-3, 48 % of the 0.014 bar**, with
eager bf16 at the same value, so it was bf16 rounding, not the compiler. The first canary after it (update 10, 122.13M) read
4.86e-4. The two read the SAME rows (slice 0, all 64 golden rows tiled to B = 2,048, relocated the same way;
`compile_trainer.fixture_index`), so only the weights differed.

**THE FINDING (CPU, X26's own checkpoints loaded strict at this checkout, B = 64 untiled, eager bf16 vs the fp32
reference on the relocated rows exactly as the gate builds them; `designs/research_state/measurements/
x26_bf16_drift_2026-10-17/`).** CPU tracks CUDA on this statistic: on the 121.14M gate weights it reads 6.29e-3 at B = 64 and
6.31e-3 at B = 2,048 against CUDA's 6.68e-3, and it relocates 608 / 2,048 rows, the same count as CUDA.
- **It is not a smooth growth.** From 2M to 113M the drift and its 16-seed dithered null (relative 2^-9, half a bf16 ulp) sit
  flat. The null median is 2–10e-4, and 0 of 476 draws exceed 0.007. From about **114M** the null steps up: 3 of 204 draws
  pass 0.014 (1.5 %, max 2.05e-2), and 21 of 204 pass 0.007. The undithered 118M checkpoint reads **1.15e-2 (82 %)**. Adjacent
  checkpoints differ by 10x (118M 1.15e-2, 120M 1.5e-3). It is a per-state LOTTERY, and the trend cannot be extrapolated to a
  crossing date.
- **ONE ROW carries it.** The policy term is 95 % of the loss difference. Inside it, golden **row 9** carries **88–100 %** of
  the drift at 117M, 118M, 119M and 121.14M. Row 9 has A = −3.23, the fixture's largest |A|. Its raw ratio sits within the
  margin of its switching bound 0.85, so the relocation moves it to **ρ = 0.85 e^(2M) = 3.59**. That gives it a weight of
  ρ|A| = 11.6 in a 64-row mean, 18 % of the whole policy term.
  - The row's bf16 log-prob shift went from 0.002–0.009 (110M) to 0.046–0.09 (116M on).
- **The cause is the FUNCTION's conditioning, smooth, not a near-tie.** A half-ulp dither moves row 9's fp32 log-prob by a
  median of 0.04–0.11 from 116M on, against about 0.004 for a well-conditioned heavy row (row 46). Its sensitivity scales
  LINEARLY with the dither from 2^-20 to 2^-7 (×8 per 2^3), so there is no discrete op near a tie (`scale.py`). bf16's shift
  on that row is the size of the fp32 function's own move under a half-ulp weight change, which every optimizer step
  dwarfs.
- **No layer outgrew bf16.** Each trunk round alone in bf16 contributes the same order of drift, with random signs; rounds 0
  and 1 dominate and the extra round is smaller. The rounds' output |max| (≈ 6) and rms (≈ 1.0–1.2) are flat across the run.
  `move_belief.move_head.weight` (the startup line's worst per-parameter ratio, 6.33 < 12) is the ENVELOPE's
  per-parameter statistic, not the loss drift's. Its loss term contributes 0 to the drift.
- **The relocation amplifies the loss statistic, and the bar was never derived on it.** The loss is CONTINUOUS in ρ (the
  clip switches the GRADIENT, not the value), so it needs no relocation. But T36's move multiplies a relocated A < 0
  row's loss weight by ≈ 4.2 (ρ 0.85 → 3.59), and by e^(2ΔM) more under a raised `--canary-shift-max`.
  `BF16_DRIFT_BARS` / `BF16_ENVELOPE_K` were derived in window D (2026-10-10) on the golden's OWN rows, five days before the
  relocation existed. On the golden's own rows, the same 170 dithered draws (114M–123M) read **max 7.7e-3 (55 %)**, 0
  over 0.014 and 1 over 0.007. The relocated reading of the same draws: max 1.5e-2, 2 over 0.014 and 6 over 0.0105.
- **A T37 raise makes the relocated reading worse.** At `--canary-shift-max 0.28` (the value §4.2's RED action stages
  from the 0.218 read), the same 170 draws read **7 over 0.014 (4.1 %, max 3.4e-2)**, against 2 at 0.18. The undithered
  118M checkpoint reads 0.01397, on the bar's edge. The golden-row reading does not depend on the margin.
- **The grad drift** (bar 0.68) reads at most 0.226 over 680 draws (33 %). It is also elevated from 114M (0.09–0.12
  undithered) and is not close.

**Forecast** (CPU; the CUDA statistic tracks it within about 6 % on the one direct comparison, so every probability below is
**UNVERIFIED on CUDA**):
- No crossing date exists, because there is no trend: through 113M the statistic is flat and far from the bar.
- Since about 114M each startup gate draws from a distribution with **≈ 1–2 % per restart above 0.014**: 5 of 374 pooled
  draws over two nulls, 90 % interval about 0.6–2.8 %.
- It is **≈ 4 % at a shift_max of 0.28**.
- Over X26's remaining ≈ 13 six-hourly restarts to 500M, that is ≈ 15–20 % for at least one refusal if the regime holds,
  and ≈ 40 % after the raise.
- The canary is not exposed the same way. Its confirmation needs both disjoint halves, and row 9 sits in one half only.
  The startup gate is single-shot.

**The responses, ranked** (cost → verdict):

| option | cost | verdict |
|---|---|---|
| **(b) read the loss on the golden's own rows** (`gen3_r1_loss_on_golden_rows_v1`, BUILT below) | none at runtime: host-side arithmetic on each arm's own log-probs, no extra forward | **THE FIX, for every pin at or after it.** It keeps the DERIVED bar and its derivation's statistic, so the teeth are unchanged by construction. It removes the relocation's ×4.2 (and the T37 raise's e^(2ΔM)) from the loss. |
| (b′) normalise the policy part by its sensitivity (s = \|ΔL_pg\| / mean(ρ\|A\|), against the shift bar) | none at runtime | REJECTED. It is a new statistic with a new, unmeasured bar, and it is about 10x looser on a forward fault in the logits; (b) needs neither. |
| (c) a check-only, raise-only override of the loss / grad drift bars, like T37 | ≈ 0.5 agent-day | NOT BUILT. X26's pin cannot take new code anyway. A raised loss bar loosens the teeth with no measured basis, and (b) makes it unnecessary. |
| (a) fp32 for one trunk round | ≈ +10 % `train_ms` (one of three rounds of the +29 % fp32 trunk; ESTIMATE, UNVERIFIED) | NO. No round dominates (comparable shares, random signs), so it would only scale the lottery. For X26 it is a TRAINING change, so it needs an amendment. |
| (d) the fp32 trunk | `train_ms` +29 % (42.1 → 54.3 s) | A working stopgap. It is a training change for X26 (Amendment 3, never silently), and it is not needed: the defect is the statistic, not bf16. |
| fp32 for the move_belief head / the logits | — | MOOT. Both already run fp32 (design (b)), and the move_belief term contributes 0 to the drift. |

**What X26 can use: none of the code.** X26's pin `9c033489` reads the relocated statistic, and the launcher refuses to move a
live pin. What it can use with no code:
1. **The rollback.** If a startup gate refuses ONLY on the bf16 loss (the `DRIFTED … loss` or `COMPILED … loss:` lines;
   never a gradient, per-parameter or margin refusal), resume the SAME run at the SAME pin from the newest periodic
   checkpoint whose CPU pre-flight reads at most half the bar. The pre-flight is `scripts/newrule.py <ckpts> 64` in the
   measurement directory; its "relocated-row drift" column is what the pinned gate judges, and it takes about 10 s per
   checkpoint on the agent cores. Adjacent checkpoints differ by up to 10x, so this costs ≤ 1–2M steps.
2. **Weigh the staged T37 raise against its cost.** It moves the per-restart refusal odds from ≈ 1–2 % to ≈ 4 % on these
   states. Today's compiled shift (0.218) is 3.3x below the refusal (M = 0.72). The canary 10 updates later read 0.1285.

**THE FIX (`gen3_r1_loss_on_golden_rows_v1`).** `compile_regions.golden_row_loss` reads each arm's loss with its policy term
re-evaluated on the golden's OWN old log-probs: L − pg(relocated old) + pg(raw old), from the arm's own log-probs and the
normalised advantages. `micro_step`'s formula is computed in float64, with the fork mask and the accumulation rung's per-micro
sum. `ClipRelocation.raw_old_log_prob` carries the rows.
- `_r1_judge` and the graphed accumulation rung hand the bf16 rule those losses (`_r1_verdict_bf16(..., losses=)`).
- Every LOSS comparison (the DRIFT and the ENVELOPE) reads them.
- The GRADIENT comparisons stay on the relocated rows, where rule 8 needs them.
- The R1 line prints `[loss on the golden's own rows; relocated rows: drift …, envelope …]`, and every canary record's
  `global` gains `drift_loss_relocated` / `envelope_loss_relocated` beside the judged values.
- The bars, `BF16_LOSS_FLOOR` and the fp32 rule are unchanged, and training is untouched (only the gate's judgement reads it).

**Proof** (`compile_regions_loss_rows_test`, each test fails on revert; two revert mutations were applied, and each was
caught):
- The reconstruction EQUALS a forward on the golden's rows (K9 learner, bf16 and fp32 trunk, within 1e-5 relative), both for
  one micro-batch and for the accumulation rung's two.
- The gate hands the rule those losses, and the line names them.
- A relocation-amplified healthy one-row shift (X26's row 9 on hand-built arms) passes on the golden's rows and is refused
  on the relocated ones.
- THE TEETH still refuse three planted faults: a broad 0.2 log-prob shift (DRIFT), a non-policy loss fault in the eager arm
  (DRIFT) and a compiled-only loss fault (ENVELOPE).
- On X26's states through production's own functions: 118M reads relocated 1.15e-2 → golden 7.0e-5, and the gate weights
  6.29e-3 → 3.14e-3 (B = 2,048: 6.31e-3 → 3.16e-3).
- **Not covered by a revert test:** the graphed accumulation rung's wiring. It runs only on CUDA with
  `--learner-cudagraphs on`; its reconstruction function is the tested one.

### The gate releases its memory (`gen3_gate_releases_memory_v1`, `gen3_gate_memory_v1`, `gen3_canary_memory_v1`, 2026-10-15)

**The failure.** `rb_xs_s2701` (a fresh `--arch production` with the X26 argv at `71351ff5`) passed the bf16 gate and was
refused by the update-fit check: demand 9,790 MiB reserved, headroom 978 < 1,024, while the dry update itself peaked at
6,579 MiB allocated. The startup ledger's "compiled regions: gate + prewarm + lock" row read +7,188 MiB reserved, peak
8,244 allocated — against +4,840 / 5,428 on the `ffe2961b` production smoke (pre-T36, no ride-alongs).

**The attribution (GPU, `e092a02b` = `71351ff5` + the memory record, the X26 argv, the old behaviour reproduced exactly:
reserved 9,790, peak 8,244).** The new per-step record (`GateMemory`, MiB allocated after / high-water):

    start 717/805; relocation forward 5,203/5,362, compiled arm 1,237/8,201, eager arm 1,247/8,201,
    fp32 reference arm 1,257/8,201, [the perturbed rung] relocation forward 5,252/8,201, compiled arm 1,278/8,244, …

T36's relocation forward (`relocate_clip_boundary`: one fp32 grad-mode forward of R1's eager body at B = 2,048, no
backward) left **4,486 MiB** allocated after it: its outputs are STASHED on the policy (the extractor's `stash` container,
the masked distribution), and through them its whole autograd graph, every saved fp32 activation. That graph stayed alive
under the COMPILED arm (~3.0 GiB of its own) until the arm's forward replaced the stashes: high-water 8,201 = 5,203 + the
arm. So the +2.8 GiB over `ffe2961b` is T36's retained relocation graph stacked under the compiled arm; the ride-along
heads are not in R1 (their update is the eager tail) and add 43 MiB of weights. The allocator then kept the 8.2 GiB peak
CACHED, and the update-fit check read that cache as the update's demand. The same rungs run in the MID-RUN CANARY, with
training state resident: at update 2 it peaked at **8,223 MiB allocated** (training state 984 + the check's transient
7,239) — so even a passing startup would have taken the canary to the card's edge every 100 updates.

**The fix — three parts, none changes a number:**
- **The relocation keeps nothing for a backward**: its forward runs under `saved_tensors_hooks` that drop every saved
  tensor (K9(b)'s probe recipe, `consistency._drop_saved`; hooks change what autograd keeps, never a kernel or a value —
  `compile_regions_memory_test` pins the reference log-probs bit-identical and that the forward saves 0 bytes, where the
  plain forward saves 2,285 tensors on the CPU golden). `release_gate_memory` then detaches the module-level stashes.
  `release_autograd_stashes` walks INTO the typed stash containers (`ExtractorStashes`, `OpStashes` and the dicts /
  tuples / dataclasses they hold — `gen3_release_walks_stashes_v1`, 2026-10-19); before, it reached only tensors held
  DIRECTLY on a module, so a forward relying on the release alone kept its graph (`consistency_test` pins the walk).
- **The gate hands its cache back**: `arm_compile_sentinel` calls `release_gate_memory` + `empty_cache()` after the
  LOCK, so the startup transient is not held where nothing else can use it.
- **The update-fit check measures what training will need** (`gen3_update_fit_v2`, `designs/training/learner_lifecycle.md`
  "The update fit check"): from an emptied cache, the dry update and then the canary's check once
  (`CompileCanary.measure`), demand = the peak reserved over both.

**The record.** Every gate prints a `🧮 [CompileRegions] gate memory` line; every canary record carries `memory`
{start_alloc_mib, transient_mib, peak_alloc_mib, reserved_mib} and its PASS line prints the card; TB gets
`compile/canary_peak_alloc_mib` and `compile/canary_reserved_mib` on a canary update. A smoke moves the FIRST canary
with `GEN3AI_CANARY_FIRST=<n>` (a test hook; the cadence after it is unchanged; unset in production).

**The verification (GPU, the X26 argv, `GEN3AI_CANARY_FIRST=2`, the fix's code at `20c07e0b`):** the gate's record reads
`start 717/805; relocation forward 1,219/2,061, compiled arm 1,237/4,217, eager arm 1,247/4,725, fp32 reference arm
1,257/5,438, …, fp32 reference arm 1,298/5,480` — peak **5,480** allocated (the `ffe2961b` smoke's 5,428 + 52), the
ledger's gate row +492 reserved (was +7,188); the update-fit demand 8,324 MiB, headroom **2,444** (was 978, FATAL);
the canary at update 2 PASSED at **5,458** allocated (training state 984 + transient 4,474), reserved 8,324; the run
held 8,324 reserved at every update (= the prediction) through 4 updates, no OOM. Full table: `learner_lifecycle.md`
"The update fit check"; `designs/endstate/design_x26_readiness.md` I31. The K9 learner golden is IDENTICAL.

### Lever C: R1's GRAPHED route (`--learner-cudagraphs {off,on}`, `gen3_r1_graphed_route_v1`, 2026-10-10)

**Why a second route.** Window B (`perf_phase_2026-10-10/README.md`, F-PB-2): R1 under Inductor's cudagraph trees
passes the startup gate bit-equal, then dies on the first update in the grad-balance probe's
`autograd.grad(retain_graph=True)` — "graph recording observed an input tensor deallocate during graph recording that
did not occur during replay". Cudagraph trees record ONE backward per forward; a retain_graph probe traverses R1's graph
a second time.

**The retain_graph inventory** (every autograd call that re-enters R1's graph, 2026-10-10):

| consumer | where | which micro-steps |
|---|---|---|
| the per-term noise sampler (`noise_scale_terms.PerTermNoiseSampler.flush_micro`, one `autograd.grad` per loss group) | `ppo.train`, before `backward()` | epoch 0's first accumulation group (`accum` micro-steps) of a cadence update (`plan.noise_terms`; EVERY update under `--adaptive-batch policy`) |
| the grad-balance probe (`grad_balance._flat_grads`, one `autograd.grad` per term) | `ppo.train`, before `backward()` | ONE micro-step of a cadence update: the first whose terms are all present (data-dependent, usually epoch 0's first) |

Not R1's graph: the capacity cosine (`capacity_telemetry.flat_grad`, its OWN eager forwards), the K9(b) behaviour probe
(its own forward, before the loop), the ride-along heads (`ridealong_terms._ridealong_step`'s
`backward(retain_graph=…)` runs on the heads' own graph over `RideAlongBatch.detached` inputs).

**The route** (`compile_regions.r1_route_for` ← `diagnostics_cadence.micro_retains_graph`, decided BEFORE each R1 call,
deterministic, no try/except): a micro-step whose per-term sampler is collecting (and has not failed) or whose
grad-balance probe is still PENDING takes the un-graphed COMPILED route; every other micro-step takes the GRAPHED route.
At the production shape (accumulation 32, 48 micro-batches × 10 epochs) a diagnostic update runs ~32 of its 480
micro-steps compiled; a plain update (9 in 10) runs all 480 graphed.

**What `on` builds.** `compile_regions.install` compiles R1 a SECOND time (`model._compiled_micro_step_graphed`): the
same function, the same kwargs and preset options, plus `triton.cudagraphs` — a second DECLARED signature in `REGIONS`
(`Signature(route="graphed", gate="learner_cudagraphs")`), one more dynamo cache entry on R1's code object. Both routes:

- **gated** — `gate_regions` runs `r1_rungs` on each route, plus the graphed route's ACCUMULATION rung
  (`_r1_accum_rule`: two micro-batches, golden slices 0 + 1, ONE `.grad` accumulating across two replays, against eager;
  a `.grad` aliasing the graph pool would read 2·g₂ for g₁ + g₂);
- **prewarmed** (`prewarm_calls`: one call per route) and **inventoried** (`assert_inventory`: 2 entries);
- **locked** by the same stance; **canaried** — `CompileCanary` judges every declared route, the graphed one first
  (its cosine is `compile/canary_grad_cosine`);
- **no silent un-graphed route** — Inductor's `cudagraph_skips` counter moving after the install is FATAL at
  `assert_inventory` and at every sentinel `check` (`compile_control.cudagraph_skips`, `CompileControl.graphed_check`);
- `lifecycle/graphed_region_calls` (TB, flag on only) counts the graphed calls per update.

**Output lifetime.** Cudagraph trees overwrite a replay's outputs at the next replay, so nothing may read R1's outputs
or stashes across micro-steps. Inventory of every read after the call: `_mo` and the extractor stashes are read inside
the same micro-step (the eager tail, the rank probe, `_MaskedPi`); `micro_step.pack` and the calibration reader COPY
(`torch.cat`) before their deferred host read; the ride-along accumulator's deferred snapshots clone the batch fields and
hold only fresh head outputs; the capacity snapshot is read in the same micro-step; `_r1_arm` clones the loss. The
risk that remains is the AccumulateGrad steal (`.grad` taking the backward's pool output on a group's first micro-step):
the accumulation rung judges it at startup.

⚠️ **ON FAILS ON THE GPU (window E, 2026-10-10, F-PE-1; `perf_phase_2026-10-10/README.md`).** The accumulation rung
died at startup.
- **The error:** "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run", in the second
  micro-batch's `backward()`, at the embedding table's gradient-addition node.
- **The cause:** AccumulateGrad took the first replay's pool output as `.grad`, and the second replay overwrote it.
- **It surfaces as a bare `RuntimeError`, not a typed `CompileTrainerError`.**
- **Do not launch with it** until a follow-up materialises `.grad` before each accumulation group (or clones the graphed
  grads out of the pool) and types the rung's error. Off is unaffected.

**OFF is byte-identical:** one `torch.compile` with the production kwargs, one cache entry, the same gate rules, prewarm,
inventory line and canary; `r1_graphed_route_test.py` pins it. **The flag is a RUNTIME perf knob** like
`--compile-trainer`: not versioned, not in `model_config.json`, never pickled (`hparams._excluded_save_params`), not
fresh-only (the launcher's restarts forward it with the argv), keyed in the shared compile cache (`COMPILE_ARG_KEYS`).
Refused (`FATAL_CONFIG`) on cpu, under `--debug`, with `--no-compile-trainer` (`combination_checks`) and when an AUTO
`--compile-trainer` resolved off (`lifecycle._maybe_compile_trainer`). It stays a MEASUREMENT lever (default off) until
its paired benchmark (PLAN.md lever C, window E) adopts it.

### 🚨 The CPU Inductor tile2d codegen bug (`gen3_cpu_inductor_scalar_v1`, 2026-10-15) — torch 2.8.0; R1 on CPU carries `cpp.simdlen = 1`

**Read this before you compile anything on CPU.** Production compiles on CUDA only (Triton); `--compile-trainer` REFUSES
a CPU device (`compile_trainer`: "a CPU compiled learner has never been gated"). A CPU compile of R1 exists as a TEST
SEAM (`compiled_first_update_test`'s CPU cell, the `compile_regions_*` CPU cells), and on 2026-10-15 the post-U3
integration sweep found that seam RED since the end state became production (`ed58a3ad`), hidden behind a stale PASS in
`slow_tier_status.json` (the 2026-10-09 row).

- **Symptom.** `torch._inductor.exc.InductorError: CppCompileError`, g++: `error: 'tmp8' was not declared in this
  scope` on a `transpose_mxn<float,16,16,false>(in_ptr29 + x1 + 18L*tmp8 + 108L*x0, 108L, tmp12, 16L)` line. The typed
  wrapper (`compile_regions._fatal_compile_errors`) reported it as "a DECLARED REGION does not compile as one graph",
  i.e. as a graph break; it now adds that the BACKEND's code generation failed and points here.
- **Bisect (the sweep, `~/.cache/gen3ai/postu3_sweep/`).** `ed58a3ad~1` passes; `ed58a3ad` (production takes the
  end-state overlay: derived move resolution on), `10b3a6b6~1`, `882a3f86~1` and HEAD fail with the same kernel.
- **The kernel, read from the failing fusion (`TORCH_COMPILE_DEBUG`, 2026-10-15).** R1's forward graph fuses 25 nodes
  into one 2-D tiled CPU kernel (`group ([16], reduction [18])`: a vector across the 16 batch rows, K = 18 their seats as
  the outer loop). Op `op3858` is `buf3858[b, k] = buf3799[b, buf21[b], k] > 0` — `move_facts`'s
  `tm_a = o.pair_type_mult[ar, o.our_active]` feeding `hits_k = seat_dmg * acc_k * (tm_a > 0)` (`move_resolution.py`),
  a per-row gather at the active slot. Three `sum`s over K of strided-column products (`buf3759[6*k + 4]`, ...) sit in
  the same kernel. Inductor's CPU backend picks the TILED kernel (`TilingSelect._select_tiling_indices`: exactly two
  contiguous iteration vars, the last one also strided elsewhere) and `CppTile2DKernel.load` then emits a TRANSPOSED
  tile load for any index whose outer stride is 1 (`need_vec_transpose`, `torch/_inductor/codegen/cpp.py`). That check
  reads only the strides of the two tile variables: it does not exclude an index that contains an INDIRECT variable
  (`tmp8`, the vector-valued gather index, defined later in the inner loop), so the transposed load, hoisted into the
  kernel's preloads, names `tmp8` before it exists. The same load in the non-tiled path and in the tile kernel's tail
  loop is emitted correctly (`tmpbuf[x0_inner] = in_ptr29[... + 18L*tmp9[x0_inner] + ...]`).
- **The smallest repro** (`cpu_inductor_tile2d_test.py::_repro`; 4 lines, no model):

  ```python
  def repro(x, idx, q):               # x [B,6,K] · idx [B] long · q [B,K,6]; B = 16, K = 18
      g = x[torch.arange(B), idx]     # a row gathered at each batch row's own slot   (arange INSIDE the graph)
      return (g * q[..., 5]).sum(-1), g   # a [B] reduction over K of a strided column; the gather stored
  torch.compile(repro, backend="inductor", fullgraph=True)(x, idx, q)   # CPU, torch 2.8.0 -> CppCompileError
  ```

  Each ingredient is needed (measured, `designs/research_state/measurements/cpu_inductor_tile2d_2026-10-15/` `ingredients.py`): replace `q[..., 5]` by a contiguous [B,K] operand, drop the
  reduction (return `g * q[..., 5]` and `g`), or capture `arange` as a tensor instead of building it in the graph (then
  the batch coordinate is indirect too and the load stays untransposed) and it compiles. **Shape threshold** (AVX-512 box, 16 fp32 lanes; `shapes.py` there): fails for B >= 16 and K >= 16 (B 16 / 17 /
  32 / 64 / 256; K 16 / 17 / 18 / 32 / 64); compiles for B 8 / 15 or K 8. **UNVERIFIED:** the threshold on an AVX2 box
  (expected 8). The K9 learner golden's micro-batch is 16, which is why the CPU cell hit it.
- **A known CLASS upstream, not this trigger's report (GitHub API, 2026-10-15; the summaries were not independently
  checked).** pytorch/pytorch#193581 (OPEN, 2026-08-14, torch 2.12.1+cu130: `tmp13` undeclared in `transpose_mxn`,
  `index_select` beside a conv) and #181621 (closed 2026-07-16, torch 2.11.0+cu130: `tmp8` undeclared in a gather). Both are
  newer than 2.8.0, so a torch bump is not a known fix; **UNVERIFIED** whether either's change covers this repro (the env
  is pinned to 2.8.0; the cu128 twin is the same torch). The repro above is small enough to file.
- **Why this graph and not the one before.** The end-state graph carries `move_facts` / `switch_facts` (derived move
  resolution, U3); ~178 lines of `agents/model` use the `x[ar, ...]` advanced-indexing idiom (31 in `move_resolution.py`,
  13 in `damage_op.py`). `move_facts` ALONE compiles on CPU at B = 16, K = 18 (forward and backward); the failing fusion needs
  its neighbours.
- **The decision: (a) a CPU-only compile option.** `compile_regions.CPU_INDUCTOR_OPTIONS = {"cpp.simdlen": 1}` turns the
  CPP backend's vectorisation, and so the tiling, off: scalar C++ kernels of the same graph, only fp32 summation order
  inside a kernel may move. It is added to R1's `torch.compile(options=)` for a CPU learner on the default Inductor
  backend only (`cpu_inductor_options`); **a CUDA learner's compile kwargs are unchanged byte for byte** (Triton never
  reaches the CPP codegen; `cpu_inductor_tile2d_test::test_install_hands_R1_the_cpu_option_and_a_cuda_learner_nothing`
  and the two production-kwargs spy tests, now declared on the CUDA device). Why not the others:
  (b) RESPELLING is not a local change: `x.gather(1, ...)` lowers to the same indirect load and fails identically (`respellings.py` there);
  only a one-hot sum compiles (bit-identical to the index in eager), but that rewrites ~178 sites of the PRODUCTION CUDA
  graph, changes the backward from a scatter into a multiply-reduce, and needs the K9 golden, the gate and the canary
  re-judged on the GPU — a numerics-bearing change to fix a CPU test seam. (c) DECLARING CPU unsupported would drop the
  only CPU coverage of the donated-buffer pin (device-agnostic AOTAutograd logic) that this test exists for.
- **Evidence** (raw reads, scripts and verbatim outputs: `designs/research_state/measurements/cpu_inductor_tile2d_2026-10-15/`). `compiled_first_update_test` CPU cell: RED at `87e493c3` (337.9 s), PASS at the fix (277.6 s,
  `slow_tier_status.json` row recorded at the fix commit); the 4-line repro: FAIL without the option, compiles with it, no
  `transpose_mxn` / `Vectorized<float>` left in the C++ (pinned, plus a TRIPWIRE that fails when torch 2.8.0 stops
  failing it, so the workaround is retired with the bug). Alternatives measured on the repro: `cpp.vec_isa_ok = False`
  and `cpp.simdlen = 1` compile; `cpp.enable_tiling_heuristics = False` does not; `max_fusion_size = 1` compiles (and keeps
  tiling) but unfuses the whole graph.
- **Other CPU compile sites are exposed in principle and were NOT changed here (out of scope):** the T2 inference
  service's `torch.compile(decide)` (`inference/service/engine.py`) and `compile_opponents.maybe_compile_extractor` (the
  offline readers) compile the extractor on CPU; at B < 16 (the offline opponent compile runs B = 1) the tiled kernel is
  not chosen. A CPU compile of the end-state extractor at B >= 16 outside R1 would hit this bug until it carries the same
  option.
- **The CUDA twin is UNREAD at the end-state graph.** The production CUDA smoke passed with compile on (2026-10-15) and the
  CUDA row of `compiled_first_update_test` holds a PASS from 2026-10-04 (88 s) — a graph older than the end state, i.e. a
  stale PASS (testing.md "The SLOW-TIER LAST-KNOWN-STATUS gate"). Its read, for the training session's next GPU window:
  `GEN3AI_GPU_LEASE_TOKEN=<token> GEN3AI_TEST_ALLOW_GPU=1 scripts/ops/gpu_lock.sh timeout 1200 "$PY" -m pytest
  src/agents/model/compiled_first_update_test.py -m slow -k on_cuda -q -p no:cacheprovider` (~2-3 min: the 2026-10-04 run
  took 88 s; the end-state R1 compiles larger). It records its row in `slow_tier_status.json`.

## From the training leaf (moved 2026-10-10)

> These sections headed `src/agents/training/CLAUDE.md` until its 2026-10-10 cleanup; moved here as they
> stood (minus statements verified FALSE). Where an earlier section of this doc says the same in more
> detail, both are current; fix both in the same pass.

### The compile flag (`--compile-trainer`, DEFAULT ON)

**`--compile-trainer`** is the GPU/LEARNER compile (auto-on for cuda) — the CUDA forward **and backward** the PPO step runs. The CPU/ROLLOUT half (the compile-opponents flag, its preload and strict variants, the forkserver preload, the cache prewarm and the cross-process revert quorum) was deleted in the deletion pass (U3): under the Rust env core every policy opponent forwards through the inference service, so the trainer compiles no opponent. What is left of the opponent compile is `agents.model.compile_opponents.maybe_compile_extractor(model, enabled, label, hide_cuda)`, used by the OFFLINE paths (the prober's counterfactual view, `agents.model.snapshot`'s opponent loads, `opponent_parity`, `compile_cache_benchmark`).
**fp32 matmul precision `highest` is the ONLY precision** (TF32 retired, deletion pass K2; `--matmul-precision`
is DELETED — `designs/deleted_flags.md`). Nothing in the trainer sets it; `metadata.json` still records the
realized value as `matmul_precision`, and every parity gate (the region gate, the canary, T2's judge, K9(b))
refuses a process at any other precision. A resume or fork of a run whose `metadata.json` recorded
`matmul_precision: high` is refused `FATAL_CONFIG` naming the pin (`model_version.retired_levers`,
`LAST_COMMIT_K2`). K9(b)'s behaviour gate (`rust_rollout/consistency.BEHAVIOUR_GATE`): `max` |Δ log π| < 1e-4,
DETERMINISTIC (`gen3_behaviour_tie_exclusion_v1`: the policy forward selects with topk / argmax, and a selection within a rounding error of its cutoff can resolve differently in T2 and in the learner — such rows, 3.7 % at a relative margin 2e-4, are EXCLUDED and every other row is judged single-shot; the offending rows are dumped to `<run_dir>/behaviour_violations.jsonl`;
`designs/training/learner_gates.md`).
🚨 **`--compile-trainer`'s startup gate (the REGION gate, `compile_regions.gate_regions`) runs on REAL
obs rows**, never zeros. The rows are the committed fixture `src/agents/model/compile_parity_obs.npz`
(R1 ALWAYS uses the K9 golden's labelled buffer — a key the run declares that the golden lacks, e.g. the fork arm's `fork_pg_m`, at its DECLARED placeholder (`compile_regions.fill_value`), an unfillable space FATAL_CONFIG, and a fill that shrinks R1's judged parameter set below the golden rows' own FATAL_CONFIG; P10-C — the old zero-label fallback left ~48 critic / intent / belief parameters unjudged on every fork run); regenerate the fixture with
`python -m agents.model.compile_parity_fixture --write` after an obs-layout change, and a stale
fixture REFUSES. It holds R1's loss and every policy gradient (cosine ≥ 0.9999 plus the per-parameter
rule, the bar chosen by weight regime — `train_verdict` has NO default bar) to eager; there is no decision-readout arm any more (the compiled rollout region R0
was deleted, P10-E — T2 gates every served bucket on load). **On FRESH weights** (a fresh
launch: the zero-init pointer head makes every legal log-prob `-log(n_legal)`, so that check cannot
fail) it ALSO runs on a seeded, bit-exactly-restored perturbation of the policy
(`agents.model.parity_probe`, `gen3_fresh_parity_probe_v1`), and a vacuous comparison REFUSES rather
than passes. The offline CPU opponent compile runs a decision-level parity check once per distinct
weights (`agents.model.opponent_parity`), which RAISES on a mismatch. 🚨 **HEAD runs torch >= 2.8 ONLY** (deletion pass K1, 2026-10-02): `utils/torch_floor.py`
exits the trainer `FATAL_CONFIG` on an older torch, because every 2.5.1 path is gone — the
extractor-only compile and its gate, and the CUDA trunk split (`gen3_inductor_trunk_split_v1`) that
2.5.1's single Inductor graph needed (it miscompiled on real rows: every default cuda run from
`28eaef29`, 2026-08-17, to 2026-09-28 — argmax agreement 70.9%, gradient cosine 0.778). A 2.5.1 run
resumes PINNED to its own commit through the launcher, which selects `gen3ai_stable`
(`designs/training/compile_flags.md` "Lane K1").
🚨 **THE COMPILE SENTINEL (`gen3_compile_sentinel_v1`) makes a silent recompile / eager fallback
FATAL.** `src/agents/model/compile_control.py` is the ONLY runtime module that touches
`torch._dynamo` — add nothing that does elsewhere. Phases: `torch._dynamo.reset()` → install + gate
the DECLARED REGION (R1) → prewarm its DECLARED signature (`compile_regions.prewarm_calls` is the
declaration) → LOCK at the END OF STARTUP, before the first real
iteration (K6; `🧊 [COMPILE LOCK]`) → RELEASE when `learn()` returns (the lock's scope is training).
While locked, any recompile, late first compile or cache-limit hit exits `[CompileSentinel] FATAL` /
`FATAL_CONFIG` (not restarted), and the message NAMES the failing guard (`UNDECLARED SIGNATURE — the
failing guard(s): …`). At update 10 and then every 100 updates the IN-RUN PARITY CANARY (`agents/model/compile_canary.py`) holds
R1 (the compiled micro-step) to eager on the K9 golden's real labelled rows at the startup gate's bars
(loss + every policy gradient — EVERY canary; there is no decision-only canary), on rows moved off the PPO clip boundary (T36,
`compile_regions.relocate_clip_boundary`, as the startup gate). A disagreement is CONFIRMED in the same update on BOTH
disjoint halves of the rows (`compile_trainer.FIXTURE_HALVES`, eager recomputed): confirmed ⇒ checkpoint + `[CompileCanary] FATAL` naming the safe
rollback point (`<run_dir>/canary_verdicts.jsonl`); unconfirmed ⇒ counted, and two consecutive ⇒ FATAL. The rank probe is HOOK-FREE (a forward
hook on a compiled module is a guard — the iteration-1 signature `8fc297a2` used to absorb). 🚨 **The learner compiles ONLY as its DECLARED REGION** (K8, `agents/model/compile_regions.py`, `designs/training/compile_flags.md` "K8 — DECLARED COMPILE REGIONS"): R1 the micro-step, `fullgraph=True` at ONE declared signature (`batch_size` rows: a ragged micro-batch is REFUSED by the dispatcher, `gen3_r1_no_ragged_v1` — the collector refuses an update that does not divide by the micro-batch, and `check_shape_stability` judges the REAL update size, `--rollout-target-samples`, at startup). The compiled rollout region R0, its eager batch-1 route and `install_rollout_region` are DELETED (P10-E, owner-approved 2026-10-03: the Rust collector serves rollouts through T2, so R0 was never called, yet it was compiled, gated, prewarmed and canaried every launch); the rank probe reads R1's stashes (no second forward, the spectra on the device); on a CUDA buffer every micro-batch is STAGED to the device by a prefetch thread (`--device-batch staged`, the default since 2026-10-01; `resident` = one device copy of the whole flattened buffer per update, +~1.1 GB of update peak; `instrumented_ppo/device_batches.py`, bit-identical batches in every mode, the same permutation). 🚨 **Startup RUNS one dry update** (`agents/training/update_fit.py`, `gen3_update_fit_v1`): one real `train()` epoch on a fixture rollout of the buffer's full shape, the learner restored bit-identically, and a first update that would not leave 1,024 MiB of device headroom (or OOMs) is `UpdateWontFit` (FATAL_CONFIG) — `designs/training/learner_lifecycle.md` "The update fit check". ONE startup gate per region (`gen3_one_gate_per_region_v1`); the trainer's compile step (`compile_trainer.preflight_compile_trainer`) compiles nothing and refuses a learner without the micro-step. 🚨 **No region runs eager silently** (`gen3_no_silent_eager_v1`): a compiled-route call whose Python body executes, a ragged micro-batch, or dynamo disabled / errors suppressed / the stance moved under the lock is a typed FATAL; every update logs `lifecycle/compiled_region_calls`, `lifecycle/eager_fallback_calls`, `lifecycle/eager_share`, `lifecycle/update_wall_s` (`designs/training/compile_flags.md`). R1's startup gate and the canary judge its per-parameter gradient by WEIGHT REGIME (`compile_regions.weights_regime`: fresh vs trained, measured bars — `designs/training/compile_flags.md`), through the SAME rungs (`compile_regions.r1_rungs`); a parameter training never moved — bit-identical to its fresh-build value in the init record every checkpoint carries (`compile_regions.record_param_init`, a `@startup_builder` in the fresh construction), or exactly 0.0, the only rule on a checkpoint with no record (the oracle arm's belief head, its labels all PAD) — is judged at the FRESH bar on the live weights and on its own perturbed rung at the TRAINED bar (`gen3_r1_unmoved_param_v1`, `gen3_r1_unmoved_init_v1`). 🚨 **Every R1 comparison PROVES its two arms independent and names a non-finite gradient** (F-XC-4, 2026-10-05): `compile_regions._r1_pair` reads `region_calls`' counters — the compiled arm dispatched the compiled route once and ran R1's Python body zero times, the eager arm the reverse — else a typed refusal (`gen3_gate_independent_arms_v1`); a NaN / inf gradient on either arm is `NonFiniteGateArmError` naming the parameters, never a cosine (`gen3_gate_nonfinite_named_v1`; `_cos` returns NaN, not 0.0, on a non-finite side). Code inside R1 must stay a static-shape program (the fold contract above). 🚨 **Every lever R1 reads is DECLARED at startup from the resolved config** (`gen3_r1_declared_levers_v1`): R1's declaration is its static flags (`MicroStatic`, resolved by `TrainSetup._micro_static`) plus the observation key set (never a rollout's data; R1 takes no per-update tensor dict), and every compiled update is held to the declaration (`compile_regions.check_r1_declared`, a typed FATAL naming the field or key). A new R1 lever is declared in that static, never by relaxing the lock (`compile_flags.md` "R1's DECLARATION"). A NEW compiled signature is added to the region table (`compile_regions.REGIONS`)
and its prewarm, never absorbed after the lock; a learner-process caller OUTSIDE the regions runs the
extractor eager already (the regions compile functions over the module and never patch `fe.forward`,
so there is no eager-route wrapper any more — `eager_extractor` was deleted with the Python fork arm, its last caller); the learner process's rollout forward (`policy.forward` / `rollout_core`, for `predict`-class callers) is ALWAYS eager — the late-shape
table in the doc lists every caller. An unknown torch or a drifted torch internal
(`_SOURCE_HASHES`) REFUSES — checked WHERE A LEARNER COMPILES (`compile_control.require_supported_torch`:
the compile preflight + every `CompileControl`, FATAL_CONFIG with the reason), never at import, so a
non-compile run never trips it (P10-D). TB: `compile/recompiles_after_lock` must stay 0.
🚨 **THE COMPILE CACHE IS THE RUN'S OWN (K3, `gen3_hermetic_compile_cache_v1`).** Every process of a
run's tree compiles into `<run>/compile_cache/` (Inductor, Triton, T2's AOT packages), declared by
`lifecycle._declare_compile_cache` the moment the run dir exists — EMPTY at a fresh launch or fork,
reused only by the run's own restart under a matching stamp (commit, clean tree, torch, interpreter,
`compile_config.config_row_hash()` — data only, so stamping never imports the sentinel), wiped otherwise; printed as `🧊 [CompileCache]`. Tests and
every other compiling process get a fresh private dir on the REAL DISK (`$GEN3AI_SCRATCH`, else
`~/.cache/gen3ai/tmp`; never tmpfs `/tmp`), deleted at exit, swept by PID if its owner died
(`agents.model.compile_cache`). **A new
`torch.compile` site must call `ensure_hermetic_cache()` first** — `compile_cache_test` fails a
production module that does not. The box-wide `/tmp/torchinductor_<user>` and
`/tmp/gen3ai_inductor_cache` are no longer used. Cost: a fresh launch always compiles cold (see the doc).
**Full detail — in [`designs/training/compile_flags.md`](compile_flags.md).**
