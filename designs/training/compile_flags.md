# Training — compile flags

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was
split by topic (it was 8,219 lines / 676 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.**

---

## Compiled CPU opponents (`--compile-opponents`, DEFAULT ON) + BLAS thread pinning

> **Two independent compile flags, split by WHO and WHERE** (renamed 2026-08-14 from the
> single `--compile-extractor`, which said neither): **`--compile-opponents`** is the
> CPU/ROLLOUT half documented in this section — frozen opponents in the env workers.
> **`--compile-trainer`** is the GPU/LEARNER half, documented below. They are orthogonal;
> a run can take either, both, or neither.

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
(a different checkpoint's `load_state_dict` does NOT recompile), and a shared
`TORCHINDUCTOR_CACHE_DIR` turns each worker's cold codegen into a cache hit.

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
  would prefix every state_dict key with `_orig_mod.`. It also calls
  `Gen3FeaturesExtractor.disable_observation_debugger()` (a method, not a reach-in assignment to
  `fe._debugger`), because the debugger's numpy asserts inside `forward` make dynamo die creating a
  guard.

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

**What ships instead (all four compose — `agents/model/compile_opponents.py`):**

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

**The caches work.** The shared Inductor cache is HIT at a promotion (13 files written, vs 6600+ at
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
## FP32 matmul precision (`--matmul-precision {highest,high}`, DEFAULT `highest`)

`gen3_matmul_precision_v1`. **`highest`** is PyTorch's own default — full FP32 matmuls, no TF32 —
and `main.train.config.apply_matmul_precision` then calls NOTHING, so the default process is
byte-for-byte what it was before the flag. **`high`** calls
`torch.set_float32_matmul_precision("high")` in the TRAINER process, letting fp32 matmuls run on
TF32 tensor cores (Ampere+; ~10-bit mantissa). It does not reach the env/opponent worker processes.
The resolved value is read back from torch and **stamped at launch** as
`🧮 [MATMUL PRECISION] <value>` (beside `🔭 [OBS SOURCE]`), and recorded per checkpoint in
`metadata.json` as `matmul_precision`. A RUNTIME perf knob: not in `model_config.json`, not
inherited on a resume (a launcher restart re-sends the original argv, so it holds across one
run's restarts). **UNVERIFIED:** its speed-up and its effect on the trained policy on this box
have not been measured.

**Under `high`, `--compile-trainer`'s startup parity gate switches to a precision-aware rule**
(`gen3_tf32_parity_gate_v1`): a fixed fp32 tolerance is the wrong question once eager and compiled
each run their OWN TF32 kernels. The first TF32 launch (`ai_v14_04_lbat_t32`, 2026-09-28) died on
the fixed 1e-4 at a compiled-vs-eager 7.62e-03 on zero obs. The rule, and the measurement behind
it, are under "The startup parity gate" below.

## Compiled GPU trainer (`--compile-trainer`, DEFAULT ON for cuda)

`torch.compile`s the LEARNER's feature extractor — the CUDA forward **and backward** the PPO step
runs. The other half of the pair above, and the larger of the two.

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
`FATAL_CONFIG` with the same message. `--no-compile-trainer` is the opt-out, and it is also how you
KEEP the ObservationDebugger — see the trade below, which every default cuda run now makes.

**⚠️ The device is only HALF the auto default, and the other half is easy to miss.**
`check_shape_stability` (below) refuses `--async-rollout` and a rollout that does not divide by
`--batch-size` — both correct for someone who ASKED for the compile, and both fatal for a DEFAULT,
because they would convert two classes of command that work today into a startup `FATAL_CONFIG`.
So `resolve_compile_trainer_auto` runs those same checks and, on a refusal, **leaves the default OFF
and says why** rather than refusing to launch:

```
⚡ --compile-trainer would be ON by default here, but this config cannot take it — leaving it
   OFF rather than refusing to launch. Reason: … (pass --compile-trainer explicitly to make
   this a hard error instead.)
```

The rule, and it generalises to any future default: **a default yields to the config the user typed
and announces it; an explicit flag refuses.** Pinned by `src/main/compile_defaults_test.py`
(`test_auto_yields_to_async_rollout_instead_of_refusing_to_launch`,
`test_auto_yields_to_a_rollout_that_does_not_divide_the_batch`, and
`test_an_explicit_flag_never_reaches_the_auto_path`, which holds the refusal in place).

**Measured** (2026-08-14, v76 `gen3_ctx_dedup_v1`, RTX 3080 Ti, the real
`MaskablePPO -> ActorCriticPolicy._build()` path, gen-9's own `cli_args`: batch 4096, PopArt on;
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

**We compile the EXTRACTOR, and the second row is
why**: the two scopes measure the same to within 0.004x — the mlp_extractor, the pointer head and
the value head contribute nothing — so the whole-policy scope buys nothing for strictly more graph
(and more surface for SB3's distribution objects and the mask path to break on). Same win, smaller
blast radius. Also confirmed: rollout 2048x48 / batch 4096 = **exactly 24 minibatches, no
remainder**, so one graph and no per-epoch recompile.

**FAIL-LOUD BY DESIGN, and deliberately asymmetric with `--compile-opponents`.** The opponent path
warns and falls back to eager (`--compile-opponents-strict` opts into raising) because it prints a
`[CompileExtractor]` line either way. Here there is nothing to notice: a silent fallback trains
perfectly correctly and just produces ~38% fewer steps/hour forever. So every failure is fatal
(`CompileTrainerError` -> `TrainExitCode.FATAL_CONFIG`, so the launcher gives up instead of
restart-looping), and the flag has no `strict` variant because there is nothing to opt into. Four
refusals, each guarding an otherwise-invisible outcome:

| refusal | why |
|---|---|
| `--device cpu` | The CPU BACKWARD does not lower — `CppTile2DKernel.store` asserts on the `atomic_add` mode. Pinned by `extractor_compiles_test::test_cpu_backward_still_does_not_compile`, which builds at `belief_grad_mode="shaping"` ON PURPOSE: under `label_only` (production since gen-11) those gather-backwards do not exist and the compile succeeds, so an unpinned test would have gone green while testing nothing. Costs nothing in practice — the compiled backward we run is CUDA, where Triton emits `tl.atomic_add` |
| compile raised | bisect the op — the whole "torch cannot compile our model" story was ONE op (see `src/agents/model/CLAUDE.md`, the `species_posterior` precedent) |
| compiled is not faster (< 1.05x) | the graph fragmented or the backend fell back per-frame; the measured figure is ~1.75x, so parity is a defect |
| compiled disagrees with eager on the committed REAL-obs rows — features, masked legal log-probs, V, or the train graph's gradient (the bars and the TF32 rule: "The startup parity gate" below) | a faster wrong model is not a win |

Every rejection **uninstalls** the compiled callable before raising, so the process never keeps
running something it just declared unacceptable.

### 🚨 The single-graph CUDA miscompile, and the trunk split (`gen3_inductor_trunk_split_v1`, 2026-09-28)

**Measured** (torch 2.5.1+cu121, RTX 3080 Ti, fp32 `highest`, `ai_v14_01_base` weights on 3,840 REAL
rows from its own `eval_traces`): with the whole extractor as ONE CUDA Inductor graph, the compiled
learner disagreed with eager by up to **7.65 on pi_features** (scale ~16). At the decision level,
**argmax agreement was 70.9%**, masked-policy TV had median 0.12 and p99 0.63, win-prob |dV| had
median 0.024 and max 0.33, and in the TRAIN graph the extractor gradient's **cosine to eager was
0.778**. dynamo `backend="eager"` and `"aot_eager"` matched exactly, and the CPU Inductor compile
(`--compile-opponents`: eval workers, sentinels, pool, stable and exploiter opponents, so every eval
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

### Lane K1 — torch 2.8 (`gen3ai_torch28`), the split OFF there (2026-09-28)

**Two envs, one code base.** `gen3ai_stable` stays torch 2.5.1+cu121 (NEVER mutated — live and pinned
runs use it); `gen3ai_torch28` is its clone with torch **2.8.0+cu126**, torchvision 0.23.0+cu126,
torchaudio 2.8.0+cu126, triton 3.4.0, sympy 1.14.0 and the cu126 `nvidia-*` wheels
(`environment_torch28.yml` — a SEPARATE file because `scripts/bootstrap.sh` re-applies
`environment.yml` to `gen3ai_stable` whenever its hash changes). The split is keyed on the torch
version: `team_transformer._SPLIT_NOT_NEEDED_ON = {"2.8.0+cu126"}` ⇒ `_CUDA_TRUNK_SPLIT` is OFF on
2.8 and ON on every other torch. A new run moves to 2.8 by its interpreter (`$GEN3AI_PYTHON=
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3`), switched at a run boundary.

**Measured, split OFF on 2.8** (RTX 3080 Ti beside a live training run, `ai_v14_06_lbat_ctrl_fix`
`final_model.zip`):

| check | fp32 (`highest`) | TF32 (`high`) |
|---|---|---|
| the real startup gate (64 fixture rows) — eval readout | PASS: features 2.4e-5, legal log-prob 9.5e-6, V 4.2e-7 | PASS (rule: e_comp ≤ 4·e_eager + 1e-4 vs an fp32 reference): features 5.1e-2 ≤ 1.7e-1, legal log-prob 2.1e-2 ≤ 7.3e-2, V 2.0e-3 ≤ 5.2e-3; the same graph re-run at fp32 passes the strict bars |
| the real startup gate — TRAIN graph gradient cosine | 1.000000 | PASS: 1−cos vs fp32 compiled 6.1e-5 ≤ 4 × eager 4.6e-5 + 1e-4 (cosine 0.99994) |
| 3,840 rows of its own `eval_traces` (steps 78M/80M/82M) — argmax agreement | 1.0000 | 0.9992 (2.5.1 WITH the split: 0.9992) |
| same — masked-policy TV p99 / max | 1.6e-6 / 2.7e-6 | 2.5e-3 / 1.8e-2 (2.5.1 split: 2.5e-3 / 1.8e-2) |
| same — win-prob \|dV\| max | 1.4e-6 | 3.5e-3 (2.5.1 split: 3.5e-3) |
| train graph, 64 rows, loss = mean pi² + mean vf² — gradient cosine | 1.000000 (rel 7.3e-7) | 0.999977 (rel 6.7e-3) |

At TF32 compiled-vs-eager differs by TF32's own rounding, identically on 2.8-unsplit and on
2.5.1-split — the gate's TF32 rule (vs an fp32 reference, plus the same graph re-run at fp32) is
what resolves a real defect there. `compile_trainer_test::test_REVERTING_the_trunk_split_FAILS_the_real_obs_gate`
now asserts, per torch: 2.5.1 ⇒ split ON and unsplit FAILS; 2.8 ⇒ split OFF and unsplit PASSES.

**Graph count** (`torch._dynamo.explain`, the extractor's `forward`, 32 fixture rows, CPU): **1 graph /
0 breaks on both torches with the ObservationDebugger dropped** — which the compiled path always does
— and 3 graphs / 2 breaks with it kept (the debugger's rate-limited logger calls `time.time()`,
`utils/logging/rate_limiter.py:14`). On CUDA the 2.5.1 split adds one; on 2.8 the CUDA explain is also **1 graph / 0 breaks** (6,870 ops) — the unsplit graph, K8's post-K1 target for the extractor.

**Speed A/B (2026-09-29, idle RTX 3080 Ti, the same saved rollout buffer, `learner_benchmark`,
K=5):** 2.8-unsplit is 1–2% FASTER than 2.5.1-split — baseline 57.21 s vs 58.45 s per update,
diag_skipped 50.47 vs 51.27, TF32 53.42 vs 54.46, epochs_half 33.08 vs 32.97 — almost all of it in the
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
update), as does the distill grad-projection. On torch 2.8 that failed twice, for two reasons:

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
graph first (prewarm order, K2 cadence, the distill projection's loop), and it would not touch the
cache hazard. Donation OFF runs the same kernels on the same inputs — `compile_control_test` pins the
gradient bit-for-bit on vs off — and costs nothing measurable: production extractor fwd+bwd on 2.8,
batch 64 ~10.1 ms and batch 2048 ~74.8 ms either way, peak memory +0.3%
(`designs/research_state/measurements/m5_k1/`).

**Tests.** `compile_control_test` (routine, both torches): install pins the row and uninstall
restores it; a donating backward REFUSES a retain-graph probe and the pinned row accepts it; the pin
changes no gradient; on 2.8, the cache key carries the tag and not donation.
`agents/training/compiled_train_probes_test.py` (`slow`; CPU, plus CUDA under
`GEN3AI_TEST_ALLOW_GPU=1`): the production policy surface, the
same signatures compiled first under the donating default, then the real pipeline (compile, reset,
production prewarm, attach) and ONE full `train()` with every first-update probe, plus a
repeated-`retain_graph`-backward check (bitwise on CPU; ≤ 1e-5 relative on CUDA, whose scatter-add
atomics give 2.1e-7). Revert `donated_buffer` ⇒ the probe raises (CPU and CUDA, measured). ⚠️ Reverting
ONLY the tag passes this toy (2.2e-7, measured) — that half is proven by the real-graph A/B and pinned
by the cache-key contract test, not by this test.

**Open (K1 finding, not fixed here):** on torch 2.8.0+cu126 a **batch-1 CUDA eval/no-grad** graph of
the production extractor fails to LOWER — Triton `CompilationError` (`'constexpr_type' object has no
attribute 'is_block'` on a fully-constant `tl.broadcast_to` index); batch 2 and 4 compile, and 2.5.1
compiles batch 1. It is reached by `--critic shaped`'s batch-1 prewarm signature and by the trainer's
in-process FINAL EVALUATION (batch 1 on the compiled forward, after the lock is released, per the
sentinel table above) — **UNVERIFIED:** that a 2.8 `--compile-trainer` run dies at its end (inferred
from the batch-1 repro, not observed on a real run).

### The startup parity gate (`gen3_compile_parity_real_obs_v1` + `gen3_tf32_parity_gate_v1`)

`compile_trainer_extractor` validates on **REAL observation rows, never zeros**. The rows are a
committed fixture, `src/agents/model/compile_parity_obs.npz`: 64 decision rows from 8 reproducible
in-process bridge battles, made by `python -m agents.model.compile_parity_fixture --write`. It can be
regenerated on any box for any layout. A missing fixture, or one of the wrong width, **REFUSES**
(`FATAL_CONFIG`) rather than falling back to zeros, and
`compile_trainer_test::test_the_committed_real_obs_fixture_matches_the_live_layout` fails the
routine gate the moment the layout changes. It checks four things, eager vs compiled:

| quantity | fp32 (`highest`) bar | healthy (after the split) | the miscompile |
|---|---|---|---|
| (pi ‖ vf) features, no-grad | 1e-4 (unchanged) | 2.4e-05 | 10.7 |
| MASKED legal log-probs (what the rollout samples and PPO's ratio reads) | 1e-3 | 1.2e-05 | legal logits off by up to 7.39 (3,840 rows) |
| V (the win-prob sigmoid under `--critic winprob`) | 1e-4 | 5.4e-07 | 0.33 (3,840 rows) |
| TRAIN graph: fwd features + extractor-gradient cosine on the gate's own loss | 1e-4 · cos ≥ 0.9999 | 1.9e-05 · 1.000000 | 7.65 · 0.778 |

The healthy column is the gate's own line on `ai_v14_01_base/final_model.zip`.
`compile_trainer_test::test_REVERTING_the_trunk_split_FAILS_the_real_obs_gate` pins that turning
the split off makes the gate refuse. Measured through the real gate: with the split OFF it FAILS at
10.7.

**At reduced precision (`--matmul-precision high`, TF32) two checks run.**

1. **The TF32 graph, against an fp32 EAGER REFERENCE.** The reference is the same weights and rows
   with matmul precision temporarily `highest`, restored even on an exception. For each quantity:
   `e_eager = max|eager_tf32 − ref|`, `e_comp = max|compiled_tf32 − ref|`, and PASS iff
   `e_comp ≤ 4·e_eager + 1e-4`. The gradient check is `1−cos(comp, ref) ≤ 4·(1−cos(eager, ref)) + 1e-4`.
   EPS equals the fp32 bar, so a quantity TF32 does not touch is held to the fp32 bar.
2. **The SAME compiled callable at `highest`, held to the strict fp32 bars against the fp32
   reference.** dynamo guards on the TF32 flag, so this is a second graph of the same trace. It
   exists because the TF32 rule alone cannot resolve a defect smaller than ~4× TF32's own rounding.
   **Measured:** a DROPPED projection bias and a projection weight scaled by 1.001 both PASS rule 1,
   and both FAIL rule 2 (3.1e-2 and 2.9e-2 against a 1e-4 bar).

**Why K = 4.** The measured `e_comp/e_eager` ratios at TF32 on the real weights:

| rows | features | legal log-prob | V | train features | grad 1−cos |
|---|---|---|---|---|---|
| fixture | 1.33 | 0.87 | 1.27 | 1.14 | 1.24 |
| zero obs | ≤ 0.92 | | | | |

On a random-init extractor on zero obs the ratio reached ≤ 2.31. K = 4 is about 1.7× above the worst healthy
ratio. The miscompile read 538, and a 1%-scaled projection weight read 14.7. **The resolution limit
is honest.** A TF32-ONLY kernel defect smaller than ~4× TF32 rounding passes; for this extractor that
is about 8e-2 on features (scale ~16), 6e-2 on legal log-probs and 3e-3 on V. A precision-independent
defect of any size is caught by rule 2.

**Measured TF32 rounding on the real extractor** (fixture rows, `ai_v14_01_base`): `e_eager` is
2.0e-2 on features, 1.5e-2 on legal log-probs and 7.2e-4 on V, and gradient 1−cos is 3.2e-5.
On zero obs it is 9.4e-3. The failed T32 launch's 7.62e-03 was this rounding, not a bug; the real
bug (the miscompile above) was invisible to it.

**Cost.** The gate adds a readout and a train step per arm. At TF32 it adds a second (fp32) graph of
the same trace. dynamo's `cache_size_limit` (8) holds with room to spare: measured 6 entries per
code object after the TF32 gate plus alternating rollout (48, no-grad) and train (512, grad) shapes,
0 cache-limit hits. At `highest` it is 4 entries.

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
prevent. Two configs get there, both decidable before training starts, both now fatal
(`check_shape_stability`, pure and unit-tested):

| refused | why |
|---|---|
| `--async-rollout` | the async collector forwards whichever envs are READY, so the rollout batch VARIES every step — an unbounded shape set, guaranteed to exhaust the cache. The error prints both measured numbers (`--async-rollout` +14% at n_envs=64 vs `--compile-trainer` +62%) so the choice is informed, not blind |
| `n_steps*n_envs` not divisible by `batch_size` | the remainder minibatch is a THIRD shape, replayed every epoch, for no benefit. The error names a concrete divisor to use instead rather than leaving you to do arithmetic |

Production is safe by arithmetic — 2048x48 = 98304 = 24 x 4096 exactly, so exactly two shapes — but
that was luck of the config until these guards existed.

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

*Why zeros and not `torch.rand`.* A random float vector is **not a valid observation** — the
ObservationDebugger rejects it outright — so it can drive the forward down branches no real battle
reaches. All-zero is the canonical "nothing known" state (every categorical id 0, every flag clear),
structurally legal, and it is what `snapshot._zero_obs` has always used on the opponent path. The
trainer path briefly diverged to `rand` for no reason and that is what disguised the OOM as a CUDA
config error.

*Where per-shape correctness IS checked.* `torch.compile` compiles lazily PER SHAPE, so the graphs
production trains with (batch `n_envs` for rollout, batch `batch_size` for train) are never the one
the startup check compiles. That gap is closed by
`compile_trainer_test::test_every_production_shape_agrees_with_eager`, which asserts compiled ==
eager at each shape on a free GPU where memory is not contended. Measured once against the live
gen-10 config on REAL observations off the rust bridge: **batch 48 -> 9.5e-07, batch 64 -> 7.2e-07,
batch 4096 -> 3.6e-06**, against a value scale of 2.111 — float32 rounding, not a wrong kernel.

**⚠️ It DROPS the ObservationDebugger, and that is a production-visible trade.** Dynamo cannot trace
the debugger's numpy asserts at all (it dies building a guard over a numpy bool), so this is
compile-or-debugger, not both. The debugger attaches at `log_level >= PERIODIC` — i.e. it is ON in
production — so this flag costs you the per-forward obs-integrity check for that run.

**With the default ON, that trade is now made by EVERY plain cuda run, with nobody having typed a
flag — which makes the announcement more load-bearing, not less.** It is said twice: once at
startup, when the auto default resolves to on (`⚡ --compile-trainer ON by default (device=cuda)`,
naming the debugger and `--no-compile-trainer`), and once from `compile_trainer_extractor` when the
debugger is actually dropped. Neither line is conditional on a launcher being attached. The opt-out
is the only way to keep the debugger.

**Mechanics.** Patches the BOUND `fe.forward`, never the module: `torch.compile(module)` returns an
`OptimizedModule` and prefixes every `state_dict` key with `_orig_mod.`, which would land in every
checkpoint of the run and make them unloadable by anything else. It runs immediately BEFORE
`_run_roundtrip_test`, which turns that existing save -> reload -> forward gate into a free check on
exactly that hazard. **Runtime perf knob**: never versioned, never in `check_compatible`, NOT
inherited on resume — but with the AUTO default that means a flagless cuda resume gets it ON, so it
is `--no-compile-trainer` you re-pass each launch, not the flag.

Tests: `agents/model/compile_trainer_test.py` (the verdicts are pure functions so every refusal is
testable without a GPU — a contract that needs a free card is a contract that gets checked rarely;
plus a CUDA test that the `state_dict` keys and a save/reload survive) and the compile itself in
`agents/model/extractor_compiles_test.py`.

### The compile sentinel (`gen3_compile_sentinel_v1`, 2026-09-28)

The owner's question: *"Is there no way to force it to raise or reject?"* — yes, on torch 2.5.1,
behind ONE adapter: **`src/agents/model/compile_control.py` is the only runtime module that touches
`torch._dynamo`** (the one exception is `team_transformer`'s in-graph `graph_break()`, which is model
code dynamo traces). It is also the only module that sets `torch._functorch.config` — the
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
| 1 `gate()` | `compile_trainer_extractor` | the startup parity gate compiles FREELY; the cache-limit detector already listens (a hit here is fatal too) |
| 2 `reset()` | `_arm_compile_sentinel`, after `_apply_grad_checkpointing` | `torch._dynamo.reset()` — drops every graph the gate compiled |
| 3 `prewarm(calls)` | same | runs every production signature NOW (`compile_trainer.production_prewarm_calls`), RNG-neutral (`fork_rng`), gradients zeroed |
| 4 `lock(where)` | the end of the first `train()` (after one real rollout + update) | `error_on_recompile = True` + a compile-START callback counting every frame compile. `🧊 [COMPILE LOCK] …` states the per-code-object entry count and the headroom to the limit |
| — `release(why)` | when `learn()` returns or raises | the lock's scope is TRAINING: the trainer's FINAL EVALUATION runs in-process on the same compiled forward at batch 1 / no-grad — new signatures by design. MEASURED on the real trainer (the end-to-end smoke below): a still-locked sentinel broke it with `RecompileError`. Prints `🧊 [COMPILE LOCK] released …` with the post-lock compile count (must be 0) |
| 5 `stats()` | every update | TB `compile/graphs_total`, `compile/recompiles_after_lock` (**must stay 0**), `compile/cache_limit_hits`, `compile/max_cache_entries_per_code`, `compile/locked`, `compile/regression_flag`, `compile/train_ms_vs_lock_baseline` |

The checks run at every rollout end and every update end (`CompileControl.attach` wraps the model's
`collect_rollouts`, `train` and `learn` as INSTANCE attributes — all three, plus `_compile_control`,
are in `_excluded_save_params`, so no checkpoint carries a live sentinel). A `RecompileError` raised at a call site, a cache-limit hit, or any
compile that STARTS after the lock is `CompileSentinelError` → **`[CompileSentinel] FATAL …` and
`os._exit(FATAL_CONFIG)`** — an exception inside `learn()` would reach `model_build`'s generic
`except` and become a restartable CRASH, and a restart would replay it. The launcher also matches
`[CompileSentinel] FATAL` / `[CompileTrainer] FATAL` in `_FATAL_CONFIG_SIGNATURES`. There is no
checkpoint save on this exit (the last periodic checkpoint stands); the weights are not wrong, the
throughput is.

**Why the reset (phase 2) — the gate's graphs spent production's slots.** The cache is keyed per
code object (`Gen3FeaturesExtractor.forward.__code__`, `TeamTransformer.forward.__code__`, the
trunk split's resume frames …), shared by every instance and every `torch.compile` wrapper of that
code — so a separate compiled callable for the gate would NOT have isolated it in 2.5.1
(`compile_control_test::test_contract_cache_is_per_code_object_shared_across_compiled_wrappers`).
The gate compiles train/grad + train/no-grad at batch 64, and under TF32 the same pair again at
`highest`; production rolls out in EVAL mode at `n_envs`, so **none of the gate's entries is a
production signature** (entries at lock = gate's + production's, exactly additive below). MEASURED
2026-09-28 on the real `ai_v14_01_base/final_model.zip` policy (RTX 3080 Ti, torch 2.5.1+cu121,
rollout batch 48, train batch 512 as the dynamic-train proxy — the train graph is dynamic in batch
either way, so 2048 lands on the same entry; `~/.claude/jobs/compile_sentinel_2026-09-28/`):

| precision | gate entries / code obj | at lock WITHOUT reset | at lock WITH reset (ships) |
|---|---|---|---|
| fp32 `highest` | 2 | max **5** (headroom 3), 13 code objects, 39 entries | max **3** (headroom 5), 9 code objects, 23 entries |
| TF32 `high` | 4 | max **7** (headroom **1**), 13 code objects, 57 entries | max **3** (headroom 5), 9 code objects, 23 entries |

TF32 without the reset sat ONE signature from the silent fallback (a `--critic shaped` batch-1
truncation value alone would have been the eighth) — the risk to the TF32 arm. **Startup cost of the
reset: ~0.** At TF32, run concurrently under identical conditions, reset+prewarm took 220.8 s vs
223.0 s for the prewarm alone on top of the gate's cache: the production graphs are compiled
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
| rollout `policy(obs)`, end-of-rollout `predict_values`, `WinProbLabelCallback` bootstrap, frozen-φ / PBRS | eval · no-grad · `n_envs` (PBRS: 2048-chunks) | prewarm (rollout) + iteration 1 (PBRS runs every rollout) |
| `evaluate_actions` | train · grad · `batch_size` | prewarm |
| `rank_probe` (first minibatch of every update) | train · no-grad · `batch_size` | prewarm |
| capacity half-batch cosine (every 50 minibatches), td-aux, distill-anchor fallback | train · grad · other sizes ≥ 2 | prewarm at `batch_size // 2` makes the train graph dynamic |
| truncated-episode `predict_values` under `--critic shaped` | eval · no-grad · **1** | prewarm (only when the critic is not `winprob`, which relabels those ends as terminal) |
| search-teacher / OPD (`ppo.py`), fork-arm `_score_pool`, `fork_driver._score`, distill grad-projection | a different KEY SET and/or a variable batch that may be 1 | `compile_trainer.eager_extractor(fe)` — the EAGER forward for the block (same params, same autograd) |
| cf terms, capacity velocity, win-prob episode-start read | — | already eager (`type(fe).forward(fe, …)`) |
| eval, snapshot ladder, search-teacher workers, the opponents | — | other processes (fresh dynamo per process) |
| `--debug` (DummyVecEnv) + `--compile-opponents` | opponents compile IN the learner process, on the learner's code objects, after the lock | **refused at startup** (`FATAL_CONFIG`): pass `--no-compile-opponents` |
| the FINAL evaluation after `learn()` (in-process, the live compiled model) | eval · no-grad · 1, a smaller obs key set | runs AFTER `release()` — recompiles freely, as before |

A launcher restart is a fresh process (fresh cache, a new gate, a new lock).

**END-TO-END on the real trainer** (2026-09-28, the T32b argv at TF32 made tiny — `n_envs` 2, a fork
of `ai_v14_01_base` into `~/gen3ai_archive/compile_sentinel_smoke/`, 3 iterations): gate PASS →
`reset` dropped the gate's 36 entries → prewarm (4 signatures, max 3 entries per code object, 229 s
incl. the reset) → `🧊 [COMPILE LOCK] after the first rollout + update: max 4 cache entries per code
object (limit 8, headroom 4)` — iteration 1 added one signature the prewarm did not (UNVERIFIED
which; the lock-after-iteration-1 design exists for exactly that) → iterations 2-3 clean (`train_ms`
12.8 s → 2.8 s), checkpoint saves after the lock fine, `Training complete`. It then exposed the
final-eval defect `release()` fixes; **the fix is unit-tested, not re-run on the card** (the card
was handed back to a live benchmark). Nothing toggles TF32 /
matmul precision / autocast after startup.

**THE BEHAVIOURAL BACKSTOP, independent of dynamo:** the baseline is the median `train/train_ms` of
the first 5 updates AFTER the lock update (which may carry iteration 1's compiles), then FROZEN; `train_ms > 1.4x` it for 3 consecutive updates prints
`⚠️ [COMPILE REGRESSION?] …` once and sets `compile/regression_flag` = 1. WARN, never fatal —
contention (an eval burst, a peer job) has the same signature.

**THE TORCH GUARDRAILS.** (1) An exact version table (`_SUPPORTED`, `2.5.1+cu121` → the
`error_on_recompile` mode); an unknown torch REFUSES. (2) **A source-hash drift tripwire** (the SB3
`instrumented_ppo._verify_upstream_unchanged` pattern): `_SOURCE_HASHES` = {torch version →
{qualname → SHA256 of `inspect.getsource`}} over every internal the adapter rests on
(`cache_size.compute_cache_size` / `is_recompilation` / `exceeds_cache_size_limit`,
`guards.get_and_maybe_log_recompilation_reason`, `convert_frame._compile` (emits the warning),
`ConvertFrame.__call__` (the soft-fail swallow), `ConvertFrameAssert.__call__` + `Tracker` +
`eval_frame._debug_get_cache_entry_list` (the per-code read), `callback.CompilationCallbackHandler`,
`OutputGraph.compile_and_call_fx_graph` (the graph counter), `torch._dynamo.reset`,
`torch.compiler.reset`). Verified at the adapter's import — every `--compile-trainer` process — and
in the routine tier; a mismatch raises naming each drifted qualname with both hashes. Re-record with
`python -m agents.model.compile_control --record` AFTER re-reading the functions and re-running
(3) the behavioural CONTRACT tests in `compile_control_test.py` (the warning's text and logger, the
raw `RecompileError`, the start callback, the counter / entry list / reset, the per-code-object
cache). Hash = the code changed; contract = the behaviour changed; both are kept.

**Per torch version** (`_SUPPORTED`, `_SOURCE_HASHES` — one row each; the contract tests run on BOTH
envs and assert each version's own semantics): **2.5.1+cu121 → `error_on_recompile`** (13 hashed
internals); **2.8.0+cu126 → `stance`** (17): the lock is `torch.compiler.set_stance(
"fail_on_recompile")`, which raises at the call site on ANY cache miss of a compiled callable — a
recompile AND the first call of a never-compiled one — compiling nothing; `release`/`uninstall` set
the stance back to `"default"`. **What moved in 2.8** (re-read at the source): `exceeds_cache_size_limit`
→ `exceeds_recompile_limit`, and `config.cache_size_limit` is now an ALIAS of `config.recompile_limit`
(the warning reads `hit config.recompile_limit (8)`; the soft failure is `RecompileLimitExceeded`,
still swallowed into eager); `get_and_maybe_log_recompilation_reason` → `..._reasons`; the
compile-START callback takes a `CallbackArgs`, fires for LAZY_BACKWARD / TRITON_AUTOTUNING /
CUDAGRAPH_RECORDING too, and now runs AFTER the recompile check — so it no longer sees a REJECTED
recompile, and a caller's `except Exception` could have hidden one. **`CompileControl.wrap_compiled`**
closes that on both versions: the learner forward is installed through it and records any lock
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
`FATAL_CONFIG`; `eager_extractor`; the version refusal and the hash tripwire; the contract tests;
and `test_the_production_extractor_locks_with_headroom_at_fp32_and_tf32` (CUDA, `slow`; skips while
a trainer holds the card).

### Every non-training model can use it

`maybe_compile_extractor` is safe to apply to ANY frozen model, because the wrapper routes
**grad-enabled calls to eager**. That matters: the compiled artifact is inference-only (under
`requires_grad` dynamo hands the graph to AOTAutograd, whose CPU backward codegen fails on this
model's scatter/`index_add` — the documented reason the June `--compile-damage-op` integration was
inference-only), and the prober backprops through this same extractor for gradient saliency.

| consumer | what is compiled | gate |
|---|---|---|
| training env workers | pool / stable / exploiter opponents | `--compile-opponents` (+ forkserver preload) |
| `eval_worker` | **the trainee** (plays every eval game) + sentinel + fixed opponents | `compile_extractor` cfg key, threaded from both eval callbacks |
| `search_teacher_persistent_worker` | trainee (per re-freeze) + opponent (per iteration) | `compile_extractor` cfg key |
| `snapshot_ladder` | both frozen ladder players | **default ON** — offline tool, nothing races it |
| prober (`session._load`) | the no-grad replay / rollout models | `--compile` (off by default) |
| `play.py` | nothing — the websocket/LADDER client loads a plain `MaskablePPO` and stays eager: one process, one battle at a time, and eager already measures **18 ms/decision** against a 150 s ladder timer, so a compile would buy latency nobody is waiting on | n/a |

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
`src/main/eval_worker_compile_test.py` now pins the wiring in the fast unit suite.

**Validation is paid once per process** (`_COMPILE_VALIDATED`), and the reuse path STILL LOGS
(`ON (reused this process's validated compile)`) — it used to return silently, which made the eval
opponent's compile look like it had never run and cost a round of doubt. A success you cannot see in
the log is a success you will not trust. The eager-vs-compiled timing answers
"does this extractor's code object compile to something faster?", and `torch.compile` keys on exactly
that code object — so a second model in the same process cannot get a different answer. Consumers that
load models in a LOOP (the search-teacher worker rebuilds its opponent every iteration) would
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
