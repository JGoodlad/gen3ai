---
name: project_throughput_compile
description: "Training throughput profile + the whole-extractor torch.compile win (--compile-extractor, 6.53x B=1 / +31% fps, BUILT not shipped); suppress_errors killed; forkserver-preload is the next lever"
metadata: 
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
  modified: 2026-08-03T23:28:45.777Z
---

> **Archived 2026-09-08** — 33 KB throughput archaeology; the operative facts (--compile-opponents / --compile-trainer ON by default) are in root CLAUDE.md and the training runbook. Preserved verbatim; nothing below is current.

**⚠️ UPDATE 2026-08-17 — THREE corrections; read these before acting on anything below.**

1. **The flag was RENAMED (2026-08-14).** `--compile-extractor` no longer exists. It split into
   **`--compile-opponents`** (CPU, frozen opponents in env workers — the +33.3% below) and
   **`--compile-trainer`** (CUDA, the learner's fwd+bwd, measured 1.75x ≈ +62% end-to-end). Every
   `--compile-extractor` mention below is the OLD spelling of the first one.
2. **The recurring promotion cost is SMALL — ~2.7%, and the flag is net +40%.** (An earlier version
   of this note claimed ~31% and a −0.3% wash; that was **WRONG and is retracted** — it generalised a
   one-time transition.) Measured n=2 on gen-14: iteration 22 cost **+1095 s with 48 *timed*
   compiles**, but that is where **self-play first activates** — the pool is seeded from empty so all
   48 workers simultaneously load a 41 MB checkpoint and pay their process's FIRST compile; it
   happens once per run. Iteration 42, the true steady state, cost **+77 s with 27 compiles, all on
   the `"reused this process's validated compile"` path** ≈ 2.7% of wall-clock, ~16 min per 25M run.
   **The caches all work** (Inductor cache HIT at promotion: 13 files written vs 6600+ at startup;
   `SnapshotPool._model_cache`; `_COMPILE_VALIDATED`). Lever T1 is **KILLED**.
   **Method scar:** the log distinguishes the expensive path (`ON — 14.21 -> 1.84 ms (7.7x)`) from
   the cheap one (`ON (reused …)`). Counting `[CompileExtractor]` lines conflates them — read the
   path, not the count. And **eval cycles are non-blocking**: gen-13 ran an 1865 s `[SELFPLAY EVAL]`
   inside a 395 s iteration, so a long eval beside a slow iteration proves nothing.
3. **The forkserver-preload hang is FIXED, not live.** `gen3_forkserver_preload_v1` (2026-08-16)
   root-caused it (fork() from a multi-threaded process; poke-env's loop thread started at extractor
   import), fixed it via lazy poke_env inits, and made it **fail-loud** — a preload that cannot prove
   the forkserver single-threaded RAISES rather than forking. `--compile-opponents-preload` is a
   normal opt-in flag now. **But it does NOT address (2)** — it runs before the workers exist, so it
   cannot anticipate a snapshot promoted 2M steps later.


**⚠️ UPDATE 2026-08-03 — `suppress_errors` DELETED; production now compiles WHOLE at 6.53×.** The
`--threat-unrevealed-outgoing` inductor crash was **ONE OP**, not an arch property: the softmax over
species logits (`BeliefHead.species_posterior`, the v36 expected-latent read) lowers to a
`[B,6,n_species]` numerator + `[B,6,1]` denominator the CPU scheduler asserts on while fusing
(`AssertionError: buf307`). Respelling it `log_softmax(...).exp()` compiles clean. `.contiguous()`,
`.clone()`, a 2-D reshape and hand-rolled `exp/sum` **ALL still fail** — the spelling is load-bearing.
Result: literal production arch **6.371 → 0.976 ms = 6.53×** (was 3.6× partial), 1 graph / 0 breaks,
max|Δ| 5.07e-07, with `torch._dynamo.config.suppress_errors` **removed** (it had been globally
silencing every backend failure in the process, forever). LESSON: bisect to the op before reaching for
a global suppression flag. Repro `tmp/inductor_crash_repro.py`; variants `tmp/softmax_variant_probe.py`;
guard `species_posterior_compiles_test.py` (`GEN3AI_COMPILE_TESTS=1`, verified to FAIL on the old spelling).
Helper also hardened: `hide_cuda` is now an EXPLICIT caller declaration (was inferred from
`torch.cuda.is_initialized()` — would have blinded the learner's GPU on the first main-process call),
late-compile failures degrade ONE model via `_eager_fallback_on_error` instead of a global flag, the
debugger drop goes through `disable_observation_debugger()` instead of poking `fe._debugger`, and the
cache-dir env mutation is no longer an import side effect. NOTE the +31% fps A/B PREDATES this fix, so
it was measured on the 3.6× partial graph — the end-to-end number should now be at least that good.

**✅ UPDATE 2026-08-16 — THE FORKSERVER PRELOAD IS REVIVED (`gen3_forkserver_preload_v1`,
`--compile-opponents-preload`).** The hang below was fixed at the ROOT, not by the planned ~12-file
model-layer refactor: **`poke_env/__init__.py`, `player/__init__.py` and `battle/__init__.py` are
now LAZY (PEP 562)** (public surface unchanged; the loop thread starts only when a player/client
module is imported), so the extractor import is SINGLE-THREADED —
`extractor_import_is_fork_safe()` now returns None and `compile_prewarm_test` pins the INVERSE of
the old hazard. The laziness also dissolved an order-dependent `battle ↔ player.battle_order`
circular import the eager inits masked. `agents/model/compile_preload.py` + the flag: compile ONCE
in the forkserver, workers inherit (~0.12 s each); FAIL-LOUD — the preload RAISES if any non-main
thread survives its compile, so the silent 2-of-48 wedge became a loud env-construction crash.
Proven live on a real 4-worker SubprocVecEnv run (compiled 41 s once, all workers forked, training
completed). Still NOT a throughput lever (~50 s per 3 h restart); the reason it exists is the
architecture now permits it safely. HISTORY below kept verbatim:

**🚨 THE FORKSERVER PRELOAD HUNG TRAINING — REMOVED (2026-08-03, shipped 59aa1bf, fixed 1a41503).**
I shipped `install_forkserver_preload` on the strength of a 6-worker probe; a real **48-env run
WEDGED**: forkserver logged a successful compile, trainer forked **2 workers instead of 48**, hung
forever — parent in `unix_stream_data_wait` on the forkserver socket, workers in `anon_pipe_read`,
box at **0.2 load**, no error/traceback (textbook [[feedback_idle_cpu_means_deadlock]]). ROOT CAUSE:
`fork()` copies MUTEXES but not the threads holding them ⇒ safe only from a SINGLE-THREADED process.
The forkserver has **two** thread sources: (1) Inductor's `compile_threads`=16 codegen pool survives
the compile (fixable: `shutdown_compile_workers()`), and (2) **poke-env's global asyncio loop thread
starts at IMPORT**, and `agents.model.features_extractor` transitively imports **37 poke_env
modules** — so merely importing the extractor makes the process multi-threaded, unfixable from
outside. REPLACED by `agents/model/compile_prewarm.py` (warm the shared on-disk Inductor cache in the
parent: 59.6s → 30.1s for 16 workers). `extractor_import_is_fork_safe()` + `compile_prewarm_test.py`
pin the hazard and FAIL-LOUD-WITH-INSTRUCTIONS if the extractor ever stops importing poke-env (the
preload is 250× cheaper, so it's worth rediscovering). **LESSON: a probe with N=6 does not validate a
fork-topology change; the failure mode of fork hazards is a HANG at scale, not an error.**

**📊 2×2 MATRIX {node,rust} × {compile off,on}, n_envs=48, literal arch — RE-RUN on the seed-fixed
tree at `a5157cc` (2026-08-03, `tmp/bridge_x_compile_ab.sh`).** Marginal fps, 4 samples/arm:

| | compile off | compile on | compile effect |
|---|---|---|---|
| **node** | 413.3 | 558.5 | **+35.1%** |
| **rust** | 417.0 | 599.4 | **+43.7%** |
| rust effect | **+0.9%** | **+7.3%** | |

Combined node_off → rust_on = **+45.0%**. Interaction 1.064 (the levers COMPOSE mildly — removing the
opponent-forward bottleneck makes transport matter more, as Amdahl predicts). 48/48 workers compiled
in both `_on` arms, 0 reverts.
**VERDICT: rust is NOT a throughput lever at production scale — +0.9% alone, +7.3% with compile.**
The `bridge_impl_throughput_benchmark.py` 1.41× is TRANSPORT-ONLY and does not survive the
`SubprocVecEnv` barrier, exactly like the node bridge's 2.1× transport → ~5% training. Rust's case is
OPERATIONAL (25× smaller child, ~10 GB saved at 48 envs) + it now supports `__RECON__`/`resumeReseed`.
The voided (fixed-seed) run had rust at +3.5%/+8.6%, i.e. the bug FLATTERED rust by inflating
`rust_off` 427.4 → the true 417.0; node arms were identical across both runs (413.1 vs 413.3), which
is the control that says the re-run is sound.
**The ep_len CONFOUND RESOLVED CLEANLY, confirming the diagnosis:** void run node 48.7/50.7 vs rust
56.9/56.2 (a systematic ~13% gap); fixed run node 51.7/54.9 vs rust 53.6/54.9 — the between-impl gap
is now smaller than the within-impl spread. So it WAS the degenerate RNG, not a sim-mechanics
divergence and not random init. The "seeded repeat" I was about to run would have been the wrong
experiment.
**⚠️ RUST IS NOT A 1.41× TRAINING LEVER — it is ~4-9%.** The 1.41× in
[[project_rust_bridge_training_enablement]] is `bridge_impl_throughput_benchmark.py`, a TRANSPORT-ONLY
microbenchmark of N parallel env workers. End-to-end it is absorbed exactly like the node-bridge
transport win was (2.1× transport → ~5% training). Rust's case stays OPERATIONAL (25× smaller child,
~10 GB RAM saved at 48 envs), not throughput.
**🚨 THE RUST HALF OF THAT MATRIX IS INVALID — measured on the FIXED-SEED bug.** The worktree was at
`bd964e1` and did NOT have `bc00d4d` ("the seedless bridge START ran on a FIXED seed — every rust
training episode replayed one dice stream"), which had landed on origin/main at 12:33 while the A/B
ran at 13:50-14:47. So every rust episode replayed ONE dice stream. That EXPLAINS the confound I had
flagged as unresolved (rust ep_len 56.9/56.2 vs node 48.7/50.7): not sim divergence, not random init
— a degenerate RNG. **The node numbers (413.1 → 552.3, +33.7% compile) stand; the rust numbers do
not.** Re-running on the fixed tree.
**LESSON (cost me a whole measurement): `git fetch` BEFORE a long benchmark, not after.** A worktree
pinned to an old commit will happily benchmark a bug someone already fixed, and the result looks
clean — tight ranges, plausible deltas, a story that fits. The `/gen3ai-ship` step-4b rebase review
is what caught it, by which point ~40 min of box time was spent.

**📊 A/B ON THE LITERAL ARCH (2026-08-03): 406.5 → 541.8 fps = +33.3%** at n_envs=48, 4 samples/arm,
ranges DISJOINT (off max 417 < on min 512), 48/48 workers compiled, 0 reverts (`tmp/literal_arch_ab.sh`
— the arm the old A/B couldn't run). **KEY READ: the per-forward win has SATURATED.** Doubling it
(3.6x→6.53x) moved end-to-end only ~31%→~33% ⇒ Amdahl: the opponent forward is NO LONGER the rollout
bottleneck. Further compiler work on this path is spent effort; the next lever must be a different
stage (obs build / parse / bridge wait / the PPO update).

**✅ EVAL VERIFIED e2e (bd964e1) + TWO VERIFICATION TRAPS.** `eval_sharding_fuzz_test.py 4 2 --compile
--neural-opponent` drives the REAL `eval_worker._run` over the bridge: `eval-trainee: ON — 3.33 →
0.67 ms (5.0x)` + `eval-opp:final_model.zip: ON`, all exactness assertions unchanged. **TRAP 1:
`--debug-eval` does NOT exercise eval_worker** — its final win-rate eval runs IN-PROCESS, shows zero
compile lines, proves nothing. **TRAP 2: a BOTS-ONLY plan never runs the OPPONENT half** (scripted
bots have no extractor ⇒ `_get_opponent_model` never fires); need a FIXED/SENTINEL item —
`--neural-opponent` reuses the trainee's own ckpt (valid: `load_foreign_opponent` gates on
`arch_signature` only). `src/main/eval_worker_compile_test.py` pins it in the FAST suite.
**BUG I INTRODUCED + FIXED: `_COMPILE_VALIDATED`'s fast path returned SILENTLY**, so the opponent
compile looked like it never ran — took instrumenting the call to disprove. Now logs `ON (reused this
process's validated compile)`. LESSON: a success you can't see in the log is one you'll re-verify.

**✅ ALL NON-TRAINING MODELS SUPPORT COMPILE (1a41503).** The wrapper routes **grad-enabled calls to
eager** (the artifact is inference-only — AOTAutograd's CPU backward fails on the scatter/index_add —
and the prober backprops for saliency), which makes the helper safe to apply anywhere: eval_worker
(the TRAINEE + sentinel + fixed opponents, via a `compile_extractor` cfg key from both eval
callbacks), search_teacher_persistent_worker, snapshot_ladder (default ON, offline), prober
`--compile` (off by default; for better-line/falsify/replay-counterfactual). play.py needs nothing —
it's RandomPlayer vs RandomPlayer. Validation is paid ONCE per process (`_COMPILE_VALIDATED`).

**(superseded) FORKSERVER PRELOAD as originally measured 2026-08-03 —**
`agents/model/compile_preload.py` + `install_forkserver_preload()` called before vec-env creation.
Measured end-to-end (`tmp/preload_integration_probe.py`, 6 workers each building their OWN extractor
with DIFFERENT weights and calling the REAL `maybe_compile_extractor`): forkserver compiles **ONCE
9.6s**, each worker then takes **0.12s** (vs ~30s each) and keeps 6.1-6.6x. The 0.12s is mostly our
OWN eager-vs-compiled timing benchmark; the dynamo cache lookup is ~0.02s (it caches on the CODE
OBJECT, which fork inherits). Live-verified in a real `train_rl_agent` run: "[CompilePreload]
forkserver compiled the extractor in 7.2s". Arch is handed over via `GEN3AI_COMPILE_PRELOAD_ARCH`
(JSON) built from the SAME table as the model — a mismatch only costs a re-trace, never correctness;
a failed preload import is ignored by CPython's forkserver ⇒ degrades to per-worker compiles.
**ENABLER REFACTOR:** `agents/model/extractor_arch.py` — `train_rl_agent` built the extractor-kwargs
dict TWICE inline (fresh path + resume path, 42 keys each, nothing tying them together; a v51 toggle
on one and not the other = resume version-checks an arch it didn't build). Now ONE table, 3 callers,
proven EXACT vs HEAD by replaying the old key→source mapping. Guarded by `extractor_arch_test.py`
incl. a source-level assert that no inline `extractor_kwargs["x"] = args.x` returns.
**LOUD FAILURE:** `_compile_warn` → stderr + launcher event stream (TUI) on every failure path;
`--compile-extractor-strict` promotes to `CompileExtractorError`. **CODE-TIME GATE:** the real-compile
test is now DEFAULT-ON (`GEN3AI_SKIP_COMPILE_TESTS=1` to opt out), +10s on the suite.
**COMPILED RESIDUAL (tmp/compiled_residual_profile.py):** 0.980 ms / **688 aten calls** (eager 6.382 ms
/ 14,375 calls ⇒ **95.2% of dispatches removed**); 1.42 us/call now vs 0.44 eager ⇒ NO LONGER
dispatch-bound. Remaining: fused inductor region 0.506 ms (42%), **`aten::addmm` 0.346 ms (29%) across
60 calls** = the Linear layers, which Inductor does NOT fuse (they go to MKL/oneDNN and at B=1 are
latency-bound, not FLOP-bound), attention 0.034 ms, dynamo guard check 0.030 ms (2.5%). ⇒ the next
lever is FEWER/BIGGER matmuls (arch) or batching across envs, not more compiler.

**⚠️ STARTUP COST + THE PRE-COMPILED-WORKER LEVER (measured 2026-08-03).** Per-worker compile, 16
workers/16 cores, wall clock to all ready: **private cache 163.4 s / cold shared 59.6 s / warm shared
30.1 s** — `TORCHINDUCTOR_CACHE_DIR` is load-bearing, and the residual ~30 s is dynamo tracing the cache
can't remove. **KEY DISCOVERY: SB3's `SubprocVecEnv` uses `mp.get_context("forkserver")` explicitly —
it IGNORES our process-wide `set_start_method('spawn', force=True)`.** So workers fork from a clean
exec'd forkserver process and inherit its memory COW. With `set_forkserver_preload` on a module that
compiles at import: compile happens **ONCE (19.4 s) in the forkserver**, every worker inherits it —
first call 3.95 ms, steady 1.06 ms, **20.4 s wall for 16 workers** (O(1) in N, vs 30.1 s O(N)-ish).
`tmp/fork_inherit_probe.py` confirms the mechanism (forked children at steady compiled speed, 6.73×
in-child, in **0.2 s**). NOT BUILT — the preload module needs the run's arch kwargs, which are assembled
inline at the model-build site; doing it right means factoring that assembly out first rather than
duplicating ~50 lines that would drift. This is the "compile once and serve" answer for SB3's vec-env
world. Probes: `tmp/compile_spawn_cost.py`, `tmp/forkserver_preload_probe.py`.

**⚠️ CORRECTION 2026-08-01 — "compile is dead" is WRONG for the WHOLE-EXTRACTOR compile. That was the
one configuration never tested** (this file's own "FOLLOW-ONS (a) compile the WHOLE extractor inference
forward"); the 0.70× below compiled ONLY the DamageOp inside `get_distribution`, where dynamo overhead
ate the win. Measured on the v50 unified config, B=1 CPU, 1 thread, min-of-N, idle box:
**extractor-only 4.74 → 0.86 ms = 5.5×**, and through the REAL path on a real checkpoint
**`policy.get_distribution` 4.84 → 0.91 ms = 5.3×** (compile ~16 s/process). ADVERSARIALLY VERIFIED:
`torch._dynamo.explain` = **0 graph breaks / 1 graph / 2073 ops**, values match eager to 9.5e-7 abs
(3.7e-7 rel), **0/16 argmax flips**, speedup IDENTICAL on a rotating 32-obs bank (not an input-reuse
artifact), and patching the BOUND `fe.forward` (not the module) keeps state_dict keys clean = resume-safe.
At B=256 (learner forward) it is 1.28×.
**THE BLOCKER IS ONE FLAG: `--threat-unrevealed-outgoing` (v36 #2) CRASHES the inductor backend**
(`BackendCompilerFailed: AssertionError: buf307`, a [1,6,400] pointwise buf = the species×moves
expected-latent marginalization). Bisected: `--threat-refine-outgoing` and `--threat-status-refine`
compile FINE; the refine LOOP itself compiles fine. So the production config is uncompilable purely
because of that one K10-null channel — v50 (`--damage-op-prefuse`) drops it as a side effect, but
dropping the flag alone is enough.
**CONSEQUENCE — compile SUBSUMES the eager micro-optimizations.** Under compile the refine loop costs
0.05 ms (vs 0.69 ms eager), so v50's +28.2% eager win shrinks to ~5%; and the whole outgoing family
(block + v34 + v39) costs **0.040 ms = 4.6% compiled vs 1.39 ms = 29% eager**, which kills the
"merge the three overlapping outgoing matrices" idea (it would triple the FLOPs to save dispatch that
compile already removed). STILL-STANDING RISKS from the saga below: the **CUDA-context explosion** (48
CUDA-visible workers × ~0.5 GB → OOM; any deployment must hide CUDA or stay in one process), per-process
compile cost, and the barrier-absorption caveat — end-to-end FPS remains UNMEASURED.

**PRODUCTION-SCALE A/B 2026-08-02 — +31.0%, RANGES DISJOINT.** n_envs=**48** (the ai_v8_03 shape, read
out of its `metadata.json` `cli_args` by `tmp/production_cmd.py`), `--async-rollout`,
`--grad-checkpointing`, `--self-play --self-play-use-cpu` against a SEEDED pool (one real snapshot +
`persist_summary(win_rate_vs_bots=0.95)` ⇒ self_play_fraction 80%), bridge=node, cuda, PopArt, full
belief/damage/zarch stack. Marginal fps **498.2 → 652.6 (+31.0%)**, 6 samples/arm, **off max 512 < on
min 614 (disjoint)**. 48/48 workers compiled, 0 failures, 0 CUDA/OOM errors, exactly ONE CUDA context
(the learner). **This KILLS the barrier-absorption prior for this lever** — the win is BIGGER at 48
envs than at 8 (+26.6%), the opposite of what absorbed levers do. Adversarial checks: the compiled
path is `pool:snapshot_*` (the real self-play opponent, not some other model), and `ep_len_mean` is
unchanged (47.4 vs 45.9, within noise) so the FPS gain is NOT shorter battles; `ep_rew_mean` −32.2 vs
−32.4 (value-preserving).
**CORRECTION to the "production can't compile" claim:** with `torch._dynamo.config.suppress_errors =
True` the LITERAL production arch (with `--threat-unrevealed-outgoing`) DOES compile — 6.48 → 1.78 ms
(**3.6×**) — because dynamo falls back to eager per-FRAME for just the failing region instead of
failing the whole compile. So dropping that flag is NOT required. (Without suppress_errors it raises
`BackendCompilerFailed`.) End-to-end at production scale was measured on the refine-0 arch, not the
literal one.
**The helper SELF-VALIDATES** (`maybe_compile_extractor`): it times eager vs compiled at load and
REVERTS to eager if compiled isn't ≥5% faster — precisely because `suppress_errors` makes a silent
no-op possible, and a knob that logs "ON" while delivering nothing is worse than one that's off.
Tests: `src/agents/model/compile_extractor_test.py` (10) + `src/main/thread_pinning_test.py` (5).

**END-TO-END A/B 2026-08-02 — IT SURVIVES THE BARRIER: +21% to +27% TRAINING FPS.** Prototype
`--compile-extractor` (BUILT, not shipped) wired into the pool / stable / exploiter opponent loads via
`snapshot.maybe_compile_extractor`. Matched two-arm run, `--exploiter` substrate (ONE frozen neural
model as the SOLE opponent in every worker, CPU, so 100% of opponent decisions are the compiled path),
n_envs=8, 40k steps/arm, threads pinned: **cumulative fps 230 → 279 (+21%)**, **steady-state MARGINAL
fps 228 → 289 (+26.6%)**. The ON arm RAMPS (176 → 278) because SB3's `fps` is cumulative and amortizes
the one-time compile. This is the first throughput lever in this file that did NOT get absorbed —
unlike the learner-only compile (+0.2%) and the transport win (~5%). CAVEATS: one run per arm,
per-iteration ranges just touch (off max 256 = on min 256); n_envs=8 on an UNSATURATED box (production
48-64 + the SubprocVecEnv barrier may absorb more); and the exploiter substrate is 100% neural
opponents, so the realized win scales with the neural-opponent fraction (production self-play mixes in
cheap bots).
**CUDA-CONTEXT GUARD VERIFIED LIVE**: with 8 compiled CPU workers running, `nvidia-smi` showed exactly
ONE process holding a context (the learner) — the June 48×0.5 GB OOM does not recur, because the helper
sets `CUDA_VISIBLE_DEVICES=""` when the model is on CPU and CUDA is not yet initialised in that process.
**TWO BUGS THE PROTOTYPE HAD TO FIX (both would bite anyone repeating this):** (1) the
`ObservationDebugger` runs NUMPY asserts inside `forward`, and dynamo dies creating a guard
("TypeError: 'numpy.bool' object cannot be interpreted as an integer") — a frozen opponent doesn't need
it, so the helper drops it; (2) `torch.compile` is LAZY, so the real compilation happened on the first
live decision, OUTSIDE the try/except — the helper now forces it with a zero-obs warmup inside the
handler and sets `suppress_errors=True` so any failure degrades to eager instead of killing a run.
**GOTCHA (not a production bug):** running `train_rl_agent.py` DIRECTLY with many neural-opponent envs
thrashes — 16 envs × multi-threaded BLAS gave load average 110 on 16 cores and **6 fps**. The launcher
already pins `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1` (`src/main/launcher/child.py:23-24`); direct runs
do not, so export them by hand.

**THROUGHPUT PROFILE (2026-06-29, live ai_v7_02 run, py-spy + nvidia-smi):** the box is the bottleneck, NOT the GPU
per se. GPU duty cycle = **mean 56% / median 87%** (bimodal: ~20% during rollout, ~90% during the grad-accum-8 PPO
update) — so the old "GPU ~86% idle" no longer holds for THIS heavy arch. **MAIN learner: 87% of time = the MODEL
FORWARD, of which ~45% is the DamageOperator** (`_boost_mult`/`_rolls`/the matrices); env-worker comm only ~6%. **ENV
WORKERS (rollout): 68% = the self-play OPPONENT inference** (same heavy model, run on CPU, per-env B=1, un-batched),
13% obs build, 9% parse, 8% bridge wait. So the bottleneck is the HEAVY MODEL (esp. the DamageOperator) hitting twice:
GPU (learner) + CPU (48 opponents). The old "obs build 88%" memory was vs-BOTS (cheap opponents); under SELF-PLAY the
neural-opponent forward dominates the workers. The DamageOp cost is OVERHEAD/dispatch-bound (a latency problem that
caps throughput), NOT FLOP-bound — proof: GPU not saturated during the forward. Fix = FUSION (torch.compile), not a
faster GPU. The 3 levers (ranked): (1) batched GPU opponent-inference server [bucket-by-pool-model, ~10 buckets/48
envs; the 68% lever, big build], (2) **torch.compile the DamageOp [DONE, below]**, (3) ablate low-value DamageOp
matrices.

**DAMAGE-OP torch.compile — BUILT 2026-06-30 (worktree bridge-cse, NOT shipped, behind `--compile-damage-op`):** the
DamageOperator compiles **CLEANLY — 0 graph breaks** (pure tensor math, constant-bound loops, fixed toggles → one
fused graph), **bit-correct** (allclose, max err 5e-7 = float reassociation), CPU speedup **17.5× at B=1 (the per-env
opponents) / 8.5× at B=64**. GOTCHA: the TRAINING BACKWARD fails — Inductor CPU codegen errors on the op's
scatter/`atomic_add` (the HP-type-belief `index_add`). So the integration is **INFERENCE-ONLY by design**: `_apply_
compile_damage_op` in train_rl_agent.py (mirrors `_apply_grad_checkpointing` — a runtime perf knob, NOT in
model_version/check_compatible, OFF byte-identical) compiles the bound `op.forward` and dispatches on
`torch.is_grad_enabled()` → compiled under no_grad (rollout / opponents / eval = the bottleneck), EAGER under grad
(the PPO update). Patches `op.forward` (instance attr) NOT the module → **state_dict keys UNCHANGED = resume-safe**
(a module-wrapping `torch.compile(op)` would prefix keys `_orig_mod.` and break resume). Re-pass `--compile-damage-op`
every launch (perf knob, not inherited). VERIFIED: full smoke (`--compile-damage-op --grad-checkpointing`) — roundtrip
PASSED, rollout episodes compiled, PPO update ran eager (no codegen crash), Training complete, save/reload clean; 751
model unit tests pass. UNMEASURED: the GPU speedup (CPU-only proven; mechanism = fewer kernel launches, holds on GPU;
didn't bench GPU to avoid contending the live run). FOLLOW-ONS: (a) compile the WHOLE extractor inference forward (op
is ~45% → ~2× the win), (b) fix the scatter backward to compile the training update too. To use: /gen3ai-ship +
relaunch with `--compile-damage-op`. See [[project_plateau_research_2026_06_25]] (ai_v7_02 = the run this speeds up).

**SHIPPED d2923e8 + MEASURED 2026-06-30 — FPS FLAT (+0.2%), as feared:** shipped `--compile-damage-op`, resumed ai_v7_02
with it from step 101.19M. `time/fps` A/B: before (compile OFF, 70-101M) mean 400 vs after (compile ON, >101M) mean 401 =
**+0.2%, FLAT**. WR/ELO unchanged (value-preserving; bot-WR ~0.89, ELO ~1975), no recompile churn / no eager-fallback (the
`ctx.batch_size` int-guard risk did NOT bite). ROOT CAUSE (predicted): `_apply_compile_damage_op` compiles ONLY the
LEARNER's policy (main process) — NOT the 48 self-play OPPONENTS (separate uncompiled snapshot models in the env WORKERS),
and the profile says those opponents are the 68%-of-rollout-worker bottleneck → compiling the non-gate stage moves nothing.
Compile kept ON (harmless, value-preserving, positions the follow-up). **THE REAL LEVER = compile the OPPONENT snapshots in
the env workers** (patch the self-play/env opponent-load path the same way; ~17× at B=1 on CPU could ~2-3× rollout throughput;
watch the 48-worker startup compile cost — shared TORCHINDUCTOR_CACHE_DIR to amortize). NOT yet built.

**OPPONENT COMPILE — BUILT 2026-06-30 (worktree bridge-cse, NOT shipped; reuses `--compile-damage-op`, one flag = learner
+ all TRAINING opponents):** shared helper `maybe_compile_opponent_damage_op(model, enabled, label)` moved into
`agents/model/snapshot.py` (learner's `_apply_compile_damage_op` is now a thin wrapper → identical impl). Wired at the 3
TRAINING opponent sites (mapped by a 5-agent workflow): pool (`snapshot_pool.SnapshotPool.load_model` after
load_model_snapshot; new `compile_damage_op` ctor arg, threaded from train_rl_agent:2207), stable + exploiter (after each
`load_foreign_opponent` in train_rl_agent). Plus `os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/gen3ai_inductor_cache')`
at module top so the ≤64 spawn-workers codegen the op ONCE + cache-hit (workers re-import under spawn). VERIFIED: real
ai_v7_02 snapshot loaded as opponent + compiled → get_distribution runs, logits allclose 4.77e-7 (value-preserving),
argmax move IDENTICAL (self-play fairness intact), state_dict 445 keys UNCHANGED (resume-safe); 1823 model+training + 208
snapshot/pool tests pass; self-play smoke ran (learner compile fired, episodes clean, no codegen crash). DEFERRED: the EVAL-
worker opponent compile (eval latency, NOT the training-FPS gate — a follow-up: eval_worker `_get_opponent_model` + a cfg key
threaded from the two callbacks). **CRITICAL UNCONFIRMED (the spec's #1 risk = worth-it gate):** the SubprocVecEnv step
BARRIER + CPU saturation across 64 workers could ABSORB the per-worker 17x like it absorbed the learner-only +0.2% (same
reason n_envs isn't the FPS lever). MUST A/B the actual `time/fps` (ship + resume ai_v7_02, compare vs the current ~400) —
if flat, the real lever is fewer/faster opponents (a batched-GPU opponent server), not per-worker compile.

**EXPERIMENT FAILED 2026-06-30 — the opponent compile CRASHES (worse than flat): SHIPPED 6cfb023, resumed ai_v7_02 with
`--compile-damage-op` → the opponent compile FIRED in all 48 workers (48 CompileDamageOp lines, pool-opponent compiling)
but then FATAL `torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: RuntimeError: CUDA error: out of
memory`. ROOT CAUSE: the 48 env-worker subprocesses have CUDA VISIBLE (inherited from --device cuda), and
`torch.compile`→Inductor INITIALIZES A CUDA CONTEXT per worker during codegen (~0.3-0.6GB each) EVEN for a CPU opponent
model → 48×~0.5GB + the learner's 5GB ≫ the 12GB card → OOM. The workers deliberately load opponents on CPU to AVOID
per-worker CUDA contexts (the log even says so), and compiling them there re-introduces exactly those contexts. RECOVERED:
relaunched ai_v7_02 WITHOUT --compile-damage-op (6cfb023 code, flag off = byte-identical to no-compile) → healthy 5GB/90%,
resumed step 102.8M. **CONSEQUENCE: `--compile-damage-op` is now a FOOTGUN on main — safe learner-only at d2923e8, but at
6cfb023 it CRASHES any self-play run.** FIX OPTIONS (follow-up, not done): (a) REVERT 6cfb023 → restore safe learner-only
(recommended — the FPS win was UNCONFIRMED and likely barrier-absorbed anyway, so not worth the CUDA-context surgery);
(b) make the worker opponent compile CUDA-free (force Inductor CPU-only / hide CUDA in the workers — but they're spawned
from the cuda main, and CUDA may already be inited). LESSON: torch.compile in CUDA-visible subprocesses inits a CUDA
context per process regardless of the model's device — a 48× context explosion. The learner-only compile was safe ONLY
because it's ONE process. This also re-confirms per-worker compile is the wrong lever — the batched-GPU opponent server
(one process, one context, batched) is the right one.

**BATCHED-SERVER DE-RISK BENCHMARK 2026-06-30 — DEAD END, don't build it (saved a multi-hour build):** isolated benchmark
of the REAL opponent forward (`policy.get_distribution`, a real ai_v7_02 snapshot). Latencies: CPU B=1 = 7ms (16.5ms
contended w/ the live run); **GPU forward = FLAT ~32ms regardless of batch (B=1:30ms … B=48:32ms) → LAUNCH-BOUND** (the
uncompiled DamageOp's hundreds of tiny kernels), NOT compute-bound. Decisions/sec: CURRENT (per-worker CPU, 48 envs/16
cores) **969**; BATCHED realistic (10 model-buckets of ~5, since the pool has ~10 distinct models across 48 envs) **152 =
0.16× = 6× WORSE** (10 sequential 32ms bucket-forwards = 315ms/step); OPTIMISTIC all-48-one-batch (unachievable) only
1.54×. A single B=1 opponent forward is even FASTER on CPU (7ms) than GPU (32ms). ALSO tested CPU-only opponent compile
(CUDA hidden = crash-safe): **0.70× (SLOWER)** — the op-alone 17× does NOT translate through the full get_distribution
path (dynamo overhead; the op isn't the dominant cost there). VERDICT: **ALL throughput levers measured are dead ends** —
learner-compile flat / GPU-opponent-compile crashes / CPU-opponent-compile slower / batched-server 6× worse. The DamageOp
is launch-bound but compiling it doesn't help the real forward + model-bucketing kills batching. STOP pursuing compile/
batching for opponent throughput. The only real levers are STRUCTURAL (ablate damage-op matrices to shrink the op / lighter
opponent / accept ~400 fps). And throughput isn't blocking anything — ai_v7_02 is at 0.92 WR beating the all-time record.
CLEANUP PENDING: revert 6cfb023 (opponent compile, the footgun) on the next /gen3ai-ship; --compile-damage-op learner-only
(d2923e8) is flat-but-harmless (could also revert it — does nothing). ai_v7_02 currently runs SAFE without the flag.

**REVERTED 2026-06-30 (worktree bridge-cse, NOT yet committed — awaiting /gen3ai-ship):** user said "revert the compile
changes." Reverted BOTH commits (d2923e8 learner + 6cfb023 opponent footgun) by `git restore --source=ee8f2b9 -- <3 files>`
(train_rl_agent.py, snapshot.py, snapshot_pool.py — ee8f2b9 is the commit right before d2923e8, and the linear log confirms
those two compile commits are the ONLY ones touching the 3 files since, so restore == exact reverse, 86 deletions). Working
tree is now pre-compile / unstaged; grep confirms 0 residual refs to compile_damage_op / maybe_compile_opponent_damage_op /
TORCHINDUCTOR_CACHE_DIR; 3060 unit tests pass (2:03). LIVE RUN UNAFFECTED: ai_v7_02 is pinned to HEAD 6cfb023 but runs
WITHOUT --compile-damage-op (byte-identical to no-compile), so the footgun was dormant; the revert only lands on next
/gen3ai-ship. NOTE at ship time: the worktree has 2 untracked `tmp_repro_crash*.py` from a PARALLEL session — must be
EXCLUDED from the commit (never `git add` them). So the whole compile saga nets to ZERO code on main — the throughput
investigation's value is the MEASUREMENTS (all levers dead), not the flag.
