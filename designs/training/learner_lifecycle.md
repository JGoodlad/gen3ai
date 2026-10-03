# The learner's DECLARED LIFECYCLE (K6) — the freeze guard, startup declaration, the static gate

M5 Lane K item K6 (`designs/endstate/program_rust_core.md`, the Lane K row, the "M5 DESIGN PRINCIPLE —
a DECLARED LIFECYCLE" paragraph and the Decision record). The principle (owner, 2026-09-28): a
training process is a long-lived server — STARTUP declares and acquires every resource the steady
state will use, then FREEZES; the STEADY STATE acquires nothing, and anything that would appear
lazily is a counted, typed failure at the point it happens. Until K6 the principle was unenforced in
the learner: the K8 inventory found the ride-along heads' Adam built lazily on the first update — by
luck, not by any check.

Code: `src/agents/training/learner_lifecycle.py` (the freeze guard, `declare_optimizer_state`,
`declare_learner_startup`, `attach`), `src/agents/training/lifecycle_decl.py` (`@startup_builder`),
wired by `main/train/lifecycle._arm_learner_lifecycle` (both the fresh and the resume path of
`model_build`, right after the compile sentinel is armed). Tests: `learner_lifecycle_test.py`.

## Two classes of check — DETERMINISTIC (single-shot FATAL) and STOCHASTIC (a leak detector)

| class | what | verdict |
|---|---|---|
| deterministic | a new optimizer, `nn.Parameter`, module, buffer or optimizer-STATE entry in the learner after the freeze; an undeclared compile signature (K6's compile half) | a typed, single-shot FATAL — a healthy run never trips it, so there is no tolerance and no warn tier |
| stochastic | CUDA memory | a LEAK DETECTOR (`CudaMemoryWatch`, "The memory half" below), warn by default; stops (checkpoint, then a typed FATAL) only when a sustained trend projects an OOM inside a declared horizon (orchestrator, 2026-09-30: the caching allocator legitimately steps up from fragmentation, larger shapes and eval cycles, so a one-off crossing would eventually kill a healthy multi-day run) |

## The freeze guard (`gen3_learner_freeze_v1`)

**When.** `learner_lifecycle.attach(model)` registers the `learner_freeze` owner on the loop's DECLARED
HOOK TABLE (`agents/training/loop_hooks.py`, `gen3_declared_loop_hooks_v1`): around-hooks on the
`learn`, `collect` and `update` points that `OwnedLoop.learn` opens. It is OUTERMOST by the table's
`HOOK_OWNERS` order, over the compile sentinel's, whatever order the two attached in. The FIRST
`collect` entry FREEZES — startup is then over: the trainer compile and its parity gate, the prewarm,
`declare_learner_startup` and `learn()`'s own `_setup_learn` / every callback's `_on_training_start`
have run. Every `collect` and `update` exit CHECKS. `learn()`'s exit RELEASES (the in-process final
evaluation after training is not the steady state).

The table itself FREEZES at `learn()`'s training start: a hook registered later is `LoopHookError`
(FATAL_CONFIG), and so are an undeclared owner and a duplicate. `_learner_freeze` and the table
(`_loop_hooks`) are in `_excluded_save_params`, so no checkpoint carries them. A duck-typed model with
no table (a test stub, a tool driving `collect_rollouts` / `train` itself) gets the SAME hook bodies as
instance-attribute wrappers (`loop_hooks.install`). (`loop_hooks_test.py`; the `sb3_reference` seam the
table also served was deleted with PPO stage 3, deletion pass U4.)

**What is frozen — the learner's object graph** (`learner_objects`): `model.policy`'s whole module
tree (extractor, pointer head, aux heads, the ride-along heads) plus every `nn.Module` / `Optimizer`
the model, the policy or a policy submodule holds directly, one container deep (a list / dict of
optimizers) or one attribute deep into a repo-owned helper object (e.g. the capacity telemetry's canary
head and its Adam). Per object:

| kind | identity | a change after the freeze |
|---|---|---|
| module | object id | `NEW MODULE <path> = <class>` |
| parameter | object id (names kept for the message) | `NEW PARAMETER <name>`; the same name with a new object is `REPLACED PARAMETER` (the optimizer still holds the old one — the dead-param class `policy._build` documents) |
| buffer | qualified NAME (a buffer re-assigned in place under its own name is not an acquisition) | `NEW BUFFER <name>` |
| optimizer | object id + its param groups' parameter ids | `NEW OPTIMIZER <path>`; `PARAM GROUPS CHANGED` |
| optimizer state | which parameters carry state | `OPTIMIZER STATE created after the freeze` (state is declared at startup — next section) |

**The verdict.** `LazyAcquisitionError` (a `main.exit_codes.FatalConfigError` → exit `FATAL_CONFIG`
(3), the launcher does NOT restart), message tag `[LearnerLifecycle] FATAL`, naming every acquisition
with its attribute path, class / shape and CONSTRUCTION SITE. Raised from the wrapped `train()` /
`collect_rollouts()`, it reaches `model_build`'s handler, which exits `exit_code_for(e)`.

**How the site is known.** While frozen, three recorders run; each only RECORDS, so an object built
for something other than the learner (an in-process opponent under `--debug`) is recorded and ignored
unless it joins the learner's graph (pinned by a test):
- torch's GLOBAL module / parameter / buffer REGISTRATION hooks (`torch.nn.modules.module.register_module_*_registration_hook`) record the stack of every registration;
- `torch.optim.Optimizer.__init__` is wrapped (restored at release) to record every optimizer built;
- a GLOBAL optimizer STEP pre-hook (`register_optimizer_step_pre_hook`) RAISES the moment an optimizer outside the frozen set steps — before it can move a weight — and records the violation STICKY, so a caller's `except Exception` cannot hide it from the next check.

A site is the innermost repo frames (torch / site-packages frames dropped), read off the frame chain
without a source-line lookup, so a recorder firing on every registration of an in-process model load
stays cheap.

**Cost.** One identity walk of the learner graph per update and per rollout (93 modules / 120
parameters / 180 buffers at `--debug`; 196 / 254 / 240 at the production surface, the K9 golden's
learner) — negligible against a ~50 s update.

**Scope limits.** The guard judges the LEARNER's graph; a resource held somewhere it cannot reach
(a module on a callback, a tensor allocated without a module) is out of its reach — a stepping
optimizer is still caught by the step hook, and memory by the leak detector. Tensors are not tracked.

## Declared at startup (`declare_learner_startup`)

`_arm_learner_lifecycle` runs it right before `attach`:

- **Every Adam / AdamW optimizer's STATE** (`declare_optimizer_state`): torch creates Adam's moments
  and step counter lazily, at the first `step()`. The helper creates them NOW with torch's OWN init —
  one `step()` over ZERO gradients for exactly the stateless trainable parameters (every other
  parameter's `.grad` is None for the call, so torch skips it), then the parameters are copied back
  BIT-EXACTLY (AdamW's decoupled weight decay moves a weight even on a zero gradient) and every state
  tensor is zeroed and the step reset to 0: the exact state the first real step would have created at
  its entry. Any other optimizer class REFUSES (typed): SGD's momentum buffer, for one, is initialised
  from the first gradient, so zeros would be wrong. Pinned bit-identical on AdamW-with-decay, Adam and
  amsgrad Adam (parameters AND every moment after three steps), and on the PRODUCTION SURFACE: the K9
  learner golden's production-surface learner, declared and frozen, runs one real update whose
  fingerprint equals the recorded golden (`test_a_production_surface_update_acquires_nothing_and_leaves_the_golden_unchanged`).
  Cost: the production optimizer's 254 state entries (2 × 3.07M floats ≈ 25 MB) exist from startup
  instead of from the first step.
- **The capacity telemetry's canary** (`--capacity-telemetry`, off in production): its head + Adam
  were built on the first minibatch; `CapacityTerms._capacity_declare` builds them at startup
  (`CapacityTelemetry.declare`, D_MODEL-wide input). A direct caller that never declared (a unit
  test) still gets the old build on first use — after a freeze that build is the FATAL.

A startup builder is marked `@lifecycle_decl.startup_builder`; the static gate reads the decorator by
name, the freeze guard proves at runtime that it ran before the freeze.

## The compile half — declared signatures, the lock BEFORE the first iteration, the canary

**The declared table.** The region table (`compile_regions.REGIONS`, prewarmed through
`compile_regions.prewarm_calls`) IS the declaration: every compiled-learner signature (region × batch ×
train/eval × grad) the steady state may reach, prewarmed at startup (`arm_compile_sentinel`). Batch 1 is never in it: it always runs eager
(`gen3_batch1_eager_v1`, `compile_flags.md`). The compile sentinel then LOCKS right after the prewarm —
before the first real iteration (was: after the first `train()`, `8fc297a2`'s interim, which absorbed
whatever iteration 1 compiled). A signature outside the table is a typed FATAL (`[CompileSentinel]
FATAL`, exit FATAL_CONFIG) that NAMES the failing guard(s): the `fail_on_recompile` stance's rejection
says nothing, so the sentinel replays the call once under `error_on_recompile` to read them (`compile_control._diagnose_rejection`).

**The iteration-1 signature `8fc297a2` absorbed — FOUND (2026-09-30).** The real trainer at the
production surface (`--debug --arch production`, CPU, dynamo `eager` backend — guards are
backend-independent; `~/gen3ai_archive/k6_k8/sigprobe/`) with the lock moved before iteration 1 was
rejected four times at the first update, on `torch_dynamo_resume_in_forward_at_281` (2.5.1's frame after
the `forward_guard` break) with guard failure `len(L['self']._modules['team_transformer']._forward_hooks)
!= 0`: the RANK PROBE (`rank_metrics.rank_probe`, first micro-batch of every update) captured the trunk
and the value CLS with forward hooks, and a hook on a compiled module is a guard. The probe is now
hook-free (`gen3_rank_probe_stash_v1`: the extractor stashes `trunk_tokens` and `value_cls` — references,
no copy), so its forward is exactly the declared `rank-probe train/no-grad` signature. ⚠️ On torch 2.8
the hooked probe did NOT recompile — dynamo skips (neither guards nor runs) a hook registered on a
submodule of an already-compiled frame — so `rank/trunk_*` and `rank/value_cls_*` were silently
MISSING on every compiled 2.8 run. `rank_metrics_test` fails on a revert to hooks on both torches.

**The in-run parity canary (`gen3_compile_canary_v1`, `agents/model/compile_canary.py`).** The startup
gate proves the compiled graph at t = 0; the canary proves it at t = N. It runs first at update
`CANARY_FIRST` (10), then every `CANARY_EVERY` (100) updates, between updates (owner, 2026-10-01: "up
it to 100"; the first run at 10 so no run of any length goes unchecked — sizing arm A's 82 updates
never reached 100), on the committed real-obs fixture
and through DECLARED signatures only. Each run compares compiled against eager on two things:
- the decision readout (masked legal log-probs, V) at the rollout signature (eval / no-grad / `n_envs`);
- the train graph: on K8, region R1's loss and every policy gradient, with the per-parameter bar
  chosen by `compile_regions.weights_regime`, on R1's gate rows — ALWAYS the K9 golden's labelled
  buffer, a key it lacks at its declared placeholder (`compile_regions.r1_batch`; P10-C: before it, a
  fork-shaped run's canary judged zero-label rows, with ~48 critic / intent / belief parameters
  unjudged); on the legacy compile, the gate's probe loss.

The train-graph check now runs at EVERY canary (`GRAD_EVERY` 1). It used to run every 4th. It costs
0.65–0.71 s at the production shape: arm C's weights, CUDA, `n_envs` 48, B = 2048, R1 through the compiled region and eager. The decision readout alone costs 0.06–0.09 s. Against a 36 s update every 100 updates that is under 0.02% (`~/gen3ai_archive/k6_k8/r1bar/canary_cost.log`, 2026-10-01).

The bars are the startup gate's, at fp32 `highest` (the only matmul precision; TF32 was retired, deletion
pass K2). Live weights are trained, so vacuity is not a refusal here.

**Persistence, not a single shot** (owner, 2026-10-01: "implement the consecutive check"). A real
miscompile is deterministic and disagrees every time. A healthy graph's rare exceedance belongs to one
batch on one weight state. Waiting for the next scheduled canary would mean ~1 h of training on a bad
graph, so a disagreement is confirmed IN THE SAME UPDATE:
- **Warn and dump** to `<run_dir>/canary_disagreements.jsonl`.
- **Re-run the whole comparison**, compiled AND eager recomputed, on the same rows and on an
  INDEPENDENT fixture slice (`compile_trainer.fixture_index` slice 1: the second half of the rows,
  tiled, the same declared shape).
- **Confirmed** (it disagrees again on both): checkpoint to `<run_dir>/final_model_canary_fatal.zip`
  (forensics only; `latest.txt` is not moved), then `CompileCanaryError` (FATAL_CONFIG, not restarted).
- **Not confirmed:** `compile/canary_unconfirmed_disagreements` counts it and training continues.
- **Two consecutive scheduled canaries that disagree are FATAL**, even when neither confirmed.

Every verdict (update, step, pass / unconfirmed / fatal) is appended to
`<run_dir>/canary_verdicts.jsonl`; the run dir is `model.behaviour_dump_dir`, which K9(b) also uses.
The FATAL names the SAFE ROLLBACK POINT: the newest `checkpoints/*_<N>_steps.zip` at or before the last
PASSING canary's step (`rollback_point`; the file is the run's, so it spans restarts).

The canary runs under `fork_rng`, restores the training mode and leaves every `.grad` None, so training
is untouched (pinned). TB scalars, written on canary updates only:
- `compile/canary_ok` (1 pass, 0 unconfirmed);
- `compile/canary_grad_checked`;
- `compile/canary_unconfirmed_disagreements`;
- `compile/canary_seconds`;
- `compile/canary_max_abs_<q>`;
- `compile/canary_grad_cosine`.

`compile_canary_test` fails on each of these: a forward that drifted (×1.01); a BACKWARD-only drift; a
planted persistent R0 fault (FATAL in the same update, with the checkpoint and the rollback point); a
one-shot blip that FATALs instead of counting; two consecutive unconfirmed disagreements that do not
FATAL; and a wrong rollback-point choice.
## The memory half — the CUDA memory TREND, wired (`gen3_cuda_memory_trend_v1`)

The detector itself (what it reads, the floor / demand / ceiling, SUSTAINED / STOP / WARN, the
calibration and the false-trip estimate) is [`learner_gates.md`](learner_gates.md) "K6 — the CUDA
memory TREND". Its call site is `learner_lifecycle.CudaMemoryWatch`, attached by `attach` with the
freeze guard (so every launch that arms the lifecycle has it; inert off CUDA):

- one `MemoryTrend` per process, STARTED at the freeze (the first rollout of `learn()`);
- a sample at the two quiescent points of every update — after `collect_rollouts` (`post_rollout`,
  the updates completed so far) and after `train()` (`post_update`);
- at every window close (2 updates) the projection is LOGGED to the run's log (stdout) as one
  `[CudaMemTrend] <level> @ update N: floor …, slope …/update, demand …, ceiling …[, projected OOM in
  K updates (horizon 25)]` line and recorded to TB (`lifecycle/cuda_*` — level, floor, demand,
  ceiling, slope, updates to the ceiling, reserved, device free, segments, streams, retries, OOMs,
  the after-freeze counters); a WARN that changes between closes is printed when it happens, and a
  changed WARN or the STOP also goes to the launcher's event channel;
- at EVERY sample, whatever the window state, one stable tag set: `lifecycle/cuda_reserved_mib`,
  `lifecycle/cuda_device_free_mib`, `lifecycle/cuda_reserved_after_freeze_mib` and
  `lifecycle/cuda_streams_after_freeze` (`gen3_reserved_after_freeze_v1`; a 3-update probe has them —
  the window-close tags need 4 updates). The section "The staged batch's stream" below says what the two
  after-freeze counters catch;
- a STOP raises `CudaMemoryLeakError` (`[LearnerLifecycle] STOP — CUDA MEMORY LEAK: …`; NOT a
  configuration error) at the update's end; the trainer's exception handler saves
  `final_model_exception.zip` (the checkpoint) and the process exits `FATAL_CUDA_LEAK` (6). A fresh
  process clears a leak, so the launcher RESTARTS from that checkpoint, loudly, at most
  `exit_codes.CUDA_LEAK_RESTART_CAP` (2) times per session, and stops for good on the next one
  (orchestrator, 2026-10-01; `src/main/launcher/cuda_leak_exit_test.py`).

`learner_lifecycle_test` drives it with a synthetic sampler: a 40 MiB/update leak against 1 GiB of
headroom STOPS with the typed FATAL after logging the projection at every window; a flat floor and a
single +200 MiB step never stop; `attach` samples in the order rollout → update. Each fails on a
revert of the wiring.

## The CUDA memory LEDGER and declared slot loads (`gen3_cuda_ledger_v1`, `gen3_declared_slot_load_v1`)

**The ledger.** Every run logs where the learner process's card goes, by startup step:
weights → the rust env core (T2 slots, staging, arena) → the extractor-only gate (2.5.1) → the
compiled regions' gate + prewarm + lock → optimizer state. Each step records allocated, reserved, its
delta, and the peak during it, in `[CudaLedger]` lines and `<run_dir>/cuda_ledger.json`
(`agents/training/cuda_ledger.py`). Every update also records these TB scalars:
- `lifecycle/cuda_update_peak_{alloc,reserved}_mib`
- `lifecycle/cuda_rollout_peak_{alloc,reserved}_mib`
- `lifecycle/cuda_{update,rollout}_end_alloc_mib` (the quiescent floor)
- `lifecycle/device_batch_mib`

MEASURED: production shape, N = 48, torch 2.8, fp32, `--env-core rust`, 3 updates; arm A's own argv
(`designs/research_state/measurements/k6_k8/memory_audit/`).

The startup steps:

| startup step | allocated (Δ) | reserved (Δ) | peak during |
|---|---|---|---|
| weights | 15 MiB | 34 MiB | 15 MiB |
| T2 | +712 MiB | +1,680 MiB | 1,111 MiB |
| the compiled regions | +185 MiB | **+4,120 MiB** | 5,034 MiB |
| optimizer state | +23 MiB | — | — |

The regions row's reserved jump is the R1 gate's eager and compiled forward + backward at B = 2048,
cached by the allocator.

Per update (the same shape, BEFORE the 2026-10-01 fit levers below): an update PEAKED at 8.75 GiB
allocated / 9.49 GiB reserved, 1.14 GiB of it the then-resident device batch; the quiescent floor is
851–855 MiB. What the update peaks at NOW is the next section's table.

**Declared slot loads.** A rust-core pool refresh is a DECLARED LOAD into its T2 slot.
- The pool loads snapshots on the CPU. On the card, each promotion's snapshot stayed in the pool's
  LRU (cap 3) beside the slot T2 had copied it into: +~33 MiB of quiescent floor per promotion on
  sizing arm A.
- A pool weight source found on the card is a typed `LazyAcquisitionError`.
- Every route's slot load goes through `rust_rollout.build.checked_slot_load`, which refuses a load
  that leaves more than 1 MiB newly allocated (`declared_slot_load_test`). It reads ALLOCATED, not
  reserved, memory. Measured 2026-10-03 (P10 F7, buckets 8 / 64 / 256): an opponent load (CPU
  source) grows neither; its transient peak (~27 MiB: the eager reference forward and the served
  clones) comes from the cache. A replica keeps its LAST eager reference forward's activation
  stash (`ExtractorStashes`; 17 MiB at 255 rows) until its next forward — bounded, replaced not
  added.
- `SnapshotPool`'s default device is the CPU, and `snapshot_pool_device_test` pins EVERY construction
  in `src/` and `tools/` (tests included) to it with one AST scan. The sizing harness's own pool was
  missed by the first fix and died on the runtime refusal (fixed in `95af710e`). The one declared
  exception was the python env core's worker pool, which inferred on the snapshot itself on the device
  `--self-play-use-cpu` chose (both deleted: the core in U3, the flag in P11).

## The update fit check (`gen3_update_fit_v1`) and fitting N = 256

**The check — MEASURED, not a lower bound** (`agents/training/update_fit.py`; it replaced
`cuda_ledger.check_device_batch_fits`, whose sum of startup rows passed N = 256 with the X26 heads and
then OOMed in update 1). At the end of startup, on both paths, after the ledger — with T2, the compiled
regions and the declared optimizer state all on the card — `dry_update` runs ONE REAL `train()` (one
epoch: the production code path unchanged, R1's compiled micro-step at the production micro size, the
eager tail, backward, accumulation, clip and optimizer steps, the micro-batch staging, the first-update
diagnostics, and K9(b)'s behaviour PROBE under the Rust core) over a FIXTURE rollout of the buffer's full
declared shape (the K9 learner golden's real labelled rows, tiled; the fixture's stored log-probs are the
policy's own, so the probe judges it correctly, warn-only, its dump in a temp dir), then restores the
learner EXACTLY — modules and optimizer state in place, the python / numpy / torch / CUDA RNG streams,
every instance attribute of the model and its modules, the logger. The run after it is bit-identical to
one without it (`update_fit_test`, on the golden: weights and logged losses). Verdict, deterministic:

    headroom = (reserved_now + device_free_now) - demand (the dry update's peak reserved) >= 1,024 MiB

1,024 = K6's 512 MiB ceiling margin + the sizing study's D-6 rule (ceiling − demand ≥ 512), so a pass is
D-6 at the first update. Below it, or an OOM inside the dry update, is `UpdateWontFit` (FATAL_CONFIG),
naming the levers; `<run_dir>/update_fit.json` keeps the reading and each `train()` segment's peak
(`phase_peaks`, booked by a `phase_hook`). Cost: one epoch, 13–17 s at startup (measured 12.7–17.4 s,
N = 48 and 256). `GEN3AI_UPDATE_FIT_SNAPSHOT=<path>` dumps the allocator's history of the dry update.
🚨 The dry update calls `model.train()` — the SAME method a benchmark worker replaces. A tool that swaps `train`
process-wide (`learner_benchmark`, `main.compile_inventory`) would take this startup call as its measurement, so
both install `learner_benchmark.learn_loop_only`: the tool is live only from the learn loop's first collection
(`designs/ops/testing.md`).

**PREDICTIVE — measured against the real first update in the same process** (RTX 3080 Ti 12 GiB, torch
2.8, fp32, rust core, `--arch production`, the X26 heads incl. `--ridealong-rnd-variants all`):

| run | dry: peak alloc / reserved | real first update: peak alloc / reserved |
|---|---|---|
| N = 48, resident batch | 7,424 / 8,566 MiB | 7,230 / 8,570 MiB |
| N = 256, all levers | 6,426 / 7,844 MiB | 6,230 / 7,910 MiB |
| N = 256, heads OFF, all levers | 6,386 / 7,814 MiB | 6,190 / 7,880 MiB |

Those reserved gaps (+66 MiB on the real update) were the staged batch's per-update SIDE STREAM, not
eval cycles: each update's new stream stranded its micro-batch cache (next section). With the copy on the
compute stream the dry update's demand IS the steady state's — 7,768 MiB predicted, 7,768 MiB at every one
of 40 real updates (N = 256, heads OFF; below). Allocated ~+195 MiB conservative. Before the probe was in the dry update it
UNDER-predicted by 1.37 GiB: the probe WAS the peak (below).

**The old N = 256 shape is refused at startup.** N = 256 + the X26 heads, resident batch, T2 uncapped:
demand 10,650 MiB against a 10,702 MiB card → headroom 51 MiB → `UpdateWontFit` (exit 3), where the
2026-10-01 launch had OOMed in update 1.

**The levers that fit N = 256 — all numerically identical** (no learning or recipe change; the trainee's
T2 rows replay at the same bucket as before, so its log-probs are bitwise unchanged):

| lever | what | measured saving |
|---|---|---|
| K9(b) probe keeps NOTHING for backward (`gen3_probe_releases_graph_v1`) | `consistency.behaviour_probe`'s eager grad-mode forward of one micro-batch (2,048 rows) ran BEFORE the epoch loop and its saved activations were the update's PEAK; it now runs under `saved_tensors_hooks` that drop them (grad mode, kernels and values unchanged — bitwise, `update_fit_test`), and the stashes that held its graph are released | N = 48 real first update 8,790 → 7,230 MiB allocated, 9,550 → 8,570 reserved |
| `--device-batch staged` (DEFAULT, `gen3_device_batch_mode_v1`) | each micro-batch gathered on the host by a prefetch thread, copied non-blocking on the COMPUTE stream (`gen3_staged_compute_stream_v1`; it was a per-update side stream, below); the resident K8.6 copy of the whole rollout is gone | −1,088 MiB of update peak (1,135 → 47 MiB on the card) |
| T2 per-slot bucket caps (`gen3_slot_bucket_caps_v1`, `--t2-opponent-bucket-cap 64`) | each lane's CUDA-graph pool is sized by its largest capture: uncapped at N = 256, 8 lanes × 232 MiB; capped, only the trainee's lane captures 256 (others ≤ 64, a 60 MiB pool) | T2 startup row +1,399 / +3,924 → +850 / +2,112 MiB (alloc / reserved) |
| T2 slots hold no ride-along heads (`gen3_opponent_inference_load_v1` + `served_replica`) | 18.2 MiB per slot at the X26 surface, never copied | T2 row +1,653 → +1,399 MiB allocated (heads then without RND variants) |

**The caps keep T2 bit-faithful — on-GPU parity at N = 256** (deferred descriptor, 2026-10-02, `~/gen3ai_archive/memfit/t2_caps_n256.json`, 31 slots, 8 lanes, the X26 heads):

| buckets | slot × bucket checks | max \|Δlog π\| | max \|ΔV\| | failures | T2 reserved | trainee lane pool | an opponent lane pool |
|---|---|---|---|---|---|---|---|
| capped (8, 64, 256), opponents ≤ 64 | 376 | 1.31e-6 | 7.2e-7 | 0 | 2,168 MiB | 1,516 MiB | 64–180 MiB |
| uncapped (8, 256) | 372 | 1.67e-6 | 7.2e-7 | 0 | 3,834 MiB | 2,336 MiB | ~210 MiB |

The staged batch's CUDA tests pass on the GPU on the compute-stream code (2 passed, `~/gen3ai_archive/staged_stream/cudatests_new.log`). The sizing study's arms B / C (N = 256, `7ef99979`) started clean through every startup gate and showed the climb the next section fixes.

**Deferred descriptors (memfit chain, 2026-10-02 01:43–02:46, `~/gen3ai_archive/memfit/status`; CPU-CONTENDED — the sizing study's meter queue ran until 02:33, so none is a clean timing):**

- **Staged vs resident update time.** At N = 48 with the X26 heads, 4 updates each. The first update and the one straddling an eval are dropped, so n = 2 for each.
  - `--device-batch staged`: 41.2 / 42.8 s.
  - `resident`: 44.1 / 46.7 s.
  - So staged is NOT slower. With n = 2 under contention, that is a direction, not a measurement.
- **Uncapped T2 at N = 256 + heads (`--t2-opponent-bucket-cap 0`):** REFUSED at startup by `UpdateWontFit` (exit 3), as designed. No throughput A/B was possible on that shape.
- **The uncontended rerun of N = 256 + heads + promotions (`ship_n256h_b`):** hit its 1,200 s hold after 2 updates (exit 124), because it was contended. It is NOT a read. The fix's own acceptance run (next section) supersedes it.

Where the peak lives now (dry update, per segment, N = 48 resident): the per-term noise-scale probe
7,424 MiB and the once-per-call grad-balance / rank probes 7,382 — first-update DIAGNOSTICS, ~1.4 GiB
above the micro-step's backward (5,997) and forward (5,476). A plain update peaks ~1.5 GiB lower (N = 256:
4,730 vs 6,230 MiB allocated).

**The micro-step's activation footprint is the model's real size** (allocator snapshot of a micro-step,
2,048 rows): ~2.5 GiB of forward activations saved for backward (the entity trunk's 61-token × 128-wide
tensors, 61 MiB each at 2,048 rows, ~40 of them; SDPA ~0.66 GiB; the damage op's pairwise status /
paths tensors ~0.34 GiB) and ~1.9 GiB of backward working set (fused reductions, the embedding-backward
buffers). Nothing found duplicated or held past its use inside R1. Recomputation (activation
checkpointing) would trade memory for a different compiled program — not a fit lever.

**N = 256 + the X26 heads, fitted** (all levers, `--self-play-start-wr 0 --promote-threshold 0`, an eval
every rollout, 5 updates, `~/gen3ai_archive/memfit/`): startup headroom 2,683 MiB; 5 evals, 4 promotions
each loaded into its T2 slot, exit 0; first update 6,230 / 7,910 MiB, the plain ones 4,73x MiB allocated;
K6's trend `OK @ update 4: demand 7.85 GiB, ceiling 8.80 GiB`; D-6 at update 4: ceiling 9,010 − demand
8,040 = 970 MiB ≥ 512 — with ~1.3 GiB of the card held by four detached ladder updaters at the time
(below). gnome-shell's 491 MiB was NOT freed for any of this.

**FINDING — the promotion's ladder updater was on the GPU** (`gen3_ladder_off_gpu_v1`). Each promotion
spawns `agents.training.snapshot_ladder` detached; it plays on the CPU but its imports opened a CUDA
context, 330 MiB of the training card each, and each outlived its promotion by 15+ min (four alive at
once at a promotion per rollout). It is now spawned with `CUDA_VISIBLE_DEVICES=""`. The startup fit
check cannot see a process spawned later; K6's trend is what watches that.

## The staged batch's stream — the after-startup climb, and the counters that see it (`gen3_staged_compute_stream_v1`, `gen3_reserved_after_freeze_v1`)

**The finding (sizing arm B, 2026-10-01: N = 256, 10 epochs, heads OFF, pin `7ef99979`).** RESERVED
climbed +66 MiB per update from the startup fit's 7,814 MiB to 9,674 MiB over updates 1–31, then stayed
flat for the remaining 51 updates; device free fell 2,515 → 854 MiB and D-6 failed at steady state
(`[CudaMemTrend] WARN` from update 30, headroom 341.5 MiB). The allocated peak (4,724 MiB) and the
quiescent floor (~980 MiB) never moved, `cuda_ooms` = `alloc_retries` = 0, and `segments_after_freeze`
read 0 the whole run. Arm C (5 epochs) climbed the same +66 MiB per update to the same 9.45 GiB demand at
update 32, so the growth was per `train()` call, not per epoch; evals (every ~20 updates), promotions
(first at ~6.0M steps) and the rank probe (every 10th update) do not line up with it.

**The cause.** `device_batches._StagedGather.__init__` built a NEW `torch.cuda.Stream` for the side-stream
copy at every `train()` call. torch hands streams out round-robin from a pool of 32 per priority. The
launcher runs every child under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(`main/launcher/child.py`), where the caching allocator keeps a segment PER STREAM and a cached block
serves only its own stream, so each new stream stranded its in-flight micro-batch blocks (+46–66 MiB)
until the pool wrapped: 31 real updates + the startup dry update = 32. Every segment counter read 0
because `memory_stats` does not count expandable segments.

**The fix: no side stream.** The copy runs non-blocking on the COMPUTE stream. The side stream bought no
overlap: it waited on the compute stream before every copy, and the compute stream waited on it after.
The micro-batches are bit-identical (a gather is exact; `instrumented_ppo_device_batches_test`). The class
fix is the static gate's `cuda_resource` kind (`src/learner_lifecycle_gate_test.py`): a `torch.cuda`
stream, CUDA graph, capture, graph pool or `MemPool` built in a training-step path — the learner, the T2
service, the rust collector / eval core / env — FAILS outside `@startup_builder` / `_setup_model` /
`_build`. A bare `__init__` does NOT exempt it, because the leak sat in a per-update helper's `__init__`
(T2's `Engine.__init__` and `_build_graphs` are declared `@startup_builder`).

**Measured, A/B** (RTX 3080 Ti, torch 2.8, fp32, rust core, `--arch production`, N = 256, n_steps 48,
10 epochs, no evals, 40 updates; run dirs `~/gen3ai_archive/staged_stream/runs/ss_ab_{old,new}_*`):

| arm | startup fit demand | update peak reserved, updates 1 → 40 | device free | streams |
|---|---|---|---|---|
| OLD (`0aad283f`, per-update side stream) | 7,814 MiB | 7,880 → 9,674 (+66/update to update 31, then flat) — B's series to the MiB | 2,515 → 854 | not counted |
| NEW (compute stream) | 7,768 MiB | **7,768 at every update** | 2,760 flat | 9 flat |

The fit check's prediction equals the steady state exactly. D-6 at steady state (heads OFF): ceiling
10,016 − demand 7,768 = 2,248 MiB, against B's 341.5.

**Acceptance — N = 256 with the X26 heads ON** (`--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5
--ridealong-opp 5 --ridealong-rnd-variants all`), production shape (n_steps 384 = 98,304 rows, 10 epochs,
the production eval cadence: evals at 2.0M and 4.0M steps, one promotion loaded into its T2 slot),
44 updates, one process (`~/gen3ai_archive/staged_stream/runs/ss_accept_n256h_1790927726`): the fit
check predicted 7,798 MiB; the update and rollout peak reserved read **7,798 MiB at every update**
(1 → 43 dumped; the 44th is the undumped last update, TECH-DEBT §2(b)), device free 2,730 flat,
`reserved_after_freeze` 0, `streams_after_freeze` 0, 9 streams, no retries or OOMs. **D-6 at steady
state: 10,016 − 7,798 = 2,218 MiB** (≥ 512). **PROJECTED, not run:** before the fix the same climb
(+1,860 MiB, measured with the heads OFF) on this fit ends at ~9,660 MiB, ~360 MiB of D-6 headroom — a FAIL. gnome-shell was NOT freed. A pure-torch check (no repo code) shows the
mechanism alone: one new stream per update adds a segment and a stream to `memory_snapshot()` every update
while `memory_stats`' segment count reads 0; on the compute stream it holds one stream and 62 MiB.

**The counters (`cuda_memory_trend`, `gen3_reserved_after_freeze_v1`).** Every sample takes a SEGMENT
CENSUS from `torch.cuda.memory_snapshot()` (`segment_census`: segments, and DISTINCT streams owning one),
which fills the segment count when `memory_stats` reads none. Measured from the FREEZE sample (the warm-up
updates included — they are after the freeze too; `segments_after_freeze` used to start at the first
windowed sample):

- `reserved_after_freeze` — reserved grown since the freeze. Above `RESERVED_AFTER_FREEZE_WARN_BYTES`
  (128 MiB) while it is still RISING, a WARN: demand the startup fit never measured. The tolerance's
  cause: the dry update is one epoch and a real update ten; the measured excess after the fix is 0.
- `streams_after_freeze` — streams holding cache that the freeze did not have. Any rise is a WARN
  (tolerance 0: every stream is a declared startup acquisition).

Both WARN, never STOP (a bounded growth that stops is not a leak); B's series WARNs at the FIRST window
(update 4) and on every window until the climb ends (`cuda_memory_trend_test`), where it used to be
silent until headroom ran out at update 30. Both are written at every sample (the memory half, above).

**What is still not covered.** T2's own `InferenceService._frozen_guard` counts `segment.all.allocated`,
which is blind under `expandable_segments:True` the same way; it POISONS on a hit, and a snapshot per
flush is too slow, so its fix is a TECH-DEBT row, not this unit (`designs/ops/TECH_DEBT_BACKLOG.md` §2(b)).

## An opponent never trains, so an opponent load acquires nothing (`gen3_opponent_inference_load_v1`)

**The failure it closes (2026-10-01).** An X26 launch at N = 256 with the ride-along heads ON and
`--self-play-start-wr 0` died at its first pool seeding. `set_self_play_target` loaded the snapshot
through `load_model_snapshot` → `InstrumentedMaskablePPO.load` → `_setup_model` →
`_ridealong_acquire`, which built an Adam over the snapshot's 111 ride-along tensors and pre-stepped
it. The step pre-hook raised "an optimizer outside the frozen set STEPPED after the freeze", and the
sticky violation FATALed the rollout end (`~/gen3ai_archive/k6_k8/memaudit/launch_n256p.log`). The
eval sentinels (`rust_eval/launch.load_sentinels`, `eval_worker`) went through the same loader.

**The structural fix: opponents are a different CLASS.** `InferenceMaskablePPO`
(`agents/training/instrumented_ppo/inference.py`) builds the POLICY and nothing a learner needs:
- no policy optimizer at all: the policy is constructed with an optimizer class that builds
  nothing, and the checkpoint's saved Adam moments are dropped, never loaded;
- no ride-along optimizer: the heads' WEIGHTS still load, so an offline reader reads them, but a
  ride-along step raises `RideAlongLifecycleViolation`;
- no rollout buffer, no loop hooks.

It refuses `learn` / `train` / `collect_rollouts` / `save` (`InferenceOnlyModelError`). The rest is
`MaskablePPO._setup_model` in its own order, except `set_random_seed(self.seed)`: an opponent load does
not seed, and it builds the policy inside `global_rng_guard.isolated_global_rng()`, so every global RNG
stream is bit-identical before and after the load ("No global reseed after the freeze" below).

**The routes:**
- **`snapshot.load_opponent_snapshot`** — the self-play pool (`SnapshotPool.load_model`, and through it
  every T2 pool refresh) and the eval sentinels (`rust_eval/launch.load_sentinels`, `eval_worker`).
- **`snapshot.load_foreign_opponent`** — inference-only by default. That covers stable opponents,
  exploiter targets, `main.anchors`, `baselines.load` and
  the offline readers. `inference_only=False` is for an offline tool that FITS the loaded model: the
  consensus warm-start's student is the one caller.
- **Not this class, and not affected:** the prober, `play.py` and the search workers load with a bare
  sb3 `MaskablePPO.load`, which has no ride-along acquisition. None of them runs inside a frozen learner.

**Every load is STRICT (`gen3_strict_checkpoint_load_v1`, P10 F4, 2026-10-03).** sb3's `load` retries
with `exact_match=False` whenever the strict error mentions `pi_features_extractor` (its "SB3 < 1.7.0"
patch), and our extractor is registered under three aliases, so ANY missing extractor key named it: a
checkpoint with `alpha_head`'s keys deleted loaded as the trainee AND as an opponent with the head at
fresh init, behind one warning. `OwnedLoop.set_parameters` (under `InstrumentedMaskablePPO` and
`InferenceMaskablePPO` alike) always loads strictly — a MISSING and an UNEXPECTED key both raise — and
answers a non-strict request with `StrictLoadError` naming the strict error. No declared non-strict
exception exists (the deleted-kwarg sanitizers pop only kwargs that build no parameters and refuse
the rest); a future one is declared there, with its reason. The bare sb3 `MaskablePPO.load` callers
(the prober, `play.py`, the search workers, a few offline tools) still carry sb3's retry.

**F-MEM: an opponent may differ from the trainee in the ride-along keys alone.** The gate is
`ModelVersion.check_opponent_snapshot_compatible`: `check_compatible` with the declared key set
(`ridealong_heads.RIDEALONG_FLAGS`, the one source of truth) ignored, in either direction. Every other
mismatch is still a `ModelVersionError`. The T2 slot identity ignores the same set:
`slots.served_state_dict` skips every `ridealong.*` tensor, `forward_fingerprint` skips the
`ridealong_*` extractor kwargs, and a served replica drops the heads, so no slot stores, copies or
compiles them.

**The trainee stays strict.** Its resume or fork (`load_model_snapshot` with its env) is the full
`InstrumentedMaskablePPO`. It acquires every ride-along optimizer in `_setup_model`, before the
freeze, and it is checked on every key: a resume that dropped a head would silently delete a trained
baseline.

**Proof on the real path (2026-10-01, RTX 3080 Ti, torch 2.8, rust core, production, the X26 heads +
`--ridealong-rnd-variants all`, `--self-play-start-wr 0 --promote-threshold 0`, an eval every rollout;
run dirs + logs in `~/gen3ai_archive/k6_k8/oppload/`):**
- At N = 48, n_steps 2,048, 3 rollouts: three promotions, each followed by a logged
  `🧠 [SELFPLAY] pool snapshot loaded INFERENCE-ONLY`, and the pool snapshots also played as eval
  sentinels. `[LEARNER FREEZE] released — 6 checks passed`, `Training complete`, exit 0. The update
  peak was 8.79 GiB allocated / 9.53 GiB reserved (`train/cuda_update_peak_*`).
- At N = 256, n_steps 384, the run passed the pool seeding and its `set_self_play_target` load, the
  step that FATALed before the fix, with no lifecycle FATAL. It then ran OUT OF MEMORY in update 1's
  R1 micro-step (10.21 GiB allocated), which is the known N = 256 OOM above and a separate unit. With the
  heads out of the T2 slots, the ledger's rust-env-core row read 1,455 MiB allocated, against 1,685 MiB
  before the fix without the RND variants.

**Tests:** `opponent_inference_load_test`. Each test fails on revert: a pool / sentinel / foreign load
under a FROZEN trainee builds no optimizer; both F-MEM directions load; a non-ride-along mismatch is
refused; the trainee resume is strict and acquires at setup; the T2 identity ignores the heads.

## No global reseed after the freeze (`gen3_no_global_reseed_v1`)

**The failure it closes (2026-10-02, the deletion-pass manifest's P1).** sb3's `_setup_model` calls
`set_random_seed(self.seed)`. So every model load re-seeded Python `random`, NumPy and torch to the
LOADED checkpoint's saved seed. A snapshot carries its run's `--seed`, so a pool refresh, an eval
sentinel or an exploiter rung rewound the loading process's global streams to their startup state.
It was measured with a probe that wrapped every global seed / draw function and hashed the three
states at every rollout and update boundary (CPU `--debug --arch production`, forced promotions, one
run per env core):

- **The Rust core (production).** The trainee process's one steady-state global-RNG consumer is the
  PPO minibatch permutation (`np.random.permutation` in `RolloutBuffer.get`, one per epoch). Python
  `random` and torch's CPU stream are never read: their hashes are constant for the whole run, and
  teams come from per-env seeded builders (`rust_rollout/teams.py`). After each pool load
  (`rust_rollout/build.py`) and each sentinel load (`rust_eval/launch.load_sentinels`), the NumPy state
  at update entry EQUALED update 1's. Updates 8–15 replayed updates 1–8's permutations, and 16–23 did
  again. Which rows train, and how often, is unchanged, so the expected effect on learning is nil. The
  M5 sizing arms (A / A2 / A′ / B / C) had it from their first pool seed on.
- **The Python core** (every pre-M5 self-play run; DELETED in U3 — recorded as history). Each `SubprocVecEnv` worker loaded its pool
  snapshot once per generation (`MaskableAgentWrapper._ensure_pool_model`). Its teambuilders and the
  heuristic bots draw from that worker's global `random`. After the load the worker's team draws
  replayed the run's first draws exactly (25, 281, 142, 104, 558, 89, 32, 30, …), and every worker
  replayed the same stream. On the real runs, per-team trainee game counts (`metadata.json`
  `team_win_rates`) are over-dispersed: variance / mean 20–1,264 on the python core against 1.7–2.0 on
  the Rust sizing arms. N0 (`ai_v14_01_base`) spans 166–5,676 games per team where uniform draws give
  2,030 ± 45. Which teams are heavy is fixed by the snapshot seed: independent seed-1001 python runs
  correlate with N0 at Spearman 0.89–0.997, while seed-42 / seed-1002 runs and the Rust arms read
  about 0. The finding and the claims it may touch are in `designs/research_state/ledger.md`
  (2026-10-02).

**The fix: a load never touches a global stream.** `InferenceMaskablePPO._setup_model` does not seed.
It builds the policy inside `global_rng_guard.isolated_global_rng()`. That scope restores torch's CPU
generator, which a module's construction draws its init from (the loaded weights then overwrite it).
It also restores any stream SEEDED inside it, CUDA's included when initialized, so the ride-along
heads' constructor `torch.manual_seed` calls stay local. A stream only drawn from inside the scope is
left advanced, because restoring it would rewind another thread's draws.

**The guard: a global SEED after the freeze is FATAL.** `LearnerFreeze.freeze()` arms
`global_rng_guard`, and `release()` disarms it. The guard wraps the seeding functions on their modules:
`random.seed`, `numpy.random.seed`, `torch.manual_seed` / `torch.random.manual_seed` / `torch.seed`,
and `torch.cuda.manual_seed[_all]` / `torch.cuda.seed[_all]`. While armed, a call from the armed
process on a thread outside an isolated scope raises `GlobalReseedError`. That error is a
`FatalConfigError`, so the run exits FATAL_CONFIG and is not restarted. The error names the call site,
and the violation is appended to the freeze's sticky list, so a swallowed raise still fails the next
`check()`. There is no allowlist: a stream that must be reproducible owns its generator (`keyed_draw`,
a `random.Random`, a `torch.Generator`). The learner golden's harness pins its update seed BEFORE its
test freezes (`learner_golden.compute(before_train=...)`), and the K9 golden is unchanged.

Its limits are deterministic and stated:
- A state RESTORE (`setstate` / `set_state` / `set_rng_state`) is not a seed and is not guarded.
- A name bound before arming bypasses the wrappers, so the static twin
  `src/global_rng_seed_gate_test.py` (routine, EMPTY allowlist) refuses `from random import seed` and
  every other seeding function bound by name or taken as a value in non-test `src/`.
- A forked child inherits the wrappers but not the arming (the PID is checked).

**Tests** (`global_rng_guard_test`, each failing on revert):
- every seeding spelling refused while armed, with its site, sticky;
- a seed inside an isolated scope is local, and an unseeded draw stays advanced;
- the freeze arms the guard;
- a pool / sentinel / foreign load leaves all three streams bit-identical (heads on and off);
- the post-load permutation is not the startup permutation;
- a heads-on pool load under a frozen learner raises nothing;
- the loaded weights are the saved ones.

## Smoke

`python src/main/train_rl_agent.py --debug --steps 3000` (2026-09-30, CPU): `🧊 [LEARNER STARTUP]
declared optimizer state: policy.optimizer +120` → `🧊 [LEARNER FREEZE] the first rollout of learn(): 93
modules, 120 parameters, 180 buffers, 1 optimizer(s) [policy.optimizer] with 120 state entries` →
`released — learn() returned; 4 checks passed` → `Training complete`.

## The static twin — `src/learner_lifecycle_gate_test.py` (routine tier, EMPTY allowlist)

An AST gate that fails any optimizer / `nn.Parameter` / `nn.Module` construction (torch or a repo
`nn.Module` subclass, resolved by name; every import spelling, `self.optimizer_class(...)`) inside a
TRAINING-STEP code path, unless the enclosing function is a `@startup_builder` or a class's
`__init__` / `_build` / `_setup_model`. The scope is DECLARED, not a call-graph walk: every function
in `agents/training/instrumented_ppo/`, the loss-term / probe modules the fold imports
(`STEP_MODULES`, each with its reason; a test fails when the fold starts importing an `agents`
module in neither `STEP_MODULES` nor `NOT_STEP_MODULES`, so the scope grows with the fold), and the
PER-STEP hooks of every SB3 callback under `agents/training` and `main/train` (`_on_training_start` /
`_init_callback` are startup). Blind spots, stated in the test: a helper outside the lists, a
`@startup_builder` called from a step path, a dynamically built class — the runtime freeze guard
covers each. Teeth: 24 violating snippets fail, 8 exemptions pass, and each scanner rule's removal
fails a teeth test. Landed with the ride-along builders marked `@startup_builder` (`_mlp`,
`build_rnd_variants`, `build_ridealong`, `_adam` — every caller is startup since `8812c565`).

**The `cuda_resource` kind** (`gen3_staged_compute_stream_v1`, 2026-10-01): a `torch.cuda` `Stream` /
`ExternalStream` / `CUDAGraph` / `graph(...)` capture / `graph_pool_handle` / `MemPool`, however spelled,
fails in the same scope PLUS `CUDA_SCOPE` (the T2 inference service, the rust rollout collector, the rust
eval core, the rust env's Python side — scanned for this kind only). Its exemptions are `@startup_builder`,
`_setup_model` and `_build` — NOT a bare `__init__`, because the per-update side stream that stranded
1.86 GiB sat in a per-update helper's `__init__`. `torch.cuda.Event` is not in the class (an event owns no
allocator state; T2 creates one per flush). Teeth: 9 violating snippets (the original leak among them),
4 exemptions, the CUDA-only scope.

## The ride-along heads under the guard

The ride-along heads (`--ridealong-*`) used to build their Adam LAZILY on the first update — the
case that made the owner ask for this guard. Since `8812c565` every head's optimizer (each RND
variant's included) is acquired at startup with its state pre-allocated, the late-build path is
deleted, and a missing optimizer is `RideAlongLifecycleViolation`. MEASURED under the guard
(2026-09-30, `--debug --arch production --ridealong-ensemble 3 --ridealong-rnd --ridealong-adv 4
--ridealong-opp 4 --ridealong-rnd-variants all`, CPU, torch 2.5.1): frozen at 366 modules, 362
parameters, 356 buffers, 6 optimizers (`model._ridealong_opt` + the five variants'
`model._ridealong_vopts[...]`, plus the policy's) — `released — learn() returned; 6 checks passed`,
`Training complete` (`~/gen3ai_archive/k6_k8/ridealong_smoke/run_stable.log`).
