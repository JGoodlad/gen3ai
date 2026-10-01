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
instance-attribute wrappers (`loop_hooks.install`). The `sb3_reference` seam applies the table too, so
both arms of the loop's equivalence A/B run under the same guards (`loop_hooks_test.py`).

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

**The declared table.** `compile_trainer.production_prewarm_calls` IS the declaration: every
compiled-learner signature (callable × batch × train/eval × grad) the steady state may reach, prewarmed
at startup (`arm_compile_sentinel`). Batch 1 is never in it: it always runs eager
(`gen3_batch1_eager_v1`, `compile_flags.md`). The compile sentinel then LOCKS right after the prewarm —
before the first real iteration (was: after the first `train()`, `8fc297a2`'s interim, which absorbed
whatever iteration 1 compiled). A signature outside the table is a typed FATAL (`[CompileSentinel]
FATAL`, exit FATAL_CONFIG) that NAMES the failing guard(s): 2.5.1's `RecompileError` lists them; on
2.8 the `fail_on_recompile` stance's rejection says nothing, so the sentinel replays the call once
under `error_on_recompile` to read them (`compile_control._diagnose_rejection`).

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
  chosen by `compile_regions.weights_regime`; on the legacy compile, the gate's probe loss.

The train-graph check now runs at EVERY canary (`GRAD_EVERY` 1). It used to run every 4th. It costs
0.65–0.71 s at the production shape: arm C's weights, CUDA, `n_envs` 48, B = 2048, R1 through the compiled region and eager. The decision readout alone costs 0.06–0.09 s. Against a 36 s update every 100 updates that is under 0.02% (`~/gen3ai_archive/k6_k8/r1bar/canary_cost.log`, 2026-10-01).

The bars are the startup gate's. Under TF32 that is the TF32 rule against an EAGER fp32 reference; the
gate's compiled-at-fp32 arm would be a separate graph, i.e. an undeclared signature after the lock.
Live weights are trained, so vacuity is not a refusal here.

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
  ceiling, slope, updates to the ceiling, reserved, device free, segments, retries, OOMs); a WARN
  that changes between closes is printed when it happens, and a changed WARN or the STOP also goes
  to the launcher's event channel;
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

Per update:
- An update PEAKS at **8.75 GiB allocated** and 9.49 GiB reserved. That is the same on a diagnostics
  update and a plain one.
- **1.14 GiB** of that peak is the device-resident batch (K8.6).
- The quiescent floor is 851–855 MiB.

**Declared slot loads.** A rust-core pool refresh is a DECLARED LOAD into its T2 slot.
- The pool loads snapshots on the CPU. On the card, each promotion's snapshot stayed in the pool's
  LRU (cap 3) beside the slot T2 had copied it into: +~33 MiB of quiescent floor per promotion on
  sizing arm A.
- A pool weight source found on the card is a typed `LazyAcquisitionError`.
- Every route's slot load goes through `rust_rollout.build.checked_slot_load`, which refuses a load
  that leaves more than 1 MiB newly allocated (`declared_slot_load_test`).
- `SnapshotPool`'s default device is the CPU, and `snapshot_pool_device_test` pins EVERY construction
  in `src/` and `tools/` (tests included) to it with one AST scan. The sizing harness's own pool was
  missed by the first fix and died on the runtime refusal (fixed in `95af710e`). The one declared
  exception is the python env core's worker pool, which infers on the snapshot itself on the device
  `--self-play-use-cpu` chose.

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
