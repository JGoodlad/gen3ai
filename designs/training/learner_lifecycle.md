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
| stochastic | CUDA memory | a LEAK DETECTOR, warn by default; stops (checkpoint, then a typed FATAL) only when a sustained trend projects an OOM inside a declared horizon (orchestrator, 2026-09-30: the caching allocator legitimately steps up from fragmentation, larger shapes and eval cycles, so a one-off crossing would eventually kill a healthy multi-day run) |

## The freeze guard (`gen3_learner_freeze_v1`)

**When.** `learner_lifecycle.attach(model)` wraps `collect_rollouts` / `train` / `learn` as instance
attributes (outermost, over the compile sentinel's). The FIRST `collect_rollouts` entry FREEZES —
startup is then over: the trainer compile and its parity gate, the prewarm, `declare_learner_startup`
and `learn()`'s own `_setup_learn` / every callback's `_on_training_start` have run. Every
`collect_rollouts` and `train` exit CHECKS. `learn()`'s exit RELEASES (the in-process final evaluation
after training is not the steady state). The wrappers and `_learner_freeze` are in
`_excluded_save_params`, so no checkpoint carries them.

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

## Smoke

`python src/main/train_rl_agent.py --debug --steps 3000` (2026-09-30, CPU): `🧊 [LEARNER STARTUP]
declared optimizer state: policy.optimizer +120` → `🧊 [LEARNER FREEZE] the first rollout of learn(): 93
modules, 120 parameters, 180 buffers, 1 optimizer(s) [policy.optimizer] with 120 state entries` →
`released — learn() returned; 4 checks passed` → `Training complete`.

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
