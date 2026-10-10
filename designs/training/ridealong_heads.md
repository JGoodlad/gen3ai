# Training — the detached ride-along heads, and STRICT checkpoint loading

> Lifted out of `src/agents/training/CLAUDE.md` on 2026-10-10. Always-current, like the leaf: update it in the same
> pass as the code. Lifecycle detail: [`learner_lifecycle.md`](learner_lifecycle.md).

## The DETACHED RIDE-ALONG heads (`--ridealong-ensemble` · `--ridealong-rnd` · `--ridealong-adv` · `--ridealong-opp` · `--ridealong-rnd-variants`)

`gen3_ridealong_heads_v1` (v126; `instrumented_ppo/ridealong_terms.py`, heads in
`agents/model/ridealong_heads.py`). Baselines that OBSERVE: a V ensemble (epistemic uncertainty),
RND novelty on the raw observation, and the Q = V + A + B main effects on PPO's own labels.
`_ridealong_update` runs right after each minibatch's `evaluate_actions` and BEFORE PPO's loss is
built: detached stashes → the heads' losses → their own backward, clip and Adam step → their grads
back to None. So it sits OUTSIDE the fold order above, and nothing in the fold can see it.
`ridealong_update_test` pins one real update ON vs OFF as bit-identical (params, PPO optimizer
state, PPO scalars, RNG). Two things to know when reading `ridealong/*`. **The heads TRAIN and READ
on epoch 0 only** (`RIDEALONG_EPOCHS` = 1: each rollout row seen once, scored before the heads train
on it). All 10 epochs cost +13 % of a GPU update on the learner benchmark; one pass MEASURED +0.59 s = 0.88 % of a 67 s update (`ridealong_step_benchmark.py`; X26's
`PREREGISTRATION.md` "Overhead").
And **the heads' Adam state is not checkpointed** (a restart resumes their weights with a fresh Adam: `_setup_model` re-acquires it). 🚨 **Nor is it SAVED or UNPICKLED any more** (`gen3_cpu_load_no_cuda_v1`): older checkpoints pickled the optimizers (CUDA tensors) into sb3's `data`, and a CPU READ of one created a CUDA context. `strict_load.NEVER_UNPICKLED` is excluded from every save and substituted on every load (`StrictCheckpointLoad.load`), and a CPU load that initialises CUDA anyway raises `CudaContextOnCpuLoad` (`designs/training/learner_lifecycle.md`).
**Every ride-along optimizer is ACQUIRED AT STARTUP** (`RideAlongTerms._setup_model` →
`_ridealong_acquire`, Adam state pre-allocated, bit-identical to lazy init). There is NO lazy build:
a step that finds an optimizer missing, or bound to other heads, raises
`RideAlongLifecycleViolation`, as K6.1's freeze guard would. Tooling that swaps `policy.ridealong`
must call `_ridealong_acquire()` again (both benchmarks do). 🚨 **Only the TRAINEE acquires: an
OPPONENT load acquires nothing** (`gen3_opponent_inference_load_v1`). The self-play pool and the eval
sentinels load through `snapshot.load_opponent_snapshot`, and every `load_foreign_opponent` (stable
opponents, exploiter targets, `main.anchors`, the offline readers) is an
`InferenceMaskablePPO` by default: policy weights only, with no optimizer, no ride-along optimizer and
no rollout buffer. It refuses `learn` / `train` / `save`. Before this fix a pool load after the freeze
pre-stepped a ride-along Adam, and K6 FATALed the X26 launch at its first pool seeding. An opponent may
differ from the trainee ONLY in the declared ride-along keys (`RIDEALONG_FLAGS`), in either direction
(F-MEM); the trainee's own resume stays strict. 🚨 **Every checkpoint load is STRICT on state-dict
keys** (`gen3_strict_checkpoint_load_v1`): `StrictCheckpointLoad.set_parameters`
(`instrumented_ppo/strict_load.py`; `OwnedLoop` inherits it, so the learner and the opponent classes
carry it) refuses sb3's non-strict "SB3 < 1.7.0" retry (`StrictLoadError`), which used to load a
checkpoint missing an extractor submodule with that submodule at fresh init. The READERS (the ladder
session `play.py`, the prober, the offline meters, `winprob_finetune`) load through
`agents.model.snapshot.load_checkpoint_strict` — a `StrictMaskablePPO`, a plain `MaskablePPO` with only
that `set_parameters` — and `src/strict_checkpoint_load_gate_test.py` fails a bare `MaskablePPO.load` /
`PPO.load`, an `exact_match=False`, or an sb3-algorithm subclass without the mixin (EMPTY allowlist).
Detail: `designs/training/learner_lifecycle.md`. The step is
K8's candidate compile region R-ride; it stays eager. **The RND variants**
(`--ridealong-rnd-variants all`, v127: `fast` / `decay` / `small` / `feat`, beside the unchanged base
RND) each step on their own Adam after the four heads. A non-finite variant disables ITSELF
(`ridealong/rndv_<name>_disabled`). `decay`'s pull toward init runs once per update, at the first step
(the accumulator marks the boundary). Their series are `ridealong/rndv_<name>_*`: loss, error mean /
median / IQR / `rel_spread` (saturation), z by class, the V-error meters, and `*_ident_ratio` (block
chimera ÷ real-row error, the identification monitor). Base logs the same `rnd_err_median` /
`rnd_err_rel_spread` / `rnd_ident_ratio`. With every variant on, the heads add 0.97 s
(1.45 %) to a 67 s update (`designs/research_state/measurements/ridealong_baseline/overhead_variants_2026-09-30.json`). B trains
only where `--opp-intent-coef > 0`, because that is what aligns the one-ahead opponent labels. The
meters are disagreement / novelty vs |V − z| (`*_auroc_err`, `*_spearman_err`, top vs bottom decile),
`rnd_z_<class>`, `adv_corr_logit`, `adv_std_starved` vs `adv_std_fed`, `q_out_of_range`. These are
MONITORING (rows the heads just trained on). The verdicts are the offline reader's,
`python -m main.ridealong_read` (`src/main/ridealong_read/`: CPU forwards on the Lane S bank and the
X4 pre-read truth turns; `--fresh-heads` reads the same checkpoint's untrained floor; every RND variant
and the amended comparisons (a)–(e); `--run-checkpoints <run>` reads a run's retained checkpoints as a
series), plus
`python -m main.ridealong_read.rnd_states` for the state-level RND reads. The pre-registered run is
EXPERIMENT_BACKLOG X26.
