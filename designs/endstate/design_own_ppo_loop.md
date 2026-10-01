# Design — owning the PPO loop (SB3 / sb3-contrib off the production path)

🚨 **ALWAYS-CURRENT (an `endstate/` doc).** A build that differs from this plan updates it in the same
commit, saying what changed and why. Scoped 2026-10-01 at the owner's request: *"Let's own the PPO loop
and do that before the cutover if it isn't too hard"*, then *"Do the PPO pay down, as long as it is
less than 2 agent days"*.

**Status:** **STAGE 1 BUILT 2026-10-01** (`gen3_owned_ppo_loop_v1`, `instrumented_ppo/loop.py`):
SB3's loop is off the production path, and equivalence is held as identity (§4: E0, E1, E2, and E3
on the Rust core; the readout is
[`research_state/measurements/own_ppo_loop/`](../research_state/measurements/own_ppo_loop/README.md)).
Stage 1 was scoped at 1.6 agent-days (range 1.3–1.9), under the 2-day bar, and was GO.
**STAGE 2 BUILT 2026-10-01** (owner-approved in full). It shipped as three units:
- the eval-dump KL-skip fix (a regime boundary for runs with a live controller);
- the declared hook table (identity);
- owned seeding (a nominal regime boundary on CUDA).

It lands before the X26 ride-along baseline. Stages 3 and 4 run after the switch.

---

## 0. The answer in one paragraph

Most of the PPO loop is **already ours**:
- **The update.** `train()` is a vendored, hash-pinned copy of `MaskablePPO.train`. Every loss term is
  ours, and fold steps 1–3a are the functional region R1.
- **Masking** is functional (`masked_categorical.py`).
- **The micro-batch** is device-resident.
- **Rollouts on the Rust core**, GAE included, are collected by Lane G's complete-game collector.

What SB3 still **executes** each iteration:
- `learn()`, which is about 35 lines;
- `_setup_learn`, `dump_logs`, `_update_current_progress_remaining`, `_update_learning_rate` and
  `_update_info_buffer`;
- the Python core's `collect_rollouts`;
- the rollout buffer's `reset` / `add` / `get` and its window-mode GAE;
- `CallbackList` dispatch;
- `Logger.record` / `dump`.

What SB3 provides **at rest**:
- the constructor and `_setup_model`, including `set_random_seed` and the policy's ortho-init;
- the `.zip` save and load, which about 30 readers depend on;
- the policy base classes and the optimizer construction;
- the VecEnv base classes, `Monitor` and `BaseCallback`.

**Stage 1** moves SB3's control flow into code we own, as a declared phase table. It keeps
bit-identical behaviour, the `.zip` format, the policy `nn.Module`, the callbacks and the logger.
**Stage 2** makes the hooks explicit and fixes a real defect the inventory found (§2.1). **Stages 3–4**
retire the buffer, the logger and the VecEnv glue during the deletion pass, and then, as a separate
later decision, the checkpoint format and the dependency.

**Push-back on "before the cutover".** Two things are true about the ordering:
- **Stage 1 must land before the DELETION PASS.** Its strongest equivalence check (§4, E2) replays the
  same recorded games through the Python env core, and the deletion pass removes that core.
- **The switch should NOT wait for stage 1.** It is a default-flag flip that keeps both cores, and the
  sizing study's arms run pinned on today's loop. Stage 1 is identity by construction (§4), so their
  verdicts carry over.

Making the switch wait would buy nothing.

---

## 1. Inventory — every SB3 / sb3-contrib touchpoint on the production path

sb3 and sb3-contrib are **2.8.0** in both `gen3ai_torch28` and `gen3ai_stable` (verified 2026-10-01).
Risk is the risk of replacing a touchpoint without changing behaviour.

### 1.1 The algorithm and its control flow

| touchpoint | where | what it does for us | risk |
|---|---|---|---|
| `MaskablePPO.learn` | sb3_contrib `ppo_mask.py:428`. Called at `main/train/model_build.py:786` (resume) and `:991` (fresh) | The outer loop: `_setup_learn` → `on_training_start(locals(), globals())`, then repeat (collect → `iteration += 1` → progress → `dump_logs` → `train()`) → `on_training_end` | **low**. 35 lines, and its ORDER is load-bearing (§2.2) |
| `BaseAlgorithm._setup_learn` | sb3 `base_class.py:383` | `start_time`; `ep_info_buffer` (deque 100); `total_timesteps += num_timesteps` on a resume (the 2026 budget-doubling incident, `model_build.py:779`); env reset when `_last_obs` is None; callback init | **low**. Vendor it |
| `OnPolicyAlgorithm.dump_logs` | sb3 `on_policy_algorithm.py:277` | `time/fps`, `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `time/*`, then `logger.dump(step=num_timesteps)` | **low**. The tag names are read by `killbar`, `tb_read`, `restart_startup`, `stall_exhibit`, the launcher's `format.py` and `plot_tb` |
| `MaskablePPO.collect_rollouts` (Python core) | `ppo_mask.py:181`, reached via `RolloutProbes.collect_rollouts` (`instrumented_ppo/rollout_probes.py:32`) | Step the VecEnv, `update_locals(locals())` **before** `rollout_buffer.add`, the truncation bootstrap, then GAE | **med**. `WinProbLabelCallback` and `DenseAuxLabelCallback` read `buf.pos` in `on_step` as the row about to be written. Vendor it with its LOCAL NAMES unchanged |
| async collector | `agents/training/async_vec_env.py:226` | `--async-rollout`'s own loop, already ours. It still calls SB3's GAE | low. Deleted with the Python core |
| Rust collector | `rust_rollout/collector.py:327` | Already ours: host steps, `on_step` with a fixed locals dict, `_update_info_buffer`, filling SB3's buffer in place | low |
| `train()` | `instrumented_ppo/ppo.py:134` | Already ours (vendored, upstream hash pinned) | — |
| `_update_learning_rate`, `self.clip_range(progress)` | SB3 base | Applies `lr_schedule(progress)` to the optimizer every `train()`; records `train/learning_rate` | **low**. `lr_schedule` is a contract with the KL controller (§1.4) |

### 1.2 Buffer, GAE and advantages

| touchpoint | where | what it does | risk |
|---|---|---|---|
| `MaskableDictRolloutBuffer` | built by `_setup_model`; rebuilt at a new size by the Rust collector (`collector.py:395`) | The container: `[n_steps, n_envs]` arrays, `reset` / `add`, and `get(batch_size)` (one `np.random.permutation` per epoch, flatten-swap) | **med**. It is the contract between ~10 writers and `train()`; the K9 golden and `device_batches` are keyed on it |
| `compute_returns_and_advantage` | SB3's stock collect; async `:284`; `rust_rollout/store.py:393` (window mode); PBRS and frozen-φ re-run it | Window GAE | low. The complete-game path already uses our `store.game_gae`, which repeats SB3's arithmetic operation for operation |
| advantage normalisation | `micro_step.py:227` (ours) | Per micro-batch | — |
| the post-collect window | `rollout_probes.py` (PBRS, frozen-φ), WinProb `on_rollout_end` (labels, λ-returns), Fork `on_rollout_end` (appends rows), ValueSidecar | Everything that rewrites the buffer between collection and `train()`. **The order is a contract:** WinProb before ValueSidecar and before Fork, both enforced only by list order in `main/train/callbacks.py` | **med** |

### 1.3 Policy, init and optimizer

| touchpoint | where | what it does | risk |
|---|---|---|---|
| `MaskableMultiInputActorCriticPolicy` (→ `ActorCriticPolicy` → `BasePolicy`) | parents of `Gen3DualHeadMaskablePolicy` (`agents/model/policy.py:90`) | `mlp_extractor`, `value_net`, `_build` (ortho-init `apply` over every Linear, gain √2, and the optimizer), `obs_to_tensor`, `predict`, `set_training_mode`, state_dict layout | **high to replace** because of init order and the global-RNG draw sequence (`project_sb3_ortho_init_clobber`; our `restore_identity_init` repairs the zero-inits after it). The K9 golden's `init_params_sha256` pins it. **Kept in stages 1–3** |
| optimizer | `policy._build` rebuilds `optimizer_class(self.parameters(), lr=lr_schedule(1), **optimizer_kwargs)` (AdamW); the ride-along heads have their own optimizers (`_ridealong_acquire`) | Built at startup, frozen by K6 | low. Already ours in substance |
| `set_random_seed(seed, using_cuda)` | `_setup_model` on every construction AND every `load` | Seeds python, numpy and torch, and **sets `cudnn.deterministic=True, cudnn.benchmark=False` process-wide on CUDA** | **FINDING**: nothing on the training path restores it (only `rust_eval/parity.py:149` does). Its throughput cost is unmeasured. Stage 2 owns seeding as a LABELLED behaviour change |

### 1.4 Schedules and controllers

| touchpoint | where | coupling |
|---|---|---|
| LR schedule | `model.lr_schedule` is reassigned as a **lambda** by `AdaptivePPOCallback` / `TwoPhaseLRCallback` (`adaptive_lr_callback.py`), by `--fork-lr` (`fork_lr.py:164`) and on resume (`model_build.py:616,660`) | SB3 applies it in `_update_learning_rate`. The lambda is **pickled into every `.zip`** (§1.8) |
| clip range | `lambda _: args.clip_range` (`model_build.py:706`), **which closes over the whole argparse namespace** (8.5 KB per zip) | Wrapped by `FloatSchedule` in `_setup_model` |
| KL controller | `AdaptivePPOCallback._on_rollout_end` reads `model.logger.name_to_value["train/approx_kl"]` (`:413`) | **The logger is used as a message bus**; see §2.1 |
| adaptive batch | `AdaptiveBatchCallback` writes `model.grad_accum_steps` | low |

### 1.5 Callbacks — 26 classes, all built in `main/train/callbacks.py:92` (`build_callbacks`)

About 11 are on by default:
- **Always on:** `_TrackingCheckpointCallback`, the LR controller, `MetricsExporterCallback`, `_HparamLogCallback`, `DoseLogCallback`, `GracefulRestartCallback`, `SignalMetricsCallback`, `RewardTermMetricsCallback`.
- **On by default:** `RankTripwireCallback` and `TeamWinRateCallback`.
- **On in a non-debug run:** one eval callback (`PerOpponentEvalCallback` or `SelfPlayCallback`).

The rest are flag-gated:
- Adaptive batch, the distill anchor and distill stop.
- The exploiter temperature and ladder.
- `WinProbLabelCallback` and `DenseAuxLabelCallback` (Python core only).
- The value sidecar, the fork arm and team PFSP.
- cf supply and the search teacher.

**What couples a callback to SB3:**
- **`self.locals`:** 10 reads in 3 files, all `.get()` — `signal_callback.py` (with an async fallback to `wave_infos`), `win_prob_callback.py` and `dense_aux_callback.py`.
- **The logger as a bus:** 6 readers of `logger.name_to_value` — the two LR controllers, RankTripwire, DistillAnchor, DistillStop and MetricsExporter — plus `compile_control.py:816`.
- **Model attributes written by callbacks:** `lr_schedule`, `grad_accum_steps`, `distill_coef`, about 10 `distill_anchor_*` fields, and the `_win_*` / `_dense_aux_*` / `_fork_metrics` stashes.
- **`env_method` / `get_attr`:** in selfplay, the exploiter callbacks, team PFSP, team win rate and reward terms.
- **`model.save`:** in checkpoint, selfplay, eval and the teacher.

`self.globals` is never read. `self.parent` is read once (DistillAnchor).

**A second, undeclared hook system** sits on top of the callbacks. These modules reassign
`model.learn`, `model.collect_rollouts` and `model.train` as instance attributes:
- `learner_lifecycle.py:571` (K6's freeze guard and CUDA memory trend);
- `compile_control.py:784` (the sentinel lock and the canary);
- `learner_benchmark.py:793`.

An owned loop must call `self.collect_rollouts` / `self.train` **through attribute lookup** so these
keep intercepting. Stage 2 folds them into the declared hook table.

### 1.6 Logger → TensorBoard

- **Setup:** `_attach_run_tb_logger` (`main/train/run_io.py:222`) calls SB3's `configure(<run>/tb, ["stdout"?, "tensorboard"])`. There is no CSV or JSON output.
- **Volume:** 204 `logger.record` call sites; 129 literal keys plus 38 f-string families.
- **Usage:** `exclude=` appears once (`train/n_updates`); `record_mean` is never used.
- **Four explicit `dump(step)` calls:** `_HparamLog`, the two eval callbacks (**in the middle of a rollout**, §2.1) and `final_eval`.
- **The tools** (`tb_curate`, `tb_inherit`, and the readers in §1.1) depend on the tag names and the `<run>/tb` event layout, not on SB3 itself.

### 1.7 VecEnv

| path | class | learner reliance |
|---|---|---|
| `--env-core python` (the default today) | `SubprocVecEnv` (spawn) over `Monitor(MaskableAgentWrapper(Gen3Env))` (`main/train/env_factory.py:318`) | `step_async` / `step_wait`, `reset`, `get_action_masks` via `env_method("action_masks")`, `Monitor`'s `info["episode"]` |
| `--async-rollout` | `AsyncSubprocVecEnv(SubprocVecEnv)` | its own collect loop |
| `--debug` | `DummyVecEnv` | as for the Python core |
| `--env-core rust` | `RustVecEnv(VecEnv)` (`rust_vec_env.py:71`): a stub whose `env_method` is a declared dispatch onto `_m_<name>`; stepping raises | only the interface type |

### 1.8 Save / load and the `.zip` format

- **What gets written.** `model.save` writes `data` (about 130 keys, cloudpickled where not plain), `policy.pth`, `policy.optimizer.pth` and `pytorch_variables.pth`; `_excluded_save_params` (`instrumented_ppo/hparams.py:511`) keeps about 25 process-local fields out.
- **What it embeds as code.** Each zip pickles our `policy_class` and `policy_kwargs` (162 KB), sb3-contrib's buffer class, and **two lambdas from our own modules** (`lr_schedule` and `clip_range`; the latter carries the argparse object).
- **Who reads it.** Every reader goes through `MaskablePPO.load` / `InstrumentedMaskablePPO.load`, including `agents/model/snapshot.py:934` `load_model_snapshot` and `:1141` `load_foreign_opponent`. That covers:
  - resume (**high** — it restores `num_timesteps`, `_n_updates`, the optimizer state, and therefore the KL-annealed LR that makes `--lr` inert on a resume);
  - the pool, baselines, play, eval workers, eval-trace generation, the prober (2 sites plus a side read through `load_from_zip_file`), counterfactuals, search, cf producers, `winprob_finetune` (which loads AND saves), the Rust eval and the inference service.
- **No policy-only loader exists.** Every reader except resume needs only `policy_kwargs` + `policy.pth`.
- **Not traced:** `main.elo`, `main.anchors` and the teacher are assumed to route through `eval_worker` / `snapshot_ladder` / `load_foreign_opponent`. This is **UNVERIFIED**.

**Stages 1–3 do not touch any of this.**

---

## 2. Findings that shape the design

### 2.1 FINDING (verified): every eval cycle silently skips one KL→LR controller step, on both env cores

**Mechanism.** The ordering is:
- SB3's `learn()` dumps the logger **before** `train()`, so `train()`'s `train/*` values sit in
  `name_to_value` until the next iteration's dump.
- The KL controller reads `train/approx_kl` from there at the next `on_rollout_end`.
- The eval callbacks call `logger.dump(step)` inside `_on_step` (`eval_callback.py:1767`,
  `selfplay_callback.py:766`), which **clears** `name_to_value`.

So on every cycle where an eval publishes during the rollout:
- the previous update's `train/*` values are written at the eval's SNAPSHOT step, which lies behind
  the current step;
- `AdaptivePPOCallback` finds no reading and returns without updating its EMA (`adaptive_lr_callback.py:413`);
- RankTripwire, DistillStop, the DistillAnchor dual and the MetricsExporter pipe each miss that reading.

**Both env cores have it:**
- **Python core:** it fires when a worker's results are collected in some `on_step`.
- **Rust core** (Lane H's eval, in process and blocking): `_launch_eval` → `_collect_pending` → `_record`
  → `dump` runs inside the launching `on_step`. This is verified by reading the code; no Rust-core
  production run with eval exists yet to observe it.

**Evidence** (read-only, the runs' own TensorBoard files, inherited files excluded):

| run | era / role | KL controller | updates | updates whose `train/*` went out at an eval step | eval cycles |
|---|---|---|---|---|---|
| `ai_v14_01_base` | N0, fresh | **live** | 739 | **37 (5.0%)** | 37 |
| `ai_v12_02_winprob_critic` | older era, fresh, 75M | **live** | 741 | **37 (5.0%)** | 37 |
| `ai_v12_11_ladder_ctrl10M`, `_19_lambda09`, `_23_rollout` | 10M ladder, fresh | **live** | 100 / 100 / 98 | 5 / 5 / 5 | 5 each |
| `ai_v14_02_lbat_ctrl` (C), `_03_lbat_e5` (E5), `_05_lbat_l95`, `_06_..._fix`, `_08_g0p_k3` | battery forks | **frozen** (`--fork-lr-freeze`) | 80 each | 4 each (5.0%) | 4 each |

In N0, the LR was unchanged after **37 / 37** skipped updates, and after **695 / 701** normal ones.

**Scope:**
- **The size of the effect.** The controller moves the rate rarely, so a skipped reading changed N0's
  LR path by at most a deferred step. The defect is a **5% reading loss, not an LR excursion**.
- **Who was affected.** It hits only runs with a live controller. Frozen forks (every battery arm
  and every v13 population arm) lose the reading but their LR is pinned anyway.
- **Bias between arms.** None is expected wherever arms share the eval cadence (every era above sits
  at exactly 1 eval per 20 updates). A comparison could be biased only between arms whose
  eval-cycles-per-update differ while the controller is live; none was found among the runs scanned.
- **Other consumers.** RankTripwire (warn by default) and the distill-stop / anchor controllers lose
  1 reading in 20.
- **The sizing study.** Its arms share the cadence, so they share the defect equally; the comparisons
  stay fair.

**Disposition: FIXED 2026-10-01** (`gen3_eval_dump_isolation_v1`, stage 2's first unit). Both eval
callbacks' `_collect_pending` run under `agents/training/logger_scope.isolated_dump`: the values pending
when the cycle starts are held aside, the cycle dumps only its own scalars, and the held values are put
back. One test per core fails on revert (`eval_dump_isolation_test.py`), and E1's pin is flipped. This
is a **REGIME BOUNDARY**: a run with a live controller from that commit onward is not comparable with
an earlier one on its LR / dose trajectory. Ledger 2026-10-01 *THE EVAL DUMP DROPPED A KL READING*.
The original plan, kept for the record:
- **Stage 1 PRESERVES it bit-for-bit**, so that equivalence is identity.
- **Stage 2 FIXES it by construction.** `train()`'s statistics are handed to the post-update hooks as
  a value, and the bus is not cleared mid-rollout. This is a **labelled behaviour change**: it lands
  with a test per core that FAILS on revert, and before the X26 ride-along baseline (orchestrator,
  2026-10-01).

### 2.2 FINDING: the TB step convention is a contract

`train/*` for update k is written at the **next** iteration's dump step, i.e. after rollout k+1. An
owned loop that dumps after `train()` would shift every `train/*` series by one rollout and break
TensorBoard parity with every existing run. **Stage 1 keeps SB3's order.** Any change to it is a
separate, labelled decision (not proposed).

### 2.3 FINDING (measured): a seeded `--debug` run is NOT reproducible

Two `train_rl_agent --debug --steps 6000 --seed 42` runs on the Python core (CPU, rust bridge,
2026-10-01) give:
- the same 106 tags;
- 66 of 103 non-wall-clock series differing from the **first** dump (`belief/hptype_n_slots` 1945 vs
  2095 — different games);
- different `policy.pth`.

So "a real-run A/B at a matched seed has identical losses" is **not measurable on the Python core**.
**The Rust core IS reproducible.** Two `--debug --arch production --env-core rust --n-envs 4
--n-steps 256 --batch-size 256 --steps 4000 --seed 42` runs (CPU, 2026-10-01; 16 updates each) give:
- all 342 non-wall-clock series identical (tag, step and value; NaN = NaN);
- the same `final_model.zip` `policy.pth` bytes.

Only wall clocks differ (`time/fps`, `*_ms`, `*_ms_per_host_step`, `trainee_decisions_per_s`).

This decides the equivalence design (§4):
- **Rust core:** the real-run A/B is read as IDENTITY.
- **Python core:** identity rests on E1 + E2, where the inputs are held identical, and its real-run
  A/B is read against the stock-vs-stock spread.

### 2.4 FINDING: checkpoints embed code

Every `.zip` cloudpickles two closures from our modules, one of them carrying the run's argparse
namespace, plus our policy class by module path. Consequences:
- loading is coupled to module paths and to SB3's class layout;
- every resume revives a stale argparse object inside `clip_range`.

This matters for stage 4 only. Stages 1–3 leave the format alone.

---

## 3. The target design

### 3.1 Stage 1 — the loop is ours (bit-identical; `gen3_owned_ppo_loop_v1`)

**BUILT 2026-10-01**, as designed below. Two details were settled during the build:
- **The reference seam is upstream end to end.** Under `sb3_reference`, every vendored method
  (`_setup_learn`, `dump_logs`, progress, `_update_info_buffer`, the Python collect) defers to its
  upstream original. Without that, upstream's `learn` would reach OUR vendored methods through the MRO,
  and the A/B would compare ours with ours.
- **`_ppo_loop_mode` is in `_excluded_save_params`**, so the `.zip`'s `data` is exactly what it was
  before.

There is one new module, `agents/training/instrumented_ppo/loop.py`. It is a mixin `OwnedLoop`, placed
before `MaskablePPO` in `InstrumentedMaskablePPO`'s bases. It owns:
- **`learn()`**: SB3's sequence written out as a **declared phase table** (`LOOP_PHASES`):
  `setup → training_start → [collect → post_collect → progress → dump → update]* → training_end`.
  The table is the single place a reader learns the order. `instrumented_ppo_hub_contract_test` pins
  it the way it pins the fold order.
- **`_setup_learn`, `dump_logs`, `_update_current_progress_remaining`**: vendored. Each upstream source
  is hash-pinned next to the existing `train` pin, so an sb3 upgrade is loud while sb3 is still
  installed.
- **The Python core's collect** (`_collect_python`): vendored from `MaskablePPO.collect_rollouts` with
  its **local variable names unchanged**, because callbacks read `self.locals`. `RolloutProbes`
  dispatches to it in place of `super()`. The Rust and async collectors are already ours and are
  unchanged.
- **Hooks keep their interception points.** The loop calls `self.collect_rollouts` and `self.train`
  by attribute, so K6's freeze guard and the compile sentinel keep wrapping them untouched. This
  avoids colliding with the K6+K8 agent's canary-persistence work.

**What stays SB3 in stage 1** (named, not trimmed silently):
- the constructor and `_setup_model` (including `set_random_seed` and the cudnn leak);
- the policy base classes and the init;
- the buffer class, its `get()` and window GAE;
- `BaseCallback` / `CallbackList`;
- the logger;
- the VecEnv classes and `Monitor`;
- save / load and the `.zip`.

**Deleted:** nothing. Added: about 250 lines plus tests.

### 3.2 Stage 2 — declared hooks and the two labelled fixes (`gen3_declared_loop_hooks_v1`)

- **The update result is a value.** `train()` returns its scalar dict, and the `post_update` hooks
  receive it. The KL controller, RankTripwire, DistillStop and the DistillAnchor dual read that value
  instead of `logger.name_to_value`. The eval callbacks' mid-rollout `dump(step)` is replaced by
  recording at the snapshot step **without clearing** the update's values. **Fixes §2.1.**
- **One test per core that FAILS on revert:** an eval cycle inside a rollout neither clears the
  previous update's `train/*` values nor skips the controller or tripwire step.
- **A declared hook table — BUILT 2026-10-01** (`gen3_declared_loop_hooks_v1`,
  `agents/training/loop_hooks.py`). `HOOK_POINTS` = learn > collect | update, `HOOK_OWNERS` =
  learner_freeze (outermost), compile_sentinel. Both `attach`es register context-manager hooks
  instead of reassigning bound methods, and a duck-typed model with no table gets the same bodies as
  wrappers. The sb3_reference seam applies the table too. Identity: E1 is unchanged, and
  `loop_hooks_test.py` pins nesting under both arms plus the freeze guard's checks on the real
  learner. The original plan: K6's freeze guard / CUDA memory trend and the compile sentinel / canary
  register as declared loop hooks, replacing the instance-attribute reassignment of
  `learn` / `collect_rollouts` / `train`. The table is frozen at startup (the declared lifecycle): a
  hook registered after `training_start` is a typed FATAL. The orders that today hang on list order
  become declared and checked: WinProb before ValueSidecar and Fork, and the post-collect window
  labels → fork → PBRS → frozen-φ.
- **Seeding is ours — BUILT 2026-10-01** (`gen3_owned_seeding_v1`, `OwnedLoop.set_random_seed`).
  It makes sb3's draws in sb3's order (python, numpy, torch, the action space, the env) and touches no
  backend flag. sb3's version set the process-wide `cudnn.deterministic = True` / `benchmark = False`
  on every CUDA construction and load.
  - **Evidence:** 0 cuDNN kernels in a production update under either setting (2.03M CUDA kernels);
    losses equal.
  - **Timing:** CONFOUNDED (a fixed T→F order, and un-restored update counters moving the diagnostics
    cadence); no claim is drawn from it.
  - **Regime boundary:** CUDA runs from this commit onward run with torch's default cuDNN flags. This
    is NOMINAL, because no cuDNN kernel runs in the update. CPU, and so the K9 golden, is unchanged.
  - Readout: `research_state/measurements/own_ppo_loop/` §5.
- **Not in stage 2:** replacing `self.locals` with a typed per-step event. It lands with the Python
  core's deletion, because only the Python-core callbacks read it.

### 3.3 Stage 3 — with the deletion pass (after the switch)

- **The buffer is ours.** Flat device tensors, written by the Rust collector directly, with window
  GAE as `store.game_gae`'s sibling. It is held to SB3's `get()` permutation, so the K9 golden stays
  bit-identical.
- **Deleted with the Python core:** the vendored Python collect, `async_vec_env.py` (287), the
  `DummyVecEnv` / `SubprocVecEnv` / `Monitor` wiring, `RustVecEnv`'s SB3 base, and
  `device_batches`' SB3 sample type.
- **The logger is ours.** A thin `SummaryWriter` writer with the same tags, the same step convention
  and the same `<run>/tb` layout, plus the launcher pipe.
- **`BaseCallback` → hook protocol** for the surviving callbacks.

### 3.4 Stage 4 — the checkpoint and the dependency (a separate decision, not proposed now)

1. **A policy-only loader** (`policy_kwargs` + `policy.pth`) for the roughly 20 readers that need only
   the policy. Low risk; it can go earlier as hygiene.
2. **An owned writer** that keeps the `.zip` layout (`policy.pth`, `policy.optimizer.pth`, a `data`
   JSON) but stores schedules as DATA, not lambdas, and drops the pickled argparse.
3. **Retire the SB3 policy base classes.** This means re-implementing `_build`'s ortho-init order
   bit-identically, or re-recording the golden's `init_params_sha256` with a reason.

Then drop `stable-baselines3` / `sb3-contrib`. **Recommendation:** decide after stage 3. SB3 at rest
costs nothing per update, while the `.zip` is the most widely read contract in the repo.

---

## 4. Equivalence plan (stage 1 must be identity)

| check | what | bar | tier |
|---|---|---|---|
| **E0** | The K9 LEARNER GOLDEN (`agents.training.learner_golden`) | bit-identical. **Necessary, not sufficient:** it calls `train()` directly and does not exercise `learn()` | routine (exists) |
| **E1** (`own_ppo_loop_test.py`, PASS; mutation-checked) | **Lockstep differential** (new, CPU, seconds): a seeded scripted VecEnv over the production spaces, serving rows of the golden's buffer, with dones on a fixed schedule, one `TimeLimit.truncated` + `terminal_observation` (the bootstrap branch) and `info["episode"]` dicts. Real callbacks that need no live env are registered — `WinProbLabelCallback`, `AdaptivePPOCallback`, `SignalMetricsCallback`, a mid-rollout `dump(step)` stand-in for eval (pinning §2.1's preserved quirk) — plus a RECORDING callback. Three iterations of upstream `MaskablePPO.learn(model, …)` (still callable unbound while sb3 is installed) vs `OwnedLoop.learn`, from the same seeded learner | EXACT: the hook trace (name, `num_timesteps`, `n_calls`, locals keys, `buf.pos`), every buffer array at each rollout end, `name_to_value` before every dump and each dump's step, params sha after each update, `_n_updates`, `_current_progress_remaining`, `ep_info_buffer` | routine |
| **E2** (`own_ppo_loop_parity_test.py`, PASS; mutation-checked) | **Rollout level on REAL games**: `rust_rollout.parity`'s record/replay machinery (the same games, keyed trainee draws). The Python-core side is filled by the stock collect and by the owned collect, in one process with the same weights | buffers byte-EXACT on every field (tighter than parity's 1e-5 bars, which compare different batch compositions) | integration (`sim`) |
| **E3** | **Short real-run A/B** at matched seed: `train_rl_agent --debug` (CPU) for ≥ 3 updates, under each loop, on BOTH env cores. The stock loop is selected through a TEST SEAM env var (`GEN3AI_PPO_LOOP=sb3_reference`), deleted in stage 3. **The control is run FIRST:** stock vs stock | If stock-vs-stock is identical (§2.3's Rust half): every non-wall TB series identical (tag set, step grid, values) and `policy.pth` bytes equal. Otherwise: the same tag set and step grid, and divergence onset and magnitude no earlier or larger than stock-vs-stock, declared before reading | one-off, recorded in `measurements/own_ppo_loop/`; the control is MEASURED (§2.3): **Rust core identical, so identity is the bar there**; Python core not reproducible, so it gets the bound. **RESULT (2026-10-01):** Rust — 342 / 342 non-wall series identical, and `policy.pth` equal in both arms AND to the pre-stage-1 control; Python — the same tag set and step grid, with values diverging from the first dump exactly as stock-vs-stock does |
| **E4** | The sizing study's carry-over | nothing extra: E0–E2 make stage 1 identity at the update and rollout level, so the arms' verdicts (run pinned on today's loop) transfer by construction | — |

Stage 2 is **not** identity on purpose (§2.1 and the seeding fix). It is gated by its fail-on-revert
tests, E0, and a re-run of E1 with the expected deltas named (the controller's step on eval cycles,
and the TB step of `train/*` on eval cycles).

---

## 5. Estimate, staging and interactions

| stage | content | agent-days | when |
|---|---|---|---|
| **1** | the loop module + phase table + upstream pins (0.3); E1 (0.35); E2 (0.2); E3 incl. control (0.35); docs: training `CLAUDE.md`, `designs/training/ppo_step.md`, `learner_lifecycle.md`, program §4 deletion list, this doc (0.2); rebase onto in-flight work + full routine gate + ship (0.2) | **1.6** (1.3–1.9) | now; must precede the deletion pass, need not precede the switch |
| **2** | update-result handoff + eval no-clear (0.35); per-core fail-on-revert tests (0.25); hook table absorbing K6 / compile wrappers (0.35); seeding (0.15, plus a GPU benchmark read under the GPU lock); docs + ledger paragraph (0.15) | **1.25** (1.0–1.6) | straight after stage 1, after the K6+K8 canary work lands; before the X26 ride-along baseline |
| 3 | own buffer, logger, callback protocol; delete the Python collect / async / VecEnv glue | 2–3 | in the deletion pass |
| 4 | policy-only loader, owned writer, drop sb3 | 3–5 | separate decision |

**Stages 1 + 2 together are about 2.85 agent-days (2.3–3.5), which exceeds the 2-day bar.** Stage 1
alone fits, and is the stage that takes SB3's loop off the production path. Stage 2 is reported
separately (orchestrator's rule: tell, don't trim).

**Riskiest parts:**
1. **E3 on the Python core** (§2.3): a seeded run there is not reproducible, so its real-run A/B only
   bounds divergence, and identity there rests on E1 + E2. On the Rust core (the switch's default and
   the sizing arms' core) the A/B is identity, as measured.
2. **The undeclared wrapper stack** (§1.5): it keeps working only through attribute lookup. E1
   asserts it with a wrapped model.
3. **A hidden SB3 behaviour** that no vendored line reproduces, for example `_setup_learn`'s env reset
   on a resume (`_last_obs` is None after `load`). E1 runs a save → load → `learn` resume leg.
4. **Stage 2's controller fix** changes LR trajectories on every live-controller run from that commit
   on. That is a regime boundary, recorded like the 2026-09-07 eval-regime boundary.

**In-flight work:**
- **The K6+K8 agent** (R1 regime bar → canary persistence → "no silent eager fallback"): touches
  `compile_canary` / `compile_regions` / `compile_control`. Stage 1 does not touch them. Stage 2's
  hook table lands **after** them.
- **The R1 declared-levers agent** (`k8-r1-declared-keys`: `TrainSetup._r1_levers` /
  `check_r1_declared`, then F-SZ-3's env-step checkpoint cadence): shares `ppo.py` (stage 1 adds one
  base class) and `learner_golden.py` (untouched). Stage 1 rebases onto it. F-SZ-3 changes
  `_TrackingCheckpointCallback`'s `n_calls` semantics, which E1's trace records; whichever lands
  second updates the trace.
- **The sizing study:** its arms run pinned; nothing here changes their code. E4 carries the verdict.

**What becomes dead** (program §4):
- **Stage 1:** the stock `MaskablePPO.collect_rollouts` / `learn` paths become unreachable in
  production; the `GEN3AI_PPO_LOOP` seam becomes deletable in stage 3.
- **Stage 3:** `async_vec_env.py` (287), the Python collect, `Monitor` / `DummyVecEnv` /
  `SubprocVecEnv` use, and `RustVecEnv`'s SB3 base.
- **Stage 4:** the sb3 / sb3-contrib dependency.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-01 | Own the PPO loop **(owner: "before the cutover if it isn't too hard"; then "as long as it is less than 2 agent days")** | Staged. Stage 1 (identity: `learn` / `_setup_learn` / `dump_logs` / the Python collect vendored as a declared phase table; `.zip`, policy, callbacks, logger and buffer kept) at 1.6 agent-days is GO. Stage 2 (declared hooks, the KL-skip fix, seeding) follows as labelled behaviour changes. Stages 3–4 come with or after the deletion pass | Everything at once (callbacks, buffer, logger, checkpoint; ~8–12 days); making the switch wait for stage 1 (it buys nothing; the arms run pinned); changing the `.zip` in stage 1 (about 30 readers) | §1 inventory; §2.3 control |
| 2026-10-01 | Equivalence for stage 1 | E0 golden + E1 lockstep differential + E2 same-games buffer identity; E3 real-run A/B read against a stock-vs-stock control | Treating a matched-seed real run as identity on the Python core: measured NOT reproducible there; the Rust core IS (342/342 series, same `policy.pth`), so on it the A/B is identity | §2.3 controls, 2026-10-01 |
| 2026-10-01 | Stage 2 **(owner: "stage 2 FULL is approved")** | Declared hooks, the KL-skip fix on both cores and owned seeding, each fix a labelled regime boundary; before X26 | Stopping at stage 1 (under the 2-day bar) | the §5 estimate |
| 2026-10-01 | Stage 2 unit: owned seeding — a NOMINAL regime boundary (CUDA) | sb3's draws, no cuDNN flag touched (`OwnedLoop.set_random_seed`); `owned_seeding_test` fails on revert | Restoring the flags after `_setup_model` (two writes to undo one); a second GPU hold to de-confound the timing (orchestrator: the zero-kernel count is decisive and the GPU queue is the bottleneck) | 0 cuDNN kernels of 2.03M per update, losses equal; the timing CONFOUNDED (fixed order, un-restored counters), no claim |
| 2026-10-01 | Stage 2 unit: the declared hook table | `LoopHooks`: around-hooks at learn / collect / update, owners ordered by a table (freeze guard outermost), frozen at training start, late / duplicate / undeclared = FATAL_CONFIG; the compile sentinel's `record` stays once per update after the canary (the no-silent-eager window's contract) | Keeping instance-attribute reassignment (nesting order = attach order, and nothing refuses a late hook); a callback-style before/after pair (the sentinel needs its guard AROUND the body) | `loop_hooks_test.py`; E1 unchanged; `compile_control_test`, `learner_lifecycle_test` unchanged (adapter path) |
| 2026-10-01 | Stage 1 built | `OwnedLoop` (`instrumented_ppo/loop.py`) after `RolloutProbes` and before `MaskablePPO`; the reference seam defers every vendored method to upstream; `_ppo_loop_mode` is excluded from the `.zip` | A reference seam over `learn` / collect only (upstream `learn` would reach our vendored methods through the MRO, and the A/B would compare ours with ours) | E0 golden unchanged; E1 / E2 exact; E3 Rust identical (same `policy.pth` as the pre-stage-1 control) |
| 2026-10-01 | The eval-dump KL skip, FIXED (stage 2, unit 1) — a REGIME BOUNDARY | `logger_scope.isolated_dump` on both eval callbacks' `_collect_pending`: the cycle dumps only its own scalars, and the update's `train/*` survive for their dump and every bus reader. Live-controller runs from this commit onward are not comparable on LR / dose with earlier ones | Moving the dump after the update (it shifts every `train/*` series in the archive); routing only the KL controller around the bus (it leaves RankTripwire / DistillStop / the anchor dual and the TB stamping broken) | `eval_dump_isolation_test.py` (one test per core, fails on revert); ledger 2026-10-01 |
| 2026-10-01 | The eval-dump KL skip | Preserved bit-for-bit in stage 1; fixed in stage 2 with per-core fail-on-revert tests, before X26 **(orchestrator)** | Fixing it inside stage 1 (that would make stage 1 non-identity and its verdict un-transferable) | §2.1: 37/739 N0 updates (5.0%), 1 per eval cycle in every run scanned |
