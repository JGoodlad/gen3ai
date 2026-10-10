# CLAUDE.md — Training (`src/agents/training/`)

Callbacks, the reward's declarations, the belief/value auxiliaries, the owned PPO loop and the eval pipeline.
**How to LAUNCH** is the root `CLAUDE.md` and `designs/ops/training_runbook.md`. This leaf keeps only the RULES and
HAZARDS to know before touching a subsystem, plus a pointer; **`designs/training/` owns the detail and is
always-current like this file — update the topic doc in the same pass as the code.**

The current world: training and eval run IN PROCESS on the Rust core (the only env core, the only collector, the only
eval engine — no Showdown server, no port). The Python trackers, reward manager, `Gen3Battle` / `RLPlayer` road and
poke-env are DELETED (T27 P6): every fold, label, reward and trace record is the Rust core's
(`src/rust_sim/src/trackers/`, `src/rust_env/`; `designs/rust_sim/trackers.md`). `turn_delta.py` keeps only the frozen
`TurnDelta` field layout; `hidden_power_tracker.py` keeps `HIDDEN_POWER_TYPE_ORDER`. Battle read-models:
`src/agents/battle/CLAUDE.md`; the observation layout: `src/agents/observation/CLAUDE.md`.

## Where the detail is — the topic map

| I am about to touch… | Read |
|---|---|
| **a trainer / launcher FLAG — adding, keeping or deleting one** | `designs/ops/flag_census.md`: a flag survives only by naming a LIVE user in its row (`src/main/flag_census_test.py`); a deleted flag gets a `designs/deleted_flags.md` row, which is also the reason a typed one is refused with |
| the PPO step, the FOLD ORDER, the owned loop, `--diagnostics-every`, the policy's GAE λ | `designs/training/ppo_step.md` |
| the learner's GIGO gates (K9): the learner golden, behaviour consistency, fail-closed non-finite; the CUDA memory trend | `designs/training/learner_gates.md` |
| the DECLARED LIFECYCLE (K6): the freeze guard, `@startup_builder`, the update-fit check, no global reseed, the debug orphan watchdog | `designs/training/learner_lifecycle.md` |
| the reward (the terminal alone), the shaped-checkpoint refusal, the no-progress clock | `designs/training/reward.md` |
| the in-loop eval, the eval COUNT ledger, `main.h2h`, `main.plateau`, `main.belief_roles`, the baseline registry, ELO / the ladder, the forensic trace | `designs/training/eval_rules.md`, then `designs/training/eval_and_rating.md` |
| self-play, the snapshot pool, stable opponents | `designs/training/self_play_and_pool.md` |
| exploiter mode, the best-response gap, the rank tripwire (distillation / search-as-teacher are DELETED) | `designs/training/exploiter_and_distillation.md` |
| per-team win-rate tracking, team blocking | `designs/training/team_curriculum.md` |
| the win-prob critic, the 250-turn cap, `--vf-coef` | `designs/training/critic_and_value_losses.md` |
| the win-prob head (`--win-prob-mode`) | `designs/training/winprob_head_and_pbrs.md` |
| the detached ride-along heads; strict checkpoint loading; opponent (inference-only) loads | `designs/training/ridealong_heads.md` |
| the training-side value sidecar | `designs/training/value_sidecar.md` |
| any supervised belief loss, the opponent-class label weight | `designs/training/belief_losses.md` |
| the recipe surface (`--arch production`'s recipe half), gradient accumulation, the noise scale, THE DOSE, `--fork-lr`, `--adaptive-batch` | `designs/training/step_size_and_batch.md` |
| the MatchupSpec, which FILE a run spec names, LINEAGE, TB inheritance | `designs/training/matchup_and_lineage.md` |
| the TB census and currencies, capacity telemetry, grad balance, `signal/`, the scaffolding gauge | `designs/training/telemetry_scalars.md` |
| a SUPPLY GUARD (`--supply-starve-cycles`, `FATAL_SUPPLY`) | `designs/training/supply_guards.md` |
| the FORK ARM (`--fork-fraction`, declared, OFF) | `designs/training/forks.md` |
| the RUST COLLECTOR, the env core, the observation source, the keyed draw, cadences | `designs/training/rust_collector.md` |
| `--compile-trainer`, the compile regions / sentinel / canary / cache, fp32 precision | `designs/training/compile_flags.md` |
| `stats.py`, `main.policy_drift` | `designs/training/offline_meters.md` |
| the stall-tail harvest / head-repair pipeline (offline) | `designs/training/stall_tail_harvest.md` |
| the counterfactual stack, the cf audit, the offline materializer — all DELETED (history only) | `designs/training/cf_grounding.md`; search-as-teacher's tombstone is `designs/training/search_teacher.md` |

Closed history — **do not update it, do not re-derive a plan from it**: the snapshot of this leaf before its
2026-10-10 cleanup, `designs/research_state/claude_md_archive/src_agents_training_CLAUDE_2026-10-10.md`;
`designs/research_state/claude_md_archive/training_leaf_faint_attribution_history.md`;
`designs/research_state/claude_md_archive/training_leaf_deleted_subsystems_history.md` (the latent-belief loss and
V_pub — a checkpoint recording either is refused by its migration).

## TensorBoard — the CURRENCY first (`gen3_tb_relevance_v1`)

🚨 **The first question about any scalar is its UNIT.** Under the win-prob critic (the only critic) the reward is the
terminal win indicator and `V(s) = sigmoid(win_head logit)`, so `train/return_*`, `train/value_loss` (a diagnostic;
its term is dropped) and `train/explained_variance` are in PROBABILITY units — **not comparable with an old shaped
run's tags of the same name**; read the run's `🎯 [CRITIC]` line first. PopArt and its currency are DELETED (L1).
🚨 **ERA RELEVANCE: a tag whose SOURCE is absent is not emitted** — the gate is on the SOURCE, never the value, so a
dead source leaves a GAP, not a confident number. ⚠️ A run pinned before `gen3_obs_margin_unconditional_v1` has a
degenerate `win_margin`: read its `win_prob/*contested*` family as absent. Detail: `designs/training/telemetry_scalars.md`. ⚠️ `edge/*` / `cell/*` liveness and `grad/*_share` are NOT effect sizes
(zero-init families look dead and working alike; read `weight_norm` and `grad_norm` as a pair).

### What to watch on a WIN-PROB run — the 28-tag dashboard

- **Stronger?** `eval/elo` + `eval/elo_ci` · `eval/win_rate_vs_bots` (saturates; a FALL is the alarm) ·
  `eval/win_rate_vs_pool` (pinned near 0.50 by the gate) · `signal/outcome_win_rate_bots` / `_pool` · `rollout/ep_rew_mean`.
- **Critic honest?** **`win_prob/critic_resolution`** (THE meter, higher is better — `critic_reliability` is not) ·
  `critic_brier` · `critic_skill` · `critic_decomp_residual` (≈0) · `win_prob/ece` · `rel_gap_b*` (NaN = a hole) ·
  **`win_prob/start_gap`** (positive = optimistic at the opening) · `train/explained_variance` · `win_prob/coverage`.
- **Optimization healthy?** `train/approx_kl` · `train/learning_rate` + **`train/dose_rate`** · `clip_fraction` ·
  `grad_norm` · **`grad/value_policy_logratio`** · `train/noise_scale_ratio_policy` (beside the total, never alone).
- **Kill condition + GIGO guards:** **`rollout/ep_len_mean`** and **`signal/draw_rate`** are PRIMARY endpoints (a
  `[0,1]` critic cannot say a timeout is worse than a loss); **`reward/untracked_abs_mean` must read exactly 0.0** and
  **`train/return_abs_max` exactly 1.0**. Contested-split questions: `python -m main.scaffolding_gauge --reliability
  --reliability-reweight` and `main.critic_gate`.

## The PPO step (`instrumented_ppo/`) — the FOLD ORDER contract

- **Two straight lines** (K8): steps 1–3a (policy loss, the belief bank, the win-prob BCE) are ONE function,
  `instrumented_ppo/micro_step.micro_step` = compile region R1; `train()` folds the rest as the declared EAGER TAIL.
  **No flag combination reorders them**; `instrumented_ppo_hub_contract_test.py` pins both orders, the mixin list and
  `MaskablePPO` LAST in the MRO. A tail fold that runs its OWN extractor forward CLOBBERS the stashes earlier steps
  read — moving a stash reader below one silently scores the wrong states.
- 🚨 **R1 is a static-shape program:** no host read (`.item()`, `float(t)`, `bool(t)`), no boolean-mask indexing /
  `nonzero` / `bincount`, no Python branch on a tensor, no numpy; a diagnostic is a `(value, weight)` pair of 0-d
  tensors. A new per-micro-batch host read in `train()` goes through the DEFERRED queue (`instrumented_ppo/host_reads.py`)
  — `host_sync_guard_test.py` fails one. The legacy `belief_bank` functions stay as the reference the static twins are
  pinned equal to.
- 🚨 **The loop is OURS** (`instrumented_ppo/loop.py`, `designs/endstate/design_own_ppo_loop.md`): the rollout buffer,
  logger, callback protocol (`loop_callbacks.py`, declared `STEP_LOCALS`) and env base (`TrainerVecEnv`) refuse any
  other layout. Must not break: the **dump stays BEFORE the update**; the **final dump** at `learn()`'s end; the abort
  runs only at a SAFE POINT (`main/train/deferred_abort.py`); hooks register only at the declared points
  (`agents/training/loop_hooks.py`; never reassign a bound method); the `step` event fires before the buffer row is
  written. A test reads an update's scalars from `testkit.record_dumps`, never `name_to_value` after `learn()`.
  **A callback that dumps the logger mid-rollout MUST use `logger_scope.isolated_dump`** (else the KL→LR controller
  skips readings — `gen3_eval_dump_isolation_v1`, a regime boundary on LR / dose trajectories).
- 🚨 **No global RNG is SEEDED after the freeze** (`global_rng_guard.py`): `GlobalReseedError`. A stream that must
  repeat owns a generator; an opponent load builds inside `isolated_global_rng()`.
- **K9** (`designs/training/learner_gates.md`): `learner_golden_test.py` pins one update's exact bytes — any fold /
  term / default change fails until re-recorded deliberately (`python -m agents.training.learner_golden record
  --reason "..."` under every interpreter with an entry; never routine). Every non-finite loss / gradient / buffer /
  KL is `NonFiniteLearnerError` (exit 4, not restarted) BEFORE the optimizer moves — never `nan_to_num` or a skipped
  step; a `where(isfinite)` on a label is a NaN hide unless it means `-inf` (use `isneginf`).
- 🚨 **K6 — the learner FREEZES at the first rollout** (`learner_lifecycle.py`): a new optimizer, `nn.Parameter`,
  module, buffer or optimizer-state entry after it is `LazyAcquisitionError` (exit 3). Build in `_build` /
  `_setup_model` / `__init__` or a `@lifecycle_decl.startup_builder`. A CUDA stream / graph / pool is a startup
  acquisition too, and a bare `__init__` does NOT exempt it. Static twin: `src/learner_lifecycle_gate_test.py`. A
  sustained CUDA leak is `FATAL_CUDA_LEAK` (exit 6). Startup runs one DRY update (`update_fit.py`).
- Benchmark phase marks in `train()`: exactly `if _ph is not None: _ph("<phase>")`, name in `phase_hook.PHASES`.
- `--policy-grad-coef` scales ONLY the clipped surrogate. ⚠️ `train/approx_kl` is the LAST epoch's mean,
  `train/clip_fraction` pools epochs. `--diagnostics-every N`: a skipped update writes NONE of the optional tags (a gap).

## The reward — the TERMINAL alone

The shaped path is DELETED (config v122): the reward is the win indicator (victory 1.0, draw 0.0, γ 1.0 — constants
of the namespace; `--gamma`, `--victory-value`, `--draw-penalty`, `--terminal-indicator`, `--critic` are DELETED and
refused with their reason). The rule is `reward_config.terminal_breakdown`; training's terminal is the Rust core's
twin. 🚨 **A resume or fork of a checkpoint trained WITH shaping REFUSES** (`FATAL_CONFIG`; run it pinned to its own
commit); it still loads as an opponent, in meters and in the prober. `material_margin.py` is the Python statement of
the `win_margin` label rule (the label itself is the Rust core's `labels/margin.rs`). Detail:
`designs/training/reward.md`.

## Eval, the ledger, baselines, ELO — the rules (detail: `designs/training/eval_rules.md`)

- **The in-loop eval is BLOCKING and in process** on the Rust eval core, every 2M steps, 100 games per opponent over
  the archetype bots + `random` + every self-play sentinel. A missing eval core is FATAL, never a fall-back. An
  OFFLINE caller declares its own eval core through `rust_eval.offline`.
- 🚨 **Trace result vocabulary is `WIN` | `LOSS` | `DRAW`** (a `DRAW` carries `meta.draw_kind`; unknown is REFUSED).
  🚨 **The capture quota PREFERS LOSSES** — a trace tree is loss-enriched by design; one that records no selection is
  SELECTION UNKNOWN. ⚠️ A protocol identifier carries the NICKNAME, not the species — resolve through the side
  reader's own-team map. A forensic recorder must never take down a run.
- 🚨 **`--eval-mirrored-pairs` and `--promotion-sprt` are REGIMES** (recorded, inherited): read pair-level
  (pentanomial) intervals, never per-game; `elo.load_rows` refuses a run whose rows span the mirrored boundary. Do not
  flip either default on your own.
- 🚨 **The eval regime is recorded and INHERITED**: pool sentinels are GREEDY and draw the trainee's teams;
  `--no-eval-sentinel-greedy` restores the old regime (+8.9 pp to the trainee); `--promote-threshold` follows it. Read
  the `⚖️  [EVAL REGIME]` line, never assume it.
- 🚨 **The eval COUNT ledger** (`eval_ledger/`): every row is written under a CLAIM for a REQUEST; every reader goes
  through `eval_ledger.read` with a spelled-out `ReaderDecl` (`src/eval_ledger_reader_gate_test.py`, EMPTY allowlist).
  Its `.ledger_index/` is a CACHE, safe to delete. The in-loop eval DUAL-WRITES it (`cycle_ledger.py`).
- 🚨 **`main.h2h` keeps the player in seat p1** — a checkpoint against itself reads 0.5 + a seat effect; read a
  pair-clustered interval. `play-many` takes at most TWO architectures per engine. Run it from the repo root.
  **`main.plateau` is Tier 1 only** (`TIER1_PLATEAU` is a candidate, never a declared plateau).
  **`main.belief_roles` plays nothing**; its adoption-gate form is `intent_logloss_conditional`.
- 🚨 **Read a baseline BY NAME** (`baselines.py`, `designs/baselines.json`): every entry is an explicit file;
  `python -m main.baselines set <name> <file> --reason "..."` is the only edit (a new opponent is a RE-MEASUREMENT).
  **Load with `baselines.load(name)`, never a bare `MaskablePPO.load`** (`BaselineLoadError` names the fix). The
  untaught meter's default opponent REFUSES at HEAD's architecture — pass `--opponent <a v144+ checkpoint>` (a new
  series) or run it pinned.
- 🚨 **ELO**: the games that selected a snapshot never rate it (ladder recipe v3: 200 FRESH games per promotion);
  every `ladder.json` carries a `recipe` stamp and a cross-run reader refuses or refits without it (`python -m
  main.elo refit [--apply] <run>`); **bump `LADDER_FITTER_VERSION` whenever the fit changes what a rating means**. The
  ladder's TRANSPORT (Rust engine vs the old poke-env rows) is a regime boundary a fit refuses to mix.
- A retained `eval_traces/step_<N>/snapshot.zip` may be a HARD LINK to a checkpoint — never write THROUGH it.

## Opponents, the pool, run specs, supply

- 🚨 **A bare run dir resolves to the run's LAST SNAPSHOT** (`gen3_last_snapshot_resolution_v1`), never
  `best_model` (exported on BOT win rate; only a last-resort rung, announced on stderr). On disagreement the file that
  trained FURTHEST wins. Every consumer goes through `fixed_opponent_pool.resolve_model_ref` — the ONE choke point;
  `run_spec_test.py` holds the census. Name the `.zip` or `@step` to pin a file. Detail:
  `designs/training/matchup_and_lineage.md`.
- 🚨 **`MatchupSpec.spec_hash()` is a measurement-regime tag**: built ONCE from the args, consumed, never re-derived;
  two hashes are not metric-comparable; a `--model` launch whose matchup differs prints `⚠️ [MATCHUP DRIFT]`.
- 🚨 **A FORK starts with an EMPTY self-play pool, which falls back to the BOT pool**; a genuine fork auto-seeds its
  parent's (`pool_seed.py`) or exits `FATAL_CONFIG`. `max_snapshots` holds on EVERY path (a scan included,
  `PoolOverCapError`); the pool's `model_config.json` is a WRITE-ONCE arch record checked before every add.
- 🚨 **Every live lever's SUPPLY is declared** (`lever_supply.py`): live but delivering zero for its floor →
  `FATAL_SUPPLY` (5); a mis-wiring → `FATAL_CONFIG` (3); neither restarts. Every supply line goes through
  `lever_supply.loud` (`emit` alone never reaches the child log).
- 🚨 **An exploiter's gain is TARGET-SPECIFIC** — read it as an opponent for the pool, measured by `python -m
  main.best_response_gap`, which refuses unmatched budget / dose / regime. Two exploiters of one archetype against one
  target are REPLICATES (each keeps its row).
- ⚠️ **A raw per-team win rate conflates pilot competence with team strength** — normalise first. Per-team series
  have **NO TensorBoard emission** (owner rule, pinned); the tracker RAISES on a pool-ORDER disagreement.

## The Rust collector and env core (detail: `designs/training/rust_collector.md`)

- `rust` is the ONLY env core and the complete-game collector the ONLY collector: a game's rows are buffered until it
  ends, GAE and the win label run on the complete game, and an update fires at `--rollout-target-samples` completed
  rows. Run sizes are declared in ONE block, `designs/production_config.json` `recipe.sizing`. A python-era WINPROB
  checkpoint moves onto rust announced as a CORE SWITCH; a SHAPED one is refused on resume / fork.
- 🚨 **A STEP-COUNTED CADENCE is TOTAL ENV STEPS** — compare `num_timesteps` to a boundary, never `n_calls`
  (`main.train.constants.checkpoint_due`; `src/main/train/cadence_n_independence_test.py`).
- 🚨 **A fresh model is BUILT AT ONE TORCH THREAD** (`utils.torch_state_guard.single_thread_build`; the count is
  recorded as `init_num_threads`, immutable in `metadata.json`). Any new fresh-parameter site uses the same helper.
- 🚨 **Startup runs BEFORE `--compile-trainer`** (the inference service deep-copies the policy as slot templates).
- 🚨 **Every micro-batch is FULL**: `--rollout-target-samples` must be a multiple of lcm(`--batch-size`, `--n-envs`);
  only the last accumulation group can be short, and it is a FULL-weight step (the dose counts it so).
- **K9(b) `--behaviour-check`** (`fatal`): before any optimizer step the learner's log π on current-version rows must
  equal the stored behaviour log-prob (|Δ| < 1e-4), DETERMINISTICALLY — rows within a rounding margin of a declared
  selection cutoff are excluded (`agents/model/selection_sites.py`), excluded share < 0.15. Detail:
  `designs/training/learner_gates.md`.
- Stochastic actions are the KEYED DRAW (replayable; not a speed lever). `--oracle-reveal` is a resume-immutable
  DIAGNOSTIC observation mode, never production. A launcher RESUME pins to the checkpoint's commit and builds that
  checkout's env core at startup.
- The trainee's `(observation_space, action_space)` has ONE builder, `agents.training.trainee_spaces`.

## Compile (`--compile-trainer`, default ON for cuda; detail: `designs/training/compile_flags.md`)

- fp32 matmul precision `highest` is the ONLY precision (TF32 retired); every parity gate refuses any other.
- 🚨 **The learner compiles ONLY its DECLARED REGION R1** (`agents/model/compile_regions.py`), `fullgraph=True` at one
  signature; a ragged micro-batch is REFUSED. A new signature goes into `compile_regions.REGIONS` and its prewarm;
  a new R1 lever is declared in the static flags (`MicroStatic`), never by relaxing the lock.
- 🚨 **The COMPILE SENTINEL makes a silent recompile / eager fallback FATAL**: `src/agents/model/compile_control.py`
  is the ONLY runtime module that touches `torch._dynamo` — add nothing that does elsewhere.
  `compile/recompiles_after_lock` must stay 0. The startup REGION gate runs on REAL rows (the committed
  `src/agents/model/compile_parity_obs.npz`; regenerate with `python -m agents.model.compile_parity_fixture --write`
  after an obs-layout change — a stale fixture refuses); the in-run canary re-checks R1 against eager.
- 🚨 **The compile cache is the RUN's OWN** (`<run>/compile_cache/`, K3). **A new `torch.compile` site must call
  `ensure_hermetic_cache()` first** (`compile_cache_test`). HEAD runs torch >= 2.8 ONLY.

## The critic, the value side and the auxiliaries

- **`winprob` is the ONLY trainable critic** (a namespace constant); the value term is the head's BCE against the
  terminal outcome. 🚨 **A DRAW is a NOT-WIN** (`y = 0`). 🚨 **The 250-turn cap is a TERMINAL, not a truncation.**
  `signal/draw_rate` counts ties, not timeouts — watch `signal/stall_rate` / episode length. `--vf-coef`'s 0.5 default
  carries no information about a BCE and is fixed for a run's lifetime. Detail:
  `designs/training/critic_and_value_losses.md`.
- **The win-prob head is a BAROMETER, not a coach**: `--win-prob-mode shaping` is REPRESENTATION shaping only;
  `win_target` / `win_mask` are training-only keys the collector fills (`store.fill_complete`). Its PBRS routes are
  DELETED.
- **The FORK ARM** (`--fork-fraction`, default 0.0 = OFF and bit-identical) is the collector's fork phase
  (`rust_rollout/fork.py`), declared and OFF under the one-ply scope. Its rules (uniform mask on the fork step,
  prefix counted once, `--fork-crn dice_and_draws`) are `designs/training/forks.md`.
- **The ride-along heads** (`--ridealong-*`) run OUTSIDE the fold order and must stay bit-identical ON vs OFF
  (`ridealong_update_test`); their optimizers are ACQUIRED AT STARTUP (`RideAlongLifecycleViolation` otherwise).
  🚨 **Only the TRAINEE acquires — an OPPONENT load is an `InferenceMaskablePPO`** (no optimizer, no buffer; refuses
  `learn` / `train` / `save`). 🚨 **Every checkpoint load is STRICT** (`StrictCheckpointLoad`; readers use
  `agents.model.snapshot.load_checkpoint_strict`; gate `src/strict_checkpoint_load_gate_test.py`). Detail:
  `designs/training/ridealong_heads.md`.
- **The value sidecar** (`--value-sidecar`, `auto` = on): it reads the rollout buffer, so it **cannot be reconstructed
  after the fact**; one header per writer session; read it with `python -m main.ops.value_sidecar_read <run>`. A
  `--debug` A/B cannot measure its cost.

## The belief losses (detail: `designs/training/belief_losses.md`)

- 🚨 **Read a belief target through `features_extractor.belief_supervision("<key>")`, NEVER a `last_*` stash** —
  under `--belief-grad-mode label_only` the stashes are stop-grad, so a loss on them trains NOTHING while every metric
  looks normal (guard: `agents/model/belief_label_only_gate_test.py`).
- Every belief label is a TRAINING-ONLY Dict-obs key, and each builder is fail-loud. ⚠️ The mask conventions TILE
  (hidden-team masks HIDDEN slots; spread / nature-EV / hp-type mask REVEALED ones). `--intent-label-bot-weight` is
  confined to the INTENT labels by design. Structural toggles are version-checked and fresh-only; every `*_coef` is
  read back on a flagless resume. The hidden-team row is X5's `hidden_team_set` on `--opp-belief-aux-coef`.

## The recipe, step size and THE DOSE (detail: `designs/training/step_size_and_batch.md`)

- **`--arch production`** writes every `recipe.fresh` knob (`designs/production_config.json`) the argv did not TYPE;
  a fresh argv differing on an UNTYPED knob is refused (`--allow-nonproduction-recipe` consents). `recipe.fork` (E5) is
  never applied by `--arch`. 🚨 **A same-run RESTART strips `--arch`** and resolves each untyped knob by exactly one
  route; a missing value REFUSES (`RecipeRestartError`), never a parser default. Values: `src/recipe_doc_gate_test.py`
  holds `designs/endstate/design_learner_recipe.md` §3.22 to the block.
- 🚨 **`--lr`, `--batch-size`, `--n-steps` are INERT on a resume**; `--fork-lr` pins a genuine fork's LR. **A FORK
  inherits the parent's LR but not its freeze** — forking a `--fork-lr-freeze` run without a `--fork-lr` is `[ForkLR]
  FATAL` (`--allow-inherited-fork-lr` opts in).
- 🚨 **The DOSE predicts a fold's collateral**: `lr × n_epochs × optimizer steps per epoch / rollout rows`
  (`agents/training/dose.py`; `python -m main.dose <run>`; live `train/dose_rate`). A dose reading names its LR
  RECORD (`source`) — pre-`rb_` runs read from TB since the 2026-10-09 checkpoint cleanup.
- 🚨 **Read the POLICY noise scale** (`train/noise_scale_ratio_policy`); never size a batch on the total.
  `--adaptive-batch policy` moves `--grad-accum-steps` only.
- **LINEAGE** (`lineage.py`, `python -m main.lineage`): the `lineage` block is IMMUTABLE (a restart must not
  re-point the parent); RECORDED and DERIVED are independent; a fork inherits its parent's TB as a TRUNCATED prefix
  (`--no-tb-inherit` opts out).

## Offline meters in this package

- **`main.policy_drift`** is a DESCRIPTOR, not a test: its probe set is pinned by sha256 (a different set is
  refused — start a new out dir); it writes under `~/gen3ai_archive/`, never `models/`. Conditional class rates are the
  primary read. Detail: `designs/training/offline_meters.md`.
- **`stats.py`** holds the shared stateless estimators (pure NumPy, seeded bootstrap only). Two near-siblings are
  deliberately NOT merged into it — read `designs/training/offline_meters.md` before "de-duplicating".
- **`main.scaffolding_gauge` is OFFLINE only** (the in-training gauge was retired, P11d); it is meaningful only on an
  archived shaped run.
- `watchdog.start_orphan_watchdog` arms only on the `--debug` path (exits when the launching parent changes);
  launcher-managed runs never arm it.

## DELETED — do not go looking for these

The Python env core and every Python collector (U3), the shaped reward (v122), PopArt / value-dist / PBRS routes
(L1), distillation and search-as-teacher (L3, config v133; distillation is the only built route to X15 — a Rust port
is ~1-2 agent-days if ever scheduled), the counterfactual training half (L4) and the offline cf stack, `cf_audit`
and `obs_materializer` (T27 P6), the Python fork arm (L5), team-side PFSP (L4), `--showdown-port` (P11) and the
eval callbacks' `server_config` seam (T27 P6 slice 6c). A checkpoint that recorded a retired lever is refused by
`model_version/retired_levers.py`; every flag is in `designs/deleted_flags.md`.
