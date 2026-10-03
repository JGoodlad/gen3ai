# The FLAG CENSUS — every surviving trainer / launcher flag names a LIVE user (deletion pass P11)

**The rule (owner-accepted, `TECH_DEBT_BACKLOG.md` §1):** a flag survives only if it names a LIVE user — a production default or `--arch production` / `recipe.*` row, a documented SOP / runbook operator knob, a registered experiment (`EXPERIMENT_BACKLOG.md`, `designs/endstate/*`), a meter, the launcher, a recorded-field reader, or a test seam a production check depends on. Everything else is deleted along with its code, its checks and its tests, and recorded in [`designs/deleted_flags.md`](../deleted_flags.md). `src/main/flag_census_test.py` fails when a parser flag has no row here (or a row names no parser flag), so a NEW flag arrives with its live user or not at all.

**Verdicts.** `KEEP` — has a live user (named). `DELETE` — no live user: deleted by this pass (a row stays here only while its batch is unshipped). `ONE-VALUED` — can only take its default after the Python core's deletion (U3) and the winprob critic: the flag is a constant with a recorded-field story; see §3. `NEEDS-OWNER` — ambiguous, or deleting it reaches shared kernels / Rust / recorded fields; one line of evidence, the owner decides.

**Evidence columns.** `typed` = recorded argvs (`models/*/metadata.json` `original_command`, else `launcher_command`; 275 runs, measured 2026-10-03) that type the flag, over ALL runs / the LAST 40 by `saved_at`. A recent run that types every flag copies a whole argv, so a high recent count of a default-valued flag is not a use.

**Generated from the parsers** (`build_parser()` + the launcher's `build_launcher_parser()`), so no flag is missed; the verdict table below is hand-judged. Counts exclude the generated `--no-<flag>` negations (`BoolFlag`) and the `--help` action.

## 0. Counts

| | trainer flags | launcher flags | total |
|---|---|---|---|
| STARTING (main `87d3a623`) | 202 | 9 | 211 |
| DELETED by P11 so far | 30 | 0 | 30 |
| NOW | 172 | 9 | 181 |
| verdicts NOW | KEEP 163 · DELETE 0 · ONE-VALUED 0 · NEEDS-OWNER 9 | KEEP 9 | |
| ENDING of this run (the P11 hand-off is the end of `deletion_pass_manifest.md`) | same as NOW | | |

## 1. The deletions this pass makes

**Shipped** (each also has a row in [`deleted_flags.md`](../deleted_flags.md); batches B1-B5, commits in the manifest's P11 row):

* `--eval-concurrency` (B1) — no reader since P6 deleted `final_eval`; an exact prefix of `--eval-concurrency-per-worker`, so its deletion needed `allow_abbrev=False`
* `--self-play-use-cpu` (B2) — read nothing on the Rust core (two startup `emit()` lines and an unused local)
* `--predict-unrevealed-mon-moves` (B2) — a clarity alias of `--opp-belief-moves-weight 0` + `--move-belief-mode revealed`
* `--snapshot-dir` (B2) — an unused override of the pool directory; no doc, no run typed it
* `--allow-nonsample-trainee` (B2) — a research override for the closed FiLM capacity studies; the vetted-sample gate is now unconditional
* `--warmstart-consensus` (B3) — consensus DISTILLATION of teacher exploiters into an init (`warmstart.py`); exploiters are opponents never teachers (L3); 0 of 275 recorded runs typed it
* `--warmstart-battles` (B3) — the warm start's battle count
* `--warmstart-bc-steps` (B3) — the warm start's BC steps
* `--showdown-port` (B4) — `--use-bridge rust` is the only transport: no server, the `server_config` was only stored; the launcher's default injection went with it
* `--eval-workers` (B4) — the Python eval-worker pool size; read only on the `env_core != "rust"` eval branch no production run reaches
* `--eval-device` (B4) — the same Python eval worker pool
* `--eval-concurrency-per-worker` (B4) — the same Python eval worker pool
* `--env-core` (P11b (a)) — the Rust env core is the only core (the Python one was deleted in U3); the flag was one-valued. The recorded `env_core` stamp and the D4 shaped-checkpoint refusal stay, keyed on the record
* `--use-bridge` (P11b (a)) — the trainer's transport is the in-process Rust core and nothing else; no reader but `production_args.py`
* `--critic` (P11b (b)) — the win-prob critic is the only trainable critic (shaped served the deleted Python core); a constant of the namespace now (`parser/objective.py`). `--win-prob-mode` is NOT deleted: it is not one-valued (see section 2)
* `--gamma` (P11b (c)) — the discount is 1.0 and nothing else trains the win-prob critic exactly; a constant of the namespace now. NOT a research lever: no registered experiment varies it (no `recipe.fork` / EXPERIMENT_BACKLOG row, `winprob_critic_needs_unit_gamma` refused every other value)
* `--victory-value` (P11b (c)) — the win indicator magnitude, 1.0 only; a namespace constant now
* `--draw-penalty` (P11b (c)) — the win indicator pays 0.0 on a tie and a timeout; a namespace constant now
* `--terminal-indicator` (P11b (c)) — the terminal is the win indicator, ON only; a namespace constant now (its generated `--no-terminal-indicator` negation went with it)
* `--exploiter-temp-start` (P11c batch 1) — the exploiter temperature curriculum: 23 recorded runs, none of the last 40; no backlog / era / end-state row names it (X29's pool-temperature anneal is a different mechanism)
* `--exploiter-temp-end` (P11c batch 1) — the exploiter temperature curriculum
* `--exploiter-temp-anneal-frac` (P11c batch 1) — the exploiter temperature curriculum
* `--exploiter-temp-mode` (P11c batch 1) — the exploiter temperature curriculum
* `--exploiter-temp-ratchet-wr` (P11c batch 1) — the exploiter temperature curriculum
* `--exploiter-temp-ratchet-factor` (P11c batch 1) — the exploiter temperature curriculum
* `--exploiter-temp-ratchet-games` (P11c batch 1) — the exploiter temperature curriculum
* `--rollout-target-band` (P11c batch 1) — an adaptive-target hook with no controller (`set_target` had no caller outside its test); 0 recorded runs; X28 is an epoch controller
* `--rollout-trigger` (P11c batch 1) — the window trigger was the python-parity schedule: only tests selected it and its oracle (the Python core's collect) is deleted; the complete-game trigger is a constant
* `--opponent-sampling` (P11c batch 1) — `generator` was the Python core's RLPlayer stream; production and the fork arm are `keyed` only; 0 recorded runs
* `--version-pinning` (P11c batch 1) — the first staleness remedy, OFF by owner decision 2026-09-29; no backlog row revisits it; 0 recorded runs

**Planned, not yet shipped:** (none left in the DELETE column — the remaining work is the ONE-VALUED sweep and the NEEDS-OWNER rows below, sized in the manifest's P11 hand-off)


## 2. NEEDS-OWNER

* `--progress-decision-tense` — no-progress-clock fix (`designs/training/reward.md`), OFF in `production_config.json`, 0 recorded runs; read by the RUST env decl (`RustEnvDecl.decision_tense`), so deleting it reaches `src/rust_env`
* `--progress-switch-freeze` — the sibling no-progress-clock fix; same Rust reach (`RustEnvDecl.switch_freeze`), OFF, 0 recorded runs
* `--win-prob-strata-weight` — opponent-stratified win-prob BCE; documented in `training_runbook.md` as an arm of the fork experiment; 2 recorded runs, not in the backlog
* `--pair-value-route` — "available but OFF" in `designs/ARCHITECTURE.md` (v95, 14-dim pair route into the critic); 0 recorded runs; a STRUCTURAL toggle (extractor module + `ModelVersion` field), so retiring it is an L4-shaped unit with a config bump
* `--td-aux-coef` — the TD-consistency aux (`td_aux.py`); ladder arm `ai_v12_13_ladder_tdaux` (`--td-aux-coef 1.0`) NOT DETECTED at matched 10M (ledger 2026-09-09); a recorded `ModelVersion` field, OFF in `production_config.json`; 9 older recorded runs
* `--grad-checkpointing` — a bit-exact memory lever read by the compile sentinel (`lifecycle`); no recipe row, no runbook workflow, 67 older recorded runs
* `--bait-bot-share` — BaitBot joins the training roster; the bait hunt is CLOSED, but BaitBot is also a RUST bot (`rust_env/src/bots/`, `bots_gate_test`, the Python class is its oracle), so deleting it is a Rust crossing
* `--bait-bot-p` — BaitBot's pivot probability, same Rust reach as `--bait-bot-share`
* `--win-prob-mode` — **NOT one-valued** (the P11 hand-off's "the winprob critic REQUIRES `shaping`" was wrong): the only refusal is `none` (`winprob_critic_needs_a_head`), so `read_only` (the win-prob head on a STOP-GRAD value pool: the critic's gradient does not reach the trunk) is a legal winprob arm. It is a registered open question — `designs/ai_v12/design_winprob_only_critic.md` §3 "keep the flag, drop `none`" and open question 5 ("whether `--win-prob-mode read_only` is the interesting arm"), a `Family.CRITIC` row of `flag_registry.py` that `arch_surface` deliberately leaves unapplied "because an experiment varies it", a recorded `ModelVersion` field, and `ARCHITECTURE.md` §3.4 names `read_only` as the stop-grad alternative; no recorded winprob run used it (all 69 recorded `critic: winprob` configs read `shaping`; the 16 `read_only` configs are shaped-era). Deleting it removes the only trunk-exposure lever of the critic, so the owner decides; left in place by P11b

## 3. ONE-VALUED

(none left — P11b deleted all eight the P11 hand-off listed, bar `--win-prob-mode`, which turned out NOT to be one-valued and is in §2: batch (a) `--env-core`, `--use-bridge`; batch (b) `--critic`; batch (c) `--gamma`, `--victory-value`, `--draw-penalty`, `--terminal-indicator`. A deleted flag's row leaves the census for `designs/deleted_flags.md`.)

## 4. The table — every flag, in `--help` order


### operational

| flag | default | typed (all / last 40) | live user | verdict |
|---|---|---|---|---|
| `--model` | — | 190 / 33 | the resume / fork entry point of every launch (launcher, `training_runbook.md`, `TRAINING_RUN_SOP.md`) | **KEEP** |
| `--run-dir` | — | 16 / 0 | the launcher passes it on every restart (`src/main/launcher/child.py`); a same-run restart needs it | **KEEP** |
| `--run-name` | — | 248 / 40 | names a fresh run's directory (runbook; 248 of 257 recorded argvs) | **KEEP** |
| `--steps` | 100000 | 273 / 40 | the run length, typed on every launch | **KEEP** |
| `--debug` | false | 5 / 1 | the 10k-step smoke of the root `CLAUDE.md` | **KEEP** |
| `--debug-eval` | false | 0 / 0 | the root `CLAUDE.md` smoke names it as the way to exercise the eval path; read by `callbacks._run_eval` | **KEEP** |
| `--n-envs` | 32 | 268 / 39 | `recipe.sizing.n_envs` (256) in `--arch production` | **KEEP** |
| `--device` | auto | 272 / 39 | every launch (`--device cuda`) | **KEEP** |
| `--tb-inherit` | true | 0 / 0 | a fork copies its parent's scalar TB events (`tb_inherit.py`, `main.tb_inherit`, root `CLAUDE.md`'s fork notes); `--no-tb-inherit` is the documented fleet opt-out | **KEEP** |
| `--seed` | 42 | 172 / 39 | every launch (`recipe`-adjacent reproducibility; the Rust core's `segment_seed`) | **KEEP** |
| `--log-level` | periodic | 192 / 39 | root `CLAUDE.md` / launcher (`--log-level periodic`) | **KEEP** |
| `--arch` | — | 6 / 6 | THE production surface (root `CLAUDE.md`, `checkargs`) | **KEEP** |
| `--allow-nonproduction-arch` | false | 0 / 0 | the consent for a deliberate ablation (`arch_surface`, `training_runbook.md`) | **KEEP** |
| `--allow-nonproduction-recipe` | false | 0 / 0 | the consent for an untyped non-production recipe knob (`recipe_surface`, `training_runbook.md`) | **KEEP** |

### hyperparameters

| flag | default | typed (all / last 40) | live user | verdict |
|---|---|---|---|---|
| `--batch-size` | 4096 | 263 / 34 | `--arch production`: `recipe.fresh.batch_size` = `2048` | **KEEP** |
| `--grad-accum-steps` | 1 | 249 / 34 | `--arch production`: `recipe.fresh.grad_accum_steps` = `32` | **KEEP** |
| `--adaptive-batch` | off | 0 / 0 | registered experiment X7 (`EXPERIMENT_BACKLOG.md`: `--adaptive-batch policy`); the controller is built and tested | **KEEP** |
| `--adaptive-batch-target` | 1.0 | 0 / 0 | a knob of the `--adaptive-batch` controller (X7) | **KEEP** |
| `--adaptive-batch-band` | 2.0 | 0 / 0 | a knob of the `--adaptive-batch` controller (X7) | **KEEP** |
| `--adaptive-batch-min-accum` | 1 | 0 / 0 | a knob of the `--adaptive-batch` controller (X7) | **KEEP** |
| `--adaptive-batch-max-accum` | 32 | 0 / 0 | a knob of the `--adaptive-batch` controller (X7) | **KEEP** |
| `--adaptive-batch-every` | 4 | 0 / 0 | a knob of the `--adaptive-batch` controller (X7) | **KEEP** |
| `--checkpoint-every-steps` | — | 121 / 24 | the checkpoint cadence, an operator knob (`training_runbook.md`; typed by 121 recorded runs) | **KEEP** |
| `--n-epochs` | 5 | 265 / 35 | `--arch production`: `recipe.fresh,recipe.fork.n_epochs` = `10` | **KEEP** |
| `--lr` | 0.0003 | 264 / 34 | `--arch production`: `recipe.fresh.lr` = `0.0003` | **KEEP** |
| `--fork-lr` | — | 46 / 30 | `--arch production`: `recipe.fork.fork_lr` = `5.6e-05` | **KEEP** |
| `--fork-lr-freeze` | false | 46 / 30 | `--arch production`: `recipe.fork.fork_lr_freeze` = `true` | **KEEP** |
| `--allow-inherited-fork-lr` | false | 0 / 0 | the documented escape of the fork-LR guard (`training_runbook.md`, root `CLAUDE.md` `--fork-lr` hazard) | **KEEP** |
| `--min-lr` | 1e-05 | 260 / 34 | `--arch production`: `recipe.fresh.min_lr` = `1e-05` | **KEEP** |
| `--max-lr` | — | 76 / 0 | `--arch production`: `recipe.fresh.max_lr` = `null` (the production value is OFF / zero: an arch toggle held for ablations, see FINDINGS) | **KEEP** |
| `--anneal-lr-start-steps` | — | 50 / 0 | `--arch production`: `recipe.fresh.anneal_lr_start_steps` = `null` (the production value is OFF / zero: an arch toggle held for ablations, see FINDINGS) | **KEEP** |
| `--anneal-min-lr` | — | 50 / 0 | the cosine floor REQUIRED by `--anneal-lr-start-steps` (a `recipe.fresh` row, null = off; `TwoPhaseLRCallback`) | **KEEP** |
| `--ent-coef` | 0.02 | 264 / 34 | `--arch production`: `recipe.fresh.ent_coef` = `0.05` | **KEEP** |
| `--policy-gae-lambda` | — | 7 / 7 | `--arch production`: `recipe.fresh.policy_gae_lambda` = `0.8` | **KEEP** |
| `--diagnostics-every` | — | 0 / 0 | the documented telemetry cadence (`designs/training/`, `training_runbook.md`) | **KEEP** |
| `--device-batch` | — | 0 / 0 | the GPU-memory design's batch-delivery mode (`resident` / `staged` / `host`; `designs/endstate/`) | **KEEP** |
| `--vf-coef` | 0.5 | 262 / 34 | `--arch production`: `production_config.json` `vf_coef` = `0.5`; `--arch production`: `recipe.fresh.vf_coef` = `0.5` | **KEEP** |

### reward

| flag | default | typed (all / last 40) | live user | verdict |
|---|---|---|---|---|

### clean_world

| flag | default | typed (all / last 40) | live user | verdict |
|---|---|---|---|---|
| `--value-sidecar` | auto | 0 / 0 | the training-side value sidecar (`main.ops.value_sidecar_read`, `training_runbook.md`); `auto` = on under winprob | **KEEP** |
| `--value-sidecar-fraction` | 0.015625 | 1 / 0 | the sidecar's sampling share (`training_runbook.md`) | **KEEP** |
| `--value-sidecar-seed` | 0 | 0 / 0 | the sidecar's sampler seed (`training_runbook.md`) | **KEEP** |
| `--progress-decision-tense` | false | 0 / 0 | no-progress-clock fix (`designs/training/reward.md`), OFF in `production_config.json`, 0 recorded runs; read by the RUST env decl (`RustEnvDecl.decision_tense`), so deleting it reaches `src/rust_env` | **NEEDS-OWNER** |
| `--progress-switch-freeze` | false | 0 / 0 | the sibling no-progress-clock fix; same Rust reach (`RustEnvDecl.switch_freeze`), OFF, 0 recorded runs | **NEEDS-OWNER** |
| `--clip-range` | 0.15 | 261 / 34 | `--arch production`: `recipe.fresh.clip_range` = `0.15` | **KEEP** |
| `--clip-range-vf` | 0.5 | 263 / 34 | `--arch production`: `recipe.fresh.clip_range_vf` = `null` (the production value is OFF / zero: an arch toggle held for ablations, see FINDINGS) | **KEEP** |
| `--opp-belief-cls-k` | — | 186 / 34 | `--arch production`: `production_config.json` `opp_belief_cls_k` = `6` | **KEEP** |
| `--opp-belief-aux-coef` | — | 254 / 34 | `--arch production`: `production_config.json` `opp_belief_aux_coef` = `0.05`; `--arch production`: `recipe.fresh.opp_belief_aux_coef` = `0.05` | **KEEP** |
| `--opp-belief-moves-weight` | 1.0 | 191 / 34 | the moves-BCE vs species-CE weight inside the belief aux term (read by `train_setup`; default 1.0 is the production value) | **KEEP** |
| `--move-belief-mode` | — | 186 / 34 | `--arch production`: `production_config.json` `move_belief_mode` = `"both"` | **KEEP** |
| `--move-belief-coef` | — | 251 / 34 | `--arch production`: `production_config.json` `move_belief_coef` = `0.05`; `--arch production`: `recipe.fresh.move_belief_coef` = `0.05` | **KEEP** |
| `--damage-op` | — | 184 / 34 | `--arch production`: `production_config.json` `damage_op` = `true` | **KEEP** |
| `--unified-damage` | off | 185 / 34 | the parse-time macro the registry, `extractor_build` and `combination_checks` refusal texts name; `resolve_config` desugars it into the component toggles (`config.desugar_umbrella_flags`) | **KEEP** |
| `--damage-outgoing` | — | 184 / 34 | `--arch production`: `production_config.json` `damage_outgoing` = `true` | **KEEP** |
| `--move-candidate-floor` | — | 245 / 34 | `--arch production`: `production_config.json` `move_candidate_floor` = `0.02` | **KEEP** |
| `--move-prior-fusion` | — | 184 / 34 | `--arch production`: `production_config.json` `move_prior_fusion` = `true` | **KEEP** |
| `--t0-species-prior` | — | 183 / 34 | `--arch production`: `production_config.json` `t0_species_prior` = `true` | **KEEP** |
| `--species-prior-fusion` | — | 185 / 34 | `--arch production`: `production_config.json` `species_prior_fusion` = `true` | **KEEP** |
| `--compile-trainer` | — | 187 / 34 | ON by default (root `CLAUDE.md`); the learner's compiled regions | **KEEP** |
| `--consequence-topk` | — | 184 / 34 | `--arch production`: `production_config.json` `consequence_topk` = `6` | **KEEP** |
| `--entity-topk-seats` | — | 186 / 34 | `--arch production`: `production_config.json` `entity_topk_seats` = `6` | **KEEP** |
| `--entity-tail-seats` | — | 186 / 34 | `--arch production`: `production_config.json` `entity_tail_seats` = `true` | **KEEP** |
| `--edge-bias-families` | — | 186 / 34 | `--arch production`: `production_config.json` `edge_bias_families` = `"d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1,c3,c2,c5,h,r"` | **KEEP** |
| `--damage-candidate-k` | — | 184 / 34 | `--arch production`: `production_config.json` `damage_candidate_k` = `0` (the production value is OFF / zero: an arch toggle held for ablations, see FINDINGS) | **KEEP** |
| `--win-prob-mode` | — | 252 / 34 | NOT one-valued: winprob refuses only `none`, so `read_only` is a legal arm — design_winprob_only_critic §3 + open question 5, a `Family.CRITIC` registry row, a recorded `ModelVersion` field; the owner decides (section 2) | **NEEDS-OWNER** |
| `--win-prob-strata-weight` | — | 2 / 0 | opponent-stratified win-prob BCE; documented in `training_runbook.md` as an arm of the fork experiment; 2 recorded runs, not in the backlog | **NEEDS-OWNER** |
| `--ridealong-ensemble` | — | 0 / 0 | `--arch production`: `production_config.json` `ridealong_ensemble` = `0`; X26's ride-along baseline heads (`EXPERIMENT_BACKLOG.md`, `main.ridealong_read`) | **KEEP** |
| `--ridealong-rnd` | — | 0 / 0 | `--arch production`: `production_config.json` `ridealong_rnd` = `false`; X26's RND novelty head | **KEEP** |
| `--ridealong-adv` | — | 0 / 0 | `--arch production`: `production_config.json` `ridealong_adv` = `0`; X26's per-action A heads | **KEEP** |
| `--ridealong-opp` | — | 0 / 0 | `--arch production`: `production_config.json` `ridealong_opp` = `0`; X26's opponent-effect B heads | **KEEP** |
| `--ridealong-rnd-variants` | — | 0 / 0 | `--arch production`: `production_config.json` `ridealong_rnd_variants` = `"off"`; X26's RND variant ensemble | **KEEP** |
| `--fork-fraction` | — | 1 / 0 | the Rust fork arm (declared, OFF; `designs/training/forks.md` section 14 checklist gates enabling it; `designs/endstate/`) | **KEEP** |
| `--fork-branches` | — | 1 / 0 | the Rust fork arm's branch count | **KEEP** |
| `--fork-contested-gap` | — | 0 / 0 | the Rust fork arm's selector quantile | **KEEP** |
| `--fork-contested-absv` | — | 0 / 0 | the Rust fork arm's /V-0.5/ admit (OFF on purpose) | **KEEP** |
| `--fork-max-per-battle` | — | 0 / 0 | the Rust fork arm's per-episode cap | **KEEP** |
| `--fork-crn` | — | 1 / 0 | the Rust fork arm's common-random-numbers mode | **KEEP** |

### capacity

| flag | default | typed (all / last 40) | live user | verdict |
|---|---|---|---|---|
| `--capacity-telemetry` | — | 41 / 0 | `--arch production`: `production_config.json` `capacity_telemetry` = `false`; `capacity/*` early warnings (`main.capacity`, `src/agents/training/CLAUDE.md`; typed by 41 recorded runs) | **KEEP** |
| `--canary-reset-steps` | — | 0 / 0 | `--arch production`: `production_config.json` `canary_reset_steps` = `1000000`; a knob of the capacity telemetry | **KEEP** |
| `--capacity-cosine-every` | — | 0 / 0 | `--arch production`: `production_config.json` `capacity_cosine_every` = `50`; a knob of the capacity telemetry | **KEEP** |
| `--capacity-velocity-every` | — | 0 / 0 | `--arch production`: `production_config.json` `capacity_velocity_every` = `50`; a knob of the capacity telemetry | **KEEP** |

### distillation (arch + heads)

| flag | default | typed (all / last 40) | live user | verdict |
|---|---|---|---|---|
| `--rank-tripwire` | — | 36 / 5 | `--arch production`: `production_config.json` `rank_tripwire` = `"warn"`; the rank/policy_pr watchdog; default `warn` registers its callback in every run | **KEEP** |
| `--rank-tripwire-drop` | — | 0 / 0 | `--arch production`: `production_config.json` `rank_tripwire_drop` = `0.2`; the tripwire's threshold (the callback reads it) | **KEEP** |
| `--intent-move-cell` | — | 184 / 34 | `--arch production`: `production_config.json` `intent_move_cell` = `true` | **KEEP** |
| `--value-entity-pool-full` | — | 180 / 34 | `--arch production`: `production_config.json` `value_entity_pool_full` = `true` | **KEEP** |
| `--pair-outcome-cell` | — | 171 / 34 | `--arch production`: `production_config.json` `pair_outcome_cell` = `true` | **KEEP** |
| `--pair-outcome-switch` | — | 171 / 34 | `--arch production`: `production_config.json` `pair_outcome_switch` = `true` | **KEEP** |
| `--switch-branch-cell` | — | 171 / 34 | `--arch production`: `production_config.json` `switch_branch_cell` = `true` | **KEEP** |
| `--conditional-threat-cell` | — | 171 / 34 | `--arch production`: `production_config.json` `conditional_threat_cell` = `true` | **KEEP** |
| `--pair-value-route` | — | 0 / 0 | "available but OFF" in `designs/ARCHITECTURE.md` (v95, 14-dim pair route into the critic); 0 recorded runs; a STRUCTURAL toggle (extractor module + `ModelVersion` field), so retiring it is an L4-shaped unit with a config bump | **NEEDS-OWNER** |
| `--intent-threshold` | — | 180 / 34 | `--arch production`: `production_config.json` `intent_threshold` = `true` | **KEEP** |
| `--op-drop-renders` | — | 180 / 34 | `--arch production`: `production_config.json` `op_drop_renders` = `true` | **KEEP** |
| `--op-believed-lean` | — | 180 / 34 | `--arch production`: `production_config.json` `op_believed_lean` = `true` | **KEEP** |
| `--intent-conditional` | — | 180 / 34 | `--arch production`: `production_config.json` `intent_conditional` = `true` | **KEEP** |
| `--item-belief` | — | 180 / 34 | `--arch production`: `production_config.json` `item_belief` = `true` | **KEEP** |
| `--history-events` | — | 180 / 34 | `--arch production`: `production_config.json` `history_events` = `true` | **KEEP** |
| `--value-entity-pool` | — | 181 / 34 | `--arch production`: `production_config.json` `value_entity_pool` = `true` | **KEEP** |
| `--beta-setvalued-coef` | — | 184 / 34 | `--arch production`: `recipe.fresh.beta_setvalued_coef` = `0.05` | **KEEP** |
| `--intent-label-bot-weight` | — | 171 / 34 | `--arch production`: `production_config.json` `intent_label_bot_weight` = `0.25`; `--arch production`: `recipe.fresh.intent_label_bot_weight` = `0.25` | **KEEP** |
| `--opp-intent-coef` | — | 185 / 34 | `--arch production`: `recipe.fresh.opp_intent_coef` = `0.05` | **KEEP** |
| `--value-threat-inject` | — | 186 / 34 | `--arch production`: `production_config.json` `value_threat_inject` = `true` | **KEEP** |
| `--td-aux-coef` | — | 9 / 0 | the TD-consistency aux (`td_aux.py`); ladder arm `ai_v12_13_ladder_tdaux` (`--td-aux-coef 1.0`) NOT DETECTED at matched 10M (ledger 2026-09-09); a recorded `ModelVersion` field, OFF in `production_config.json`; 9 older recorded runs | **NEEDS-OWNER** |
| `--policy-grad-coef` | — | 3 / 0 | `--arch production`: `production_config.json` `policy_grad_coef` = `1.0` | **KEEP** |
| `--move-latent` | — | 184 / 34 | `--arch production`: `production_config.json` `move_latent` = `true` | **KEEP** |
| `--move-belief-latent-coef` | — | 184 / 34 | `--arch production`: `production_config.json` `move_belief_latent_coef` = `0.05`; `--arch production`: `recipe.fresh.move_belief_latent_coef` = `0.05` | **KEEP** |
| `--unified-moves` | — | 250 / 34 | a flagless fresh argv resolves it to `both`, which turns `--damage-op` / `--move-latent` ON (`config.desugar_umbrella_flags`) — the BARE-argv default's route to the unified move system | **KEEP** |
| `--damage-topk` | — | 246 / 34 | `--arch production`: `production_config.json` `damage_topk_k` = `6` | **KEEP** |
| `--damage-matrices` | — | 244 / 34 | the ONLY CLI spelling of the two `production_config.json` toggles `damage_matrices_incoming` / `damage_matrices_outgoing` (registry rows `cli_name`, `checkargs`) | **KEEP** |
| `--spread-belief` | — | 250 / 34 | `--arch production`: `production_config.json` `spread_belief` = `true` | **KEEP** |
| `--spread-belief-coef` | — | 243 / 34 | `--arch production`: `production_config.json` `spread_belief_coef` = `0.05`; `--arch production`: `recipe.fresh.spread_belief_coef` = `0.05` | **KEEP** |
| `--spread-belief-nature` | — | 236 / 34 | `--arch production`: `production_config.json` `spread_belief_nature` = `true` | **KEEP** |
| `--hp-belief-mode` | — | 184 / 34 | `--arch production`: `production_config.json` `hp_belief_mode` = `"composed"` | **KEEP** |
| `--hp-type-belief-coef` | — | 241 / 34 | `--arch production`: `production_config.json` `hp_type_belief_coef` = `0.05`; `--arch production`: `recipe.fresh.hp_type_belief_coef` = `0.05` | **KEEP** |
| `--item-belief-coef` | — | 0 / 0 | `--arch production`: `production_config.json` `item_belief_coef` = `0.05`; `--arch production`: `recipe.fresh.item_belief_coef` = `0.05` | **KEEP** |
| `--allow-belief-grad-mode-change` | false | 0 / 0 | the documented escape of the v41 belief-grad-mode resume gate (`designs/training/belief_losses.md`) | **KEEP** |
| `--belief-grad-mode` | — | 237 / 34 | `--arch production`: `production_config.json` `belief_grad_mode` = `"shaping"` | **KEEP** |
| `--n-steps` | 2048 | 266 / 36 | `--arch production`: `recipe.sizing.n_steps` = `384`; `recipe.sizing.n_steps` (384) | **KEEP** |
| `--grad-checkpointing` | false | 67 / 0 | a bit-exact memory lever read by the compile sentinel (`lifecycle`); no recipe row, no runbook workflow, 67 older recorded runs | **NEEDS-OWNER** |
| `--weight-decay` | 1e-05 | 184 / 34 | `--arch production`: `recipe.fresh.weight_decay` = `1e-05`; `recipe.fresh.weight_decay` | **KEEP** |

### eval_subprocess

| flag | default | typed (all / last 40) | live user | verdict |
|---|---|---|---|---|
| `--eval-shard-games` | 25 | 184 / 34 | sizes the Rust eval core's shard units (`rust_eval`, `training_runbook.md`) | **KEEP** |
| `--bait-bot-share` | 0.0 | 3 / 0 | BaitBot joins the training roster; the bait hunt is CLOSED, but BaitBot is also a RUST bot (`rust_env/src/bots/`, `bots_gate_test`, the Python class is its oracle), so deleting it is a Rust crossing | **NEEDS-OWNER** |
| `--bait-bot-p` | 0.6 | 3 / 0 | BaitBot's pivot probability, same Rust reach as `--bait-bot-share` | **NEEDS-OWNER** |
| `--eval-freq` | — | 5 / 0 | the eval cadence (every real run) | **KEEP** |
| `--eval-games` | — | 26 / 0 | games per opponent per eval cycle (`design_ladder_campaign.md` types `--eval-games 200`) | **KEEP** |
| `--snapshot-ladder-games` | 100 | 184 / 34 | the per-promotion ladder tax (`ladder.json` is the headline ELO) | **KEEP** |
| `--forensic-win-quota` | 5 | 11 / 0 | the eval trace quota the prober reads (root `CLAUDE.md`) | **KEEP** |
| `--forensic-loss-quota` | 10 | 11 / 0 | the eval trace quota the prober reads (root `CLAUDE.md`) | **KEEP** |
| `--forensic-draw-quota` | 5 | 11 / 0 | the eval trace quota the prober reads (root `CLAUDE.md`) | **KEEP** |
| `--keep-eval-snapshots` | 10 | 184 / 34 | artifact retention (`artifact_retention.py`, run-dir disk bound) | **KEEP** |
| `--keep-eval-trace-steps` | 0 | 184 / 34 | artifact retention | **KEEP** |
| `--keep-stalls` | 50 | 184 / 34 | artifact retention | **KEEP** |
| `--keep-crashes` | 10 | 184 / 34 | artifact retention | **KEEP** |
| `--self-play` | false | 167 / 22 | `--arch production`: `recipe.fresh.self_play` = `true`; the self-play pool (recipe.fresh `self_play`) | **KEEP** |
| `--fork-pool-seed` | true | 0 / 0 | a fork auto-seeds its parent's pool (root `CLAUDE.md` fork hazard) | **KEEP** |
| `--allow-empty-pool` | false | 0 / 0 | the consent for a fork with an empty pool (root `CLAUDE.md`, `supply_guards.md`) | **KEEP** |
| `--supply-starve-cycles` | — | 0 / 0 | the declared lever-supply floors (`designs/training/supply_guards.md`) | **KEEP** |
| `--promote-threshold` | — | 1 / 0 | the pool-promotion gate (`TASK_BACKLOG.md` T6, `training_runbook.md`) | **KEEP** |
| `--eval-sentinel-greedy` | — | 92 / 34 | the eval regime (root `CLAUDE.md` opponent-regime boundary; recorded and inherited) | **KEEP** |
| `--eval-mirrored-pairs` | — | 0 / 0 | T17, BUILT and default OFF, flipped at the X26 baseline (`TASK_BACKLOG.md`) | **KEEP** |
| `--promotion-sprt` | — | 0 / 0 | T6, BUILT and default OFF, flipped with T17 (`TASK_BACKLOG.md`) | **KEEP** |
| `--self-play-temp` | 1.0 | 114 / 20 | the pool opponents' sampling temperature (read by the self-play and eval callbacks) | **KEEP** |
| `--bot-weights` | — | 75 / 0 | biases the heuristic-opponent pick (`resolve_bot_weights`; `designs/training/`) | **KEEP** |
| `--heuristic-floor` | — | 75 / 0 | the self-play curriculum floor (`snapshot_pool.heuristic_fraction`) | **KEEP** |
| `--self-play-start-wr` | — | 0 / 0 | the first-promotion threshold (`TECH_DEBT_BACKLOG.md` P2 names it) | **KEEP** |
| `--self-play-full-wr` | — | 0 / 0 | the curriculum's saturation threshold (`snapshot_pool.heuristic_fraction`) | **KEEP** |
| `--pfsp-scale` | 0.0 | 44 / 0 | the ARM-C lever (`design_ladder_campaign.md`, era plan row 1) | **KEEP** |
| `--n-sentinels` | 5 | 157 / 34 | pool sentinels evaluated per cycle (`rust_eval.build`) | **KEEP** |
| `--pool-spread` | false | 30 / 0 | spread retention of the pool (`design_ladder_campaign.md`) | **KEEP** |
| `--team-block-episodes` | 1 | 196 / 34 | team-blocked trainee draws (`design_ladder_campaign.md`) | **KEEP** |
| `--team-wr-tracking` | true | 0 / 0 | the per-team win-rate callback (ON by default; `team_win_rates.json`) | **KEEP** |
| `--stable-opponents` | — | 46 / 10 | the cross-run opponent window of the population loop (era plan row 1) | **KEEP** |
| `--stable-opponent-temp` | 1.0 | 167 / 20 | the stable opponents' temperature | **KEEP** |
| `--stable-opponent-mastered-wr` | 0.8 | 167 / 20 | the mastery threshold of the stable window | **KEEP** |
| `--stable-opponent-selfplay-share` | 0.2 | 191 / 20 | the stable-window share (0.40 in the new lineage) | **KEEP** |
| `--stable-opponent-pfsp` | false | 17 / 5 | the PFSP selection inside the stable window | **KEEP** |
| `--exploiter` | — | 94 / 12 | the exploiter of the population loop (era plan row 1; `main.best_response_gap`) | **KEEP** |
| `--exploiter-keep-bots` | false | 40 / 12 | the exploiter's bot-mix floor (`best_response_gap`) | **KEEP** |
| `--exploiter-bot-fraction` | 0.5 | 114 / 34 | the exploiter's bot-mix fraction | **KEEP** |
| `--trainee-team` | — | 17 / 0 | SPECIALIST mode (`TECH_DEBT_BACKLOG.md` P1, `eval_trace_gen`, `promote_teams`) | **KEEP** |
| `--trainee-teams` | — | 83 / 14 | MULTI-SPECIALIST mode (`main.promote_teams`) | **KEEP** |
| `--allow-untaught-teacher` | false | 0 / 0 | the documented override of the untaught-slice guard (`matchup_setup`); the untaught meter is live | **KEEP** |

### env_core

| flag | default | typed (all / last 40) | live user | verdict |
|---|---|---|---|---|
| `--rollout-target-samples` | — | 0 / 0 | `--arch production`: `recipe.sizing.rollout_target_samples` = `98304`; `recipe.sizing.rollout_target_samples` (98,304) | **KEEP** |
| `--rust-env-front` | — | 0 / 0 | `proc` (production) vs `ffi` (`rust_collector.md`) | **KEEP** |
| `--rust-env-threads` | — | 0 / 0 | the Rust core's worker threads (`rust_collector.md`) | **KEEP** |
| `--rust-env-profile` | — | 0 / 0 | `selfcheck` is the build every test run uses (`rust_collector.md`) | **KEEP** |
| `--rust-env-refusal-budget` | — | 0 / 0 | a production FATAL budget (`rust_collector.md`) | **KEEP** |
| `--rust-env-respawn-budget` | — | 0 / 0 | a production FATAL budget (`rust_collector.md`) | **KEEP** |
| `--trainee-slots` | — | 0 / 0 | named by `recipe.sizing.trainee_slots` (`null` = 1) in `--arch production`. **FINDING (P11c batch 1):** its only non-default use was `--version-pinning per_game`, deleted there — above 1 it now reserves T2 trainee slots nothing plays. The list is closed (standing rule 9), so it stays KEEP; it is the next one-valued candidate for the owner | **KEEP** |
| `--t2-buckets` | — | 0 / 0 | `--arch production`: `recipe.sizing.t2_buckets` = `null`; `recipe.sizing.t2_buckets` | **KEEP** |
| `--t2-opponent-bucket-cap` | — | 0 / 0 | the T2 graph-pool memory cap (`gen3_slot_bucket_caps_v1`) | **KEEP** |
| `--t2-lanes` | — | 0 / 0 | `--arch production`: `recipe.sizing.t2_lanes` = `null`; `recipe.sizing.t2_lanes` | **KEEP** |
| `--t2-backend` | — | 0 / 0 | `graph` / `eager` / `aot` inference backends (`inference/service`) | **KEEP** |
| `--rust-eval-envs` | — | 0 / 0 | the Rust eval core's env count (`rust_eval.build`) | **KEEP** |
| `--behaviour-check` | — | 0 / 0 | K9(b), default `fatal` (`rust_rollout/consistency.py`) | **KEEP** |

### launcher (`main/launcher/run.py` `build_launcher_parser`)

| flag | default | typed (all / last 40) | live user | verdict |
|---|---|---|---|---|
| `--restart-interval-hours` | 3.0 | — | the periodic full-process restart (`TRAINING_RUN_SOP.md`, root `CLAUDE.md`) | **KEEP** |
| `--restart-grace-minutes` | 20.0 | — | the scheduled restart's fallback window (`src/main/launcher/CLAUDE.md`) | **KEEP** |
| `--max-crash-restarts` | 3 | — | the crash auto-restart breaker (`src/main/launcher/CLAUDE.md`) | **KEEP** |
| `--nice` | 10 | — | every run is niced (root `CLAUDE.md`) | **KEEP** |
| `--no-pin` | False | — | documented: skip worktree pinning (`src/main/launcher/CLAUDE.md`) | **KEEP** |
| `--dry-run` | False | — | THE offline resolve of a launch (root `CLAUDE.md`, `training_runbook.md`) | **KEEP** |
| `--sync-to-main` | False | — | documented: pin a resume to HEAD (`src/main/launcher/CLAUDE.md`) | **KEEP** |
| `--pin-commit` | None | — | documented: pin a batch to one commit (`--pin-to-hash` is its legacy spelling) | **KEEP** |
| `--allow-torch-switch` | False | — | the consent for a resume under another torch (root `CLAUDE.md`) | **KEEP** |
