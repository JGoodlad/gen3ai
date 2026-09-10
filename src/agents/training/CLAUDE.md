# CLAUDE.md — Training (`src/agents/training/`)

Callbacks, reward manager, episode/turn tracking, stall detection, the belief/value auxiliaries and
the bot-eval pipeline. **How to launch training** (commands, flags) lives in the root `CLAUDE.md`
→ Training / Launcher and in [`designs/ops/training_runbook.md`](../../../designs/ops/training_runbook.md);
this file documents the subsystems' internal design. The `TurnDelta` fold and the
LiveView/TurnView/LegalActions read-models it consumes are in `src/agents/battle/CLAUDE.md`; the
obs-build performance gate is in `src/agents/observation/CLAUDE.md`.

## Where the detail is — the topic map

**This leaf keeps each subsystem's heading, the rules and hazards an agent must know BEFORE
touching it, and a pointer. `designs/training/` owns the detail and carries the same
always-current obligation as this file — update the topic doc in the same pass as the code.**

| I am about to touch… | Read |
|---|---|
| the reward registry, PBRS, the no-progress clock, the two entropy boosts | [`designs/training/reward.md`](../../../designs/training/reward.md) |
| the PPO package's module map or a source-level pin on `train()` | [`designs/training/ppo_step.md`](../../../designs/training/ppo_step.md) |
| bot eval, the untaught meter, the critic gate, ELO / the ladder / Hodge, the baseline registry | [`designs/training/eval_and_rating.md`](../../../designs/training/eval_and_rating.md) |
| self-play, the snapshot pool, stable opponents | [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md) |
| exploiter mode, the warm start, distillation + the off-slice anchor | [`designs/training/exploiter_and_distillation.md`](../../../designs/training/exploiter_and_distillation.md) |
| team-side PFSP, per-team win-rate tracking | [`designs/training/team_curriculum.md`](../../../designs/training/team_curriculum.md) |
| `--critic`, PopArt, the value-tail weight, TD-aux, the value-dist head, the 250-turn cap | [`designs/training/critic_and_value_losses.md`](../../../designs/training/critic_and_value_losses.md) |
| the win-prob head, win-prob PBRS, either frozen-φ route | [`designs/training/winprob_head_and_pbrs.md`](../../../designs/training/winprob_head_and_pbrs.md) |
| the training-side value sidecar, its two flags, its cost, `main.ops.value_sidecar_read` | [`designs/training/value_sidecar.md`](../../../designs/training/value_sidecar.md) |
| any supervised belief loss, or the opponent-class label weight | [`designs/training/belief_losses.md`](../../../designs/training/belief_losses.md) |
| gradient accumulation, the noise scale, the DOSE, `--fork-lr`, `--adaptive-batch` | [`designs/training/step_size_and_batch.md`](../../../designs/training/step_size_and_batch.md) |
| the MatchupSpec, run-spec resolution provenance, LINEAGE, TB inheritance | [`designs/training/matchup_and_lineage.md`](../../../designs/training/matchup_and_lineage.md) |
| the TB census detail, capacity telemetry, grad balance, `signal/`, the scaffolding gauge | [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md) |
| the counterfactual audit, the cf label plumbing, the prefix-sharing materializer | [`designs/training/cf_grounding.md`](../../../designs/training/cf_grounding.md) |
| search-as-teacher, OPD, the win-prob one-ply teacher | [`designs/training/search_teacher.md`](../../../designs/training/search_teacher.md) |
| the stall-tail harvest / head-repair pipeline | [`designs/training/stall_tail_harvest.md`](../../../designs/training/stall_tail_harvest.md) |
| either `--compile-*` flag, BLAS pinning | [`designs/training/compile_flags.md`](../../../designs/training/compile_flags.md) |
| `stats.py`, the replay-imputation probe | [`designs/training/offline_meters.md`](../../../designs/training/offline_meters.md) |

Closed history — **do not update it, and do not re-derive a plan from it**:
`designs/research_state/claude_md_archive/training_leaf_faint_attribution_history.md` and
`designs/research_state/claude_md_archive/training_leaf_deleted_subsystems_history.md`.

## TensorBoard export census — every scalar, and its CURRENCY

> 🔒 **Two anchors here are PINNED to this file** by
> `tb_relevance_test.py::test_the_census_table_carries_an_era_column`: the string
> `gen3_tb_relevance_v1` and the heading *What to watch on a WIN-PROB run*. Keep both. Everything
> beneath them — the site/tag counts, the group table, the NOISE / REDUNDANT / CONDITIONAL
> classification, the five per-group tag tables and the annotated dashboard — is in
> [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).

**THE FIRST QUESTION ABOUT ANY SCALAR HERE IS WHAT UNIT IT IS IN**, because this trainer runs
value quantities in **four different currencies at once** and three of them look like floats:

| currency | is | who is in it |
|---|---|---|
| **RAW REWARD** | the units `--victory-value` is in, undiscounted, pre-PopArt | every `reward/*` term, `--draw-penalty` |
| **RAW SHAPED RETURN** | `Σγᵏr` in raw-reward units | `train/return_*`, `rollout_buffer.{values,returns}`, `train/explained_variance` |
| **POPART-NORMALIZED RETURN** | `(raw − μ)/σ`, σ moving over the run | `train/value_loss`, `signal/adv_*`, the value-dist support, every `distill/*_value_mse` |
| **PROBABILITY** | `[0, 1]`, outcome units, undiscounted | every `win_prob/*`, `cf/*` labels, `eval/win_rate_*` |
| ⚠️ **PROBABILITY, under `--critic winprob`** | the same `[0,1]`, but it is now ALSO what `rollout_buffer.values` / `returns` / `train/explained_variance` are in | the row above **plus** `train/return_*`, `train/explained_variance`, `train/value_loss` (unnormalized — PopArt is refused) |

⚠️ **A number is only comparable to another number in the SAME currency**, and the two most
frequently confused pairs are `train/return_std` (raw) against `popart/sigma` (the estimate OF it,
also raw — these two SHOULD track), and `train/value_loss` (normalized, ≈O(1)) against
`train/return_abs_max` (raw, ~30). WHEN POPART IS ON the conversion in force is `popart/mu` and
`popart/sigma`, and whether it is CURRENT is `popart/norm_return_*` (below); with PopArt off — which
`--critic winprob` REFUSES it into — there is no conversion and `train/value_loss` is already raw. Full background:
`designs/learning/popart_value_scale_and_currencies.md`.

🚨 **`--critic winprob` COLLAPSES the four currencies into one, which changes what several tags
MEAN without changing their names** (`gen3_winprob_critic_mode_v1`). The reward is the terminal WIN
INDICATOR, `V(s) = sigmoid(win_head logit)` and PopArt is refused — so `train/return_mean` reads a
win RATE, `train/value_loss` is an unnormalized MSE in probability units (a diagnostic; its term is
dropped from the loss), `train/explained_variance` is EV in the P(win) currency, and the
POPART-NORMALIZED row of the table above is empty because there is no normalizer. **A `winprob`
run's `train/*` value tags are not comparable with a `shaped` run's**, and nothing in the tag names
says so — read the run's `🎯 [CRITIC]` startup line first. The one tag that IS comparable across
the two is the `win_prob/` family, which was in probability units all along.

**ERA RELEVANCE — a tag whose SOURCE is absent is not emitted (`gen3_tb_relevance_v1`).** The
`--critic winprob` era changes no tag NAME but removes the SOURCE behind several of them, and the
recorders kept publishing — flat constants and byte-identical duplicates a reader cannot tell from
a measurement. Classified from the live arm's own tfevents (216 tags): **166 LIVE · 31 NOISE, all
now GATED · 19 REDUNDANT · 0 DEAD**, plus ~48 correctly-silent CONDITIONAL ones. 🚨 **The gate is
on the SOURCE, never on the value**, so a shaped run's tag set stays byte-identical and a dead
source leaves a GAP rather than a confident number. ⚠️ **A run pinned before
`gen3_obs_margin_unconditional_v1` carries a degenerate `win_margin`** — read its whole
`win_prob/*contested*` family as absent.

### What to watch on a WIN-PROB run — the 28-tag dashboard

The first two blocks are in PROBABILITY units, which is the era's whole point. Read the run's
`🎯 [CRITIC]` startup line first — a `shaped` run's `train/*` value tags are not comparable with
these. Each tag's currency and reading rule: the topic doc.

**Is it getting stronger?** `eval/elo` + `eval/elo_ci` (the only cross-run number) · `eval/win_rate_vs_bots` (saturates — read a FALL as an alarm) · `eval/win_rate_vs_pool` (pinned near 0.50 by the gate; leaving it is the news) · `signal/outcome_win_rate_bots` · `_pool` · `rollout/ep_rew_mean`.

**Is the critic HONEST?** **`win_prob/critic_resolution`** (the G1 PRIMARY meter, HIGHER is better) · `critic_reliability` · `critic_brier` · `critic_skill` · `critic_uncertainty` · `critic_base_rate` · `critic_decomp_residual` (must sit at ≈0) · `win_prob/ece` · `mce` · `rel_gap_b0…b9` (a NaN bin is a HOLE, never a zero error) · **`win_prob/start_gap`** (the PAIRED episode-start read; **positive = optimistic at the opening board**) · `train/explained_variance` · `win_prob/coverage` (a fall here invalidates every line above it).

**Is the OPTIMIZATION healthy?** `train/approx_kl` · `train/learning_rate` + **`train/dose_rate`** · `clip_fraction` · `grad_norm` · **`grad/value_policy_logratio`** · `grad/{policy,value,aux}_share` · `train/noise_scale_ratio_policy` (read BESIDE `train/noise_scale_ratio`, never alone) · `signal/adv_raw_mean` **as a ratio to** `adv_raw_std`.

**The G7 KILL condition and the two GIGO guards.** **`rollout/ep_len_mean`** and **`signal/draw_rate`** are PRIMARY endpoints, not monitored ones — a `[0,1]` critic cannot represent "a timeout is worse than a loss", so stalling is this era's registered failure mode. **`reward/untracked_abs_mean` must read exactly 0.0** (the startup composition census and the reward folds agree) and **`train/return_abs_max` must read exactly 1.0** (the reward stream IS the win indicator the `V = P(win)` identity rests on). Plus `time/fps`.

**What to read when the contested split is ABSENT:** the family is gated off rather than duplicated
on a spread-free margin, so the contested-vs-blowout question is answered offline by
`python -m main.scaffolding_gauge --reliability --reliability-reweight` and by `main.critic_gate`.

## The PPO step (`instrumented_ppo/`) — and the FOLD ORDER contract

`instrumented_ppo` is a PACKAGE whose `__init__.py` is a pure re-export hub; the module map and the
two source-pin rules (a pin that says "in `train()`" should read `ppo.train_step_source()`; a
monkeypatch follows the SYMBOL) are in
[`designs/training/ppo_step.md`](../../../designs/training/ppo_step.md). **The contract below stays
here, because it is the thing a fold edit must not get wrong.**

**THE FOLD SEQUENCE is deliberately NOT split**, and the reason is the contract below. `train()` is
~1,220 lines in one module — of which the epoch loop is ~1,020 — because the ORDER the terms are
folded in is straight-line source order, and that is only checkable by reading while it stays one
straight line. What DID move out is everything AROUND the sequence: the pre-loop setup
(`train_setup`) and the metrics export (`metrics_export`), neither of which folds a term, plus the
per-rollout probes (`rollout_probes`), which `train()` does not call at all. `ppo.py` is **1,331
lines** — its floor with the loop intact is ~1,200, so the file-size ratchet's 1,000-line TARGET is
unreachable here without splitting the sequence, which is the thing that must not happen.

Per minibatch:

1. the upstream PPO loss (`policy_grad_coef·policy_loss + ent_coef·entropy + vf_term` — `--policy-grad-coef`
   scales ONLY the clipped surrogate, never entropy/value/aux; at the 1.0 default the UNSCALED
   `policy_loss` tensor is used, byte-identical to upstream, and 0.0 removes the policy-gradient
   term alone — the arm-F pure-distill/aux phase. Training-only, the `td_aux_coef` provenance
   class: recorded, `_resolve`-inherited on a flagless resume, never gated)
2. the belief bank — species/moves aux, opponent intent (+ set-valued β), move / spread /
   nature-EV / HP-type / item belief, move-latent
3. the win-prob BCE, then the CF-twin on-policy mirror
4. the value-dist HL-Gauss CE
5. the distill family — the policy term (full KL, or the top-K/action-CE form with the optional
   advantage gate under `--distill-target action` — gen3_distill_target_gate_v1), value MSE, the
   value-feature hint
6. search-teacher AWR, then OPD
7. **TD-AUX**
8. **the counterfactual block** — cf-winprob, cf-evidential, cf-twin, cf-shadow, **q-winprob**

**No flag combination reorders these.** Each term is guarded by its own `if <x>_on:`; a term that
is off contributes nothing and moves no one. **Steps 7 and 8 are last because they each run their
OWN extractor forward, which CLOBBERS the minibatch's stashes** (`last_win_prob_logits`,
`last_spread_belief`, …) that steps 2-4 read. Moving a stash-reading fold below step 7 does not
crash — it silently scores the wrong states. `instrumented_ppo_hub_contract_test.py` pins the
7-before-8 half by reading the source, along with the mixin base list (a dropped mixin removes a
whole family of loss terms without breaking an import) and `MaskablePPO` staying LAST in the MRO
(or `_excluded_save_params`'s `super()` stops reaching upstream and checkpoints start pickling a
`threading.Lock`). It also walks the package's own import graph TRANSITIVELY from the hub, so a
module reached only through `train_setup` still counts as reachable — requiring a direct edge from
`__init__`/`ppo` would forbid a decomposition rather than check one.

## Reward redesign — registry + PBRS + the no-progress clock (`reward_manager.py`, `progress_clock.py`)

**Where the reward lives — six modules, one import path.** `reward_manager.py` re-exports every
public name the other five declare (`reward_bias_terms` · `reward_config` — the `_REGISTRY`, the
reward's source of truth · `reward_potentials` · `reward_composition` · `reward_weights`), plus
`reward_verify.py` (the `GEN3AI_REWARD_VERIFY=1` shadow twin) and `progress_clock.py`.
🚨 **Changing a value in `reward_weights.py` is a RETRAIN-class change, not a knob.**
🚨 **THE SEQUENCE IS NOT SPLIT, AND THAT IS THE DESIGN** — the ORDER `process_turn_reward` folds
its terms in is a CONTRACT and stays one straight line there, exactly like `instrumented_ppo/ppo.py`'s
minibatch fold. 🚨 **A PATCH TARGET FOLLOWS THE SYMBOL** (`_encode_incoming_block` is read in
`reward_potentials`, not `reward_manager`); `src/test_stub_vacuity_gate_test.py` fails a stale one
rather than letting it pass.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/reward.md`](../../../designs/training/reward.md).**

## State-conditioned defensive-exploration entropy (`--defensive-entropy-boost`)

`gen3_defensive_entropy_v1` — multiply the per-decision entropy bonus on decisions where the env
flags a *productive* defensive option (`defensive_opportunity`), so the model EXPLORES heals more
without any reward change. **The mechanism is ORTHOGONAL to the reward**: it surfaces the option and
changes nothing about its value, so the existing anti-stall pressure stays the guardrail.
`--defensive-entropy-boost 1.0` (the default) is OFF and byte-identical;
`--defensive-entropy-anneal-frac` fades it. Training-only, NOT version-locked, settable on resume.
⚠️ Watch the stall-rate canary.
**Full detail — in [`designs/training/reward.md`](../../../designs/training/reward.md).**

## State-conditioned BAIT-exploration entropy (`--bait-entropy-boost`)

`gen3_bait_entropy_v1` — the same mechanism on a different flag, and it exists to answer ONE
question. The bait verdict (ledger *E4 VERDICT*, 2026-08-23) closed the hunt with a stated
mechanism — **exploration starvation at a saturated action**: the whiff sits at p≈0.97, so the
alternatives at p≈0.01–0.03 are never sampled and their advantage is never realized. Everything
upstream of the action was already cleared. This flag tests the mechanism's own claim — that the
policy would fix this if it merely SAMPLED the alternatives.
**Full detail — in [`designs/training/reward.md`](../../../designs/training/reward.md).**

## MatchupSpec — the declared matchup (`matchup_spec.py`)

**The ONE explicit declaration of what a run's battles look like** — built ONCE in `train_rl_agent`
(`MatchupSpec.from_args(args)`), then CONSUMED, never re-derived (the `plan.json` pattern). It
exists because one week produced four independent failures with a shared root: *the matchup a run
plays was assembled implicitly across seams that nothing forced to agree*.

🚨 **`spec_hash()` is a MEASUREMENT-REGIME TAG: two runs or eras with different hashes are NOT
metric-comparable.** It is stamped into `metadata.json`, every `eval_results.jsonl` row,
`eval_manifest.json`, each checkpoint sidecar and `snapshot_history`; a `--model` resume whose
declared matchup differs prints `⚠️ [MATCHUP DRIFT]` with the field-level diff. **The two sides are
independent BY CONSTRUCTION** (`trainee_teams` / `opponent_teams`), so the mirror-bug class is
structurally closed.
**Full detail — in [`designs/training/matchup_and_lineage.md`](../../../designs/training/matchup_and_lineage.md).**

## Faint attribution in the trace (`gen3_faint_attribution_v1`)

`BattleRecorder` names the newly-fainted species by a SET DIFFERENCE over the two snapshots'
`*_fainted_species`, never by labelling a count change with the mon that was active at the
DECISION — and it emits one event per species, because **one side can lose two mons in a turn**.
Gate: `poke_env_gaps/faint_attribution_fuzz_test.py`, validated against the sim's own protocol log.
⚠️ **A protocol identifier carries the NICKNAME, not the species** — this pool holds teams with
localized nicknames (`Triopikeur` = Dugtrio), so any protocol-vs-our-data comparison must resolve
identifiers through poke-env's `battle.team` map. ⚠️ **A forensic recorder must never take down a
run**, which is why a mis-read falls back to a slightly-wrong label rather than raising. The
measured defect and its revert numbers are CLOSED history:
`designs/research_state/claude_md_archive/training_leaf_faint_attribution_history.md`.

## THE BASELINE REGISTRY (`baselines.py` · `designs/baselines.json` · `python -m main.baselines`)

**A baseline is the thing a result is read AGAINST, and this module is the ONE accessor over the
named set** (`gen3_baselines_registry_v1`). 🚨 **Read a baseline BY NAME, never by copying a path**
— `production`, `v9_long_baseline`, `v9_fold_parent`, `famine_comparator`,
`untaught_meter_opponent` and friends. Torch-free and offline; `resolve()` is the only call that
touches `models/`.

```python
from agents.training import baselines
baselines.get("v9_fold_parent").spec        # "ai_v9_59_R2ACTION_0827/final_model.zip"
baselines.resolve("famine_comparator")      # through fixed_opponent_pool.resolve_model_ref
baselines.protected_files()                 # {run: [rel path]} — the grooming keep-list
```

🚨 **Every entry is EXPLICIT** (a `.zip`, a `.json` or an `@step`, never a bare run dir), so the
last-snapshot rule below cannot move what a name points at while its run keeps training.
🚨 **A NEW OPPONENT IS A RE-MEASUREMENT, NOT A RENAME** — untaught-meter levels are not comparable
across opponents, so `python -m main.baselines set <name> <file> --reason "<ledger title>"` is the
only legal edit, and it PRINTS the ledger line to append rather than writing one.
**Full detail — in [`designs/training/eval_and_rating.md`](../../../designs/training/eval_and_rating.md).**

## Bot evaluation (subprocess, non-blocking)

**Flat schedule, full roster, non-blocking.** Eval fires every `EVAL_FREQ_STEPS` (2M) for
`EVAL_GAMES` (100) games per opponent (`--eval-games N` overrides), uniformly over the eight
archetype bots plus `random` and every self-play sentinel — no maturity tiers, no per-opponent
caps, no roster flag. It **skips a cycle while the previous one is still running**, so a heavier
roster self-throttles instead of needing tuned ceilings.

🚨 **THE FORENSIC TRACE'S RESULT VOCABULARY is `WIN` | `LOSS` | `DRAW`** (`gen3_trace_result_v2`,
`trace_result.py`); a `DRAW` carries `meta.draw_kind` (`timeout` vs `tie`), and an unknown result is
REFUSED rather than coerced. A timeout arrives wearing a LOSS's flags, which is how it went
unnoticed for the whole archive. 🚨 **The capture quota PREFERS LOSSES and draws have their OWN
bucket** — a trace tree is a loss-enriched sample by design, each cycle's manifest states the rule
in words, and a tree that records none is SELECTION UNKNOWN, never uniform.
**Full detail — in [`designs/training/eval_and_rating.md`](../../../designs/training/eval_and_rating.md).**

## Self-play opponents (`--self-play`, gated behind pathology hunting)

`SelfPlayCallback` replaces `PerOpponentEvalCallback` and the training opponents become frozen
snapshots of the agent itself, drawn from a directory-backed `SnapshotPool` (`snapshot_pool.py`;
state reconstructed from `<run_dir>/snapshots/` on every restart — no manifest). Design:
`designs/ai_v5/`. 🚨 **A FORK starts with an EMPTY pool, and an empty pool does not disable
`--self-play` — it falls back to the BOT pool**; a genuine fork auto-seeds its parent's and exits
`FATAL_CONFIG` if it still has none (`pool_seed.py`).
**Full detail — in [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md).**

## WHICH FILE a run spec names — the ONE resolution rule (`gen3_last_snapshot_resolution_v1`)

**A bare run directory resolves to the run's LAST SNAPSHOT, not to `best_model/best_model.zip`.**
Owner ruling, 2026-09-06. `best_model` is exported on **BOT win rate** — an opponent set with
nothing to do with what a teacher is being distilled FOR — and probe H8 measured the consequence:
for 2 of 8 unfunded R5F teachers the exported file was a ~0.93M-step exploiter rather than the
~2.93M final, and **nothing recorded which file was used**. Every meter this programme banks scores
a run at its END, so the last snapshot is what the metrics already measure.

### The rungs, for a BARE run dir (no `@step`)

| # | rung | file |
|---|---|---|
| 1 | `latest_txt` | `<run>/latest.txt` — a run-RELATIVE path (root CLAUDE.md); resolves both forms it can hold (`checkpoints/checkpoint_<N>_steps.zip` and the bare `final_model.zip`) |
| 2 | `highest_checkpoint` | the highest-step `checkpoints/checkpoint_<N>_steps.zip`, **including** the SIGUSR1 `checkpoint_forced_<N>_<HHMMSS>.zip`; legacy run-root copies too |
| 3 | `final_model` | `final_model.zip` / `final_model_interrupted.zip` (the higher of the two) |
| 4 | `best_model_fallback` | `best_model/best_model.zip`, then the legacy `<run>/best_model.zip` — **LAST**, only for a run that has nothing else, and it says so on **stderr** when it fires |

Two more rungs are not ladder steps at all — they are the ways a caller names a file outright, and
both **bypass the ladder entirely**: `explicit_step` (`<run>@<step>` → that checkpoint) and
`explicit_zip` (a path ending `.zip`, **`best_model/best_model.zip` included**, used verbatim).
**Naming the file is how you pin it.** Each rung also reports a coarse `rule` — `explicit_step` /
`explicit_zip` / `last_snapshot` (rungs 1-3) / `best_model_fallback`.

### 🚨 DISAGREEMENT: the higher `num_timesteps` wins, not the earlier rung

Rungs 1-3 are three names for "the end of this run", and they disagree in **both** directions:

* a **COMPLETED** run writes `latest.txt → final_model.zip` *after* its last periodic checkpoint, so
  `latest.txt` is AHEAD of `checkpoints/`. Measured on the eight R5F runs (2026-09-06):
  `final_model.zip` @**28,115,184** vs the highest checkpoint @**28,067,760** — **47,424 steps
  apart**, and rung 1 fires for every one of them;
* an **INTERRUPTED** / crashed run can leave `latest.txt` naming a file a later
  `final_model_interrupted.zip` has since passed.

Taking the earlier rung is right in the first case and wrong in the second, so neither ordering is
the rule. The rule is **the file that trained furthest**, with the rung order used only to break a
tie — or to decide when NO candidate declares a step at all (an unreadable zip). `num_timesteps` is
read from the SB3 zip's plain-JSON `data` member (`lineage.checkpoint_num_timesteps` — no torch, no
model load), falling back to the `checkpoint_<N>_steps.zip` filename. `best_model` is not on that
tier at all: it is a different SELECTION rule, so it never competes on steps and loses to every
other rung even when it trained further.

**Every consumer goes through ONE choke point** —
`agents.training.fixed_opponent_pool.resolve_model_ref(path, step=None)` → a `ResolvedModel`
carrying the rung, the rule and `num_timesteps`. It serves `--distill-teacher`,
`--win-prob-pbrs-source`, `--stable-opponents`, `--exploiter`, `--exploiter-ladder`,
`--warmstart-consensus` and `--distill-anchor-parent`; `run_spec_test.py` holds the census that
fails, naming the file and its flags, when one of them stops. 🚨 **EVERY TEACHER LOADED BEFORE
2026-09-06 WENT THROUGH THE OLD RULE and recorded nothing about it** — `main.lineage` says so
rather than re-resolving under today's rule, because a current answer presented as history is worse
than no answer. **NOT VERSIONED**: this changes which FILE a run loads, never a weight shape.
**Full detail — in [`designs/training/matchup_and_lineage.md`](../../../designs/training/matchup_and_lineage.md).**

## Stable (cross-run) opponents (`--stable-opponents`, `fixed_opponent_pool.py`)

Load a frozen model from **another, already-finished run** as a **fixed opponent** — measured
against in eval AND (under `--self-play`) played against in training. Which FILE a run dir resolves
to is the ONE rule above. Design: `designs/ai_v5/design_stable_opponents.md`.
**Full detail — in [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md).**

## Exploiter mode (`--exploiter`, `MaskableAgentWrapper._exploiter_player`)

A clean opponent-mix front-end for the league **exploiter** role: train a dedicated agent against
ONE fixed foreign model as the **sole opponent every episode**, to surface (and then patch, by
folding the exploiter back as a stable opponent / pool member) the non-robustness a *self-play* Nash
cannot see. It needs **no `--self-play` / `--stable-opponents` / share fiddling**.
**Full detail — in [`designs/training/exploiter_and_distillation.md`](../../../designs/training/exploiter_and_distillation.md).**

## Team curriculum — team-side PFSP (`--team-pfsp`) and per-team win-rate tracking (`--team-wr-tracking`)

Two independent instruments on the TEAM axis. **`--team-pfsp {off,measure,var,onesided}`** (default
`off`, byte-identical) biases the TRAINEE's team sampling toward the pool teams it is weakest on,
with a floor, a cap and `--team-block-episodes` for per-team gradient density.
**`--team-wr-tracking`** (DEFAULT ON) is instrumentation only — a running per-`team_sha` record of
wins/games stratified by opponent class, riding `metadata.json`'s `team_win_rates` block.

⚠️ **A raw per-team win rate conflates PILOT COMPETENCE with TEAM STRENGTH** (the ai_v8 team-PFSP
finding). Anything spending budget on this signal must normalize against a team-strength baseline
first; the artifact carries that sentence in its own `notes` field. 🚨 **NO TensorBoard emission**
(owner rule: per-team series are noisy spam), pinned by a test that fails on "just one scalar".
🚨 **Same pool SIZE is not the same pool ORDER** — both aggregators verify per-index team identity
across workers and the tracker RAISES on disagreement. Both are training-only, not version-locked,
and both take an `env_method` PULL rather than an info-dict thread, because that is the seam that
works identically under `--async-rollout`.
**Full detail — in [`designs/training/team_curriculum.md`](../../../designs/training/team_curriculum.md).**

## ELO / skill rating (`elo.py`, `bot_elo_calibration.py`, `main.elo`)

Once training is mostly self-play pool play, win rate stops being legible — `win_rate_vs_pool` is a
treadmill pinned near 0.50 **by construction** and `win_rate_vs_bots` saturates. The ELO subsystem
gives a single **absolute** number, anchored to the fixed bots.

🚨 **THE EVAL OPPONENT REGIME IS A RECORDED, INHERITED PROPERTY OF A RUN** (2026-09-07,
`gen3_eval_sentinel_greedy_default_v1`). Pool sentinels are **GREEDY by default** and draw the
**trainee's own teams**; `--no-eval-sentinel-greedy` restores the old greedy-trainee-vs-stochastic-
sentinel regime, whose asymmetry read **+8.9 pp [+7.0, +10.7]** in the trainee's favour on the same
frozen pair the dense ladder plays symmetrically. `--promote-threshold` follows the regime (0.55
greedy / 0.65 stochastic) and an explicit value still wins. Both are `ModelVersion` fields (config
**v112**) with argparse default `None`, so **a flagless resume or launcher restart INHERITS the
checkpoint's regime** rather than silently crossing an opponent-regime boundary (rule of evidence
15); every launch prints `⚖️  [EVAL REGIME] …` naming both resolved values and their source. Under
the symmetric regime the dense ladder **REUSES** the pairs a cycle already measured (≈500 battles
saved per promotion) — gated on the row's own `sentinel_regime` stamp, both halves required.

**Full detail — in [`designs/training/eval_and_rating.md`](../../../designs/training/eval_and_rating.md)
and [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md).**

## Rollout collection: sync barrier vs `--async-rollout` (`async_vec_env.py`)

The default `SubprocVecEnv.step()` is a **per-step barrier**. `--async-rollout` swaps in
**`AsyncSubprocVecEnv`** (per-env `send_step`/`poll_ready`/`recv_step` + **drain-safe
`env_method`**, which stashes in-flight step results before any barrier RPC) and
`collect_rollouts_async`. It keeps every worker in flight, batch-forwards whichever envs are READY,
and writes each env's transition into **its own buffer column**. It is **exactly on-policy** — a
scheduling change, not an APPO-style algorithm change — and the per-decision mask rides in the Dict
obs, so no wrapper changes. Off by default; ignored under `--debug`. Measured **+14% FPS at the
production `--n-envs 64`** (1489→1695, heuristic opponents); design + benchmark table:
`designs/ai_v5/design_async_rollout.md`.

🚨 **This is why several callbacks are `env_method` PULLS rather than info-dict threads** (reward
terms, team PFSP, per-team win rates): the async collector wave-batches, so callback locals cannot
recover which buffer ROW a step landed on. A capture that needs the row is INLINED into
`collect_rollouts_async` instead (`WinProbLabelCallback`'s terminal capture).

## The two compile flags (`--compile-opponents` · `--compile-trainer`, both DEFAULT ON)

**Split by WHO and WHERE** (renamed 2026-08-14 from the single `--compile-extractor`, which said
neither): **`--compile-opponents`** is the CPU/ROLLOUT half — the frozen opponents in the env
workers, plus BLAS thread pinning. **`--compile-trainer`** is the GPU/LEARNER half (auto-on for
cuda) — the CUDA forward **and backward** the PPO step runs, and the larger of the two. They are
orthogonal; a run can take either, both or neither.
**Full detail — in [`designs/training/compile_flags.md`](../../../designs/training/compile_flags.md).**

## Gradient-balance + value-scale diagnostics (`grad_balance.py`)

The dual-head extractor shares ONE transformer trunk between policy, value and a dozen auxiliaries,
and all of their gradients compete there. A read-only `autograd.grad(retain_graph=True)` probe on
ONE minibatch per `train()` measures each head's pull **on one common denominator**, so every
`grad/*_share` sums to ~1 and any term crowding out the rest is read directly. "Shared" is the
DECLARED `SHARED_TRUNK_PHASES` allow-list — it excludes `cls_pool` and both projection heads, so
only truly contested params count.

**`grad/value_policy_logratio` is the gauge to read** (`log10(‖g_value‖/‖g_policy‖)`, 0 = balanced)
— it is AUX-INDEPENDENT, where `grad/value_share` moves with how many auxiliaries are on. It is
also what `--vf-coef`'s startup announcement quotes rather than recomputing.
⚠️ **`edge/<fam>_*` and `cell/<name>_*` liveness are NOT effect sizes** — every family and cell
enters ZERO-INIT, so a dead one is bit-identical in the logs to a working one; read `weight_norm`
and `grad_norm` as a PAIR, and only an ablation measures importance.
**Full detail — in [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).**

## Live capacity telemetry (`--capacity-telemetry`, `capacity_telemetry.py`)

**Three continuous saturation early-warnings that ride the train loop** — the plasticity canary, the
half-batch trunk-gradient cosine, and feature velocity. They exist because every previous answer to
*"is the network out of capacity?"* was an expensive one-shot probe returning a NUMBER at a MOMENT,
and saturation is a trend.
**Full detail — in [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).**

## THE VALUE LOSS has a MODE — `--critic {shaped,winprob}` (`gen3_winprob_critic_mode_v1`)

**Default `shaped`; a flagless run is byte-identical.** Design of record:
`designs/ai_v12/design_winprob_only_critic.md`; the model-side half is `src/agents/model/CLAUDE.md`
→ *The CRITIC MODE*.

| | `shaped` | `winprob` |
|---|---|---|
| the value TERM | `vf_coef · _value_loss_from_se(...)`, or the HL-Gauss CE under `value_from_dist` | `vf_coef · _win_prob_loss(...)` — the head's **BCE against the terminal outcome** |
| the scalar `value_loss` | the loss | a DIAGNOSTIC only (its term is dropped), computed UNCLIPPED |
| `--win-prob-coef` | weights the auxiliary BCE, tagged `aux` | refused — the BCE is the value loss now |
| PopArt · `--value-dist-*` · every `--win-prob-pbrs-*` | available | **REFUSED** |

🚨 **`winprob` REQUIRES all four of `--no-hand-shaping --terminal-indicator --victory-value 1.0
--draw-penalty 0`**, each named by its own `combination_checks` refusal, so the undiscounted return
is exactly `1{win}` and at `--gamma 1.0` `V(s) = P(win|s)` with no approximation term.
🚨 **THE COST IS STATED, NOT BURIED: a critic bounded in [0,1] cannot represent "a timeout is worse
than a loss."** The anti-stall pressure is the obs deadline clock plus `--arm-no-progress-tax`, and
**stall rate and mean episode length are PRIMARY, kill-condition-bearing endpoints on a `winprob`
arm.** 🚨 **A `winprob` run's `train/*` value tags are not comparable with a `shaped` run's.**

🚨 **`critic_resolution` IS the meter; `critic_reliability` is not.** A base-rate forecaster scores
a perfect 0 reliability and a useless 0 resolution — the committed baseline measured this head at
reliability ~0.002 against a resolution of 0.062 out of an available 0.182, so a promotion that
improves ECE and leaves `critic_resolution` flat has moved the meter that was never the disease.
The `win_prob/critic_*` family comes from `scaffolding.reliability_table`, **imported, never
re-implemented**, and an unmeasurable rollout publishes **`{}`, never zeros**.

🚨 **A DRAW IS SCORED AS A NOT-WIN BY DECISION** (`y = 0`), never dropped and never 0.5 — that is
what makes "P(win)" literally P(win). ⚠️ `signal/draw_rate` counts **ties and not timeouts**, so the
series that watches the 250-turn cap is `signal/stall_rate` / mean episode length.
🚨 **THE 250-TURN CAP IS A TERMINAL, NOT A TRUNCATION** — this env never truncates in the SB3 sense;
a cap forfeit used to arrive as `truncated`, so SB3 bootstrapped `V(s_last)` onto a 0 reward at
γ=1 and the timeout left the loss entirely with **a TD error of identically zero**. Fixed in
`wrappers.resolve_episode_end`, under `winprob` only. **`--gamma` is a flag now and is INERT ON A
RESUME like `--lr`.**

### `--win-prob-strata-weight` — the BCE's opponent MIX (`gen3_winprob_strata_weight_v1`, v115)

**Default `0.0` = OFF and the loss is BIT-identical; `--critic winprob` is REQUIRED** (refused
otherwise — under `shaped` that BCE is an auxiliary diagnostic, not the value loss). Each state's
BCE term is multiplied by its opponent CLASS's weight `w_c ∝ freq_c ** (−s)`, capped at **8×** and
renormalised so the **mean weight over the rollout buffer is exactly 1** — it re-prices the MIX
without moving the loss SCALE, so an arm cannot confound "re-weighted the classes" with "raised the
critic's learning rate". At `s = 1` every class contributes equally; `s` interpolates.

🚨 **WHY, and it is arithmetic rather than a hunch.** Only **10.2 % / 14.4 %** of the terminal 0/1
label's variance lies BETWEEN (cycle, opponent) cells, so a head minimising BCE buys its resolution
from the board and its own team — which is cheaper — and never conditions on the opponent. The head
refit proved the fault is the TARGET, not the head, on both substrates
([`winprob_head_refit_2026-09-09`](../../../designs/research_state/measurements/winprob_head_refit_2026-09-09/README.md)
§6/§11). This raises that share directly, with **no new labels and no rollout cost**.

🚨 **THE VOCABULARY IS THE FOUR `opp_class` CODES** — `bot` / `pool` / `stable` / `exploiter` — and
per-BOT identity is **not available**: the archetype is drawn per EPISODE in
`MaskableAgentWrapper._select_episode_opponent` and never reaches the observation. So the lever
balances the between-CLASS share and leaves within-class heterogeneity in episode proportion.
⚠️ **THE CAP BINDS AT THE PRODUCTION MIX, deliberately**: at ~10 % bots / ~90 % self-play, `s = 1`
asks for 10× and gets 8×, so the objective splits **44/56, not 50/50** — a 4.4× re-pricing with a
bounded per-row weight. Read `win_prob/strata_share_*`, `strata_w_entropy` (1.0 = balanced) and
`loss` vs `loss_unweighted`. 🚨 **`strata_active` 0 vs an ABSENT family are different facts**: the
family is published whenever the flag is on, so 0 means "on, but one class present / no labels yet"
(every `--debug` run and any run before the pool seeds) and ABSENT means the flag is off.

### `--win-prob-lambda` — the BCE's TARGET (`gen3_winprob_lambda_v1`, v116)

**Default `1.0` = OFF and BIT-identical; `--critic winprob` is REQUIRED** (refused otherwise — under
`shaped` the buffer's `values` are a PopArt-normalised shaped return, not a probability, so blending
them into a BCE target is a category error). Below 1.0 each state's target stops being its episode's
terminal bit and becomes a **λ-return over the collector's RECORDED values**:

```
row t ENDS its episode   ⇒  G[t] = y                       (the outcome, exactly)
otherwise                ⇒  G[t] = (1−λ)·V(s[t+1]) + λ·G[t+1]
```

γ = 1 and the clean-world stream is terminal-only, so an n-step return **is** `V(s[t+n])` and the
λ-average collapses to that one backward pass. A state `d` steps from its terminal keeps weight
**λ^d** on the outcome. The loss is unchanged — the same masked-mean BCE, now against a SOFT target,
which is exactly what a proper scoring rule generalises to.

🚨 **WHY.** The strata flag's mechanism from the other side. One terminal bit copied to ~30 states is
a very noisy objective, only **10.2 % / 14.4 %** of whose variance lies BETWEEN (cycle, opponent)
cells, so the weak axes shrink toward the marginal — the critic barely separates opponents at turn 1
(spread ratio **~0.1**) although **mid- and late-game values already separate them at ~0.5–0.8**. The
information exists inside the episode; λ moves it backward along a far less noisy channel than the
terminal draw ([`winprob_head_refit_2026-09-09`](../../../designs/research_state/measurements/winprob_head_refit_2026-09-09/README.md) §6/§11).

🚨 **`V` IS THE RECORDED, PRE-UPDATE VALUE** (`rollout_buffer.values`, which under this critic *is*
`sigmoid(win logit)`), not a re-forward inside `train()`: a target recomputed from the current
weights would move under its own gradient across the 10 epochs. **`--win-prob-lambda-truncated
{bootstrap,mask}`** picks the buffer-boundary convention for an episode with no terminal inside the
rollout — `bootstrap` (the default) targets `V(s_T)` from the same `model._last_obs` forward SB3's
GAE bootstrap uses and **UNMASKS** rows that carry no target today, `mask` leaves them excluded. It
is a flag so a read can separate "the targets moved" from "there are more rows"
(`win_prob/lambda_unmasked` counts them), and it is INERT at λ = 1.0 because the recursion is
skipped whole. An episode that ended with **no recorded outcome** is never unmasked — it would train
the head against a fabricated label.

Read `win_prob/lambda_target_shift`, `lambda_bootstrap_frac` and **`lambda_loss` vs
`lambda_loss_terminal`** (both scored on the SAME recorded predictions, so their difference is the
target change and not a step of learning). 🚨 **An ABSENT `win_prob/lambda_*` family means λ = 1.0**,
and nothing else. ⚠️ The value SIDECAR's `target` column follows the flag: under λ < 1 it holds the
λ-return, not the raw outcome. Composes with `--win-prob-strata-weight` (that one weights ROWS, this
one re-aims them) and with the cf labels (disjoint state sets — the cf term never touches
`win_target`).

### `--win-prob-dense-aux` — 25 DENSE TARGETS BESIDE the BCE (`gen3_dense_aux_v1`, v117)

**Default `0.0` = OFF and BIT-identical** — bit-identical by not BUILDING the head, so there is no
module, no obs key, no callback and no term. **`--critic winprob` is REQUIRED**, and
`--win-prob-mode != none` is a `flag_registry` `requires` the extractor constructor enforces.

The three flags above all act on ONE loss. This one does not touch it: it adds a small MLP on
`value_pooled` — the same tensor the win head reads — predicting for every state the episode's
END-OF-BATTLE facts, back-filled the way the win bit is.

| outputs | target | scored by |
|---|---|---|
| `0..11` | SURVIVAL of slot k (our 6, then theirs 6, in the OBSERVATION's own team order) | BCE |
| `12..23` | slot k's FINAL HP FRACTION at termination | BCE against the soft target |
| `24` | TURNS LEFT, `log1p(terminal_turn − this_turn) / log1p(250)` | BCE against the soft target |

`aux_loss = coef × mean(the three masked-mean terms)` — a mean of TERMS, so twelve survival outputs
cannot outvote the one turns output. The per-side KO counts are **DERIVED** from survival
(`6 − Σ survived`) and published as meters, never predicted: a count is a sum over slots some of
which are masked, so it is the one target that could not honour the mask.

🚨 **WHY.** Four 10M levers on the same one bit moved nothing at ±0.01 on bot resolution (ledger
*THE ARMS AT 400 GAMES*), and only ~10 % of that bit's variance lies BETWEEN opponents. KataGo (Wu
2019 §3) answers a one-bit terminal signal by ADDING targets that share its cause — ownership of
every point, and the final score — for a large reported gain in learning efficiency. Per-Pokémon
end-of-battle outcomes are our analogue, and each is a fact about a NAMED ENTITY the state's own
observation carries, so the gradient runs along exactly the axes a pooled bit cannot separate.

🚨 **TWO MASKS, ANDed with the episode-known bit.** A slot is scored only where it HAS an
end-of-battle fact (an opponent mon never revealed has none — MASKED, never fabricated as "alive at
full HP", a label that would be wrong in a DIRECTION) **and** where it names an entity THIS state's
observation carries (opponent slot order is REVEAL order, so an early state simply has fewer;
scoring an unrevealed slot would anchor a label to a feature block encoding nothing — this arm's own
defect, one level down). Read `win_prob/aux_masked_frac` before reading any aux loss.

🚨 **THE HEAD'S INPUT IS NOT DETACHED, and that IS the arm.** It is not called by the forward at all
(the `CfEvidentialHead` contract), so pi/vf are bit-identical at an ARBITRARY weight in it; the
training term applies it to the stashed, LIVE `value_pooled`, so its gradient reaches the shared
trunk exactly as the win-prob loss does under `shaping`. **`grad/dense_aux_share` must NOT read 0**
— unlike `grad/cf_evidential_share`, whose 0 is the verification.

🚨 **λ DOES NOT REACH THESE TARGETS.** They are terminal FACTS, not returns, and there is no recorded
per-state estimate of "slot 4's final HP" to blend. Structural, not conventional: the λ recursion
overwrites `win_target`/`win_mask` and names no `aux_*` key. Under λ < 1 the two coexist.

Read **`aux_auc_own` vs `aux_auc_opp`** (survival AUC per SIDE — a pooled one would hide the
asymmetry the arm is built to move; a side with one class absent is OMITTED, never logged),
`aux_hp_mae` / `aux_turns_mae` (⚠️ the interpretable reads — a BCE against a SOFT target has a
non-zero entropy floor, so the loss numbers alone cannot say whether the head is good),
`aux_coverage` and `aux_ko_mae_*`. ⚠️ **STRUCTURAL**: `dense_aux` is gated by a bool compare in
`check_compatible`, so a resume may RE-DOSE the coefficient but may not add or drop the head.

**Full detail — the currency argument, the cap-terminal measurement, the `--vf-coef` BCE
announcement and every value-side flag below — is in
[`designs/training/critic_and_value_losses.md`](../../../designs/training/critic_and_value_losses.md).**

### The other value-side flags, in one place

| flag | default | what it does, and the one thing to know |
|---|---|---|
| `--vf-coef` | `0.5` | multiplies a BCE under `winprob`, an MSE on a PopArt-normalised shaped return under `shaped` — **the 0.5 default carries no information about the first**. The startup announcement prints the raw BCE and the value/policy shared-trunk gradient RATIO (`10 ** grad/value_policy_logratio`); it never divides by `|policy loss|`, which is ≈0 by construction on epoch 1. Fixed for a run's lifetime |
| `--use-popart` | off | normalizes the value target so the value gradient stops swamping the trunk. **Requires an explicit `--clip-range-vf none`**; watch `grad/value_policy_logratio` fall toward 0. Refused under `winprob` |
| `--value-tail-weight` | `0.0` | CVaR blend over the worst 10% squared errors at all three value sites. β=0 is byte-identical to `F.mse_loss`. **Resume-IMMUTABLE** (`check_value_tail_weight`) |
| `--td-aux-coef` | `0.0` | the Bellman identity as an explicit loss over CONTIGUOUS pairs the PPO permutation destroys. 🚨 **Pre-registered band 1.0–3.0; `λ ≤ 0.1` measured significantly WORSE than control** — the small-coef regime is to be avoided, not treated as "a bit of the effect". Episode boundaries DROP the pair, never zero it |
| `--value-dist-mode` / `--value-dist-coef` | `none` / `0.0` | HL-Gauss categorical readout off `value_pooled`, **interpretability only** — ledger K1 killed it as a win-rate lever. Validate PIT ≈ uniform, never win rate. REFUSED under `winprob` |

## The PRIVILEGED true-team value channel (`--value-true-team`)

`gen3_value_true_team_v1` (v114), the critic ladder's **arm-5 CEILING PROBE**
(`designs/research_state/winprob_critic_ladder_2026-09-08.md` §L1): how much of the win-prob
critic's residual error is irreducible uncertainty about the opponent's team? It is the only
channel in the tree that gives the network information the observation does not already carry.

The plumbing is the belief labels' exactly. `Gen3Env` declares ONE more training-only obs key,
`opp_true_team` `[6, POKEMON_FULL_DIM]`, and fills it from **`battle2.team`** — agent2's own battle
view, the same privileged source `belief_species` / `belief_spread` / `item_label` / `hp_type_label`
already read — through `agents.observation.true_team.build_true_team_block`, which calls the SAME
`PokemonEncoder.encode` the flat vector's opp slice uses. Unlike those, this key is **not a label**:
it enters the FORWARD, on the value side only, via `TrueTeamValueReadout`'s zero-init injection into
`value_pooled` (`pi_combined` never contains `value_pooled`, so `pi` is bit-identical at any
weight). It must therefore also be present at EVAL, or the arm's own meters would read a V the run
never trained — `RLPlayer` emits it from the `_opp_player` back-reference `LocalBattleRunner` sets,
which is the transport for bridge training, bridge eval and the counterfactual replay driver. At
ladder play there is no runner and the all-zero "unknown" block goes instead.

Model half + the four contracts (vf-only, augment-not-replace, presence-follows-the-local-sim,
raise-not-skip): [`designs/model/readouts_and_value_routes.md`](../../../designs/model/readouts_and_value_routes.md).

## The win-probability head (`--win-prob-mode` / `--win-prob-coef`) and its two PBRS routes

A calibrated **P(win|state)** supervised by the Monte-Carlo episode OUTCOME, back-filled onto every
step of an episode by `WinProbLabelCallback`; the trailing in-progress episode gets `win_mask=0` and
is **never trained toward a fabricated label**. `win_target`/`win_mask` are TRAINING-ONLY obs keys
read only by the loss, so the outcome cannot leak into the forward.

🚨 **`--win-prob-mode shaping` carries NO behavioral force, and the word has misled readers.** It is
**REPRESENTATION** shaping — the BCE gradient reaches the shared trunk; there is no gradient path
anywhere from *predicting wins* to *choosing winning actions*, because the logit is a SIDE readout
never concatenated into pi/vf. **The head is a BAROMETER, not a coach.** It is also
self-referential: its labels are outcomes under the CURRENT policy, so a habitual whiff that still
wins 55% teaches it "55%", never "the whiff was the mistake".

**The routes that ARE pointed at behaviour**, all OFF by default and byte-identical when off:
`--win-prob-pbrs-coef` (route 1 — `γφ(s′) − φ(s)` folded into the REWARD, `shaped`-critic only),
`--win-prob-pbrs-source <ckpt>` (a FROZEN foreign φ, so Ng's invariance holds **exactly** rather
than approximately), and `--win-prob-pbrs-frozen` (the ACTOR-ONLY variant for `--critic winprob`:
it writes **only** `rollout_buffer.advantages`, so `V ≡ P(win|s)` is preserved bit-for-bit).
⚠️ **Read `pbrs/frozen_phi_mean` FIRST and it must be FLAT** — φ is a fixed function of state, so a
mean that wanders means the frozen source is not the thing being read.
⚠️ **`train/pbrs_reward_share` is the WRONG meter for sizing a coefficient** (its denominator moves
with EPISODE LENGTH — measured 2.1–3.1× off — and it reads `NaN`, never `0.0`, on an empty unshaped
stream); **`train/pbrs_episode_dose` is the meter the ladder is sized in**, the shaping's whole
per-episode budget priced against one win.
**Full detail — in [`designs/training/winprob_head_and_pbrs.md`](../../../designs/training/winprob_head_and_pbrs.md).**

## The training-side VALUE SIDECAR (`--value-sidecar`, `gen3_value_sidecar_v1`)

**Detail: [`designs/training/value_sidecar.md`](../../../designs/training/value_sidecar.md).**

Every other instrument that reads the critic reads **eval** battles. This one reads the **training
buffer** — the value PPO actually used, against `win_target`, the label the BCE actually minimises.
Once per rollout at `_on_rollout_end`, a seeded 1/64 of buffer states is appended to
`<run>/value_sidecar/rows.jsonl`. Read it with `python -m main.ops.value_sidecar_read <run>`.

- **`--value-sidecar {auto,on,off}` (default `auto` = ON under `--critic winprob`)**, plus
  `--value-sidecar-fraction` (1/64) and `--value-sidecar-seed` (0). None of the three reaches
  `model_config.json`, so there is no `MODEL_CONFIG_VERSION` implication.
- 🚨 **CALLBACK ORDER IS LOAD-BEARING AND SILENT IF WRONG.** It MUST be appended after
  `WinProbLabelCallback` — that callback's `_on_rollout_end` is what replaces the `win_target` /
  `win_mask` placeholders with the Monte-Carlo label. Registered earlier it reads ZEROS and writes a
  file that looks exactly like a critic scoring an unbroken run of losses. `main.train.callbacks`
  appends them in that order, `value_sidecar_test.py` pins it, and at runtime an all-zero mask over
  a whole rollout is REPORTED (`labels_unfilled`) rather than written as data.
- 🚨 **It cannot be reconstructed after the fact.** It reads the rollout buffer, which is gone the
  moment `train()` returns. A run launched without it has no training-side read, ever.
- 🚨 **`target` HAS ONE NAME AND TWO MEANINGS, and only the header says which** — the terminal 0/1
  OUTCOME at `--win-prob-lambda 1.0` (every arm before arm 8), a soft **λ-RETURN** below it
  (`SIDECAR_SCHEMA` 2). Three arms landed on exactly 155,137 rows each; **matching row counts are
  not evidence of a matching quantity.** The outcome is written as its OWN `outcome` /
  `outcome_known` column — the λ recursion overwrites `win_target` in place, and the bit is
  otherwise gone (the λ-return holds it at weight λ^d for an unrecorded `d`, and **`win_margin` is
  a per-turn MATERIAL margin, not an outcome**). `ep_complete` follows the terminal mask, never
  `target_known`, which `bootstrap` truncation widens.
- 🚨 **ONE HEADER PER WRITER SESSION, not per file** — a resume used to append its rows under the
  first process's header, hiding a mid-file change of meaning. `value_sidecar_read` reads the
  header FIRST, labels every table with the quantity it scored, and REFUSES a mid-file change by
  ROW INDEX and a `--compare` across a quantity boundary. A schema-1 file and a schema-2 file at
  λ = 1.0 compare EQUAL on purpose.
- 🚨 **`critic_read` (eval) and `value_sidecar_read` (training) answer DIFFERENT questions.**
  Neither supersedes the other; a disagreement is a finding about GENERALISATION.
- ⚠️ **`opp_class` now rides the win-prob gate too**, not just the intent labels — a win-prob arm
  normally has no intent loss, so the by-opponent-class slice was otherwise empty on exactly the
  runs the sidecar exists for. It stays a label key the network never reads
  (`designs/ARCHITECTURE.md` §7). There is no opponent NAME, snapshot STEP or ladder RATING: the
  first two never reach the observation and the third does not exist at training time.
- **Cost, measured 2026-09-08 at production shape: 19.3 ms median per rollout, 0.57 MB — 0.016% of
  a hostile 120 s rollout.** 🚨 A `--debug` smoke A/B CANNOT measure this (the ON arm came out 6.5 s
  *faster*); use `src/agents/training/value_sidecar_benchmark.py`, which measures the numerator
  directly and warns on contention rather than rescaling.

## Step size, batch size and THE DOSE (`--grad-accum-steps` · `--fork-lr` · `--adaptive-batch`)

**`--grad-accum-steps K`** runs K `batch_size` micro-batches per optimizer step, so the accumulated
gradient is the **exact** gradient of a `batch_size·K` batch at one micro-batch's activation peak;
`K=1` is byte-identical to upstream. It is also what makes the McCandlish **gradient noise scale**
free (`train/noise_scale_ratio` ≫1 ⇒ noise-limited, ≪1 ⇒ over-batched).

🚨 **`train/noise_scale` is measured on the TOTAL gradient, and on this tree the total gradient is
mostly not PPO.** A dozen dense supervised auxiliaries have agreeing per-example gradients, which
DEFLATES `B_simple` — the live runs read the total at 0.001–0.05 ("over-batched 16–1000×") on runs
whose **`train/noise_scale_ratio_policy`** read noise-limited. **Read the policy term; never size a
batch on the total.** `--adaptive-batch policy` closes that loop, moving K only — never
`--batch-size`, which would be the unbounded shape set `--compile-trainer` refuses.

🚨 **`--lr`, `--batch-size` and `--n-steps` are INERT on a resume** — SB3 restores the checkpoint's
own values, so a FORK inherits whatever the parent's KL controller had annealed to. **`--fork-lr`
pins it** and fires ONLY on a genuine fork (a checkpoint outside the run dir); `--fork-lr-freeze`
makes it constant and persists across every restart. 🚨 **The quantity that predicts a fold's
collateral is the DOSE**, `lr × n_epochs / (batch_size × grad_accum_steps)` — three folds launched
at the same `--lr` ran at 1.00× / 6.62× / 3.19× v8's rate and nothing in any of them said so. Read
it with `python -m main.dose <run>` or the live `train/dose_rate`. ⚠️ **The dose is a product of TWO
controllers** (KL-lr and adaptive-batch), so watch `train/dose_rate`, never either loop's own series.
**Full detail — in [`designs/training/step_size_and_batch.md`](../../../designs/training/step_size_and_batch.md).**

## LINEAGE — who forked whom (`lineage.py`, `python -m main.lineage`)

`metadata.json`'s **`lineage`** block states the fork graph instead of implying it: `role`
(`fresh`/`fork`/`fold`/`exploiter`), `fork_parent`, `teachers`, `exploiter_target`, a walked
`ancestry` and an `ancestry_stop` saying where the chain went dark and why. 🚨 **IMMUTABILITY is the
whole feature** — the existing value always wins, because a launcher restart re-derives the block
and would silently re-point the recorded parent at the DRIFTED student. 🚨 **RECORDED and DERIVED
are INDEPENDENT**: a block REGEXed out of `original_command` is a recorded GUESS, and the CLI's
header says both. **The FRESH form is explicit** (`fork_parent: null`) — "no block" and "no parent"
are different facts. A fork also inherits its parent's TB curves as a TRUNCATED prefix
(`tb_inherit.py`, `--no-tb-inherit` opts out); truncation is not optional, or parent-only progress
draws inside the fork's own step range.
**Full detail — in [`designs/training/matchup_and_lineage.md`](../../../designs/training/matchup_and_lineage.md).**

## The `signal/` group — advantage density × outcome entropy (`gen3_signal_rate_metrics_v1`)

**How much action-attributable learning signal is PPO actually receiving?** Two always-on, flagless
scalar families answer it live. Pure observability: no gradient path, no extra battle, no env call —
a handful of numpy means per rollout.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).**

## The SCAFFOLDING GAUGE — `train/scaffolding_gauge` + `python -m main.scaffolding_gauge`

🚨 **THIS GAUGE IS A `--critic shaped` INSTRUMENT AND IS DEGENERATE ON THE PRODUCTION RUN.** It
measures the divergence between TWO readouts; under `--critic winprob` there is one — the win-prob
head IS the critic, so the gauge compares a head with itself and its rank correlation is 1 by
construction. Read it on a shaped run, or on an archived one; do not read it as a scaffolding
measurement of a terminal-only run, which has no scaffolding to measure.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).**

## ⚠️ Reading a belief target: `belief_supervision(...)`, never `last_*`

Cross-cutting rule for **every** belief loss below (`gen3_belief_label_only_v1`). Under
`--belief-grad-mode label_only` the extractor's `last_move_belief_logits` / `last_spread_belief` /
`last_hp_type_logits` / `last_spread_nature_logits` / `last_spread_ev` / `last_alpha_logits` stashes
are **stop-grad publications** — that is how the mode stops the policy/value gradient reaching a
belief head through any of its forward consumers. A supervised loss must therefore read its target
through **`self.policy.features_extractor.belief_supervision("<key>")`**, which returns the LIVE
tensor (and the identical object under `shaping`/`detached`).

A loss that reads the `last_*` attribute instead trains **nothing** under `label_only`, and does so
**silently** — the loss value, its gradient norm and every `belief/*` metric look completely normal,
because the loss is still computed; only the graph behind it is gone. The accessor raises a
`KeyError` on an unknown key so a typo cannot degrade into that, and
`agents/model/belief_label_only_gate_test.py::test_every_belief_loss_still_trains_its_head` is the
guard that each key still deposits gradient on its own head. The full four-route table is in
`src/agents/model/CLAUDE.md` → `--belief-grad-mode`.

## The supervised belief losses

Six supervised belief heads, all folded through **`belief_bank.py`** — one declarative ROW per head
(stash/attr/obs spec · coef key · metric prefix) and `compute(site=…)` at the THREE original
`train()` positions, where the site tag is what preserves the float-addition sequence exactly. A
seventh belief is a row, not a slice. Every head emits **`mask_rate`** under its own prefix — the
uniform per-head coverage key, comparable across heads and batch sizes where the older `n_slots`
counts are not. ⚠️ **The mask conventions TILE**: hidden-team masks HIDDEN slots, the
spread/nature-EV/hp-type heads mask REVEALED ones.

| flag | default | supervises | label |
|---|---|---|---|
| `--opp-belief-aux-coef` (+ `--opp-belief-moves-weight`) | `0.0` | the opponent's still-hidden mons — species CE + moves BCE, **order-invariant (Hungarian)** over the anonymous believed slots | `belief_species`/`belief_moves`, from agent2's own team |
| `--move-belief-mode` / `--move-belief-coef` | `off` / `0.0` | the reinjected moveset, over two DISJOINT slot populations (revealed = direct BCE, unrevealed = Hungarian) | `known_moves` / `belief_moves` |
| `--spread-belief-coef` (+ `--spread-belief-nature`) | `0.0` | the hidden derived stats the `DamageOperator` consumes; the nature⊕EV decomposition supervises it structurally | true `mon.stats`, and the nature/EVs **deterministically INVERTED** from them |
| `--hp-type-belief-coef` | `0.05` | the discrete Hidden-Power type posterior | `hp_type_label`/`hp_type_mask` |
| `--intent-label-bot-weight` | `1.0` (OFF) | a per-sample weight on the opponent-intent (α/β) LABELS produced against a **bot** | the existing `opp_class` obs key |

🚨 **Every belief label is a TRAINING-ONLY Dict-obs key read only by the loss** — the model forward
reads only `obs["observation"]`, so privileged truth cannot leak — and each builder is **fail-loud**
(a non-contiguous `species_known`, an out-of-vocab num) rather than mis-slotting supervision.
🚨 **`--intent-label-bot-weight` is confined to α/β and that is a design claim**: the other beliefs
are TEAM TRUTH, which does not depend on who is piloting, so discounting a bot's rows there would
throw away valid labels. Only INTENT is behaviour; at 1.0 the loss is **bit-identical**. Every
structural toggle is version-checked and fresh-only; every `*_coef` is training-only and **read back
on a flagless resume**.
**Full detail — in [`designs/training/belief_losses.md`](../../../designs/training/belief_losses.md).**

## `stats.py` — the package's SHARED small-sample statistics

**Where a stateless estimator lives once the second consumer exists** — `wilson_ci`, `spearman`,
`cluster_bootstrap_ci` / `cluster_bootstrap_diff_ci`, `sd_true_excess` and the `MIN_CELL_N` /
`MIN_SUBCELL_N` floors. Pure NumPy in, floats out: no labels, no battles, no checkpoints, no
filesystem, no torch, no RNG except an explicitly seeded bootstrap. That is the admission rule; a
helper that has to know what a *decision* or a *bias map* is belongs beside the instrument that owns
the concept. ⚠️ **Three near-siblings elsewhere are deliberately NOT merged into it** (the
NaN-refusal pair in `scaffolding.py`, `winprob_finetune.label_noise_variance`,
`main/q_amortization.spearman`) — the reasons are in the module docstring and in
[`designs/training/offline_meters.md`](../../../designs/training/offline_meters.md), so nobody
"de-duplicates" a shipped instrument's output by accident.

## `cf_audit` — the counterfactual audit instrument (`cf_audit.py`)

**Three modules, one instrument**: `cf_audit.py` owns the frame, the sampler, the label schema, the
bias map and the CLI; `cf_audit_render.py` turns a finished bias map into markdown (its two
formatting rules are the ABSENT-vs-ZERO ones — a row of zeros makes "no head" and "no uncertainty"
read identically); `cf_audit_twin.py` holds the twin-head paired read and the shadow read.
`cf_audit` re-imports every name, so the historic import paths still resolve.
**Full detail — in [`designs/training/cf_grounding.md`](../../../designs/training/cf_grounding.md).**

## Prefix-sharing materialization (`obs_materializer.materialize_branches`)

K counterfactual arms of one decision share an identical prefix; the materializer replays it once,
snapshots the player's whole battle/tracker state at the branch decision, and restores it per arm —
**exactly equivalent to per-arm `materialize_decisions`, bit-for-bit** (59/59 arms byte-identical,
15.4 → 5.3 ms per arm). The per-arm restore is serialized ONCE and rebuilt per arm rather than
deep-copied (1.98 → 0.22 ms). ⚠️ A graph that will not pickle **falls back to deepcopy and says so
once on stderr** — a 9× regression nothing mentions is the failure shape this tree keeps eating.
**Full detail — in [`designs/training/cf_grounding.md`](../../../designs/training/cf_grounding.md).**

## Counterfactual win-prob grounding (`--cf-records` / `--cf-winprob-coef`, `gen3_cf_label_plumbing_v1`)

The **trainer-side plumbing** for `designs/ai_v10/design_counterfactual_value_grounding.md` — its gate
**G3**, which is explicitly "tap + buffer + flags at coefficient zero, byte-identity gated". Rung **R1**
only: tight Monte-Carlo P(win) labels, delivered to the **win-prob head**. The label PRODUCER is a
separate, out-of-process program (`cf_producer.py`, § *The label PRODUCER DRIVER* below);
**nothing in this section produces a label**, and the two halves share only a file format.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/cf_grounding.md`](../../../designs/training/cf_grounding.md).**

## The STALL-TAIL HARVEST + head-repair pipeline (`main.harvest` → `winprob_finetune` → `main.harvest_meter`)

An **offline, three-stage pipeline** that manufactures win-probability labels for the population
probe O convicted, fits the win-prob head on them with the trunk frozen, and measures the result
against a battle-level holdout. It is the ai_v12 head-repair backbone and it writes nothing into
`models/` — artifacts land under `utils.paths.harvest_dir()` (repo-root `harvest/`, gitignored,
`$GEN3AI_HARVEST_DIR` overrides).

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/stall_tail_harvest.md`](../../../designs/training/stall_tail_harvest.md).**

## `replay_imputation_probe` — the own-side imputation meter (`replay_imputation_probe.py`)

A **meter, not a lever**: how far would our observation move if the only thing we knew about our OWN
side were what a public Showdown replay had shown by now? It plays reproducible bridge battles,
overwrites our not-yet-revealed moves / item / spread with the top Smogon-prior candidate, and
re-encodes — with truth re-encoded a THIRD time after the restore and required to be bit-identical,
because a leaked restore would make every later "truth" a previous decision's imputation. There is
**no transcoder and no `|request|` synthesis here**, deliberately.
**Full detail — in [`designs/training/offline_meters.md`](../../../designs/training/offline_meters.md).**

## Two DELETED subsystems

The **latent-belief loss** (`--opp-belief-latent-coef`, the SimSiam predictor, v75) and the
**public-replay value aux V_pub** (v88 `gen3_dead_flag_purge_v1`) are **gone** — the first was a side
readout never fed forward that cost ~13% of the train step, the second measured NULL and was never
ON in a production generation. A checkpoint recording either is REFUSED by its migration. The
reasoning generalises to every aux head on this trunk and is kept as closed history:
`designs/research_state/claude_md_archive/training_leaf_deleted_subsystems_history.md`.

## Exploiter distillation (`--distill-teacher` / `--distill-coef` / `--distill-value-coef` / `--distill-value-feat-coef`)

`gen3_exploiter_distill_v1` — pour a frozen per-team SPECIALIST into the generalist so it learns to
PILOT that team, closing the amortization gap the self-play average cannot. `--distill-teacher`
takes `TEACHER:TEAM` colon pairs (a bare run dir → that run's **LAST SNAPSHOT**; the `🧪 [DISTILL]`
line states the resolved file, its step and the rung per teacher); the env emits a training-only
`distill_mask` key, and per teacher `distill_coef · KL(π_teacher ‖ π_student)` is folded masked to
that teacher's states, the per-teacher mean-KLs AVERAGED so a small-coverage teacher still
contributes comparable gradient. `--distill-team-bias` (0.4) keeps the rest as pool rehearsal. OFF
is byte-identical; training-only, NOT version-locked, inherited on a flagless resume.
**Full detail — the off-slice anchor, the stop rule, the advantage-gated/action-form target and the
rank tripwire — is in [`designs/training/exploiter_and_distillation.md`](../../../designs/training/exploiter_and_distillation.md).**

## Search-as-teacher (`--search-teacher`, `teacher/` package)

Selective **Expert Iteration**: each cycle, search + rollout-confirm the worst loss craters of
recent eval traces and distil the VERIFIED-better action into the policy via an advantage-weighted
CE aux loss (AWR), or the full improved distribution via `--opd-coef` KL. Off by default and
byte-identical. The "expert" is the prober's `better_line` beam plus the rollout-confirm tiers.

🚨 **The distilled advantage is the CONFIRMED win-rate improvement, never the critic's optimistic
backed-up value** (the Spore 95%-vs-62% lesson), and an opponent that cannot be resolved exactly is
**SKIPPED, never approximated** — distilling "A\* beats a proxy" is a soundness failure, not a
degrade. 🚨 **BOTH PHASES OF A CYCLE RUN IN CHILDREN**; selection used to run INLINE on the training
step (measured 48.1 s over 9 traces, 350.2 s over 60) and "non-blocking" was half true for a year —
`teacher/step_block_ms` now records what the teacher costs the training step as a series.
🚨 **A FAILED SELECTION IS A REPORTED CYCLE, not a silence** (`teacher/selection_failures_total`).
⚠️ **`--search-teacher-mode winprob_oneply`** (ai_v12 routes 2+3, nothing has run it) swaps only the
selection and production halves, and its **CONFIRMATION step is a REQUIREMENT, not a refinement** —
the WINNER'S CURSE: a separation procedure certifies the leaf's residual differential bias as much
as signal, and unlike PBRS a distillation target has **no invariance shield**.
**Full detail — in [`designs/training/search_teacher.md`](../../../designs/training/search_teacher.md).**

## Process liveness guards (`watchdog.py`)

Two daemon-thread watchdogs keep a hung/abandoned run from lingering:

- **`start_subprocess_watchdog`** — for the `SubprocVecEnv` path. A crashed worker leaves the
  parent blocked on a pipe `recv` forever; this thread polls `processes` and `os._exit(1)`s the
  moment a worker dies with a nonzero exitcode. Started *after* env construction (and, in
  self-play, after `_maybe_engage_self_play` rebuilds the env), right before `learn()`. It is a
  **no-op on the `--debug` DummyVecEnv path** (no worker processes to watch).
- **`start_orphan_watchdog`** — for the `--debug` smoke path, which has no worker watchdog. A
  smoke run is a child of the launching shell/agent; if that parent dies the run is orphaned
  (PPID changes) and a hung smoke (e.g. a vanished `9XXX` server) would otherwise sit as a
  multi-GB zombie indefinitely. This thread captures the launching PPID up front and `os._exit`s
  when `os.getppid()` *changes* (by-change, not `== 1`, so PID-namespace subreapers count).
  Started early in `main()` inside the `if args.debug:` block — before team/env/server setup —
  so a startup hang is covered too. **Real launcher-managed runs keep a live parent and never
  arm it.** Regression test: `watchdog_test.py` (subprocess-driven orphan + no-false-fire).

## Showdown port threading (the `server_config` seam)

`train_rl_agent.py --showdown-port <port>` builds **one** `ServerConfiguration` in `main()`
via the single constructor `localhost_server_configuration(port)` (in
`poke_env.ps_client.server_configuration`) and threads it to **every** Showdown client —
the training-env players (carried into the `SubprocVecEnv` spawn workers via the env-factory
closures), eval, and self-play. Every player-creating callback takes a `server_config` param
(defaulting to port 8000 for standalone use) and builds its players from it — **never** from a
bare `LocalhostServerConfiguration` constant. `server_port_threading_test.py` is the
regression guard: it fails if any of these callbacks hardcodes the default port instead of
threading the configured one (the original bug had the now-retired replay recorder connecting
to :8000 while training ran on :8001; eval forensic traces inherit the same guard).
There is no environment variable; `train_rl_agent.py`'s own default is 8000, but the **launcher**
overrides it to 8001 before forwarding (see `src/main/launcher/CLAUDE.md`). The launcher
forwards `--showdown-port` verbatim (it strips only launcher-owned flags).
