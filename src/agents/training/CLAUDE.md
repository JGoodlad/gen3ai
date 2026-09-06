# CLAUDE.md — Training (`src/agents/training/`)

Callbacks, reward manager, episode/turn tracking, stall detection, and the bot-eval pipeline.
**How to launch training** (commands, flags) lives in the root `CLAUDE.md` → Training /
Launcher; this file documents the subsystems' internal design. The `TurnDelta` fold and the
LiveView/TurnView/LegalActions read-models it consumes are documented in
`src/agents/battle/CLAUDE.md`. The obs-build performance gate is in
`src/agents/observation/CLAUDE.md`.

## TensorBoard export census — every scalar, and its CURRENCY

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
`train/return_abs_max` (raw, ~30). The conversion in force is `popart/mu` and `popart/sigma`, and
whether it is CURRENT is `popart/norm_return_*` (below). Full background:
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

**Counts, measured 2026-09-06** — 153 static `logger.record(` sites across `src/agents/training`,
`src/main/train`, `src/main/eval_worker.py` and `src/main/elo.py`; 185 distinct tags observed
across three CPU smokes (a plain run, a `--win-prob-mode read_only` run, and a `--use-popart` run).
The two do not match and should not: one site can emit a whole dict (`f"reward/{k}"`), and many
sites are flag-gated off in any one run. **Recount before quoting** — `tmp_census.py`'s recipe is
`grep -rn "logger.record(" src/agents/training src/main/train src/main/eval_worker.py` for the
sites and an `EventAccumulator` walk of a run's `tb/` for the tags.

| group | sites | tags seen | cadence | currency | computed in |
|---|---:|---:|---|---|---|
| `reward/` | 1 | 46 | **per rollout** | RAW REWARD | `reward_term_callback` ← `reward_term_stats` |
| `train/` | 53 | 23 | per rollout (`train()`) | MIXED — see per-tag below | `instrumented_ppo/ppo.py`, `grad_balance`, `run_io` |
| `win_prob/` | 5 | 42 (+10 under `--critic winprob`) | per rollout | PROBABILITY | `ppo.py` ← `value_terms`, `calibration`, `scaffolding.reliability_table` |
| `eval/` | 35 | — | **per EVAL CYCLE** | win rate / ELO / reward | `eval_callback`, `selfplay_callback` |
| `eval_final/` | 2 | 10 | once, at run end | win rate | `main/train/final_eval.py` |
| `signal/` | 4 | 12 | per rollout | NORMALIZED (adv) / probability (outcome) / rate (draw) | `signal_metrics`, `signal_callback` |
| `popart/` | 5 | 5 | per rollout | raw (μ,σ) + unitless (norm) | `ppo.py` |
| `grad/` | (dynamic) | 16 | per rollout | unitless shares | `grad_balance` |
| `rank/` | 6 | 18 | per rollout | unitless | `rank_tripwire`, `rank_metrics` |
| `belief/` | 1 | 8 | per rollout | accuracy / CE | `belief_bank` |
| `distill/` | 7 | — | per rollout | KL / MSE / rate | `distill_terms`, `distill_anchor*`, `distill_stop_callback` |
| `cf/` | 5 | — | per rollout | probability + counts | `cf_terms`, `cf_label_buffer` |
| `teacher/` · `opd/` | 11 | — | per cycle / rollout | CE / KL | `teacher/callback`, `ppo.py` |
| `team_pfsp/` · `hparams/` · `capacity/` · `defent/` · `baitent/` · `value_dist/` · `td_aux/` · `q_winprob/` | 20 | — | per rollout | see each section | their own callbacks |

### The five groups the diagnostic contract names

#### `reward/` — WHAT THE REWARD IS MADE OF (`gen3_reward_term_export_v1`, 2026-09-06)

Per-rollout, RAW REWARD units, one triple per ACTIVE term of this run's composition plus four class
rollups and four totals. Full rationale — including why the share is `|·|`-weighted and why the
residual is a GIGO guard rather than a rounding term — is in
`agents/training/reward_term_stats.py`'s module docstring.

| tag | is |
|---|---|
| `reward/<term>_mean` | Σterm ÷ decisions. **A PBRS term should read ≈0** over an episode-complete window — that is the telescoping, visible |
| `reward/<term>_abs_share` | `Σ\|term\| / Σ_terms Σ\|term\|` — this term's share of the reward stream's MOVEMENT. **The shares partition to 1** |
| `reward/<term>_abs_mean` | `Σ\|term\|` ÷ decisions, the un-normalized magnitude |
| `reward/class_{terminal,pbrs,bias,refund}_{mean,abs_mean,abs_share}` | the same, rolled up by `RewardClass` |
| `reward/total_{mean,abs_mean}` · `reward/n_decisions` | the stream itself and the window size |
| `reward/untracked_abs_mean` | **THE GIGO GUARD** — `mean\|bd.total − Σ tracked\|`. Reads exactly 0.0 when the startup composition census and the folds agree; anything else means they do not |

The tracked set is derived from `reward_class_composition(config)` — the SAME `_pbrs_term_active` /
`_bias_term_active` predicates the folds are gated on — so the exported terms cannot disagree with
the startup line. Under the default composition that is 10 terms (1 terminal + 7 PBRS + 1 bias +
the refund mechanism) → 46 tags; under `--no-all-shaping-pbrs` it is 28 terms → ~100 tags. Bounded
by the REGISTRY, never per-team.

**Transport: an `env_method` PULL, not an info-dict thread.** The reward is computed in the env
WORKER, and under `--async-rollout` the callback's step locals arrive wave-batched with no way to
recover which buffer row a step landed on — the same reason `TeamWinRateCallback` uses this seam.
`AsyncSubprocVecEnv.env_method` is drain-safe, so one seam covers both collectors, and
`RewardTermAccumulator.drain()` zeroes the window so a double pull cannot double-count. ALWAYS ON,
no flag: the accumulator folds only the ACTIVE terms (9 of 35 under the production composition).

#### `win_prob/` — the head's PREDICTION, its CALIBRATION, and the paired episode-start read

Per rollout, PROBABILITY units, epoch 0 only (by epoch 3 the policy that produced a pair is not the
policy it is attributed to).

| tag | is | added |
|---|---|---|
| `loss` · `acc` · `brier` · `pred_mean` · `label_mean` · `coverage` | the pre-existing fit meters | |
| `brier_contested` · `acc_contested` · `contested_frac` · `contested_label_mean` · `brier_material` · `skill_vs_material` | the information-value half, restricted to material-EVEN decisions | |
| **`ece`** | 10-bin count-weighted Expected Calibration Error. **Brier is a PROPER score and decomposes as reliability − resolution + uncertainty, so it can stay flat while calibration drifts**; this isolates the reliability term | ✅ 2026-09-06 |
| **`mce`** | the WORST readable bin's gap — ECE is an average, so a head badly wrong only on the confident tail holds a small ECE | ✅ |
| **`rel_gap_b0` … `rel_gap_b9`** | the reliability HISTOGRAM, one scalar per bin, so the SHAPE of the miscalibration is readable. **A bin under 100 samples publishes NaN**, which TensorBoard renders as a hole — a 3-sample bin's "error" is sampling noise | ✅ |
| **`rel_n`** | rows the diagram was built from | ✅ |
| **`contested_*`** | every one of the above, restricted to `\|win_margin\| < 0.25` — a blowout's P(win) is trivially recoverable from material, so the pooled ECE is flattered by exactly the states nobody needs the head for | ✅ |
| **`start_pred_mean` · `start_realized_mean` · `start_gap` · `start_n`** | **THE PAIRED EPISODE-START READ** — what the head says at the LEAST-informed state against what those very episodes went on to do | ✅ |
| **`start_*_{bots,pool,stable,target}`** | the same, split by opponent class — **`start_*_pool` IS "the self-play win probability at episode start vs the realized self-play win rate"** | ✅ |

🚨 **THE EPISODE-START READ IS PAIRED, AND THAT IS THE WHOLE POINT.** `win_target` is back-filled by
`WinProbLabelCallback` from the episode's own outcome to every step of that episode, so at an
episode-START row it IS the realized outcome of the game that starts there. Prediction and
realization therefore come from ONE set of episodes and `start_gap` is a paired difference — not
the difference of two independently-windowed averages, which would carry the two windows'
disagreement as well as the head's error. **POSITIVE = optimistic at the opening board.** Cost: one
EAGER forward over the episode-start rows (capped at `_WINPROB_START_MAX_ROWS` = 1024, a
deterministic prefix, never sampled) once per `train()` — eager `type(fe).forward` for the
capacity-probe's reason (both compile flags patch the BOUND attribute, and a second obs shape
through the compiled entry point would add a dynamo graph for a diagnostic).

⚠️ **The per-class split is OPPORTUNISTIC**: it needs the `opp_class` obs key, which the env emits
only alongside the opponent-intent labels (`--opp-intent-coef > 0`). Without it the POOLED read
still ships, and `signal/outcome_win_rate_<kind>` carries the realized per-class rate
unconditionally — so the self-play realized rate is never missing, only its paired partner is.

**`win_prob/vs_critic_divergence` does not exist under that name; the scalar is
`train/scaffolding_gauge`** — `(1 − Spearman ρ(V, P(win)))/2` over epoch 0's paired reads, with
`train/scaffolding_rho` and `train/scaffolding_n` beside it. It is NOT renamed: the name is
non-obvious but not misleading, it is the subject of a documented section and an offline CLI
(`python -m main.scaffolding_gauge`), and dashboards read it. See *The SCAFFOLDING GAUGE* below.

#### `signal/` — advantage density and the REALIZED per-class win rate

| tag | is | currency |
|---|---|---|
| **`adv_raw_mean`** | mean RAW GAE advantage — the NO-HARM watch. A mean far from 0 relative to `adv_raw_std` is a systematically MIS-CENTRED critic, and `normalize_advantage` erases it per minibatch so nothing else can report it. **Read as the ratio to `adv_raw_std`** | NORMALIZED ✅ 2026-09-06 |
| `adv_raw_std` · `adv_raw_abs_mean` · `adv_kurtosis` | the density and its shape | NORMALIZED (kurtosis scale-free) |
| `outcome_entropy[_<kind>]` · `outcome_n[_<kind>]` | `p(1−p)` over a rolling 200-episode window | probability |
| **`outcome_win_rate_<kind>`** | **the REALIZED per-class win rate.** `p(1−p)` is SYMMETRIC about 0.5, so `outcome_entropy_pool = 0.16` means p = 0.2 **or** 0.8 and nothing in the export said which — for two generations only the entropy shipped per kind. Free: the same deque, one more mean | probability ✅ 2026-09-06 |

#### `popart/` — and whether the currency conversion is CURRENT

| tag | is |
|---|---|
| `mu` · `sigma` | what the normalizer BELIEVES the return mean and scale are. **Should TRACK `train/return_mean` / `train/return_std`** |
| `value_weight_norm` | the POP rescale staying bounded |
| **`norm_return_mean`** | `mean((returns − μ)/σ)` — **≈0 when the conversion is current.** Far from 0 is an offset the value head has to carry itself | ✅ 2026-09-06 |
| **`norm_return_std`** | `std((returns − μ)/σ)` — **≈1 when the conversion is current.** Drifting from 1 is PopArt LAGGING the return scale, and the value gradient is then mis-scaled against the shared trunk by exactly that factor | ✅ 2026-09-06 |

μ and σ alone say what the normalizer believes; these two apply the conversion to THIS rollout's own
returns and say whether the belief is current. Free — a mean and a std over an array
`value_scale_metrics` has already read. Emitted only under `--use-popart`.

#### `train/` — value-function health, and the EXPLAINED-VARIANCE currency question

| tag | is | currency |
|---|---|---|
| `explained_variance` | `1 − Var(returns − values)/Var(returns)`, over the whole rollout pooled | **see the note below** |
| `return_mean` · `return_std` · `return_abs_max` | the value TARGETS' scale. **`return_std` IS the value-target std** | RAW SHAPED RETURN |
| `value_pred_std` | the critic's own output spread | RAW SHAPED RETURN |
| `value_loss` | the fitted loss | NORMALIZED under PopArt, raw otherwise |
| `policy_gradient_loss` · `entropy_loss` · `loss` · `approx_kl` · `clip_fraction` · `clip_range[_vf]` · `grad_norm` · `n_updates` | the stock PPO step | unitless / loss units |
| `scaffolding_gauge` · `scaffolding_rho` · `scaffolding_n` | the shaped critic vs the win-prob head | unitless (rank) |
| `noise_scale[_ratio][_<term>]` · `dose_rate` · `effective_batch` · `grad_accum_steps` · `train_ms` | the step-size controllers | see their sections |

🚨 **`train/explained_variance` IS THE SAME NUMBER IN BOTH CURRENCIES, and a second "normalized"
tag would be a duplicate curve rather than a second measurement.** EV is
`1 − Var(y − ŷ)/Var(y)`, and PopArt applies the SAME affine map `(·−μ)/σ` to both `y` and `ŷ`
(`policy._critic_value` de-normalizes, so `rollout_buffer.values` and `returns` are both RAW). A
shared affine map cancels: `Var(a(y−ŷ))/Var(a·y)` is unchanged for any `a ≠ 0`, and the `−μ` cancels
inside both variances. **So SB3's default is computed on the RAW shaped-return arrays, and the
PopArt-normalized EV is numerically identical to it.** That is worth stating rather than shipping,
because "which currency is this EV in?" is a question a reader will otherwise ask on every run.

⚠️ **`train/value_target_std` does not exist and is not needed: it is `train/return_std`.** The
value targets ARE `rollout_buffer.returns`. Not renamed — `return_std` is accurate, sits beside its
own `return_mean`/`return_abs_max` family, and dashboards read it.
Likewise **`train/advantage_mean` / `train/advantage_std` are `signal/adv_raw_mean` /
`signal/adv_raw_std`** — the `signal/` group is where the raw pre-normalization advantages are read
(the ONE place they still exist), and duplicating them under `train/` would give two names to one
number.

### What is NOT exported, deliberately

* **A per-team reward or win-rate SERIES.** Owner rule (design_flywheel_tick_tock.md §6b): per-team
  curves are noisy spam. The per-team win-rate table rides `metadata.json`'s `team_win_rates` block
  instead, and `TeamWinRateCallback` has a test that FAILS if anything is emitted to TensorBoard.
* **A reliability histogram bin's COUNT.** `rel_gap_b<k>` publishes NaN below 100 samples, so an
  under-populated bin renders as a hole; adding 10 count tags to say the same thing would double the
  group for no reading.
* **The opponent's identity beyond its CLASS.** Only the `OPP_CLASS_*` integer crosses the env-worker
  pipe, so `_bots` / `_pool` / `_stable` / `_target` are real and finer identity (which heuristic,
  which pool snapshot) is not. `signal/outcome_entropy_rung` is the one finer split, and it exists
  only because `ExploiterLadderCallback` keeps its own per-rung window in the parent process.

## The PPO step (`instrumented_ppo/`) — and the FOLD ORDER contract

**`instrumented_ppo` is a PACKAGE** (2026-08-23; it was a single 2,152-line file, the last entry
on the size ratchet's grandfathered list). `__init__.py` is a pure re-export hub, so every
`from agents.training.instrumented_ppo import <name>` resolves unchanged:

| module | holds |
|---|---|
| `ppo.py` | `InstrumentedMaskablePPO` + `train()` — the vendored upstream override and **the whole fold sequence** |
| `hparams.py` | every after-construction knob `train_rl_agent` sets (`value_tail_weight`, the belief/intent/cf coefficients, `grad_accum_steps`, …) with the rationale comment each carries, plus `_excluded_save_params` |
| `noise_scale.py` | the McCandlish gradient-noise-scale estimator + the rate-limited NSR advisor + `noise_ratio_sample`, the read seam `--adaptive-batch` steers by |
| `noise_scale_terms.py` | the PER-LOSS-TERM half of it — is the total reading the POLICY gradient's, or the dense aux heads'? |
| `distill_terms.py` | search-teacher AWR · OPD · the exploiter-distillation family (policy KL — or the top-K/action-CE form with the advantage gate, `_gated_action_distill_loss` — value MSE, the FitNets hint) |
| `value_terms.py` | the win-prob BCE · the value-dist HL-Gauss CE · `_value_loss_from_se` |
| `aux_terms.py` | the `belief_bank` / `td_aux` / `cf_terms` delegates |
| `constants.py` | `_VALUE_TAIL_FRAC` · `_WIN_CONTESTED_TAU` · `_NOISE_SCALE_EMA_DECAY` |

**`train()` is deliberately NOT split**, and the reason is the contract below. It is ~1,250 lines
in one module because the ORDER the terms are folded in is straight-line source order, and that is
only checkable by reading while it stays one straight line. Per minibatch:

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
`threading.Lock`).

The upstream-drift hash check (`_verify_upstream_unchanged` + `_EXPECTED_UPSTREAM_TRAIN_HASH`)
stays in the HUB on purpose: `instrumented_ppo_test` patches that global on the module object it
imports, so moving it into a submodule would have left the patch reaching a different global than
the function reads — a test that still passes, for the wrong reason.

## Reward redesign — registry + PBRS + the no-progress clock (`reward_manager.py`, `progress_clock.py`)

> **Where the reward lives.** `reward_manager.py` (the terms, the folds, `RewardConfig`,
> `RewardBreakdown` and the composition census) · `reward_weights.py` (every tunable MAGNITUDE —
> weights, bonuses, thresholds, clamps; re-exported by `reward_manager`, so the old import path
> still resolves) · `reward_verify.py` (the `GEN3AI_REWARD_VERIFY=1` shadow twin) ·
> `progress_clock.py` (the no-progress clock the reward READS). Changing a value in
> `reward_weights.py` is a RETRAIN-class change, not a knob.

The reward (`Gen3RewardManager`) is organised as a **registry of class-tagged terms**
(design `designs/ai_v5/design_markovian_reward_and_features.md`). Every `RewardBreakdown` field is one
entry in `RewardBreakdown._REGISTRY` mapping name → `RewardClass`. The **BIAS class is folded
generically** off the registry (`_fold_bias_refund` sums `registry_fields(BIAS)`); TERMINAL and the
PBRS terms are **explicit named folds** (`_fold_material_pbrs` / `_fold_belief_pbrs` /
`_fold_status_pbrs` + the v13 `_fold_{progress,hazard,boost,opp_boosts}_pbrs`) because each PBRS term
carries its own `_prev_phi_*` telescoping state a generic
loop can't hold — `process_turn_reward` reads as a short phase sequence over these helpers:

- **TERMINAL** (`win_loss`, the ±30) — emitted as-is; never shaped/flag-affected. Out of scope.
- **PBRS** (always telescoping, objective-neutral; `Φ(terminal)=0`): `pbrs_material` (the material
  potential **Φ_mat**, design §2), `pbrs_belief` (the shipped incoming-KO belief PBRS — RENAMED from
  the mis-named `pbrs_material`), `pbrs_status` (the non-damaging-tempo status potential **Φ_status**,
  design §2.7 — `bias_redesign`- OR `all_shaping_pbrs`-gated, see below), and the **four v13/v14 end-state
  potentials** (see **End-state PBRS** below): `pbrs_progress` (**Φ_progress** =
  −`no_progress_penalty`·`progress_clock.value()`, the anti-stall clock as a telescoping potential —
  **`--stall-pbrs`-gated**; the other three are **`--all-shaping-pbrs`-gated**),
  `pbrs_hazard` (**Φ_hazard** = `HAZARD_WEIGHT`·(opp − our spike layers), design §2.6), `pbrs_boost`
  (**Φ_boost** = `BOOST_WEIGHT`·Σmax(0,our-active-boost)·hp_frac, the stored offense), and
  `pbrs_opp_boosts` (**Φ_opp_boosts** = −`OPP_BOOST_WEIGHT`·Σmax(0,opp-active-boost), the phaze value),
  and `pbrs_roar` (**Φ_roar** = −`ROAR_BOOST_WEIGHT`(0.25)·Σmax(0,opp-active-boost), the **DEDICATED**
  phaze-out-boosts PBRS — **folded INTO `--all-shaping-pbrs`** (no separate flag/version, owner request);
  same state-potential shape as `pbrs_opp_boosts` but its own weight, so a successful Roar pays out
  `+ROAR_BOOST_WEIGHT·(stages cleared)`. A PBRS can't be action-keyed without becoming a BIAS, so it IS the
  same potential — under `--all-shaping-pbrs` the two STACK; safe, both telescope to 0 → policy-invariant,
  the effect is just stronger proportional roar shaping).
  The field holds `γ·Φ(s′)−Φ(s)`; `PBRS_GAMMA` MUST ==
  the PPO gamma (asserted in `train_rl_agent.py` after the model is built — the manager is built first,
  in the env factory, so it can't assert in `__init__`).
- **BIAS** (everything else) — additive shaping whose additive↔telescoping mix is set by
  `--bias-additivity` λ∈[0,1] (`RewardConfig.bias_additivity`, default 1.0). Implemented as
  **accumulate-and-refund**: each BIAS term emits its current per-turn value; the manager accumulates
  `_bias_acc` and emits `bias_refund = −(1−λ)·Δacc` (the low-variance accumulator-potential spread). At
  **λ=1 the refund is identically 0** → byte-identical to the old additive biases (the no-op the
  registry-coverage / no-op-equivalence tests pin).

**Φ_mat** (`_compute_phi_mat`) = `MAT_HP_WEIGHT·(Σ our_hp − Σ opp_hp) + MAT_ALIVE_WEIGHT·(n_alive_ours
− n_alive_opp)`, over the **declared team size** (unrevealed opp mons = full-HP-alive → `Φ_mat(s_0)≈0`,
no opp-reveal jumps, no start-state variance). It REPLACES the old unconditional `hp_ours/hp_opp/
faint_ours/faint_opp` base spine — material no longer banks the lead, so every win returns +30 / loss
−30 (the clutch-vs-dominant fix). The old asymmetric `−0.75 FAINT_MATERIAL_PENALTY` is REMOVED (folded
into `MAT_ALIVE_WEIGHT=1.25`, a state potential, not a bias). The `+2.0` explosion literal is deleted
(survive-Explosion credit rides Φ_mat); `explosion_block` is kept.

**Φ_status** (`_compute_phi_status` / `_fold_status_pbrs`, `pbrs_status`) = `STATUS_TEMPO_WEIGHT·(opp_tempo
_statused − our_tempo_statused)` over **non-fainted par/slp/frz mons only** (`_TEMPO_STATUSES`). It
restores the *standing* value of a held non-damaging status that the event-form `status` reframe drops —
sleep/freeze/para "lose the opponent turns", value `Φ_mat` can't see (Toxic/burn/poison value is the chip
→ already in `Φ_mat`, so they're excluded to avoid a double-bridge). Nobody is statused at `s_0` →
`Φ_status(s_0)=0`, `Φ_status(terminal)=0` → it telescopes to **zero net** (policy-invariant dense signal,
not a net bias). **Gated on `bias_redesign`** (the default count-diff `status` BIAS already pays the
standing value → folding `Φ_status` there double-counts; OFF → `pbrs_status≡0`, `_prev_phi_status` stays
None, byte-identical default). It adds **no** resume-immutable field — it rides the existing
`bias_redesign` flag (design §2.7 / §7.4 hedge).

**The no-progress clock** (`ProgressClock`, `progress_clock.py`) is an episode-scoped
`turns_since_progress` counter **owned by `EpisodeTracker`** (NOT LiveView — it is cross-turn state;
precedent = `HiddenPowerTracker`). It is updated at `record()`/`embed_battle` time (so the obs is fresh
— poke-env runs `embed_battle` before `calc_reward`), and read by BOTH the obs encoder (`value()` →
`reactive_layout["turns_since_progress"]`, absolute obs column **1602**) and the reward
(`last_penalty` → `no_progress_tax`), so **obs and reward key on one value**. The ternary predicate per decision window: PROGRESS (our-attributed damage ≥3% / status
landed / hazard layer / forced opp commit / **an our-owned residual — Toxic/poison/burn or Leech
Seed/Curse/Nightmare — chipping the opp NET-down** → reset), DENIED (freeze), NO_OP (deliberate
wheel-spin → increment + charge, gated off on forced-switch windows and when no switch is legal).
DENIED splits two ways (`_denial_kind`): **exogenous** (miss / Protect-block / cant) is ALWAYS frozen;
a **productive heal** is frozen only for `HEAL_FREEZE_GRACE`=2 consecutive windows — a SUSTAINED heal
with no progress (the self-play mirror heal-war) then falls through to NO_OP and CHARGES, so the
250-turn stall finally registers. **Rest-loop (`gen3_rest_loop_stall_v1`):** a REST that already happened
this episode for the same species — i.e. our active woke and re-Rested — gets NO heal-grace at all
(`_update_rest_loop` sets `_is_rest_loop`, read in the heal branch), so a wake-then-re-Rest is a NO_OP
stalled turn the moment it repeats; a mon carrying **Sleep Talk** is exempt (looping Rest is a legitimate
act-while-asleep strategy, and our own moveset is fully known so the check is exact). **Setup-progress
(`gen3_setup_progress_v1`, unconditional correctness fix — clauses (vi)/(vii)/(viii) of `_is_progress`):** the
predicate had NO clause for an own stat-boost rising, a Substitute being made, or a Wish being cast, so a
PRODUCTIVE setup turn (a first Calm Mind / Dragon Dance / Swords Dance / Curse / Belly Drum, a fresh Sub, or a
Wish cast) was charged identically to an idle wheel-spin — the one stall-break route the reward actively
discouraged. Three clauses now count a **NON-redundant** setup as PROGRESS: our active's Σ positive boost
stages STRICTLY rose (a +6-capped repeat leaves the sum unchanged → still charged) OR a Substitute was NEWLY
created (a failed re-Sub while one is up → still charged) OR a **WISH was SUCCESSFULLY cast** (`gen3_wish_wired_v1`
— a pending ~50%-maxhp heal; a double-Wish FAILS → outcome 'fail' → still charged, keyed on the move id like
the Rest/Spikes clauses). Read from `live.ours.active.{boosts,volatiles}` + the delta's move id, with
`_prev_our_boost_sum`/`_prev_our_has_sub` trackers mirroring the spikes-layer pattern; the +6 cap +
Sub-can't-restack + its 25% HP cost bound how long it can keep resetting; in gen3 only our own move raises
our boosts and a switch-in is boostless (boosts reset on switch), so a pivot/opp action can't false-credit.
**Always-on** (not flag- or version-gated — a clock-predicate correctness fix like `gen3_rest_loop_stall_v1`,
but with NO `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` bump, so an in-flight run picks it up on resume). The
residual-PROGRESS
branch is what keeps a *winning* Toxic/Leech
defensive stall from being taxed (the discriminator is the opp net-losing HP; a heal-war where they
out-heal the tick still charges) — and because it runs FIRST, a winning *rest-stall* (Rest while Toxic
chips the opp down) is exempt too — validated end-to-end by `progress_clock_fuzz_test.py` (bridge, real
battles: a winning-residual window is never charged). The env (`gen3_env.py`) folds the delta once at
embed time, updates the clock, caches it for `calc_reward` (no double fold), and wires
`reward_manager.progress_clock = tracker.progress_clock`.

**Three futile-move short-circuits** (BEFORE the PROGRESS check, so an incidental opp switch — or, for
(3), a winning residual via clause (v) — can't launder them): **(1) capped Spikes** — Spikes used at the
3-layer cap can never add a layer, so it is charged as a NO_OP directly (a layer-ADDING Spikes still
resets via the hazard clause); **(2) filler RapidSpin** — RapidSpin with NO spikes on our side to clear is
a 20-BP filler pseudo-attack, so its trivial chip is barred from counting as progress and it falls through
to the NO_OP charge (a spin that genuinely clears our hazards, lands a KO, or is RNG-denied is handled
normally); **(3) wasted Refresh** (folded into `gen3_rest_loop_stall_v1`, `_is_wasted_self_cure`) — a self-status-cure
move (`cures_self_status`, i.e. Refresh) used with no status to cure (`our_status_cured is None`, not a cant)
does nothing, so it is charged as a NO_OP directly — crucially even when our Leech Seed / Toxic is chipping
the opp NET-down (which clause (v) would otherwise credit as progress), killing the observed
Refresh-spam-while-seeded stall (a Refresh that ACTUALLY cures a status sets `our_status_cured` → not wasted →
normal path). The first two target the self-play Spikes/RapidSpin wheel-spin loops the flat anti-spam taxes
missed; the third targets degenerate self-cure spam during a passive residual stall.

### The clock's two intent-restoring fixes — `--progress-decision-tense` / `--progress-switch-freeze`

**Both default OFF; a flagless run is byte-identical to what every generation through gen-15
trained** (proved by `gen3_data_obs_parity_integration_test`'s committed golden and by
`progress_clock_test.py`'s recorded default trace, captured against the pre-fix implementation).
They are `RewardConfig` fields — resume-immutable, value-checked, recorded in `model_config.json`,
`MODEL_CONFIG_VERSION` 105, **no `ARCH_SIGNATURE` bump** — and are threaded onto the clock by ONE
call, `ProgressClock.apply_reward_config(cfg)`, used by both `gen3_env.py` and `reward_tracker.py`
so training and eval cannot drift on what the clock does.

**Why they exist, in one line each.** Probe M censused what the tax actually charges
(`designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md`) and probe N traced the
term's intent against its implementation (`.../no_progress_tax_review_2026-08-29.md`). Between them
they found that **79% of all charges land on the two paths below**, and that neither is what the
design specified.

| flag | what it changes | measured motivation |
|---|---|---|
| `--progress-decision-tense` | both window GATES (the forced-switch sit-out **and** the trapped-vs-wall charge suppression) read the decision that OPENED the window instead of the one after it | the sit-out lands on **19,503 full-agency decisions** (13.2%, the costliest class at −5.1pp) while the **zero-agency post-faint replacement is charged 63.9%** of the time — **36.3% of all charges** |
| `--progress-switch-freeze` | a VOLUNTARY switch that fails `_is_progress` FREEZES the window (no increment, no charge) rather than being taxed | `_is_progress` is offense-only, so no switch can satisfy it by its own doing: **−0.101** expected charge per voluntary switch vs **−0.010** per move, and within the switch branch the discrimination is **INVERTED** (Δ mean `d_out` **+0.0103** [+0.0076, +0.0131] — the charged switches are worth *more* win probability). **42.7% of all charges** |

**The tense fix is ONE off-by-one in ONE call, and it has two halves that must move together.**
`phase_is_forced_switch` reads `curr_ctx.phase` — the phase of the request that CLOSES the window —
because it was minted eleven days earlier for the obs history slot, where that is the correct read.
`ProgressClock` reused it for "was the decision that opened this window forced", and no test could
catch it: **both readings are true statements about the same delta**. The same call also passed the
upcoming request's `legal` to the helplessness gate, so a mon genuinely trapped at `t` is charged
whenever its successor could switch. The fix adds a NEW field, `TurnDelta.decision_was_forced_switch`
(`prev_ctx.phase == "forced_switch"`), and threads `legal_prev` alongside `legal`;
**`phase_is_forced_switch` is deliberately untouched** — the obs decoder, `opp_intent_labels` and
`reward_manager` all want the closing tense, and re-pointing it would silently change what they mean.

**F2b, not F2a.** Probe N specified two spellings and this is the one that ships: the alternative
(give `_is_progress` a switch clause keyed on belief-delta or type-matchup) would reintroduce exactly
the hand-coded switch heuristic `928a00b` deleted on the argument that switching value is LEARNABLE
from Φ_mat + `pbrs_belief` + the terminal. The freeze is instead the **composition-corrected reading
of the original intent**: the design's "a pure tempo-pivot pays the toll once" was written for a
reward that also paid the same pivot `switch_base +0.5` / `se_switch +0.2` / `escape_threat +0.25`;
`928a00b` zeroed every one of those and explicitly kept the tax, so the sign of the net switch
incentive flipped with nobody re-deriving the term. The **honest cost** is that a pure A↔B
switch-loop becomes free — anti-stall survives via the move turns between pivots, `--draw-penalty`
and the 250-turn forfeit, and **stall rate / mean game length is the canary** on any arm that runs it.

**Both are RETRAIN-CLASS when ON, and the blast radius is measured rather than argued.** `n` is the
obs scalar as well as the charge basis (that identity is the Markovian design's whole premise — a
fix that moved only the reward would break it), so turning either on changes the observation stream.
`progress_clock_obs_confinement_integration_test.py` captures the golden 6-battle set under each arm
and reports every differing cell: **`--progress-decision-tense` 49/991 decisions, `--progress-switch-freeze`
153/991, and in both cases the ONLY column that ever differs is 1602** — `turns_since_progress`
itself. No other block moves, no dim moves, the trajectory does not branch.

**Server-free reward parity (`reward_tracker.py`).** The offline reward path (`RewardTracker`, used by
`BattleRecorder` + the eval `RewardTrackingMixin`) has no `Gen3Env` to own the clock, so it OWNS a
per-battle `ProgressClock` itself and advances it before each `process_turn_reward` — mirroring the
env's embed-time timing. Without this, eval traces scored `no_progress_tax`=0 (clock absent) and the
prober **understated the training penalty on every stall/no-op turn**; now the recorded reward matches
training (the gate is still `all_shaping_pbrs`/`bias_redesign` in the run's `RewardConfig`, so a
default-config run stays byte-identical).

**Anti-stall terminal (`--draw-penalty`, DEFAULT −35.0).** The trainee FORFEITS a
stalled battle at the turn cap (`gen3_env` `ForfeitBattleOrder` at turn ≥ `StallConfig.threshold`), so
a 250-turn stall ends as a forfeit-**loss** (`lost=True`), NOT a tie. The terminal therefore detects a
timeout by **`live.turn >= _TIMEOUT_TURN_CAP`** (synced to `StallConfig.threshold`), not by won/lost:
`if won: +30; elif finished: draw_penalty if timed_out else −30`. At the default −35.0 a stall-to-cap
is strictly worse than a clean loss, which cancels the γ=0.9999 discount pull of delaying an inevitable
−30. `--draw-penalty -30` restores the historical default (a tie scored identically to a decisive
loss); it was tuned under the additive-BIAS regime `--all-shaping-pbrs` replaces, which is why the two
defaults flipped together (owner decision 2026-08-18 — see **The reward COMPOSITION** below).
Resume-immutable, value-checked (`MODEL_CONFIG_VERSION 6→7`, `check_reward_config`).

**Staged rollout (`RewardConfig.bias_redesign`, `--bias-redesign`, default OFF).** OFF = the
**single-variable default run**: today's anti-spam taxes + roar/status/spikes, so the ONLY reward
change vs the live baseline is the material clutch-fix (clean attribution). ON = the no-progress clock
SUBSUMES the escalating anti-spam family (repetition/bouncing/dead-matchup/struggle suppressed) and the
clock charge is active. The `turns_since_progress` OBS scalar is present EITHER way (the clock always
tracks it), so both arms share one architecture and can A/B by resume. `--bias-additivity` /
`--mat-alive-weight` / `--bias-redesign` are resume-immutable, value-checked by
`ModelVersion.check_reward_config` (the same machinery as `--vf-coef`). Tests: the `reward_*_test.py`
per-term spec family over `reward_test_fakes.py`
(registry coverage, Φ_mat telescoping + terminal-zeroing, **Φ_status non-damaging-only + gated-off-default
+ telescopes-to-zero**, bias no-op + parameterized blend, the bias_redesign reframes, the full
ProgressClock predicate), plus the updated `reward_manager_test.py`.

**Belief-risk-scaled switch BIAS lever (`--switch-bias-weight`, default 0.0 = OFF).** The shipped
`pbrs_belief` is policy-INVARIANT (a telescoping potential) so it can't move a *converged* under-switch
preference — verified on `run_20260607_102632`: switch-mass still inverts vs P(KO), stay-and-die ≈ 61%
== the V1 control. The fix (`design_reward_switching.md §7`, `impl_step6`) adds two **BIAS-class** terms
that *do* tilt the objective: `stay_risk_tax = max(−w·risk, −2.0)` for STAYING into a high imminent-KO
spot a safe pivot could escape, and `escape_risk_bonus = w·0.5·risk` for escaping it (asymmetric < the
tax → no farm). `risk = max(phys_pko,spec_pko)·(1−P(outspeed))` from the incoming belief. Hardened gates
(red-teamed): never tax a **trapped** stay (`_cur_can_switch` from the decision-time `ctx.mask`), an RNG
fizzle (`our_failed_to_move`), a KO'ing stay (`opp_fainted`), or a forced stay (a `_prev_safe_pivot`
bench mon with raw P(KO) ≤ `SAFE_PIVOT_PKO_MAX`=0.35 must exist; the escape bonus needs it too). Snapshots
are decision-time (set end of last turn / in `record_action`), read before `_fold_belief_pbrs` overwrites
them. **Reward-only — no obs/arch change** (ARCH unchanged; `MODEL_CONFIG_VERSION 4→5`), resume-immutable
(`check_reward_config`). Being BIAS-class it rides `--bias-additivity`, so a fixed weight at **λ=1 vs λ=0**
is the causal A/B for "is it the objective tilt that helps." Tests: `reward_bias_terms_test.py::TestSwitchBias`.

**HP-scaled self-KO penalty (`--self-ko-hp-penalty`, default 0.0 = OFF).** A grounded floor-leak fix
(2026-06-12 forensics on ai_v5_11): the policy confidently (median P≈0.5) explodes **healthy** mons —
~38% of all Explosion/Self-Destruct selections are at ≥80% HP (incl. turn-1 full-HP Metagross),
human-obvious blunders that cost ~0.95 mon. Mechanism (ruled out reward+exploration first): the
**reward is correct** (a healthy non-trade Explosion scores ≈−2.7; the finishing-blow mis-credit is
already guarded), but Φ_mat is **symmetric for a 1-for-1 trade** (our −hp/−alive cancels theirs → ~0),
so on the 77%-of-the-time trade the critic learns to value the post-self-KO board POSITIVELY
(measured `dV ≈ +2.9`), which **neutralizes the −2.7 reward in the PPO advantage** (`r+γV′ ≈ +1.5`, 74%
≥0) and the policy never un-learns it. (It is NOT the old ① active-value readout — the no-① baseline explodes
just as much; that toggle is deleted, v88.) The fix is a **BIAS-class** term `−w·(our active HP fraction at decision time)` charged
when our mon self-KOs (`our_move_id ∈ SELF_KO_MOVES` + `we_fainted` + not `our_failed_to_move`), using
the `_our_active_hp_before` snapshot from `record_action`. Scaling by HP **spares the legitimate low-HP
"explode a dying mon for a KO"** (≈0 penalty). A static pre-check showed `w≈2.5` flips the healthy-trade
advantage negative; in a retrain the critic's over-valuation also drops as the TD target sharpens.
Reward-only — no obs/arch change (no `ARCH_SIGNATURE` bump; `MODEL_CONFIG_VERSION 11→12`),
resume-immutable (`check_reward_config`). **Validate by watching `win_rate_vs_bots` (82%→~95% target)
and the healthy-explosion rate fall.** Tests: `reward_end_state_test.py::TestSelfKoPenalty` (unit) +
`self_ko_penalty_fuzz_test.py` (bridge — real Explosion turns net exactly `−w·hp`, 0 elsewhere, OFF
byte-unchanged).

**De-bias cleanup (`--drop-redundant-bias` / `--drop-switch-bias`, default OFF).** A distortion audit
(ranking the BIAS terms by their ability to move the converged optimum away from win-maximization)
flagged three TIER-1 distorters; these two flags ZERO them in `_apply_bias_drops`, called **right
before** `_fold_bias_refund` so the dropped terms leave the bias accumulator too. Both default OFF =
byte-identical (the no-op tests pin it); each is resume-immutable + value-checked
(`MODEL_CONFIG_VERSION` v13, `check_reward_config`), no `ARCH_SIGNATURE` bump (reward-value only).
- **`--drop-redundant-bias`** drops `stall_tax` (a raw-turn-count ramp that also taxes a *winning*
  long game — the progress-aware `no_progress_tax` clock + the `--draw-penalty` terminal already cover
  stalling) and `matchup_penalty` (the same incoming-KO threat signal as the telescoping `pbrs_belief`
  PBRS term, but BIAS-class/additive → it distorts where `pbrs_belief` is policy-invariant).
- **`--drop-switch-bias`** drops the HAND-CODED switch-strategy subsidy (`switch_base`,
  `switch_bouncing_tax`, `escape_threat_switch`, `se_switch`, `pivot_protect/status/damage`,
  `sleep_out/in`) — switching value is LEARNABLE from `Φ_mat` + `pbrs_belief` + win/loss, so
  hand-rewarding it is a `provide-vs-learn` violation that biases the objective.

Two flags (not one) so the low-risk redundant removes can be attributed separately from the
behaviorally-uncertain switch family (which may have been doing real exploration-acceleration work).
The historical worst distorter — `finishing_blow` rewarding a self-KO Explosion — is already fixed
(guarded + the `+2.0` literal deleted), so it is not in scope. Tests:
`reward_bias_terms_test.py::TestBiasDrops` + `snapshot_test.py` (resume-immutability + v12→v13 migration).

**End-state PBRS — TWO switches (`--all-shaping-pbrs`, DEFAULT ON; `--stall-pbrs`, default OFF;
v14/v15).** The FINAL stage of the staged PBRS rollout: convert the last BIAS shaping to
policy-invariant telescoping potentials. Deliberately TWO switches so the stall tilt (which carries a
documented regression risk) can be A/B'd separately from everything else — which is also why only the
first of them defaults on.
- **`--all-shaping-pbrs` ("everything but stall")** — (1) **folds** `Φ_hazard` =
  `HAZARD_WEIGHT`·(opp − our spike layers, design §2.6), `Φ_boost` = `BOOST_WEIGHT`·Σmax(0,our-active
  boost)·hp_frac, `Φ_opp_boosts` = −`OPP_BOOST_WEIGHT`·Σmax(0,opp-active boost), **and `Φ_status`**
  (its gate is now `bias_redesign OR all_shaping_pbrs`, so the tempo-status standing value is carried
  even without `--bias-redesign`); (2) **zeros EVERY BIAS term EXCEPT the anti-stall tilt
  `no_progress_tax`** — so `status`, `stall_tax`, `matchup_penalty`, the switch family, the anti-spam
  family, `spikes`/`futile_*`/`boost_utilized`/`roar`, and the redundant good-outcome bonuses
  (`finishing_blow`/`explosion_block`/`status_wasted`) all go. It also **activates the clock charge**
  (gate `bias_redesign OR all_shaping_pbrs`) so `no_progress_tax` is live as the kept tilt.
- **`--stall-pbrs` ("stall")** — **folds `Φ_progress`** = −`no_progress_penalty`·`progress_clock.value()`
  (the anti-stall clock as a telescoping potential) and **zeros `no_progress_tax` + `stall_tax`**, so the
  anti-stall signal is policy-invariant too.

Run **both** ⇒ the WHOLE BIAS class is zero → TERMINAL + PBRS only (fully policy-invariant). Run **only
`--all-shaping-pbrs`** ⇒ everything-else is PBRS but the progress-aware `no_progress_tax` survives as the
single acknowledged BIAS tilt (insurance against stall-regression — watch the stall-rate canary; the
terminal `--draw-penalty` remains the objective anchor either way). The zeroing lives in
`_apply_pbrs_suppression(bd)` (loops `registry_fields(BIAS)`, skipping `no_progress_tax` under
`all_shaping_pbrs`; zeroing the two stall terms under `stall_pbrs`), called **after** all PBRS folds +
the `_last_attack_had_effect` read and **before** `_apply_bias_drops` → `_fold_bias_refund`, so zeroed
terms leave the bias accumulator. Each new fold early-returns unless its switch is set, so with both OFF
the `_prev_phi_*` slots stay None and the four `pbrs_*` fields stay 0.0 — the byte-identical
`--no-all-shaping-pbrs` baseline (pinned by the no-op-equivalence + registry-coverage tests).
Composes with the v13 drops (orthogonal,
run after). `--no-all-shaping-pbrs` is the fallback and restores the additive objective in full.
Resume-immutable + value-checked alongside the **now-recorded `no_progress_penalty`**
(Φ_progress's weight) — `MODEL_CONFIG_VERSION` v14/v15, `check_reward_config`, no `ARCH_SIGNATURE` bump.
Tests: `reward_pbrs_progress_hazard_test.py::{TestProgressPBRS, TestHazardPBRS}`,
`reward_pbrs_boosts_roar_test.py::{TestBoostPBRS, TestOppBoostsPBRS}`,
`reward_end_state_test.py::{TestEndStateDrops, TestAllShapingPbrsNoOpDefault}` + `snapshot_test.py` (resume-immutability + v13→v14 +
v14→v15 migration).

### The reward COMPOSITION — stated at launch, recorded in `metadata.json`

**A launch says what its reward is MADE OF.** `reward_class_composition(config)` (pure, in
`reward_manager.py`) returns the per-class ACTIVE-term census —
`{terminal, pbrs, bias, bias_terms, pbrs_terms}` — where ACTIVE means *"this config does not
structurally force the term to zero"* (it mirrors the `_fold_*_pbrs` early-returns,
`_apply_pbrs_suppression`, `_apply_bias_drops`, `_apply_progress_clock`, and the three weight-gated
terms). `format_reward_composition` renders the one line `train_rl_agent` emits at startup, to
stdout AND the launcher Events panel; the dict is written to `metadata.json` as
`reward_composition`, carried forward across saves like `cli_args`. It is duck-typed on field
names, so a recorded `ModelVersion` can be censused offline without reconstructing its config.

| config | composition |
|---|---|
| **default** | `1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)` |
| `--no-all-shaping-pbrs` | `1 TERMINAL + 2 PBRS + 25 BIAS` |
| `--stall-pbrs` (with the default) | `1 TERMINAL + 8 PBRS + 0 BIAS` — the zero-bias destination |
| **`--no-hand-shaping`** | `1 TERMINAL + 0 PBRS + 0 BIAS` — the CLEAN WORLD (see below) |

#### The CLEAN-WORLD switches (`gen3_clean_world_config_v1`, config v105)

**Four resume-immutable fields, every default equal to today's behaviour**, so a flagless launch is
byte-identical. Spec: probe N
(`designs/research_state/measurements/no_progress_tax_review_2026-08-29.md` §5).

| flag | default | what `false` / the other value does |
|---|---|---|
| `--hand-shaping` | ON | **the master.** All EIGHT `_fold_*_pbrs` early-return AND the whole BIAS class is zeroed, `no_progress_tax` included |
| `--pbrs-material` | ON | drops Φ_mat (the term had NO flag at all before) |
| `--pbrs-belief` | ON | drops the EMITTED Φ_belief term only — see the mutation note below |
| `--victory-value` | 30.0 | the ±terminal, promoted off the `reward_weights.VICTORY_VALUE` module constant |

🚨 **Why a master flag rather than "just turn `--all-shaping-pbrs` off": the two halves are
ANTI-CORRELATED across it.** `all_shaping_pbrs` does two jobs — it folds five potentials *and* it is
`_bias_term_active`'s master gate — so `--no-all-shaping-pbrs` silences the potentials while
**reviving 25 BIAS terms**. "No hand PBRS **and** no BIAS" sat in a hole between the two settings and
no combination of the pre-existing flags could reach it (asserted, in `clean_world_config_test.py`,
so nobody "simplifies" `hand_shaping` away later).

⚠️ **State this honestly in any write-up.** Every PBRS term is **policy-INVARIANT by construction**
(`Φ(terminal)=0`, telescoping), so removing them **cannot change the optimal policy** — it changes
learning dynamics and conceptual complexity. The clean-world claim's real content is "the hand terms
cost more in interference and tuning than they buy in credit assignment", never "the hand terms bias
the objective". The only class that biases the objective is BIAS, and that was flag-zeroable before.

🚨 **THE OUTCOME ORDERING is the clean arm's largest hazard.** `draw_penalty = -35 < -30` exists so
that stalling to the 250-turn cap is strictly worse than losing cleanly. At a ±1 terminal a
`draw_penalty` of `0.0` **inverts** that: the stall becomes the best non-winning outcome, and with
`no_progress_tax`, `stall_tax` and Φ_progress all removed, nothing else opposes it. The owner's
ruling is **draw = loss**. `resolve_config` prints a loud `[Reward] ⚠️ ORDERING` line whenever
`draw_penalty > -victory_value` — a warning, not an error, because an arm may want it — and
`--victory-value <= 0` is refused outright. **Register stall rate + mean game length as a PRIMARY
safety endpoint on this arm, not a secondary one.**

🚨 **THE ORDERING GUARD IS ONE OF THREE, and the other two are about SCALE** (`_terminal_scale_guards`
in `main/train/config.py`, added by the R1 adversarial review). `--victory-value` is the first flag
that can move the RETURN SCALE, and two older flags are quietly denominated in that same scale:

| `[Reward] ⚠️` line | fires when | why it is not just style |
|---|---|---|
| `ORDERING` | `draw_penalty > -victory_value` | a draw beats a loss ⇒ stalling is optimal for a losing agent |
| `TERMINAL SCALE` | `\|draw_penalty\| > 3 × victory_value` | `--victory-value 1.0` alone INHERITS the −35.0 default: a timeout 35× a clean loss, so the composition advertised as "1 TERMINAL" is really a stall-avoidance objective |
| `VALUE-DIST SUPPORT` | a value-dist head, PopArt OFF, and the achievable returns either fall outside `[vmin, vmax]` or span < 8 atoms | with PopArt off the HL-Gauss target is the RAW return, so the atom support and the terminal are in the SAME units — and under `--value-from-dist` that quantized `E[Z]` **is** the critic feeding GAE |

The third is the one that matters for the registered arm, because the clean/sparse arms run
**without PopArt** (ledger `2d38a4a`) while production carries `value_from_dist=True` over a
support of `[−12, +12]` / 51 atoms. A ±1 terminal there lands inside ~4 atoms — a critic quantized
to ~0.5 on a ±1 scale — and nothing downstream tells that apart from a well-fitted one
(`value_dist/mean_abs_err` looks *better* as the support widens). All three are warnings, never
refusals: a launch that works today must not become a `FATAL_CONFIG`. Under PopArt the target is
`popart.normalize(returns)`, the support lives in units of standard deviations, and the guard is
skipped — which is every run ever launched. Pinned by `ai_v12_intersection_test.py` §1-§2.

**The clean-world reward flag set, verbatim** (`CLEAN_WORLD_REWARD_FLAGS` in
`clean_world_config_test.py`; the dense signal is `--win-prob-pbrs-*`, below):

```
--no-hand-shaping --victory-value 1.0 --draw-penalty -1.0
```

Two implementation notes worth keeping:

- **`victory_value` covers the PRE-CAP TIE too.** `finished and not won and not lost and turn < cap`
  shared the decisive-loss branch as a hardcoded `-VICTORY_VALUE`; it now reads the field, so a ±1
  arm cannot score a rare tie at −30 beside a −1 loss. `MAT_HP_WEIGHT` / `MAT_ALIVE_WEIGHT` are
  calibrated against the 30 scale — moot under `--no-hand-shaping`, which is the composition the ±1
  terminal exists for.
- **`--no-pbrs-belief` gates the EMITTED FIELD ONLY.** `_fold_belief_pbrs` also snapshots the
  decision-time KO risk and safe-pivot flag, which the belief-scaled BIAS terms read; the manager's
  standing rule is that a gate skips a COMPUTE, never a cross-turn mutation.
- **The folds and the census are now ONE declaration.** Every `_fold_*_pbrs` calls
  `_hand_pbrs_on(name)` → `_pbrs_term_active`, the same predicate `reward_class_composition` reads.
  They were two hand-maintained copies of the same conditions, which is exactly how a census can
  advertise a composition the folds do not implement.

**Why it exists.** The v8→v9 drift (`designs/research_state/ledger.md`, 2026-08-18):
`--all-shaping-pbrs` simply stopped being passed at the fresh-generation boundary, so every
`ai_v9_*` run through gen-14 trained the 26-term additive objective while every validated `ai_v8_*`
run had trained the near-policy-invariant one. Nothing failed. Reward config is **training-only** —
no `ARCH_SIGNATURE` bump, absent from `check_compatible` — and no launch line stated the
composition, so the change was unobservable for a year. The census is the counter-measure and the
seed of the **launch-diff gate** the ledger registers: the field a new generation's resolved
command is diffed against its reference generation's.

⚠️ **The ledger's prose says "8 PBRS + 1 BIAS" (v8) and "3 PBRS + 28 BIAS" (v9); the census says
7/1 and 2/26.** The census is the measured one and the difference is definitional, not a
disagreement about the regimes: it counts terms a config can actually EMIT, where the hand-count
took the PBRS registry class size (8) — `pbrs_progress` gates on `--stall-pbrs`, which is off in
both regimes — and did not subtract the weight-gated BIAS terms (`stay_risk_tax` /
`escape_risk_bonus` at `switch_bias_weight` 0, `self_ko_penalty` at `self_ko_hp_penalty` 0). The
shape claim the ledger makes — ONE acknowledged bias term vs a couple of dozen additive ones —
holds exactly.

Pins: `src/main/reward_defaults_test.py` (both defaults, both opt-outs, both compositions, the
`RewardConfig` ↔ `ModelVersion` default agreement, and the actionable resume FATAL).

#### The census also drives a fast path — `_active_bias` (`gen3_reward_skip_suppressed_v1`)

`process_turn_reward` used to compute every BIAS helper and then hand the results to
`_apply_pbrs_suppression`, which under the production composition zeroes ~20 of them — a movedex
walk (`dead_matchup_tax`), two effectiveness loops (`se_switch` / pivot) and a 12-mon status scan,
every decision, for numbers immediately overwritten with 0.0. The manager now derives
`self._active_bias` ONCE at `__init__` **from `_bias_term_active`** — the same function the census
above reads, never a second hand-copied name list (the v79 hand-copied-family-set lesson) — and
`_bias_active("<field>")` gates each pure value computation. Each gate names the field it ASSIGNS,
so a rename breaks the assignment beside it instead of silently un-gating a term.

**It is legal because activeness is a per-run CONSTANT**: every flag `_bias_term_active` reads
(`all_shaping_pbrs`, `stall_pbrs`, `bias_redesign`, `drop_*`, `switch_bias_weight`,
`self_ko_hp_penalty`) is resume-immutable and value-checked by `check_reward_config`, so a
constructor-time active set can never go stale mid-run. Where the mirror is imprecise it errs
ACTIVE (it does not model the progress clock's extra zeroing of repetition/struggle/dead_matchup
under `--bias-redesign`), which costs time, never correctness.

**The cut is COMPUTE-only, never a cross-turn MUTATION.** `_update_opp_se_threat`,
`_compute_spikes_bonus` (`_prev_opp_spikes`), `_compute_status_reward` (`_prev_*_statused`),
`_apply_switch_outcome` (`switch_count` / bounce depth / `_last_switched_from`) and the
`_last_opp_seen_by` update all stay **ungated**, so the manager's observable state is identical
turn for turn whether the skip fires or not. The one exception is `_compute_dead_matchup_tax`,
skipped whole despite mutating `_consecutive_dead_matchup_stays`, because that counter has ZERO
readers outside `reward_manager.py` — suppressed, it is write-only, not observable state.

**Measured** (2026-08-23, order-alternated same-process A/B, both arms on the same decision;
absolutes contaminated by a busy box, ratios are the claim): **~1.08× on `process_turn_reward`**
across four ~1500-decision runs, and a load-free **−20.3% Python calls per call**. Under
`--no-all-shaping-pbrs` (nothing suppressed) the ratio is 0.990× — a no-op, as required. Riding
along: `registry_fields` memoized, `total` summing a cached field-NAME tuple instead of
re-deriving `dataclasses.fields()` (which measured 7.4% of the stage — as much as the whole BIAS
family), and the Φ_opp_boosts/Φ_roar Σ (the same potential at two weights) computed once.

**Gates — this is THE OBJECTIVE, so bit-identity, not approximation.**
`reward_skip_parity_fuzz_test.py` plays real bridge battles and compares EVERY breakdown field
(with `!=`) between the production manager and a `_shadow=True` twin, across the three
compositions **on the same decision stream** — which is also what makes its trigger-coverage table
meaningful, since the `--no-all-shaping-pbrs` arm's firings are exactly what the production arm
skipped. It additionally asserts per turn that the skip is the suppression's exact COMPLEMENT, and
fails INCONCLUSIVE if the corpus never fired the required signals. `GEN3AI_REWARD_VERIFY=1`
(`reward_verify.py`) is the shadow mode: a lockstep full-computation twin asserted bit-identical
every turn — no CLI flag, because the skip is an internal swap and a default branch nobody runs is
the untested one. Derivation pins: `reward_skip_parity_test.py`.

**`reward_tracker.py` parity holds BY CONSTRUCTION** — it builds the same `Gen3RewardManager`
through the same factory with the run's `RewardConfig`, and this change adds no constructor input
the tracker path doesn't thread, so eval traces / falsify / `cf_mc_return` inherit it unchanged.

⚠️ **Do NOT "optimize" the Φ potentials by carrying/telescoping Φ** — recompute-from-the-memoized-
view IS the exactness guarantee, and the reasoning lives in `_pbrs_step`'s docstring where someone
would try it. The one expensive Φ input, `pbrs_belief`'s `encode_block` at **60% of the stage**,
got the safe answer instead — a content-keyed memo, next.

#### The belief-block memo — `IncomingBeliefMemo` (`gen3_belief_block_memo_v1`)

Φ_belief's `encode_block` measured **60.0% of `process_turn_reward`** (`compute_team_block` 42.5 /
`_attacker_threat` 10.8) — the largest single item anywhere in the per-decision CPU budget, and
`reward_manager.py` is the tree's **only** per-decision caller of that pipeline. It now answers
from a per-manager, **content-keyed** cache
(`agents/observation/incoming_damage_encoder.IncomingBeliefMemo`), threaded as
`encode_block(live, memo=…)` and cleared at `reset()` (episode scope).

**Two caches, both on content:**
- `attacker_state_key(live)` → the `AttackerThreat`. The key is `(species, types, move_ids, status,
  atk/spa/spe stages, our reflect, our light screen, weather)` — the opponent active keeps all of
  those for runs of turns.
- `(attacker_key, Defender)` → that mon's `PER_MON` row (`inc.compute_mon_row`, factored out of
  `compute_team_block` for exactly this). `Defender` is a frozen dataclass over primitives rebuilt
  fresh from the board every call, so it **is** its own key — no coverage question on that side.

Plus one algebraic identity in the inner loop: the crit branch computes
`gen3_damage_max(..., screen=False, …)`, which with **no screen up** has argument-for-argument the
same inputs as the modal call, so `dmax_crit == 2·dmax` exactly and the second formula evaluation
is skipped on the overwhelmingly common screenless board.

**Why content-keyed and not `_state_epoch`-scoped.** An epoch key is strictly WEAKER here: same
epoch ⇒ same content ⇒ the content key hits anyway, so the epoch adds no hit it does not already
have — while a per-turn scope would DELETE the cross-turn reuse that is the entire win. The reward
path calls `encode_block` exactly ONCE per decision, so a turn-scoped cache has a structural hit
rate of zero. Content-keying also makes the clone/deepcopy question vanish: unlike a cache keyed on
an object identity or a `battle_tag`, a cross-arm hit here is *correct* rather than a hazard.

**And it is NOT the telescoping Φ that `_pbrs_step` refuses.** Nothing accumulates and nothing is
carried; dropping the memo at any moment changes only speed (pinned:
`test_clearing_at_any_point_changes_nothing`). Content, not history.

**Measured** (2026-08-23, the same order-alternated same-process A/B; box at load 31-36, so
absolutes are contaminated and the RATIO is the claim): **~1.25× on `process_turn_reward`** over
five runs of 2000-5500 paired decisions (1.321 / 1.254 / 1.186 / 1.214 / 1.294), plus a load-free
**−24.0% Python calls per call** (484.2 → 368.1, a `sys.setprofile` count — a different instrument
from the cProfile figure above, not comparable to it). Cache hit rate 48-58% of lookups under
random play. Against the ~23-27% reward share that is ~5-6% of worker CPU.

**Gates.** The key's completeness is proven twice, and neither proof is a reading of the code by
eye: `incoming_damage_memo_test.py` **AST-walks `_attacker_threat`** for every attribute reached
from `live` (including through `getattr`) and fails if one appears that the key does not carry —
the "enumerate the doors, not the reads you noticed" discipline of the `live_view` epoch memo,
applied to a pure function — and the same file pins `AttackerThreat`'s field list, since the proof
is a claim about that constructor's arguments. `reward_skip_parity_fuzz_test.py` adds the
**differential** half on real boards: every decision's key is recorded against the freshly-derived
belief, and a key seen twice must carry the identical belief — 3,956 repeat-key tests over 1,003
distinct keys in a 4,959-decision run (verified failing: a deliberately under-keyed build reports
`two boards share attacker key … but differ on ['spa_tail','spa_mean']`).
That same fuzz covers the memo end-to-end **by construction**, because `_shadow=True` means
"compute everything the slow way": the twin runs with the skip disabled AND `_belief_memo = None`,
so `GEN3AI_REWARD_VERIFY=1` is a live per-field test of key completeness in any run. Memo hit
counts are printed and FLOORED — a clean run in which the cache never served anything would be
evidence about the uncached path only.

#### `reset()` clears every `_prev_phi_*`, DERIVED (`gen3_prev_phi_reset_v1`)

`Gen3RewardManager.reset()` cleared its PBRS carry-overs from a hand-written list, and that list
omitted `_prev_phi_roar` for the whole of its life: eight potentials declared in `__init__`, seven
cleared, nothing anywhere to notice. The set is now derived from the instance (`_prev_phi_fields`),
and the pin (`reward_manager_test.py::test_reset_clears_every_prev_phi_potential`) derives it the
same way — a hand list in the test would have passed the whole time too.

**The leak was benign by COINCIDENCE, not by contract**, which is why it is fixed rather than
documented. `_pbrs_step` zeroes Φ at a terminal fold, so a normal episode end leaves
`_prev_phi_roar == 0.0`; the next episode's first window then charges `γ·Φ_roar(s₁) − 0.0` where
the correct fresh start (`prev is None`) charges `0.0`. Equal only when `Φ_roar(s₁) == 0` — and s₁
is the board **after turn 1 resolves**, not the opening board, so an opponent that opens with
Dragon Dance / Calm Mind breaks it. Measured over random-play bridge battles: non-zero at the first
window in **3/185 battles (1.6%)**, a one-off ≈ −0.25 per positive boost stage — rare under random
play, and a trained boost-opener raises it. The second channel the
coincidence never covered at all: a `reset()` that lands MID-battle leaves the last **non-terminal**
Φ_roar, an arbitrary value. (Before `--all-shaping-pbrs` became the default in 2026-08 the fold
never ran at all, so `_prev_phi_roar` stayed `None` and there was nothing to leak.)

## State-conditioned defensive-exploration entropy (`--defensive-entropy-boost`)

`gen3_defensive_entropy_v1` — the answer to "the model under-uses Recover/Soft-Boiled/Wish/Refresh/Heal Bell
when safe" that does **NOT** touch the reward (so it can't create a stall incentive). Instead of biasing toward
healing (which would force you to hand-draw the good-defense-vs-stall line), it **explores** defensive moves more
and lets the *existing* anti-stall reward (the `--draw-penalty` + the no-progress clock) be the guardrail: the
model only KEEPS healing if the returns reward it, and a heal-war that drifts to a 250-turn draw is punished as
before. **The mechanism is ORTHOGONAL to the reward** — it explores the defensive option more but changes
nothing about its value, so if the critic learns healing is net-negative here (no-progress clock / racing
meta), the boost will NOT override that; it only surfaces the option. *Contingent* virtuous loop: IF the model
**discovers** defense is valuable (the returns must reward it), the self-play **opponents** become defensive
too, so the distribution self-enriches toward the patient meta self-play currently lacks.

- **The flag (`gen3_env._defensive_opportunity`).** Per decision, the env emits a training-only
  `defensive_opportunity` Dict-obs key = 1.0 when the trainee's ACTIVE mon has a *productive* defensive option:
  a legal `is_heal` move with HP below `_DEFENSIVE_HEAL_HP`=0.85, OR a legal self-cure (Refresh) while statused,
  OR a legal team-cure (Heal Bell/Aromatherapy) while any party member is statused; else 0.0 (forced switch →
  no moves → 0). Never raises (hot path). Read ONLY by the entropy term — never enters the pi/vf forward.
- **The boost (`instrumented_ppo`).** The per-decision entropy bonus is multiplied by `defensive_entropy_boost`
  on flagged decisions: `entropy_loss = -mean((1 + (B_eff−1)·flag)·entropy)`. `B=1.0` = OFF (byte-identical;
  also identical on any minibatch with no flagged decisions). `B_eff` anneals B→1 linearly over
  `--defensive-entropy-anneal-frac` of training (`_defensive_entropy_boost_eff`, 0 = constant) so exploration
  fades as the policy learns. The standard `train/entropy_loss` metric stays UNWEIGHTED; new `defent/*` metrics
  (`flagged_frac`, `boost_eff`, `entropy_flagged` vs `entropy_unflagged`) confirm the boost fired and raised
  entropy where intended.
- **Threading.** `--defensive-entropy-boost` (default 1.0) + `--defensive-entropy-anneal-frac` (default 0.0);
  the env emit is gated on `boost > 1.0`; the coefs are set on the model like `ent_coef` — **training-only, NOT
  version-locked, settable on resume** (no `model_config`/`ARCH` change). Try `--defensive-entropy-boost 3.0`.
  **Caveat (be honest):** the model already *samples* heals ~24% in safe spots, so exploration helps mainly at
  rare policy-collapse states (low HP + safe + revenge-killer coming) and can't manufacture a "heal→win" signal
  self-play lacks — it's complementary to, not a substitute for, a teacher/league. Watch the stall-rate canary.
  Tests: `defensive_entropy_test.py`.

## State-conditioned BAIT-exploration entropy (`--bait-entropy-boost`)

`gen3_bait_entropy_v1` — the same mechanism as the defensive boost above, on a different flag, and it
exists to answer ONE question. The bait verdict (`designs/research_state/ledger.md` → *E4 VERDICT*,
2026-08-23) closed the hunt with a stated mechanism: **exploration starvation at a saturated action** —
the whiff sits at p≈0.97, so the alternatives at p≈0.01-0.03 are never sampled and their advantage is
never realized. Everything upstream of the action was cleared: α/β know the switch, the critic already
ranks an alternative above the whiff in 21/23 loop decisions, and the E4 substrate arm moved the cells
and changed **11 decisions in 780**. What was never tested is the mechanism's own claim — that the
policy would fix this if it merely SAMPLED the alternatives. This flag is that test, and it is the
cheapest instrument that can separate the two remaining stories.

- **The flag (`gen3_env._bait_opportunity`).** Per decision, the env emits a training-only
  `bait_opportunity` Dict-obs key = 1.0 when the attack we would most likely click (`last_move` if it is
  still legal and damaging — the RE-CLICK — else the highest-base-power legal attack) deals **ZERO**
  damage to an **alive, revealed opponent BENCH** mon. Bench, not active, because in gen 3 the switch
  resolves first: the decision that whiffs is taken while the immune mon is still benched, which is also
  what makes the flag line up with the offline detector's whiff states. The zero-damage predicate is
  `baitbot.blocks` → `gen3_mechanics.effective_multiplier` → `data/` — ONE predicate shared with the
  scripted BaitBot opponent, so the flag fires on exactly the boards BaitBot exploits and no immunity
  table is hand-copied. Never raises (hot path); read ONLY by the entropy term, never in the pi/vf forward.
- **Three scope decisions, all deliberate.** (1) **REVEALED bench only** — using agent2's true team was
  available (the key is privileged) and refused: boosting entropy on a distinction the policy cannot make
  adds sampling noise with no learnable signal, and gen-15 settled that perception is not the gap.
  (2) **Ability immunities count once revealed** (Levitate/Water Absorb/Volt Absorb/Flash Fire), the same
  information the policy holds; type immunity always counts. (3) **The α half of the proposed predicate is
  NOT shipped** — α is published by the extractor inside the LEARNER's forward, and the flag is built in
  the env worker *before* any forward exists (the eval-time capture reads it off an in-process `RLPlayer`,
  a seam training does not have). There is nothing to emit as a second key; an α-gated variant would have
  to live at loss time and gate on the live policy's own moving α, which is a worse instrument for a probe.
- **The boost (`instrumented_ppo`).** Identical arithmetic to the defensive boost, on the same annealing
  schedule (`_annealed_entropy_boost`, shared so the two cannot drift): `entropy_loss = -mean(w·entropy)`
  with `w = (1 + (B_def−1)·flag_def)·(1 + (B_bait−1)·flag_bait)`. **Overlap semantics: multiplicative.**
  Each factor is exactly 1 off its own flag, so either boost alone is byte-identical to running it alone,
  and a decision flagged by both gets the product (they are near-disjoint in practice — "a heal is legal"
  vs "our attack is dead into their bench"). `B=1.0` = OFF, byte-identical *including on a fully populated
  flag column*. `train/entropy_loss` stays UNWEIGHTED; `baitent/{flagged_frac, boost_eff, entropy_flagged,
  entropy_unflagged}` say whether the boost fired and where.
- **Threading.** `--bait-entropy-boost` (default 1.0) + `--bait-entropy-anneal-frac` (default 0.0); the env
  emit is gated on `boost > 1.0`; the coefs are set on the model like `ent_coef` — **training-only, NOT
  version-locked, settable on resume** (no `model_config`/`ARCH` change, nothing in `flag_registry` — it
  reaches no extractor).

**The pre-registered readings** (write them down before the run, per the hunt doc's own rule):

| observation | reading |
|---|---|
| whiff / re-click rate falls under the boost and **STAYS** down past the anneal | **SAMPLING was the block.** The mechanism the verdict named is right, the correction is realizable on-policy, and the cheap lever generalizes to other saturated actions. |
| falls under the boost and **REVERTS** as `boost_eff → 1` | **CREDIT is convicted.** The alternatives were sampled, their advantage was estimated, and the policy still went back — so the correction has to arrive off-policy: R1/R2's counterfactual labels and the search-teacher/OPD inherit, exactly as the E4 entry's closing paragraph predicted. |
| never falls, at a healthy `baitent/flagged_frac` | neither — the boost did not move behaviour at all; read `entropy_flagged` vs `entropy_unflagged` first to confirm the boost actually reached the policy. |
| never falls, at a near-zero `baitent/flagged_frac` | a **DOSE** finding, not a mechanism finding: the states were not in the rollout. Raise exposure (a BaitBot-shaped opponent in the pool) before concluding anything. |

⚠️ `flagged_frac` is the exposure reading and must be quoted with any verdict — the E4 entry's own
ecology finding (BaitBot-shaped opponents propagate baiting through self-play, pivots 574→773) is why a
dose number is not optional here. **MEASURED at build time** (`--debug --steps 10000
--bait-entropy-boost 3.0`, 2026-08-23, default bot roster, early policy): `flagged_frac` **0.005-0.016**
— i.e. ~1% of decisions are bait boards at the default opponent mix, with `entropy_flagged` 1.74-1.88 vs
`entropy_unflagged` 1.65-1.69. That is the dose a probe arm inherits unless it deliberately raises
exposure, and it is small enough that a BaitBot-weighted pool is worth considering in the same launch.

Tests: `bait_entropy_test.py` (predicate units incl. the four ability immunities with negative controls,
the anneal, and the loss on the REAL `train()` path — OFF byte-identical with every row flagged, the boost
inert on unflagged rows, and the exact identity `(ent_coef=c, boost=B) ≡ (ent_coef=B·c, boost=1)`) +
`bait_opportunity_integration_test.py` (`sim`: the emission path through a real bridge battle, and the flag
CROSS-CHECKED against `main.prober.loops` — every detector `immune` whiff whose arrival was already
revealed must have been a flagged decision; **23 immune whiffs / 21 cross-checked / 0 disagreements** on a
PINNED matchup. The teams are pinned deliberately: drawn from the pool, only 2 of 14 sample-team pairs
produced any immune whiff at all and the cross-checked count ranged 0-48 run to run — a test whose sample
size is a random variable cannot carry a floor. The reverse direction is deliberately not asserted — an
opportunity predicate fires before the mistake, so it fires on states that never become whiffs).

## MatchupSpec — the declared matchup (`matchup_spec.py`)

**The ONE explicit declaration of what a run's battles look like** (design:
`designs/ai_v8/design_matchup_config.md`, P0 built). One week produced four independent failures with a
shared root — *the matchup a run plays is assembled implicitly across seams that nothing forces to
agree*: the eval worker rebuilt its own default teams (specialists measured OOD), the env's single
`team=` fed BOTH sides (the training mirror), training/eval play modes drifted (stochastic
noise-farming), and the launcher's exit summary resolved "Last model" to a global-glob golden. The spec
makes the matchup EXPLICIT: built ONCE in `train_rl_agent` (`MatchupSpec.from_args(args)`), then
CONSUMED — never re-derived — by the consumers (the `plan.json` pattern).

- **`TeamSource`** — where one side's teams come from; its `build(all_teams, sample_teams)` is the ONLY
  constructor of that side's `Gen3Teambuilder` (the env factory no longer assembles builders inline).
  Kinds: `pool` (opponent default), `default_biased` (trainee default — full pool + 10% sample-team
  bias, `DEFAULT_TRAINEE_BIAS_PROB`), `pinned` (`--trainee-team`), `pin_multi` (`--trainee-teams` — a
  SMALL FIXED SET sampled uniformly, the z-near multi-team exploiter / 1-vs-3-team A/B; `pin_str`
  mirrors `pin_strs[0]` so single-team consumers keep working, and unlike a single pin z_arch VARIES
  across the set), `pin_biased` (the future
  `--trainee-team-prob` shape — supported, no CLI yet). Each is byte-parity with the legacy
  construction (pinned by `matchup_spec_test.py`). **The two sides are independent BY CONSTRUCTION**
  (`trainee_teams` / `opponent_teams` → `Gen3Env(team=, opponent_team=)`) — the mirror-bug class is
  structurally closed.
- **`PlayMode`** — how the frozen-NN opponents select actions (greedy | stochastic@temp, schedule
  fixed | anneal | ratchet). Descriptive in P0 — the executors (RLPlayer temp, the anneal/ratchet
  callbacks) already exist; the spec records the intent so echo/provenance say what a metric was
  measured under. `eval_opponent_play` defaults greedy; `eval_trainee_teams` defaults to
  `trainee_teams` (**the eval-OOD fix made structural**: eval pilots what training pilots).
- **Provenance** — `to_dict()` (pin fingerprints via sha1, not full text) + `spec_hash()` (a 10-hex
  **measurement-regime tag**: two runs/eras with different hashes are NOT metric-comparable) are
  stamped into `metadata.json` beside `cli_args` (`_matchup_spec` / `_matchup_spec_hash`).
- **Startup echo** — `summary_lines()` emits a `🧭 [MATCHUP <hash>]` block to the launcher Events
  panel: trainee teams, opponent teams + mix, exploiter target + play mode, eval regime — one glance
  at what the run actually plays.
- **The realized-matchup fuzz** (`poke_env_gaps/matchup_realized_fuzz_test.py`, bridge, no server) is
  the permanent mirror-catcher: it drives the REAL construction path (spec → builders →
  `Gen3Env(team=, opponent_team=)` → bridge) over real battles and asserts per episode that the
  trainee fields EXACTLY the declared pin, the opponent does NOT (the mirror signature), and opponent
  rosters VARY across episodes. P1+ (not built): controllers keyed on eval play modes, per-row regime
  tags, per-opponent team pools.

### Matchup provenance (what a run trained/evaled against — the diligence layer)

Four self-describing records, all metadata-only + additive (old readers unaffected), closing the
"a row/trace/checkpoint can't say what regime produced it" gap the OOD-eval era exposed:

- **`eval_results.jsonl` rows carry `matchup_hash` + `externals`** (`append_eval_result_row`):
  each append-only ladder row is stamped with the run's CURRENT declared-matchup hash (rows from
  different regimes/eras are distinguishable IN-FILE, not by dates), and the per-cycle vs-target
  record (`{ext label: {win_rate, counts}}` — e.g. the exploiter VERDICT) now survives in the
  jsonl instead of only the overwritten `latest_eval` + TensorBoard. Externals stay OUT of `bots`
  (the ELO fit's ladder is untouched).
- **`metadata.json:matchup_history`** (append-only, maintained by `save_model_snapshot` from the
  `cli_args` stamp): one `{hash, spec, recorded_at}` entry per ERA — a resume that changes the
  declared matchup appends a new era instead of silently overwriting the old one (cli_args keeps
  only the latest). Saves without cli_args (the periodic-checkpoint path) preserve it.
- **The resume MATCHUP-DRIFT guard** (`train_rl_agent`, warn-not-fatal): a `--model` resume whose
  declared matchup hash ≠ the run's recorded one emits a loud `⚠️ [MATCHUP DRIFT]` + the
  field-level diff (`matchup_spec.describe_drift`) — a mid-run curriculum change is legitimate,
  doing it SILENTLY is not. Launcher restarts forward flags verbatim → never fire it.
- **`eval_manifest.json` records the eval REGIME**: `matchup_hash`, `trainee_team_sha` (the pin
  the trainee piloted; None = pool), `opponent_pins` ({ext label: sha} for fold-back-pinned
  opponents) — a trace dir is self-describing about HOW its numbers were measured.
- **Checkpoint sidecars + `snapshot_history` entries carry `matchup_hash`** (via
  `record_checkpoint` → `_build_snapshot_entry`, like the `latest_eval` stamp) — each checkpoint
  is self-describing about what it was training against as of its save, robust to later eras.

Readers: `snapshot._read_matchup_hash(model_dir)` (current era) /
`snapshot.read_recorded_matchup(model_path)` (the drift guard's input). Tests:
`snapshot_test.py::test_matchup_*`/`test_eval_row_*`/`test_checkpoint_sidecar_*`,
`matchup_spec_test.py::test_describe_drift_*`, `eval_callback_test.py::test_eval_manifest_records_the_regime`.

## Faint attribution in the trace (`gen3_faint_attribution_v1`)

`BattleRecorder` writes one `<side>:<species>:fainted` event per faint. It detected the faint by
COUNT (`*_fainted_count` went up) and then labelled it with **`prev_ctx.*_active`** — the mon that
was active when the DECISION was made. That is the wrong mon whenever a switch resolved on the same
turn, and the trace then contradicted its own battle log two lines above:

```
we switch cloyster → jolteon
opp explosion → jolteon (now 0%)
we cloyster fainted            ← the protocol says JOLTEON fainted
```

**Measured on `ai_v9_17_tdaux_lam3_0818`: 25 of 466 turns named a mon that had not fainted.** Two
shapes produce it — WE switch and the switch-IN eats the hit; or the OPPONENT switches a mon in and
it dies the same turn (Claydol → Dugtrio, our Ice Beam KOs Dugtrio).

**The fix reads the newly-fainted species as a SET DIFFERENCE** over the two snapshots'
`*_fainted_species` — which `BattleContext` already carried, so no new state was needed. A set
difference rather than an HP transition because the second shape has no previous HP to fall from:
Dugtrio was never revealed before the turn it died on.

Two things followed from it, both of which the fuzz found rather than the design:

- **The HP-delta slot was wrong in the same way.** `our_ref` picked `prev_ctx.our_active` on a faint
  turn, so a switch-in that died had its damage read off the row of the mon that left (the recorded
  `hp_delta` read `+0%` while the switch-in went 273 → 0). It now uses the actually-fainted species.
- **ONE SIDE CAN LOSE TWO MONS IN A TURN.** An opponent mon is KO'd, its forced replacement switches
  in and dies to Spikes — both inside turn 34. The old `if delta.*_fainted:` shape could emit at
  most one event per side, so the second faint was silently unreported (1 of 36 faints in a
  4-battle fuzz). `_newly_fainted` returns a LIST and the caller emits one event per species.

**Blast radius: forensic only.** These event strings are read by the prober (the battle-log
timeline, `summary_flags`' `faint` flag) — the reward, the obs and the TurnDelta all compute faints
from their own state, so nothing in training consumed the wrong label. That is also why the fallback
is a slightly-wrong label rather than a raise: a forensic recorder must never take down a run.

**Gate: `poke_env_gaps/faint_attribution_fuzz_test.py`** (bridge, no server) — real battles with a
real `BattleRecorder`, validated against the **protocol log** (`|faint|pNa: Species`), which is the
sim's own statement and not another of our derived structures. It asserts species, side and
completeness per turn, and REPORTS its trigger coverage (`switch-in deaths`) so a clean run that
never exercised the bug says so instead of passing quietly. Measured: **123 faints / 50 switch-in
deaths / 0 mis-attributions**, and **44 mis-attributions when the fix is reverted**.

⚠️ **A protocol identifier carries the NICKNAME, not the species.** The team pool contains teams
whose nicknames are LOCALIZED species names (`Triopikeur` = Dugtrio, `Airmure` = Skarmory), which
reported 10 false failures until the harness resolved identifiers through poke-env's own
`battle.team` map. Any future protocol-vs-our-data comparison needs that map.

## Bot evaluation (subprocess, non-blocking)

**Flat schedule, full roster.** Eval fires every `EVAL_FREQ_STEPS` (2M steps) and plays
`EVAL_GAMES` (100) games per opponent — overridable per run with `--eval-games N` (threaded to both
callbacks via the `_schedule()` seam; n=100 → ±0.098 per-cell 95% CI, n=200 → ±0.069; the recorded
`n_games` tracks the actual cycle size) — one cadence, one game count, applied uniformly to
every bot *and* every self-play sentinel (no maturity tiers, no per-opponent caps). The
roster is the full set of eight archetype bots — both the v1 and v2 of each
(`heuristic`/`heuristic2`, `staller`/`staller_v2`, `aggressive`/`aggressive_v2`,
`setup_sweep`/`setup_sweep_v2`) — plus `random` as the eval-only "is-the-model-broken"
floor (excluded from `win_rate_vs_bots`). All nine are the single source of truth in
`_EVAL_OPPONENT_SPECS` / `eval_opponent_names()`, shared by the bot path, the self-play
path, and the worker. There is no roster flag — every bot always plays, because they play
differently and the playstyle diversity is the point. The flat numbers are safe precisely
because eval is non-blocking and **skips a cycle while the previous one is still running**
(below): a heavier roster self-throttles to a sparser cadence instead of needing tuned
ceilings.

### ⚠️ GLOBAL-RANDOM COUPLING — the five seeds a paired-arm design must set

A drawer that reaches into a **process-wide** RNG couples itself to every other drawer in the
process. Two players interleave their `choose_move` calls inside one battle; two paired **arms**
interleave them *differently* (the searched arm awaits an executor, the control runs inline). So a
decision that consumed the shared stream lands differently in the two arms **with no treatment
involved**, which is precisely what a paired design claims cannot happen.

**It is measured, not hypothetical**, and it was found by a FAILED INTEGRITY CHECK rather than by
review. The transfer-coefficient cell
(`designs/research_state/measurements/transfer_coefficient_cell_2026-08-29.md` §4) ran a paired-arm
falsifier whose zero-overrule units MUST be the same battle in both arms. It passed **exactly** on
the deterministic bots (2,693 pairs, A−B = 0.0000, **zero** divergences) and failed on **exactly the
two stallers** (755 pairs, 4 divergences), whose Protect coin (`_PROTECT_PROBABILITY = 0.6`) came
off the global module. Unbiased noise (3 favoured A, 1 favoured B), but it widens every paired
interval for free.

⚠️ **"The two stallers are the roster's only source of randomness" was WRONG, and this doc said it.**
The follow-up census (`designs/research_state/measurements/global_random_sweep_2026-08-30.md`) found
**four more** and the stallers were the *smallest*. The falsifier above could not have caught the
biggest one: it conditions on zero-overrule units, and the overrule rate against `random` is 1.00,
so that bot contributed **no units at all** — a subject a falsifier gets zero units from has not
been exonerated by it.

| what draws | when | seed kwarg | env hook |
|---|---|---|---|
| **every player's** `choose_random_move` + `DEFAULT_CHOICE_CHANCE` (`poke_env.Player`) | per decision | `rng_seed=` | **`$GEN3AI_PLAYER_SEED`** |
| the **team draw** (`Gen3Teambuilder`) | per battle | `rng_seed=` | **`$GEN3AI_TEAM_SEED`** |
| the **policy's action sample** (`RLPlayer`, torch's default generator) | per decision, when `stochastic` | `policy_seed=` | **`$GEN3AI_POLICY_SEED`** |
| the **self-play pool draw** (`SnapshotPool.sample`) | per episode | `rng_seed=` | **`$GEN3AI_POOL_SEED`** |
| the two stallers' **Protect coin** (`agents/opponents.py`) | conditional | `protect_seed=` | **`$GEN3AI_STALLER_SEED`** |

The first is the widest: `choose_random_move` is `RandomPlayer`'s *entire policy*, the fallback of
all sixteen scripted bots, and `DEFAULT_CHOICE_CHANCE` fires inside the RL players too — so even an
all-deterministic-bot roster has a shared-stream consumer in it. The third is the one a
`random`-only grep never finds: torch has its own process-wide generator, and `stochastic=True` is
the **default** for the pool and stable cross-run opponents.

**Every fix is OPT-IN and every default is unchanged, byte-for-byte.** With no seed by either route
the RNG *is* the `random` module (or torch's default generator), so the call site makes the same
call on the same stream in the same order; an unseeded instance does not even carry the attribute.
An unparseable env seed **raises** rather than falling back — a seed that was meant to be set and
silently was not would make an arm look reproducible while it is not.

**Any paired-arm design over battles should set all five:**

```bash
GEN3AI_PLAYER_SEED=1 GEN3AI_TEAM_SEED=2 GEN3AI_POLICY_SEED=3 GEN3AI_POOL_SEED=4 \
GEN3AI_STALLER_SEED=5   <harness>
```

Measured stake — 2 real bridge battles per arm under the **same fixed sim seed**, arm B burning
1234 unrelated global draws first: **unseeded the arms played different games** (84/145 turns vs
212/233, different winners); **seeded they were identical battle for battle.** A fixed sim seed
bought nothing on its own.

Caveat: a flat seed makes two instances draw the same *sequence* — reproducibility, not
independence (their decisions still differ, because their legal-order lists do). Pass distinct
`rng_seed=` values where the two sides must be independent as well.

Tests: `global_random_coupling_test.py` (47, all four new seams) and
`opponents_test.py::TestStallerProtectRng`. Each seam carries a **revert arm** — unseeded, the same
interleaving pulls the two apart — so if that ever passes, the per-instance RNG has stopped being
the difference and the rest of the suite is asserting nothing.

`PerOpponentEvalCallback` (non-self-play path) does **not** eval in-process. On each
scheduled step it snapshots the live weights (`model.save`) and spawns `--eval-workers`
(default 3) `main.eval_worker` subprocesses that **work-steal at battle granularity** from a
shared pool, load the **frozen** snapshot, and play against the shared Showdown server (or the
in-process bridge) **without pausing training**. **The trainee's eval teambuilder follows the
run's `--trainee-team` pin** (`trainee_team_str` in the worker cfg → `eval_worker._build_trainee_tb`;
threaded by BOTH callbacks): a specialist run is measured piloting ITS OWN team. The worker used to
hardcode the default full-pool builder, so every specialist eval (win rates / ELO / `vs_ext`
verdicts) measured the model piloting random teams it never trained on — pure OOD; the
"ai_v7_05–08 plateau" was this instrumentation gap, not the training (see `eval_worker_test.py`,
the fix's pin). No pin → the default pool builder, byte-identical. **The companion TRAINING-side bug
(the mirror):** PokeEnv feeds its single `team=` kwarg to BOTH internal env agents, and the
per-episode opponent Players are decision-functions over `battle2` (agent2 does the networking), so
agent2's `_team` decides the opponent's REAL team — a `--trainee-team` pin therefore also pinned the
OPPONENTS, turning every specialist run's training into a single-team MIRROR vs bot pilots
(genuinely-won ~100% training WRs, fake curriculum; a probe on the exact path measured the same
checkpoint at 1.000 mirror vs 0.483 with real opponent teams). Fixed by the `Gen3Env(opponent_team=…)`
post-init seam (the `_battle_class` injection pattern), threaded unconditionally from the env factory
(`opponent_teambuilder`); `None` = the pre-fix both-sides behavior. Pinned by `gen3_env_test.py`. Each opponent's `EVAL_GAMES` are split into
**shard units** of `--eval-shard-games` (default 25 → 4 shards/opponent); a worker claims units
(atomic `O_EXCL` lock per `unit_id`), plays them, and publishes one `shard__<unit_id>.json` of
**raw** counts; the parent pools an opponent's shards back into one **exact** result. This is the
long-tail fix — when fewer opponents remain than workers, the straggler's remaining games spread
across idle workers instead of one worker grinding a whole opponent alone (workers are capped by
unit count, not opponent count). The whole mechanism lives in the **`eval_sharding/` package**
(below); when all workers finish the parent merges → TensorBoard + TUI + best-model (the winning
snapshot is promoted by copy, not re-saved). Forensic traces land under
`<run_dir>/eval_traces/step_<N>/<opponent>/` as a per-captured-battle triple (`write_battle_record`,
`battle_recorder.py`): `<outcome>_s<shard>_NNN_summary.json` (the human-readable per-decision dump —
each invocation also carries a **`belief`** block, the model's top-`BELIEF_TOPK` (3) most-likely species
per still-HIDDEN opp slot, present ONLY when the hidden-opponent belief is on and a slot is un-revealed;
`RLPlayer._decode_belief` → `inference/belief_decode`, see `src/agents/model/CLAUDE.md` — and an
**`opp_intent`** block, the v67 `α`/`β` heads' read of what the OPPONENT was about to do: `α` a ranked
list of NAMED believed moves plus `SWITCH`, `β` the candidate switch-ins each named by the model's own
species posterior. Present only under `--opp-intent-coef>0`, so an intent-off run's trace is unchanged;
it is what the prober's `EXPECT` line and the web replay's per-turn *expect* line read) +
`<outcome>_s<shard>_NNN_states.npz` (raw obs/logits/values **+ the chosen `actions`** for the prober
and offline obs replay) +
**`<outcome>_s<shard>_NNN_replay.html`** — a self-contained, **browser-watchable** Showdown replay of
that battle (poke-env `save_replay` over the accumulated protocol stream). The first two are
prober-only; the HTML lets a human just open the game in a browser (no checkout, no prober) — the
only watchable replay for *non-stall* eval battles (stall games still get their own `stalls/*.html`).
The `s<shard>_` prefix namespaces the files so concurrent shards of one opponent never collide.
The filename stem is built by the single helper **`trace_filename_stem(outcome, trace_tag, idx)`**
(`<outcome>_<trace_tag><idx:03d>`) — the **one source of the naming contract** the prober's
`discovery._FNAME_RE` must invert. (When sharding added the `s<shard>_` infix, the prober's regex
didn't follow → every sharded trace parsed as outcome `"?"` and the **whole prober went blind**;
`eval_callback_test.test_trace_naming_contract` now pins that `discovery` parses exactly what
`trace_filename_stem` emits, so the producer↔consumer pair can't silently drift again.)
On a BRIDGE run each trace also gets a fourth sibling,
`…_NNN_reconstruction.json` — the battle's **full-information reconstruction record** (resolved
PRNG seed + both packed teams + the raw command log), captured at the bridge layer and joined to
the trace by battle tag (`utils/bridge/reconstruction.py`). It makes the battle fully replayable
and turn-re-rollable offline (`replay_battle` / `reroll_turn`), and
`agents.training.obs_materializer` can rebuild the trainee's one-sided obs from it bit-for-bit
(guarded by `obs_roundtrip_fuzz_test.py`). It is referee-view data in a **separate artifact** on
purpose — nothing in the obs/training path reads it (the one-sided/omniscient wall; see the bridge
README). Websocket eval simply doesn't produce it (degrades gracefully). 🚨 **THE CAPTURE QUOTA IS OUTCOME-CONDITIONAL, and the manifest now RECORDS it**
(`gen3_trace_selection_manifest_v1`): `EvalRLPlayer` persists at most `_FORENSIC_WIN_QUOTA` (5) wins
and `_FORENSIC_LOSS_QUOTA` (10) losses per opponent per cycle (scaled per shard unit), so the traces
are a LOSS-ENRICHED sample by design — and every consumer that averages over them (`calibration`,
`falsify_scan`, `main.scaffolding_gauge`) used to inherit that skew with nothing on disk saying so.
`record_eval_selection` patches each cycle's manifest at COLLECT with, per opponent,
`battles_played` / `battles_won` / `traces_written` / `traces_won` plus the derived
`capture_rate_win` / `capture_rate_loss` and the rule in words; the counts ride the existing shard
plumbing (`ShardResult.traces_{written,won}`, defaulted so a legacy shard still deserializes).
**Absent reads as SELECTION UNKNOWN, never as uniform** — a legacy tree, or a cycle that crashed
before collecting (the block is written `null` at launch). One declaration,
`agents/training/trace_selection.py` (pure stdlib, imported by the prober and the gauge too), so
producer and consumers cannot drift. All
three sit alongside a per-cycle
**`eval_manifest.json`** (`write_eval_manifest`) recording exactly which model produced them
— `num_timesteps`, `git_hash` + `arch_signature` (read from the run's `metadata.json` /
`model_config.json`), and a `snapshot` pointer. The eval snapshot is normally ephemeral
(`model.save` → workers load → deleted in `_cleanup`) and the eval `step` rarely lines up with
a persisted `<run>/checkpoints/checkpoint_<N>_steps.zip`, so the prober can't reload the *exact* weights unless
they're retained: `--keep-eval-snapshots N` copies the snapshot into
`eval_traces/step_<N>/snapshot.zip` (keeping the N most-recent) and points the manifest at it.
The prober consumes the manifest to load the exact model, falling back to the nearest
checkpoint. **The trainer grooms the traces it writes**: after each cycle
`_prune_eval_traces` keeps only the `--keep-eval-trace-steps` (default 20) most-recent eval
step dirs, and `_prune_eval_snapshots` keeps the `--keep-eval-snapshots` (default 10)
most-recent snapshots — so `eval_traces/` stays bounded without any external task
(`python -m main.prober.groom` is the manual fallback). **The same cycle also bounds the run's
two append-only debug dirs** via `_prune_run_artifacts` (`artifact_retention.py`, a dedicated
module — not bolted onto this busy callback): keep the `--keep-stalls` (default 50) most-recent
`stalls/stall_*.html` replays and the `--keep-crashes` (default 10) most-recent
`crashes/restart_err_*.txt` launcher diagnostics, newest-by-mtime, `0` = keep all. Same
producer-grooms-its-own-data contract; `python -m agents.training.artifact_retention <run_dir |
models_dir> [--apply]` is the manual fallback (dry-run by default; sweeps every run under a
`models/` tree). The eval summary itself is
written to `metadata.json` as a **top-level `latest_eval`** block (step-labeled, NOT
nested under a checkpoint) — robust to the async timing (an eval can finish after a
newer checkpoint, or before any checkpoint exists); `save_model_snapshot` carries it
forward so a later checkpoint never erases it. That top-level block is the canonical,
timing-robust record; **additionally, `record_checkpoint` stamps a point-in-time copy
of the then-current `latest_eval` into each checkpoint's entry** (both the per-checkpoint
sidecar `.json` and the run-level `snapshot_history` entry, under a `latest_eval` key) so
each checkpoint carries the most-recent eval+pool stats as of when it was saved. The
embedded block keeps its own `step`, so storing it under a possibly-newer checkpoint never
mislabels which weights were measured (`snapshot._read_latest_eval` reads it; the union
builder `_build_snapshot_entry` keeps sidecar + history in lockstep).

The frozen snapshot makes parallel eval correct (a worker can't read mutating in-memory
weights), and the fresh process returns all eval memory to the OS on exit (no fragmentation
in the trainer). Behaviors:
- A trigger that fires while the previous cycle still runs is **skipped** (logged) — on CPU
  an eval can outlast its interval; cadence just goes sparser.
- A worker crash is **logged-and-continued**, never fatal (its opponents are just missing
  for that cycle).
- **The hung-cycle watchdog is CONTENTION-SCALED** (`eval_cycle_timeout()` =
  `scale_timeout(_EVAL_CYCLE_TIMEOUT_SEC)`, 30 min baseline, shared by BOTH callbacks;
  `gen3_contention_robust_timeouts_v1`). Eval is the path most exposed to load — it runs
  concurrently with training *by design*, so it is contended 100% of the time, and the bullet
  above already concedes "an eval can outlast its interval". Firing early does **not** merely
  lose a cycle: `_abort_pending_cycle` kills the workers and collects **PARTIAL** results, which
  feed `win_rate_vs_bots` (the curriculum ramp), `win_rate_vs_pool` (the promotion gate) and the
  ELO fit — and the survivors are whichever shards got scheduled, not a random subsample. So a
  merely-slow cycle must never be mistaken for a hung one. The partial-coverage warning no longer
  asserts "worker crash mid-opponent" as the cause either (an overrun-kill produces an identical
  shortfall); it states the fact and appends `describe_contention()` so the reader can tell which
  happened. ⚠️ **Tests must read `eval_cycle_timeout()`, never the raw constant** — the two
  hung-cycle tests built a past timestamp from `_EVAL_CYCLE_TIMEOUT_SEC` and so passed on an idle
  box while failing on a loaded one; `GEN3AI_TIMEOUT_SCALE=6 pytest src/ -m "not integration and
  not e2e"` is the check that catches that class.
- **An operator can force an off-cadence eval** from the launcher's `f` button (confirm →
  SIGUSR2). The signal handler (`train_rl_agent._setup_signal_handlers`) only flags a
  process-global `request_forced_eval()`; whichever eval callback is active CONSUMES it on its
  next `_on_step` (the shared `eval_callback._ForcedEvalMixin._maybe_force_eval`, mixed into BOTH
  callbacks so the path can't drift) and launches a cycle immediately. A request that arrives
  while a cycle is already in flight is **rejected** and reported to the launcher Events panel —
  the same skip-while-running rule as the normal cadence. The forced launch consumes the current
  cadence bucket (`_last_eval_step = num_timesteps`) so the schedule check can't double-launch the
  same step; the next boundary still fires normally. Tests: `eval_callback_test.py` /
  `selfplay_callback_test.py` (`test_force_eval_*`).
- **Graceful shutdown waits for eval to finish**: a scheduled restart is self-initiated by
  `GracefulRestartCallback` at a rollout boundary and the launcher won't force-kill until the
  child overruns the deadline by `--restart-grace-minutes` (20 min), so the drain budget is a
  full `_ABORT_EVAL_DRAIN_SEC` (10 min) AFTER the checkpoint is saved — long enough for a CPU
  eval to complete. Even the pathological forced-SIGTERM case (already overran → ~90s SIGKILL)
  is safe: the checkpoint is saved first, only the in-flight eval can be lost.
- **On resume the last eval is re-published to the TUI** from the resumed checkpoint's
  `metadata.json` (`replay_last_eval_to_tui`), so the eval panel isn't blank until the next
  cycle. This covers the **self-play `pool` block too** — the aggregate (`win_rate_vs_pool`,
  `mean_reward_vs_pool`, monotonicity, snapshot count) and every per-sentinel row are
  re-published from the saved block, with the saved step tags, so Pool/sentinel rows survive
  a restart exactly like the bot rows (no waiting a full cadence for fresh numbers). Safe
  because the pool only changes at an eval-collect — the same moment the block is persisted —
  so the saved rows match the pool reconstructed from `snapshots/`. A pre-seed eval persists an
  empty `sentinels` list, which isn't re-published (nothing to show yet).
- **The cadence ANCHOR is restored on resume — CLAMPED to the current step**
  (`_ForcedEvalMixin._restore_last_eval_step`, shared by BOTH callbacks). `_last_eval_step` is
  in-memory and resets each process, so it is restored from metadata; otherwise the resumed step
  sits far past a boundary and a fresh `0` would eval on step 1. That is right for a launcher
  RESTART and **wrong for a FORK**: `resume_eval_metadata` is the SOURCE run's run-level
  `metadata.json`, whose `latest_eval.step` is where *that* run last evaluated, not the step of the
  older checkpoint being forked from. Measured 2026-08-21 on an exploiter fork of gen-17's
  9,084,672-step checkpoint out of a run that reached 25M: the anchor restored to **24,000,000**,
  and since the cadence test is `(now // freq) > (anchor // freq)`, the fork would have launched
  **ZERO eval cycles** until it itself reached 26M — no `win_rate_vs_*`, no `eval_results.jsonl`
  row, no ELO. A gate arm whose verdict IS an eval metric silently produces nothing to read.
  Clamping to the model's `num_timesteps` restores the intended meaning and the next boundary after
  the fork point fires normally; a restart is unaffected (its recorded step is at or behind the
  loaded one, so the clamp never bites) and the clamp announces itself with an `anchor is AHEAD`
  event that states the FACT rather than asserting a cause — a crash-restart that rewound past a
  completed eval reads identically to a fork.
  ⚠️ It reads **`self.model.num_timesteps`**, not `BaseCallback.num_timesteps` — the latter is a
  mirror SB3 only syncs inside `_on_step`, so at `_init_callback` time it is still `0` even on a 9M
  resume, and reading it would clamp every restart to 0 and re-eval on step 1 (observed live before
  the fix, as `this model is at 0`). Same family as `_warn_if_fork_pool_empty`: a fork inherits the
  base's weights but none of its run-directory state, and the silent failures live in that gap.
  Test: `eval_fork_cadence_test.py` (both callbacks, parametrized).

| Flag | Default | Notes |
|------|---------|-------|
| `--eval-workers` | `5` | Eval subprocesses per cycle; work-steal **shard units** from a shared pool. Capped at the unit count (≈ opponents × shards-per-opponent, so sharding lets the full pool help). Self-play doubles this (→ `10`) since sentinel matchups run the model for both players. |
| `--eval-games` | `None` (=`EVAL_GAMES`, 100) | Games per **opponent** per eval cycle. Raise for tighter sentinel/promotion CIs (200 → ±0.069) at proportionally more eval compute — work-stolen across the workers, off the training path. Shards/opponent = eval-games / `--eval-shard-games`. |
| `--eval-shard-games` | `25` | Games per work-steal **shard unit** (battle-level work-stealing). Each opponent's `EVAL_GAMES` split into chunks any idle worker drains → the long tail collapses to one shard (≈4-shards-per-opponent default = ~4× shorter tail). Smaller = finer tail collapse but more player builds / (on websocket) more connection churn — the bridge is preferred for fine shards. `>= EVAL_GAMES` ⇒ one shard/opponent = the original opponent-level behaviour. Aggregation is exact (Σwon/Σfinished etc.); see the package below. |
| `--eval-device` | `cpu` | Device for eval-worker inference. `cpu` decouples eval from the training GPU. |
| `--eval-concurrency-per-worker` | `1` | Battles each worker overlaps **within** its claimed opponent (single-thread asyncio latency-hiding — NOT multi-core). `1` = today's sequential play. Threaded to the constructor's `eval_concurrency` → `cfg["concurrency"]` → `run_local_battles(concurrency=)` (bridge) / the player's `max_concurrent_battles` (websocket). See the concurrency note below. |
| `--keep-eval-snapshots` | `10` | Retain the N most-recent eval weight snapshots in `eval_traces/step_<N>/snapshot.zip` (~27MB each; default ≈270MB) for bit-exact prober replay. `0` writes the identity manifest only; the prober then loads the nearest persisted checkpoint. The trainer auto-prunes to this cap each cycle. |
| `--keep-eval-trace-steps` | `20` | The trainer keeps only the N most-recent eval **step dirs** under `eval_traces/` after each cycle (`0` = keep all), so forensic data stays bounded. `python -m main.prober.groom` is the manual fallback. |
| `--keep-stalls` | `50` | Each cycle keep only the N most-recent `stalls/stall_*.html` replays (`0` = keep all). A self-play run writes thousands (~80 KB each); this caps the dir. `artifact_retention.py`; CLI fallback `python -m agents.training.artifact_retention`. |
| `--keep-crashes` | `10` | Each cycle keep only the N most-recent `crashes/restart_err_*.txt` launcher diagnostics (`0` = keep all). Same module/CLI as `--keep-stalls`. |

**TD-residual tail metric (`eval/td_resid_tail_*`).** Each cycle also folds a **left-tail
statistic of the per-decision critic surprise** δ(t) = r(t) + γ·V(s_{t+1}) − V(s_t) — the same
formula the prober uses (`main/prober/session/core.py::ProbeSession._td`, the single source of
truth). `BattleRecorder`
accumulates δ live (one-step delayed backfill, closing each transition at the next `record()` when
the reward is finalized and V(s′) is known; the last decision has no δ). It costs **zero extra GPU**:
δ is computed only over the battles eval already captures forensically (where `need_aux=True` already
paid for V(s)), pooled per opponent (one `EvalRLPlayer` per matchup → `td_tail()`), and folded as a
**CVaR@5%** (mean of the worst 5%, `TD_TAIL_FRAC`; single min below `TD_TAIL_MIN_SAMPLES`=20). It
rides the exact win-rate plumbing — worker `shard__<unit_id>.json` (raw δ pooled across shards) → `merge_eval_results` →
`eval/td_resid_tail_vs_<opponent>` + `eval/td_resid_tail_mean` (TB + TUI), the `metadata.json`
`latest_eval` block (per-opponent + pool aggregate), and the append-only `eval_results.jsonl`. The
run's `model.gamma` is threaded into the worker (`base_cfg["gamma"]`) so the live δ matches the
prober's offline recompute (guarded by `td_residual_parity_fuzz_test.py`). More-negative = the critic
got blindsided more often — the **leading indicator for the critic-coverage obs work** (it moves in a
cycle or two, where saturated win-rate / gate-pinned `win_rate_vs_pool` / wide-CI ELO don't).

**Intra-worker concurrency (`--eval-concurrency-per-worker`, default `1` = sequential).** Each
worker overlaps up to N battles **within** its claimed opponent. This is **single-thread asyncio
latency-hiding, NOT multi-core** — everything (the obs build + PyTorch forward in `choose_move`, the
bridge/server I/O) runs on the one `POKE_LOOP` thread with BLAS pinned (`OMP/MKL=1`), so concurrency
only overlaps the time a worker is *blocked* on the bridge subprocess / websocket round-trip with
another battle's forward. The ceiling is **one core of compute**: a single-core bridge benchmark
(`/tmp/eval_concurrency_bench.py`, NN trainee vs bot and vs NN sentinel) measured ~**2.0× decisions/sec
at conc=3** on spare cores (plateau ~3; bot eval ≈2.0×, the heavier NN-vs-NN ≈1.8×) — i.e. about half
the per-decision wall-time at conc=1 was bridge I/O wait. **The old `_EVAL_SUBPROCESS_CONCURRENCY` = 1
default and its "measured slower" note were the *saturated* regime** (eval contending with training's
64 env workers for already-full cores — there the extra event-loop overhead nets negative); on **spare
cores (idle box / the cycle tail)** it's a clean ~2×. So the live gain runs between 1× and 2×
depending on how saturated the box is during the eval window; default stays `1` (opt-in). It does
**not** use idle cores at the tail — that needs *process-level* sharding (chunk one opponent across
workers); concurrency stacks multiplicatively on top of that (≈`2 × #shards`). Cross-opponent
parallelism is still the `--eval-workers` (5) subprocesses work-stealing the pool.

### Battle-level work-stealing (`eval_sharding/` package)

The *process-level* tail fix above is the `eval_sharding/` package — a small, deeply-encapsulated
unit with a narrow interface (4 focused files, no mega-file):

- **`units.py`** — `EvalItem` (one opponent the parent declares) + `ShardUnit` (a chunk of its
  games) + `plan_units(items, shard_games)`, a **pure** partition: split each item's games into
  ≤`shard_games` chunks (Σshards == n_games exactly), ordered LPT-ish (cost-descending items, shards
  round-robined) so every opponent starts early and the expensive ones lead.
- **`results.py`** — `ShardResult` (raw additive metrics: won/finished, reward+turn sums, the raw δ
  list — never a reduced ratio) + `aggregate`, which pools an opponent's shards back **exactly**:
  win_rate=Σwon/Σfinished, reward/ep_len count-weighted, and the TD tail by **pooling raw δ then one
  `td_tail`** (a CVaR can't be averaged). `td_tail` + its constants live here (the single source of
  truth; `eval_callback` re-exports them, so the dependency is one-way `eval_callback → eval_sharding`).
- **`pool.py`** — `ShardedEvalPool`, the deep coordinator. Parent: `write_plan(run_dir)` →
  `collect(result_dir)`. Worker: `from_plan(run_dir)` → `claim_next(claim_dir)` / `publish(...)`. It
  hides every filesystem mechanic; the worker never touches a lock file, the parent never touches a
  shard file. The plan (`plan.json`, items + shard_games) is the **single source of truth** both
  sides read — neither reconstructs the universe independently, so they can't drift.
- **`merge_eval_results`** is now a thin delegate to `ShardedEvalPool.collect` returning the same
  `merged` shape every downstream consumer already reads (record_per_opponent / build_bot_eval_block
  / record_elo / pool & externals blocks are **untouched**), plus additive `counts` (exact W/L) and
  `coverage` siblings.

**Exactness caveat (documented, by design):** win_rate / reward / ep_len are exact regardless of
`shard_games`. `td_resid_tail`'s *aggregation* is exact (pool the raw δ, compute the CVaR once), but
the *captured-battle sample* it's computed over shifts slightly with the shard count — the forensic
capture quota is per-unit (scaled `max(1, ⌈quota/shards⌉)`), so which battles contribute δ depends on
the split. It's a sampled diagnostic either way. Forensic trace files are namespaced by a per-unit
`trace_tag` (`{outcome}_s{shard}_{idx}`) so concurrent shards of one opponent don't collide in the
shared `eval_traces/step_<N>/<opponent>/` dir. Per-cycle `run_dir` is wiped at cleanup (and cleared
at launch), so no lock/shard/plan ever leaks across cycles. Sentinel/fixed opponent models are cached
per worker by path (immutable within a cycle → safe; the version check rides the first load) so a
fine split doesn't pay an N× 27MB deserialize. Worker rewrite: `eval_worker._play_unit` (one fresh
trainee + opponent per unit → independent measurement) + a per-worker model cache; tests:
`eval_sharding_test.py` (partition + aggregation-exactness property + claim-once + coverage),
`eval_sharding_fuzz_test.py` (real bridge battles through the real worker → exact pooled result).

### Rating-model seam (`rating.py`) — extensibility for Glicko-2 / TrueSkill

The live skill rating is anchored Bradley-Terry (`elo.py`), a *global batch* fit. `rating.py` is the
**ready drop-in point** for a different model without re-plumbing: `MatchRecord` (exact counts +
draws + `period_id` + optional opponent priors — the union BT, Glicko-2 and TrueSkill all need),
`RatingResult`, a `RatingModel` **batch** protocol, and `BradleyTerryRating` — a thin adapter over
`elo.fit_pairwise` whose ratings+SE are **byte-identical** to the live fit (pinned by `rating_test.py`).
`eval_rows_to_match_records` bridges the existing `EvalRow` history. The live `record_elo` path is
**deliberately unchanged** (zero risk): the seam exists and is tested, but routing through it buys
nothing until a new model is actually wanted — and Glicko-2 is *sequential* (period-by-period RD
carry-forward), so it needs the `SequentialRatingModel` sibling sketched in the module footer, not the
batch `fit`. Data fidelity is already in place: `eval_results.jsonl` now carries exact per-opponent
`counts` (additive, backward-compatible), so a future Glicko backfill has an exact ladder even under
partial shard coverage (where `win_rate × n_games` would be ambiguous).

## Self-play opponents (`--self-play`, gated behind pathology hunting)

When `--self-play` is set, `SelfPlayCallback` replaces `PerOpponentEvalCallback` and the
training opponents become frozen snapshots of the agent itself, drawn from a directory-backed
`SnapshotPool` (`snapshot_pool.py`; state reconstructed from `<run_dir>/snapshots/` on every
restart — no manifest). Design lives in `designs/ai_v5/`. Key behaviors:

- **Eval + promotion are NON-BLOCKING (frozen-snapshot subprocess), mirroring
  `PerOpponentEvalCallback`.** Self-play eval no longer runs in-process on the training thread.
  On a trigger step `SelfPlayCallback` freezes the live weights to disk (`model.save`) and
  spawns `--eval-workers`×2 (default 10) `main.eval_worker` subprocesses that **work-steal BOTH
  the bot roster AND up to `--n-sentinels` pool sentinels** (default 5; all split into shard units)
  from one shared pool (the
  worker's `_play_unit` SENTINEL branch plays the frozen trainee greedy vs each sentinel stochastic);
  training continues immediately. On a later
  `_on_step` poll the parent merges per-opponent + per-sentinel results → `win_rate_vs_bots` /
  `win_rate_vs_pool` / `sentinel_monotonicity`, records to TensorBoard + the TUI + metadata.json
  (with the `pool` block), persists `win_rate_vs_bots` (feeds `heuristic_fraction` next run),
  saves best by **copying** the frozen snapshot, and — if `win_rate_vs_pool > --promote-threshold`
  — **promotes the FROZEN snapshot into the pool by file-copy** (`SnapshotPool.add_from_path`):
  the live model has advanced since launch, so re-saving `self.model` would promote the wrong
  weights. Sentinels load via `load_model_snapshot` against the pool's shared `model_config.json`
  using `current_model_version(mappings)` — a stale-arch snapshot fails with `ModelVersionError`,
  never loads silently. The **only** training-thread work per cycle is the `model.save` freeze +
  one cheap `opponent_default_stats` IPC at collect; all battles / model loads / inference run in
  the worker processes, and the trainer holds no live eval connections (the worker rebuilds
  opponents/teambuilders/mappings itself). Skip-while-running, worker-crash-logged-and-continued,
  graceful-shutdown `drain()`, and resume-republish all behave exactly as the bot path above. The
  launch→poll→collect→drain mechanics are the **shared** `eval_callback.spawn_eval_workers` /
  `merge_eval_results` / `persist_eval_snapshot` / `prune_eval_*` / `replay_last_eval_to_tui`
  helpers, so the two non-blocking paths can't drift. `--debug --self-play --debug-eval` uses a
  fast eval cadence (every 4k steps, 3 games) so a short CPU smoke exercises seed → pool eval →
  promotion (a plain `--debug` smoke skips all eval by default — see `--debug-eval`).
- **Curriculum: thresholded ramp + LIVE per-episode fraction.** `heuristic_fraction`
  (`snapshot_pool.py`) is **0% self-play below `SELF_PLAY_START` (0.55)** — a weak model trains
  100% vs bots, no cycles wasted on a useless self-opponent — then smoothsteps `0.55→0.80` up to
  **90% self-play** (`HEURISTIC_FLOOR`=0.10 keeps a few % vs real bots for anti-forgetting). The
  three anchors are **configurable** — `--heuristic-floor` / `--self-play-start-wr` /
  `--self-play-full-wr` (defaults = the constants) thread through both the startup fraction and the
  live push, so a run can keep the coverage-punishing bots in the mix longer (raise `full` to ramp
  slower, raise `floor` for a bigger permanent bot slice). `--bot-weights name=w,…` additionally
  biases WHICH heuristic each episode draws (e.g. `aggressive_v2=3,heuristic2=3` → ~3× emphasis on
  the loss-analysis-flagged coverage bots; unlisted bots stay 1.0, omitted → uniform) — the weighted
  pick lives in `MaskableAgentWrapper._select_episode_opponent`, an O(1) in-memory `rng.choices`
  with zero per-step cost. All three default to the original behavior, so an unset run is unchanged.
  Crucially the heuristic-vs-pool split is **no longer fixed per process**: every training env
  picks its opponent **per episode** in `MaskableAgentWrapper.reset()` from a live
  `self_play_fraction`, and `SelfPlayCallback` pushes the fresh fraction (+ a `pool_generation`)
  to all envs via `training_env.env_method("set_self_play_target", …)` **after every eval**, so
  the ratio tracks measured strength mid-run with no restart. The opponent is a pure decision
  function over `env.battle2` (env.agent1/agent2 do the networking), so swapping it between
  episodes is free and safe — built `start_listening=False` (no idle connections), and the
  in-episode stale-decision path is untouched. The pool-vs-heuristic **coin flip is per-episode**
  (so the live fraction is honored exactly), but the pool **snapshot is (re)sampled+loaded only
  once per `pool_generation`**, NOT per episode: `load_model` deserializes a ~27MB MaskablePPO,
  and doing it every episode against an N-deep pool (LRU `lru_cache_size`=3) thrashed the workers
  — they blocked in `reset()` on the deserialize, dropping CPU to ~40% and FPS from ~1400 to ~500
  (regression fixed in `_select_episode_opponent`). A `pool_generation` bump (after a seed/promote)
  makes the worker re-scan + re-sample, so promotions become training opponents within a
  generation; diversity comes from 48 envs sampling independently + rotating each generation, not
  from per-episode churn. (`_n_pool_envs` / the `_maybe_engage_self_play` env-rebuild are gone.)
- **Opponent-mix reporting (`train/selfplay_fraction` / `train/stable_fraction` /
  `train/nonbot_fraction`).** The curriculum coin `sf` (`1 − heuristic_fraction(win_rate)`) pushed to
  the envs and persisted to `summary.json` is the **challenge-ENTRY** probability (= pool +
  un-mastered stable, *when* the challenge pick returns non-None) — NOT the pool share. So the
  reported metrics are derived separately by `SelfPlayCallback._opponent_mix_fractions(sf, pool_ready)`,
  a pure mirror of `MaskableAgentWrapper._select_episode_opponent` (it does **not** change selection).
  The four mutually-exclusive opponent types (bot / pool / un-mastered-stable / mastered-stable) sum
  to 1; the metrics report **`train/selfplay_fraction` = P(pool)** (REPOINTED — it used to log `sf`),
  **`train/stable_fraction` = P(any stable)** (un-mastered in the challenge **+** mastered in the
  weighted floor — a mastered stable "becomes a bot" so it's NOT in `sf`), and **`train/nonbot_fraction`
  = pool + stable** (= 1 − bot; bot is left implicit). `nonbot` is independent of the stable challenge
  share (it cancels); the per-bucket split needs three **reporting-only** inputs threaded into the
  callback from `train_rl_agent` (the capped `stable_challenge_share`, the `--bot-weights` vector, and
  `len(OPPONENT_CLASSES)` — the floor roster, which excludes eval-only `random`).
  With no stable opponents these reduce to `selfplay_fraction = nonbot = sf·P`, `stable = 0`.
  `_opponent_mix_fractions` is a hand-written **mirror** of the wrapper's selection, so the anti-drift
  guard is `wrappers_test.py::test_mix_fractions_match_actual_sampling`: it runs the REAL
  `_select_episode_opponent` thousands of times and asserts the empirical pool/stable shares match
  the analytic fractions (the per-case `selfplay_callback_test.py::test_opponent_mix_*` pin the math
  itself). A future selection change that isn't mirrored fails that cross-check.
- **Seeding is GATED on competence; the pool is a SLIDING WINDOW (nothing pinned) by default.** The
  pool is seeded only once win rate clears `SELF_PLAY_START` (at startup via `_maybe_seed_pool`, or the
  moment it crosses mid-run in `_collect_pending`), so the first self-play opponent is a
  *competent* model — never the random/weak step-0 seed of old. By default nothing is pinned: the
  oldest snapshot (incl. the seed) ages out as the window slides past `max_snapshots`, so the floor
  stays a recent self; anti-forgetting is the heuristic floor, not a pinned seed.

### 🚨 A FORK starts POOLLESS — auto-seed, and REFUSE the silent bot fallback (`pool_seed.py`)

**`SnapshotPool` derives its whole state from a directory, so a FORK begins in a new run dir whose
`snapshots/` is EMPTY — and an empty pool does NOT disable `--self-play`. It falls back to the BOT
pool.** A fold launched to replicate a parent that trained at 90% self-play therefore trains at ~0%
self-play, silently, with the argv, the startup banner and every metric still saying self-play. It
has now cost or nearly cost two cells:

* **2026-08-18** — three 3M-step `ai_v9_17_tdaux_*` forks off a 25M base ended with 1, 2 and 1
  snapshots against the base's 12. Essentially all 9M fork-steps were bot games, voiding a three-arm
  A/B whose gates were defined on the self-play regime.
* **2026-09-02** — the three-dose cell. Caught at launch by reading the startup line, before a
  GPU-hour was spent, and fixed by hand.

**THE MANUAL FIX WAS HALF A FIX THE FIRST TIME, and that half is the point.** Copying the parent's
`snapshot_*.zip` files alone still printed `self_play_fraction=0%`: the STARTING fraction comes from
`SnapshotPool.load_persisted_win_rate()`, which reads the pool's **METADATA**, not its zips. A pool
with 14 snapshots and no metadata reads as a competent-model pool the ramp has not opened yet —
exactly as wrong as an empty one, and it looks healthier.

**THE FILE SET the pool's loader reads out of its own directory** (audited 2026-09-02; `pool_seed.py`
copies exactly this, and `pool_seed_test.py` pins the two names against `SnapshotPool`'s own class
attributes so a rename there breaks the test rather than un-copying a file):

| path | read by |
|---|---|
| `snapshot_*.zip` | `_scan()` — they ARE the pool's entries |
| `summary.json` | `_SUMMARY_FILE` — `load_summary` / `load_persisted_win_rate`; carries `win_rate_vs_bots` (the ramp input) plus `self_play_fraction` / `last_eval_step` / `seeded` / `pool_generation` |
| `win_rate_vs_bots.txt` | `_WIN_RATE_FILE` — the legacy single-float fallback |
| `model_config.json` | NOT by `SnapshotPool` itself: `load_model_snapshot` looks beside the `.zip` and then one dir up, so without it every pool opponent arch-checks against the RUN ROOT's config instead of the pool's own |

Nothing else in the directory is read — there is no manifest, which is the whole reason the class is
directory-derived.

**THE SEEDING RULE** (`agents.training.pool_seed.prepare_pool`, called once from `train_rl_agent`
**before** the run's `SnapshotPool` is constructed — the starting fraction is read at construction,
so seeding afterwards would land the files and still announce 0%):

> `--self-play` ON **and** a genuine FORK **and** this run's pool is EMPTY ⇒ copy the fork parent's
> pool — every `snapshot_*.zip` plus every metadata file above — and print one line:
> `🌱 [SELFPLAY] [pool] seeded N snapshots + metadata from <parent run> (win_rate_vs_bots=…) [files: …]`

FORK-vs-RESTART is `main.train.fork_lr.is_same_run_checkpoint`, **IMPORTED, never re-derived** (a
second predicate for the same question is a second answer waiting to disagree), so a launcher restart
never re-seeds — re-seeding there would overwrite the run's own grown pool with the parent's stale one
every few hours. A **non-empty pool is never touched**, so the hand-seeded arms of a running cell keep
exactly the pool they were given when a later arm syncs this code. A **FRESH** run (no `--model`) is
unchanged: it legitimately starts poolless and grows one, and the win-rate gate is what stops it
seeding a random-weights opponent. The parent run dir comes from `lineage.fork_parent(run_dir)` when
the run already records one (immutable, so it names the ORIGINAL parent even after the launcher swaps
`--model`), else the `--model` path's own run dir — which is the only answer available on a fork's
first process, before any save has written a lineage block.

**THE REFUSAL.** If `--self-play` is on, the run is a FORK, and the pool is STILL empty after that
step — the parent has no pool, or `--no-fork-pool-seed` was passed — the launch exits
**`FATAL_CONFIG`** naming the three ways out (seed by hand, `--allow-empty-pool`, drop `--self-play`)
rather than quietly training against bots. `FATAL_CONFIG` and not `parser.error` because a restart
would hit the identical config, so the launcher must give up instead of looping.

| flag | default | |
|---|---|---|
| `--fork-pool-seed` / **`--no-fork-pool-seed`** | ON | opt out of the auto-seed. Declared POSITIVELY so `BoolFlag` generates the `--no-` form — declaring `--no-fork-pool-seed` would have generated `--no-no-fork-pool-seed` |
| **`--allow-empty-pool`** | OFF | explicit consent to the bot fallback on a fork. Never needed by a fresh run |

Both are **training-runtime** flags: they reach no extractor, scale no loss and change no weight
shape, so they are not in `agents/model/flag_registry.py`, not on `ModelVersion`, and not in
`check_compatible`; they land in `metadata.json`'s `cli_args` like every train-loop knob and the
launcher forwards them verbatim (`launcher/pool_seed_flag_forwarding_test.py`).

**PROVENANCE.** `seed_pool` writes `<pool_dir>/pool_seed.json` (parent run dir + name, pool dir, N,
the snapshot names, the metadata files copied, and the resulting `win_rate_vs_bots`), and
`run_io._run_lineage` attaches it to the run's lineage block as a **SIBLING key
`lineage.pool_seeded_from`** — never an edit to `fork_parent`. The block is written ONCE at fork
creation and frozen thereafter (`save_model_snapshot`: the existing value always wins), and the pool
is seeded earlier in that same process, so the fact is available exactly when the block is built and
no later restart can add or change it. ⚠️ The record is deliberately NOT written into
`metadata.json` at seed time: `run_io._resolve_fresh_model_dir` treats an existing `metadata.json` as
*"this name is already a run"*, so writing one before the first checkpoint would make a crashed
pre-save fork un-relaunchable under its own name.

**Verified end to end** (2026-09-02, CPU `--debug`, a scratch run root): a hand-built parent pool at
`win_rate_vs_bots=0.901250` seeded into a fork reproduced the parent's own startup line exactly —
`Pool has 1 snapshots, win_rate_vs_bots=90.12% → self_play_fraction=90%` — and the same fork with the
parent's pool hidden exited **3** with the three-way message. Gates:
`agents/training/pool_seed_test.py` (34), including the negative control that the **zips alone** read
`self_play_fraction=0%`, and the idempotence check that a hand-seeded pool comes back byte-identical.

- **PFSP / league-lite (`--pfsp-scale`, `--pool-spread`; both OFF → byte-identical).** A pure recency
  window is a near-50% echo chamber (recent selves beat each other ~evenly), so it never up-weights the
  *kind* of self the trainee is actually losing to. Two opt-in knobs turn it into a prioritised
  curriculum:
    - **`--pfsp-scale S` (default 0.0)** — `SnapshotPool.sample()` blends a per-snapshot HARDNESS factor
      into the weight: `weight = recency × (1 + S·(1 − p))`, where `p` is the trainee's measured win-rate
      vs that snapshot. A self it loses to (`p→0`) is sampled up to `1+S`× more; one it dominates (`p→1`)
      keeps factor 1 — never starved, so coverage is preserved. An unmeasured snapshot uses the mean of the
      known rates (average difficulty); with **no** rates yet (cold start) every factor is 1 ⇒ pure recency.
      The per-snapshot win-rates are exactly the sentinel win-rates the eval already measures: each cycle
      `SelfPlayCallback._update_pfsp_ema` EMA-smooths them (`_PFSP_WR_EMA_BETA`=0.5, to damp ~100-game eval
      noise) and `_prune_and_push_pfsp` prunes the map to the live pool and pushes it to every env via
      `env_method("set_opponent_win_rates", {step: p})` (mirrors the `set_self_play_target` push;
      `MaskableAgentWrapper.set_opponent_win_rates` → `SnapshotPool.set_win_rates`). The map survives resume
      in `summary.json` (`pfsp_win_rates`). Headline signals: `eval/pfsp_hardest_win_rate` (the most
      up-weighted self) + `eval/pfsp_tracked_snapshots`. Try `1.0–2.0`.
    - **`--pool-spread` (default off)** — replaces the oldest-evicted window with **spread retention**
      (`SnapshotPool._evict_spread`): always keep the newest + the oldest (a weak early self = a forgetting
      tripwire PFSP can up-weight) and thin the most-redundant interior snapshot (smallest neighbour
      step-gap) to an even ladder. So PFSP weights over a genuinely diverse range of selves, not a
      recent-selves cluster. Pairs with `--pfsp-scale`; alone it just diversifies the window.

  Both are threaded into the `SnapshotPool` at **both** construction sites (the per-env-worker pool that
  samples, and the trainer-side pool used for honest sentinel-weight telemetry); off → no extra IPC and the
  legacy sampling/eviction byte-for-byte.

  **REVIVAL VERIFICATION (2026-08-18) — it SURVIVED; nothing needed repair.** PFSP was built
  ai_v8-era and never production-enabled, so gen-16 wanting it ON required checking whether code
  that no test-suite failure would have protected still worked across the fresh-generation reset,
  the frame deletion and two signature bumps. It did: **70/70 existing tests green unmodified**, and
  every call site is intact — both `SnapshotPool` constructions (env-worker + trainer-side), the
  `_update_pfsp_ema` fold in `_collect_pending`, the `_prune_and_push_pfsp` env push, the
  `summary.json` `pfsp_win_rates` resume-load, and `MaskableAgentWrapper.set_opponent_win_rates`.
  A `--debug --self-play --debug-eval --pfsp-scale 2.0 --pool-spread` CPU smoke ran to
  `Training complete` (exit 0). **What that smoke does NOT show, and why it can't:** pool seeding is
  gated on `win_rate_vs_bots >= SELF_PLAY_START` (0.55) and a fresh debug model sits at ~4%, so the
  pool stays empty and PFSP never weights anything — the smoke proves the flags launch and thread,
  not that they skew.
  **The gap the revival actually closed was in the TESTS, not the code.** Every pre-existing test
  exercised ONE link with the other side mocked (pool math / callback EMA / wrapper forwarding), so
  a green suite said nothing about the composition — the thing a revival has to prove. Two
  end-to-end tests now run measured win-rates through callback → `env_method` → wrapper →
  `SnapshotPool` → `sample()`: `test_measured_winrates_skew_real_sampling_end_to_end` asserts the
  empirical 40k-draw distribution matches the analytic weights (at `pfsp_scale=2.0`, win-rates
  0.1/0.5/0.9 and recency off ⇒ factors 2.8/2.0/1.2 ⇒ shares **0.467 / 0.333 / 0.200**, and the
  self we lose to is drawn **2.33×** as often as the one we dominate), and
  `test_pfsp_off_makes_no_push_and_no_skew_end_to_end` asserts the same composition at
  `pfsp_scale=0` makes **no IPC call at all** and leaves the draw uniform under the same win-rates.

  **Honest caveats (it's a partial-coverage curriculum, not a full PFSP league):** (1) only the
  **`--n-sentinels` (default 5) evenly-spaced sentinels** the eval measures per cycle get a fresh win-rate —
  the other snapshots fall back to the cohort
  mean (treated as average difficulty), so on a 20-deep pool the default PFSP actively re-prioritises ≈¼ of the pool per
  cycle and an un-remeasured snapshot keeps its **last** EMA (a staleness bias toward selves you *used* to lose
  to — watch `eval/pfsp_hardest_win_rate` is tracking a moving target, not a fossil). (2) The `1 +` floor in the
  weight keeps coverage but makes the tilt mild: a self at `p=0.1` vs one at `p=0.5` differ only `(1+S·0.9)/(1+S·0.5)`
  (≈1.4× at `S=2`), and in a healthy gate-pinned pool the sentinel win-rates cluster near 50% so the realised
  prioritisation is modest — lean toward the high end of `S` (or beyond) if you want it to bite. PFSP touches
  **only which frozen opponent is sampled** — never the rollout, GAE, value target, promotion gate, or the
  `win_rate_vs_bots` curriculum ramp — so it cannot corrupt training; the worst case is "does little." A denser
  sentinel count under PFSP + a decay-toward-neutral for stale entries are the obvious follow-ups (deferred).
- **Full roster (v1 + v2 of every archetype).** Training (`OPPONENT_CLASSES`) and eval
  (`eval_opponent_names()` / `_EVAL_OPPONENT_SPECS`) both use all eight archetype bots —
  `{Heuristic, Heuristic2, Staller, StallerV2, Aggressive, AggressiveV2, SetupSweep,
  SetupSweepV2}` — because they play differently and the extra playstyle diversity is the
  point. There is no roster flag; the same nine names (eight bots + `random`) feed every
  path. `Random` is eval-only (a cheap "is the model broken" floor, excluded from
  `win_rate_vs_bots`); it is never a training opponent.
- **Resume state in `summary.json`.** `SelfPlayCallback` writes
  `<snapshot_dir>/summary.json` each eval (`win_rate_vs_bots`, `self_play_fraction`,
  `last_eval_step`, `seeded`, `pool_generation`) — `SnapshotPool.persist_summary`/`load_summary`.
  Read at `train_rl_agent` setup → the initial `self_play_fraction` (so a strong resumed model
  starts at the right ramp level, not the 0% cold-start) and the seed-gate decision. Distinct
  from the prober's `eval_traces/*/summary.json`; the legacy `win_rate_vs_bots.txt` is still read
  as a fallback.
- **Opponents sample, they don't argmax.** Training opponents are built with `stochastic=True`
  (now the `RLPlayer` default) so the learner trains against the policy's full action
  distribution — a richer, less-exploitable signal than the greedy move. Temperature is
  `--self-play-temp` (default `1.0` = the policy's own distribution; >1 flatter). **The measured
  trainee is always greedy** (`stochastic=False`) — that's what gives `win_rate_vs_bots`
  (curriculum) and `win_rate_vs_pool` (promotion) a stable, comparable control signal. The bots
  are deterministic rule-based players. The **pool sentinels default to stochastic@`--self-play-temp`**
  (mirroring how they act as training opponents) — so a sentinel matchup is greedy-trainee vs
  stochastic-sentinel, a deliberate asymmetry that inflates `win_rate_vs_pool` by a ~constant
  temperature handicap (≈15–20 pts; the [ELO caveat](#elo--skill-rating) below). **`--eval-sentinel-greedy`
  makes the sentinels greedy too** (`_play_unit` builds the sentinel opponent `stochastic=False`), so the
  matchup is best-vs-best and `win_rate_vs_pool` / the snapshot ELO reflect real skill (≈50% vs a
  recent self, ramping with sentinel age). It's eval-only — TRAINING opponents stay stochastic — and
  it auto-lowers `--promote-threshold` to `0.55` (else the handicap-free pool win rate never clears
  the 0.65 gate and the pool freezes). Default off so the live metric stays continuous until opted in.
- **Opponent snapshots are version-checked.** They load via `load_model_snapshot` (not a raw
  `MaskablePPO.load`), and `SnapshotPool` writes a shared `model_config.json` next to its
  snapshots, so an arch-mismatched snapshot fails with a clean `ModelVersionError` instead of
  loading mismatched weights.
- **The opponent RE-DECIDES on a stale decision; the trainee crashes** — split by who *owns* the
  decision. `SingleAgentWrapper` polls the opponent's `choose_move` on the *training* thread while
  POKE_LOOP mutates its battle, so by serialize time the captured snapshot (`ctx.legal`) can diverge
  from the live battle: POKE_LOOP parses an **in-flight turn-resolution during the model forward**,
  advancing `battle.turn` one ahead of `ctx.turn` (proven by the race trace — mutual Arena-Trap
  Dugtrios, the turn resolves mid-decision). `assert_decision_current` / `action_to_order` raise
  `StaleDecisionError`; handling then splits:
  - **Opponent** — its decision is *internal* to `step` (SB3 never sees it), so `RLPlayer.choose_move`
    catches the error and **re-decides on the now-current request**, bounded (`_OPP_REDECIDE_MAX`),
    with a valid default fallback only if the battle never settles. It must always return a valid
    order: SB3 has **no failed-step path** (a raise kills the `SubprocVecEnv` worker → parent hangs →
    worker-watchdog `os._exit`s → launcher restart). Each attempt's `embed_battle()` records its
    would-be decision into the rolling turn-history, so `choose_move` snapshots the tracker before
    the loop and `EpisodeTracker.restore()`s on a stale attempt — the superseded decision leaves
    **no phantom turn** in the opponent's turn-history obs (only the committed one survives; guarded
    by `redecide_rollback_fuzz_test.py` + `episode_tracker_test.py`). The re-decide guards only up to
    the order `choose_move` RETURNS; `SingleAgentWrapper.step` then re-serializes it via
    `self.env.order_to_action`, re-reading the battle **one more time** — a second, narrower window
    where it can finish/flip-to-wait under us (`ValueError ... not in valid orders ['/choose
    default']`). On that the wrapper falls back to the default order rather than crash (guarded by
    `single_agent_wrapper_test.py` + `order_to_action_race_fuzz_test.py`).
  - **Trainee** — its action is *SB3's*, computed outside `step` and not re-runnable mid-step, so a
    stale trainee decision **crashes** (`gen3_env`, no fallback): acting on it would corrupt its
    `(obs, action) → (reward, next_obs)` transition. Empirically it doesn't hit this — gated by the
    env's `race_get` request-wait (17 h vs-bots + self-play, zero trainee staleness).
  `_settle_opponent_battle` is a **pre-drain** that only trims how often the opponent re-decides — it
  can't drain *in-flight* messages, which is why re-decide (not settle) is the fix. The comprehensive
  `assert_decision_current` (every axis: moves+disabled, switches+species,
  force_switch/trapped/maybe_trapped/wait/struggle) is the detector; `train/selfplay_opp_redecide_rate`
  surfaces the resolved-race rate. **Full context — mechanism, the race trace, why it was hard, and the
  verification tiers — is in `race_fuzz_README.md`.** (`GEN3_FORCE_SELFPLAY` forces 100% self-play for
  the stress; `GEN3_RACE_TRACE=1` dumps the per-battle cross-thread interleaving into the
  `StaleDecisionError` **and** into the `race_get` silent-stall crash — see below. `StaleDecisionError`
  lives in `agents/action/mapper.py`.)
  - **Force-switch request-delivery deadlock (`_AsyncQueue.race_get`, `env.py`) — FIXED.** A
    *different* failure from the stale-decision race, and a latent bug **inherited verbatim from
    upstream poke-env 0.15.0**: `race_get` races a per-agent `queue.get()` against the
    `_waiting`/`_trying_again` coordination events, and can drop a request the server already
    delivered into the `battle_queue`. Two ways: **(1) stranding** — `asyncio.wait(FIRST_COMPLETED)`
    returns the instant any waiter completes, so an already-set **stale** event wins before the
    equally-ready `queue.get()` runs → `race_get` returns `None`, the agent is marked not-to-move,
    and its request sits unread; **(2) orphan theft** — `race_get` `cancel()`s the pending
    `queue.get()`, which a later `put` can resurrect to dequeue-and-discard the request.
    `_trying_again` goes stale because `env.step` cleared it only on the `None` path, and a
    re-request makes the battle non-`None`, skipping that clear. The trigger is the mutual
    Arena-Trap Dugtrio self-play mirror (trapped-switch `[Unavailable choice]` → stale
    `_trying_again`, then a faint → a `wait`+`forceSwitch` pair whose force-switch is stranded);
    rare (~1/8600 battles), so it only surfaced once self-play was on. **Fix:** `race_get` now
    `cancel()`s **and `await`s** the get to settle it (recovering its item, never orphaning it) and
    **prefers a queued battle over a stale event**, and `env.step` clears `_trying_again` the moment
    its agent receives a battle. Repro + regression guard: `forceswitch_deadlock_fuzz_e2e_test.py`
    (needs a `9XXX` server; `--widen` surfaces the timing race); unit coverage of both failure modes
    in `async_queue_disconnect_test.py`.
  - **Silent-stall watchdog (now a should-never-fire backstop).** Independently of the fix above,
    `race_get` bounds its wait by `_RACE_GET_TIMEOUT_S` (120 s, ~100× a normal step; override with
    `GEN3_RACE_GET_TIMEOUT_S`) and on a silent stall **raises `ShowdownException`** — a hard crash
    that propagates uncaught through the wrapper step chain to the SubprocVecEnv worker, so SB3
    discards the in-flight rollout (no fabricated transition reaches backprop) and the launcher
    restarts from the last checkpoint. It **crashes, never recovers in place** (recovering would feed
    PPO a stale `(obs, action) → (reward, next_obs)`). With `GEN3_RACE_TRACE=1` the wedged battle's
    interleaving is appended to the crash message via `race_trace.dump_recent()` (wedged battle
    ordered last so its newest events survive the launcher's last-100-line crash-file tail; the full
    trace is in `launcher_child.log`). `env.step` also emits `ENVSTEP` enter/race trace lines under
    `GEN3_RACE_TRACE` for debugging this handshake. Kept as defense-in-depth against any future
    request-delivery regression.
- **Self-play engages in the first process, not only after a restart.** The env is built before
  the model exists (the model needs the env's spaces), so on the first self-play process
  `_maybe_engage_self_play` seeds the pool from the loaded weights and rebuilds the env with
  pool opponents (then `set_env`). The worker watchdog is started *after* this, just before
  `learn()`. Later restarts find the pool already populated and skip the rebuild.
- **`--debug --self-play --debug-eval` exercises the real path** (seed → pool eval → promotion)
  on a fast eval cadence, so a CPU smoke against a `9XXX` server validates the wiring without
  disrupting the `:8001` training server (`--debug` skips all eval by default — `--debug-eval`
  opts in). `selfplay_opponent_fuzz_test.py` covers the opponent load + legal
  play (both modes) + version check in-process via the local bridge (no server).

## WHICH FILE a run spec names — the ONE resolution rule (`gen3_last_snapshot_resolution_v1`)

**A bare run directory resolves to the run's LAST SNAPSHOT, not to `best_model/best_model.zip`.**
Owner ruling, 2026-09-06, verbatim: *"I would either prefer us do best against target or just do the
last snapshot. I feel like best against target will always have a nuance that we need to keep track
of, whereas the last one is probably what our metrics would measure anyway."*

**WHY IT CHANGED.** `best_model/best_model.zip` is exported on **BOT win rate** — an opponent set
with nothing to do with what a teacher is being distilled FOR. Ledger 2026-09-06 (probe H8,
*exploiter off-slice competence*) measured the consequence: for 2 of 8 unfunded R5F teachers
(`ai_v9_94_R5F02`, `ai_v9_98_R5F06`) the exported file was a **~0.93M-step exploiter rather than the
~2.93M final**, so "the teacher" a fold distilled from was neither the last snapshot nor the best
against its target — and **nothing recorded which file was used**. It made `teacher_distance`'s
UNF budget covariate (3.07M) heterogeneous (≈2.43M mean) on the very axis it had found
rank-indistinguishable from D_off. Every meter this programme banks scores a run at its END, so the
last snapshot is what the metrics already measure.

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

### Every consumer goes through the ONE choke point

`agents.training.fixed_opponent_pool.resolve_model_ref(path, step=None)` → a `ResolvedModel`
(`zip_path`, `config_path`, `run_base`, `run_dir`, `rung`, `rule`, `num_timesteps`). The flags it
serves: **`--distill-teacher`** and **`--win-prob-pbrs-source`** (`main/train/model_build.py`),
**`--stable-opponents`** and **`--exploiter`** (via `resolve_stable_opponents`),
**`--exploiter-ladder`** (`exploiter_ladder.py`), **`--warmstart-consensus`** (`warmstart.py`) and
**`--distill-anchor-parent`** (`main/train/callbacks.py`). `run_spec_test.py` holds the census that
fails, naming the file and its flags, when one of them stops.

**`_resolve_zip_and_config(path, step)` is a FROZEN 3-tuple wrapper over it** — the offline probe
scripts under `designs/research_state/measurements/arch_transfer_2026-09-05/`
(`content_locality_v2`, `exploiter_competence`) import it by name to reproduce exactly the call
`model_build.py` makes for a teacher. **They measured the OLD rule's files, by design, and stay as
records of it.** New call sites that want the rung or the step should call `resolve_model_ref`.

### Provenance — a fold now records which file it loaded

* `metadata.json`'s **`lineage`** block: every model reference (`fork_parent`, each entry of
  `teachers`, `exploiter_target`) carries `resolved_file`, `resolved_num_timesteps`,
  `resolution_rung` and `resolution_rule`. `python -m main.lineage <run>` prints them.
* **Startup lines**: `🧪 [DISTILL]` emits one `teacher <k>: <spec> -> <zip> @<N> steps [rung=… rule=…]`
  per teacher; `🐴 [STABLE]` and `🥊 [EXPLOITER]` emit the same per opponent
  (`FixedOpponentEntry.provenance()`); `🧊 [WinProbPBRS]` names its frozen φ the same way.

🚨 **EVERY TEACHER LOADED BEFORE 2026-09-06 WENT THROUGH THE OLD RULE** (`best_model` first, then
`final_model.zip`, then `<run>/best_model.zip`) and recorded nothing about it. A pre-change run's
teacher identity is therefore **not recoverable from its metadata**, and `main.lineage` says
`resolved file not recorded (pre gen3_last_snapshot_resolution_v1)` rather than re-resolving it
under today's rule — a current answer presented as history is worse than no answer. A reference
this change DID try and fail to resolve records `unresolved`, so the two are distinguishable.

**NOT VERSIONED.** This changes which FILE a run loads, never a weight shape, so it is absent from
`ModelVersion.check_compatible` / `arch_signature` by design, and no checkpoint on disk becomes
incompatible with it. Gates: `agents/training/fixed_opponent_pool_test.py` (each rung, both
disagreement directions, the explicit-form passthroughs, the frozen 3-tuple, the entry's
provenance), `agents/training/run_spec_test.py` (the choke-point + consumer census),
`agents/training/lineage_test.py` (the recorded fields and the legacy message).

## Stable (cross-run) opponents (`--stable-opponents`, `fixed_opponent_pool.py`)

Load a frozen model from **another, already-finished run** as a **fixed opponent** — measured
against in eval AND (under `--self-play`) played against in training. Design:
`designs/ai_v5/design_stable_opponents.md`. Which FILE a run dir resolves to is the ONE rule above
— the run's **LAST SNAPSHOT**, with `best_model/best_model.zip` as the last-resort fallback.

**Training-mix participation (Stage 2) — "tossed in like a sentinel, becomes a bot when mastered":**
a stable opponent rides the *existing* pool-vs-heuristic split in `MaskableAgentWrapper`
(`wrappers.py`), no new source-model abstraction:
- **CHALLENGE bucket** (the self-play pool branch, competence-gated by `self_play_fraction`): the
  pool gets the BULK; un-mastered stable opponents share a **capped minority slice**
  (`STABLE_CHALLENGE_SHARE` = 0.20 in `wrappers.py`), so a single fixed opponent can never dominate
  training (multiple un-mastered ones SHARE the 20%, so the total stays bounded). It only enters the
  mix once the model clears `SELF_PLAY_START` (a weak model trains on bots first), and only under
  `--self-play` (without it, stable opponents are eval-only — a startup NOTE says so).
- **FLOOR bucket** (the heuristic-bot branch): once the trainee **masters** it
  (`win_rate_vs_ext_<run>` ≥ `--stable-opponent-mastered-wr`, default `0.80`, for
  `_MASTERY_CONFIRM_CYCLES`=2 consecutive cycles — a noise guard since the irreversible flip is
  one-way), it "becomes another bot" — moved to the always-on coverage floor (weighted like an
  unlisted bot). The eval callback tracks a **monotonic** mastered set + a per-label streak counter,
  recomputed each cycle (→ resume-safe), and pushes it via `env_method("set_stable_mastered", …)`,
  exactly like `set_self_play_target`. The recompute+push runs **early** in `_collect_pending` (with
  the training-mix telemetry below), so this cycle's challenge↔floor flips show up in both the pushed
  env state and the reported fractions. **Resume note:** the mastered set lives only in callback
  memory, so after a launcher restart a previously-mastered opponent reverts to the challenge bucket
  until the first post-restart eval re-confirms it (self-healing; bounded by the eval cadence).
- **Training-mix share is reported, not just eval win rate.** The stable opponents' actual slice of
  the training mix shows up in `train/stable_fraction` (challenge un-mastered + floor mastered), with
  `train/selfplay_fraction` (pool) and `train/nonbot_fraction` (their sum); see the Curriculum
  subsection's **Opponent-mix reporting** bullet above for the exact decomposition.
- **Dynamic within-slice selection (`--stable-opponent-pfsp`, default off).** A FLAT capped share
  splits the stable slice UNIFORMLY over the un-mastered opponents — so a generalist hardening against
  several exploiters at once spends equal budget on the axis it already handles and the one it's
  failing. Under `--stable-opponent-pfsp`, `MaskableAgentWrapper._pick_stable` weights the
  un-mastered-stable pick by **`1 − win_rate`** (floored 0.05) — the exploiter it's LOSING to worst
  gets most of the slice, and each fades as mastered (win_rate→1 ⇒ weight→0), then the mastery flip
  retires it to the floor. Win-rates are the same `win_rate_vs_ext_<label>` eval already computes,
  EMA-smoothed (`_PFSP_WR_EMA_BETA`) and pushed each cycle via `SelfPlayCallback._push_stable_mastered`
  → `env_method("set_stable_win_rates", …)` (mirrors the pool PFSP `set_opponent_win_rates`). **The
  TOTAL pool-vs-stable share is unchanged** (still `--stable-opponent-selfplay-share`), so the
  opponent-mix telemetry + the `test_mix_fractions_match_actual_sampling` anti-drift guard are
  unaffected — only WHICH un-mastered stable opponent is picked shifts. Training-only (not
  version-locked, forwarded on resume like `--pfsp-scale`); OFF = uniform, byte-identical. **Pairs
  with a raised `--stable-opponent-selfplay-share`.** Motivation: a flat 0.35 share (≈12% exposure
  each of 3 exploiters) left ai_v7_14's hardening flattening at ~0.30 vs the exploiters; the dynamic
  focus + a raised share is the fix. Tests: `wrappers_test.py::test_stable_pfsp_*`.
- The stable-opponent players are **built once per worker** (`load_foreign_opponent` in the env
  factory), so no per-episode reload; each plays **stochastic** at `--stable-opponent-temp` in
  TRAINING but **greedy (temp 0)** in EVAL (a clean yardstick).
- **Surfaced in the launcher Events panel** (via `emit`, like the `[SELFPLAY]` startup lines): a
  `🐴 [STABLE] N cross-run opponent(s): ext_<run> — eval greedy; training ≤<share> of self-play until
  mastered (win_rate ≥ <wr>)` line at startup (and a `🏇 [SELFPLAY] Mastered stable opponent(s) …`
  line on the challenge→floor flip), and each eval-summary event gains a `stable <pct>%` field. (Per-opponent `eval/win_rate_vs_ext_<run>` also rides the normal eval Metrics table.)

- **CLI:** simplest form is just the run dir — `--stable-opponents models/ai_v5_5_popart_N_0607`,
  which resolves to that run's **LAST SNAPSHOT** (the rung table above; it was `best_model` until
  2026-09-06);
  the opponent is **labelled by the run-dir name** (`ext_ai_v5_5_popart_N_0607`, derived
  `best_model`/`snapshots`-aware so a direct `…/best_model/best_model.zip` path still yields the run
  name, not `best_model`). Optional per-entry suffixes: `@<step>` (a specific checkpoint, which
  BYPASSES the ladder), `:<name>` (rename). The resolved zip, its `num_timesteps` and the rung that
  picked it ride the entry (`FixedOpponentEntry.{resolution_rung, resolution_rule, num_timesteps}`)
  and are printed on the `🐴 [STABLE]` / `🥊 [EXPLOITER]` startup lines. **Per-opponent weights (`=<weight>`) are rejected** with a clear
  message (not supported). Knobs: `--stable-opponent-temp` (default 1.0 — the *training* play
  temperature; eval is always greedy) and `--stable-opponent-mastered-wr` (default 0.80 — the
  challenge→floor flip). Parsed + resolved at startup by `fixed_opponent_pool.resolve_stable_opponents`.
- **Compatibility = the OBSERVATION FAMILY only** (two axes: obs family vs model family — see the
  design §3). The gate is **same `arch_signature`** (`ModelVersion.check_opponent_compatible`,
  the obs-family proxy); a mismatch is a **startup FATAL** (`[StableOpponent] FATAL` →
  `TrainExitCode.FATAL_CONFIG`, surfaced to the TUI, no restart). Loaded inference-only via
  `snapshot.load_foreign_opponent` (`env=None`), which **skips `check_compatible`** — so
  `use_popart`/`vf_coef`/reward differences (irrelevant to an opponent's forward, which never reads
  the value head) don't block it. The example `models/ai_v5_5_popart_N_0607` shares HEAD's arch, so
  it loads despite being PopArt-on.
- **Label namespace `ext_<run>`** — underscore separator (NOT `ext:`) so the emitted metric tags are
  **uniform** with the rest (`eval/win_rate_vs_ext_<run>`, like `eval/win_rate_vs_sentinel_0`), no
  colons in TensorBoard. `is_external` (`startswith("ext_")`) keeps them out of the bot aggregates.
  Both eval callbacks (`PerOpponentEvalCallback` + `SelfPlayCallback`) add the `ext_` labels as
  `FIXED` `EvalItem`s (so they shard + ride the same plan); the worker's `_play_unit` FIXED branch
  (`eval_worker.py`) plays the **greedy trainee vs the greedy stable opponent** (a clean yardstick).
- **Metric set (deliberate, uniform across both callbacks):** per opponent —
  `eval/win_rate_vs_ext_<run>`, `eval/mean_reward_vs_ext_<run>`, `eval/mean_ep_len_vs_ext_<run>`;
  plus `eval/win_rate_vs_external` ONLY for a mini-league (2+ — with one it duplicates its row; it's
  an `_EVAL_SUMMARY` "vs External" row, not a fake per-opponent row); plus a `metadata.json:latest_eval`
  `externals` block. Kept **OUT of** `win_rate_vs_bots` (`bot_mean` excludes them), `win_rate_vs_pool`,
  the best-model aggregate, the `td_resid_tail_mean` headline, and **the ELO FIT itself** (no ladder
  distortion). **NOT emitted for ext:** `td_resid_tail` (a bot/sentinel critic-coverage diagnostic).
  The TUI renders each by its run name with an `(ext)` tag.
- **ELO shown in the eval table** (`record_external_elos`): the elo column for an `ext_` row PREFERS
  the opponent's **own recorded ELO** — read at startup from its `best_model.json` sidecar (or run
  `metadata.json`) `latest_eval.elo` into `FixedOpponentEntry.source_elo` (`_read_source_elo`). It's a
  well-fit, bot-anchored rating (cross-run-comparable since the bot anchors are stable) — e.g. 1902 for
  `ai_v5_5_popart_50m_0607`. **Fallback** (`external_elo`) when the opponent carries no recorded ELO:
  invert the BT win prob from the trainee's live rating + win rate (`R_opp = R_trainee −
  (400/ln10)·logit(wr)`, clamped ≈±676) — a rough single-edge estimate. Recorded as
  `eval/elo_vs_ext_<run>`; the opponent is NEVER a player in the fit itself (no ladder distortion).
- **`best_model/` is self-contained.** Saving the best model copies the run's `model_config.json` AND
  writes a `best_model.json` sidecar (`copy_run_config_to_best_model` + `write_best_model_sidecar`,
  both called from both eval callbacks' best-save). `best_model.json` reuses
  `snapshot.write_checkpoint_metadata` (the per-checkpoint sidecar code) so it carries the
  `latest_eval` block **incl. the run's ELO** —
  `best_model/{best_model.zip,model_config.json,best_model.json}` co-located (arch gate + carried ELO,
  no parent search). Backfilled for existing `models/*/best_model/` dirs.
- **Per-opponent pinned teams (the league FOLD-BACK contract).** A SPECIALIST stable opponent —
  one whose run pinned `--trainee-team` — pilots **ITS OWN team** here, not the shared pool
  (otherwise a trapper exploiter folds back piloting random teams and the pressure it was trained
  to apply evaporates — the realized-matchup lesson applied to the opponent side).
  `resolve_stable_opponents` reads the pin from the opponent run's `metadata.json:
  cli_args.trainee_team` (`_read_trainee_pin`) into `FixedOpponentEntry.team_str` — **fail-loud**:
  a recorded pin whose file is missing raises, and a pin that no longer matches the run's recorded
  MatchupSpec `pin_sha` raises (never a silent pool fallback). TRAINING: the env factory builds a
  per-entry pinned builder and `MaskableAgentWrapper._apply_opponent_team` switches
  `env.agent2._team` **per episode** to match the selected opponent (agent2 does the opponent-side
  networking, so its `_team` decides the opponent's real team — the mirror lesson); unpinned
  episodes restore the pool builder (the SAME instance, so team-draw RNG streams are unchanged);
  with no pinned opponent anywhere the wrapper never touches `agent2._team` (byte-identical). EVAL:
  `team_str` rides `to_cfg()` → the `EvalItem` → `eval_worker._fixed_opponent_tb`, so the FIXED
  branch measures the opponent piloting its pin (eval matches training, same rule as the trainee's
  own pin). The `[STABLE]`/`[EXPLOITER]` startup lines annotate `[pilots ITS OWN pin: <file>]`.
  Guard: `poke_env_gaps/opponent_pin_fuzz_test.py` (bridge, real battles — pinned episodes field
  EXACTLY the pin, bot episodes the pool).
- **Tests:** `fixed_opponent_pool_test.py` (parse + resolve + the arch FATAL gate + the pin
  resolve/fail-loud/sha cases + `register_exploiter_for_eval` dedup),
  `snapshot_test.py::*opponent*/*foreign*` (the loader + `check_opponent_compatible`), and the
  end-to-end `stable_opponent_fuzz_test.py` (bridge, no server — resolve + arch FATAL + foreign
  load + legal stochastic play) + `opponent_pin_fuzz_test.py` (the fold-back realized-team guard).

## Exploiter mode (`--exploiter`, `MaskableAgentWrapper._exploiter_player`)

A clean opponent-mix front-end for the league **exploiter** role: train a dedicated agent against
ONE fixed foreign model as the **sole opponent every episode** — to surface (and then patch, by
folding the exploiter back as a stable opponent / pool member) the non-robustness a *self-play* Nash
can't see. It needs **no `--self-play` / `--stable-opponents` / share fiddling** — point `--exploiter`
at the target and it's the only opponent.

- **Target resolution** reuses the stable-opponent path exactly: `--exploiter <run-dir|checkpoint
  spec>` → `resolve_stable_opponents` (a single `FixedOpponentEntry`, arch_signature-gated) +
  a weights-load validation in the main process (corrupt zip = startup `[Exploiter] FATAL` →
  `FATAL_CONFIG`, no restart loop). Emits a `🥊 [EXPLOITER]` line to the launcher Events panel.
- **Opponent mix**: the env factory builds ONE `RLPlayer` over the target per worker (stochastic at
  `--stable-opponent-temp`, a moving target), and `MaskableAgentWrapper._select_episode_opponent`
  **short-circuits** the whole challenge/floor/pool/stable selection when `exploiter_player` is set —
  the target is `self.opponent` every reset. `None` (default) = the normal selection, byte-identical.
- **Team-source guarantee — an exploiter may ONLY EVER pilot a vetted SAMPLE team.** The curated
  `data/teams/sample/` set is the tournament-proven, rock-solid roster; the ~687 `other` teams are
  bulk-downloaded and unvetted. `matchup_spec.validate_exploiter_trainee_is_sample(matchup,
  sample_teams)` (called at startup in `train_rl_agent`, FATAL → `FATAL_CONFIG`) enforces that a
  `mix_kind == 'exploiter'` run with a pinned `--trainee-team` pins a team whose strip-normalized
  fingerprint is in the sample set — else it refuses to launch with a clear message. Out of scope:
  non-exploiter runs (any pin allowed), and an exploiter with an UNPINNED trainee (a full-pool
  exploiter, not a single-team specialist). The shipped TSS pin IS a sample team, so it passes. **A
  multi-team exploiter (`--trainee-teams` → `pin_multi`) validates EVERY member likewise** (each of the
  N teams must be a sample). Tests: `matchup_spec_test.py::test_exploiter_*sample*` +
  `::test_pin_multi_*` + the e2e FATAL.
- **Widening the curated set — `python -m main.promote_teams`.** The refusal message above says
  *"promote this one into the sample set first if it is proven"*; a fleet larger than the curated set
  (40 teams against 32 curated) is where that stops being hypothetical. The tool is a **seed-recorded
  UNIFORM RANDOM draw**: exclusions (already-taught ∪ rev-4-pending ∪ the 2 held-out transfer
  instruments, from `designs/ai_v12/promotion_exclusions.json`) → a `random.Random(seed)` shuffle of
  the sorted eligible pool → `validate_teams_locally` → copy into `data/teams/sample/` →
  `PROMOTION_MANIFEST.{md,json}`. **Random rather than ranked is an owner ruling** (ledger
  2026-08-30): a hand-picked or headroom-ranked fleet makes its own result a selection estimate
  rather than an unbiased one, so archetype/folder composition is REPORTED, never corrected. A team
  that fails validation is REPLACED by the next candidate in the same shuffle and recorded — never
  silently dropped, which would shrink the fleet. `--dry-run` plans, `--draw-only` emits the manifest
  for review, `--root <copy>` rehearses the whole promotion on a tree copy, `--verify-exclusions`
  re-derives the exclusions from each run's `metadata.json` (`read_recorded_trainee_teams`) and
  `--regenerate-exclusions` REWRITES them from it.

  🚨 **The exclusion artifact ROTS, and its own committed tests could not see it.** It was first
  built from **frozen argv files** in a session-scoped job directory, before the runs they describe
  had launched — and the launched rev-4 runs dealt different teams: all three arms disagreed with
  their `metadata.json`, naming 4 teams rev-4 never pinned and missing 4 it did. *A frozen argv is a
  plan; `metadata.json` is the record.* **The union stayed 26 and eligible stayed 693 throughout**,
  so the pre-existing assertions (union size, per-category counts, `719−26=693`) all passed while
  the list was wrong — **a count-shaped check cannot see a membership error**. The gate is now
  `::test_the_committed_exclusions_agree_with_recorded_run_provenance`, which names the offending
  team ids in BOTH directions and refuses to pass vacuously when no run dir is present; verify,
  repair and gate share one derivation (`recorded_provenance`) so a check and a repair cannot
  disagree. Repaired 2026-08-31 —
  `designs/research_state/measurements/exclusions_and_artifacts_repair_2026-08-31.md`. ⚠️ The
  eligible COUNT was unchanged but the DRAW was not: re-running the committed demo at its own seed
  moved 21 of 40 positions, because a seeded shuffle of the *sorted* eligible list is reproducible
  against a FIXED set and not stable across a change to it.

  ⚠️ **Promotion MOVES a team between manifests; it must not list it in both.** `TeamLoader`'s
  universe is the `teams.json` files, deduped by resolved *path* rather than by text — so a team in
  `sample/teams.json` that is still in `others/<author>/teams.json` is loaded twice and drawn as an
  opponent twice as often. That is the `yak_attack`-66%-of-draws defect, recreated on exactly the
  teams the fleet measures. The tool therefore de-lists from the source manifest (leaving the `.txt`
  on disk, so the change is reversible from the manifest alone) and then re-loads through
  `TeamLoader` to assert the pool total is unchanged and no sha appears twice. Two more traps it
  closes: `validate_teams_locally` returns the same `{"valid": False}` shape for a broken node bridge
  as for an illegal team, so a known-good **positive control** rides in every batch and a failure
  aborts instead of "replacing" the whole pool; and the manifest is **write-once** (a second draw
  under a different seed is refused without `--force`, so the seed cannot be re-rolled until the
  composition looks good). Gates: `src/main/promote_teams_test.py` (24 tests, ~0.3 s, no node).
- **Mutually exclusive with `--self-play`** (arg-parse error — the exploiter needs no pool). Because
  it's not self-play, `_opp_version` (the arch gate for the foreign load) is set explicitly for this
  path before the factories are built. Training-only; not version-locked.
- **Temperature-annealing curriculum (`gen3_exploiter_temp_anneal_v1`, `--exploiter-temp-start`).** A
  from-scratch trainee vs a STRONG frozen target is crushed every game — the PPO advantage is ~0 (all
  losses look equally bad) and it never gets a foothold. This anneals the target's SAMPLING TEMPERATURE
  over training — a difficulty curriculum via opponent STOCHASTICITY (not by swapping opponents): start
  the target HOT (`--exploiter-temp-start`, e.g. 2.0 → flatter logits → noisier/weaker play, so the
  trainee wins some games and gets a learning signal) and linearly anneal it to `--exploiter-temp-end`
  (default 1.0 = the target's true play distribution) over `--exploiter-temp-anneal-frac` of `--steps`
  (default 0.2), held after. `ExploiterTempAnnealCallback` (`exploiter_temp_callback.py`) computes the
  temp from SB3's `_current_progress_remaining` each rollout (both the sync and async collectors call
  `on_rollout_start`) and pushes it to every env's exploiter `RLPlayer` via
  `env_method("set_exploiter_temperature", T)` — the `set_self_play_target` idiom;
  `MaskableAgentWrapper.set_exploiter_temperature` sets `RLPlayer._temperature` (read fresh each
  `choose_move`). Metric: `train/exploiter_temp` (TB + TUI). **Training-only** — no weight-shape/forward
  change, NOT version-locked, forwarded verbatim on resume (where `_current_progress_remaining` reflects
  the resumed step, so the anneal continues from the right point). Registered ONLY when
  `--exploiter-temp-start` is set → an off run makes no push (byte-identical, opponent plays at the fixed
  `--stable-opponent-temp`). Composes with `--exploiter-keep-bots` (the from-scratch specialist recipe:
  a bot floor + a temp-ramped strong target). Tests: `exploiter_temp_callback_test.py` (schedule +
  push/change-guard), `wrappers_test.py::test_set_exploiter_temperature_*`.
  - **Two modes (`--exploiter-temp-mode {fixed,ratchet}`, default `fixed`).** `fixed` = the linear
    time schedule above. **`ratchet` = DYNAMIC, win-rate-driven, one-way** (`ExploiterTempRatchetCallback`):
    the fixed schedule has to GUESS the right starting temperature (empirically ai_v7_06's fixed 2.0 start
    was too weak — a 1983-ELO target flattened by temp ~2 is still a wall for a from-scratch net, so half
    the games yielded ~no advantage signal). Instead, start the target near-trivial
    (`--exploiter-temp-start` HIGH, e.g. 5.0) and ratchet the temp DOWN (`*= --exploiter-temp-ratchet-factor`,
    default 0.9, floored at `--exploiter-temp-end`) only when the trainee's measured **training** WR vs the
    target clears `--exploiter-temp-ratchet-wr` (default 0.55, near the ~0.5 max-advantage-signal zone) over
    a window of `--exploiter-temp-ratchet-games` (default 500) target-games. It **never raises** the temp,
    so a plateauing trainee can't comfort-trap the controller into weakening the opponent (the failure mode
    of a symmetric setpoint controller) — mirroring the one-way stable-opponent mastery flip. The signal is
    the TRAINING WR at the current temp (NOT the greedy eval WR, which reads ~0 forever early): the wrapper
    counts per-episode outcomes vs the target (`_record_exploiter_outcome` / `exploiter_winrate_totals`,
    bot episodes excluded), and the callback diffs the cumulative totals via `env_method` each
    `on_rollout_end`. **Resume-safe:** the ratcheted temp is persisted to `<run>/exploiter_temp_state.json`
    and restored on a launcher restart (else a fresh child resets to the easy `temp_start` and undoes the
    ratcheting; the WR window restarts fresh). Metrics: `train/exploiter_temp` + `train/exploiter_target_wr`
    (hovers near the threshold) + `train/exploiter_temp_ratchets`. Requires `--exploiter-temp-start >
    --exploiter-temp-end`. Tests: `exploiter_temp_callback_test.py` (`_decide` one-way/floor + windowed
    control loop + resume round-trip), `wrappers_test.py::test_exploiter_winrate_totals_*`.
- **Pool-ladder curriculum (`gen3_exploiter_pool_ladder_v1`, `--exploiter-ladder`,
  `exploiter_ladder.py`).** The SECOND difficulty axis, and the complement of the temperature
  curriculum above: that one keeps ONE opponent and makes it play NOISILY; this keeps play honest and
  swaps in a genuinely **WEAKER** opponent — an earlier frozen snapshot — promoting up a ladder that
  ENDS at the `--exploiter` target. Motivation is the same C1 hypothesis (a full-strength near-twin
  from step 0 makes nearly every episode a loss, and PPO's advantage is a *within-batch* contrast, so
  a batch of uniform losses says almost nothing about WHICH decision was bad) attacked along strength
  rather than stochasticity. The two are orthogonal knobs and compose.
  - **Two input forms.** An ORDERED comma-separated list of checkpoint specs in the
    `--stable-opponents` grammar (`path[@step][:label]`), weakest first — the order is the user's and
    is not re-sorted; or **`auto:<run_dir>`**, which draws `--exploiter-ladder-rungs` (default 4)
    **evenly-ELO-spaced** snapshots from that run's `snapshot_ladder/ladder.json`. 🚨 The auto draw
    orders by **ELO, not by step** — training is not monotone in strength (measured in
    `ai_v9_27_extremedial_probe_0823`: 42.0M rates 1888.6, the WEAKEST of its 20 snapshots, while
    45.0M rates 2087.4), so "the earliest N snapshots" would build a curriculum that is not a
    curriculum. The `--exploiter` target is ALWAYS appended as the terminal rung (deduped when the
    list already ends there), so the default auto ladder is 5 rungs, and a ladder that resolves to
    the target alone is refused rather than silently being the no-ladder run.
  - **The gate.** Promote one rung when the trainee's **training** WR vs the **CURRENT rung** ≥
    `--exploiter-ladder-gate` (default 0.55, the `--exploiter-temp-ratchet-wr` value and the same
    reasoning) over a completed window of `--exploiter-ladder-window` games (default 500, the same
    disjoint-window semantics as `--exploiter-temp-ratchet-games`). **No demotion, terminal rung
    sticky** — the same one-way property, for the same anti-comfort-trap reason.
  - **The WR is per-rung by construction, not by convention.** The wrapper carries a rung index +
    its own `(games, wins)` pair (`exploiter_rung_totals`), zeroed in the same operation that swaps
    the rung in; the callback DROPS worker rows whose index isn't the live one (a worker that hasn't
    reset since the push). So a promotion window can never pool games played against two different
    opponents. Bot episodes under `--exploiter-keep-bots` were already excluded (only episodes whose
    opponent IS the exploiter player count), so **the bots keep their independent per-episode share
    at every rung** — the ladder changes who the non-bot opponent is, never how often it appears.
  - **The swap rides the established opponent mechanism.** `env_method("set_exploiter_rung", index,
    zip, config)` — the `set_self_play_target` / `set_exploiter_temperature` idiom, change-guarded so
    a steady rung costs no IPC. The worker DEFERS it to the next `reset()` (an opponent's brain must
    not be replaced mid-battle, and the episode in flight must be scored against the rung it was
    actually played against), then assigns into the persistent `RLPlayer` exactly as
    `_ensure_pool_model` does for a self-play snapshot. The loader is INJECTED from `env_factory` (a
    `(zip, config) -> model` closure owning the arch gate, device and `--compile-opponents` policy),
    so rungs compile like any frozen opponent and the wrapper stays free of model-loading policy.
    **The swap changes WEIGHTS ONLY** — the target's pinned team (fold-back), its temperature, and
    the bot fraction are untouched, so the curriculum varies exactly one thing.
  - **Resume-safe.** Live rung + per-rung counts + the promotion log (step, labels, WR, games) go to
    `<run>/exploiter_ladder_state.json` (atomic, on every promotion + every 20 rollouts) and are
    restored on a launcher restart — **by LABEL first**, so an edited ladder resumes at the same
    OPPONENT rather than at whatever now sits at that index; corrupt/partial state fails soft to rung
    0. Without this a 3-hourly restart would silently drop the trainee to rung 0 and the curriculum
    would never finish. Metrics: `train/exploiter_rung`, `train/exploiter_rung_wr`,
    `train/exploiter_ladder_promotions`.
  - **Training-only** (no weight-shape/forward change → never versioned, no `flag_registry` entry;
    it lands in `cli_args`/`metadata.json` like every train-loop knob) and **registered only when
    `--exploiter-ladder` is given**, so an off run adds no callback, makes no `env_method` call, and
    builds its wrapper with `exploiter_rung_loader=None` (every rung branch inert). Rungs are
    resolved + arch-gated + load-validated in phase 2 (`matchup_setup`), so a bad rung is a
    `FATAL_CONFIG` at startup, not a crash in every env worker. Tests:
    `exploiter_ladder_test.py` (52: the ELO draw, both input forms + every malformed one, the gate
    incl. never-demote/terminal-sticky/live-rung-only, the state artifact + the launcher-restart
    resume, the deferred swap seam, and the OFF byte-identity half).
- **The target AUTO-registers for eval** (opponent-parity Proposal A,
  `fixed_opponent_pool.register_exploiter_for_eval`): `--exploiter` alone now produces the verdict
  metric `eval/win_rate_vs_ext_<target>` — the resolved target entry is appended to the eval-side
  fixed-opponent list, DEDUP-guarded (same resolved zip or colliding label → unchanged), so the
  historical `--exploiter X --stable-opponents X` recipe is byte-identical. Eval-only by
  construction (exploiter mode excludes `--self-play`, so the appended entry never joins the
  training mix). And per the fold-back contract above, a SPECIALIST target (its run pinned
  `--trainee-team`) is faced — and eval-measured — piloting **its own pinned team**
  (`exploiter_team` in the wrapper; the startup `[EXPLOITER]` line annotates the pin).
- **Usage:** `--exploiter <target> --model <target's checkpoint>` — init the exploiter from a strong
  checkpoint so it has a baseline to exploit from (the AlphaStar exploiter init). The verdict
  metric vs the target is automatic (above); an explicit `--stable-opponents <same target>` is
  harmless (dedup). The run dir defaults to a readable `models/exploiter_vs_<target>/` (not a
  date-stamp); override with `--run-name <name>`. Tests: `wrappers_test.py::test_exploiter_*`
  (sole-opponent + off-unchanged) + `test_pinned_*` (per-opponent teams),
  `fixed_opponent_pool_test.py::test_exploiter_*registration*`.

### Consensus warm-start (`--warmstart-consensus`, `warmstart.py`) — EXPLOITER-ONLY

`gen3_exploiter_consensus_warmstart_v1` — a low-bias INIT for a NEW exploiter: BEFORE training, build a
competent, archetype-NEUTRAL warm start by **disagreement-gated CONSENSUS distillation** of N mature
teacher exploiters into `--model` (the generalist init), then init the exploiter from it. The
`build_consensus_target` math (pure, in `warmstart.py`): `consensus` = mean of the N teachers' masked
action distributions; `d` = mean pairwise **Jensen-Shannon disagreement**; a quantile-normalized gate
`g∈[0,1]` sets a per-state temperature `T = 1 + (tmax−1)·g`; `target = softmax(log consensus / T)` over
legal actions — **SHARP where the teachers AGREE** (universal decisions the new exploiter just inherits)
and **FLAT where they DISAGREE** (archetype forks left high-entropy → the new exploiter specializes
FREELY, unbiased). BC also carries a KL anchor toward the student's OWN distribution (`anchor_coef`) so
the warm start RETAINS the generalist's competence. Distills in **function space** (teacher outputs) —
weight-averaging FAILED (from-scratch exploiters live in different loss basins → the average collapsed;
`tmp/average_exploiters.py`).

**Why EXPLOITER-ONLY (guarded, `parser.error` without `--exploiter`):** this SEEDS a new model with the
consensus + freedom to diverge. It must NOT touch generalist training, whose objective is the OPPOSITE —
absorb the DIVERGENT per-team specializations (that is `--distill-teacher`, one teacher per team-masked
state). Distilling the consensus into the generalist would sharpen agreement and blur divergence,
erasing the specialization it is trying to learn (and the generalist already ≈ the consensus → circular).
`--self-play` is excluded automatically (exploiter mode already forbids it).

**Integration.** `train_rl_agent` builds it ONCE into `<run>/warmstart/warmstart_consensus.zip` (via
`run_consensus_warmstart`, live over the local bridge) right before the model load, then re-points
`--model` at it. **Idempotent under launcher restarts:** skipped entirely once ANY training checkpoint
exists (the normal resume path continues the trained state); the warm start is arch-identical to `--model`
(its `model_config.json` is copied), so the resume-immutable checks stay valid. Standalone:
`python -m agents.training.warmstart --student <run> --teachers <run,...> --out <dir>`. OFF (flag unset) =
byte-identical. Knobs: `--warmstart-battles` (200), `--warmstart-bc-steps` (4000). Tests:
`warmstart_test.py` (the pure consensus/JS/gate/temperature math: identical→0 disagreement, sums-to-1 over
legal, mask respected, sharpens-agreement/flattens-disagreement, `tmax=1` recovers plain consensus).

## Team-side PFSP (`--team-pfsp`, `team_pfsp_callback.py`)

The TEAM-axis complement to the opponent-side `--pfsp-scale`: bias the TRAINEE's team sampling toward
the pool teams it is weakest on, so training spends gradient where the win-rate says there's headroom
instead of uniformly over ~700 pool teams (the documented "uniform team sampling = headroom" gap).
Four modes: **`off`** (default → byte-identical), **`measure`** (TRACK + persist the per-team
self-play win-rate WITHOUT biasing sampling — pure observability), **`var`** (measure + bias,
symmetric variance), **`onesided`** (measure + bias, losing side held at MAX).

- **Variance weighting + cap + floor.** For pool team `i` the weight is `raw_i = --team-pfsp-floor +
  w(p_i)` where `p_i` is the team's self-play win-rate EMA (seed 0.5 → an unmeasured team gets the
  MAX weight → explored), then capped `w_i = min(raw_i, --team-pfsp-cap·mean(raw))` (no team is
  sampled more than `cap`× the uniform share — the over-representation bound). **`var`**: `w(p) =
  p·(1−p)` — peaks at 50% and decays to the floor at BOTH extremes, so it self-ignores both the teams
  we crush AND the truly-lost teams. **`onesided`** (owner-requested, the z_arch/FiLM companion):
  `w(p) = 0.25 for p < 0.5, else p·(1−p)` (continuous at 0.5) — every sub-50% team stays MAXIMALLY
  sampled and only mastery retires a team, because under the conditioning hypothesis the weak-team
  tail is exactly the learnable headroom (the amortization gap): "truly lost" is the claim under
  test, not a sampling prior to bake in. The floor keeps nothing fully starved either way.
  `compute_team_pfsp_weights` is the pure, unit-tested math.
- **Team-blocked episodes (`--team-block-episodes`, default 1 = off, byte-identical).** Each env
  holds its drawn TRAINEE team for N consecutive episodes before redrawing
  (`Gen3Teambuilder.set_block_episodes`; the WHOLE draw is held — bias branch, PFSP weights,
  tracking index — so weights apply at redraw and outcomes attribute to the blocked team for the
  whole block; each SubprocVecEnv worker unpickles its own builder copy ⇒ blocks are per-env). The
  per-team gradient-DENSITY counter to the sample starvation the retired FiLM group measured
  (`film/noise_scale` ran ≈ 8–9× the batch before the v78 zarch deletion took that metric with it;
  the DENSITY argument stands on its own): per-episode redraw gives ~700 teams × ~4 episodes
  (~140 decisions) per rollout;
  at ~64 (≈ `n_steps`/ep_len — the phase-transition value) each env carries ONE team per rollout at
  ~2k decisions (~15× density) AND the block spans an update boundary, so the env replays the team
  right after its gradient landed (the mini-exploiter learn-and-retest loop — the piece of the
  exploiter regime per-episode redraw never provides). Acceptance: the fixed-matchup ablation
  probe's intact-vs-ablated gap widening. Trainee side
  only (opponent draws stay per-episode); training-only, NOT version-locked, resume-forwarded.
- **Self-play only, pool teams only.** The per-team win-rate is measured ONLY on self-play POOL battles
  (bots wash the signal out — we win ~0.99 vs bots): `MaskableAgentWrapper.step` records the outcome to
  the trainee's `Gen3Teambuilder` (`self.env.agent1._team`) only when `self.opponent is
  self._pool_player`. A bias/distill-pinned team (the `--distill-team-bias` branch) yields
  `_last_pool_idx=None` → its battle is never tracked (those teams get fixed exposure via the bias, not
  the win-rate weighting).
- **Centralized aggregation (NOT per-worker — ~700 teams makes a single worker's counts too sparse; NOT
  info-dict threading — that breaks under `--async-rollout`).** Each worker's teambuilder accumulates
  LOCAL windowed `(wins, games)` per pool team; `TeamPFSPCallback` every `update_every` (3) rollouts
  PULLs them from all workers via `env_method("drain_team_pfsp_counts")` (drain-zeroes each window), SUMs
  by pool index, EMA-smooths a global per-team win-rate, computes the capped weights, and PUSHes them
  back via `env_method("set_team_pfsp_weights", w)` → the teambuilder samples with
  `random.choices(weights=…)`.
- **Auditability + GIGO guard.** Each pool team carries a `team_sha` fingerprint
  (`sha1(team_str.strip())[:10]` — the SAME convention as `matchup_spec.pin_sha` / the archetype
  artifact, so a key JOINS every provenance record). The callback pulls them ONCE
  (`env_method("get_team_pfsp_keys")`) and verifies the per-INDEX team identity is IDENTICAL across
  every worker (**same pool SIZE ≠ same pool ORDER** — a diverged order would silently mis-attribute
  win-rates, which the cheap per-cycle size-only belt can't catch), then logs the weakest measured
  teams by `sha@win-rate` so the weighting is inspectable (which teams/archetypes the budget
  concentrates on), not an anonymous min/max scalar. Metrics
  `team_pfsp/{min_wr,max_wr,n_measured,weight_spread}`.
- **Persisted artifacts (both `measure` and `var`) → offline "which exploiter next".** Each update the
  callback writes to the run dir: `team_winrates.json` (the latest snapshot — per-team `{sha, win_rate,
  games, archetype}` sorted WEAKEST-FIRST, atomic-replaced; the weakest teams = candidate exploiter
  targets, and `archetype` is joined from `gen3_team_archetypes.json` via `team_sha` so it reads
  "weakest = stall-class") and an appended `team_winrates_history.jsonl` row `{step, wr:{sha:wr}}` (so
  the per-team win-rate is trackable OVER TIME offline — trends + noise, not just the latest). `measure`
  gives this signal on ANY self-play run without changing the team distribution.
- **Training-only, not version-locked.** Threaded into the TRAINEE teambuilder only (both the
  `matchup.trainee_teams.build` and the distill `Gen3Teambuilder` paths); the opponent builder is
  untouched. Registered ONLY when `--team-pfsp != off` (off → no callback, no `env_method`, exact-legacy
  `random.choice` → byte-identical); `var` pushes weights, `measure` never does. Forward it like
  `--pfsp-scale` on resume; no `model_config`/`ModelVersion` entry.
- **Tests.** `utils/teambuilder_test.py` (off==uniform RNG-identical, weighted sampling, record/drain,
  the cap+floor weight math), `team_pfsp_callback_test.py` (cross-worker aggregation, the pool-size GIGO
  guard, the `update_every` throttle, None-worker filtering).

## Per-team win-rate tracking (`--team-wr-tracking`, DEFAULT ON, `team_winrate_callback.py`)

A first-class running record of how the trainee does **piloting each team**, keyed by `team_sha`.
The training loop always knew which team an episode piloted and how it ended; nothing kept the
record, so the three flywheel consumers that need it — the deficit thermostat, **headroom
capture's denominator**, and slice-curation evidence — each had to be a scratch script. This is
**instrumentation only: no prioritization consumer ships with it**, by design.

⚠️ **THE CONFOUND, and it is written into the artifact rather than only into this file.** A raw
per-team win rate conflates **PILOT COMPETENCE with TEAM STRENGTH** (the ai_v8 team-PFSP finding:
team-PFSP win rate was confounded by team strength). "Our win rate with team T is low" does not
mean "we pilot T badly". Anything that spends budget on this signal must first normalize against a
**team-strength baseline** — e.g. T's pool-average win rate under a reference pilot. The artifact
carries that sentence in its `notes` field so it travels with the numbers, plus the reminder to
read `by_class`: a pre-self-play curriculum phase is ~all `bot` episodes, where every team reads
~0.99.

- **The seam is an `env_method` PULL, not an info-dict thread — and that is the async decision.**
  Each worker's `Gen3Teambuilder` accumulates a windowed per-team, per-opponent-class count
  (`record_team_wr_outcome`), fed by `MaskableAgentWrapper._maybe_record_team_wr` at the terminal
  step beside the existing `win_outcome` capture; `TeamWinRateCallback._on_rollout_end` drains
  every worker (`drain_team_wr_counts`) at a rollout boundary. **This works identically under
  `SubprocVecEnv` and `--async-rollout`** because `AsyncSubprocVecEnv.env_method` is drain-safe (it
  stashes in-flight step results before the barrier RPC), whereas an info-dict route would have to
  know which buffer ROW a terminal landed on — knowledge only the async collector has, which is why
  the team-PFSP precedent avoided that route for the same reason.
  `test_aggregation_reads_env_method_and_never_the_info_dicts` pins it by feeding the callback a
  deliberately contradictory `self.locals["infos"]` and asserting the result ignores it.
- **The default uniform draw stays RNG-identical.** With `--team-pfsp off` (the default)
  `_draw_team` is `random.choice(self.packed_teams)`, which returns the team and not its index. The
  index is recovered by a **reverse dict lookup** (`_pool_index_by_packed`, built at construction),
  never by re-drawing it — so the byte-identity baseline is untouched
  (`test_default_uniform_draw_is_rng_identical_with_tracking`). Side effect worth knowing:
  `--team-block-episodes` caches `_last_pool_idx` for the block, which on the default path used to
  be `None`, so a blocked default run can now attribute its whole block to the team it held.
- **Stratified by opponent class** (`MaskableAgentWrapper.OPP_CLASS_*` / `OPP_CLASS_NAMES`), so a
  rate can always be split back out by who it was measured against. A bias/distill-pinned yield
  (`_last_pool_idx is None`) is never attributed to a pool team.
- **NO TensorBoard emission — owner rule** (design_flywheel_tick_tock.md §6b: per-team series
  would be noisy spam; "let's not spam it if the data won't be nice"). Pinned by
  `test_NOTHING_is_emitted_to_tensorboard` — a future "just one scalar" regression fails there.
- **The table rides `metadata.json`** as the top-level `team_win_rates` block (written via
  `snapshot.record_team_win_rates`, carried forward across checkpoints by `save_model_snapshot`
  exactly like `latest_eval` — one artifact per run holding per-team AND per-opponent records
  side by side):
  `{step, updated_at, n_teams_seen, n_games, opp_classes, notes, teams: {sha: {n, wins, wr,
  archetype, by_class}}}`. **RAW COUNTS, not a smoothed rate** — headroom capture needs a
  denominator, which is exactly what team-PFSP's EMA throws away. `archetype` is joined via
  `load_team_archetypes` on the same `team_sha`. **Restart-safe by load-and-continue**, and keyed
  by sha rather than pool index so a pool that was reordered or resized between runs still joins
  (`test_reload_is_keyed_by_sha_so_a_REORDERED_pool_still_joins`). A corrupt file starts fresh.
- **GIGO guard, throwing.** Counts arrive per pool INDEX and are keyed to a sha by the worker's own
  key list; if any worker's list disagrees the callback **raises**. Same pool SIZE is not the same
  pool ORDER, and a diverged order would attribute every per-team number to the wrong team.
- **Deliberately NOT coupled to `--team-pfsp`, and the overlap is real enough to state.**
  `--team-pfsp measure` also tracks a per-team win rate and also writes an archetype-joined
  `team_winrates.json`. Four differences make it unusable as this instrument: it is **off by
  default**; it measures **self-play POOL battles only** (bots wash out its weighting signal), so a
  pre-self-play generation records nothing; it keys per pool **INDEX** with the sha only for an
  audit line; and it stores an **EMA rate**, not counts. The two share the builder's "which team
  did I just yield" draw index (`_last_pool_idx`) and **nothing else** — separate counter tables,
  separate accessors, separate artifacts, deliberately differently-named files
  (`team_win_rates.json` vs `team_winrates.json`). If the owner later wants one tracker,
  consolidating team-PFSP's `measure` mode onto this table is the direction, not the reverse.
- **Flag class: training-runtime, like `--team-pfsp`.** Never reaches the extractor, scales no loss,
  changes no weight shape ⇒ **no `ARCH_SIGNATURE` bump, not in `model_config.json`/`ModelVersion`,
  not in `check_compatible`, and deliberately not in `agents/model/flag_registry.py`** (that
  registry's scope is extractor architecture toggles — the `--td-aux-coef` /
  `--intent-label-bot-weight` precedent, which are recorded on `ModelVersion` only because they
  scale a loss and want flagless-resume inheritance; this one does neither). Forwarded verbatim by
  the launcher like any non-launcher flag. `--no-team-wr-tracking` opts out (no callback, no
  `env_method`, the wrapper hook returns immediately).
- **Verified end to end** by a `--debug --steps 4000` CPU smoke: **96 teams / 103 games** recorded,
  archetypes joined (`semi_stall`, `balance`, `hyper_offense`), `by_class` correctly all-`bot` on a
  fresh run, the `notes` caveat present, and `teams/n_teams_seen` / `teams/n_games` on the TB
  event file. The four `wr_*` scalars need a team past the 10-game floor, which a 719-team uniform
  pool does not reach in 4000 steps — a `--trainee-team`-pinned smoke exercises them, and all six
  keys are pinned numerically by `test_sparse_tb_keys_are_summaries_not_one_series_per_team`.
- **Tests.** `team_winrate_callback_test.py` (29): the `team_sha` convention agreement with
  `team_archetypes.team_sha` incl. strip-normalization, the RNG-identity claim, the builder
  accumulator + drain-zeroing + bias-yield exclusion + PFSP-table independence, the wrapper hook and
  its off path, the callback's running math across workers AND windows, per-class restriction,
  `min_games`, the `update_every` throttle, None-worker filtering, the throwing order guard, the TB
  key set with hand-computed values, the artifact shape + confound note, the archetype join and its
  missing-artifact fallback, restart reload incl. the reordered-pool case and a corrupt file, and
  the `env_method`-not-infos seam claim. Plus `utils/teambuilder_test.py` (the off-path RNG identity
  now also asserts the index resolves while PFSP still ignores it).

## ELO / skill rating (`elo.py`, `bot_elo_calibration.py`, `main.elo`)

Once training is mostly self-play **pool play**, win-rate stops being legible: the promotion
gate only promotes when `win_rate_vs_pool > promote_threshold` and the pool is a *sliding window
of recent selves*, so `win_rate_vs_pool` is a treadmill pinned near 50-65% **by construction** —
it cannot trend up however much the model improves; `win_rate_vs_bots` saturates near 100%. The
ELO subsystem gives a single **absolute** number that genuinely rises with skill, anchored to the
fixed bots.

- **No new battles.** Every eval cycle already plays the trainee (greedy) vs all 9 bots and vs
  up to `--n-sentinels` (default 5) pool sentinels, `EVAL_GAMES` each — a full tournament-matrix
  row. `record_elo`
  (`eval_callback.py`, shared by BOTH callbacks) appends that row to an **append-only
  `<run>/eval_results.jsonl`** (`snapshot.append_eval_result_row`) — the canonical, restart-safe
  source of truth, distinct from the overwritten `metadata.json:latest_eval`.
- **The model = anchored Bradley-Terry** (`elo.fit_elo`): `P(i beats j)=σ((Rᵢ−Rⱼ)·ln10/400)`,
  fit in **batch** by penalized MLE (weak Gaussian prior keeps 100-0 records finite), SE from the
  inverse Hessian. Each bot is a player `bot:<name>`, each snapshot `snap:<step>` — a snapshot is
  the SAME player whether it appears as a cycle's trainee or later as a sentinel (unified by
  step), which links the whole ladder. Batch-BT (not online K-factor Elo) is drift-free and
  re-runnable; the fit is a few Newton steps over ~tens of players. **Not Glicko-2**: its
  volatility models skill drift, but snapshots are *frozen* — the drift is the *sequence* of
  snapshots (the ELO-vs-step curve); the per-player uncertainty (Glicko's valuable part) is the
  Hessian SE.
- **Anchor = a precomputed bot-vs-bot round-robin.** `python -m agents.training.bot_elo_calibration`
  plays all 36 bot pairs toward `--target-games` (default 5000) **in-process via the bridge — no
  server** (safe alongside a live run; it does use CPU — throttle with `--concurrency`), fits BT
  (`elo.fit_pairwise`, `random` pinned at `base`=1000), and writes the anchor. **Artifact split:**
  the immutable bot anchor (ratings, SEs, the 9×9 win-matrix, a non-transitivity `fit_quality`) is
  the only runtime input, so it lives in **`data/gen3_bot_elo_anchors.json`**; the raw game-count
  **store** (resume state) and the **heatmap** PNG are calibration provenance/viz, so they live with
  the ELO design work under **`designs/ai_v5/elo_calibration/`** (override with `--games-store` /
  `--heatmap`). The
  live/offline fits then **pin all 9 bots** to those high-confidence ratings and fit only
  snapshots — so a snapshot is well-grounded from its first cycle, and because the anchor is
  identical across runs, **snapshot ELOs are comparable run-to-run**. **Regenerate when bot logic
  changes** (the json records `git_hash` + date). Graceful fallback when the file is absent:
  `random` pinned at `base`, other bots float (rank/trend preserved, scale not cross-run-stable).
  Bots build once and are reused across pairs (`reset_battles` between) — building warms the data
  singletons (~4.5 s each), so per-pair rebuilds dominated cost; the full 5000-game job is a
  many-hour, run-overnight one-time cost.
- **The RAW matrix, at higher resolution.** `python -m agents.training.bot_matchup_matrix`
  accumulates the same round-robin (same bots, same team sampling, same bridge driver — it calls
  the calibration's own `_build_bot`/`_play_chunk`) as **raw per-pair `wins_a`/`wins_b`/`draws`/`n`**
  toward 10 000 games/pair in resumable chunks → `data/gen3_bot_matchups.json`. Draws stay
  separate and it **never writes the anchor** (regenerating that is an owner decision).
- **Live (each eval cycle).** `record_elo` refits and records `eval/elo` + `eval/elo_ci` (95% CI
  half-width) to TensorBoard + the TUI dict, and stamps `elo`/`elo_ci` into `metadata.json:
  latest_eval` (so the resume-republish path shows ELO immediately after a restart — the saved
  headline is authoritative; and if a resumed checkpoint predates the `elo` field,
  `replay_last_eval_to_tui` **fits** the saved block's win rates via `elo.fit_from_block` to recover
  both the headline and each opponent's ELO, so the badge never blanks for a full cadence). The
  launcher
  surfaces a `🏅 ELO 1532 ±40` badge (`app.py::_elo_badge`) + an `elo` column in the eval panel:
  the model's rating on the `all` row, and each opponent's anchored ELO on its row
  (`_record_opponent_elos` records `eval/elo_vs_<bot>` + positional `eval/elo_vs_sentinel_<i>` to
  the TUI). The live number is the best estimate from data SO FAR (batch-BT is global → early
  points retro-adjust; the single-cycle per-sentinel ELO is rough — only the trainee is
  bot-anchored each cycle); the offline CLI re-fits canonically over the full per-snapshot history.
- **Offline (`python -m main.elo <run_dir>`).** Loads results (`--source auto|log|tb|meta` —
  `tb` **backfills an already-running run straight from TensorBoard, zero training change**), fits,
  and prints a ranked ladder + writes `elo_ratings.json` + an Elo-vs-step `elo_curve.png` (CI band
  + bot anchor lines). `--out` defaults to `<run>/elo/`; point elsewhere to analyze a LIVE run
  without writing into it.
- **Caveat (acceptable, noted in code):** by default the trainee is greedy but the sentinels are
  stochastic@temp, so a snapshot's rating blends greedy strength (when it's the cycle's trainee)
  with stochastic strength (when it's a later sentinel) — a roughly uniform shift that preserves the
  trend, but it does mean the same snapshot is scored in two regimes. **`--eval-sentinel-greedy`
  removes this** — sentinels play greedy too, so every snapshot is scored greedy in both roles and
  the ELO ladder is internally consistent (at the cost of a one-time scale shift vs prior cycles;
  the bot-anchored scale is preserved since trainee-vs-bot records are unchanged). Tests:
  `elo_test.py` (synthetic-ladder recovery, anchoring, perfect-score, loaders, `fit_pairwise`).

### Frozen-snapshot ELO ladder — the dense, pay-once resolution (`snapshot_ladder.py`)

The live ELO above is RESOLUTION-limited at the frontier: the fixed bots have SATURATED (we sit
~400 Elo above them, out on the flat tail of the logistic — a 10-Elo trainee move shifts its
bot-WR by ~0.5% against a 1.9%/200-game noise floor), so the bots pin the absolute LEVEL but the
fine ordering rides on the sparse, near-50% sentinel edges (±15 Elo CIs). Fix from the other side:
a promoted snapshot is FROZEN, so snapshot-A-vs-snapshot-B is a STATIONARY Bernoulli — measure it
ONCE (dense round-robin) and it is permanent. On each promotion, `SelfPlayCallback._spawn_snapshot_ladder_update`
fires a **DETACHED** `python -m agents.training.snapshot_ladder <run> --promote <step>` subprocess
(bridge, off the training path) that plays the new frozen node vs the current frozen pool
(`--snapshot-ladder-games`, default 100/pair; 0 disables) and appends to
`<run>/snapshot_ladder/games.jsonl` (**forever, race-safe line appends; a measured pair is NEVER
replayed**). `fit_ladder` combines that dense frozen-vs-frozen matrix with each snapshot's
historical bot edges (from `eval_results.jsonl` — the anchor connection) → an anchored BT fit
(`fit_pairwise`, bots pinned) written to `<run>/snapshot_ladder/ladder.json` (the sidecar metric);
`_record_ladder_elo` surfaces the latest promoted node's rating as `eval/ladder_elo` (+`_ci`) on
TB/TUI — the high-resolution counterpart to the saturated `eval/elo`. Snapshots load via
`load_foreign_opponent` (their own saved config → PopArt/toggles honored, `check_compatible`
skipped). `--backfill` pays the one-time back tax over the whole current pool (idempotent — skips
measured pairs); `--fit-only` refits without playing. `ladder.json.fit_quality.mean_abs_err`
QUANTIFIES non-transitivity (a scalar Elo is lossy if the pool is rock-paper-scissors — the dense
matrix at least measures it). Tests: `snapshot_ladder_test.py` (store accumulation/symmetry,
measure-once contract, fit-recovers-ordering, sidecar read).

### Hodge decomposition — the SPINE and the WIDTH (`hodge.py`)

A scalar rating is a **transitive** model by construction, so a BT fit cannot see a cycle: two
snapshots with identical ELO can have a lopsided head-to-head. `ladder.json`'s
`fit_quality.mean_abs_err` NOTICES the residue but reports it as one unitless number with **no
noise floor**, which cannot answer the only question that matters — *is the non-transitivity real,
or is it binomial noise on 100-game edges?* HodgeRank answers it by splitting the measured flows

```
Y_ij = logit(p_ij)   =   (r_i − r_j)   +   R_ij        w_ij = n_ij·p_ij·(1−p_ij)
                          ───────────       ────
                          TRANSITIVE        CYCLIC     (Fisher info of a logit = the weight)
```

where `r` is the weighted-least-squares (graph-Laplacian) solve — BT's quadratic cousin, reported
BESIDE the BT ratings so the estimators' disagreement is visible. The split is **exactly
w-orthogonal** (`Σw·Y² = Σw·(rᵢ−rⱼ)² + Σw·R²`, pinned by a test), so spine and width cannot be
traded against each other by refitting. Units: 1 logit = 400/ln10 ≈ **173.72 ELO**.

**The noise floor is the whole instrument.** Two nulls, both reported: an exact-mean analytic one
(`E[Σw·R²] = Σ(1 − w_e·Reff_e)` — per-edge effective resistance, i.e. Foster's `E−V+C` spread over
edges) and a **parametric bootstrap** that simulates games from the fitted transitive model and
re-runs the whole pipeline. `width_rms_excess = √(raw² − null²)` is the width that survives, with a
p-value for "width > noise".

**Width SCOPE — a pendant edge's residual is identically zero.** A player with one measured
opponent has that single edge as its whole normal equation, so counting it only inflates Σw and
deflates the RMS. Width statistics therefore default to the **triangle-supported subgraph**; the
spine is always fit over every edge. `n_triangles` + `n_width_edges` ride with every read.

- **Offline — THE instrument.** `python -m main.elo <run>` prints the block and writes it into
  `elo/elo_ratings.json` under `hodge` (flags: `--no-hodge`, `--hodge-bootstrap N`, `--hodge-seed`,
  `--hodge-with-bot-rr`). The graph is exactly `fit_ladder`'s: the dense frozen matrix
  (`snapshot_ladder/games.jsonl`) + every cycle's bot/sentinel edges. The static bot round-robin is
  **excluded by default** — its 36 edges carry ~2700 games each against a ladder edge's 100, so on
  the Fisher weighting they would carry ~99% of Σw and the "width" would become a property of the
  immutable shared anchor rather than of this run. `main.endofrun`'s §1 block carries the same read
  for the run and its `--ref`.
- **Live — two scalars beside `eval/elo`**, recorded by `record_elo` on the same cadence:
  `eval/hodge_width_elo` (excess width, ELO) and `eval/hodge_cyclic_fraction` (null-adjusted).
  Both also ride in the `eval_results.jsonl` row's `hodge` block for offline replotting, and a
  cycle whose graph had **no testable triangle** writes `recorded: false` + a reason there and
  records NOTHING to TB (never 0-as-a-stand-in, never NaN — a missing point and a suppressed one
  look identical in TensorBoard, and only one is a fact about the graph).

⚠️ **THE STAR-GRAPH SUBTLETY — read this before quoting a live width.** A cycle's own new games are
a **star** (trainee vs each opponent). A star is a tree; a tree has no cycles; so a width computed
on the cycle's games alone is *identically zero* — a fake instrument that would read "no
non-transitivity" forever. The triangles come from joining the trainee's edges to the **static
bot-vs-bot round-robin** in `data/gen3_bot_elo_anchors.json` (which does ship the raw 9×9
`win_matrix` + per-pair `pair_games`, so those are MEASURED edges; a future anchor carrying only
`ratings` falls back to edges reconstructed from them, which are transitive by construction and act
purely as a pinning prior — flagged in `caveats` when it happens). So the live metric means exactly
**"the trainee's matchup deviation from its own rating, over trainee×bot×bot triangles"** — nothing
about the pool's width. Sentinel edges are in the FIT (real spine information) but on no triangle,
so they are excluded from the width scope. And the live read is **weak by construction**: ~100-game
edges put the noise floor around 35-60 ELO, so only a gross cyclic profile clears it (measured on
gen-15's 12 cycles: p between 0.13 and 0.93, i.e. never significant on its own). **The offline
dense-ladder read is the real instrument; the ELO-reading rules below apply unchanged — never
narrate a mid-run width.**

**First reading (gen-15, `ai_v9_18_gen15_v8rewards_0818`, 21 players / 174 edges / 814 triangles,
300 bootstrap reps):** spine 939 ELO, width raw 58 → null 36 → **excess 46 ELO, p = 0.005**; cyclic
energy 6.3% raw / 3.8% null-adjusted; **3 significant 3-cycles**, all snapshot-vs-snapshot
(16M > 20M > 18M > 16M, curl +217 ELO z=4.3; 8M > 20M > 18M; 8M > 20M > 14M). gen-14 on the same
read (same 21/174/814 shape): spine 765, excess **26 ELO, p = 0.0033**, 2.2% null-adjusted, and **0
individually-significant cycles** — its width is real but diffuse. So both ladders are
overwhelmingly spine (96-98%) and both carry cycle content that is **not sampling noise** — the
first evidence here that the BT gate is a lossy projection by a *measured* amount, and that the
loss is bigger on gen-15. (p ≈ 0.003-0.005 is the bootstrap's floor `1/(B+1)`, not a coincidence:
no null replicate reached the observed width.) Tests: `hodge_test.py`.

#### 🚨 Reading an ELO: `ladder.json`, at matched SNAPSHOT COUNT, never mid-run

**Read `<run>/snapshot_ladder/ladder.json` — not `eval/elo`, not the per-cycle TB scalar.** On
gen-10's completed ladder the two agree at the end (24M: dense 2079 vs sparse 2102) but the dense
CI is **±10 vs ±29**. Precision is the reason to prefer it; it is *not* immune to the drift below.

**A snapshot's rating keeps moving until it stops gaining opponents.** Anchored BT is a GLOBAL
BATCH fit — every added player re-solves every rating — and the movement is a **systematic
downward bias on the newest node**, not noise. Measured over gen-10's 12 successive refits
(`snapshot_ladder/updater.log` records each one):

| snapshot | first fit | final fit (n=12) | drift |
|---|---|---|---|
| 2M | 1790 | 1705 | **−85** |
| 4M | 1945 | 1844 | **−101** |
| 12M | 2089 | 2021 | **−68** |
| 14M | 2088 | 2044 | **−44** |

Mechanism, and the SE is the tell (12M: 25.9 → 18.4 as it fell 2089 → 2021): a fresh snapshot's
only edges are ~90% wins over the bots. A saturated edge says *"≥380 Elo above"* with a likelihood
that is **flat upward**, so the MLE is inflated and under-constrained. The near-50% frozen-vs-frozen
edges are sharply informative, and as they accumulate they pull the chain down onto the anchor.
Dense measurement buys resolution; it does not buy an early answer.

**Therefore, two rules:**

1. **Never narrate a mid-run ELO or a mid-run delta.** The gen-10 12M delta read +108, +82, +73,
   +64 before settling at **+11** — four reported "results", all artifacts. Wait for the run to end.
2. **Cross-run comparison must be at matched snapshot count `n`, not matched step.** Both runs'
   node at n=k carries the same inflation, so it cancels; a live run's newest node against a
   finished run's *final* value does not. Worked example — gen-11 at n=7 vs gen-10's **n=7** fit
   (recoverable from `updater.log`) reads 14M: 2082 vs 2088 = **−6, tied**; against gen-10's n=12
   final it reads **+38**, which is the drift and nothing else.

`n_frozen_pairs_measured` / `n_pairs_possible` in `ladder.json` is the completeness check — a fit
at 21/21 pairs is internally complete but only 7 nodes deep, and depth is what the bias tracks.

## Rollout collection: sync barrier vs `--async-rollout` (`async_vec_env.py`)

The default `SubprocVecEnv.step()` is a **per-step barrier** — the trainer waits for the slowest of
N env workers every step, so a slow battle turn / heavy opponent forward / oversubscription jitter
stalls the whole batch and the GPU policy-forward never overlaps CPU env-stepping. `--async-rollout`
swaps in **`AsyncSubprocVecEnv`** (per-env `send_step`/`poll_ready`/`recv_step` over the pipes +
**drain-safe `env_method`** — the eval callback's `set_self_play_target`/
`opponent_default_stats` fire mid-collection, so the override stashes in-flight step results before
any barrier RPC to avoid a pipe desync) and **`collect_rollouts_async`**, dispatched by
`InstrumentedMaskablePPO.collect_rollouts` when `model._async_rollout` is set.

The collector keeps every worker continuously in-flight, batch-forwards whichever envs are READY
(dynamic batch), and writes each env's transition into **its own buffer column**
(`MaskableDictRolloutBuffer`); collection ends when every column has `n_steps`. It is **exactly
on-policy** — PPO freezes the policy during collection, so this is a *scheduling* change (overlap
forward with stepping, drop the max-latency barrier), NOT an APPO-style algorithm change. Bookkeeping
(`num_timesteps`, GH-#633 timeout bootstrap, `_update_info_buffer`, `_last_*` carry-over, per-column
GAE) mirrors the stock loop exactly. The per-decision **mask rides in the Dict obs**
(`obs["action_mask"]`, = `last_ctx.mask`), so no per-env `env_method` and no wrapper change.

**Measured FPS (bridge, GPU forward, steady-state, heuristic opponents):** +20% at `--n-envs 16`;
**+14% at the production `--n-envs 64` (1489→1695)**; `--async-rollout --n-envs 32` matches `sync@64`
FPS with half the envs (≈half the env/bridge RAM). Off by default (stock `SubprocVecEnv`), ignored
under `--debug`. Caveat: benchmarked with heuristic opponents — re-bench under `--self-play` for the
production-regime number. Full design + benchmark table: `designs/ai_v5/design_async_rollout.md`.

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
| compiled disagrees with eager (> 1e-4) | a faster wrong model is not a win |

Every rejection **uninstalls** the compiled callable before raising, so the process never keeps
running something it just declared unacceptable.

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

## Gradient-balance + value-scale diagnostics (`grad_balance.py`)

The dual-head extractor shares ONE transformer trunk between the policy and value heads
(`src/agents/model/CLAUDE.md`); both losses' gradients compete there. When the value loss
dominates (large / unclipped, big-return scale) it **swamps the trunk** and the policy barely
updates — visible before only *indirectly* as suppressed `train/approx_kl` + `train/clip_fraction`
while `train/explained_variance` races ahead. `InstrumentedMaskablePPO.train()` now measures it
**directly** via the pure helpers in `grad_balance.py` (no SB3 / logging coupling → unit-tested in
`grad_balance_test.py`), recorded once per `train()` call through the standard logger → TensorBoard
**and** the launcher TUI (the new scalars ride the generic `MetricsExporterCallback` →
`ipc.send_metrics` path with zero extra wiring; ordering/labels live in `launcher/format.py`).

- **Gradient balance — every head's *pull* on the shared trunk, on ONE common denominator.** Sampled
  on the first minibatch (graph alive) by **read-only** `autograd.grad` probes (`retain_graph=True`, so
  the real `loss.backward()` is unaffected) against the shared-trunk params. "Shared" =
  `SHARED_TRUNK_PHASES = {embeddings, pokemon_encoder, team_transformer, assembler}` (the allow-list
  is the single source of truth), which **excludes** `cls_pool` (head-private `our_cls`/`their_cls`/
  `value_cls` queries) and both projection heads — only *truly contested* params count. With the
  belief / move / latent / move-latent / win-prob / value-dist auxiliaries there are now **many**
  competitors, not just value-vs-policy, so **every `grad/*_share` is on the SAME total**
  `T = ‖g_pi‖ + ‖g_vf‖ + Σ‖g_aux‖` — the shares are mutually comparable, **sum to ~1**, and any one
  term crowding out the rest is read off directly. (L1-of-norms — an upper-bound proxy, not a variance
  decomposition, since `‖a+b‖ ≠ ‖a‖+‖b‖` — but the same convention for every term.)
  - `grad/policy_share` + `grad/value_share` — the two RL heads' slices of the **whole** pie (ALWAYS
    present). Each is weighted by the live `ent_coef` / `vf_coef`, so `value_share` is a `vf_coef`
    tuning read — but it now *moves with the aux count* (it is value's slice of the full pie), so prefer
    the aux-independent `value_policy_logratio` below for the pure value/policy balance.
  - `grad/aux_share` (only when ≥1 aux is on) — Σ of all the aux shares, the **total non-RL draw** on
    the trunk: one curve for "are the scaffolds collectively crowding out policy/value".
  - `grad/value_policy_logratio` = `log10(‖g_value‖/‖g_policy‖)` — the **AUX-INDEPENDENT** value-vs-policy
    imbalance (a pure ratio of the two RL norms, unchanged by how many auxiliaries are on), *linear &
    non-saturating* (0 = balanced, >0 = value dominates, <0 = policy dominates, e.g. ≈+1.8 at a 66:1
    swamp). The legible gauge for **watching a PopArt / `vf_coef` fix land** — it moves linearly toward 0
    where `value_share` would crawl. `vf_coef` is **fixed per run** (recorded in `model_config.json`,
    FATAL to change on resume — it rescales this very gradient; tune on a fresh run; see
    `src/agents/model/CLAUDE.md` → resume-immutable training hparams).
  - `grad/policy_value_cosine` — scale-invariant (hence `vf_coef`-independent) structural-conflict
    signal: <0 ⟹ the two RL heads pull the trunk in opposing directions.
  - `grad/policy_norm_shared` / `grad/value_norm_shared` — the weighted norms, for absolute context.
  - **Per-aux breakout** (each present only when ITS head is active this minibatch — passed as the
    `aux_terms` dict): `grad/{species_belief, move_belief, move_latent, win_prob, value_dist}_*`,
    each with `_share` (on the common `T`), `_norm_shared`, and `_policy_cosine` (<0 = that aux fights
    the policy). So the species CE, move BCE, SimSiam latent, move-latent grading, win-prob and value-dist
    pulls are **attributable individually** (the old combined `belief_share` lump is gone) — watch each
    sit small (~a few %); a spike with a degrading policy → lower THAT term's coef. `win_prob`/`value_dist`
    are ≈0 under `read_only` (stop-grad), real under `shaping`.
  - **`grad/distill_share`** (`gen3_grad_distill_share_v1`): the exploiter-distillation **policy KL**'s
    own entry in the same dict — the dose meter `design_advantage_gated_distillation.md` §6.2
    dose-matches the G1/G2 arms on (gradient share, not coefficient). Policy KL ONLY, deliberately:
    the value-side distill coefficients are held fixed across those arms, so folding them in would
    compress the very differences the meter reads. When distill is on, the once-per-`train()` sample
    waits for a minibatch with a live distill term (unless the whole rollout holds no teacher-team
    rows, in which case it samples immediately rather than suppressing the probe); distill off → not
    logged, zero cost.
- **Per-edge-family LIVENESS — `edge/<fam>_weight_norm` + `edge/<fam>_grad_norm`**
  (`edge_family_metrics`, sampled once per `train()` right after the backward so `.grad` is still
  populated; parameters only, so the forward — and therefore the CPU opponent path — pays nothing).
  Every family enters as a ZERO-INIT `Linear(cell → 2·n_heads)`, which means **a family that never
  learns anything is bit-identical in the logs to one that works**: both write zero into the
  attention bias and neither says a word. The v79 `h` (pair-history) family shipped into a
  production run with exactly that blindness, and the only recourse would have been a post-hoc
  ablation at run end.
  - `weight_norm` — how far the map moved off its zero init. **Has it learned anything?**
  - `grad_norm` — how hard the loss is pushing it right now. **Does anything want it to?**
  - Read as a PAIR: both ~0 = genuinely dead (the cell carries nothing the loss can use); weight ~0
    with grad > 0 = still climbing off init (the expected early reading); weight > 0 with grad ~0 =
    converged and contributing. Weight norm alone cannot separate the first two.
  - ⚠️ **Neither is an EFFECT SIZE.** Both scale with the cell's input magnitude, so a family with
    larger-magnitude inputs shows a bigger gradient regardless of usefulness — measured at init on
    the gen-12 config, `h` reads the largest `grad_norm` of all 16 families (0.0100 vs d3's 0.0032),
    which says it is alive and being pushed, **not** that it is the most useful. The per-family
    ABLATION audit remains the only thing that measures importance, and these must never be quoted
    in its place.
- **Per-CELL LIVENESS — `cell/<name>_weight_norm` + `cell/<name>_grad_norm`**
  (`cell_family_metrics`, same window, same backward, same parameters-only cost). The identical gap
  one layer over: `SwitchBranchMoveCell`, `PairOutcomeMoveCell`, `PairOutcomeSwitchCell` and
  `ConditionalThreatCell` each enter through a **ZERO-INIT `proj` Linear** — deliberately, so that
  ON-at-init is byte-identical to OFF and any measured effect is something the run LEARNED — which
  means an enabled cell that never learns contributes exactly zero to every action logit and looks
  exactly like one that works. gen-16 turns four of them on at once, in the run meant to decide
  whether the switch-branch channel kills the bait-loop pathology, where **"the behaviour did not
  change" and "the cell never came off zero" must not be the same observation**
  (`designs/research_state/bait_loop_hunt.md` §6 makes this the launch-window check).
  Read the pair exactly as the edge families' — and under the same ⚠️: a parameter magnitude is not
  an effect size. `CELL_FAMILIES` is DECLARED, not duck-typed, so a renamed cell breaks the test
  rather than going quietly unmonitored. Nothing is emitted for a cell that is off — it is absent
  from the extractor, and a zero would read as "enabled but dead", a different claim.
- **Value scale — PopArt prep.** From the full rollout buffer: `train/return_mean` / `train/return_std`
  / `train/return_abs_max` (exactly the `(μ, σ)` + tail an adaptive return normalizer / PopArt's ART
  half tracks) and `train/value_pred_std` (the value head's actual output spread). Watch these to SEE
  the non-stationary value-scale drift (reward annealing / policy improvement) that a static `vf_coef`
  cannot follow. Plus `train/grad_norm` (pre-clip total grad norm, mean over minibatches → grad-clip
  activity).

Cost: **2 partial backward passes on ONE minibatch per `train()` call**, plus **one more per ACTIVE
auxiliary** (species/move/latent/move-latent belief + win-prob + value-dist → up to ~8 total when every
head is on; each is the `aux_terms` dict's per-term `autograd.grad`) — all on the single sampled
minibatch, negligible vs the `n_epochs × n_minibatches` the loop already runs + trivial NumPy stats. The
probe is a **no-op**
(records nothing) when `shared_trunk_parameters` finds no matching modules (a non-Gen3 policy). **Why
it exists:** to prepare for **reducing `vf_coef`** and **adding return normalization (PopArt)** — both
target the value→trunk pressure, which can now be tuned to a number instead of inferred. (The
`+INSTRUMENTATION` markers in `instrumented_ppo.py` flag the added lines; the upstream-drift hash check
is unaffected since it hashes only `sb3_contrib.MaskablePPO.train`.)

## Live capacity telemetry (`--capacity-telemetry`, `capacity_telemetry.py`)

**Three continuous saturation early-warnings that ride the train loop.** They exist because every
previous answer to *"is the network out of capacity?"* here has been an expensive one-shot probe —
a rank sweep, an ablation, an offline battery — each returning a NUMBER at a MOMENT. Saturation is
not a moment; it is a trend, and a trend measured twice is a line through two points. Everything
here is cheap enough to run on every `train()`, so the reading that matters (the SHAPE of the curve
over tens of millions of steps) exists at all.

**Read every one of these as a TREND. None has a meaningful absolute value.**

| scalar | what it answers | alarm shape |
|---|---|---|
| `capacity/canary_loss` | how well a small detached head fits K=4 synthetic obs targets (EMA) | rising |
| `capacity/canary_recovery` | post-reset ÷ pre-reset loss of the target that was last re-seeded | **degrading from reset to reset, at a MATCHED `canary_age`** |
| `capacity/canary_age` | env steps since the last reset | — (the x-axis `recovery` must be read against) |
| `capacity/canary_loss_reset` · `canary_resets` · `canary_steps` | the reset target's own EMA · reset count · updates this `train()` | `canary_steps` = **0** with the flag ON ⇒ the probe is measuring nothing |
| `capacity/halfbatch_cosine` | do two halves of one minibatch agree on the shared trunk? | falling to 0 / **negative** |
| `capacity/halfbatch_grad_norm_ratio` | are the two halves' gradients comparable in size? | ≪1 ⇒ one half dominates; read it before believing a low cosine |
| `capacity/feature_velocity` · `_cos` · `_rel` | how far a FROZEN 256-row probe batch's `value_pooled` moved since the last measurement | falling **while `train/grad_norm` holds** ⇒ weights move, functions do not |

### 1. The plasticity canary — the centerpiece, and the only SUPPLY-side probe

A small head (`LayerNorm → Linear → ReLU → Linear`, K outputs) regresses the trunk's **detached**
`value_pooled` onto K=4 synthetic targets that are pure functions of the observation. Every
`--canary-reset-steps` env steps (default 1,000,000) **ONE target is re-seeded, round-robin** — and
that reset is the whole instrument. Re-fitting a *brand-new* random function of the obs, from the
same representation, with the **same head weights** (they are deliberately NOT re-initialised),
measures how much usable structure the representation still SUPPLIES. A trunk that has collapsed
onto the policy's current answers re-fits slower and plateaus higher.

**THE TARGET FAMILY — quoted here because a deferred OFFLINE probe must use the SAME one or the two
instruments do not cross-validate.** `capacity_telemetry_test.py` pins the arithmetic literally:

```
seed(k, e) = 20260823 + k + 1_000_000 * e         # k = target index, e = its reseed count
P[:, k]    = torch.randn(obs_dim, generator=torch.Generator("cpu").manual_seed(seed(k, e)))
target_k   = tanh( obs @ P[:, k] / sqrt(obs_dim) )
```

The generator is **CPU-seeded always**, so the same `(k, e)` gives the same column on any device and
in any process. `tanh` bounds the target into (-1, 1) so `canary_recovery` is a ratio of two
comparable scales; an unbounded target would make the two sides of that ratio incomparable.

⚠️ **It measures the REPRESENTATION's richness, not the policy's headroom, and those are different
claims.** A rising `canary_loss` says the trunk carries less recoverable obs structure than it did.
It does not say the policy would be stronger if it carried more. Treat it as a signal that
something is narrowing, then go find out what — the same discipline `dV`-ablation results get here.

**The gradient CANNOT reach the trunk, and that is structural rather than careful.** The head is
owned by the **PPO object**, not the extractor — no `state_dict` key, no policy-optimizer position
(the ai_v6_13 "128 vs 5" class is unreachable) — it trains through its own Adam over its own
parameters, and `PlasticityCanary.step` detaches its input unconditionally. Three assertions on a
real `MaskablePPO` measure that on the actual parameter update and on `.grad`, with a LIVE graph
handed to it on purpose.

### 2. Half-batch trunk-gradient cosine — INTERFERENCE

Every `--capacity-cosine-every` minibatches (default 50), the current minibatch is split in half,
each half's gradient on the **shared trunk** is taken, and the cosine between them is logged. Two
halves of one on-policy batch are i.i.d. draws from the same distribution, so a healthy batch has
them broadly agreeing; a cosine trending to zero or negative means the batch is increasingly
fighting itself — capacity going into trading one part of the state space against another instead
of improving on both.

* **TRUNK is `grad_balance.shared_trunk_parameters`** — the existing allow-list (`embeddings`,
  `pokemon_encoder`, `team_transformer`, `assembler`), reused rather than re-defined so this cosine
  and the `grad/*_share` family are talking about the same weights.
* **The surrogate is the PLAIN PPO objective** (clipped policy loss + `vf_coef`·MSE), not the run's
  full fold. The question is whether the two halves agree about the RL objective; folding in a dozen
  auxiliaries would make the answer a statement about the auxiliaries instead. PopArt and
  tail-weighting are skipped for the same reason — both are monotone rescalings of the same
  per-sample residual, so they move the gradient's LENGTH, not this angle.
* **Advantages are sliced from the caller's already-normalized tensor.** Re-normalizing each half
  against its own mean/std would inject a difference the batch does not have and bias the cosine down.
* **Orientation, not a threshold**: on a fresh `--debug` run at ~6k steps the cosine reads **0.99**
  with `halfbatch_grad_norm_ratio` 0.86 (measured 2026-08-23) — an untrained policy's two halves
  agree almost perfectly, which is what "healthy" looks like at the start. There is no calibrated
  alarm LEVEL and this is not one; what a saturating run is expected to show is that number
  *descending* over tens of millions of steps.
* **It corrupts nothing**: `th.autograd.grad`, never `backward()`, so `.grad` is never written and
  an in-flight grad-accumulation group survives bit-for-bit. That is a test, not a comment — the
  gate populates a real accumulated gradient and real Adam state first, because an all-`None`
  `.grad` would hide exactly the failure worth catching.

### 3. Feature velocity — do the functions still move?

One **fixed** probe batch of 256 obs rows, captured from the first rollout after launch and never
changed, is forwarded every `--capacity-velocity-every` `train()` calls (default 50) under
`no_grad`; the mean L2 displacement of `value_pooled` since the previous measurement is logged.
Read it **beside `train/grad_norm`**: weights moving while functions do not is the fingerprint of a
network burning gradient on a representation that has stopped changing. `feature_velocity_rel`
divides by the representation's own norm, because a falling raw velocity can also mean the features
merely shrank. The forward is the EAGER `type(fe).forward` with an observation-key-only dict, for
the reasons `cf_terms` gives (the compile flags patch the BOUND `fe.forward`, and a second obs
shape through the compiled entry point would add a graph for a diagnostic); the
ObservationDebugger is suppressed, since these are replayed rows.

### Where it sits in `train()`, and the flag's class

**It is NOT a fold step.** All three probes run at the END of the minibatch body — after the loss
fold, after `loss.backward()`, after the optimizer step — so nothing they do can reach `loss` or
`.grad`, and the placement is the proof. The one thing taken from inside the fold is a SNAPSHOT of
this minibatch's `value_pooled`, grabbed right after `evaluate_actions` for the same reason steps
2-4 of the fold sit where they do: the TD-aux / counterfactual folds run their own forward and
REPLACE the stash. A stale or missing snapshot is a **skip** (row-count checked), counted in
`capacity/canary_steps` rather than silently mis-pairing features with observations.

The flag is the **`training_coef` class** (`td_aux_coef` / `cf_records`): an argparse entry
defaulting to `None`, a `_resolve` line, and a recorded `ModelVersion` field (config **v101**,
`gen3_capacity_telemetry_v1`) — never in `check_compatible`. It is NOT `structural` (no module in
the policy tree, no `state_dict` key, forward bit-identical), NOT `resume_immutable` (changing it
mid-run changes nothing about training, so there is nothing to protect), and NOT `runtime` — a
runtime knob's value is unrecoverable after the fact, and a DIAGNOSTIC whose provenance is
unrecoverable is a number nobody can interpret later. It is not a `flag_registry` row either: that
registry declares **extractor** toggles, and nothing here reaches the extractor.

**OFF is byte- AND cost-identical**: no head, no optimizer, no projection matrix, no probe batch,
no extra forward or backward — one boolean per minibatch. Gated both ways (an OFF run's update
equals a no-telemetry run's; an ON run's update equals the OFF run's, exactly).

**Overhead, measured** (2026-08-23, CPU, real 2,047,958-param policy over the live 2501-dim obs, 80
minibatches per `train()` = the production ratio, 10 reps/arm INTERLEAVED, `OMP_NUM_THREADS=4`, quiet
box). Two independent clean runs:

| run | OFF median | ON median | overhead (median / minima) |
|---|---|---|---|
| 1 | 5797.2 ms | 5943.4 ms | **+2.52% / +2.42%** |
| 2 | 5800.6 ms | 5938.6 ms | **+2.38% / +2.32%** |

Inside the <3% budget, with the two arms' 10-sample ranges **disjoint** in both runs. That figure is
a CONSERVATIVE bound for production: it runs the BASELINE extractor chain (no damage op, no belief
heads), whose forward is far cheaper than the production one, and a cheaper denominator makes the
probes' share LARGER — smaller again on CUDA under `--compile-trainer`, where the canary's tiny eager
MLP is noise against a compiled forward+backward. Reproduce:
`src/agents/training/capacity_overhead_benchmark.py`.

⚠️ **A third run, taken while the 4-worker test suite shared the box, read +4.28%** — and its OFF
arm alone spread 5840→6618 ms (13%) where a clean arm spreads 1.7%. `warn_if_contended` did NOT
fire (the one-minute load average lags a job that just started). The interleaving saves the SIGN of
the effect under contention but not its size, so: read the within-arm spread before believing the
delta, and take the number on a quiet box.

🚨 **THE CANARY'S STATE IS NOT CHECKPOINTED, and this is the honest limitation.** `_capacity_state`
is in `_excluded_save_params`, so the head, its Adam state, the projection matrix and the frozen
probe batch are all re-created on every resume — and the launcher restarts every 3 hours. The
canary's loss therefore JUMPS at each restart and `canary_recovery` restarts its curve. The trade
was taken deliberately (persisting a diagnostic's optimizer into every checkpoint is worse), and the
usable reading is **compare recoveries WITHIN a restart window**: at production throughput a 3-hour
window is ~16M env steps, so a 1M-step reset interval still fires ~16 times inside one. The startup
banner says so out loud, because a silent ON here is a misreading waiting to happen.

## THE VALUE LOSS has a MODE — `--critic {shaped,winprob}` (`gen3_winprob_critic_mode_v1`)

**Default `shaped`; a flagless run is byte-identical.** Design of record:
`designs/ai_v12/design_winprob_only_critic.md`. The model-side half (the route, the version gate)
is `src/agents/model/CLAUDE.md` → *The CRITIC MODE*; this is what `train()` does about it.

| | `shaped` | `winprob` |
|---|---|---|
| the value TERM | `vf_coef · _value_loss_from_se(...)`, or the HL-Gauss CE at `vf_coef` under `value_from_dist` | `vf_coef · _win_prob_loss(...)` — the head's **BCE against the terminal outcome** |
| noise-scale group | `value` | **`value`** |
| the scalar `value_loss` | the loss | a DIAGNOSTIC only (its term is dropped, `_vf_term = 0.0`), and computed UNCLIPPED |
| `--win-prob-coef` | weights the auxiliary BCE, tagged `aux` | refused — the BCE is the value loss now |
| gate | `win_prob_coef != 0` | `win_prob_coef != 0` **or** the critic — the head's own loss cannot be switched off by a coefficient |

**The `"value"` tag is the point of `gen3_value_diagnostics_v1`'s sibling finding, applied one
critic over.** §1.4 of the design records that under `--value-from-dist` the REAL critic loss was
folded as `_ntg.add("aux", …)` while `_vf_term` was 0.0 — so `train/noise_scale_value` spent that
entire era describing a term with weight zero, and the grad-balance probe had to compensate
separately. The promoted BCE joins `value`, and `grad/value_share`'s term follows the critic
(`win_prob_term if critic_winprob …`) for the same reason: a `grad/value_share` measuring the
FROZEN scalar head's pull is the 2026-07-22 catch (`grad/value_dist_share` stuck at ~0.05).

**`win_prob/critic_*` — the P(win)-currency reliability read, once per rollout.** Under `winprob`
only, from `agents.training.scaffolding.reliability_table` — **imported, never re-implemented**, so
the live number and `python -m main.scaffolding_gauge --reliability` are the same statistic and a
run's series is comparable with the committed 2026-09-06 baseline. It reads the rollout BUFFER's
`values` (which under this critic ARE the P(win) GAE bootstrapped from) against
`win_target`/`win_mask`, i.e. the DEPLOYED quantity — where the `win_prob/ece` family above reads
the head's logits per minibatch. Keys: `critic_{brier,skill,ece,mce,reliability,resolution,uncertainty,decomp_residual,base_rate,n}`.

🚨 **`critic_resolution` IS the meter; `critic_reliability` is not.** A base-rate forecaster scores
a perfect 0 reliability and a useless 0 resolution. The committed baseline measured this head at
reliability ~0.002 against a resolution of 0.062 out of an available 0.182 — already calibrated in
the MEAN, starved of SEPARATION — so a promotion that improves ECE and leaves `critic_resolution`
flat has moved the meter that was never the disease. ⚠️ It is computed on the TRAINING population
rather than the loss-enriched eval quota, so it needs no selection reweighting and its LEVEL is
**not** comparable with the offline gate's — only its trend. An unmeasurable rollout publishes
**`{}`, never zeros**: a calibration of nothing and a perfect calibration must not render the same.

**What `winprob` does to the REWARD, and the one thing it gives up.** The stream becomes the
TERMINAL **win indicator** alone (`--no-hand-shaping` + `--terminal-indicator` +
`--victory-value 1.0` + `--draw-penalty 0`, all four REQUIRED and each named by its own
`combination_checks` refusal), so the undiscounted return is exactly `1{win}` and, at `--gamma 1.0`,
`V(s) = P(win | s)` with no approximation term. The cost is stated rather than buried: **a critic
bounded in [0,1] cannot represent "a timeout is worse than a loss."** `--draw-penalty`'s
`−35 < −30` ordering is unrepresentable there, so the anti-stall pressure is the obs deadline clock
plus **`--arm-no-progress-tax`** (design gap B4) — which re-arms `no_progress_tax` alone under
`--no-hand-shaping`, without reviving the other 24 BIAS terms. **Stall rate and mean episode length
are PRIMARY, kill-condition-bearing endpoints on a `winprob` arm.**

**THE DRAW BRANCH IS EXPLICIT, and `signal/draw_rate` states its frequency** (design §3.2 / gap
B9). `battle.won` is a TRI-STATE — True / False / **None**, the last being a draw or the 250-turn
timeout — and `MaskableAgentWrapper.step` used to reach `0.0` for the third case through a boolean
test, i.e. by accident. It is now a named branch: **a draw is scored as a NOT-WIN by decision**
(`y = 0`), because that makes "P(win)" literally P(win); 0.5 would make the critic systematically
wrong exactly where stalling tempts; and masking the episode out would leave its ~250 decisions
with no learning signal at all. It is SCORED, never dropped — `info["win_draw"]` rides beside
`win_outcome`, `SignalMetricsCallback` counts it on the terminal scan it already runs (both rollout
paths), and `signal/draw_rate` + `signal/n_terminals` publish the rate per rollout.

⚠️ **It is `signal/draw_rate`, not the `train/draw_rate` the design proposes**, for two reasons:
that callback carries a PINNED prefix contract (every row it emits must start with `signal/`), and
on the merits the draw rate is an OUTCOME statistic whose literal siblings — `signal/outcome_win_rate`,
`signal/outcome_entropy` — are computed from the same terminal `info` in the same loop.

⚠️ **The label and the objective DISAGREE about draws under the shaped terminal, and that is a real
property of that composition rather than a defect here.** The label scores a timeout as a not-win
(`y = 0`, the same as a loss) while the reward pays `--draw-penalty` (−35, i.e. WORSE than the −30
loss). Under `--terminal-indicator` they agree: a timeout pays 0.0, exactly like a loss.

**`--gamma` is now a flag** (design gap B6; it was hardcoded at `model_build.py`'s
`InstrumentedMaskablePPO(...)` call). Its `shaped` default is `reward_weights.PBRS_GAMMA` itself
rather than a retyped `0.9999`, and the `PBRS_GAMMA == reward_config.gamma == model.gamma` assert
is now **GATED on a hand potential actually being folded** on both build paths — under
`--no-hand-shaping` every `_fold_*_pbrs` early-returns, so there is nothing for the invariance
claim to be about and an ungated assert would refuse a coherent run. It is **INERT ON A RESUME**
like `--lr`: SB3 restores the checkpoint's own γ, the resume path SAYS so, and it re-points
`reward_config.gamma` at the value actually in force so the two cannot silently disagree.

## PopArt value-target normalization (`--use-popart`)

The fix for the swamping the diagnostics above reveal. `train()` reads `self.popart =
getattr(self.policy, "popart", None)` (built by the policy when `--use-popart`; see
`src/agents/model/CLAUDE.md` → PopArt for the math + version-checking). When present: once per
`train()` (before the epochs) `popart.update(self.rollout_buffer.returns, self.policy.value_net)`
advances the running `(mu, sigma)` **and** POP-rescales `value_net`; the value loss then becomes
`MSE(popart.normalize(returns), popart.normalize(values))` — the **normalized**-space loss, so the
value gradient into the shared trunk drops by ≈`sigma²` and stops swamping the policy. The policy's
value sites de-normalize, so `rollout_buffer.values` / GAE / advantages stay real-unit — the policy
path is untouched. **`--use-popart` requires an explicit `--clip-range-vf none`** (errors otherwise —
self-documenting config; clipping is unnecessary with value normalization, and would clip in
un-normalized units). New diagnostics ride the same generic metrics path:
`popart/mu`, `popart/sigma` (watch them track `train/return_mean`/`return_std`),
`popart/value_weight_norm` (POP keeps it bounded). Under PopArt `train/value_loss` is the normalized
loss (≈O(1)) and `grad/value_policy_logratio` should fall from a large positive value toward ~0 (the
aux-independent value/policy balance — `grad/value_share` also drops but moves with the aux count, so the
log-ratio is the cleaner confirmation it worked).

## Tail-weighted value loss (`--value-tail-weight`)

A probe-driven critic-tail lever (off by default). A representation probe found the critic's TD-residual
tail is fat and barely anticipated (the V-tail crater the `eval/td_resid_tail` CVaR@5% already tracks),
so `InstrumentedMaskablePPO._value_loss_from_se` replaces the plain `F.mse_loss` at all **three** value
sites (PopArt-normalized / unclipped / clipped) with a **CVaR blend**:
`value_loss = (1−β)·MSE + β·mean(worst _VALUE_TAIL_FRAC=10% squared errors)`, computed in whichever
space the branch uses (NORMALIZED under PopArt, so the tail selection matches the loss scale). At **β=0
it is `se.mean()`, byte-identical to `F.mse_loss`** (the default no-op). β>0 makes the critic prioritise
the big over-claim misses it under-prices; it is **symmetric in error sign**, so V stays an unbiased
mean estimate and the GAE advantages the policy reads are unaffected — a weighting change, not a new
target. The hparam is set on the model after construction (like `_async_rollout`), **resume-immutable**
(recorded in `model_config.json`, FATAL to change on resume via `ModelVersion.check_value_tail_weight`,
`MODEL_CONFIG_VERSION` v11; excluded from `check_compatible` since a frozen opponent never runs the value
loss), and **not weight-shape** (no `ARCH_SIGNATURE` bump). The v10
`value_active_readout` value-head fix that used to pair with it is **deleted** (v88
`gen3_dead_flag_purge_v1` — it was never enabled in a gen-8+ run and the multi-seed readout /
`--value-threat-inject` superseded it; a checkpoint recording it ON is refused by the migration). Validate by watching
`eval/td_resid_tail` fall.
Tests: `instrumented_ppo_test.py` (β=0 == MSE, β>0 == the exact blend).

## TD-consistency auxiliary (`--td-aux-coef`, `td_aux.py`)

**What it fixes.** The critic's only signal is a PER-STATE regression, `MSE(V(s_t), G_t)`. That
constrains each state's LEVEL and says nothing about the DIFFERENCE between two adjacent states — so
independent per-state noise ε in V arrives in `ΔV` at `2·Var(ε)`, exactly where the truth is nearly
constant. Since ΔV is what GAE reads, that is injected advantage noise on **every** transition, not
just the dramatic ones. `--td-aux-coef λ` adds the Bellman identity the critic already owes, as an
explicit loss:

```
loss += λ · mean_pairs[ ( V(s_t) − r_t − γ·V(s_{t+1}) )² ]
```

Both residual ends carry gradient (the residual-gradient / Baird form — see the *Cons* in the
pre-registration). `λ = 0.0` is the default and the whole block is skipped, so an OFF run is
byte-identical. **Pre-registered band: 1.0–3.0, 3.0 the favourite; `λ ≤ 0.1` measured significantly
WORSE than control offline, so the small-coef regime is to be avoided, not treated as "a bit of the
effect".** Full pre-registration (rung-1 evidence, the honest ceiling, the rung-2 gates):
`designs/research_state/levers/td_consistency_aux.md` (ledger C5). Do not edit that file — it is the
pre-registration.

**Where the pairs come from — this is the whole engineering problem.** `RolloutBuffer.get()` yields
a RANDOM PERMUTATION, so a PPO minibatch contains **no adjacent pairs at all**; the pairs have to be
drawn from the buffer's surviving `[n_steps, n_envs]` structure. `td_aux.sample_contiguous_pairs`
draws `TD_AUX_STATES` (512) rows as contiguous per-env runs of `TD_AUX_SEG_LEN` (16) and pairs their
adjacent rows. Four facts make it correct:

- **Row convention.** After the first `get()`, `observations` are `swap_and_flatten`ed to ENV-MAJOR
  (`row = env·n_steps + t`), so temporal adjacency survives; the sampler returns rows in exactly
  that convention and `_td_aux_term` **raises** if `generator_ready` is False rather than indexing
  an un-flattened array (which would silently mis-pair states with rewards at any `n_envs > 1`).
- **`rewards` / `episode_starts` are NOT in `get()`'s flatten list**, so they stay `[n_steps,
  n_envs]` and are read in their native shape (rewards are swapped to env-major at use).
- **Episode boundaries DROP the pair, never zero it.** `episode_starts[t+1] == 1` means the
  successor begins a new episode, so (t, t+1) is not a transition; zeroing would train
  `V(s_t) → r_t` at every battle end. This also disposes of SB3's time-limit bootstrap (which folds
  `γ·V(s_term)` into the stored reward at the done step): that row's successor always starts an
  episode, so the pair never forms.
- **Segments, not random pairs.** L contiguous states serve L−1 pairs off L forwards — the
  "K+1 forwards serve K pairs" economy the pre-registration calls for, ~2× cheaper per pair than
  independent pair sampling. Rung 1 also found whole-battle batching beat a random-permutation
  control by 12%, so the within-segment correlation is a feature.

**It runs per MINIBATCH, with its own sample and its own critic forward** — modelled on the
search-teacher / OPD folds, not on the once-per-`train()` diagnostic probes. Those are read-only;
this one carries gradient, and a once-per-`train()` fold would give it ONE contribution against the
value loss's `n_epochs × n_minibatches` (~240 in production), so λ would have to be ~240× rung-1's
band to mean the same thing. Cost is bounded by `TD_AUX_STATES`, not by `batch_size`: one extra
512-state critic forward per minibatch, ≈10% of the train step at production shapes.

**The value path is `policy.predict_values`, never a hand-rolled one.** That method is what routes
to the DISTRIBUTIONAL head's mean under `--value-from-dist` (where the scalar `value_net` is FROZEN)
and applies PopArt's de-normalization — reading `value_net` directly would train a critic the run
does not use.

**Units.** `predict_values` returns REAL-unit values and the buffer's rewards are real-unit, so the
raw residual is real-unit. But under PopArt the value loss trains in NORMALIZED space, so the
residual is divided by σ — which *is* the normalized-space residual, since
`normalize(V) − normalize(r + γV′) = (V − r − γV′)/σ` (the μ cancels). λ therefore keeps the meaning
rung 1 calibrated in both regimes; σ = 1.0 with PopArt off.

**Metrics (`td_aux/` prefix).** `resid_rms` is the headline — the quantity being minimised, the live
counterpart of the offline ΔV-dispersion instrument, and it should FALL. `resid_mean` (SIGNED) is
the no-harm watch: rung 1's decomposition says this is dispersion suppression, so a bias drifting
away from ~0 means the residual-gradient term is shifting the LEVEL rather than tightening it — read
it beside `train/explained_variance`. Also `loss`, `n_pairs`, `scale` (the σ the residual is
expressed in) and `pair_drop_frac` (share of candidate pairs lost to episode boundaries). The
shared-trunk pull rides `grad/td_aux_share` + `grad/td_aux_policy_cosine`; the term reaches the trunk
through the CRITIC path only, so `td_aux_share` against `value_share` is the read for "is the
consistency term crowding out the level regression it is meant to complement".

**Class: `training_coef`.** Scales a loss, touches no forward pass ⇒ NO `ARCH_SIGNATURE` bump, NOT in
`check_compatible` and no `check_*` of its own; recorded on `ModelVersion` (`MODEL_CONFIG_VERSION`
v90) purely for provenance and so a **flagless resume inherits it** via `_resolve`, exactly like
`--opp-belief-aux-coef`. It is deliberately NOT in `agents/model/flag_registry.py` — that registry's
scope is extractor architecture toggles, and this reaches the extractor not at all.

Tests: `td_aux_test.py` — the sampler (env-major row convention, (t, t+1) adjacency, boundary pairs
DROPPED not zeroed, the all-boundary degenerate → `None` not 0.0, the segment economy, fail-loud on a
flattened `episode_starts`), the residual math on a hand-built case, the PopArt scale identity, both
ends carrying gradient, and on a REAL `train()`: coef-0 byte-identity (asserted twice — identical
parameters AND the sampler monkeypatched to raise, so a future sampler change cannot perturb an off
run), coef>0 moving the update and logging every metric, gradient landing on `value_net`, and the
un-flattened-buffer refusal.

## Gradient accumulation (`--grad-accum-steps`)

A **GPU-memory lever** for keeping a large effective batch when the full minibatch OOMs. Stock
`MaskablePPO.train()` does one `forward → backward → optimizer.step()` **per minibatch**, so
`batch_size` couples the effective-batch size to the activation-memory peak — there is no
`accumulation_steps` knob upstream. `InstrumentedMaskablePPO.train()` adds one: with
`--grad-accum-steps K` it runs K `batch_size`-sized **micro-batches**, summing their gradients, and
calls `optimizer.step()` only **once per group of K**. Because gradients are additive and each
micro-loss is scaled by `1/K`, the accumulated gradient is the **exact** gradient of one
`(batch_size·K)` batch — but the backward graph only ever holds **one micro-batch's** activations.
So `--batch-size 4096 --grad-accum-steps 4` trains with the dynamics of `--batch-size 16384` at ~¼
the activation peak (the `DamageOperator`'s `[B,6,~416]` tensors + the grad-balance probe's retained
graph scale with the micro-batch, not the effective batch). `K=1` (default) is **byte-identical to
upstream** (one step per minibatch).

- **The step is gated on a full group** (`micro_in_group == accum`); a **trailing partial group**
  (#minibatches not divisible by accum) is flushed at epoch end with its accumulated grad rescaled
  `accum/micro_in_group` so the short group's step has the right magnitude. Grad-norm clipping
  (`max_grad_norm`) is applied **once per optimizer step** (per group) — i.e. to the full
  effective-batch gradient, exactly as the big batch would clip it.
- **Bit-exact when the rollout divides cleanly.** The accumulation math reproduces a literal
  `batch_size·K` batch to the float32 noise floor (~3e-8, empirically) **when `batch_size` divides the
  rollout (`n_steps·n_envs`) AND `K` divides the minibatch count** — then every group is `K` equal-size
  micro-batches. Production power-of-2 configs satisfy this (e.g. rollout 131072, `--batch-size 4096
  --grad-accum-steps 4` → 32 micro-batches, 8 groups, exact). For a NON-divisible rollout the single
  smaller remainder minibatch in the **final group of each epoch** is weighted as if full-size — a
  bounded mis-weighting of one remainder per epoch (≈8e-5 on params in a toy probe; negligible vs a
  100k-sample rollout, and no worse than stock SB3, which gives that remainder minibatch its own
  full-weight optimizer step).
- **KL early-stop** (`target_kl`, `None` by default so this path is dormant) discards the partial
  group (`zero_grad`, no step) on a trip — a true `(batch_size·K)` batch checks KL over the whole
  effective batch and would discard it as one unit.
- **The other (always-present) non-identity is per-micro-batch advantage normalization**
  (`normalize_advantage`, default on): stock SB3 already normalizes advantages *per-minibatch*, so
  here the normalization sample is the micro-batch (e.g. 4096) rather than the effective batch
  (16384). The difference is the normalization sample size — statistically negligible for batches of
  thousands (and it is this term, not the accumulation math, that the bit-exact check above isolates
  by running with `normalize_advantage=False`). (The grad-balance probe also samples on the first
  **micro**-batch instead of the first minibatch — a smaller, cheaper, still-representative sample; its
  `retain_graph` memory shrinks with the micro-batch.)
- **Not version-locked / not in `model_config.json`.** It is a pure train-loop knob (no forward
  change, no weight-shape effect, no `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` bump) — like `batch_size`
  / `n_epochs`, **forwarded as a CLI flag on every launcher resume** (set on the model in both the
  fresh-build and resume paths of `train_rl_agent.py`; surfaced in `_model_hparams` for the sidecar).
  Change it freely on resume; only the *effective* batch (`batch_size·K`) matters for dynamics, so
  `--batch-size 16384` (K=1) and `--batch-size 4096 --grad-accum-steps 4` continue a run identically.
- **The upstream-drift hash check is unaffected** (it hashes only `sb3_contrib.MaskablePPO.train`).

Tests: `instrumented_ppo_test.py` — `test_grad_accum_matches_full_batch` runs the REAL `train()` on a
minimal `MaskablePPO` and asserts `K=accum` over `batch/K` micro-batches reproduces the parameter
update of `K=1` over the full batch to `rtol=1e-4` (parametrized over a divisible 16=4×4 **and** a
non-divisible 15=5×3 case that exercises the partial-group rescale), plus default-is-1 + source-marker
guards.

### Gradient noise scale (`train/noise_scale`) — "is the batch big enough?"

A **free byproduct of accumulation** (only emitted under `--grad-accum-steps >= 2`) that answers *how
big a batch is enough* with a number instead of intuition: the McCandlish et al. 2018 **simple
gradient noise scale** `B_simple = tr(Σ)/|G|²` — the critical batch size where gradient noise stops
dominating. Below it, a bigger batch buys ~linear per-step progress; above it, diminishing returns
(you're averaging out noise that was already small, and could shrink the batch for more update steps).

The estimator needs the squared gradient norm at **two batch sizes** — and accumulation produces
exactly that for free each `train()`: ‖g‖² of one micro-batch (`B=batch_size`, read from `.grad`
right after the first micro-batch's backward, un-scaled by `accum²`) and of the accumulated first
group (`B=batch_size·accum`, the pre-clip norm `clip_grad_norm_` already returns). From the model
`E‖Ĝ_B‖² = |G|² + tr(Σ)/B`, two `(B, ‖Ĝ_B‖²)` points pin both `|G|²` and `tr(Σ)` (`_noise_scale_estimate`,
pure/unit-tested). Both single-call estimates are noisy (either can go negative), so the **numerator and
denominator are EMA'd separately** (`_NOISE_SCALE_EMA_DECAY`=0.99 ≈ a few-hundred-call window) and only
then divided — and the scalar is emitted only once both EMAs are positive (so a warmup transient never
logs a garbage value). Cost: one extra global grad-norm read per `train()` (the group norm is reused
from clipping); no extra backward. EMA state is **process-local** (resets on a launcher restart →
re-converges in a few hundred calls; not saved).

Two scalars ride the standard logger → TensorBoard + launcher TUI (`format.py` labels `noise scale` /
`noise/batch`, in the train column by `train/grad_norm`):
- **`train/noise_scale`** = `B_simple` (compare directly to your effective batch `batch_size·accum`).
- **`train/noise_scale_ratio`** = `B_simple / (batch_size·accum)` — the actionable read: **≫1 ⇒
  noise-limited** (enlarge the effective batch), **≪1 ⇒ diminishing returns** (you have more than
  enough; could shrink for more/cheaper update steps), **~1 ⇒ the sweet spot**.
- **The NSR advisor (`_noise_scale_advice` / `_emit_noise_scale_warnings`)** — when the SMOOTHED
  ratio leaves the band, a `⚠️ [NOISE]` warning goes to the launcher **Events panel** (via
  `main.launcher.ipc.emit`; plain print standalone) naming the concrete fix: ratio > 2 → "raise
  `--grad-accum-steps` ~ratio× (free — no VRAM/FPS cost, same rollout)"; < 0.5 → "over-batched,
  lower it for more steps per sample". **Rate-limited to one warning per key per 30 min** and
  suppressed for the first ~20 EMA folds (warm-up false-alarm guard). Pure decision logic
  unit-tested (`instrumented_ppo_test.test_noise_scale_advice_bands_and_fixes`). A FiLM-group half
  (`film/noise_scale*`, `--film-grad-accum-steps` and `_GroupGradAccumulator`) measured the same
  thing for the conditioning params until v78 and was deleted with the zarch family.

Tests: `instrumented_ppo_test.py` — `test_noise_scale_estimate_recovers_known_values` (the two-point
math recovers a planted `|G|²`/`tr(Σ)` exactly), `_smaller_batch_is_noisier_sign`, `_global_grad_sq`
matches a manual sum, and `_logged_only_when_accumulating` (real `train()`: skipped at accum=1, EMA
updated + scalar emitted at accum=2).

### 🚨 The total is NOT the policy gradient — the PER-TERM noise scale (`noise_scale_terms.py`)

**`train/noise_scale` is measured on the TOTAL gradient, and on this tree the total gradient is
mostly not PPO.** The loss is the clipped surrogate + the value term + the entropy bonus + a dozen
DENSE supervised auxiliaries (belief heads, win-prob, spread/nature/HP-type, value-dist, TD-aux,
the counterfactual family) + a distillation KL on a fold. A supervised head's per-example gradients
**agree** — its target is a label, not an advantage — so its `tr(Σ)` is small and its `|G|²` is not.
Mixing it into the total therefore **DEFLATES** `B_simple = tr(Σ)/|G|²`, and the run reads
"over-batched" while the term you are actually trying to train may be starved. Acting on the total
in that state shrinks the batch the policy gradient needed.

That confound is not hypothetical here: the live runs read `train/noise_scale_ratio` **0.001 early
and 0.05 late** on the generalists and **1.1** on the v8 fold, i.e. "over-batched 16-1000x" — a
conclusion no batch-size decision should rest on until the policy term has been read on its own.

**Five groups, the SAME estimator.** `PerTermNoiseSampler` accumulates each group's gradient over
the same two batch sizes the total already uses for free (one micro-batch, and the accumulated
first group of epoch 0) and feeds them through the SAME `_noise_scale_estimate` two-point solve and
the SAME separately-EMA'd numerator/denominator. The math is not forked — the whole point of the
comparison is that a disagreement can only be the *gradient*, never the estimator.

| group | is |
|---|---|
| `policy` | the clipped surrogate AS FOLDED (`_policy_grad_term`; at the 1.0 default that is `policy_loss` itself) |
| `value` | `vf_coef · value_loss` (0.0 and therefore absent under `value_from_dist`) |
| `entropy` | `ent_coef · ent_loss_used` — **degenerate at `--ent-coef 0`** (a 0.0-scaled tensor still folds, so the group is present but its norms are 0 and both EMAs stay non-positive ⇒ nothing is emitted, which is the right answer, not a gap) |
| `aux` | every belief / win-prob / value-dist / TD-aux / search-teacher / OPD / counterfactual term, as ONE bucket (`grad/<term>_share` already breaks the heads out individually) |
| `distill` | the `--distill-coef` family — separated because it comes and goes with a fold and its dose is the thing being tuned |

Three scalars per group, beside the existing pair:
- **`train/noise_scale_<g>`** — that group's own `B_simple`.
- **`train/noise_scale_ratio_<g>`** — over the effective batch. **`_ratio_policy` is the headline.**
- **`train/noise_scale_share_<g>`** — `|G_g|² / |G_total|²`, i.e. who owns the true gradient's
  squared length. ⚠️ **The shares do NOT sum to 1 and must not be read as a partition**:
  `|G_total|² = ‖Σ_g G_g‖²` carries the cross terms, so groups pulling together sum above 1 and
  groups fighting sum below it.

**The advisor now reads BOTH.** `_noise_scale_advice` takes the policy-term ratio, quotes it inside
the OVER-BATCHED / NOISE-LIMITED bands, and — when the two land in different bands **or** differ by
≥3x inside one — emits its own `total_vs_policy_disagree` warning naming the aux deflation and
pointing at `train/noise_scale_share_*`. **That disagreement is the finding this exists for**, so it
is a warning of its own rather than a footnote on the total's. The policy ratio is read from the EMA
state (`_per_term_ratio`), not from the last fold, so a call the cadence did not sample still quotes
it.

**It cannot change training, structurally.** The tagger is threaded through the fold as
`loss = loss + _ntg.add("aux", term)` and **`add` returns its argument unchanged**, so the loss
expression is tensor-for-tensor the one that was there (`_ent_term` merely names a sub-expression
whose operations and order are unchanged). Gradients come from `torch.autograd.grad(…,
retain_graph=True)`, which never writes `.grad` — the same read-only mechanism
`grad_balance_metrics` has used per-term on every `train()` for generations, **which is also why
`--compile-trainer` is not a new risk**: the compiled backward is already called repeatedly with
`retain_graph` by that probe. Any exception retires the probe for the call with one printed line and
leaves the step untouched.

**COST — measured, and the default follows the measurement.** The probe costs `n_groups` extra
backward traversals on `accum` micro-batches of a sampled `train()`, against
`n_epochs × n_minibatches` fwd+bwd for the call — so **the overhead is governed by minibatches per
`train()`, not by batch size**. It self-reports (`train/noise_per_term_ms`) against a new
`train/train_ms` (the whole call's wall clock, recorded as `train()`'s last line — the honest
denominator for this and every future probe's cost claim):

| shape (epochs × minibatches per `train()`, accum 2) | `noise_per_term_ms / train_ms` |
|---|---|
| `--debug --n-steps 1024 --batch-size 512` (5 × 2 = 10 units, 4 live groups) | **24.3%** (5 calls) |
| `--debug --n-steps 1024 --batch-size 128` (5 × 8 = 40 units, 4 live groups) | **7.9% / 8.0%** (two runs, 11 calls each) |
| production `--n-steps 2048 --n-envs 64 --batch-size 16384 --n-epochs 10` (10 × 8 = 80 units, 5 groups) | **≈5.0% — EXTRAPOLATED** (2× the units, 1.25× the groups), not measured on GPU |

*(Both measured rows are CPU `--debug` runs on a box carrying a live fleet. That does not
invalidate them: the numerator and denominator are wall clocks from the SAME `train()` call, so
contention stretches both and the RATIO is what survives — which is exactly why `train/train_ms`
was added rather than an external stopwatch.)*

**Default ON** (`PpoHyperparameters.noise_scale_per_term`), because the production shape is well
under the 10% bar. Peak extra memory is one gradient accumulator per live group
(`n_groups × Σ|params|` ≈ 5 × ~16 MB), freed at the end of the call.

**⚠️ WARM-UP: `_policy` is the LAST tag to appear, and that is the signal, not a gap.** A group is
emitted only once both its EMAs are positive. For a strongly noise-limited term `|G|²` is genuinely
near zero at these batch sizes — with `accum=2` the estimate is `2·g_big − g_small ≈ 0` — so its
single-sample estimate SIGN-FLIPS and only the average resolves it. So EVERY reading here —
per-group AND the total — folds through the ONE `noise_scale.debiased_ema`: effective decay
`min(decay, 1 − 1/(n+1))`, i.e. a plain running MEAN until the `1/(1−decay)` window fills and the
exponential decay takes over, which is Adam's `ema / (1 − beta^t)` spelled as a decay. One negative
first sample therefore cannot suppress a tag for hundreds of calls. Measured effect on a 12k-step
debug smoke: without the debiasing `_policy` never emitted in 11 calls; with it, it emits by call
~10.

🚨 **THE TOTAL USED TO BE THE EXCEPTION, AND IS NOT ANY MORE (`gen3_noise_scale_warmup_v1`,
2026-09-03).** `train/noise_scale`'s EMA anchored on its FIRST sample at a fixed decay 0.99, so
after two samples it read `0.99·x₁ + 0.01·x₂` — the first sample, essentially, for its first few
hundred calls. That is why the first production reading on R5F15 had to be published as
"provisional, n=2", and it is the mechanism behind the smoke below in which the total never emitted
at all across 11 calls while every per-term tag did. Both halves now warm up identically, so a
young run's `train/noise_scale` and `train/noise_scale_ratio_policy` are comparable to each other
from the first reading. ⚠️ **A run's `train/noise_scale` series is NOT comparable across this
change** during its first ~100 folds — the fix moves early values by construction; the steady state
past the warm-up window is unchanged. ⚠️ And an EMA is now `(value, COUNT)`: priming
`_noise_ema_s`/`_noise_ema_g2` by hand without also setting `_noise_ema_n` leaves the fold on
sample 1, which takes the next sample whole (the one live edge, pinned in the test file).

**FIRST READING (2026-09-01, two `--debug --steps 12000 --n-steps 1024 --batch-size 128
--grad-accum-steps 2` runs, CPU, default flags so `aux` is small and `distill` absent).** It
reproduces the confound in miniature. Run A: `train/noise_scale_share_value` = **1.00001** and
`train/noise_scale` == `train/noise_scale_value` to five figures — **the "total" IS the value
term**, contributing ~100% of |G|² — with `noise_scale_ratio` = **0.081** ("over-batched 12×").
Run B (post-debiasing) put the policy tag on the board: `noise_scale_ratio_policy` = **6.2 then
2.7** ("noise-limited") on a run whose total ratio read **0.074–0.090**. Same run, same call, ~30–80×
apart, in the direction the total hides. Do NOT read those numbers as the production runs' answer
(different device, batch, flag set, and eleven calls) — read them as the instrument working: the
term PPO actually optimizes says *noise-limited* where the total says *over-batched*.

**One thing that smoke exposed, and that is now FIXED (2026-09-03).** The TOTAL's own EMA had the
same anchor-on-first-sample fragility the per-term half was debiased for — in run B the total's
`tr(Σ)` EMA started negative and `train/noise_scale{,_ratio}` therefore never emitted at all across
11 calls, while every per-term tag did. It was deferred at the time because the byte-identity of
the total series was a requirement of that work; it has since cost a reading (R5F15's
"provisional, n=2"), so the deferral was paid off with the shared `debiased_ema` above. The
revert-catcher is
`instrumented_ppo_noise_scale_terms_test.py::test_a_negative_first_sample_no_longer_suppresses_the_total_for_hundreds_of_calls`,
which reproduces exactly this failure. ⚠️ **A CONSTANT synthetic stream cannot detect this class**
— an anchored fold reads a constant correctly too — so the constant-stream test is the analytic
anchor and the outlier test is the guard.

Two levers, both ENV/constant rather than CLI flags — the probe changes no training math, so it
never belongs in `model_config.json` and should not have to survive a resume's argv:
- **`$GEN3AI_NOISE_SCALE_PER_TERM=0`** turns it off for a process (wins over the class default).
- **`_NOISE_PER_TERM_EVERY`** (`constants.py`, currently `1`) samples one `train()` call in N,
  dividing the cost directly. It slows the per-group EMA's convergence in wall-clock, never its
  value — the EMA is per SAMPLE. **Raise it on a config with few minibatches per `train()`**, which
  is the only regime where this probe is expensive.

Tests: `instrumented_ppo_noise_scale_terms_test.py` — the per-group fold recovers a planted
`B_simple` per group (with `aux` planted 4000x below `policy`, the confound itself); `share` is
pinned as NOT a partition; the sampler's `small_sq`/`big_sq` are checked against independently
computed gradients; a partial group and a group that first appears on a later micro-batch both yield
nothing; `.grad` is never written; `add` returns the identical object; a raising probe self-disables;
**the byte-identity gate** runs two identically-seeded fresh models (a `train()` on the toy is *not*
reproducible from a restored `state_dict` — three consecutive restores drift ~5e-4 with the probe
absent — so the arms are fresh, and a third OFF arm is the control); and the advisor's disagreement
family. A source scan asserts the tags in `train()` and `NOISE_TERM_GROUPS` are the same set.

## THE DOSE, and pinning a fork's step size (`--fork-lr` / `--fork-lr-freeze`, `dose.py`)

**`--lr` is INERT on a resume.** `main/train/model_build.py`'s resume path restores the checkpoint's
optimizer LR and prints `(arg --lr=… ignored on resume)` — correct for a launcher RESTART (the KL
controller should keep the rate it settled on) and wrong for a FORK, which then inherits whatever
the PARENT had annealed to. `--batch-size` and `--n-steps` are inherited the same way.

**The quantity that predicts a distillation fold's collateral is the DOSE, not the LR** (ledger M7):

```
updates_per_env_step = n_epochs / (batch_size * grad_accum_steps)
dose_rate            = lr * updates_per_env_step
```

`grad_accum_steps` is in the DENOMINATOR because K micro-batches are summed into ONE optimizer step
(see *Gradient accumulation* above), so two runs at the same `--lr` differ 8× in dose when one
accumulates 16 micro-batches and the other 2. Measured over the archive's own sidecars:

| run | eff. batch | epochs | lr median | dose_rate | vs v8 |
|---|---:|---:|---|---|---|
| `ai_v8_14_distill3_0725` | 32,768 | 7 | 1.004e-4 | **2.145e-8** | 1.00× |
| `ai_v9_59_R2ACTION_0827` (rev-2) | 4,096 | 10 | 5.814e-5 | 1.419e-7 | **6.62×** |
| `ai_v9_70_R3ACTION_0828` (rev-3) | 4,096 | 10 | 2.804e-5 | 6.845e-8 | 3.19× |
| `ai_v9_92_R5F00_0831` | 16,384 | 10 | 6.977e-5 | 4.258e-8 | 1.99× |

Three folds launched with the same `--lr` ran at three different rates, and nothing in any of them
said so — the controller's inherited state was a hidden confound in every fold comparison. Two
flags and one recorded block close that.

### `--fork-lr FLOAT` (resume-only) — and the fork-vs-restart rule

Sets the resumed model's **optimizer LR**, its **`model.lr_schedule`** and the **KL controller's
`_current_lr`** at load. All three, because each is a separate no-op risk: SB3 re-installs the
schedule's value at the top of every `train()` (so the optimizer alone would be overwritten on the
first update), and the controller's multiplicative ladder starts from wherever it thinks it is (so
seeding it from the checkpoint would walk straight back there). The pin is still clamped into
`[--min-lr, --max-lr]` — a bound the user set is a bound.

🚨 **It applies ONLY on a genuine FORK.** The launcher re-invokes the same argv every
`--restart-interval-hours` into the same run dir, so a flag that fires "on resume" fires every few
hours forever and would reset the adapted rate each time. `main/train/fork_lr.py` keys on WHERE the
resumed checkpoint lives — outside the run dir ⇒ FORK; `<run>/checkpoints/*.zip` or `<run>/*.zip`
(the legacy root layout) ⇒ RESTART. That is the predicate `run_io._resolve_fresh_model_dir` already
uses for its clobber guard and `launcher/checkpoint.resolve_fork_resume_model` uses to decide
whether a restart re-inits from the source; the launcher SWAPS `--model` to the fork's own
checkpoint once the fork has progress, so restart #2 of a fork reads RESTART for the same reason a
plain resume does. `<run>/warmstart/…` is deliberately a FORK — the consensus warm-start is an INIT
built from foreign teachers, not this run's own progress. A fresh run is REFUSED (use `--lr`).

### `--fork-lr-freeze` — a constant, recordable step size

Disables the KL adaptation **and** the two-phase cosine (`frozen` on both callbacks, plus
`freeze_at`), so the LR stays at `--fork-lr` exactly. A fold experiment wants a constant dose; an
adapting LR makes it a per-rollout variable nothing records. Unlike the pin it is a **property of
the RUN** and DOES persist across every periodic restart — re-read from the pin recorded in
`metadata.json`, or from the argv a launcher restart reproduces verbatim.

### The recorded `dose` block, and `python -m main.dose`

Every metadata write (and every checkpoint sidecar, through the one `_model_hparams` dict) carries
`dose`: `lr_now` · `lr_flag` (what `--lr` said, so the inertness is VISIBLE) · `fork_lr` ·
`lr_frozen` · `batch_size` · `grad_accum_steps` · `effective_batch` · `n_epochs` ·
`updates_per_env_step` · `dose_rate_now` · `kl_controller` {target_kl, kl_factor, lr_factor,
min_lr, max_lr, phase} · `fork_lr_pin` when one was applied. **metadata.json ONLY** — never
`model_config.json`, which is the weight-shape record `check_compatible` reads (root CLAUDE.md's
provenance rule). Live: `train/dose_rate` + `train/effective_batch` every rollout, because a groomed
run keeps no sidecars and the rate alone is ambiguous (a falling `dose_rate` is the KL controller
annealing OR an operator having raised `--grad-accum-steps`, and only the second moves the batch).

⚠️ **The `kl_controller` field is a PLAIN-DATA SNAPSHOT, never the callback.** `model.save()`
cloudpickles the model's `__dict__`, and an LR callback back-references the model and SB3's
`Logger`, which carries a `_contextvars.Context` and cannot be pickled — stashing the live object
breaks EVERY save in the run at the pre-train round-trip smoke (observed while building this, the
`_correction_buffer` hazard again). The snapshot is taken AFTER the pin so a freeze is captured.

`python -m main.dose <run>…` answers the same question for runs already on disk, from what they
already wrote down: median LR over the **checkpoint sidecars** (preferred over `snapshot_history`,
which is CAPPED at ~15 rows while sidecars keep every un-groomed checkpoint; then the run-level
`current_lr` as a single point), the shape from the SAME rows, and a ratio against a `--reference`
run (default `ai_v8_14_distill3_0725`). A run whose shape MOVED mid-flight is flagged rather than
averaged. Torch-free and model-free, so it reads a run whose architecture drifted past current code.

**Flag class: training-runtime.** Neither flag reaches the extractor, scales a loss or changes a
weight shape ⇒ no `ARCH_SIGNATURE` bump, not in `model_config.json`/`ModelVersion`, not in
`check_compatible`, and deliberately **not** in `agents/model/flag_registry.py` (whose scope is
extractor architecture toggles). They land in `metadata.json`'s `cli_args` like every train-loop
knob, and the launcher forwards them verbatim.

Tests: `src/main/fork_lr_test.py` (the discrimination rule incl. the warm-start case, the four
decisions, the freeze surviving a restart from the record AND from the argv, the three-site pin, the
clamp, the freeze holding across a KL excursion a control arm demonstrably moves on, and the three
config refusals), `src/agents/training/dose_test.py` (the arithmetic against v8's own recorded row,
the block, the pickle-safety of the snapshot), `src/main/dose_test.py` (source precedence, step
ordering, the shape-moved flag, the CLI, and that importing it pulls in no torch).

### `--adaptive-batch` — CLOSING the loop on the noise scale (`gen3_adaptive_batch_v1`)

Everything above is a **reading**. The NSR advisor printed *"raise `--grad-accum-steps` ~N×"* into
the Events panel and a human typed it on the next relaunch. `--adaptive-batch {off,total,policy}`
turns that into a controller — the second one this trainer runs, beside the KL-driven lr loop.
**OFF by default; an `off` run registers no callback at all and is byte-identical** (pinned by
`test_the_callback_cannot_change_the_ppo_update` + `test_the_flag_defaults_to_off_and_registers_no_callback`).

**THE RULE, in one paragraph.** Every rollout the controller reads the smoothed noise-scale ratio
of the chosen term and the number of EMA folds behind it. It does nothing until the EMA is warm
(20 folds — the NSR advisor's own warm-up, because a single-sample `B_simple` can SIGN-FLIP) and at
least `--adaptive-batch-every` rollouts (default 4) have passed since the last move. Then, if the
ratio has left `[target/band, target·band]` (defaults 1.0 / 2.0), K is **DOUBLED** when it is ABOVE
(noise-limited: each update is mostly sideways) and **HALVED** when BELOW (over-batched: buy update
steps instead of averaging), clamped into `[max(2, --adaptive-batch-min-accum), --adaptive-batch-max-accum]`.
An unreadable ratio, a cold EMA, a within-band reading or a clamp is a **named no-op**, reported
ONCE (a silently idle loop is indistinguishable from a broken one; a loop that says so every
rollout is noise).

**Why K and never `--batch-size`** — three independent reasons and all three matter: (1) SHAPE —
`--compile-trainer` keys graphs on shape against a `cache_size_limit` of 8, so a moving batch size
is the unbounded shape set `check_shape_stability` exists to refuse, and dropping to eager is
invisible (~1.75×); moving K leaves every forward shape byte-identical. (2) MEMORY — the activation
peak is one micro-batch, so K is the one batch lever with no VRAM cost. (3) EXACTNESS — K
micro-batches summed **is** the gradient of a `batch_size·K` batch. `check_shape_stability` takes
`n_steps`/`n_envs`/`batch_size`/`async_rollout` and *not* K, which is the proof rather than the
claim (`test_shape_stability_does_not_depend_on_k`), and a source scan fails any assignment to
`batch_size` in the controller module.

**🚨 THE FLOOR IS 2, NOT `--adaptive-batch-min-accum`.** The noise-scale estimator needs gradient
norms at TWO batch sizes and gets the second from the accumulation group, so at K=1 it emits
nothing — a loop allowed to reach K=1 would blind the signal it steers by and could never climb
back out. The requested floor is raised to 2 and the raise is ANNOUNCED at startup.

**`policy` is the mode to use.** It steers by `train/noise_scale_ratio_policy`; `total` steers by
the legacy scalar. The section above is the whole argument: the total is ~100% the value term plus
a dozen dense supervised aux heads and reads "over-batched" on runs whose policy term reads
"noise-limited", so sizing on the total shrinks the batch the policy gradient needed. `policy`
REQUIRES the per-term probe, and `$GEN3AI_NOISE_SCALE_PER_TERM=0` alongside it is a
`parser.error` rather than a loop that silently never reads anything.

**⚠️ WHY THE STEP IS 2× AND THE BAND MUST BE ≥ √2 — measured, and it corrected the design.** K is
the ratio's denominator, so a move changes the reading INSTANTLY and exactly: doubling K halves the
ratio. A correction therefore crosses to the *other* side of the band only when `target·band < ratio`
and `ratio/2 < target/band` can both hold, i.e. iff **`band² < 2`**. The first draft of this
documented the boundary as 2.0; running the test found `band=1.5` settling cleanly and the algebra
put the real boundary at **√2 ≈ 1.4142** — pinned by a parametrized test straddling it (1.30 and
1.41 chatter forever; 1.50 and 2.0 settle). The overshoot window `(target·band, 2·target/band)` is
only 0.6% wide at 1.41, so the test starts *just* outside the band on purpose: a start further out
takes several one-directional moves and lands in band, which is progress, not chatter, and would
have passed for the wrong reason. Default 2.0 sits comfortably above; a narrower band is for a
smoke that WANTS movement in a handful of rollouts.

**THE TWO-CONTROLLER INTERACTION — read this before tuning either.** The KL lr controller and this
one are COUPLED through the update: at a fixed `target_kl`, a larger K means each optimizer step
consumes more data, so per-step KL falls, so the lr controller RAISES lr. **That is intended** —
the batch loop fixes an update's signal-to-NOISE, the lr loop fixes its STEP SIZE — but it means
the effective **dose is a product of two controllers**, and the scalar to watch is `train/dose_rate`,
never either loop's own series. Two controllers chasing each other on one timescale is the classic
oscillation, and they are separated by their SIGNALS rather than merely their cadences: the lr loop
reads a KL EMA at `α=0.20` (half-life ~3 rollouts) and this one reads the noise-scale EMA at decay
`0.99` (a several-hundred-call window) — ~30–100× slower-moving by construction. `--adaptive-batch-every`
(4) is the second-order guard on top of that, and the lr loop's own 7-rollout post-move cooldown
means the fast loop has re-settled before the slow one looks again.

**PERSISTENCE is free and deliberately so.** `_model_hparams` already writes `grad_accum_steps`
into every checkpoint sidecar, straight off the model attribute the callback owns — so a moved K is
persisted by the EXISTING checkpointer with **no new key and no edit to the checkpoint path**.
`build_callbacks` reads it back with `read_checkpoint_metadata` (the same sidecar `handoff_lr`
rides in) and hands it to the callback as `resume_accum`, which installs it in
`_on_training_start` — i.e. AFTER `model_build` applied the CLI `--grad-accum-steps`, so the
controller's own history wins over the launch argv. A restart that TIGHTENS `--adaptive-batch-max-accum`
re-clamps rather than reinstating the old K. **The noise-scale EMA itself is process-local and does
NOT persist**, so after a restart the loop re-warms (20 folds) before it may move again — the right
behaviour, not a gap: a cold EMA is not a reading.

**The three series to read**: `train/grad_accum_steps` (K in force for the `train()` that follows),
`train/effective_batch` (`batch_size·K`), `train/adaptive_batch_ratio_used` (the exact number each
decision was made on). Every move also emits one Events line naming the ratio, the direction, the
old and new effective batch, and that the dose moved.

**⚠️ Like the two `--compile-*` flags, it is NOT inherited on a flagless resume** — a bare
`--model … --steps …` gets `off`. The launcher forwards its own recorded argv verbatim, so a
launcher-managed run keeps it across every periodic and crash restart; a hand-typed resume must
re-type it.

**SMOKE (2026-09-01, CPU `--debug --steps 34000 --n-steps 1024 --batch-size 128 --grad-accum-steps 2
--adaptive-batch policy --adaptive-batch-every 1 --adaptive-batch-band 1.5`).** Read back from
`tb/`: `train/grad_accum_steps` held at **2 for the first 20 rollouts** (the warm-up refusing to
act — and `noise_scale_ratio_policy` read 0.09–0.99 through that window, so the guard is what kept
a cold EMA from driving K DOWN), then **2 → 4 → 8 → 16 → 32** on readings of **432 → 24.1 → 6.76 →
3.38**, every one above the band's 1.5, before parking at the `max_accum` clamp with the
report-once line. Do not read those magnitudes as a production answer — a 128-row micro-batch on a
toy is not the production shape — read them as the loop tracking the series it is supposed to.

Tests: `adaptive_batch_callback_test.py` — the pure controller (planted ratio sequences → K
trajectory, the walked-feedback hysteresis, the √2 boundary, cadence, both clamps, the floor,
every unreadable-ratio form, constructor validation); the read seam returning exactly the recorded
value under the same emit gate; the callback mutating `grad_accum_steps` and nothing else
(behavioural + source scan + the `check_shape_stability` signature); report-once; the sidecar
round-trip through real `record_checkpoint`/`read_checkpoint_metadata`; and the byte-identity gate
(two identically-seeded fresh models, one with the callback attached and not moving K).

## LINEAGE — who forked whom (`lineage.py`, `python -m main.lineage`)

**Every exploiter, fold, funding fork and dose arm is a fork of some parent, and every comparison
the ledger makes is a claim about that graph.** Until `gen3_run_lineage_v1` that graph was
recoverable only by REGEXing `--model` out of `metadata.json`'s recorded `original_command` —
brittle in the obvious ways (a renamed flag, a quoted path, a `--model=X` spelling) and silent in
the worst one: **a failed parse reads exactly like a fresh run.** `metadata.json` now states the
answer instead of implying it.

**The block** — one top-level `lineage` key, written ONCE at fork creation:

```
lineage: {schema, role, fork_step, recorded_at,
          fork_parent: {path, resolved_path, run_dir, run_name, git_hash, arch_signature,
                        model_config_version, num_timesteps, sha256, created_at} | null,
          teachers: [ …same shape… ], exploiter_target: {…} | null,
          ancestry: [ {run_name, run_dir, git_hash, arch_signature, fork_step, role,
                       model_path, source} … ],
          ancestry_stop: {at, reason}}
```

`role` is `fresh` / `fork` / `fold` (`--distill-teacher`) / `exploiter` (`--exploiter`, which wins
— a double-sided exploiter is an exploiter that also distils, and its TARGET is what identifies
it). `ancestry` is walked through each parent's OWN block, nearest ancestor first, bounded and
cycle-safe on realpaths; `ancestry_stop` says where the chain went dark and why, because *"the
chain ends at a fresh root"* and *"the parent directory is gone"* are different facts a bare list
conflates.

🚨 **IMMUTABILITY is the whole feature, and it reuses `original_command`'s mechanism verbatim**:
`save_model_snapshot` reads the existing value first and the existing value always wins. A launcher
run restarts every few hours and an idempotent FORK has its `--model` swapped to the fork's OWN
latest checkpoint on each relaunch (`launcher.checkpoint.resolve_fork_resume_model`), so a block
re-derived on a restart would silently re-point the recorded parent at the DRIFTED student — the
exact failure `distill_anchor_callback` has a module of prose defending against. Belt and braces:
`build_lineage` also returns `None` on a same-run restart, decided by
**`main.train.fork_lr.is_same_run_checkpoint`, IMPORTED rather than re-derived** (a second
predicate for the same question is a second answer waiting to disagree; `<run>/warmstart/…` is
deliberately a FORK there, and the seam captures `args.model` BEFORE `--warmstart-consensus`
re-points it, or a warm-started exploiter would record itself as its own ancestor).

**The FRESH form is explicit** (`fork_parent: null, role: "fresh", ancestry: []`) — "no block" and
"no parent" are different facts and only one of them is a measurement.

**THE ACCESSOR is the API** — `agents.training.lineage.fork_parent(run_dir) -> ForkParent | None`
and `.ancestry(run_dir)`. It returns the recorded block's parent when there is one and otherwise
DERIVES it from `original_command`, printing
`[lineage] WARNING: derived from original_command (legacy run, pre-lineage)` to stderr.
`ForkParent.derived` says which, so a legacy guess is never mistaken for a recorded fact. Every run
on disk today is legacy, so the derive path is not a corner case — but it lives in exactly ONE
place, marked as legacy, instead of in each consumer.

⚠️ `distill_anchor_callback.resolve_anchor_parent`'s `original_command` branch is the CURRENT
consumer of that regex and should move to this accessor — same answer, one implementation, and the
recorded block preferred where a run has one.

**The CLI** is `python -m main.lineage <run>…` (torch-free and model-free, so it reads a run whose
architecture drifted past current code — which is most of `models/`). It prints the tree with each
node's `git_hash` / `arch_signature` / `role` / `fork_step`, plus the run's teachers and target, and
**flags a broken link**: a parent run directory that is gone, a parent checkpoint whose sha256 no
longer matches the file on disk, and an `arch_signature` that CHANGED across a link (a fork cannot
have loaded a differently-shaped parent, so the recorded parent is wrong). `--json` for scripts;
exit 1 when anything is flagged. `--backfill` derives a block for a LEGACY run and writes it marked
`"derived": true` — **dry-run unless `--apply`**, and it REFUSES a run that already records one,
because a backfill that overwrote a recorded parent with a re-parsed guess would defeat the point of
recording it.

**The seam is two lines.** `run_io._run_lineage(args, model_dir, model_path=…, fork_step=…)` is the
one place that knows which argparse fields carry the parent, teachers and target; `model_build`
builds the block once per path and passes it to every `save_model_snapshot` call. All the work is in
`agents/training/lineage.py`, which is torch-free and reads a checkpoint's `num_timesteps` out of
the SB3 zip's plain-JSON `data` member via `zipfile` — never by loading the model.

Tests: `lineage_test.py` (41) — the fork block incl. the sha256 and the zip/filename step read; the
same-run restart building nothing and the recorded block surviving a restart byte-for-byte against
a DIFFERENT offered parent; the fresh null form; ancestry over two recorded levels, a legacy stop,
a missing directory, a CYCLE and the depth limit; the accessor's recorded-vs-derived split and its
warning; the three integrity checks; the CLI on a synthetic tree; backfill's dry-run, apply and
refusal; and the seam pins (both build paths ask for it, every save carries it, the pre-warm-start
capture, and `dose` staying current while `lineage` stays immutable).

## The `signal/` group — advantage density × outcome entropy (`gen3_signal_rate_metrics_v1`)

**How much action-attributable learning signal is PPO actually receiving?** Two always-on, flagless
scalar families answer it live. Pure observability: no gradient path, no extra battle, no env call —
a handful of numpy means per rollout.

**THE PAIR IS THE INSTRUMENT. Neither number is readable alone.**

| scalar | recorded by | is |
|---|---|---|
| `signal/adv_raw_std` | `instrumented_ppo/ppo.py::train()` | population std of the rollout's RAW GAE advantages |
| `signal/adv_raw_abs_mean` | same | `E|Â|` — the outlier-robust companion to std |
| `signal/adv_kurtosis` | same | EXCESS kurtosis (Fisher; normal = 0), **scale-free** |
| `signal/outcome_entropy` | `signal_callback.py::SignalMetricsCallback` | `p(1−p)` over a rolling 200-episode window, POOLED |
| `signal/outcome_entropy_{bots,pool,stable,target}` | same | the same, split by `MaskableAgentWrapper.OPP_CLASS_*` |
| `signal/outcome_win_rate`, `signal/outcome_n[_<kind>]` | same | the window's `p` and its depth — so a thin split is visible as thin |
| `signal/outcome_entropy_rung` | `exploiter_ladder.py::ExploiterLadderCallback._record` | `p(1−p)` of the LIVE `--exploiter-ladder` rung's gate window |

### Why the pair — the MIRROR PARADOX

Outcome entropy is **maximal (0.25) against a near-twin**, which is exactly the regime where a
single action's effect on the outcome is *smallest* and the games are closest to coin flips. So a
high `outcome_entropy` is **not** "lots of signal" — it is "lots of outcome VARIANCE", which only
becomes signal to the extent the critic localizes it onto actions. That localization is what
`adv_raw_std` / `adv_kurtosis` measure. Read the 2×2:

| outcome entropy | advantage density | reading |
|---|---|---|
| high | high | decisive moments exist and the critic finds them — healthy |
| high | **LOW** | coin-flip games nothing can be attributed to — the mirror paradox, or a stale critic |
| **LOW** | high | a lopsided matchup, but its few live decisions are sharp — a curriculum problem |
| low | low | the opponent is a wall or a pushover — no gradient to be had |

`adv_kurtosis` is the third axis and it is the one that distinguishes *shape* from *scale*: exploit
signal is sparse — a few decisive turns inside a long stretch of forced or irrelevant ones — so a
healthy rollout is HEAVY-TAILED (positive). Near 0 or negative means the advantage mass is smeared
evenly across decisions, i.e. nothing is being localized even though the std may look fine.

### ⚠️ UNITS — within a run freely, across runs only cautiously

The advantages ride the run's own **PopArt-normalized returns** (`--use-popart`, default on), whose
σ moves over training. `adv_raw_std` / `adv_raw_abs_mean` are therefore in *this run's current
normalized-return units*, not a fixed scale: their TREND is meaningful, their absolute level is not
portable. Across two runs with different reward composition, `gamma`/`gae_lambda`, or PopArt state,
only **`adv_kurtosis`** — scale-free by construction — compares directly.

### ⚠️ This is a TRIPWIRE, not the attributable-share measurement

`signal/` tells you *when* to go and measure; it does not do the measurement. The gold standard for
how much of an outcome was actually action-reducible remains the **offline counterfactual
decomposition** — `python -m main.prober.query falsify-scan` (the luck / unattributed /
proven-`policy_reducible` crater bracket) and `cf_audit.py`. Those re-roll the real dice and sweep
alternative actions; `signal/` only reports the critic's own opinion of its rollout.

### Where each half is measured, and why there

**Advantage density is read ONCE per `train()`, off `self.rollout_buffer.advantages`, BEFORE the
epoch loop** — because that is the last point at which the raw GAE advantages still exist. The
minibatch loop applies `normalize_advantage`, which forces std→1 per minibatch and so *destroys the
quantity being measured*. Composes with `--grad-accum-steps` for free (the read is per-rollout, not
per-optimizer-step) and is untouched by `--compile-trainer` (it is numpy over the buffer, outside
any traced graph). Degenerate rollouts are NaN-safe: an empty buffer publishes nothing, a constant
rollout reports a real std/abs-mean with `adv_kurtosis` **NaN** — TensorBoard drops NaN, so the
curve gaps rather than reporting a fabricated 0.0 that would read as "evenly smeared".

**Outcome entropy rides the info dicts the loop already sees.** `MaskableAgentWrapper.step` publishes
`info["win_outcome"]` (which the win-prob head already used) and, new here, `info["opponent_class"]`
— purely additive keys, so nothing downstream changes. `SignalMetricsCallback` pushes each `done`
into rolling per-kind deques in `_on_step` and records in `_on_rollout_end`.

**`--async-rollout` IS covered.** The stock collector publishes `infos`/`dones` in the callback
locals; `collect_rollouts_async` publishes `wave_infos`/`wave_dones` (a wave = a macro-step over
whichever envs came ready). The callback reads whichever pair is present. Unlike
`WinProbLabelCallback` — which needs the `(step, env)` BUFFER ROW and therefore cannot use the wave
batching at all — outcome entropy is a per-episode aggregate with no row alignment, so the wave form
carries everything it needs. The advantage half is transport-agnostic (both paths call
`compute_returns_and_advantage` into the same buffer). Works under `--debug` — the callback is in
`build_callbacks`' unconditional base list.

**Which opponent splits are REAL.** The wrapper's four `OPP_CLASS_*` values are the whole of what
survives the env-worker boundary — only the integer crosses the pipe. So `bots` / `pool` / `stable`
/ `target` are real, and finer identity is **not**: which *heuristic* bot (the class collapses
random and every heuristic into one), and which *pool snapshot* (its provenance — the step it was
frozen at — is held by `SnapshotPool` in the parent and never reaches the wrapper). `_rung` is the
one finer split that exists, and it exists only because the ladder callback keeps its own per-rung
window in the parent process.

State is process-local and NOT checkpointed — a launcher restart re-warms the windows in a few
hundred episodes, the same contract the noise-scale EMAs take.

Tests: `signal_metrics_test.py` — hand-computed moments, an independently-written closed form, the
sparse-vs-spread kurtosis discrimination at MATCHED std, scale-freeness, every degenerate input,
the rolling-window eviction, the kind routing, both rollout paths' locals, the
`OPP_CLASS_SUFFIX` ↔ wrapper-constant pin, and the byte-identity of `train()` with the read
monkeypatched out.

## The SCAFFOLDING GAUGE — `train/scaffolding_gauge` + `python -m main.scaffolding_gauge`

**How far apart are the two value readouts, and is the gap closing?** The critic estimates the
**shaped** return (PopArt units, `gamma`-discounted); the win-prob head estimates the **game**
(outcome units, no discount distortion, no PopArt drift). Neither is a repair of the other — the
two-head structure is the automatic consequence of choosing shaped rewards. What their DIVERGENCE
measures is the reward scaffolding still doing work, and its trajectory is the registered signal
for when shaping coefficients can begin annealing toward the pure game.

Pure math in **`agents/training/scaffolding.py`** (numpy only, no torch, no filesystem), shared by
the live scalar and the offline CLI so the two can never drift apart.

| scalar | recorded by | is |
|---|---|---|
| `train/scaffolding_gauge` | `instrumented_ppo/ppo.py::train()` | `(1 − Spearman ρ(V, P(win))) / 2` over epoch 0's paired reads. 0 = identical ordering, 0.5 = independent, 1 = inverted |
| `train/scaffolding_rho` | same | the raw ρ, so nothing is hidden by the transform |
| `train/scaffolding_n` | same | rows the ρ was computed from |

**ALWAYS ON when the win-prob head exists** (`--win-prob-mode != none`), flagless, gated on the
head's EXISTENCE and not on `win_prob_coef` — a `read_only` head at coefficient 0 still says
something worth curving. A run with no head publishes **no key at all**, so the curve is absent
rather than flat at zero.

**Where it is read, and why there.** Inside the minibatch loop, right after the win-prob block and
BEFORE the cf-twin fold clobbers the extractor stashes: that is the one place both readouts exist
for the SAME states from the SAME forward (`evaluate_actions` produced `values` and stashed
`last_win_prob_logits`). **Epoch 0 only** — by epoch 3 the policy that produced a pair is not the
policy the pair would be attributed to. The logit is NOT sigmoided: the sigmoid is monotone, so ρ
is identical and float32 ranks never saturate.

### ⚠️ UNITS — the rank form is the ONLY one that is live-legal

`V` is a PopArt-normalized SHAPED return; there is no general unit conversion to a probability. The
live scalar is therefore **rank-based and claims ORDERING only** — nothing about magnitude or
calibration. It also goes **AMBIGUOUS at the PBRS constancy endpoint**: under a good frozen
potential, all evaluative content migrates into the reward stream and `V_shaped` is driven toward a
CONSTANT (ledger db9bb5c), at which point ρ degenerates into noise and a falling curve cannot be
told from V running out of variance to rank with. Read it beside the value-scale meters, and beside
the offline constancy row.

The magnitude question is answered OFFLINE, by `python -m main.scaffolding_gauge <run>`, which
walks the run's own `eval_traces` (model-FREE — recorded `values` + `win_probs`, so it works on a
run whose checkpoints no longer load) and ships **both** gauges per checkpoint step:

* **rank gauge** — the same statistic as the live scalar, unit-free.
* **calibrated-affine gauge** — fit `q = clip(a·V+b, 0, 1)` against the REALIZED per-battle
  outcomes on that slice, then report `rms|q − P(win)|` in probability units. The map is a
  **per-checkpoint FIT, not a conversion**, and it does not transport. Part of every residual is
  the affine family being a worse outcome predictor rather than the heads disagreeing, and that
  part ships as `readout_penalty` = Brier(readout) − Brier(head): a large `rms` with a large
  penalty is a readout finding, not a divergence finding.
* **the constancy sanity row** — `v_std` / `v_iqr` / `dispersion` plus the within-vs-between-battle
  split, i.e. the db9bb5c prediction as a one-line check a frozen-φ arm's battery can quote
  (`--constancy` prints only this block). Low `v_std` with `within_frac ≈ 0` is the look-alike
  FAILURE: V has become a per-battle matchup lookup, not a flattened potential.

Every offline CI is a **CLUSTER bootstrap over BATTLES**, because outcome labels are per-battle and
broadcast to every state; an i.i.d. interval over states would be fabricated tightness of roughly
`sqrt(states-per-battle)`. And the step-to-step curve is **not** a controlled comparison — each
point carries whatever the eval quota sampled at that checkpoint, so a verdict needs arm-vs-control
at matched step.

```bash
python -m main.scaffolding_gauge models/<run>               # table + <run>/scaffolding_gauge.json
python -m main.scaffolding_gauge models/<run> --plot        # + a 3-panel PNG
python -m main.scaffolding_gauge models/<run> --constancy   # only the db9bb5c row
python -m main.scaffolding_gauge models/<run> --reliability --reliability-reweight   # section (4)
```

### The RELIABILITY block (`--reliability`) — the head against the TRUTH, not against V

Both gauges above compare the two READOUTS to each other. `--reliability` adds the third question,
which is the one a calibration gate needs: **how far is the win-prob head from the realized
outcome?** Opt-in, so the default JSON and render are byte-stable (pinned by a test). It emits per
checkpoint, stratified `all` / `bot` / `pool` / per-opponent, with cluster-bootstrap CIs over
battles: `brier`, the Brier **`skill`** score against the slice's own base rate, `ece` / `mce`, the
Murphy **`reliability` − `resolution` + `uncertainty`** split with its binning residual reported,
and the per-bin reliability curve. Math: `scaffolding.reliability_table`.

**Read `resolution`, not `reliability`, as the meter.** A base-rate forecaster scores a perfect 0
reliability and a useless 0 resolution — `designs/learning/win_prob_decomposition.md` axis 2 is the
statement that the blur, not the level, is this project's critic disease, and the 2026-09-06
baseline measured exactly that shape on `ai_v9_59_R2ACTION_0827` (reliability 0.0013–0.0020 against
a resolution of 0.062 / 0.045 out of an available 0.182 / 0.165).

**Read `bot` and `pool` separately; a pooled row describes neither.** Axis 3's ecology split
measured the head's mean bias FLIPPING SIGN between the two populations, so the split is the
default rendering and `opponent_class` is where the `sentinel_*` ⇒ pool rule is declared.

🚨 **`--reliability-reweight` is not optional on this tree's traces.** The eval recorder's quota is
loss-enriched — measured on `ai_v9_59_R2ACTION_0827`, the captured outcome rate is **0.46** against
the same cycles' recorded **0.901 vs bots / 0.702 vs pool** — so an unweighted table scores the head
against a population it was never deployed against. The flag importance-weights each opponent's rows
back to the win/loss mix the cycle itself recorded (weights constant within a
battle, so the clustering survives), and reports Kish `ess` beside `n` so the cost is visible. It
**REFUSES** when the true rates cannot be resolved rather than falling back — an unweighted table
looks identical and answers a different question. The size of the correction is the finding: raw, the
same traces read ECE 0.237 / 0.281 and skill +0.071 / **−0.080**; reweighted, ECE 0.025 / 0.035 and
skill **+0.336 / +0.265**. Reading raw-first inverts the verdict.

**TWO SOURCES for the true rates, in preference order** (`gen3_trace_selection_manifest_v1`). Each
cycle's own **`eval_manifest.json` selection block** wins where it exists — the recorder writes the
per-opponent played/won counts into the same directory as the traces, so there is no cross-file
join and, in particular, no positional sentinel inference. Everything it does not cover falls back
to **`eval_results.jsonl`** with the existing behaviour, unchanged; a run with neither still
REFUSES. On a legacy tree the manifest half is empty, so the numbers this tool prints there do not
move (pinned as a byte-identity test on a synthetic tree). `--reliability` additionally emits a
`trace_selection` block and prints, per step, the capture rates and **which source** the
reweighting used — a step that records no selection is labelled **SELECTION UNKNOWN**, never read
as uniform.

Record: `designs/research_state/measurements/winprob_critic_baseline_2026-09-06/`. The design that
consumes it as a gate: `designs/ai_v12/design_winprob_only_critic.md`.

A run whose `win_probs` column is all NaN (`--win-prob-mode none`) **REFUSES** with that diagnosis
rather than curving zeros — "the two readouts agree perfectly" and "there is no second readout"
must not render the same.

Tests: `scaffolding_test.py` — the three known regimes (monotone ⇒ exactly 0, inverted ⇒ exactly 1,
independent ⇒ ~0.5), affine-rescale invariance, the constant-axis NaN, the affine gauge's
`readout_penalty` convicting the FAMILY on a constructed step function while a linear control
collapses it, the cluster bootstrap beating an i.i.d. one by ~`sqrt(50)`, and the live scalar's
byte-identity + NaN-safety + epoch-0-only read. For the reliability block: a calibrated forecaster
reading REL→0 with RES>0, a base-rate one reading exactly 0 skill (the meter's whole point), the
Murphy identity holding to its own reported residual, `p == 1.0` landing in the last bin rather than
a phantom one, and uniform weights reproducing the unweighted table bit-for-bit.
`main/scaffolding_gauge_test.py` folds a constructed three-regime trace tree end to end, plus a
two-opponent-CLASS fixture where the pooled row INVERTS the bot verdict (the ecology confound as a
test) and a loss-enriched-quota fixture where the reweighting moves the base rate onto the cycle's
and takes reliability from 0.32 to 0.

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

## Hidden-opponent belief aux loss (`--opp-belief-aux-coef`)

The training half of the in-place belief feature (model side in `src/agents/model/CLAUDE.md` →
`BeliefSlots`/`BeliefHead`, v16). Off by default. Two pieces live here:
- **Labels (`gen3_env.py`).** When `emit_belief_labels` (set from `--opp-belief-aux-coef>0`), `step()`
  and `reset()` merge two PRIVILEGED int64 Dict-obs keys into the trainee obs: `belief_species[6]`
  and `belief_moves[6,4]` — the opponent's still-hidden mons (species/move NUMs), sourced from
  `battle2.team` (agent2's own full team). The believed-slot mask is read **straight from the obs
  vector's per-slot `species_known`** (the SAME signal `BeliefSlots` keys its injection on) — single
  source of truth, so the label's believed slots can never diverge from where the model fills
  unknown-mon tokens. The pure builder is `agents.observation.belief_labels`. These keys are
  **training-only** (eval/self-play/inference never declare/need them) and read ONLY by the loss — the
  model forward reads only `obs["observation"]`, so the omniscient labels can't leak. **Fail-loud:**
  `_belief_labels` raises if the obs `species_known` is not leading-contiguous (a broken encoder
  packing invariant), rather than mis-slotting supervision.
- **Loss (`instrumented_ppo.py` `_belief_aux_loss`).** `train()` reads the per-minibatch stashed
  logits (`policy.features_extractor.last_belief_logits`, set by the `evaluate_actions` forward) + the
  label keys, and folds `opp_belief_aux_coef·(species_CE + moves_weight·moves_BCE)` into the loss.
  **Order-invariant (Hungarian / DETR):** the k believed-slot predictions are matched to the k hidden
  mons by per-sample min-CE-cost assignment (k! perms enumerated, vectorised per distinct k), so the
  anonymous slot tokens collectively cover the hidden SET rather than each chasing a reveal-shifting
  fixed target. Perf: species log-softmax on the GATHERED believed slots (not full `[B,6,S]`); moves
  BCE skipped when `moves_weight==0`; accuracy/P-R diagnostics under `no_grad`. **Fail-loud:** an
  out-of-vocab label id (impossible on real Gen-3 nums) RAISES — corrupt num pipeline, not a silent
  drop. Returns `None` on an empty (zero-believed) minibatch to avoid NaN-poisoning.
- **Metrics (`belief/*` — its OWN TB prefix, not `train/`, matching the `grad/`/`popart/`/`win_prob/`
  groups; rendered in the launcher TUI directly BELOW the `train/` block in the train column).** Headline
  `species_acc` + `species_acc_above_chance` (anchored to
  `1/n_species`); `moves_precision`/`moves_recall` (the opaque BCE alone can't tell if the ~4 true
  moves rank high); `coverage` (fraction of decisions with ≥1 believed slot) + `k_mean` (so acc is
  interpretable — k=1 vs k=5 differ); `species_ce`, `moves_bce`, `aux_loss`; plus `mask_rate` — the
  **uniform per-head coverage key** (`gen3_belief_mask_rate_v1`): fraction of the B×6 slot grid the
  head scored this minibatch. EVERY belief head emits it under its own prefix (`belief/mask_rate`
  hidden-team, `belief/spread_mask_rate`, `belief/natureev_mask_rate`, `belief/hptype_mask_rate`),
  comparable across heads and batch sizes where the older `n_slots` counts are not — the label-coverage
  baseline the belief-unification consolidation will judge per-head non-inferiority against. Note the
  conventions TILE: hidden-team masks HIDDEN slots, the spread/nature/hp-type heads mask REVEALED
  ones. **ALL SIX supervised belief losses live in `belief_bank.py`** (the design_unified_belief
  §4 code-shape fold, 2026-08-16): one declarative ROW per head (stash/attr/obs/param arg spec ·
  coef key · metric prefix · the `aux_loss` historic key for hidden-team) and `compute(site=…)`
  loops replace the six inline verticals at their THREE original train() positions
  (`hidden_move` = hidden-team Hungarian + move-belief BCE · `latent` = move-latent grading ·
  `revealed` = spread/nature-EV/hp-type) — the site tag is what preserves the float-addition
  sequence exactly (byte-identical), the old `InstrumentedMaskablePPO._*_loss` statics remain as
  aliases, and a seventh supervised belief is now a row, not a slice
  (`belief_bank_test.py::test_sites_partition_the_registry` pins the partition). **Balance:** the
  shared-trunk grad-balance probe (`grad_balance.py`) reports `grad/species_belief_share` (this CE's
  share of the common trunk-pull total) + `grad/species_belief_policy_cosine` — the principled "is the
  aux DOMINATING / fighting the policy" signal (and `grad/aux_share` for the COMBINED non-RL draw).
  **Tuning is empirical:** start `--opp-belief-aux-coef` small (0.1–0.3) so
  `species_belief_share` lands at a few %; confirm `species_acc_above_chance` climbs in warmup; if the
  policy degrades (`train/approx_kl` spikes, `entropy` collapses, `explained_variance` drops) while
  the share is high, the aux is fighting the actor → lower the coef. `--opp-belief-moves-weight`
  balances CE vs BCE (species dominates at 1.0). Both coefs are **training-only** (like `ent_coef`,
  NOT version-locked); the `opp_belief_slots` arch toggle they imply IS version-checked, and
  `--opp-belief-aux-coef` is **read back from the saved config on a flagless resume** (so a launcher
  restart preserves belief-ON instead of FATALing).
- **Tests.** Unit: `belief_aux_loss_test.py` (Hungarian order-invariance + min-cost-matching, empty
  guard, grad, fail-loud out-of-vocab, perf fast-path), `agents/observation/belief_labels_test.py`,
  `agents/model/belief_slots_test.py` (incl. end-to-end gradient flow through the stash to the belief
  params + shared trunk). **Fuzz** (real bridge battles, no server):
  `poke_env_gaps/belief_labels_fuzz_test.py` validates the emitted labels against the ACTUAL opponent
  team, the single-source mask invariant, the moves-⊆-moveset invariant, and the no-leak width check
  over thousands of live decisions:
  `python src/agents/training/poke_env_gaps/belief_labels_fuzz_test.py [n_battles]`.

## Move-belief reinjection loss (`--move-belief-mode` / `--move-belief-coef`)

The training half of the move-belief feature (model side: `src/agents/model/CLAUDE.md` → MoveBelief,
v17). The predicted moveset is REINJECTED into the opp token (it flows to both heads), AND supervised:
- **Labels (`gen3_env.py`).** When `move_belief_mode != "off"` (or species-belief on), the trainee obs
  carries `belief_moves[6,4]` (hidden slots, shared with the species aux) and — when mode ∈
  {revealed, both} — `known_moves[6,4]`: each REVEALED slot's FULL privileged moveset (so the head learns
  the as-yet-unrevealed moves). Both are training-only, sourced from `battle2.team`; builder
  `agents.observation.belief_labels.build_known_move_labels`. (`known_moves` keeps its name — it holds the
  privileged-*known* moveset of a revealed mon; the `revealed`/`unrevealed` mode names refer to the MON.)
- **Loss (`instrumented_ppo.py` `_move_belief_loss`).** Reads `last_move_belief_logits` + the move
  labels, folds `move_belief_coef · BCE` over two DISJOINT slot populations: **revealed** slots (direct
  multi-label BCE on `known_moves` — slot==species, no matching) and **unrevealed** slots (order-invariant
  Hungarian BCE on `belief_moves` — the believed slots are anonymous; cost is the assignment-relevant
  `-(pred·target)`, a cheap einsum). `mode` selects which population(s) are scored. Mode is read off the
  extractor (single source); coef is a model attr (training-only).
- **Metrics (`belief/move_*`).** `bce`, `precision`, `recall`, `revealed_slots`, `unrevealed_slots`,
  `loss`. The move-loss gradient ALSO reaches the trunk via the reinjection, so it is broken out on its
  own as `grad/move_belief_share` (+ `_norm_shared`/`_policy_cosine`) on the common trunk-pull total.
- **Versioning.** `move_belief_mode` (str) is the version-checked structural toggle (fresh-only;
  `unrevealed`/`both` additionally REQUIRE `--opp-belief-aux-coef>0` so the hidden slots carry
  learned tokens); `move_belief_coef` is training-only, **read back on a flagless resume**. It used
  to auto-force `--attend-unrevealed-opponents`; at v78 that toggle became **config_only frozen ON**,
  so the prerequisite holds by construction and the auto-force branch is deleted. The revealed-vs-unrevealed axis is the defensible-vs-omniscient A/B.
- **Tests.** Unit: `move_belief_loss_test.py` (direct-BCE, Hungarian order-invariance + min-cost match,
  mode gating, grad, fail-loud), `agents/model/move_belief_test.py` (module mask-gating + grad +
  per-mode wiring + off byte-identical), `belief_labels_test.py` (`build_known_move_labels`),
  `snapshot_test.py` (version gate + threading).

## Latent-belief loss — DELETED (v75)

`--opp-belief-latent-coef`, the `opp_belief_latent` arch toggle, the `BeliefHead` SimSiam predictor,
the `belief_target_slots` training-only obs key and the env work that built it are **gone**. Recorded
here because the reasoning generalises to every aux head on this trunk:

- **It was never fed forward.** The latent was a side readout — stashed for the loss, never
  concatenated into `pi` or `vf`. Contrast `--opp-belief-cls-k`, which appends its pooled belief to
  BOTH projections and therefore buys the policy something at inference time.
- **It cost ~13% of the train step.** Measured per-flag on an idle box with interleaved arms:
  marginal **+341 ms** of train time at the production batch, against a `cls_k=6` costing +349 ms
  that *does* feed forward, and a `spread_belief` costing +72 ms. The train step is ~89% of
  production wall at 10 epochs (an EXTRAPOLATION from a measured 61% at 2 epochs — see the
  `--compile-trainer` section for the provenance and its caveat), so this was real throughput.
- **Its own probe had already concluded decodable ≠ helps** (the belief latent/BYOL role-geometry
  probe: species geometry decodes strongly, and nothing downstream was shown to use it).

**Predicting the opponent's unrevealed mons is untouched.** `BeliefSlots` still fills the hidden opp
slots with learned tokens, the species CE and moves BCE still supervise them, and the T0 species
prior still feeds the physics. What is gone is the *second, graded* way of saying the same thing.

Migration: `MODEL_CONFIG_VERSION` 75 REFUSES a config that recorded `opp_belief_latent=True` (the
predictor carried parameters, so such a state_dict has keys the live extractor cannot accept) and
pops it when false. `sanitize_dead_extractor_kwargs` applies the same rule to a saved zip's
`features_extractor_kwargs`.

## Spread-belief supervision loss (`--spread-belief-coef`)

The training half of the THIRD belief leg (model side: `src/agents/model/CLAUDE.md` → SpreadBelief, v25).
The `SpreadBelief` head predicts the opponent's hidden SPREAD (the 5 derived stats {atk,def,spa,spd,spe}) and
the `DamageOperator` consumes it for damage + outspeed. WITHOUT this loss the head is **unsupervised** — it
gets only the weak/unaligned gradient leaking back through the op, so it sits at the usage-mean prior, which
**over-estimates the largest-EV stat** (the modal Smogon set maxes it) → the op mis-prices damage/outspeed
against the *modal* opponent, not the real one. Off by default (`--spread-belief-coef 0`). Two pieces:
- **Label (`gen3_env.py` → `belief_labels.build_known_spread_labels`).** When `emit_spread_labels`
  (= `--spread-belief` AND `--spread-belief-coef>0`), `_spread_labels` (INDEPENDENT of the species/move
  belief path, so `--spread-belief` works standalone) merges two TRAINING-ONLY Dict keys: `belief_spread`
  [6,5] (the TRUE derived stats of each REVEALED opp mon, matched BY SPECIES against agent2's own team's
  computed `mon.stats` — the privileged ground truth Gen 3 hides from the trainee even once the species is
  revealed) + `belief_spread_mask` [6] (1 = supervised). Believed/pad/incomplete-stat slots → mask 0. Read
  ONLY by the loss; the model forward reads only `obs["observation"]`. SPREAD_STAT_ORDER == the op's
  `_SB_ATK.._SB_SPE` consumption order (pinned by `spread_belief_loss_test` — the GIGO/order-mismatch guard).
- **Loss (`instrumented_ppo._spread_belief_loss`).** Reads the extractor's stashed `last_spread_belief`
  [6,5] (the believed stat VALUES the op consumes) + the label keys; folds `spread_belief_coef ·
  smooth_l1((believed − true)/_SPREAD_LOSS_SCALE)` over the masked (revealed) slots. The gradient flows
  believed → `stat_head` → opp tokens → trunk, so it is broken out as its OWN per-head share
  `grad/spread_belief_share` on the common-denominator grad-balance probe (it does NOT gate the
  probe-sample timing — it scores on near-always-present REVEALED slots). **Leak-safe:** the believed
  stats are a MODEL OUTPUT (the op's input), not a label; the true-spread label is training-only, read
  only here.
- **Metrics (`belief/spread_*`).** `mae` (believed-vs-true error in RAW stat points — should fall),
  `largest_bias` (signed error on each mon's LARGEST true stat — the "over-estimates the largest EV"
  diagnostic, → 0 as the head learns), `n_slots` (supervised slots/minibatch), `mask_rate` (the
  uniform coverage key — see the `belief/*` metrics bullet above), `loss`.
- **Nature/EV decomposition (`gen3_nature_ev_belief_v1`, v40, `--spread-belief-nature`).** The fix for the
  stuck `largest_bias`: the additive head predicts the DERIVED stat directly (a point estimate BETWEEN the
  nature ×1.1/×0.9 modes); the generative head predicts a NATURE categorical ⊕ Smogon prior + per-stat EVs ⊕
  prior and COMPUTES the derived stat, so the asymmetry + EV budget are structural. A SECOND loss term
  `_nature_ev_belief_loss` (nature CE + EV smooth_l1 over REVEALED slots, folded at the SAME
  `spread_belief_coef`, metrics `belief/natureev_{nature_acc,nature_ce,ev_mae,n_slots,mask_rate}`) supervises the
  decomposition DIRECTLY (the derived loss alone is many-to-one). Label: the TRUE (nature, EVs)
  **deterministically INVERTED** from agent2's `mon.stats` (`damage_tables.invert_nature_evs`, GIGO-guarded —
  gen3 hides them, so we invert the visible derived stats), emitted by `gen3_env._spread_labels` as
  training-only `belief_nature`/`belief_ev`(+masks), cached per battle. The op-side
  `--spread-belief-nature-marginalize` (an exact 3-point quadrature of P(KO) over the believed nature
  distribution) is **DELETED** (v66): measured on gen-8's own checkpoint across 1,075,200 alive
  (defender, candidate) cells it moved |ΔP(KO)| by 0.00000 at p50/p90/p95 and 0.00047 at p99, because a
  peaked nature posterior (top-1 mass 0.75) makes marginalising ≈ evaluating at the mode. Sound theory,
  absent magnitude — ledger K1's shape. Smoke: `nature_acc` rises toward the true nature,
  `largest_bias` trends to 0.
- **Versioning.** `spread_belief` (the head) is the version-checked structural toggle (v25, fresh-only);
  `spread_belief_coef` is **training-only** (inherited on a flagless resume, like `move_belief_coef`). The
  loss adds NO forward/weight change → no `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` bump (a checkpoint trained
  at coef 0 can resume with coef>0 to start supervising — like enabling any aux).
- **Tests.** Unit: `spread_belief_loss_test.py` (masking, scale-normalised smooth_l1, grad ONLY to
  supervised slots, the `largest_bias` over-estimate detector, off→None, the stat-order GIGO pin),
  `belief_labels_test.py` (`build_known_spread_labels` species-match + mask + incomplete-stat skip). **Fuzz**
  (real bridge battles, no server): `poke_env_gaps/belief_labels_fuzz_test.py` validates `belief_spread` ==
  the actual revealed opp mons' true derived stats (`mon.stats`), believed/pad slots zero (no leak), and the
  OFF env declaring no spread keys, over thousands of live decisions. End-to-end smoke (`--debug
  --unified-moves both --spread-belief --spread-belief-coef 0.1 --n-steps 64`) confirms the roundtrip + the
  loss runs + `belief/spread_*` metrics.

## Opponent HP-type belief loss (`--hp-type-belief-coef`)

The training half of `gen3_typed_hp_belief_v1` (model side: `src/agents/model/CLAUDE.md` → DISCRETE typed
Hidden Power, v51). The opponent's Hidden Power is reasoned about ONLY as the 16 discrete typed moves; the
`HPTypeBelief` head supplies the type half of `P(HP_t) = presence · P(type=t)`, and this CE is its direct
supervision.
- **Label (training-only, privileged).** `Gen3Env._hp_type_labels` reads agent2's OWN team for each
  REVEALED opp mon's true Hidden Power type (the typed move-id suffix → `belief_labels.build_hp_type_labels` /
  `hp_type_idx_from_move_id`, in the `HIDDEN_POWER_TYPE_ORDER` index space) and emits the `hp_type_label` [6]
  / `hp_type_mask` [6] Dict keys (mask=1 only at a revealed slot whose species runs HP). Gen 3 NEVER reveals
  the opp HP type, so this can't ride the obs vector — it is leak-safe (a separate Dict key, read ONLY by the
  loss; the obs vector width is unchanged). Emitted when there is a move belief AND `--hp-type-belief-coef>0`.
- **Loss (`instrumented_ppo._hp_type_belief_loss`).** Reads the extractor's stashed `last_hp_type_logits`
  [6,16] (the prior⊕delta posterior) + the label keys; folds `hp_type_belief_coef · cross_entropy` over the
  masked (revealed-HP) slots. Gradient flows posterior → `hp_type_head` → opp tokens → trunk (joins the
  per-head grad-balance probe as `grad/hp_type_*`); `aux_probe_terms["hp_type"]`.
- **It is no longer the head's ONLY signal.** Since v51 the move-belief BCE labels use each Hidden Power's TRUE
  TYPED num, so the multi-label BCE lands directly on the composed typed channels — which trains the type
  posterior AND the presence channel jointly, through one gradient path. The damage operator's gradient rides
  the same channels. So `--hp-type-belief-coef 0` no longer means "unsupervised": it means "no dedicated CE on
  top". The default is **0.05**.
- **Metrics (`belief/hptype_*`).** `acc` (top-1 HP-type accuracy — should climb well above the 1/16≈0.06
  chance; a short bridge smoke reaches ~0.8 quickly since the head cold-starts at the Smogon prior), `loss`,
  `n_slots`, `mask_rate` (the uniform coverage key — see the `belief/*` metrics bullet above).
  `hp_type_belief_coef` is **training-only** (inherited on a flagless resume, like
  `spread_belief_coef`). The old version-checked `hp_type_belief_mode` is DELETED — the head is unconditional
  whenever there is a move belief, and it no longer requires `--damage-op`.
- **Tests.** Unit: `model/hp_type_belief_test.py` (the Σ-typed-equals-presence constraint, both certain-fact
  eliminations, the immune-bug regression, the op having no HP source of its own, the CE loss masking,
  `build_hp_type_labels`, the 16-axis GIGO pin, the v51 migration). **Fuzz** (real bridge battles): the
  extended `poke_env_gaps/belief_labels_fuzz_test.py` validates `hp_type_label` == each revealed HP-mon's true
  type, the TYPED move labels == the real opponent movesets, mask 0 on revealed-no-HP / believed / pad slots
  (no leak), and the OFF env declaring no HP-type keys. End-to-end smoke (`--debug --use-bridge=node
  --unified-moves both --spread-belief --hp-type-belief-coef 0.05`)
  confirms the roundtrip + `belief/hptype_*`.

## Opponent-class label weight (`--intent-label-bot-weight`, default 1.0 = OFF)

`gen3_intent_label_bot_weight_v1` — a per-sample weight on the opponent-intent (α/β) LABELS
produced against a heuristic **bot**; every other opponent class (pool / stable / exploiter) keeps
1.0. It exists because a bot's tendencies are not the meta's, and the curriculum guarantees the
head meets them first: `heuristic_fraction` is **0% self-play below `SELF_PLAY_START`**, so a fresh
generation trains 100% vs bots until the pool seeds. Measured on gen-11, supervised intent rows ran
**100% bot at 2M and ~7% from 6M on** — and bot rows score differently (info gain 0.124 nats vs
pool 0.254, accuracy flat ~0.50 all run). The risk this knob addresses is imprinting: α/β learning
a decision tree during the ramp and carrying it into pool play.

**The mechanism.** It reuses the EXISTING identity source — the `opp_class` obs key
(`gen3_opp_class_v1`), tagged once per episode by `MaskableAgentWrapper._select_episode_opponent`,
pushed onto the env at `reset()`, emitted beside the α/β labels by `Gen3Env._opp_intent_labels`,
shifted with them by `align_labels_to_predictions`, and already read in `train()` for the
stratified metrics. **No new obs key was added**; the key that splits the dashboards is now also
the key that weights the loss. `agents.model.opp_intent.intent_losses` takes a `bot_label_weight`
and folds it as

```
loss = Σ_i w_i · ce_i / n_sup        w_i = W on bot rows, 1.0 elsewhere
```

— weighted **before the mean, at the unchanged `n_sup` denominator**. Normalising by `Σw` instead
would make a 100%-bot minibatch identical to an unweighted one, i.e. do nothing in exactly the
regime the knob exists for; with `n_sup` a `w ≡ 1` batch reproduces the plain mean, so the
`--opp-intent-coef` semantics are untouched.

**Composition with the masks.** The masks run FIRST. A row masked by `INTENT_IGNORE` (unmodeled
seat, unrevealed β switch-in, non-switch decision) is dropped, and the weight multiplies only the
survivors — a masked bot row contributes nothing at any weight, and `W = 0` legally means "score
bot rows for the metrics, train on none of them".

**It is confined to α/β and that is a design claim, not an oversight.** The other supervised
beliefs — species, move, item, spread, nature/EV, HP-type — are **team truth**: what the
opponent's team IS does not depend on who is piloting it, so discounting a bot's rows there would
throw away valid labels. Only INTENT is behaviour. The `belief_bank` rows never see `opp_class`
(pinned by `opp_class_plumbing_test::test_only_the_intent_loss_takes_the_weight`).

**Diagnostic: `opp_intent/label_bot_frac`** — the bot share of the α rows actually SUPERVISED this
minibatch. The per-class `alpha_n_supervised_*` counts carry the same information but are gated on
≥2 rows and are counts, so nothing reported the ratio. It is emitted **whether or not the weight is
set**, because the decision to set it is made off this number. The existing stratified metrics are
untouched — they measure the head, and a weighted loss must not move an accuracy.

**Default 1.0 is a deliberate no-op.** At 1.0 the original unweighted `cross_entropy` call is taken
unchanged, so the loss is **bit-identical** (not merely close — pinned by exact equality over three
opponent mixes). Lowering it is a **generation/fork decision, not this change**: it moves the
supervision distribution, so it belongs at a launch boundary where it can be attributed.

**Pre-registered decision path.** Decide at the gen-16 launch, beside the B-move supervision call:
run the fork A/B **W=1.0 vs W=0.25**, gated on **`opp_intent/alpha_acc_pool`** (the `_pool` suffix,
never the bare key — the bare one is a moving mix). W=0.25 wins only if `alpha_acc_pool` is
non-inferior or better; a fall there means bot rows were carrying real signal and the knob goes
back to 1.0. `label_bot_frac` sizes the manipulation before the arm is run — if it is already ~0 at
the steps that matter, the arm is not worth a generation slot.

**Class: `training_coef`.** It scales a loss and touches no forward pass ⇒ no `ARCH_SIGNATURE`
bump, not in `check_compatible`, no `check_*` of its own; recorded on `ModelVersion`
(`MODEL_CONFIG_VERSION` v97) for provenance and so a **flagless resume inherits it** via `_resolve`,
exactly like `--td-aux-coef`. It is deliberately NOT in `agents/model/flag_registry.py` — that
registry's scope is extractor architecture toggles, and this reaches the extractor not at all
(same call as `--td-aux-coef`).

Tests: `agents/model/intent_label_bot_weight_test.py` (bit-identity at 1.0 on every mix, the
hand-computed weighted mean, the all-bot scale-down, non-bot classes never discounted, W=0 killing
the gradient, proportional gradient scaling, mask composition on both axes, β taking the same
per-row vector, `label_bot_frac`, the stratified metrics unmoved, the CLI/ModelVersion/migration
legs) and `agents/training/opp_class_plumbing_test.py` (the whole `opp_class` chain, which nothing
covered before it became load-bearing: the two hand-mirrored class tables agreeing, the wrapper tag
per opponent kind, the reset-time push onto the env, the env emission, the one-ahead shift, the
episode-boundary drop, buffer shuffle-alignment on a real `MaskableDictRolloutBuffer`, and the
train-loop call site).

## Win-probability head (`--win-prob-mode` / `--win-prob-coef`)

The training half of the tri-state win-probability head (model side: `src/agents/model/CLAUDE.md` →
win-probability head, v22). A calibrated **P(win|state)** the shaped critic can't give — supervised by the
Monte-Carlo episode OUTCOME. Off by default (`--win-prob-mode none`). Three pieces live here:

- **The label is a FUTURE quantity** — the outcome is only known when the battle ends, so (unlike the
  per-step belief labels, which are privileged info known *each* step) it CANNOT ride as a real per-step
  obs key. The plumbing reuses the obs-dict-label STORAGE path with post-hoc population:
  - **`gen3_env.py`** declares two TRAINING-ONLY obs keys when `emit_win_target` (`--win-prob-mode != none`):
    `win_target` [1] + `win_mask` [1] (float32), and emits PLACEHOLDER zeros each step (`_merge_training_keys`).
    The rollout buffer therefore stores + shuffles them automatically (the belief-label path). Read ONLY by
    the loss; the model forward reads only `obs["observation"]`, so the OUTCOME can't leak.
  - **`MaskableAgentWrapper.step` (`wrappers.py`)** sets `info["win_outcome"]` (1.0 win / 0.0 loss-or-tie,
    from `battle1.won`) at the done step (before the VecEnv auto-resets).
  - **`WinProbLabelCallback` (`win_prob_callback.py`)** captures each terminal outcome during collection
    (SYNC: in `_on_step` at `rollout_buffer.pos`; ASYNC: the `collect_rollouts_async` collector records it
    inline at the env's just-written `(t, i)` buffer row — it owns the row, the wave-batched `on_step`
    can't recover it), into a shared `model._win_terminal_scratch` [n_steps, n_envs]. At `_on_rollout_end`
    (before `train()`) it propagates each episode's outcome BACKWARD to all its steps (γ_win = 1, undiscounted
    → P(win|s) = "probability this state leads to a win") and OVERWRITES the buffer's `win_target`/`win_mask`
    placeholders. The trailing IN-PROGRESS episode (no terminal yet in-buffer) gets `win_mask=0` and is
    excluded — never trained toward a fabricated label. Only added to the callback list when the head is on.
- **Loss (`instrumented_ppo.py` `_win_prob_loss`).** `train()` reads `last_win_prob_logits` (stashed by the
  `evaluate_actions` forward) + `rollout_data.observations["win_target"]`/`["win_mask"]`, folds
  `win_prob_coef · masked-BCE`. read_only vs shaping differ ONLY in whether the extractor stop-grads the
  head's input (the trunk gradient) — the loss term itself is identical. Folded whenever the extractor's
  `win_prob_mode != none` AND `win_prob_coef != 0`.
- **Metrics (`win_prob/*` — its OWN TB prefix, not `train/`, matching the `grad/`/`popart/`/`eval/`
  groups).** Calibration: `acc` (top-1 win/loss) + `brier` (lower = predicted P(win) tracks the win
  rate); `pred_mean` vs `label_mean` (base-rate-collapse watch); `coverage` (fraction with a known label);
  `loss`. **Information value (the aggregate Brier hides it — a blowout's P(win) is trivially recoverable
  from material):** `brier_contested`/`acc_contested` restrict to CLOSE games (`|win_margin| <
  _WIN_CONTESTED_TAU`=0.25, the normalized material margin from `_compute_phi_mat`, emitted as the
  `win_margin` obs key) — judge `brier_contested` vs a 50/50 game's ~0.25 no-skill floor;
  `contested_frac`/`contested_label_mean` (≈0.5 confirms even); and **`skill_vs_material`** = the Brier
  skill score vs a material-only baseline (`P_mat = clip(0.5+0.5·margin)`) — **>0 ⇒ the head adds info
  beyond counting mons** (the headline value number; `brier_material` is the baseline for context). The
  shared-trunk pull rides `grad/win_prob_share` (the `grad_balance_metrics(aux_terms=…)` `"win_prob"` entry) — **≈0 under
  read_only** (stop-grad, the live confirmation the diagnostic isn't perturbing the policy), real under
  shaping (watch it sit small; a spike with a degrading policy → lower `--win-prob-coef`).
- **Versioning.** `win_prob_mode` (str) is the structural + resume-IMMUTABLE toggle (any change FATALs;
  threaded into `current_model_version` / `arch_toggles_from_model` so a win-prob-ON self-play run doesn't
  FATAL on its own sentinels); `win_prob_coef` is training-only, **read back on a flagless resume**.
- **Forensic trace + prober.** `RLPlayer._win_prob` (`inference/player.py`) reads the stashed
  `last_win_prob_logits` at trace-capture time (sigmoid ⇒ P(win)) into the per-decision `state`, which
  `BattleRecorder.states_arrays` writes as a `win_probs` npz array (NaN = no head / not captured, parallel
  to `values`). The prober renders **P(win) + ΔP(win)** in the Summary + Outcome panels beside CRITIC's
  V/ΔV — "how a move moved the win odds" — model-free from that array (`engine.WinProbView`); `None`/absent
  on a non-win-prob run. See `src/main/prober/CLAUDE.md`.
- **Tests.** Unit: `agents/training/win_prob_test.py` (loss masking + None guards + the callback MC-fill
  backward-propagation + in-progress masking + sync-capture-at-pos + async-skip), `agents/model/
  win_prob_head_test.py` (module build, off byte-identical projection dims, the read_only-stop-grad /
  shaping-flows gradient gating, the v22 version gate). End-to-end `--debug --use-bridge=node
  --win-prob-mode read_only` smoke confirms the roundtrip + `train/win_prob_*` metrics + `win_prob_share`=0.

🚨 **`--win-prob-mode shaping` carries NO behavioral force, and the word has misled readers.** It is
**REPRESENTATION** shaping: the BCE gradient reaches the shared trunk, so outcome-predictive features
get a subsidy there. There is no gradient path anywhere from *predicting wins* to *choosing winning
actions* — the logit is a SIDE readout, never concatenated into pi/vf (leak-safety, since its label is
the privileged future outcome), so the policy is free to ignore the subsidised features and V compresses
to its own target regardless. **The head is a BAROMETER, not a coach.** It is also self-referential: its
labels are outcomes under the CURRENT policy, so a habitual whiff that still wins 55% teaches it "55%",
never "the whiff was the mistake". Action-level badness needs a counterfactual contrast the state label
structurally lacks. That is why "shaping has been live for generations and the bait loops persist" was
never a dose mystery — the live mode was never pointed at behavior. The routes that ARE pointed at it
are below and in `designs/ai_v12/design_winprob_behavior_coupling.md`.

## Win-prob PBRS reward shaping (`--win-prob-pbrs-coef`, `winprob_pbrs.py`, ai_v12 route 1)

**OFF by default (`0.0`) and byte-identical when off** — the module is not even imported. Design:
[`designs/ai_v12/design_winprob_behavior_coupling.md`](../../../designs/ai_v12/design_winprob_behavior_coupling.md).
**Nothing has run this; no arm is registered.**

The reward-level route that gives the barometer force. With `φ(s) = σ(win-prob logit)`, DETACHED:

```
r'(s, a, s')  =  r(s, a, s')  +  coef · ( γ·φ(s') − φ(s) )
```

A move that drops the model's own win probability now costs literal reward, and the drop flows through
GAE → advantage → policy gradient. It **SUPPRESSES without knowing the alternative** (softmax
renormalization redistributes the suppressed mass), which is the complement of what a distillation
target does — see the design doc's §2.1.

- **THE SHIELD.** Potential-based shaping (Ng, Harada & Russell 1999) leaves the **optimal policy set
  unchanged** for any *fixed* φ: the shaping telescopes to `γ^T·φ(s_T) − φ(s_0)`, a constant per start
  state. A miscalibrated φ therefore costs learning SPEED, not correctness.
- ⚠️ **THE CAVEAT THE SHIELD DOES NOT COVER: our φ is LEARNED and DRIFTING.** Exact invariance holds
  **per rollout** (PPO freezes the policy during collection and φ is read once, after it, with the
  collection-time weights) and degrades to **approximate** invariance across rollouts, bounded by φ's
  drift over one credit-assignment window. Operationally: **prefer a MATURE base**; a fresh run tests
  the shield's worst case. The one reassuring fact is the G0 bias map's diagnosis — the head's defect is
  **RESOLUTION, not offset** — and a blurry potential is a WEAK one, not a wrong one (a φ constant over
  a set of states contributes nothing over it and cannot mislead within it).
- **WHERE IT RUNS, and why there.** Env workers hold no model, so the reward cannot be shaped where it
  is produced. `InstrumentedMaskablePPO.collect_rollouts` applies it **after collection, before
  `train()`**: read φ for the whole buffer in one batched `no_grad` forward, add the term to
  `rollout_buffer.rewards` **in place**, then RE-RUN `compute_returns_and_advantage`. That window is the
  only one that works — both collectors compute GAE as their last act, and PopArt reads
  `rollout_buffer.returns` at the top of `train()`, so the shaping lands in **RAW reward space** and
  PopArt normalizes the shaped returns (the only order that keeps the value loss in the units of the
  stream being optimized).
- **Both collectors are COVERED, not documented around.** The φ read is a batched re-forward rather than
  a per-step callback capture *because* `--async-rollout` forwards a wave of envs at a time and its
  callback locals cannot recover the env→row mapping (the same reason `WinProbLabelCallback`'s terminal
  capture had to be inlined into `collect_rollouts_async`). One re-forward gives both paths the
  identical quantity, at ≈ one forward pass over the rollout — roughly `1/n_epochs` of one epoch.
- **The two conventions, which are NOT the same case.** **TERMINAL** (`episode_starts[t+1] == 1`, the
  identical test SB3's own GAE uses for `next_non_terminal`, so the two notions of "terminal" cannot
  drift apart): **φ(s′) := 0**, which is what makes the per-episode discounted sum telescope to exactly
  `−coef·φ(s_0)`. **BUFFER-BOUNDARY TRUNCATION** (the episode is still running when the rollout ends):
  φ(s′) is the **bootstrap** φ(s_T) from `model._last_obs`, *not* 0 — forcing 0 there is the classic
  PBRS bug, a phantom penalty for the rollout ending. `TimeLimit.truncated` (the 250-turn deadline)
  arrives as `done=True` and takes the terminal branch, which here is arguably *correct* rather than an
  approximation: that cap IS the forfeit deadline and the reward manager scores it as a real outcome.
- **φ carries no gradient, structurally.** The forward is `no_grad` and the result is numpy before it
  touches the buffer, whose `rewards` is a numpy array — no tensor, no graph, no path back.
- **Config gates (the ONLY gates — nothing version-checks a training-only coefficient).** Negative is
  refused (it inverts the potential; the theorem still holds for `−φ`, so it would train, converge and
  be wrong). `> 0` with `--win-prob-mode none` is refused at config time: the potential IS the head, and
  under `none` no head is built, so the shaping would be a silent no-op. A missing head at runtime is a
  `WinProbPbrsError`, never a skip.
- **Metrics: `train/pbrs_shaping_mean`, `train/pbrs_shaping_absmean`, `train/pbrs_phi_mean`,
  `train/pbrs_reward_share`.** Under `train/` deliberately — this is a property of the reward stream PPO
  is fitting, not of the head. **`pbrs_reward_share` is the one to watch**: mean |shaping| over mean
  |UNSHAPED reward|, i.e. how much of the return signal the coefficient has replaced. Quoted against the
  unshaped stream on purpose, so the ratio does not flatter itself as the coefficient rises.
  ⚠️ **It reads `NaN`, never `0.0`, when the unshaped stream is empty** (R1 adversarial review). Under
  `--no-hand-shaping` the unshaped stream is TERMINAL-ONLY, so any rollout that ends no episode has
  `mean|r| == 0` exactly — and the shaping is then 100% of the reward. The old `0.0` sentinel was the
  reading an operator scans past ("negligible") for the one case where it is everything, in precisely
  the arm the metric exists to watch. Same ABSENT-never-zero rule as `train/q_winprob_loss`.
  **`pbrs_reward_share` is still the WRONG meter for sizing on that stream, and NaN only fixes the
  worst reading.** Where it IS defined its denominator is "±V ÷ episode length", so it moves with the
  EPISODE LENGTH rather than with the coefficient — measured at 2.1-3.1x the true dose across the
  clean-world launch smokes. Hence three companions whose denominator is a CONSTANT — the run's own
  terminal magnitude: **`train/pbrs_episode_dose`, `train/pbrs_episode_dose_n`,
  `train/pbrs_terminal_share`.**
  **`pbrs_episode_dose` is the meter the coefficient ladder is sized in**: the mean |discounted
  shaping sum| of a COMPLETE episode ÷ the terminal magnitude. By the telescoping identity that is
  `coef·E[φ(s_0)]/V` — the shaping's entire per-episode budget priced against one win, i.e. *"this
  run's shaping is worth X% of a win"*. It also checks the telescoping in production rather than only
  in the test: a value that drifts from `coef·phi_mean` means the terminal/truncation convention is
  not doing what it claims on real episodes — and it separates a FROZEN φ from a live one at a glance
  (measured over three launch-smoke iterations: frozen `0.231/0.234/0.228`, live `0.187/0.104/0.087`).
  `pbrs_episode_dose_n` reports the episodes it averaged, so "no complete episode this rollout" never
  reads as "the dose is small". `pbrs_terminal_share` is the per-step companion, always defined.
  The denominator is `model.win_prob_pbrs_terminal_scale`, DERIVED from `--victory-value` in
  `apply_training_hparams` (both build paths) — not a knob, never in the loss. The class default is
  `0.0`, and at `0.0` the two companions are **omitted** rather than divided by a fictitious 30, so a
  smoke/unit test/frozen opponent that never sets it invents no denominator.
- **Versioning.** Training-only, the `td_aux_coef` class exactly: config **v104**, recorded on
  `ModelVersion` for provenance + flagless-resume read-back, never in `check_compatible`, no
  `ARCH_SIGNATURE` bump. Forwarded on both build paths by the one `_TRAINING_HPARAMS` row.
- 🚨 **THE OTHER CONSTANT `--victory-value` SILENTLY INVALIDATES: the distributional critic's
  SUPPORT** — guarded by `_terminal_scale_guards` (R1's F1), which prints
  `[Reward] ⚠️ VALUE-DIST SUPPORT vs TERMINAL SCALE` when the dist head is on, PopArt is OFF and the
  raw-return support either fails to bracket `max(victory, |draw|)` or quantizes it into too few
  atoms. Same genre as the coefficient re-sizing — a constant calibrated against a scale, carried
  across a change of scale. The LAUNCH RULE it implies is carried by
  `designs/ai_v12/launch_runbook.md` §6.3: the guard warns, nothing stops the run.
- **Tests.** `agents/training/winprob_pbrs_test.py` (22): the telescoping identity on a hand case and
  over 40 random episode layouts; the truncation-vs-terminal split; an off-by-one revert-catcher on the
  `episode_starts[t+1]` test; grad-disabled + detached-to-numpy (fails if the `no_grad` is deleted);
  coef-0 buffer identity + the source contract that the import is local to the non-zero branch; the
  raw-reward/GAE-recompute order; chunk-boundary coverage; the loud-refusal path; both config gates; the
  v104 migration; and the frozen-φ group below. Five revert-catchers verified failing on a
  deliberate revert.

### FROZEN φ (`--win-prob-pbrs-source <ckpt>`, `gen3_winprob_pbrs_source_v1`, config v105)

**The caveat above, removed.** The invariance theorem assumes φ is a **fixed** function of state;
ours is a head inside the network being trained. `--win-prob-pbrs-source` points the potential at a
**frozen foreign checkpoint** instead, so the shield holds exactly rather than approximately.
Absent (the default) ⇒ the live head, byte-identical to what v104 shipped.

- **One seam, one loader.** Only `winprob_pbrs.phi_model(model)` changes: it returns
  `model._winprob_phi_source or model`, and `buffer_potentials` / the bootstrap read it. The
  loading is `--distill-teacher`'s path verbatim — `fixed_opponent_pool._resolve_zip_and_config`
  → `snapshot.load_foreign_opponent` → `set_training_mode(False)`, in `main/train/model_build.py`.
  A bad path is `os._exit(FATAL_CONFIG)`, never a crash-restart loop.
- ⚠️ **A FULL frozen extractor forward is required; there is no head-only shortcut.**
  `WinProbHead.forward` consumes `value_pooled` — the whole-board value pool produced by *that*
  network's own trunk with its own weights. Running the frozen head over the LIVE trunk's pooled
  features computes a function of a representation the head never saw, AND it would drift with the
  live trunk, destroying the exact property the frozen source buys.
- **Cost.** The frozen forward **REPLACES** the live-φ one rather than adding to it, so the compute
  is unchanged (~1/`n_epochs` of one epoch). New: one frozen extractor of memory (the
  `--distill-teacher` class, which the tree already runs at N ≥ 3) and one load at startup.
- ⚠️ **Two forwards on the post-rollout obs now, and the split is load-bearing.** `last_values` is
  the **GAE bootstrap** and must stay the LIVE critic's; φ(s_T) must come from the φ network. With
  no source the two coincide and it stays ONE forward exactly as before. Frozen φ on the buffer
  rows with a LIVE φ on the last row would break the telescoping at every truncation boundary.
- **A prior-generation φ is viable and is the point** (that is where a MATURE potential lives).
  `load_foreign_opponent` validates the obs FAMILY (`arch_signature`), and `_phi_obs` passes only
  the keys the source's own space declares — the same filter the distillation teachers use.
- **`--win-prob-mode` governs the LIVE head only** here, i.e. whether it trains as a diagnostic.
  `read_only` is the right choice on this arm: risk-free, and it keeps a live φ trajectory to
  compare against the frozen one — a free measurement of how far the potential has drifted from
  the run's own beliefs.
- **`--compile-trainer` interaction: the source is left EAGER, deliberately.** The compile patches
  the bound `forward` of the LIVE policy's extractor for the per-minibatch train step; the frozen
  source runs once per **rollout**, so a second Inductor graph would buy a warm-up and nothing else.
  ⚠️ **UNEXERCISED:** a real CUDA `torch.compile` with a frozen source attached has not been run —
  `compile_trainer_extractor` refuses a non-cuda device, so the CPU test tier cannot reach it. What
  IS tested is the seam that makes it safe (the compile module never names `_winprob_phi_source`;
  replacing the live extractor's bound `forward` with a poisoned callable leaves φ unchanged).
- **The coefficient carries the [−1,+1] mapping, NOT a `2p−1` spelling of φ.** They are equivalent
  up to `coef ← 2·coef` plus a per-step constant `coef·b·(γ−1)`, and at `b = −1` that constant is
  `+1e-4·coef` per step — small, but a wrongly-signed **stall incentive** in an arm that has deleted
  every anti-stall term. It also breaks `successor_potential`'s `φ(terminal) := 0` convention, which
  is correct for a [0,1] potential and is the MIDDLE of a [−1,+1] one. Write
  `--win-prob-pbrs-coef 2c`; keep φ = σ(logit).
- **Provenance.** Recorded on `ModelVersion` (`win_prob_pbrs_source`) and **inherited on a flagless
  resume** (`_resolve`), because a resume that silently reverted to live-φ would swap exact
  invariance for approximate with nothing saying so. Listed in `_excluded_save_params` — a frozen
  foreign model is never pickled into our checkpoint. Startup prints the resolved zip, its
  `arch_signature` and its `config_version`: a clean-world run is uninterpretable if the identity
  of its frozen potential is not pinned.
- **Config gate.** A source with no positive coefficient is refused — it would load a whole extra
  network, forward it once per rollout, and multiply the result by zero.
- **THE correctness test** (`winprob_pbrs_test.py`): point the frozen source at the run's **own
  current checkpoint**, through the real `load_foreign_opponent`, on a real `Gen3FeaturesExtractor`
  — every φ must come back **bit-identical** to the live path. A head-only shortcut fails it, and so
  does any obs-key or eval-mode discrepancy. Its anti-vacuity twin drifts the live weights and
  requires the frozen φ not to move while the live φ does.

## `stats.py` — the package's SHARED small-sample statistics

**`agents/training/stats.py` is where a stateless estimator lives once the second consumer exists.**
It holds `wilson_ci`, `spearman`, `cluster_bootstrap_ci` / `cluster_bootstrap_diff_ci` and
`sd_true_excess` — pure NumPy in, floats out: no labels, no battles, no checkpoints, no filesystem,
no torch, no RNG except an explicitly seeded bootstrap. That is the admission rule; a helper that
has to know what a *decision* or a *bias map* is belongs beside the instrument that owns the
concept. They were lifted out of `cf_audit.py` on 2026-09-06 (the file-size ratchet's first cut of
the 1,000–2,000 band, 1439 → 1279 lines), which imports them straight back, so
`from agents.training.cf_audit import wilson_ci` — which `cf_producer` and the tests do — still
resolves. The arithmetic is unchanged, and that is EVIDENCE rather than a promise: the parity golden
above was captured before the move and reproduced byte-for-byte after it.

⚠️ **Three near-siblings elsewhere in the tree are deliberately NOT merged into it**, and the module
docstring carries the reasons so nobody "de-duplicates" a shipped instrument's output by accident.
`scaffolding.py`'s `spearman_rho` and its own `cluster_bootstrap_ci` use the **NaN** refusal
convention (TensorBoard drops NaN, so a degenerate slice leaves a GAP in the live
`train/scaffolding_gauge` curve) where this module returns `None` for a JSON report, and its
bootstrap is strictly more general — it resamples ROW INDICES and evaluates an arbitrary `stat_fn`,
which is what lets `reliability_table` compose with it. `winprob_finetune.label_noise_variance`
subtracts the same `p̂(1−p̂)/(n−1)` identity but PER ROW with a heterogeneous `n`.
`main/q_amortization.spearman` is the one true duplicate — same shape, same `None` convention, an
exact `std() == 0` flatness test instead of the relative-tolerance one here — and moving its call
site is a behaviour change (a *near*-flat row starts refusing instead of reporting float noise)
that wants its own pass and its own evidence. `hodge.py` and `elo.py` carry no general-purpose
statistics at all; everything in them is bound to the rating model.

## `cf_audit` — the counterfactual audit instrument (`cf_audit.py`)

```bash
python -m agents.training.cf_audit models/<run> \
    [--rollouts 8] [--states 200] [--step N] [--checkpoint PATH] [--impl rust] [--out DIR]
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
```

**Offline and standalone — it trains nothing.** Given a run's bridge-eval traces (the ones with a
`*_reconstruction.json` sibling) and a loadable checkpoint, it manufactures the value labels the
on-policy stream structurally cannot produce: for a sampled decision it plays the **recorded**
action and rolls the rest of the battle out live **R** times with fresh post-divergence dice
against the RELOADED real opponent, and takes the win rate. Training sees one Monte-Carlo sample of
each state's value; this sees R of the same state.

It emits two things, independently useful:

| output | what |
|---|---|
| `<out>/bias_map.json` + `bias_map.md` | predicted win-prob vs tight-MC per stratum, with `sd_true_excess`, battle-clustered CIs, the sampler design and full accounting |
| `<out>/cf_labels/labels_<producer>_<step>.jsonl` | label rows in the **shared v1 schema** — the contract a training-side consumer reads |

**The meter is `sd_true_excess`, NOT the mean gap.** G0 (2026-08-22) measured the population-mean
predicted−MC gap at |0.05|–|0.07| *with a sign that flips with the population you weight to*, while
the true within-decile spread of P(win) is 0.11–0.36 — the per-state error is 2–6× the aggregate
offset, so the head's defect is **resolution**, not an optimism offset. The estimator subtracts the
R-rollout binomial floor from the observed within-cell variance:

```
Var(MC | cell) = Var(true p) + E[sampling var];   E[p̂(1−p̂)]/(R−1)  is EXACTLY unbiased for p(1−p)/R
sd_true_excess = sqrt(max(0, Var(MC) − E[p̂(1−p̂)]/(R−1)))
```

Subtracting the floor is what makes it a claim about the world rather than about R. **A lever that
merely re-centres the head moves the mean gap and leaves this untouched** — and would be scored a
success by the wrong meter, which is exactly why the meter is stated here.

**The EVIDENTIAL read — the pre-registered meter for `--cf-evidential`, and the reader it was
missing.** The Beta head reads the same `value_pooled` as the scalar one, so it **cannot remove** the
blur G0 measured; the only success available to it is *confessing* it — wide exactly where the states
behind a confidence bin disagree. So the meter is not the loss but
**`width_vs_blur_spearman`**: the rank correlation, ACROSS STRATA, between the head's mean epistemic
width and the measured `sd_true_excess`. When the audited checkpoint carries a `cf_evid_head`,
`cf_audit` forwards it over the labelled states and the resolution table gains `evid_width_mean` /
`evid_precision_mean` columns beside each decile's `sd_true_excess`.

- **Rank, not Pearson** — the claim is an ordering ("wider where blurrier") and the two quantities
  are not on a common scale (a Beta's std vs the within-cell sd of an R-rollout mean).
- **The CI is a bootstrap over BATTLES**, and each draw rebuilds the strata from scratch through the
  same `resolution_cells` the point estimate uses. A draw that loses a thin decile to the minimum-n
  floor is dropped and reported as `draws_usable` — a CI whose resamples ran different arithmetic
  from its point estimate is a CI of nothing.
- **A FLAT width scores `None`, never 0.** "Wide everywhere" and "width unrelated to blur" are the
  same null in outcome but different findings in diagnosis, and the flat-to-1-ulp case (a weighted
  average of a constant) otherwise falls through to a `corrcoef` that divides by ~1e-17 and reports
  a confident correlation of float noise.
- **A checkpoint without the head OMITS the columns** and prints a one-line note. Zeros would render
  "this run has no head" identically to "this head claims no uncertainty". The read is
  **best-effort** throughout: the audit's products are the labels and the bias map, so a model that
  will not load (architecture drift — measured **2026-08-13: 79 of 79 archived runs**; the tree
  carries 100 checkpoint-bearing runs as of 2026-08-23 and the 0-of-N has not been re-measured)
  costs the run its evidential columns and nothing else. `accounting.evidential_scored` says how
  many states were scored.
- Reads the head through `ProbeSession.probe_model()` → `ProbeModel.cf_evidential_batch()`. That
  method exists because the extractor forward **never calls the head**, so unlike `win_prob_at` there
  is no stash to read: it forwards the extractor and applies the head to `stash.value_pooled`
  itself — the same thing `_cf_evidential_term` does, which is what makes the offline number
  comparable with the live `cf/evid_*` scalars.

**Label trust before map trust.** The ANCHOR arm runs FIRST: recorded action + recorded dice must
reproduce the recorded battle outcome. Below `--anchor-tolerance` (default 0.9) the tool exits **3**
and writes NO labels — the bias map is still written, marked `label_trust_passed: false`, for
diagnosis only. A factory whose replay is not exact is GIGO, and a map computed from it measures the
bug. Pinned by `cf_audit_integration_test.py`.

**Selection-awareness.** Eval traces over-capture losses (an explicit win/loss quota), so a pooled
gap convicts the critic of the sampler's sins. Every aggregate is computed *within* an outcome
stratum and recombined at the frame's own population shares; every CI is a bootstrap over
**battles**, never states.

**Sampling** is `(confidence decile × battle outcome × turn tercile)` with a declared
`CONVICTION_BOOST` on the high-confidence-from-lost-battles region (the "0.827 class", the
population R1 supervises). The weights, the seed and `SAMPLER_VERSION` are written into every bias
map — a silent priority change is a distribution-shift confound for every downstream readout.

**The shared label schema (v1)** — one JSON object per line; treat it as a contract, version it
rather than editing it in place:

```json
{"schema": 1, "kind": "mc_winprob", "battle": "<record path>", "decision_idx": 12,
 "obs_sha1": "<sha1 of the obs float32 bytes>", "obs_npz": "<states.npz>::obs",
 "obs_inline": null, "label": 0.625, "n_rollouts": 8, "wilson_lo": 0.30, "wilson_hi": 0.86,
 "policy_step": 24000000, "opponent": "heuristic", "created_unix": 1.77e9}
```

`obs_npz` names the array and `decision_idx` selects its ROW; `--inline-obs` swaps that for a
base64 float32 payload when the traces won't travel with the labels. `obs_sha1` is always present
so a consumer can verify the row it loaded is the row that was labelled.

**Known coverage bounds, printed in the accounting and never silent:** turn-1 decisions (one per
battle, 3.35% of move decisions) and forced-switch rounds (the re-roll layer anchors at
start-of-turn move rounds).

⚠️ **The two have DIFFERENT standing, and the turn-1 one changed on 2026-08-23.** Forced-switch
rounds are a structural limit of the re-roll anchor. Turn 1 is not: it was a rust `search_driver`
defect (`at_turn_start` compared `BattleState::turn`, which still reads 0 at the pre-commit first
boundary), **fixed by `gen3_search_turn1_open_v1`** — both impls now open turn 1, and node always
could. So turn-1 decisions ARE labelable, and the `turn_1_unopenable` skip key is a retained
misnomer for a **sampler** bound (`cf_producer.MIN_LABELABLE_TURN = 2`), not a capability limit.
Lowering it widens the declared candidate distribution by ~3.35% and is deliberately left as its
own change: missing a label is free, silently re-weighting the sampler is not.

**Cost** is the rollouts, not the materializer: an R=8 label is ~0.9 s at load ~7 and ~2.8 s at load
~25 — *more* load-sensitive than `loadavg/cpus` predicts, so any throughput figure taken beside a
trainer is a lower bound. Prefix sharing (below) does not apply to a rollout-to-end label, which has
one arm; it is the lever for the one-ply counterfactual (`lookahead`) path.

**Tests.** `cf_audit_test.py` (pure: the stratifier, the schema writer, and the EXTRACTION PARITY
GOLDEN — every public readout on one synthetic fixture, JSON-serialised canonically and pinned by
digest plus a dozen named values, captured from the tree BEFORE the statistics moved to `stats.py`
and reproduced byte-for-byte after), `stats_test.py` (the estimators themselves — `sd_true_excess`
validated at ZERO true effect AND at a known nonzero one, the clustered bootstrap and its
difference-of-means sibling, Wilson, Spearman) and `cf_audit_integration_test.py` (`sim`: a real
bridge battle it plays itself, run end to end at R=2 — including the anchor refusal).

## `replay_imputation_probe` — the own-side imputation meter (`replay_imputation_probe.py`)

A **meter, not a lever**: it answers "how far would our observation move if the only thing we knew
about our OWN side were what a public Showdown replay had shown by now?" — the #1 risk in
`designs/research_state/metamon_replay_feasibility.md` §2.7, which Metamon cannot quantify because
they have no ground truth and we can because we own a simulator. There is **no transcoder and no
`|request|` synthesis here**; those are that memo's Gap 1 and deliberately out of scope.

```bash
export PYTHONPATH=$PYTHONPATH:src
python src/agents/training/replay_imputation_probe.py 20 [--impl rust] [--json out.json]
```

**Mechanism.** It plays reproducible bridge battles (the `record_fixture_battle` recipe — pinned
teams, a per-player RNG, a fixed sim seed) with a seeded-random policy that decides on the TRUE
obs, and at every decision: encodes truth cold (`assembler=None`), snapshots our own mons,
overwrites their **not-yet-revealed-by-then** moves / item / EVs+nature with the top Smogon-prior
candidate for the species, bumps `Gen3Battle._state_epoch` so the `live_view` memo rebuilds,
encodes again, restores, and bumps again. Truth is re-encoded a THIRD time after the restore and
required to be bit-identical — the meter mutates the live battle, so a leaked restore would make
every later decision's "truth" a previous decision's imputation, and that is gated rather than
believed.

**Reveal tracking models a REPLAY, not poke-env** (`track_own_reveals`, pure over raw protocol
lines, 37 unit tests in `replay_imputation_probe_test.py`): moves on use, item on activation,
spreads never. The sharp edges each have a test — Sleep Talk's callee IS in the user's set,
Metronome's and Mirror Move's are NOT, Struggle is never a set move, Knock Off on the opponent
does not disclose ours, and the species comes from the switch DETAILS (this pool carries localized
nicknames, the same trap `search_dividend.determinize` documents).

**Result (2026-08-24, 20 battles / 2,640 decisions), recorded in full in the memo.** The error is
structurally confined to the our-team block — opp_team, active context, global env, pair history
and the H-B event window are *exactly* zero everywhere, being opponent- or log-derived. Inside it,
`moves` carries almost all of it (relL2 0.56 early → 0.36 late), `items` is ~free in gen3ou
(0.036), and **`spread` is a flat floor that never decays** (0.268 / 0.268 / 0.256) because no
battle progress ever reveals an EV spread. The early-game confound is confirmed at ~2.7× (whole-obs
relL2 0.364 at turns 1–5 vs 0.136 at 16+). ⚠️ The `reactive.active_req_moves` row it prints is an
ARTIFACT of holding the request at truth and is marked `*` in the output — read its direction,
never its size.

## Prefix-sharing materialization (`obs_materializer.materialize_branches`)

K counterfactual arms of one decision share an identical prefix, and the materializer used to
replay it from turn 1 for **every** arm — the measured bottleneck of the counterfactual label path
(`arm_ms = 4.78 + 0.853·turn`, of which prefix replay is `2.53 + 0.855·turn`; the branched turn is
~0.5 ms and the obs encode ~1.8 ms). `materialize_branches` replays the prefix once, snapshots the
player's whole battle/tracker state at the branch decision, and restores it per arm.

- **Contract: exactly equivalent to per-arm `materialize_decisions`, bit-for-bit.** Measured on 6
  gen-17 eval battles / 59 decisions / 452 arms: **59/59 byte-identical**, **15.4 → 5.3 ms per arm
  (2.91×)**, rising with the branch turn (3.7–3.9× at turn 26–28) because the part it removes is the
  part that is linear in the turn. Gate: `obs_materializer_branch_integration_test.py`, which
  compares EVERY arm rather than a sample.
- The clone SHARES append-only immutable records (`BattleEvent`, `BattleContext`) instead of copying
  them — a **contract, not an inference**, and the reason the gate compares every arm: a broken
  contract shows up as arm 2+ reading history arm 1 mutated.
- **The per-arm RESTORE is serialized ONCE and rebuilt per arm, not deep-copied per arm**
  (`_PlayerSnapshot._freeze`, 2026-08-23). Once the prefix is shared, `restore` becomes the single
  largest cost in the loop: measured on a live search-dividend oracle decision it was **3.69 ms of
  the materializer's 6.45 ms per arm — 57% of it**, because a restore is three `deepcopy`
  traversals of the battle graph and deepcopy re-walks and re-dispatches every node every time.
  Pickling each master once at snapshot time and `loads`-ing per arm measures **1.98 → 0.22 ms
  (9.1×)** on the same graph against a one-off 0.66 ms to freeze. Equivalence rests on three
  things: **three separate blobs** (one per structure, reproducing the three independent memos —
  a single blob would ALIAS the 12 objects reachable from both `battles` and `trackers`); **pins
  honoured via `persistent_id`**, so a `Logger` / `MappingProxyType` / immutable record comes back
  as itself; and `GenData` added to the pin set, because it declares itself a singleton with
  `__deepcopy__` and pickle honours no such hook. A graph that will not pickle **falls back to
  deepcopy and says so once on stderr** — a 9× regression nothing mentions is the failure shape
  this tree keeps eating. Gates: the every-arm bit-identity test above, plus
  `obs_materializer_test.py` for the graph contract.
- `lookahead` uses it for its whole `(candidate × seed)` sweep.
## Counterfactual win-prob grounding (`--cf-records` / `--cf-winprob-coef`, `gen3_cf_label_plumbing_v1`)

The **trainer-side plumbing** for `designs/ai_v10/design_counterfactual_value_grounding.md` — its gate
**G3**, which is explicitly "tap + buffer + flags at coefficient zero, byte-identity gated". Rung **R1**
only: tight Monte-Carlo P(win) labels, delivered to the **win-prob head**. The label PRODUCER is a
separate, out-of-process program (`cf_producer.py`, § *The label PRODUCER DRIVER* below);
**nothing in this section produces a label**, and the two halves share only a file format.

**Why the win-prob head and why head-only first.** The G0 bias map (ledger 2026-08-22; 2,204 tight-MC
labels) found the head's defect is **RESOLUTION, not an optimism offset** — population-mean gaps are
|0.05|–|0.07| while the true within-decile spread of P(win) is 0.11–0.36, 80–95% of it real
state-to-state variance. Only tight-MC labels carry that within-bin separation; a single realized
outcome (what the on-policy BCE eats today) structurally cannot. The head is MC-native, so R1 needs no
route change and owes no C4 gate. `--cf-head-only` defaults **TRUE** because the safe stage comes first:
the term trains the head's own params and provably cannot perturb the trunk.

**The four pieces:**

- **The record TAP (`cf_records.py`, `--cf-records`, default OFF).** The bridge emits a `__RECON__`
  reconstruction record at the end of every episode; **training discards it** (`BridgeSession` keeps a
  single overwritten slot), which is precisely why a label producer cannot reach a training decision.
  `--cf-records` threads a `recon_sink` callable into `attach_bridge_transport`, and each env worker
  writes the record into `<run_dir>/cf_records/` as a **count-capped ring** (`--cf-records-keep`,
  default 512). Crash-safe (`.tmp` + `os.replace`), filenames sort chronologically
  (`<time_ns>_<pid>_<tag>_reconstruction.json`) so the prune needs no `stat`, and the cap is **GLOBAL** —
  every worker prunes the shared dir and a lost delete race is swallowed, which is what keeps the bound
  across `n_envs` AND across launcher restarts. **The cap only bounds the directory because the `.tmp`
  is bounded too**: `prune` matches on `RECON_SUFFIX`, so a `<...>_reconstruction.json.tmp` is invisible
  to it — a failed write therefore unlinks its own tmp, and `prune` additionally sweeps tmps OLDER than
  the oldest kept record (a crash between `open` and `os.replace` cannot unlink its own; a tmp being
  filled right now is newer than every record on disk, so the sweep can never race a writer). Without
  that, the full disk this module promises to survive leaked one file per episode per worker, forever.
  **The automatic prune is THROTTLED to one write in `prune_every` (16).** It is a full `readdir`
  running on the bridge reader's coroutine — the path every env step waits behind — and pruning per
  write paid that scan ~512 times to delete ~512 files. The price is a bounded transient overshoot
  (≤ `prune_every` unpruned writes per live writer) and it degrades gracefully **because the cap is
  global**: every writer sweeps the WHOLE directory, so one worker's next sweep collects every other
  worker's backlog, and a process that dies mid-backlog has its leftovers collected by the next
  one's first sweep. Bound: `keep + prune_every·n_writers` transiently, `keep` again the moment any
  writer sweeps. `prune()` itself is unthrottled (a caller that wants the cap now can have it).
  The artifact shape is byte-for-byte the one
  `reconstruction._write_artifact` writes, so `ReconstructionRecord.load()` reads a ring file directly.
  A write failure warns once and is swallowed — a full disk must not crash a run. `--cf-records`
  without `--use-bridge` is REFUSED (a websocket run emits no such frame; the flag would be a silent
  no-op).
- **The LABEL BUFFER (`cf_label_buffer.py`).** Watches `<run_dir>/cf_labels/labels_*.jsonl`, remembering a
  per-file byte OFFSET so an appending producer is read incrementally and a partial trailing line waits
  for the next poll instead of counting as malformed. **The offset is keyed on `(name, inode)`, and the
  map is pruned to the files still on disk** — a producer that DELETES and RECREATES `labels_x.jsonl`
  (an in-place rotation) gets a new inode, and keying on the name alone made the buffer seek past the
  new file's first `offset` bytes and drop those rows with no counter and no warning. "Never a silent
  accept" has a mirror: never a silent DROP. Schema v1 is in the module docstring; obs resolve
  `obs_inline` > `obs_npz` > skip. **Everything unexpected is a COUNTED skip, never a crash and never a
  silent accept**: unknown `schema`, unknown `kind`, malformed JSON, out-of-range label, unresolvable
  obs, an obs whose width ≠ this run's, and an `obs_sha1` that disagrees with its own bytes (the GIGO
  guard — it warns loudly once). `obs_npz` resolves `<path>::<key>` and **`decision_idx` selects the
  ROW** of a 2-D array (which is what `cf_audit` emits by default, one battle's whole obs matrix per
  row) through a small per-file LRU, so N rows of a battle open the archive once instead of N times.
  FIFO at `capacity`, and **staleness expiry** at `--cf-label-lag-steps`
  (default 150 000 ≈ one production PPO iteration): `age == bound` survives, `age == bound + 1` does not,
  enforced at ingest AND on every poll. `0` disables expiry.
- **The label-QUALITY trio (task #28), landed before the coefficient ever goes live.** At coefficient
  zero none of these costs anything; the moment the term is on, each is a silent change to what the
  critic is taught.
  - **DEDUP on the obs digest, keep-NEWEST.** A producer that re-labels a decision it already shipped
    (an overlapping cycle, a re-run over the same trace tree, a truncate-and-rewrite) would give that
    one state N× the weight of every other — a change to the sampler's declared distribution with no
    flag and no counter, which design decision-of-record 3 forbids. The resident row is REPLACED, not
    appended beside. Keep-newest because a fresher label is a strictly better estimate of the same
    state (measured under a policy closer to the consumer, and carrying more evidence if R grew), and
    the replacement re-enters at the FIFO tail rather than inheriting the old row's position.
    `cf/labels_replaced_total`. Measured before: a 5-row file rewritten in place left fill **6**.
  - **SYMMETRIC staleness** — the bound is on `abs(current_step − policy_step)`. A crash-restart
    resumes from the last checkpoint, so `num_timesteps` moves BACKWARDS while the label files still
    carry pre-crash steps; under a one-sided test those rows are **immortal** and quietly become the
    whole buffer. Live tell, measured: `cf/label_age_steps_p50` reading **−4,999,000**. Future rows
    expire like stale ones, are counted separately (`cf/labels_future_total`) and trip a one-time
    loud warning naming the cause — a negative age is a diagnosis, not noise.
  - **The ObservationDebugger is SUPPRESSED around the CF forward** (`--no-compile-trainer` runs, the
    only ones that still have it). The CF rows are recorded FOREIGN states — other episodes, other
    policy steps, read off disk — and the debugger's premise is "this is the board we are about to
    act on"; it was being handed 256 replayed rows per minibatch and reporting their integrity
    against the live env's expectations. `Gen3FeaturesExtractor.suppress_observation_debugger()` is a
    context manager that restores on the way out (including on an exception) — deliberately NOT
    `disable_observation_debugger()`, which is permanent and is the compile path's trade.
- **The LOSS (`instrumented_ppo._cf_winprob_term`).** Per minibatch — the `_td_aux_term` / search-teacher /
  OPD shape, and for the same reason: the labelled states are recorded PAST decisions, absent from this
  rollout, so they cannot ride `rollout_data`, and a once-per-`train()` fold would make the coefficient
  mean something different from every other aux. `_cf_sample_and_forward` samples up to `CF_SAMPLE_SIZE`
  (256) rows and runs ONE extractor forward (`{"observation": …}` is the only key the model reads); the
  term applies the win-prob head to `stash.value_pooled` — **detaching iff `cf_head_only`**. It
  The forward runs under **`no_grad` unless something downstream actually wants the graph** — the
  condition is computed exactly (`cf_head_only` OR a dead `cf_winprob_coef`), not assumed, because the
  one arm that needs it is `--no-cf-head-only` with a live coefficient, and silently dropping the
  graph there would turn the trunk-open A/B into two copies of head-only. Both heads still train
  their own params either way: `head(value_pooled)` is applied OUTSIDE the context, which is pinned
  on the parameter update rather than argued. It
  deliberately does NOT read `last_win_prob_logits`: that stash is produced under the extractor's own
  `win_prob_mode`, which governs the ON-POLICY win-prob BCE; this term's trunk exposure is a separate
  decision, and re-applying the head makes the two independent by construction. It CLOBBERS the
  minibatch's extractor stashes, so it is folded beside `_td_aux_term`, after every loss that reads one.
  The **evidential term (below) shares that ONE sample and that ONE forward** — two samples would pay
  twice for the block's whole cost and would make the two terms disagree about which states they scored.
- **The scalars.** `cf/*` is **producer liveness and is published whenever a buffer exists**, even if not
  one label ever arrived — `cf/buffer_fill`, `cf/label_age_steps_p50`, `cf/labels_ingested_total`,
  `cf/labels_expired_total`, `cf/labels_future_total`, `cf/labels_replaced_total`,
  `cf/labels_skipped_total`, plus `cf/rows_sampled` (rows the fold actually CONSUMED this `train()`,
  summed over minibatches — residency and throughput are different questions, and only the second
  goes to zero when a producer dies while its last labels are still resident). That is deliberate: an empty buffer that does not
  announce itself is this tree's oldest failure mode (the search-teacher's silent starvation), and a flat
  `labels_ingested_total` is unambiguous evidence the producer stopped, which reads completely differently
  from a rising `labels_expired_total` (a producer that is running but lagging). `train/cf_loss` +
  `train/cf_grad_share` are the TERM, only when it folded; `cf_grad_share` is lifted from the
  grad-balance probe's shared denominator (so it is comparable with `grad/policy_share`) and reads
  **exactly 0.0 under `--cf-head-only`** — that is its verification, not a defect.

**Flag class — the `td_aux_coef` class** (`gen3_cf_coef_provenance_v1`, config **v100**). All four are
**training-only** — no forward, no weight shape, not in `agents/model/flag_registry.py` (which declares
EXTRACTOR toggles, and none of these builds a module), and **never in `check_compatible`**: a frozen
eval/pool/distill opponent runs no loss at all, so gating a loss coefficient there would be a false
rejection that breaks league play. But they ARE `ModelVersion` fields, recorded for provenance and
**read back on a flagless resume** via `_resolve`.

> ⚠️ **They were the `--opd-coef` class until 2026-08-22, and the failure that bought the promotion is
> invisible by construction.** An R1 arm resumed without re-typing `--cf-winprob-coef 1.0` kept
> training, kept logging, and simply stopped applying the term it was launched to measure — no error,
> no FATAL, just a metric that goes quiet. It was strictly worse than a symmetric loss, because the
> three STRUCTURAL cf flags (`--cf-evidential` v98, `--cf-twin-heads` / `--cf-shadow-critic` v99) were
> already recorded and GATED, so a flagless resume kept the HEAD and dropped the COEFFICIENT that
> drives it. "The launcher forwards every non-launcher flag verbatim" was the old mitigation, and it
> only ever covered a launcher-managed resume — never a bare `train_rl_agent.py --model …`.
>
> The same pass found the enabling defect underneath: `--cf-evidential` / `--cf-twin-heads` /
> `--cf-shadow-critic` each HAD a `_resolve` line and each had an argparse `default=False`, and
> `_resolve` only fires on `None` — so the line was dead and the presence test that checks for it
> passed anyway. `flag_registry_test.test_cli_flags_argparse_default_is_none` is now the gate for the
> reachability half; it found three more live flags in the same state (`value_threat_inject`, ON in
> the gen-17 production config, and `opp_intent_coef`, which `opp_intent` is DERIVED from — both would
> have made a flagless resume of PRODUCTION FATAL at `check_compatible`).

`--cf-winprob-coef > 0` REQUIRES `--win-prob-mode read_only|shaping` — `none` does not build a
`WinProbHead`, so a live coefficient would fold nothing for a whole run; the parser refuses it, and the
loss independently no-ops if the head is somehow absent.

### The LIKELIHOOD: `--cf-label-likelihood {binomial,bce}` (default **`binomial`**, `gen3_cf_binomial_likelihood_v1`)

The label schema carries `label` **and** `n_rollouts`, so the row's win COUNT is recoverable —
`w = round(label · n_rollouts)` — and the flat BCE was throwing that away. A 0.75 label from 4
rollouts and a 0.75 from 16 are the same number carrying **four times the evidence**; scoring them
identically is a modelling error, not a weighting preference.

```
w = round(label·n)            NLL_i = −[ w_i·log q_i + (n_i − w_i)·log(1 − q_i) ]
term = Σ NLL_i / Σ n_i        (mean NLL per ROLLOUT)
```

- **`binomial` is the DEFAULT**, and that is a deliberate break with the usual "new option defaults
  to old behaviour" rule: `--cf-winprob-coef` has never been live in a production run, so there is
  no trained behaviour to preserve and nothing to be compatible with. `bce` stays as the explicit
  A/B arm.
- **The normalization is `Σ NLL / Σ n`**, not `Σ NLL` and not `/mean(n)`. Two properties buy it: a
  producer that changes its R does not silently change the effective coefficient, and **at `n ≡ 1`
  it reduces EXACTLY to the mean BCE** the flat path computes (a one-rollout label is already 0 or
  1, so the round is the identity and `Σn = B`). That exact agreement is pinned bit-for-bit, which
  is what makes `binomial` a strict generalisation rather than a different objective.
- Computed through `softplus` (`−log σ(z) = softplus(−z)`), stable where `log(sigmoid(·))`
  underflows. A row whose producer omitted `n_rollouts` parses as 0 and is clamped to **one**
  observation — never a divide-by-zero, never a silently dropped row.
- Training-only, the `td_aux_coef` class: no forward, no weight shape, never gated — but recorded
  (config v100) and **read back on a flagless resume**.
- `cf/n_rollouts_mean` rides beside `cf/loss` — under the binomial likelihood the loss is per
  rollout, so a producer that quietly changed R would otherwise move the loss with no visible cause.

### The EVIDENTIAL Beta head: `--cf-evidential` + `--cf-evidential-coef` / `--cf-evidential-reg` (`gen3_cf_evidential_head_v1`, v98)

**What it is for, and what it is NOT for.** G0 convicted the win-prob head of **RESOLUTION**: the
population-mean gaps are |0.05|–|0.07| while the true within-decile spread of P(win) is 0.11–0.36.
A point estimate cannot represent that spread at all. This head reads the same `value_pooled` and
therefore **cannot remove the blur** — it has no information the scalar head lacks. What it can do
is **CONFESS** it: emit a Beta whose width is large exactly where the states behind a confidence bin
disagree. A confessed width is actionable (the factory's priority sampler can label the states the
critic knows it cannot separate; the awareness stack can read it); a point estimate that is silently
wrong is not.

- **`CfEvidentialHead` (`agents/model/aux_value_heads.py`)** — the `WinProbHead` bottleneck widened
  from 1 logit to 2, mapped by `softplus(·) + 1` so **α, β ≥ 1**: the Beta stays UNIMODAL (α<1 puts
  mass at an endpoint, turning "uncertain" into "certain of both extremes") and the uniform
  `Beta(1,1)` is exactly reachable, so maximum ignorance is a representable state.
- **The loss is the Beta-Binomial MARGINAL likelihood** of the row's counts — `p` integrated out,
  not plugged in: `NLL = −[log B(α+w, β+n−w) − log B(α, β)]` (lgamma-based; `log C(n,w)` is dropped
  as a constant in α,β). That is the correct evidential objective for count data, and it does two
  things at once: pulls the mean toward `w/n` AND grows the precision `α+β` only as far as
  consistency across states supports. Normalized by `Σn` like the scalar term, so the two
  coefficients are in the same units (nats per rollout). Checked against
  `scipy.stats.betabinom.logpmf`, not against a re-derivation of itself.
- **`--cf-evidential-reg` (default 1e-3) is the standard evidential-overconfidence guard**:
  `KL(Beta(α,β) ‖ Beta(1,1))`, closed form via digamma/lgamma, exactly 0 at the reachable floor. It
  rides INSIDE the coefficient, so coefficient zero kills the regularizer too. Nothing in the
  likelihood bounds `α+β` on locally-consistent data, and an inflated precision makes the width —
  the entire product — meaningless.
- **ALWAYS DETACHED, with no mode to change that.** Unlike `win_prob_mode` / `value_dist_mode` there
  is no read_only/shaping split: the head feeds nothing forward, so letting it shape the trunk would
  be a training change with no consumer to justify it. `train/cf_evidential_grad_share` reads
  **exactly 0.0 by construction** — published so the contract is a live measurement, not a docstring.
- **It is not called by the extractor forward at all** (the training-side term applies it to the
  stashed `value_pooled`), and it is built **LAST** in `Gen3FeaturesExtractor.__init__`. So OFF is
  byte-identical AND **ON-at-coefficient-0 is BIT-identical in pi/vf** — a stronger claim than the
  two precedents make, and one that depends on the build order: a module inserted mid-constructor
  shifts the init RNG stream for everything after it.
- **Metrics `cf/evid_*`**: `nll`, `reg`, `alpha_mean`, `precision_mean` (α+β — the claimed
  evidence), `epistemic_std_mean` (**the headline**), `pred_mean`, `n`; plus
  `train/cf_evidential_loss` and `train/cf_evidential_grad_share`. Read `nll` and `precision_mean`
  together: a falling NLL with a runaway precision is the head buying its loss with certainty it has
  not earned. A per-decision `(α, β)` stash lands on `fe.last_cf_evidential` for a future trace
  capture; **the npz capture itself is NOT wired** (deliberately deferred). ⚠️ Note when picking that
  up: the stash is written **only by the train loop**, so wiring it through `RLPlayer` would capture
  nothing — the extractor forward never calls the head, so an honest per-decision capture has to
  CALL it at record time (as `ProbeModel.cf_evidential_batch` does) and add an npz key.
- 🔒 **THE PRE-REGISTERED READ, for the experiment that has not run yet:** the predicted Beta's
  width should **CORRELATE with the measured `sd_true_excess` per stratum** (the `cf_audit` bias
  map's meter). Wide everywhere and wide nowhere are the same null. A falling `nll` with a flat
  width-vs-`sd_true_excess` correlation is the standing learns≠helps kill, not a result.
  **That correlation now has a reader**: `cf_audit`'s `width_vs_blur_spearman` (§ *The EVIDENTIAL
  read* above) computes it with a battle-clustered bootstrap CI, so the meter is an instrument
  rather than an intention.

**Flag class — the split, and why.** `--cf-evidential` is **STRUCTURAL** and IS in
`agents/model/flag_registry.py` (v98, `cli`/`structural`): it is a `Gen3FeaturesExtractor`
constructor kwarg that builds a MODULE, which is exactly the registry's declared scope, and the
`win_prob_mode` / `value_dist_mode` precedent. It gets a `ModelVersion` field, a `check_compatible`
bool compare, a `MODEL_CONFIG_VERSION` bump to **98** with a migration defaulting pre-v98 configs
OFF, and a `snapshot.current_model_version` keyword (so a frozen eval/pool opponent's gate sees it).
**No `ARCH_SIGNATURE` bump** — optional side head, obs family unchanged, the value_dist precedent.
The gate matters more here than usual: because the head is never called by the forward, a mismatched
resume produces **no shape error anywhere**, so `check_compatible` is the only thing standing between
a flipped flag and a run that silently supervises a freshly-random head for good. The two
**coefficients** are training-only (the `td_aux_coef` class): deliberately NOT in the registry — they
are loss weights set on the model, never reaching the extractor — but RECORDED on `ModelVersion` and
`_resolve`-inherited since config **v100**, so a flagless resume cannot keep this head and drop the
coefficient that supervises it.

`--cf-evidential-coef > 0` REQUIRES `--cf-evidential`, refused at the CLI. Unlike the win-prob case
the head cannot be added later to rescue a live coefficient: it is a state_dict change, so the
mistake would cost a whole run AND FATAL the resume that tried to fix it. The `cf_labels/` directory
is created when **either** consumer is live, so an evidential-only run is not silently starved.

**Gates.** `instrumented_ppo_test.py` pins the byte-identity that G3 is: a POPULATED buffer at
`cf_winprob_coef=0` yields the same parameter update as no buffer at all (the fold is gated on the
COEFFICIENT, not the buffer), and so does a live coef with no head. The two `cf_head_only` halves are
measured on the parameter update rather than asserted about a detach call — head-only moves the head and
leaves the trunk bit-identical; `--no-cf-head-only` moves the trunk. The same file pins the binomial
likelihood's exact properties as pure-function facts (`binomial == bce` bit-for-bit at `n≡1`; the
gradient ratio is exactly `n₂/n₁`; per-rollout normalization; `w` recovery; the `n=0` degradation) and
the evidential fold's three (ON-at-coef-0 byte-identical with the head in the optimizer; a live
coefficient reaching ONLY `cf_evid_head` — trunk AND win_head bit-identical; one shared sample and one
shared forward for both terms, counted). `agents/model/cf_evidential_head_test.py` holds the head's
maths (scipy cross-check, the hand-computed uniform-Beta anchor, `KL(Beta(1,1)‖Beta(1,1)) == 0`, the
regularizer actually moving α,β toward 1, the 1/√12 std anchor), the BIT-identity of ON's pi/vf, that
the forward never calls it, and the v98 gate + both migration legs.
`cf_label_buffer_test.py` covers FIFO, the exact expiry boundary (past AND future, both inclusive),
incremental polling, the partial-line case, every skip counter, dedup keep-newest + the
rewrite-converges case, the `obs_npz` row index and its per-file cache bound, the ring's
cap/atomicity/race-tolerance, the prune throttle's declared overshoot bound, **the launcher-restart
cap across sequential processes** (the one G3 sub-claim that used to stand on construction alone),
and that `batch_tensors` carries the rollout COUNT rather than just the ratio. The CF forward's two
guards are pinned in `instrumented_ppo_test.py` on the *stashed tensor* and the *parameter update*
rather than on a `with` statement: no graph under head-only, a graph in the trunk-open arm, both
heads still receiving their own gradients under `no_grad`, and the debugger suppressed-then-restored
(including on an exception). `main/cf_flags_test.py` covers
the defaults, both `--no-` spellings, the three new refusals and `checkargs`. End-to-end: a
`--debug --steps 10000` CPU smoke with fixture labels built from REAL episode obs.

### The TWIN HEADS + the SHADOW CRITIC (`--cf-twin-heads` / `--cf-shadow-critic`, `gen3_cf_twin_heads_v1`, v99)

**The owner-authorized amendment to the SIGNED R1 pre-registration** (ledger 2026-08-22 evening,
"Three owner sign-offs" item 3). It changes what the arm's primary comparison *is*, so read this
before reading the runbook's §2.

**The problem it solves.** R1 as signed compared two RUNS — an arm with `--cf-winprob-coef` and a
control without. Two runs differ in every random draw they ever make, and the primary meter carries
a MEASURED floor of ~39% of its own variance (`tmp/hidden_info_floor_report.md`). So a cross-run
difference has to clear noise the design cannot control, and a null would be uninterpretable.

**The design: three win-prob heads on ONE trunk, differing ONLY in their label stream.**

| head | module | trained by | isolates |
|---|---|---|---|
| **A** (control) | `win_head` — the EXISTING head, untouched | the on-policy single-outcome BCE, at `win_prob_coef` | — |
| **B** (coverage) | `cf_twin_head_b` | A's loss **+** the cf-labelled states with **SINGLE-OUTCOME** labels (n≡1) | **B−A = coverage/prioritization** |
| **C** (treatment) | `cf_twin_head_c` | A's loss **+** the same states with **TIGHT-MC** labels (n=R) | **C−B = pure variance reduction** |

That factorial is the mechanism split. `C−A` remains the original R1 claim; the amendment's value is
that it now decomposes. Because all three read the same `value_pooled` on the same rows in the same
minibatch, the trunk, the states, the seeds and the hidden-information floor are **identical by
construction**, not matched by design.

- **B and C are `WinProbHead` — the same class and capacity as A.** A difference of architectures
  would be a second explanation for every difference of scores, and nothing downstream would say so.
- **Head-only ALWAYS in v1.** Both twins read a DETACHED `value_pooled` in *every* term they take,
  including the on-policy mirror. So this measures the **LABEL effect on a trunk that is frozen with
  respect to them**; trunk exposure and policy transfer stay CROSS-RUN questions (runbook §0a,
  unamended). `train/cf_twin_grad_share` reads exactly 0.0 — published so the contract is a live
  measurement.
- **The mirror rides `win_prob_coef`, not `cf_twin_coef`.** All three heads must carry a
  bit-identical copy of the control objective, or B−A would confound "extra states" with "a
  different base objective".
- **B and C pull EQUALLY HARD.** `_cf_binomial_nll` normalizes by `Σn`, so a row's gradient is
  `(q − target)/B` whatever its n. B's n≡1 rows and C's n=R rows therefore differ only in the
  TARGET — which is what makes C−B a read of label PRECISION rather than of effective learning rate.
- ⚠️ **`cf/twin_b_coverage` is the FIRST thing to read.** A producer shipping no `outcome_label`
  trains B on nothing; B then equals A, the pre-registered C−B contrast silently becomes C−A, and
  every other counter reads healthy. That is the one way this arm produces a confident wrong answer.
  B's fold is skipped rather than trained on a zero-filled absent label, and the scalar says so.

**The SHADOW CRITIC** is the other half and a different job: a passive `ShadowValueHead` trained on
**`mc_return`** labels — the mean realized **shaped return** over the producer's rollouts, in the
units the live critic V actually predicts. It **never computes an advantage, never enters GAE,
feeds nothing forward, and reads `value_pooled.detach()` unconditionally** (the `pubval` structural
precedent). Swapping the live critic for an MC-grounded one is critic SURGERY and owes the C4
offline gate; this head is the **staged promotion path** that earns or refuses that gate without
risking a run.

- **The frame.** Under PopArt the head's raw output IS the normalized value and the target is
  `popart.normalize(mc_return)` — `_value_distill_mse`'s handling, for its reason (the coefficient
  stays scale-comparable with the value loss). Every reported metric is DE-normalized to real
  shaped-return units, which is the only frame a reader can interpret.
- 🔒 **THE METER is `cf/shadow_shadow_vs_live_v`** — the SIGNED real-unit mean of (shadow − live V)
  on the same states, with the live V taken off the *same* forward through `policy._critic_value`
  (never a hand-rolled `value_net` call, which under `--value-from-dist` reads a head the run does
  not use). A shadow sitting systematically BELOW the live critic is a live critic that is
  optimistic about the states the factory samples, **measured against ground truth rather than
  argued from a calibration curve**. `cf/shadow_live_v_vs_label` is its direct half; read them
  together, because the shadow is itself a fitted head and can be wrong too.

**The LABEL SCHEMA decision, and why it is not a version bump.** The three streams ride ONE row
(`outcome_label`, `mc_return` + `mc_return_n`, `reward_sha1` as additive-optional v1 fields) rather
than arriving as separate `kind`s. Two reasons, the first decisive: **`CfLabelBuffer` dedups on the
obs digest**, so a second row for the same state would collide and one would silently replace the
other. And one-row-per-state makes "heads B and C saw identical states" *structural* rather than
hoped-for. `schema` stays **1** because it is a REFUSAL gate — a consumer skips every row whose
version it does not know — so bumping it would make a new producer's output unreadable by an
existing trainer, which is the opposite of backward compatible. Old consumers ignore the new keys;
new consumers supervise nothing extra when they are absent.

**`mc_return` carries a REWARD DIGEST and is REFUSED on a mismatch.** A shaped return is a fact
about a board *under a reward composition*, so a return measured under a different `RewardConfig`
is a measurement of a **different value function**, not a noisier sample of ours — and there is no
shape error or range violation that would catch it. `reward_config_digest(config)` (a stable sha1
over every `RewardConfig` field) is stamped by the producer and handed to the buffer by the
trainer; a mismatch drops the **field** (never the row — its win-prob labels are still good), counts
`cf/labels_mc_return_rejected_total`, and warns once by name. The digest is only passed when
`--cf-shadow-coef > 0`: a run with no shadow head must not reject rows over a field it does not read.

**The producer side** (`cf_producer.py`): `outcome_label` is free (it already computes the recorded
outcome for the critic-surprise term). `mc_return` needs the server-free reward path —
`agents/training/cf_mc_return.py` wraps `RewardTracker`, keeps the per-turn rewards *in order*, and
folds them with `--gamma`. Two non-obvious facts live there: **`RewardTracker` accumulates an
UNDISCOUNTED total** (a return is `Σγᵏr` from a particular state, so the rewards must be captured
per turn), and **the divergence turn's own move is SCRIPTED**, so a tracker hooked only into the
live `choose_move` would begin at T+1 and its return would be missing `r_T` and carry an extra
factor of γ — against the very state the label is FOR. That is why `install_scripted_prefix` grew an
`on_scripted_decision` hook (default None, byte-identical): it REPORTS, and the producer's closure
decides. The reward config is read from the run's own `metadata.json` `cli_args` through the SAME
`RewardConfig.from_args` the trainer uses; when it cannot be read the default is used and the fact
is printed LOUDLY, because the digest will then simply not match and the trainer will say so.

**⚠️ TWO SEAM BUGS shipped in the first version of the `mc_return` path and were caught by
adversarial review, not by the tests.** Both produced plausible-looking labels; keep them in mind
before moving either seam.

1. **`action_to_order` is NOT a valid recording seam.** It looks ideal (the commit point; it raises
   `StaleDecisionError` on a superseded attempt) — but `counterfactual._invert_choice` calls it in a
   **LOOP over every legal index** to recover a recorded choice's action number, on every scripted
   decision of the prefix. Recording there fired 6-9 times per scripted turn with actions that were
   never played, each advancing the STATEFUL reward function. The seams are `_predict_best_action`
   (caches the committed `(idx, mask)`) + the player's own `choose_move` (the once-per-decision
   boundary; it must be wrapped BEFORE `install_scripted_prefix`, which captures it as its live
   delegate) + `_battle_finished_callback` (the terminal reward).
2. **The hook must `arm_at_next()` AND `note()`, in that order.** Arming alone left the first LIVE
   decision at T+1 as the armed one, so `r_T` was dropped and every label was `G(s_{T+1})` against an
   obs row for `s_T` — biased by whatever happened on the divergence turn (a KO there is the largest
   single shaping term), i.e. **correlated with the state and shaped exactly like a real signal**.

Both are pinned in `cf_mc_return_test.py`, the second with an explicit negative control showing the
buggy shape, because neither is visible in any scalar. Note what did NOT catch them: the
bridge-backed composition test asserted only that an `mc_return` was PRESENT. **A composition test
that checks presence rather than value is a presence test.**

**Two counters, not one, and the distinction is the same one twice.** `cf/labels_skipped_total` is
the ROW-level GIGO meter and must keep partitioning the input with `labels_ingested_total`; an
optional FIELD that is malformed or out of range ACCEPTS the row and counts into
`cf/labels_field_skipped_total`, and a reward-digest refusal counts into
`cf/labels_mc_return_rejected_total`. Folding any of these into the first would make "is the
producer feeding me garbage" climb at the ingestion rate on a buffer refusing nothing.

**The discount comes from the RewardConfig, not from a flag.** `reward_config_digest` hashes every
field including `gamma`, so folding the return at `cfg.gamma` puts the discount under the same GIGO
guard as the reward. `--gamma` survives only as an explicit override, and its help says what that
costs: a mistyped value ships returns folded against a different value function with the digest
still matching and every liveness counter reading healthy.

**⚠️ The ONE coupling head-only does NOT remove: the global gradient CLIP.**
`clip_grad_norm_` scales every gradient by `max_norm / total_norm` over ALL parameters, so any term
with a non-zero gradient anywhere perturbs the policy and value updates in the last bits. It is
tiny at a sane coefficient and it is shared by every aux this tree runs — but it is not zero, and an
arm claiming a bit-identical trunk must know which of the two mechanisms it is claiming.
`instrumented_ppo_test.py::test_the_only_coupling_between_a_headonly_term_and_the_trunk_is_the_GLOBAL_CLIP`
pins the pair: with the clip active the updates differ, with it raised out of the way they are
bit-identical. A genuine gradient leak would survive both.

**Flag class.** `--cf-twin-heads` and `--cf-shadow-critic` are **STRUCTURAL**, in
`agents/model/flag_registry.py` (v99, `cli`/`structural`), with `ModelVersion` fields, bool compares
in `check_compatible`, a `MODEL_CONFIG_VERSION` bump to **99** with a setdefault-False migration,
`snapshot.current_model_version` keywords, and **no `ARCH_SIGNATURE` bump** (optional side heads,
obs family unchanged) — the `cf_evidential` precedent exactly, and the gate matters for its reason:
the forward never calls these heads, so `check_compatible` is the ONLY thing that can catch a
flipped flag. `--cf-twin-coef` / `--cf-shadow-coef` are training-only (the `td_aux_coef`
class), deliberately not in the registry but recorded and **read back on a flagless resume** since
config **v100** — a within-run paired comparison whose coefficient silently zeroed on restart would
report B−A ≈ C−B ≈ 0 and look like a null result.

Refusals, all at the CLI: `--cf-twin-coef > 0` requires `--cf-twin-heads`; `--cf-shadow-coef > 0`
requires `--cf-shadow-critic` (both are state_dict changes and cannot be added mid-run to rescue a
live coefficient); and **`--cf-twin-heads` requires `--win-prob-mode read_only|shaping`**, because
the twins mirror head A's loss and `none` builds no head A — the arm's control arm would silently
not exist.

**The AUDIT read** — `cf_audit` gained `attach_twin_heads` (one more batched forward, same
best-effort contract as `attach_evidential`) and two blocks:

- 🔒 **`twin_paired` is the amended PRIMARY.** Per row, `brier = (pred − mc)²` and
  `abs_err = |pred − mc|` for each head, **differenced across heads on the same row**, with a
  battle-clustered bootstrap CI on the difference. Two properties buy it over `sd_true_excess` here:
  the hidden-information floor **cancels exactly** (it is a property of the STATE, identical in
  every arm — the amended §2 argued it cancels at matched *step*; twins strengthen that to matched
  *state*), and no stratification means no selection correction is owed. **SIGN: these are ERROR
  scores, so a NEGATIVE difference means the first-named head is better.**
  ⚠️ A near-zero contrast with a near-zero `mean_abs_pred_diff` is a **coverage/dosage** reading,
  not the pre-registered null — the label streams did not separate the heads and there is nothing
  to decompose yet.
- **`twin_resolution`** is the G0 continuity link: each head's own `sd_true_excess` binned by its
  own prediction. Its cells are **UNWEIGHTED** and the block says so in its own `weighting` field —
  the population re-weighting is unavailable for B and C, because the eval frame carries only head
  A's predictions, so their decile membership over the whole frame is unknown. Absolute levels here
  are NOT comparable with the bias map's `population_weighted_sd_true_excess`.
- **`shadow`** carries `shadow_vs_live_v` (signed, battle-clustered) and `shadow_vs_live_v_abs`.

**Gates.** `agents/model/cf_twin_heads_test.py` holds the heads' contracts (the shadow's UNBOUNDED
range — a sigmoid creeping in would clamp every label while the MSE fell; the twins' identical
architecture; their INDEPENDENT init, so `cf/twin_b_vs_c_abs` at step 0 is not reading its own
initialization; the BIT-identity of ON's pi/vf for each flag and both together; that the forward
never calls any of them; the v99 gate on both flags and both migration legs; the registry rows; the
`current_model_version` threading). `instrumented_ppo_test.py` pins the routing and the isolation:
coefficient-zero byte-identity for each half, a live coefficient reaching ONLY its own heads (with
the clip raised — see above), **the ROUTING pin** (B's loss equals the binomial NLL of the OUTCOME
and demonstrably NOT of the tight-MC label, with the two set to opposite extremes), B's n≡1
weighting, B's skip-and-count when no row carries an outcome, the mirror's coefficient and its
detach, the shadow's PopArt frame and masking, and that all FOUR cf terms share ONE sample and ONE
forward. `cf_label_buffer_test.py` covers both schema directions (an old row still ingests; a new
row carries both streams), the out-of-range field skip that keeps the row, the reward-digest
refusal and its counter, the coverage scalars, and the masks in `batch_tensors`.
`cf_mc_return_test.py` pins the oldest-first discount and the deliberate one-decision arming delay.
`cf_audit_test.py` pins the sign convention, the honest null, the refusal to compare heads it does
not have, and the shadow block's signedness. `main/cf_flags_test.py` covers the defaults, both
negation forms, the four refusals and `checkargs`. **The composition** is
`cf_producer_integration_test.py` (`sim`): a real bridge battle → the ring → one producer cycle →
the REAL `CfLabelBuffer`, now additionally asserting every row carries a valid `outcome_label`,
that at least one carries an `mc_return` with its digest, that the buffer keeps them, and that a
buffer configured with a FOREIGN digest refuses the `mc_return` while keeping the row. A second
test in that file covers the **MULTI-CYCLE** seam the first one holds fixed: a checkpoint lands
between cycles, the producer reloads it and RE-STAMPS the rows, and the real buffer holds the two
vintages at their two different ages — plus a poisoned row (`obs_sha1` disagreeing with its own
bytes) costing exactly itself beside the good ones. That is the leg the 2026-08-23 R1 composition
smoke found a live defect in.

### The PER-ACTION Q WIN-PROB HEAD (`--q-winprob-mode` + `--q-winprob-coef` / `--q-winprob-onpolicy-coef`, `gen3_q_winprob_head_v1`, v107)

**The problem, stated as a cost.** Every value readout this tree owns evaluates a STATE, so a
per-action win probability is not a read — it is eleven simulator re-rolls plus eleven forwards,
because the successors have to be manufactured first. That is exactly why probe L's ranking "is not
a quantity the network computes" (it is the head composed with a simulator, and PPO performs no such
composition). `QWinProbHead` amortizes the composition: one forward, eleven `P(win|s,a)`, scored
from the pointer head's own action tokens. The architecture half is in
`src/agents/model/CLAUDE.md` → `QWinProbHead`; this section is the training half.

🚨 **THE STARVATION TRAP — read this before setting either coefficient.** On-policy data labels
exactly ONE action per state, and probe L measured the policy sampling its own better-ranked
alternative at a median **p = 0.002**. A Q head trained on that stream is untrained precisely on the
never-tried moves — i.e. **confidently wrong on the entire set a per-action readout would ever be
consulted about**, because the shared scorer generalizes the taken-action signal onto the unvisited
columns with nothing to correct it. The head's primary labels are therefore COUNTERFACTUAL, from the
same R1 factory the rest of this block feeds on (ledger 229e9f1).

**THE LABEL CONTRACT** is an ADDITIVE-OPTIONAL extension of the existing v1 row — the schema version
deliberately does NOT move, for the reason stated at `cf_label_buffer`: `schema` is a REFUSAL gate,
so bumping it would make a new producer's output unreadable by an existing trainer.

```json
"q_labels": [{"action": 7, "label": 0.62, "n_rollouts": 16}, ...],   // per-ACTION counterfactual
"taken_action": 7                                                    // for the weak fallback only
```

`q_labels` is a **list of objects, never parallel arrays**. Three same-length lists can be written in
the wrong order by a producer and read as valid by the consumer; a per-action object cannot. Each
entry names its own index in the ACTION SPACE (`[switch x6, move x4, struggle]`) — the same index the
policy's logits, the action mask and the Q head's column `a` use. A malformed entry is a counted
FIELD skip (`q_labels_*`), so the row survives with its three other label streams intact: a producer
bug in one stream must not cost the trainer the rest. Duplicate actions collapse keep-LAST (two
entries for one action are the producer contradicting itself; summing them would invent evidence).

**THE LOSS** is `q_masked_binomial_nll` — the scalar cf term's likelihood restricted to the labelled
cells, normalized by `Σ(mask·n)`. Two invariances come out of that normalizer and both are pinned:
the coefficient's meaning is independent of the producer's R (an R=16 label pulls exactly 4x an R=4
one — that IS the likelihood of the data, not an emphasis choice) **and** of the minibatch's label
DENSITY. A whole-grid `Σn` would make the term shrink as coverage fell, which is the opposite of what
a starving factory should do to a loss. At full coverage the function equals
`cf_terms.cf_binomial_nll` EXACTLY, which is what makes "the same likelihood, restricted" a fact
rather than an analogy. **An unlabelled cell contributes zero to numerator and denominator alike** —
never a zero target, which is indistinguishable from a confident "this action loses".

**TWO COEFFICIENTS, AND THE SPLIT IS THE POINT.** `--q-winprob-coef` weights the counterfactual
stream. `--q-winprob-onpolicy-coef` weights the WEAK fallback — the recorded battle's realized
outcome as a single-sample label for the ONE action that was taken, at `n ≡ 1` so its per-row
gradient magnitude matches a counterfactual row's and only the TARGET differs. It defaults to **0.0**
and should usually stay there; it exists so a starved-factory run has something to show, not as a
substitute. Separate coefficients so the two can never be confused in a run's provenance, and its
metrics carry an `onpolicy_` prefix so its numbers can never be read as the grounded stream's.

**BOTH ARE HEAD-ONLY, STRUCTURALLY.** The head's inputs are detached inside the EXTRACTOR forward
(`q_winprob_mode` has no `shaping` value), so no coefficient can route a gradient into the trunk and
`grad/q_winprob_share` reads exactly 0.0 by construction — the verification, not a defect.

**⚠️ Both folds RE-APPLY the head rather than reading `last_q_winprob_logits`, and the reason is not
the same as `_cf_winprob_term`'s.** That term re-applies its head because `win_prob_mode` governs a
different decision than `cf_head_only`. This one does it because `cf_sample_and_forward` runs under
`th.no_grad()` whenever nothing downstream needs a graph — a condition computed from the *scalar*
term's settings, which know nothing about this one — so a term folded from that stash would train
**exactly nothing** while every metric looked healthy. It reads the pointer stash from the same
forward and RAISES on a batch-size disagreement (the `_critic_value` stale-stash precedent):
structurally impossible, therefore loud rather than degrading.

**METRICS — `q_winprob/*`**, its own prefix so a per-ACTION number can never be read as the per-state
win head's.

| key | read |
|---|---|
| `label_coverage` / `labels_per_row` | **FIRST.** Coverage is "is the factory running"; **`labels_per_row` is "is it running at more than one action per state"** — i.e. the number that separates a real counterfactual stream from the on-policy trickle this head exists to avoid |
| `loss` · `abs_err` · `bias` | the fit on labelled cells |
| `pred_spread` vs `label_spread` | **the discriminating pair.** A head that learned nothing per-ACTION still scores well on `abs_err` by predicting each state's mean; a `pred_spread` far below `label_spread` is a head that amortized the VALUE and not the SEARCH. Computed only over rows with ≥2 labelled actions, since a one-action row's spread is 0 by construction |
| `onpolicy_*` | the weak fallback's, and not evidence about the grounded stream |
| `train/q_winprob_loss` | **ABSENT, never 0.0, when the fold starves** — a defaulted zero is a perfect score for a head that trained on nothing |

**THE OFFLINE METER (E5 step 5)** is `python -m main.q_amortization <run_dir>`: the head's per-action
row against the prober's own one-ply `lookahead` sweep — Spearman, top-1 agreement, and the
**amortization residual**. Shrinking ⇒ the AlphaZero ratchet (search's value has migrated into the
net; search must deepen to add anything); stubbornly large on a class of states ⇒ those states
genuinely need live search, a triage signal for the ladder time manager. Two caveats live in the
script and belong here too: it is a **PREDICTIVE** meter and says nothing about whether the policy
plays better (iteration 2's lesson — keep it distinct from the behavioral dividend), and its ground
truth is **itself a model read**, since `lookahead` scores each re-rolled successor with the same
checkpoint's critic. `--self-check` runs the init-state sanity (zero-init ⇒ P = 0.5 everywhere ⇒ a
total tie) with no checkpoint, no traces and no simulator; it is gated in the suite.

**Status: LATENT.** Mode `none`, both coefficients 0.0, and **the producer does not yet emit
`q_labels`** — that is the next piece and the one that decides whether any of this measures
anything.

### The label PRODUCER DRIVER (`cf_producer.py`) — the piece that runs the loop

```bash
nohup nice -n 10 python -m agents.training.cf_producer \
    models/<run> [--rollouts 8] [--top-n 3] [--records-per-cycle 4] \
    [--max-labels-per-hour 2000] [--anchor-every 50] [--impl rust] \
    > models/<run>/cf_producer.log 2>&1 &
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
```

The tap rings records; the buffer consumes label rows; **this walks one to the other.** It is a
long-lived **standalone sidecar run beside a live trainer** — the `snapshot_ladder` /
`bot_matchup_matrix` pattern — and deliberately NOT auto-spawned by the trainer: producer and
consumer share only a file format (that is `cf_label_buffer`'s whole premise), and a producer the
trainer owned would make a label-path failure a *training* failure.

#### 🚨 THE DUTY CYCLE — the number that decides whether ANY of this works

The producer can only stamp a label with the step of the newest `checkpoints/` zip, and the buffer
expires a row more than `--cf-label-lag-steps` behind the live policy. **Those two flags define a
fraction, and until 2026-08-23 nobody computed it:**

```
duty cycle = --cf-label-lag-steps / (env steps between checkpoints)
```

The denominator is the trap. SB3's `CheckpointCallback.save_freq` **counts VEC-ENV CALLS, not env
steps** — one `_on_step` per `vec_env.step()`, which advances `n_envs` envs at once — and it was a
bare hardcoded `50000` in `main/train/callbacks.py`, read as "50k steps" by everyone including the
R1 design. At `--n-envs 48` it is **2,400,000 env steps** against a 150,000-step bound: a **6.25%**
duty cycle. Measured on the live `ai_v9_29_rev1_0823`: **6 labels ingested against 255 expired in
two hours**, with every counter on both sides reading healthy — the producer was producing, the
buffer was expiring, and neither knew the other's number.

Two things close the class:

* **`--checkpoint-every-steps <env_steps>`** (trainer) sets the cadence in the unit a reader means.
  Default `None` = the historical `50000` vec-calls, byte for byte, so a flagless resume is
  unchanged; a value is converted back by ceil-division (`main.train.constants`).
* **The launch REFUSES a duty cycle under 25%** and PRINTS it when healthy. With `--cf-records` on
  and a live `--cf-twin-coef` / `--cf-winprob-coef`, `main/train/config.py` computes it, names all
  three numbers plus both remedies, and exits `FATAL_CONFIG` (not `parser.error` — a restart would
  hit the identical config, so the launcher must give up rather than loop). `--debug` prints and
  is exempt. *A quantity nobody computes is how this shipped, so it is now printed on every launch
  that has both halves on.*

At the production shape that is `--checkpoint-every-steps 150000 --n-envs 48` → 3125 vec-calls →
a 100% duty cycle.

#### 🚨 THROUGHPUT — the ~8 s/label was the POLICY FORWARD, not the drivers

Profiled 2026-08-23 on `/tmp` copies of live `ai_v9_29_rev1_0823` ring records against that run's
own checkpoint, beside the live trainer (load 15-25):

| stage | share | note |
|---|---|---|
| **rollouts** (R × `replay_counterfactual`) | **93%** | of which **93% is `choose_move`** |
| `scan_record` (replay + obs materialize) | 0.3% | ~170 ms/record, ONCE |
| `replay_battle` (offline replay driver) | 0.2% | ~35 ms/record, ONCE |
| `score` (the ranking forward) | 0.1% | |
| record parse / label write | <0.1% | |

Inside a rollout: **832 `choose_move` calls at 15.4 ms** against 0.40 ms of scripted-prefix
`embed_battle`, 0.14 ms of `_invert_choice`, and a 9 MB rust child spawn. A B=1 CPU decision
measured **26.3 ms eager → 4.1 ms compiled (6.4×)**; the extractor alone is 21.5 → 3.3 ms, i.e.
~82% of it.

**That contradicts the banked ~162 ms/label cost model, and the correction is the useful part.**
That model was measured on the *materializer* path (one-ply labels, whole-prefix replay per arm),
where cold driver spawns dominate and a warm rust `SearchSession` is 289×. This producer is a
different shape: it replays each record **once** (`scan_record` takes `chunks=` from the single
`replay_battle`) and its labels are **rollouts to end**, which spawn no offline driver at all — they
play a live bridge battle whose cost is policy forwards. So a **warm/persistent search-driver
session buys ~0.2% here, and prefix sharing ~3%** (a scripted prefix decision is 0.4 ms against
4-26 ms for every live one, and cloning the mid-battle state would have to clone two poke-env
players' trackers as well). *When a cost model is carried across a path shape, re-measure before
building to it.*

**What shipped (`gen3_cf_producer_compiled_rollouts_v1`), and the three shape hazards it closed:**

* **`--compile-extractor` now DEFAULTS ON.** It is a fallback, not an opt-in;
  `--no-compile-extractor` is there for when the compile is the suspect (a compile *failure*
  already degrades to eager on its own). It costs ~40 s **once per PROCESS, not per checkpoint**:
  dynamo keys on the CODE object and the weights are graph inputs, so the next refresh reuses the
  graph — measured **1.1 s for an entire second `load_snapshot`** against 39.7 s for the first.
* **`score` forwards ONE ROW AT A TIME under compile.** Dynamo specializes on the batch dimension,
  so a single `B=29` scoring call in front of B=1 rollouts forced a re-trace: measured **79.4 s for
  the first label's 8 rollouts against 3.0 s for the second**. Row-wise costs ~0.12 s/record against
  ~0.04 s batched and keeps exactly one signature alive. Eager snapshots keep the batched forward.
* **The mask is cast to `float32` there too.** A materialized mask is `int8` and a live one is
  `float32`, and a compiled graph guards on DTYPE as hard as on shape — **19.5 s on the first
  scored row** before the cast.
* **The compiled graph is warmed through the LIVE call signature** (`_warm_the_compiled_graph`).
  `maybe_compile_extractor` warms with `{"observation": …}` alone; every real call also carries
  `action_mask`, and a dict's KEY SET is part of the guard — so the compile looked warm and the
  first real decision re-traced for **19.5 s**, charged to whichever record was first. It is now a
  startup cost that announces itself.
* **`--rollout-concurrency` (default 1) is a KNOB, not a win.** Measured a wash over 10 paired
  label-arms (conc=1 mean 3.86 s vs conc=8 4.17 s, no consistent sign): every policy forward *and*
  every protocol parse runs on poke-env's single `POKE_LOOP` thread, so overlapping arms finds
  almost no idle to fill. The arms are independent by construction (own players, own bridge child,
  own post-divergence dice) and one that dies costs that arm alone.

**Measured before/after — a REAL one-cycle run, same 6 records, same checkpoint, same box:**

| | cycle wall | per label (excl. one-time load/compile) |
|---|---|---|
| before (`d78aa81`, eager) | 198.5 s @ load 17 | **10.8 s** |
| after (compiled) | 99.4-102.9 s @ load 20-22 | **3.2 s** |

Interleaved arm-by-arm on the same decisions (the load-fair form): **8.09 s → 1.81 s of rollout
wall per label, 4.5×**. The eager arm reproduces the live producer's own 8.2 s/label, which is what
says the harness measured the right thing. **The label output is unchanged**: identical key set,
the same decisions selected with the same ranking, and the real `CfLabelBuffer` ingests 18/18 rows.
The only difference anywhere is `priority.win_prob` in the **6th decimal** (0.670049 → 0.670050) —
Inductor's arithmetic, the documented max|Δ| ~5e-7, on a field nothing thresholds.

#### The producer/retention race (`records_vanished`)

The trainer OWNS `cf_records/` — every env worker prunes it to the newest `--cf-records-keep` (512)
— and this process only reads it, so a record can be enumerated and then deleted before it is
opened. Measured on the same run: **176 records lost to `FileNotFoundError` across 67 cycles**,
with "538 pending" against a ring of 512 (the excess is a guaranteed loss by arithmetic). Three
properties, all load-bearing, none of them a change to the ring's semantics:

* **Records are taken NEWEST FIRST.** The ring deletes from the OLD end, so the oldest pending
  record is the one already promised away — and the loop walked exactly that end. Newest-first puts
  the deletion end of the ring at the low-value end of the work queue. (It is independently the
  right sampler order: a newer record came from a policy closer to the one the label supervises.)
* **The batch is READ AT ENUMERATION TIME** (`CfProducer._load_batch`). The window the ring wins is
  enumerate → anchor (a full scripted replay, seconds to minutes) → claim + fsync → open; reading
  immediately collapses it, and everything downstream works from an in-memory record.
* **A vanished file is a COUNTED BENIGN SKIP** — `records_vanished` in the state file, on the
  heartbeat, and one explanatory log line — **never an exception path.** As an exception it landed
  in `skip_reasons` as `error:FileNotFoundError`, indistinguishable from a corrupt record, and on
  the ANCHOR path it reached `anchors_errored`, where an ordinary ring deletion could **exit 3**
  and stop the factory. The remedy is a larger `--cf-records-keep`, which a restart can raise.

Each cycle: poll `<run>/cf_records/` for unprocessed records → refresh the freshest `checkpoints/`
snapshot (via `latest.txt`, else the highest-stepped zip; its step is stamped on every label) →
replay each record ONCE (which yields the realized outcome, every decision's obs, its mask, its
action index and its committed choice string, via `obs_materializer.scan_record`) → forward the
snapshot over the candidates → label the top `--top-n` by the declared priority → roll each out
`--rollouts` times → write one NEW file per batch to
`<run>/cf_labels/labels_cf_producer_<step>_<seq>.jsonl`.

**"The freshest checkpoint" means BOTH names a resumable checkpoint is written under** — the
periodic `checkpoint_<step>_steps.zip` *and* the FORCED `checkpoint_forced_<step>_<HHMMSS>.zip` that
SIGUSR1 writes (the launcher TUI's `c` key). Reading only the first was not cosmetic: a forced save
was reachable solely through `latest.txt` and then, because its step did not parse,
`resolve_latest_checkpoint`'s key ranked it **below every periodic zip** — so forcing a checkpoint
mid-run walked the producer BACKWARDS onto an older snapshot and it went on stamping that older
step, with every counter on both sides reading healthy. Found by the R1 multi-cycle composition
smoke (2026-08-23), which is the only thing that had ever run a producer across a checkpoint
boundary; pinned by `cf_producer_test::TestCheckpointResolution` (the resolver) and
`cf_producer_integration_test::test_a_new_checkpoint_mid_run_restamps_the_labels_and_the_buffer_takes_both`
(the label path as a whole moving forward).

⚠️ **THE ECOLOGY DECISION — read this before quoting any label this producer wrote.**
A training record carries **no opponent identity**. The tap's `__RECON__` frame holds the resolved
seed, both packed teams and the committed choices, and nothing that says *which policy* sat on the
other side — a self-play pool snapshot, one of the nine heuristic bots, or the trainee's own
weights. The label therefore cannot name the opponent it was measured against, and a value claim
that cannot name its population is not a value claim (the G0 rule: *never quote "the critic is
optimistic by X" without naming the population — the sign depends on it*). So v1 makes the
approximation **explicit rather than guessed**: every rollout is played by the **CURRENT snapshot
on BOTH sides, sampling stochastically at temperature 1.0** — the regime the training actor itself
plays in. That matches the ~90% self-play share of the training mixture, and it is wrong in a
KNOWN direction for the rest: on an episode whose opponent was a bot, a weaker opponent is replaced
by a stronger self-like one, so that label is biased LOW. Every row carries
`opponent: "self_current"` — never a bot name it cannot verify — so a reader can always tell a
producer label from a `cf_audit` label, whose opponent IS identified. Closing the approximation
means threading the opponent's identity through the training-side tap; it is not a change to
`cf_producer.py`. **Stochastic is the load-bearing half of the regime**, not a style: a greedy copy
of a net is strictly stronger than a temp-1.0 sample of it, and greedy rollouts biased the prober's
sentinel labels LOW by a measured +0.037 [+0.007, +0.066].

**Which side is the trainee.** A training record names none, so `_trainee_side` answers from the
transport's own invariant: `BridgeSession` seats `env.agent1` — the trainee — on **p1**, always. A
record that DOES name a trainee (an eval sibling handed to this tool) is honoured instead.

🚨 **A rollout that reaches the 250-turn cap is a DRAW AT CAP and scores 0.5** — never a win or a
loss (`gen3_cf_draw_at_cap_v1`, fixed 2026-08-23). Both sides of a rollout stall-forfeit at
`MAX_TURNS`, so at the cap BOTH forfeit and the recorded winner is decided by which `FORCELOSE` the
sim processes first — a fact about ordering, not about the position. **Measured over 16 capped
lines on `node` and `rust` alike, the ordering is not even a coin flip: p1's forfeit is always
processed first, so p1 always loses**, and `_trainee_side` puts the trainee on p1 always. Every
capped rollout therefore used to score a hard 0, biasing tight-MC P(win) labels **DOWNWARD** on
exactly the stall-shaped positions where the cap is reachable — an *upward* bias was guessed when
the class was first noted, and the guess was wrong. A genuine tie went the same way (`outcome ==
"win"` is False for a tie) and is likewise 0.5 now. The count rides out as **`n_capped` beside
`n_rollouts`** on every row (an ADDITION; the buffer reads a fixed key set and ignores the rest,
and `schema` stays 1) and as `rollouts_capped` in the state file + the heartbeat — because a 0.5
built from 8 draws-at-cap and a 0.5 built from 4 wins and 4 losses are the same number about
different positions, and no reader can re-derive which afterwards. `wilson_lo`/`wilson_hi` now take
a fractional success total, so with draws in the sample the interval is an approximation that errs
narrow; `n_capped` is what says how much. Detection is exact rather than heuristic —
`replay_counterfactual` returns `capped = finished and turn >= turn_cap_of(both players)`, and
`_handle_stall` forfeits at every decision from the cap turn onward, so a battle can never resolve
normally on or after it. Gated by `cf_producer_test::TestDrawAtCap` (revert-verified),
`counterfactual_test::test_battle_outcome_flags_a_battle_that_ENDED_AT_THE_CAP`, and the `sim`
`cf_producer_integration_test::test_a_rollout_that_reaches_the_TURN_CAP_is_a_draw_on_either_seat`,
which plays the same fixture board from BOTH seats at a forced low cap and requires one label.

⚠️ **Labels written before that fix carry no `n_capped`, and cap-reaching is NOT re-derivable from
them** — the rollouts leave no artifact and a capped 0 is indistinguishable from a played-out 0. On
`ai_v9_29_rev1_0823` (999 rows / 333 source records / 7,992 rollouts as of 2026-08-23) what IS
derivable bounds it: every label sits at a decision turn ≤ **96** (p50 9, p90 25), so a rollout
needs ≥154 further turns to cap; and the cap's base rate in the surrounding training population is
**2 stall events across the 4,097 episodes in the record ring's 5.1-minute window (~0.05%)**, which
puts the expected count in the single digits of ~8,000 rollouts. Treat that as a base-rate estimate,
not a measurement of the corpus.

⚠️ **A record written BEFORE 2026-08-24 by the rust bridge carries no `forcelose` entry at all, so a
scan of its `commands` reads a false 0.** The rust `sim_bridge` pushed `commands` only in
`handle_choose`, while node has always pushed `['forcelose', <side>]` — so under the PRODUCTION
default (`--use-bridge rust`) a forfeited battle's record looked exactly like one that played on.
Two consumers read that field and both were silently wrong: the offline replay path
(`search::feed_recorded_cmd` has a `"forcelose"` arm; `recorded_turn_choices` stops at one) never
reproduced the forfeit, and `record_is_full_replay_anchorable`'s forfeit exclusion was **INERT**
(`anchors_skipped_unanchorable` reading 0 on the live run means the exclusion never fired, not that
there were no forfeits — it is what misled the #34 census). **FIXED at the record writer**
(`handle_forcelose` now pushes the entry, at the site node does), which repairs the record itself
rather than one of its readers. Gate:
`bridge_impl_parity_test::test_a_forfeited_battles_record_logs_the_forcelose_command`, over BOTH
impls — verified failing on the rust arm when the push is reverted. **Records already on disk are
frozen wrong**; a forfeit census over an old rust corpus is not re-derivable from `commands`.

**The sampler is DECLARED and VERSIONED** (`cf_producer_priority_v1`), written into the state file
AND every label row, because a silent priority change is a distribution-shift confound for every
downstream readout (design decision-of-record 3):

| term | what | weight |
|---|---|---|
| `critic_surprise` | `\|P(win\|s) − realized outcome\|` — the **conviction region** G0 measured at +0.23, and the population R1 exists to supervise. A single realized outcome cannot say whether the head was wrong or the dice were (53% of that class was genuinely winning); tight-MC labels are the only instrument that separates them, so they are spent here first | **1.00** |
| `policy_entropy` | the masked action distribution's entropy ÷ `log(n_legal)` — the decisions the policy has not made up its mind about. **Normalized by the support size** so a 2-way coin flip outranks a 9-way near-certainty, which raw entropy inverts | **0.35** |

A tie (the turn cap) scores outcome **0.5**, not a loss — it is uninformative about conviction, not
evidence the head was wrong. A checkpoint with **no win-prob head** has no surprise term at all;
the producer says so once and ranks on entropy alone rather than reading a missing head as a
confident 0.0. Candidates are start-of-turn **move rounds** at turn ≥ 2 only (a forced-switch round
has no valid recorded answer to script; the turn-1 bound is the same one `cf_audit` declares — a
retained SAMPLER choice since `gen3_search_turn1_open_v1` made both impls able to open turn 1, no
longer the driver limitation it was introduced for).

**Crash safety, and what it costs.** A record is claimed in `<run>/cf_producer_state.json` and the
state file is **fsync-replaced BEFORE its rollouts run**. So a crash mid-record loses that record's
labels and can NEVER double-label. That direction is deliberate: the buffer dedups on the obs
digest, so a duplicate is survivable — but it is also a silent re-weighting of the declared
sampler, and a record aged out of the ring unprocessed is simply a record that was not labelled.
Missing a label is free; mis-weighting the sampler is not. Pinned on the ORDER (the state file must
already be durable when `process_record` raises), not argued.

**The anchor rule, inherited from `cf_audit`.** At startup and every `--anchor-every` records, one
record is replayed FULLY SCRIPTED through the live bridge (`divergence_turn=None` — the correctness
oracle) and must reproduce the winner the offline replay driver reports. On failure the producer
**exits 3 and writes nothing further**: a factory whose replay is not exact is GIGO, and every
label after it would be a measurement of the bug. This anchor is *stronger* than `cf_audit`'s —
nothing is played by a policy, so a MISMATCH is unambiguously a defect rather than a die roll. An
anchor that CRASHED counts as a FAILURE, never a pass.

⚠️ **But a CRASH and a MISMATCH must not print the same sentence, and until 2026-08-23 they did.**
The two refusals reach the same exit 3 and have opposite diagnoses: a mismatch says the replay is
inexact; an exception (a wedged bridge child, a transport error, a contention `ProgressTimeout`)
never returned a verdict, so it says nothing about exactness — it refuses because an anchor that
did not complete has certified nothing. `main` printed the MISMATCH text for both, and that turned
ONE flaky `cf_producer_integration_test` failure into an investigation of a replay-exactness gap
that did not exist. They are now counted apart (`anchors_errored` in the state file and the
heartbeat, beside `anchors_run`/`anchors_reproduced` — the split `cf_audit` has always had) and
rendered apart by the pure `anchor_refusal_message`, which appends `describe_contention()` when the
exception was a timeout. Same rule as everywhere else in this tree: **a timeout is never a semantic
outcome**, and a message must never assert a cause the code has not established. Pins:
`cf_producer_test::TestAnchorRefusal::test_an_anchor_{ERROR_is_counted_and_reported_apart_from_a_MISMATCH,TIMEOUT_self_diagnoses_instead_of_accusing_the_replay}`.

#### ⚠️ The FORFEIT class — the one thing a live scripted replay cannot adjudicate

**Root-caused 2026-08-23**, from the single intermittent `ANCHOR REFUSED` the R1 composition test
hit. `record_is_full_replay_anchorable` now EXCLUDES it, visibly and by count
(`anchors_skipped_unanchorable`); it is a declared coverage bound of the oracle, in the same family
as `cf_audit`'s turn-1 and forced-switch bounds — **not** a retry, and never a second attempt at the
same record.

- **The mechanism.** A battle that reaches `StallConfig.threshold` (= `MAX_TURNS`, 250) is ended by
  ONE side forfeiting, which the bridge logs as `['forcelose', <side>]` in `record.commands`.
  `install_scripted_prefix` builds each side's script as `[c for (s, c) in commands if s == side]`,
  so `'forcelose'` matches NEITHER side and is dropped — the scripted replay has no way to reproduce
  the recorded forfeit. Instead **both** players re-derive one from their own `_handle_stall` at
  turn ≥ 250, and whichever `FORCELOSE` the bridge processes first loses. In the recording only ONE
  side could forfeit at all (in training, the trainee; in the composition test, the
  `RecordingFuzzPlayer` against a plain poke-env `RandomPlayer` that has no stall handling), so the
  replay can hand the win to the side that actually LOST — the exact `scripted full replay → win,
  record says <opponent>` signature the ledger recorded.
- **MEASURED. 1037 fresh battles played and rung exactly as `_play_and_ring` does, `--impl node`:
  4 refusals, 0 errors (0.39%). ALL FOUR were forfeit-terminated records — 4 of the 16 that reached
  250 turns (25%) — and 0 of the 1021 non-forfeit records refused** (95% upper bound 0.29% for any
  other class). The per-forfeit flip rate is itself a race, so treat 25% as order-of-magnitude and
  possibly load-dependent: the four splits were 0/8 in one batch and 2/2 in another. Re-anchoring one refusing record refused **7/12 and 8/12** across two batches, so it
  is a RACE, not a property of the record; every non-forfeit record re-anchored **40/40 identical**.
  Rebuilding the anchor with the OPPONENT's stall threshold unreachable — mirroring the recording —
  made that same record **12/12 correct**. That is the mechanism proof.
- **It is a faithfulness LIMIT, not a defect the anchor can report.** The offline replay driver gets
  it right every time because it replays the ORDERED command log including the `forcelose`; two
  poke-env players driven concurrently cannot reproduce that ordering.
- ⚠️ **Lowering the stall threshold to force the class did NOT reproduce it** — 384 battles at
  threshold 25, 381 of them forfeit-terminated, **0 refusals** — and reading that as a clearance is
  what nearly closed this investigation early. A turn-25 board is not a turn-250 Struggle endgame;
  making a rare event common changed the thing that decides it. Force the *condition*, then confirm
  on the real one.
- 🔴 **TASK, not fixed: the ROLLOUT path inherits the same asymmetry.** A label's rollouts play both
  sides with `RLPlayer`s that both stall-forfeit, whereas the recorded training battle had only the
  trainee forfeiting — so a rollout reaching the 250-turn cap can be scored a WIN purely because the
  opponent's forfeit landed first, biasing the labels of long games upward. The anchor exclusion does
  not touch it.

**And the whole path is NODE-ONLY** — the composition test passes with `POKESIM_SIM_BRIDGE_BIN` and
`POKESIM_SEARCH_DRIVER_BIN` pointed at nonexistent files, so no `src/rust_sim` binary, stale or
otherwise, participates in it. The stale-main-binary trap is not in play here.

**A second, stricter check ships with it, and it is honest about never having fired.**
`replay_counterfactual` now returns `script_exhausted` — the sides that ran OUT of recorded commands
and finished on the live policy, which a `divergence_turn=None` full replay can only do after
diverging — and the anchor refuses on it even when the winner happens to match. It exists because
the fallback policy is random, so a script desync only flips the WINNER about half the time. It did
NOT catch the forfeit class above (that race consumes no script at all), and it was **empty on every
one of 274 instrumented healthy replays**, so it costs a correct run nothing. Pin:
`cf_producer_test::TestAnchorRefusal::test_a_full_replay_that_RAN_OUT_of_script_fails_the_anchor_even_on_a_matching_winner`.

⚠️ **ONE latent desync found by inspection and NOT reproducible** (`install_scripted_prefix`,
`utils/bridge/counterfactual.py`): when the mask is empty the scripted player returns
`choose_default_move()` **without popping the script**, but the live player it is replaying DID
emit that `/choose default` and the bridge DID record it as a command — so the script would sit one
entry ahead for the rest of the battle. Unreachable in this fuzz (0 `default` commands in 356
gen3ou records, so the branch never fires) and left alone deliberately rather than "fixed" blind.

**Observability.** A separate process has no TensorBoard, so it prints one **heartbeat line per
cycle** and keeps `<run>/cf_producer_state.json` human-readable (indented; sampler + weights +
totals + the last heartbeat + skip reasons):

```
[cf_producer] cycle 2 | snapshot step 29,867,520 | records 1 pending / 3 done | labels 2
              (+6 total, 6/h) | anchor 1/1 | PRODUCING | load 23.6 | 9.2s
```

The trainer-side half of the contract is the `cf/*` scalars — `cf/labels_ingested_total` going flat
is what a dead producer looks like from over there (see the R1 runbook's launch-window table).

**Two guards on running beside a live trainer.** `--max-labels-per-hour` (default 2000, a sliding
one-hour window) keeps it a sidecar. `--stale-checkpoint-minutes` (default 90) **pauses production**
when no NEW checkpoint has appeared for that long — the trainer is probably gone, and a producer
grinding against a frozen snapshot either burns the box filling a buffer whose rows will expire, or
teaches the current policy an ancestor's values. It keeps WATCHING (a restarted trainer resumes it)
and announces itself exactly once in each direction. `--lag-warn-steps` (default 150 000, matching
the buffer's `DEFAULT_LAG_BOUND`) warns once when the snapshot in hand falls that far behind the
newest checkpoint.

**`obs_materializer.scan_record`** is the new read primitive under it. An eval trace ships its obs
and action indices in `states.npz`; a training record ships **neither** — only the seed, the teams
and the committed choice strings — so the only route to a training decision's observation is to
replay the one-sided protocol AND recover the action history by inverting those choices through the
real mapper. `scan_record` does both in ONE replay and returns `RecordDecision(index, turn, action,
choice, mask, obs)` rows. It shares `_InvertingReplayPlayer` with `infer_action_indices` (which
stays track-only), and both go through one `_encode_or_track` step so the two replay players cannot
drift on the one operation where drift would silently change an obs rather than fail.
`scan_record(capture_choices=True)` additionally fills each row's `.choices` with the FULL legal
action → sim-choice-string map at that decision (`gen3_cf_q_labels_v1`, below) — asked for INSIDE
the replay it already runs, because asking afterwards costs a `materialize_from_record` prefix
replay per labelled decision. OFF by default and byte-identical off.

#### The PER-ACTION stream (`--q-labels`, `gen3_cf_q_labels_v1`) — the supply side of the Q head

The v107 `QWinProbHead` (above) shipped as a **trained consumer of a stream nobody wrote**: mode
`none`, both coefficients 0, and a producer that emitted no `q_labels`. This closes that: the same
tight-MC rollout, once per **legal action**, on the **same dice**. It is `--no-q-labels` by default
and byte-identical off — including the dice, whose salt now routes through `cf_q_labels.q_arm_salt`
but is verbatim the string `_rollout` always used (pinned by a test, since a change there would make
every existing label file incomparable).

| flag | default | what |
|---|---|---|
| `--q-labels` / `--no-q-labels` | **OFF** | emit `q_labels` + `taken_action` on each swept row |
| `--q-top-n N` | 1 | how many of a record's `--top-n` labelled decisions get swept |
| `--q-rollouts R` | 0 = follow `--rollouts` | R per SIBLING arm |
| `--q-max-actions K` | 0 = every legal action | cap the arms per swept decision |

**THE PAIRING IS THE POINT, and it is asserted rather than remembered.** The sweep's product is a
RANKING ("is Rock Slide better than Earthquake here?"), and at R=8 the per-arm standard error is
~0.18 — so on independent dice a 0.1 gap between two siblings is invisible. Every arm therefore
takes `cf_q_labels.q_arm_seeds`, whose salt is a function of the DECISION and carries **no action
term**; `assert_paired_dice` adjudicates at the seam on the seeds each arm ACTUALLY received (never
re-derived — a check that recomputes its own input proves only that one function is deterministic).
⚠️ The pairing covers the SIM DICE only: both sides are a stochastic snapshot at temperature 1.0 and
`Categorical.sample` draws from torch's global RNG, so the policy draws are an unpaired residual. It
biases nothing (both arms draw the same policy) and cannot be closed by seeding — the arms diverge
immediately and stop drawing the same NUMBER of samples.

**The recorded action's arm is FREE, and its q-label is an IDENTITY.** The row's own `label` IS the
recorded action's counterfactual label — same salt, same R, same substituted choice — so at
`--q-rollouts == --rollouts` it is lifted verbatim and `q_labels[recorded] == label` exactly (pinned
in the unit tests AND in the `sim` composition test). At a DIFFERENT R it is re-rolled instead,
because an anchor measured over more arms than the siblings it anchors makes `q[recorded] −
q[other]` a comparison between two sample sizes.

**The selection rule is DECLARED (`cf_q_sweep_v1`, stamped on every row) for the same reason
`SAMPLER_VERSION` is.** Recorded action first; the rest in a **deterministic decision-keyed
shuffle**, truncated by `--q-max-actions`. Both obvious orders are wrong here: descending policy
probability rebuilds the on-policy starvation the head exists to escape (probe L: median p=0.002 on
the better-ranked alternative), and action index is a systematic preference for SWITCHES, since the
space is `[switch x6, move x4, struggle]` and a prefix of it is all switches. `K=0` sweeps
everything and has no bias to declare.

🚨 **COST MULTIPLIES BY THE ARM COUNT — meter it, do not estimate it.** A swept decision costs R
rollouts per legal action instead of R total. `--max-labels-per-hour` therefore counts every
per-action arm it actually ROLLS (the reused recorded arm costs nothing, so it does not), which
keeps the cap a **cost** cap rather than letting the sweep silently multiply a sidecar's box load by
its arm count. The state file and the heartbeat carry `q_rows`,
`q_entries_total`, `q_arms_rolled`, `q_arms_reused`, `q_rollouts_total`, `q_wall_seconds` and a
separate `q_skip_reasons` (a lost ARM is not a lost RECORD, so it never touches `records_skipped`);
`(q_arms_rolled + q_arms_reused) / q_rows` **is** the measured multiplier. A sweep that exhausts the
throttle mid-decision ships a SHORT block rather than a broken one — every entry in it is a real
measurement and the consumer masks the rest.

**MEASURED 2026-08-29** — 90 `cf_records` of `ai_v9_72_R3SELF_0828` against **its own v107
checkpoint** (of the 37 archived runs holding `cf_records`, the only one current code can still
load), CPU, `--impl rust`, `nice -n 15` beside a live trainer at load ~27-33, compiled extractor at
9.3×, `--rollouts 4 --top-n 1 --q-top-n 1 --q-rollouts 4 --q-max-actions 0`:

| producer | | consumer (the REAL `CfLabelBuffer`) | |
|---|---|---|---|
| records / skipped | 90 / **0** | ingested / skipped | 90 / **0** (0 field skips) |
| **arms per row** | **7.70** (3-9) | `cf/q_label_coverage` | **1.0000** |
| arms rolled / free | 603 / 90 | `cf/q_labels_per_row` | **7.70** |
| **throughput** | **1.98 s/entry** | labelled (s,a) cells | **693 / 990 = 70.0%** |
| `q[recorded] == label` | **90/90** exactly | sweep wall / cycle | 1 375 s of 1 619 s |

Folded through the real loss kernel on that batch, `q_masked_binomial_nll` reads **0.693147 = log 2
to six places** at a zero-init head (the P = 0.5 prior it must be) and 0.4218 fitted, with the
gradient on every UNSWEPT cell **exactly 0.0** — the masked form's whole safety property, measured
rather than asserted. So the cost reads as **~7.7× a plain label**: ~15 s of sweep per row against
~2 s for the row's own. Full record: `designs/CHANGELOG.md` → *The PER-ACTION LABEL FACTORY*.

**The arithmetic lives in `cf_q_labels.py`, not in the producer** — pure, so the pairing rule, the
selection rule and the wire shape are testable without a simulator. `q_labels` is a **LIST OF
OBJECTS** each naming its own action index (never parallel arrays — see `cf_label_buffer`'s
docstring), it rides the SAME row as the per-state label (the buffer dedups on the obs digest, so a
second row for one state would collide), and it is **additive-optional at schema v1**: the sweep may
never bump `schema`, which is a REFUSAL gate, so a v2 row would be unreadable by every existing
trainer. An arm whose rollouts ALL failed is OMITTED rather than shipped at `n_rollouts: 0`, because
the consumer builds its mask from PRESENCE and a zero-evidence entry would mask ON a cell whose
target is the `0.0` fallback — a confident loss for an action nobody measured. `taken_action`
travels with `q_labels` — the consumer-facing name for the index the row already carried as
`recorded_action`, and deliberately not given its own flag, so nothing is offered the
on-policy-only regime the stream exists to escape.

**Tests.** `cf_q_labels_test.py` (pure: the salt has no action term and is byte-identical to the
historical one, a smaller R is a PREFIX of a larger one, `assert_paired_dice` raises on divergent
AND on merely shorter lists, the recorded action always survives a cap, a capped sweep does not
prefer switches — measured over 400 decisions, with the index-ordered rule pinned as the
counterfactual it fails — and the wire shape incl. the zero-evidence drop).
`cf_producer_test.py` (pure: the priority arithmetic incl. the entropy normalization and
the tie rule, the state file's claim-before-work order and its bounded processed set, the
producer/retention race — `TestProducerRetentionRace` deletes a record mid-cycle and asserts a
counted skip, newest-first order, that a preloaded record survives its file, and that a vanished
anchor record is not an anchor FAILURE — the throttle
and its sliding window, the stale-trainer pause + resume, the anchor's refusal / cadence /
crash-is-a-failure, the ecology field on every row, checkpoint resolution, and that every help
string renders). The THROUGHPUT contract has its own classes: `TestScoreForwardSignature` (a
compiled snapshot scores one row at a time, an eager one keeps the batched forward, the mask
reaches the graph as `float32` either way, and the chunking changes not a single number),
`TestCompiledGraphWarmUp` (the warm-up forwards BOTH obs keys — the live signature — and a warm-up
that raises is survivable), and `TestRolloutArms` (arms aggregate to the same label sequential or
overlapped, one dead arm costs that arm alone and the arms after it still run, every arm gets its
own players and its own dice, all-arms-dead is a skip rather than a phantom 0.0).
**The deliverable is `cf_producer_integration_test.py` (`sim`)**: a REAL bridge
battle → the REAL `CfRecordRing` in the TRAINING tap's shape (**`trainee_username` stripped**) →
ONE REAL producer cycle → the REAL `CfLabelBuffer`, asserting every row INGESTED with **zero skips**,
digests verifying, correct `policy_step`, and — the strongest assertion in the file — that the obs
the producer *materialized* is **bit-identical** to the obs the LIVE player encoded, which is the
only thing that proves the inverted action history did not desync the encoder's trackers. Both
halves of a two-process contract had unit tests when the last two contract bugs shipped; neither
test ever ran the other half's real output, which is why this file runs the composition.

The PER-ACTION stream is covered at both altitudes for the same reason. Unit
(`cf_producer_test.py`): OFF leaves the row's key set and its DICE byte-identical; the sibling arms
demonstrably receive one seed list; a producer that derives seeds per action RAISES (the regression
expressed as the bug); the check reads the base arm's OBSERVED seeds; `q_labels[recorded] == label`
and the recorded arm is not rolled twice; each budget knob bites; the cost meter round-trips through
the state file; each per-action label counts against the throttle; and the schema round-trips
through the real `CfLabelBuffer` — including a **deliberately shuffled** list reading identically
(the object-not-arrays property demonstrated, not asserted), a malformed entry costing the FIELD and
not the row's other three streams, and an OLD row still reading on the NEW consumer.
`sim`: `test_the_PER_ACTION_stream_composes_ring_to_buffer` runs the whole thing on a real battle
into the real buffer's per-action columns, and
`test_scan_record_recovers_the_FULL_choice_map_at_every_decision` checks the capture against the one
entry known independently — the map's value at the RECOVERED action index must be the string the
side actually committed, since a wrong map is a silently MISLABELLED action rather than an error.

## The STALL-TAIL HARVEST + head-repair pipeline (`main.harvest` → `winprob_finetune` → `main.harvest_meter`)

An **offline, three-stage pipeline** that manufactures win-probability labels for the population
probe O convicted, fits the win-prob head on them with the trunk frozen, and measures the result
against a battle-level holdout. It is the ai_v12 head-repair backbone and it writes nothing into
`models/` — artifacts land under `utils.paths.harvest_dir()` (repo-root `harvest/`, gitignored,
`$GEN3AI_HARVEST_DIR` overrides).

```bash
export PYTHONPATH=$PYTHONPATH:src
# 1. HARVEST — mine late-game states, score them with the subject, label by re-seed multi-rollout
python -m main.harvest --subject models/<run>/final_model.zip \
    [--states 300] [--rollouts 32] [--min-turn 60] [--workers 2] [--inline-obs] [--dry-run]
# 2. FIT — head-only, trunk frozen, binomial NLL, turn-slice re-weighted
python -m agents.training.winprob_finetune <harvest_dir> --subject models/<run>/final_model.zip
# 3. METER — probe O's battery, PAIRED, pre vs post, on the held-out battles
python -m main.harvest_meter <harvest_dir> --head <finetune_out>/head_best.pt
```

**Why it exists.** Probe O measured the head ending above 0.5 on **34.8%** of the tails of games it
loses by construction. Ledger `b63a96f` banked the account: the labels are correct, and all three
mechanisms are DATA-shaped — BCE optimizes the average and cap-game final turns are epsilon of the
buffer; strong-position-at-turn-249 barely exists in training at all. So the head's problem is a
**census** problem (discrimination mass at the late time-slices), not a propagation one — MC labels
already stamp the terminal onto every step. This pipeline manufactures the missing mass.

### The label schema is a CONTRACT (`agents/training/harvest_schema.py`)

Gzipped JSONL, eleven pinned fields plus an obs locator triple; `validate_row` runs on every write
AND every read. It is deliberately **separate from `cf_audit`'s v1 schema**, which is a single-run
bias-map contract with a live consumer (the trainer's label ring): the harvest spans many runs'
traces scored by ONE subject, carries the selection `priority` that drew each row, and pays a
binomial likelihood over `k`/`n`. Widening `cf_audit`'s schema in place would make every existing
reader tolerate columns absent from every row it has ever seen.

`load_obs` is the ONE resolver both sides call, and it verifies `obs_sha1`. That is not ceremonial:
`cf_audit` shipped a real bug where `obs_npz` rows ignored `decision_idx` and **every** default-mode
label was rejected as architecture drift, because both halves of the contract had tests and neither
ran the other's real output.

### Three measured facts that shaped the sampler — each one changed the design

1. **The subject must be RE-SCORED, always.** A trace's recorded `win_probs` came from whichever
   checkpoint played it. Measured: the subject's re-scored φ differs from the recorded column by up
   to **0.135 on that subject's own traces**. `phi_head` is a fresh forward (~2.9 ms/state batched);
   the rollouts likewise run the subject as trainee via `ProbeSession(ckpt_override=…)`, which is
   what makes `k/n` an estimate of *the subject's* value rather than of the run that recorded it.

2. **Priority alone fills the sample with doomed states.** A 40-state draw came from 4 battles, all
   losing. A head fit on nothing but doomed late states has a trivial way to score perfectly — say 0
   at every late turn — which is a BIAS, not a repair, and it would wreck the thing the head already
   does RIGHT (probe O: `LONG_WIN` reads **0.986** at 128 median turns; the failure is a right tail
   on the doomed side, not a length effect). Hence `--drag-frac` (0.6).

3. **Capping the doomed share was NOT enough.** With the remainder also ranked by priority, a
   300-state draw took **1** state from a won battle: the gap term is `|phi − realized|`, and a
   correctly-read win scores ~0 on it, so the control stratum was selected out of existence by the
   very rule that makes the doomed stratum good. Hence `--general-win-frac` (0.5). The meter's
   long-win control is only an honest test because of this.

### 🚨 THE FAILED PILOT — sample the REGION THE METER READS, or the fit gets worse in both directions

The most important thing this pipeline knows, and it was bought by running it. Pilot 1 (200 states
/ 41 battles / 6,281 rollouts, priority-ranked with no tail stratum) produced a head that was
**worse on every population**, and the long-win control is what convicted it:

| population | metric | pre | post | paired diff, CI95 | |
|---|---|---:|---:|---|---|
| held-out long losses (n=86) | `detect_le05` | 0.977 | 0.302 | −0.674 [−0.767, −0.570] | **SIG** |
| held-out long losses | `phi_T` | 0.070 | 0.607 | +0.537 [+0.492, +0.580] | **SIG** |
| LONG-WIN control (n=40) | `phi_T` | 0.943 | 0.567 | −0.376 [−0.447, −0.299] | **SIG** |

Both directions collapsed toward ≈0.6 — the head lost its dynamic range and became nearly
constant. **This is not "late means lost"** (that would have moved the two populations in opposite
directions); it is regression to the *sample* mean, and the cause is a measured distribution
mismatch:

| | fit set (harvest) | eval set (meter) |
|---|---|---|
| turn range | **60–152**, p50 90 | **96–239**, p50 118 |
| mean MC label | **0.621** — states the subject WINS | ~0 — doomed by construction |

**29.3% of the meter's eval turns were above the harvest's maximum turn.** The head was fit on
mid-game positions it wins 62% of the time, never shown a losing tail, and then asked about one.
That is extrapolation, and extrapolating from a 62%-win sample onto a 0%-win population lands on
the sample mean.

The fix is `--tail-frac` (default 0.5): a reserved share of the doomed budget goes to a battle's
last `TAIL_K = 5` decisions — `TAIL_K` matches the meter's `K_TAIL` and probe O's K **on purpose**,
because that is the region being scored. Fitting the same REGION on DIFFERENT battles is
generalization, not leakage: the battles themselves are held out by the producer. `--tail-frac 0`
ablates it back to the pilot-1 behaviour.

**The durable lesson, which generalizes past this pipeline: a label factory that never samples the
region its meter scores is extrapolating, and no amount of label quality fixes it.** Pilot 1's
labels were excellent — 31.4 adjudicated rollouts per state, 1.86% timeouts, a 0.052 noise floor.
They were labels for the wrong states. **And note what caught it**: every stall-tail metric alone
would have read this as "detection got worse", which is ambiguous with a bad fit; it was the
untouched LONG-WIN control moving the *other* way that identified the collapse as loss of dynamic
range. That is what the control is for, and it is why the meter refuses to report without one.

### 🚨 The cap-record blocker — 8 of 48, and it is the KNOWN rust `forcelose` gap

A 250-turn game ends by FORFEIT (`Gen3Env.action_to_order` returns `ForfeitBattleOrder` at
`MAX_TURNS`), logged as a `["forcelose", side]` command. **A record without it never terminates, and
the offline replay driver refuses it** — *"replayed all N commands but battle has not ended (turn
250)"*, asserted by BOTH impls in the `replay` verb that `materialize_from_record` depends on. So
every model-scored offline path is blocked on such a record, not just this one.

Measured archive-wide: **689 cap records, 543 (79%) carry the forfeit.** Within the current-arch
corpus it is **8 of 48**, and the split is *exactly* the documented boundary — the rust `sim_bridge`
pushed `commands` only in `handle_choose` until **2026-08-24** (§ *The FORFEIT class* above):

| runs | caps | replayable |
|---|---:|---:|
| `…_0819` … `…_0824` (pre-fix) | 40 | **0** |
| `…_0825` … `…_0828` (post-fix) | 8 | **8** |

So the scarcity is **transient** — every post-fix run adds replayable caps — and it is why the
doomed-tail population is caps **plus long losses** rather than caps alone (`Candidate.meter_class`).
That widening is faithful, not convenient: probe O's own framing is a heavy right tail on the doomed
side, and caps are where it concentrates (4.3× the ordinary-loss rate), not the only place it lives.
The two classes are stratified separately everywhere and the meter never pools them, because the
head reads them very differently (probe O: `detect_le05` 0.652 on caps vs 0.94–0.95 on long losses).

**The missing forfeit is NOT synthesized.** Appending `["forcelose", trainee_side]` in memory would
make all 40 replay, and the battle is a LOSS at exactly 250 turns so a trainee forfeit is
overwhelmingly likely. It is refused because "overwhelmingly likely" is not a basis on which a
LABEL FACTORY may manufacture an ending — a record may lack its terminal because the *opponent*
forfeited, and inventing our own loss would fabricate the very outcome being measured. Skipped,
counted as `cap_record_unterminated`, published in the manifest.

### Holdout is decided by the PRODUCER, before a single label is bought

The split is battle-level, computed from `--seed`, written to `holdout.json`, and the sampler
**refuses to draw a candidate from a held-out battle** — exclusion happens before ranking, so no
later slicing can readmit one. Leakage is unrepresentable rather than forbidden. It is **stratified
by class**: with 8 cap battles, an unstratified 35% draw can easily take 0 or all 8, and either way
one arm of the meter loses the class the exercise is about.

### A timeout is its own bucket

`n_rollouts` counts ADJUDICATED rollouts — one that reached a terminal, i.e. win, loss **or TIE**;
everything else (`unfinished`) lands in `provenance.n_timeout` and is excluded from both numerator
and denominator. Folding a timeout in would make a busy box read as a losing position — the error
that once let a starved parity run report 39/40 timeouts as a clean PASS. The cap forfeit itself
adjudicates normally (it is recorded as a LOSS), so it needs no special case.

⚠️ **A TIE is a semantic outcome and belongs in the DENOMINATOR — it went into the timeout bucket
until the R1 adversarial review.** `counterfactual._battle_outcome` emits four values (win / loss /
tie / unfinished) and `label_one` computed `n = win + loss`, so a tie was filed as a timeout, which
is none of the three things this section says that bucket counts. Every dropped rollout is a
NON-win, so `k/n` overstated P(win) on exactly the states where a game can end even — and the
owner's clean-world ruling says the same thing in the reward's language (a draw scores `-victory`,
i.e. as a loss). Ties are now adjudicated and counted separately in `provenance.n_tie`, because a
denominator that silently absorbed a second outcome class is one nobody can audit.

### ⚠️ TWO different `--holdout-frac` flags, and conflating them would look like leakage

They are disjoint by construction and neither is the other:

| flag | splits | for |
|---|---|---|
| `main.harvest --holdout-frac` | **battles**, out of the doomed-tail population, BEFORE any label is bought | the METER's test set. These battles contribute zero training states. |
| `winprob_finetune --holdout-frac` | rows of the ALREADY-harvested set, by `battle_tag` | the fit's own train/val split, for epoch selection |

The fit's val split is drawn only from battles the harvest already chose, so it can never touch a
meter-held-out battle — the producer excluded those before ranking. Both are battle-level; the
consumer's `split_by_battle` calls `assert_battle_disjoint` on its own result before returning it.

### The fit is head-only, and that is STRUCTURAL

`winprob_finetune` runs in two phases: a no-grad forward of the frozen trunk caching `value_pooled`
(the `WinProbHead`'s only input, via `ProbeModel._value_pooled_batch` so the numbers stay comparable
with live `cf/*` scalars), then an Adam fit over `head.parameters()` alone. The trunk is not frozen,
it is **absent** from phase 2 — and `_assert_head_only` raises if any param group holds anything
else. Loss is the binomial NLL `k·softplus(−z) + (n−k)·softplus(z)`, slice-weighted and normalized
by `Σ w·n`, asserted exactly equal to the live trainer's `cf_terms.cf_binomial_nll` and to mean BCE
at `n ≡ 1`. Slice weights are inverse-frequency over declared edges (`SLICE_EDGES = (60, 80, 100,
130, 170, 250)`, `SLICE_VERSION`), rescaled to mean 1 so the learning rate means the same thing
across datasets. **Best epoch is chosen by the PLAIN val NLL**, not the re-weighted one — the
re-weighting is an optimizer device, and scoring with it would make the meter agree with the device
by construction.

### The ANCHOR (`--anchor-coef`, default **0.3**) — what makes the fit SAFE

The second thing the pilots bought, and the reason a flagless run is now non-destructive. A harvest
is a **prioritized** sample — selected precisely where the head is wrong — so its label mean sits
far from the population's, and a 6-parameter head fit on it with nothing holding it back collapses
toward that sample mean. Measured, on held-out battles, with the long-WIN control as the
falsification:

| | fit-set label mean | long-loss `detect_le05` | LONG-WIN control `phi_T` | |
|---|---:|---|---|---|
| pilot 1, no anchor | 0.621 | −0.674 [−0.767, −0.570] | **−0.376** [−0.447, −0.299] | REGRESSION |
| pilot 2 (tail stratum), no anchor | 0.427 | −0.326 [−0.430, −0.221] | **−0.165** [−0.212, −0.119] | REGRESSION, half the dose |
| pilot 2, **anchor 0.3** | 0.427 | **±0.000** | **−0.033** [−0.045, −0.023] | **control HOLDS** |

The damage scaled with the sample-mean offset across two independent runs — which is what
identifies the mechanism as selection bias rather than a bad hyper-parameter.

`anchor_coef * mean((z − z0)^2)`, where `z0` is the SUBJECT's logit on the same cached
`value_pooled`, captured **before** the resume branch can touch the head (taking it afterwards
changes the objective mid-run and makes a resumed fit diverge — which is exactly how the bug was
caught, by `test_resume_reproduces_the_uninterrupted_run_bitwise`). It is a per-EXAMPLE penalty,
not a weight penalty: the quantity that must not drift is the head's FUNCTION on the real state
distribution, and an L2 on six parameters says nothing about that.

**Dose.** 1.0 and 3.0 both stop at best-epoch 0 — the anchor dominates and the fit never moves. So
0.3 is the largest dose that still lets the labels speak, and it is the default because shipping
0.0 would ship a setting measured to be destructive. `--anchor-coef 0` opts out and reproduces the
pilots exactly.

⚠️ **At 0.3 the fit is SAFE but not yet BENEFICIAL**: every categorical metric is unchanged and only
`phi_T` moves — the right way on cap endings (−0.109, n=3) and slightly wrong on long losses
(+0.024). The pilot demonstrates the pipeline and its guard rails; **it does not answer the
reducibility question**, which needs more labels than 359 states across two runs.

### The meter is PAIRED, and its control is the whole falsification

`harvest_meter` re-scores each held-out battle's last **K = 5** decisions through both heads and
reports probe O's metrics — `detect_le05` (the substantive criterion), `detect` (as REGISTERED,
whose "declining" half probe O showed saturates in every class), `miss`, `overconf`, `c3band` — with
a bootstrap over BATTLES, never states.

Two properties that are not incidental. **"Pre" is the SUBJECT's reading, not the trace's**, so
these numbers are legitimately not comparable to probe O's published levels. And the **LONG-WIN
CONTROL** (long won battles touched by neither arm) is reported beside the detection rate because a
head that learned "late means lost" scores *perfectly* on every stall metric while being strictly
worse than the head it replaced. `verdict_lines` names that outcome **FAILED RUN** in those words
rather than leaving a reader to notice. `graft_head` likewise **refuses** a graft that changed
nothing — a silent no-op would make the post arm a bitwise duplicate of the pre arm and every metric
would read a perfect, perfectly confident null.

**Tests** `main/harvest_test.py` (39, unmarked, 0.35 s) — schema round-trip, the `decision_idx`
indexing contract and its digest refusal, the cap-record skip, class separation, holdout
stratification/determinism/disjointness, the per-battle cap, the outcome-balance property that
mechanism 3 above exists for, priority monotonicity + the no-evidential-head fallback, CRN pairing
at the `replay_counterfactual` seam, the timeout bucket, and the meter's statistics including the
saturating registered criterion and the FAILED-RUN verdict. Plus
`agents/training/winprob_finetune_test.py` (34, 3.7 s) for the consumer.

## Public-replay value aux — V_pub — DELETED (v88 `gen3_dead_flag_purge_v1`)

The v43 pubval subsystem (`--pubval-mode`/`--pubval-coef`, `agents.training.pubval`,
`pubval_calibration`, `data/gen3_pubval.json`, `PubValHead`, `_pubval_loss`, the parity fuzz) is
**deleted** — it measured NULL as a lever and was never ON in a production generation. A checkpoint
recording `pubval_mode != "none"` is refused by the v88 migration (re-read it from the git_hash in
its metadata.json); `"none"` pops silently. The raw replay corpus (`replays/showdown/gen3ou/`,
local-only) and the design doc (`designs/ai_v8/design_public_info_value.md`) remain for history.

## Distributional value head (`--value-dist-mode` / `--value-dist-coef`)

The training half of the v29 interpretability side head (model side: `src/agents/model/CLAUDE.md` →
distributional value head). A categorical readout off `value_pooled` whose softmax is the critic's
predicted **return DISTRIBUTION** — the shape the scalar V collapses (sharp = confident, wide =
uncertain, bimodal = coinflip). **Phase A** (interpretability-only): it does NOT replace the scalar
critic, so the GAE/advantage/value-loss path is untouched — this loss is an ADD-ON, like the win-prob
aux. Design + the K1 honesty frame: `designs/ai_v6/design_distributional_value_critic.md`.

- **Loss (`instrumented_ppo._value_dist_loss`).** **HL-Gauss** (Farebrother et al. 2024): build a
  Gaussian-smoothed soft target by integrating `N(target, σ_g²)` (σ_g = 0.75·Δ) over each atom's bin,
  with the two EDGE bins absorbing the outer tails (graceful out-of-support handling), then cross-entropy
  against the head's `log_softmax`. `train()` reads the stashed `last_value_dist_logits` + the rollout
  return as the target, **PopArt-normalized when the scalar critic is** (so the target lands in the head's
  support space — set `--value-dist-vmin/vmax` to a normalized range like ±5 under `--use-popart`). Folded
  at `value_dist_coef`. Pure + static → unit-tested in `value_dist_loss_test.py`.
- **Metrics (`value_dist/*`).** Aggregate interpretability health under its own TB prefix (the
  `grad/`/`popart/`/`win_prob/` group convention): `ce`, `entropy` + `std` (fall as the critic sharpens),
  `pit_mean` (≈ 0.5 ⟺ **calibrated** — the PIT anchor), `mean_abs_err` (`|E[Z] − return|` in support
  units). Ride the generic logger → TensorBoard + launcher TUI (`value_dist/*` labels in `format.py`).
- **Versioning.** `value_dist_mode` (str) + `value_dist_bins` (int) are version-checked structural toggles
  (fresh-only); the support `vmin`/`vmax` is resume-immutable (`check_value_dist`); `value_dist_coef` is
  **training-only**, read back on a flagless resume (like `win_prob_coef`). Threaded into
  `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles`.
- **Forensic trace + prober.** `RLPlayer._value_dist` reads the stashed logits at capture (softmax ⇒ the
  per-atom distribution) → `BattleRecorder.states_arrays` writes a `value_dist [T, bins]` npz array (key
  OMITTED when the head is off → the prober's KeyError "unavailable" path; NaN rows = uncaptured). The
  prober renders the per-decision **histogram** + mean/std/P10–P90/entropy/bimodality
  (`engine.build_value_dist` → `ValueDistView`, model-free; in the Summary panel + the `analyze` CLI). See
  `src/main/prober/CLAUDE.md`.
- **Honesty gate.** Ledger **K1 already killed the distributional critic as a WIN-RATE lever** (sub-Gaussian
  residuals — no tail). This is justified on INTERPRETABILITY only; its strongest use is upgrading the
  prober calibration/`falsify-scan` luck-vs-mistake split (predicted spread vs realized return = a
  within-model PIT). "Learns ≠ helps" — validate calibration (PIT ≈ uniform), not win-rate.
- **Tests.** Unit: `value_dist_loss_test.py` (HL-Gauss math + diagnostics), `agents/model/
  value_dist_head_test.py` (module build, off byte-identical, grad gating, the v29 version gate),
  `main/prober/engine_test.py` (`build_value_dist`). End-to-end `--debug --debug-eval --use-bridge=node
  --value-dist-mode read_only` smoke captures a trace whose npz carries `value_dist`.

## Exploiter distillation (`--distill-teacher` / `--distill-coef` / `--distill-value-coef` / `--distill-value-feat-coef`)

`gen3_exploiter_distill_v1` — pour a frozen per-team SPECIALIST (an exploiter) into the generalist so it
learns to PILOT that team, closing the amortization gap the self-play average can't. `--distill-teacher`
takes `TEACHER:TEAM` colon pairs (comma-separated, N teachers — a checkpoint dir → that run's **LAST
SNAPSHOT** since `gen3_last_snapshot_resolution_v1`, `best_model.zip` before it; see *WHICH FILE a run
spec names* above, and note the `🧪 [DISTILL]` line now states the resolved file, its step and the rung
per teacher), bound
to its Showdown team file); the env emits a training-only integer `distill_mask` obs key (0=none, k=teacher
k) on states where the trainee pilots team-k (biased there by `--distill-team-bias`, default 0.4; rest =
pool rehearsal → no forgetting). In `train()`, for each teacher a frozen forward gives π_teacher and
`distill_coef · KL(π_teacher ‖ π_student)` is folded, masked to that teacher's states; the per-teacher
mean-KLs are AVERAGED (a small-coverage teacher still contributes comparable gradient). Reuses the
`evaluate_actions` forward's stashed `_last_pi_distribution` (bit-identical, one fewer forward). Metrics
`distill/{kl, agree_rate, tK_kl, tK_coverage, n_teachers_active}`. OFF (coef 0 / no teacher) byte-identical;
training-only, NOT version-locked (inherited on a flagless resume). Validated (ai_v7_16→_19): offense
transfers (TSS-piloting 0.475→0.75) and HOLDS under the double-sided recipe (see the memory).

🚨 **THE TEACHER SPEC GRAMMAR IS `<run|zip>[@<step>]:<teams|*>`, AND THE `@step` IS SPLIT BY ONE
FUNCTION** (`gen3_run_spec_split_v1`, 2026-09-05). `;` separates teachers, `,` separates a
teacher's teams, `*`/`auto` expands to exactly the teams that teacher recorded training on, and the
teacher half is a RUN SPEC — the same `path[@step]` half `--stable-opponents` takes, pinning
`<run>/checkpoints/checkpoint_<step>_steps.zip` instead of the run's `best_model`. It is split by
**`agents.training.run_spec.split_run_spec`**, the only implementation in the tree. Before that it
was not split at all here: `'<run>@<step>:*'` handed the whole string to
`matchup_spec.read_recorded_trainee_teams`, which found no `metadata.json` beside a directory that
does not exist and returned `[]` — **the same answer a real generalist run gives**. A fold written
the obvious way therefore reported teachers that taught NOTHING, and the only witness was the team
count in the `🧪 [DISTILL]` startup line (the wrong-answer-on-a-success-path class, exactly like the
era's `--steps` no-op). Two throwing guards close it, at the producer and the consumer: the reader
now **RAISES `FileNotFoundError` naming the path** when the path does not exist (`require_teams=True`
additionally raises when a run that DOES exist recorded no pin — what the `'*'` resolver asks for),
and **`distill_spec.check_teacher_spec`** is the launch-time refusal, called from
`resolve_config` as a `parser.error` (`FATAL_CONFIG` class) for every teacher that resolves to ZERO
teams. `main.checkargs` prints its findings offline from that one function, so the launch answer and
the offline answer cannot drift — the `main.train.combination_checks` contract, in a separate
function only because that module explicitly excludes anything touching the filesystem or a teacher
spec. **The two surfaces differ in ONE declared parameter, `check_paths`, and the asymmetry is
deliberate:** `resolve_config` passes `False` (structural findings only) because on a real launch a
bad PATH is already answered loudly downstream — `model_build` exits `FATAL_CONFIG` naming the
teacher it could not load, `apply_distill_team_bias` raises on a team file it cannot open — and
re-asking would newly refuse a `--distill-coef 0` CONTROL arm whose teacher run has since been
archived, which works today and which nothing about this defect argues against. `main.checkargs`
passes `True` and also reports a missing run dir / team file, because OFFLINE there is no
downstream to answer — through its own `resolve_path` hook (default identity, so `resolve_config`
is unchanged), since `models/` lives in the MAIN checkout and a worktree would otherwise report a
teacher that is right there as missing. `_resolve_zip_and_config`
does the split at ITS entry, which is what makes the four `step=None` callers (`--distill-teacher`,
`--win-prob-pbrs-source`, `--distill-anchor-parent`, `--warmstart-consensus`) `@step`-capable at
once. Gate: `src/agents/training/run_spec_test.py`, including an **AST census** that fails, naming
the file, when a run-spec consumer re-derives the `@` split locally.

🚨 **THE COEFFICIENT GATES THE LOSS, NOT THE TEAM BOOKKEEPING** (`gen3_distill_bias_at_coef0_v1`).
`--distill-teacher` is what turns `--distill-team-bias` on, at ANY coefficient — including
`--distill-coef 0`, which is the CONTROL-arm shape: same teachers, same teams, same 0.4 bias, no
loss. That is not a nicety; it is the only way an arm holds the trainee's team distribution constant
against its treatment arm. It used to be the other way round — `args._distill_pairs` was populated
only above coef 0, so `ai_v9_58_R2CTRL_0827` trained at an EFFECTIVE bias of **0.0** while its argv,
its `metadata.json` and its startup banner all said 0.4, and the rev-2 capstone's arms silently
differed in the one variable its design pinned. What stays coefficient-gated is everything that
costs or changes something: no teacher is LOADED at coef 0 (N frozen networks and a forward per
minibatch, to be multiplied by zero), no loss is folded, and `_distill_species` stays `None` so the
env does NOT emit the `distill_mask` obs key — emitting it would move the observation SPACE of a run
with no distill term to read it. Two guards make the silence unrepresentable: an explicit
`--distill-team-bias > 0` with no `--distill-teacher` is a `parser.error` (the flag's argparse
default is `None`, resolved to 0.4 in `resolve_config`, exactly so a typed bias is distinguishable
from an unset one), and `--distill-teacher` with `--trainee-team/--trainee-teams` is refused at every
coefficient because the bias REPLACES the trainee teambuilder and would discard the pin. Gate:
`src/main/distill_team_bias_test.py` — it MEASURES the draw (4000 draws, 0.4 ± 0.04; pre-fix 0 of
4000), because the flag's value was never the thing that was wrong.

- **VALUE distillation (`gen3_exploiter_value_distill_v1`, `--distill-value-coef`, default 0 = OFF).** The
  policy KL is POLICY-ONLY — the student pilots the teacher's team with its own amortized (~4-dim) critic,
  so it mimics the MOVES but never gets the teacher's per-team VALUE (confirmed: value_cls effective rank
  FLAT across _14→_18→_19). This adds `distill_value_coef · MSE(V_teacher, V_student)` on the SAME
  teacher-team states, in the student's PopArt-normalized frame (`_value_distill_mse`, a static testable
  helper mirroring `_distill_loss`; teacher V from a frozen `predict_values`, real-unit → normalized).
  **Coherent despite V^π being policy-relative** because the policy KL simultaneously drives
  π_student→π_teacher there, so V_teacher becomes the right target — hence it **requires `--distill-coef > 0`**
  (arg-parse guard). Metrics `distill/{value_mse, tK_value_mse}`. **The A/B lever:** policy-only
  (`--distill-value-coef 0`) vs policy+value (>0), read out by the value_cls effective-rank probe
  (`rank_metrics.py`, `tmp/value_rank_compare.py`) — does distilling the value ENRICH it. OFF byte-identical
  (no teacher predict_values forward); training-only. **Distributional-value distill** (distil the teacher's
  `ValueDistHead` return distribution, enabling later archetype-token conditioning) is a future follow-on.
- **FITNETS value-FEATURE distillation (`gen3_exploiter_value_feat_distill_v1`, `--distill-value-feat-coef`,
  default 0 = OFF).** Matching only the teacher's SCALAR V CRYSTALLIZES the critic — the A/B on ai_v7_20
  confirmed `distill/value_mse` falls but the value_cls effective rank DROPS (4.15→3.55): a scalar target has
  only ~1 dim of information, so the critic collapses onto it. The FitNets (Romero 2015) "hint" fix distils
  the teacher's INTERMEDIATE representation instead: `distill_value_feat_coef · (1 − cos(value_pooled_student,
  value_pooled_teacher))` on the SAME teacher-team states, where `value_pooled` is the extractor's 128-dim
  value-CLS pool (`features_extractor.last_value_pooled`, stashed EVERY forward — the hint layer). So the
  trunk inherits the teacher's per-team value STRUCTURE, not just its output. **COSINE, not MSE** (the loss
  choice from the geometry analysis `tmp/fitnet_analysis.py`): the four teachers' value subspaces are low-rank
  (PR ~3–5 even for specialists), COMPLEMENTARY (TSS orthogonal 0.04–0.07 to the others, collective effRank
  ~12), and NON-competing (all pull-cosines positive) — so a scale-free directional pull transfers the correct
  structure without over-constraining a low-rank target the way a raw-magnitude MSE would; the student/teacher
  are common-ancestor forks (all forked from _14), so their `value_pooled` bases are approximately shared and a
  direct (regressor-free) cosine is meaningful (a lower bound on alignment). The student hint (from the
  `evaluate_actions` forward, WITH grad) + each teacher's hint (captured under `no_grad` right after the KL's
  `get_distribution` forward, detached — no extra teacher forward) go through the static `_value_feat_distill`
  (masked mean cosine distance, per-teacher averaged like the KL). **Requires `--distill-coef > 0`** (the
  policy KL makes the teacher's `value_pooled` the right target — V^π is policy-relative). Metrics
  `distill/{value_feat_dist, tK_value_feat_dist}`. **The A/B lever:** scalar (`--distill-value-coef`) vs FitNets
  (`--distill-value-feat-coef`), read out by the value_cls effective-rank probe — does the HINT enrich the
  critic where the scalar crystallized it. Composes with the scalar term (both can be on). OFF byte-identical
  (no teacher `value_pooled` read); training-only, NOT version-locked (inherited on a flagless resume).

  🚨 **The metric is a DISTANCE (`1 − cos`), and the ORIGINAL key name said the opposite.**
  `distill/*_value_feat_cos` records the loss term, so it FALLS toward 0 as the two hints ALIGN — a
  reading of 0.005 means cos ≈ 0.995, i.e. near-PARALLEL. That name produced a real inverted report
  ("the hint is near-orthogonal") off exactly that data. Both metric sites now publish
  **`*_value_feat_dist` as the canonical key**; `*_value_feat_cos` carries the identical value and is
  kept ONE release for TensorBoard continuity, after which it goes. Read the `_dist` key, and treat a
  `_cos` number quoted in any earlier note as a distance that may have been read as a similarity. Pin:
  `instrumented_ppo_test.py::test_value_feat_metric_is_published_under_the_distance_name_too`.

### The OFF-SLICE ANCHOR + the live collateral meters (`--distill-anchor-coef` / `--distill-anchor-mode` / `--distill-anchor-monitor` / `--distill-anchor-parent` / `--distill-anchor-ref` / `--distill-anchor-ema-tau` / `--distill-anchor-refresh-every` / `--distill-anchor-proj-samples` / `--distill-anchor-target-kl` / `--distill-stop`)

`gen3_distill_offslice_anchor_v1` — **a fold's net is teacher content MINUS overshoot damage on the
UNTAUGHT distribution**, and every number above measures only the first half. The 2026-08-31
licensing probe (`designs/research_state/measurements/lr_licensing_probe_2026-08-31.md`) measured the
second: lowering the distill step cut OFF-SLICE collateral **39%** with on-slice absorption unchanged
on all six teacher × seed arms — so the damage is a *systematic direction the distill gradient
carries off the taught slice*, not noise. This ships both halves of the response.

🚨 **BOTH INSTRUMENTS ARE ON BY DEFAULT FOR A FOLD** (`gen3_distill_instruments_default_v1`,
2026-09-03). `--distill-anchor-monitor` and `--distill-stop warn` resolve ON whenever
`--distill-teacher` names at least one teacher **and** `--distill-coef > 0`; every other run is
byte-identical to before. They were opt-in until then, and the cost of that is on the record: one
batch of seven fold arms carried them on three argvs and not the other four, so a pre-registered
cross-check could not be run on the arms that mattered — **an ABSENT series in a column of numbers
reads like a zero**. Neither instrument perturbs training (the monitor attaches no loss term and
changes no parameter; `warn` only logs), so the only thing an opt-in bought was the chance to
forget it. Opt out with `--no-distill-anchor-monitor` / `--distill-stop off`.

Three rules the default follows, and each is load-bearing:

* **A teacher at `--distill-coef 0` is NOT a fold.** The anchor's off-slice split reads the
  `distill_mask` obs key, which the env emits only for a run with a live distill term — and
  `resolve_config` refuses the anchor without one. So the distillation-free arm is untouched, and
  the default can never turn a working command into a usage error.
* **THE DEFAULT YIELDS; AN EXPLICIT FLAG REFUSES** (the `--compile-trainer` rule, one flag over). A
  fold with no resolvable parent — neither `--distill-anchor-parent` nor `--model` — **WARNS and
  leaves the instrument off**, and records `distill_anchor_monitor_source="default-no-parent"` in
  `metadata.json`'s `cli_args` so the missing series is visible rather than silent. A typed
  `--distill-anchor-monitor` there still exits `FATAL_CONFIG`.
* **A live `--distill-anchor-coef` does not ALSO default the monitor on** — it already attaches the
  parent and already emits every meter, so a second name for one thing would be the only effect.
  The stop rule still defaults on there: its dependency is the attached PARENT, not the flag that
  attached it.

The resolved values ride `metadata.json`'s `cli_args` alongside their provenance —
`distill_anchor_monitor_source` and `distill_stop_source`, each one of `cli` / `default` /
`default-off` / `default-no-parent` — so a later reader sees what actually ran, not what was typed.
Resolution lives in `main.train.config._resolve_fold_instruments`, beside the other resolved
defaults, and therefore happens **before** the `cli_args` snapshot. A launcher RESTART re-runs it on
the same forwarded argv and lands on the same answer; the monitor's `parent` reference is a
re-read of the fork-parent path on every launch and persists nothing, so nothing extra rides the
sidecar for it (the MOVING `ema`/`periodic` references and the dual's coefficient still do).

**THE METER (free, and the reason to reach for this first).** Whenever a frozen parent is attached —
including at `--distill-anchor-coef 0` under `--distill-anchor-monitor`, the pure-instrument arm —
every `train()` records, per minibatch and averaged over the call:

| scalar | reads |
|---|---|
| `distill/collateral_kl` | mean `KL(π_ref ‖ π_student)` on the **OFF-slice** rows — the damage. Under a MOVING reference (below) this is a RATE, not a displacement |
| `distill/collateral_kl_vs_parent` | the same against the **FROZEN PARENT**, in EVERY reference mode — the accumulated DISPLACEMENT, and the number the untaught-team meter correlates with. Identical to the row above under the default `parent` reference (one forward, not two) |
| `distill/anchor_ref_age_rollouts` | WHAT the anchor is anchored to: rollouts since the reference was last refreshed (`parent`/`periodic`), or the nominal EMA window (`ema`) |
| `distill/on_slice_kl` | the same on the **on-slice** rows — how far the taught slice has moved |
| `distill/teacher_agreement_on_slice` | student↔teacher top-1 agreement, averaged over ACTIVE teachers — the content |
| `distill/off_slice_frac` | fraction of the minibatch that is off-slice (a sanity check on `--distill-team-bias`) |
| `distill/anchor_loss` | the folded term, **0.0 under monitor-only** — a measured zero, not a gap |
| `distill/anchor_kl` | the unweighted anchor KL (the loss before the coefficient) |
| `grad/distill_anchor_share` | the anchor's shared-trunk gradient share, on the SAME denominator as `grad/distill_share` — the pair IS the dose reading |

**THE REGULARISER.** `--distill-anchor-coef C` folds `C · mean_off-slice KL(π_parent ‖ π_student)`
into the loss, in the **`distill` noise-scale group** (it is part of the fold's dose, not an aux
head). `--distill-anchor-mode` picks the support: `off_slice` (default — the anchor never fights the
teacher on the teacher's own rows; the gradient there is *exactly* zero, pinned by a test) or `all`,
which exists so an arm can TEST that exclusion rather than assume it. The METERS are mode-invariant.
A third value, **`grad_project`**, is not a row set at all but a different MECHANISM — the
source-separated gradient projection; it has its own subsection below.

🚨 **THIS IS NOT R3-SELF, AND CONFUSING THEM WOULD REPEAT A −9pp RESULT.** The rev-3 SELF fold used
self-distillation as the fold **TARGET** at production dose and was destructive. A target DRIVES
steps; an anchor REMOVES freedom. Size `--distill-anchor-coef` as a *fraction* of `--distill-coef`,
never near it. The direction is FORWARD `KL(parent ‖ student)`, matching the teacher term so the two
read on one scale — note the licensing probe reported the REVERSE (`KL(now ‖ original)`), so a probe
figure and a `collateral_kl` figure are comparable in TREND, not in absolute value.

🚨 **THE RESTART RULE — the one way this feature fails silently.** The parent is **re-loaded from the
ORIGINAL FORK-PARENT PATH on every launch**, never from the current checkpoint. It has to be: a
launcher run restarts every few hours, and an idempotent fork's `--model` is swapped to the fork's
OWN latest checkpoint on each relaunch (`launcher.checkpoint.resolve_fork_resume_model`), so
"anchor to what we just loaded" would re-anchor to a DRIFTED policy at every restart — the trust
region would follow the student, ratchet by ratchet, constraining nothing while still reading as ON.
Resolution (`agents.training.distill_anchor_callback.resolve_anchor_parent`), in order:
`--distill-anchor-parent` → `<run>/metadata.json`'s **immutable** `original_command` → its `--model`
→ this process's `--model` (correct on a fork's FIRST launch, where the run dir is fresh and
`resolve_launch_run_dir` refuses to fork onto an existing run). The resolved path AND its route are
PRINTED at startup (`⚓ [DISTILL-ANCHOR] parent = … (via original_command)`), because a wrong parent
is a silent no-op and the only defence against a silent no-op is a loud statement of the choice.
Verified live: a relaunch with `--model <the fork's own final_model.zip>` resolved
`via original_command` back to the fold parent. Unresolvable ⇒ `FATAL_CONFIG`, never "anchor off".

**Where it lives.** The loss + meters are `instrumented_ppo/distill_anchor.py` (pure helpers +
`distill_anchor_step`, ONE call site in `ppo.py`); the parent resolution + the frozen load are
`agents/training/distill_anchor_callback.py`, registered from `main/train/callbacks.py` — a
CALLBACK rather than an `apply_training_hparams` row precisely because `_on_training_start` runs on
every launch, which is the cadence the re-load needs. The parent is in `_excluded_save_params`: a
pickled copy would be a second, wrong answer that survives restarts. Requires `--distill-coef > 0`
(the slice IS the `distill_mask` obs key, which the env emits only for a live distill). OFF and
byte-identical with no parent attached — and byte-identical with a parent attached at coefficient 0,
which is what makes the monitored control arm comparable rather than a third condition. Gate:
`src/agents/training/distill_anchor_test.py`.

#### THE REFERENCE — `--distill-anchor-ref {parent,ema,periodic}` (default `parent`)

**WHY THE REFERENCE IS NOT REFRESHED THE WAY PPO'S CLIP REFERENCE IS.** The clip and the anchor
bound DIFFERENT QUANTITIES, and the cadence follows from that rather than from taste. PPO's clip
bounds the per-update **RATE** against the policy that collected the data, so it is re-read every
rollout by construction — an update is only meaningful relative to the behaviour it is an update
*of*. The anchor bounds the accumulated **DISPLACEMENT** from the fold start: that is the quantity
the 2026-08-31 licensing probe measured, and it is what rev-4's untaught robbery is made of. And
that collateral is **SYSTEMATIC** — the same off-slice direction every step — which is precisely
the regime a following reference does not resist, because it moves along with the drift and reads a
small KL the whole way down.

So the default is FIXED, which is **Learning-without-Forgetting**'s design (Li & Hoiem 2016: distil
against a snapshot of the model taken *before* the new task). The alternative is **ACER**'s trust
region (Wang et al. 2016: KL to an average-policy network, α ≈ 0.99), and it exists here for one
specific reason: **a fixed reference cannot tell a GIFT from a ROBBERY.** v8's fold changed
off-slice switching behaviour by **+5.4pp** and that change was GOOD — but to a fixed anchor it is
displacement like any other, so a large enough coefficient suppresses it. An average lets slow
consistent improvement through (the average follows it) while still taxing fast overshoot (the
average lags it). `ema` is the arm to have if the fixed anchor turns out to suppress the gift.

| `--distill-anchor-ref` | the reference is | knob | precedent |
|---|---|---|---|
| **`parent`** (default) | the FIXED frozen fold parent — **byte-identical to what the anchor shipped with** | — | LwF |
| `ema` | a Polyak average of the STUDENT, `ref ← τ·ref + (1−τ)·student` | `--distill-anchor-ema-tau` (0.99) | ACER |
| `periodic` | re-snapshot from the student every N rollouts | `--distill-anchor-refresh-every` (8; **0 = never = `parent`**) | — |

**All three are INITIALISED FROM THE PARENT**, so at fold start they hold the same reference and
only diverge as the student moves. The degenerate settings collapse exactly: `--distill-anchor-ema-tau
1.0` reproduces `parent`'s update and its KL bit-for-bit, and `0.0` makes the reference the current
student, so the anchor loss goes to ~0.

**THE UPDATE CADENCE IS ONE PER `train()` CALL, taken in `_on_rollout_end`.** SB3's `learn()` is
`collect_rollouts → callback.on_rollout_end() → train()`, one train per rollout, and there is **no
hook after `train()` at all** — so this is the only per-`train()` cadence a callback can have, and
it is the right resolution for the quantity: a per-optimizer-step reference would cost
`n_epochs × n_minibatches` times as much for a number nobody reads that finely, and would have
required a second seam in `ppo.py`. The phase that follows is "the reference used inside `train()` k
is the average of the policies that produced the data", which is the correct side of the boundary —
the anchor is a trust region on the policy being updated, not on the update.

**THE WINDOW, in the units that matter.** `1/(1−τ)` **train() CALLS**, and one call is one rollout,
so at the production shape (`--n-envs 48 --n-steps 2048` ⇒ **98,304 env steps per rollout**):

| τ | window (train calls) | window (env steps) |
|---|---:|---:|
| 0.9 | 10 | ~0.98M |
| **0.99 (default)** | **100** | **~9.8M** — most of a generation |
| 0.999 | 1000 | ~98M — longer than any run here, i.e. effectively `parent` |

**THE DISPLACEMENT METER IS EMITTED IN EVERY MODE, and that is the point of separating the two
keys.** `distill/collateral_kl` reads the ANCHOR's own reference, so under `ema`/`periodic` it is a
RATE, not a displacement — and the displacement is what the untaught-team meter correlates with. So
`distill/collateral_kl_vs_parent` always reads `KL(frozen PARENT ‖ student)` on the off-slice rows,
whatever the anchor is anchored to, and the frozen parent stays loaded in all three modes for
exactly this (~2M params; memory is not the constraint). Under `parent` the two are the SAME number
computed once, so the default arm pays **no second forward**. `distill/anchor_ref_age_rollouts` says
what the anchor is anchored to: rollouts since the reference was last refreshed under
`parent`/`periodic` (so `parent`'s rises for the life of the fold, which is the honest reading), and
the nominal EMA window under `ema`, because a geometric average has no age.

**MEASURED on a `--debug` fold** (a 3k-step teacher, then a 15k-step fold at `--distill-coef 0.2
--distill-anchor-coef 0.05`, `distill/*` read back from `tb/`) — the two meters doing the two jobs:

| arm | `collateral_kl` (vs the anchor's reference) | `collateral_kl_vs_parent` | `anchor_ref_age_rollouts` |
|---|---|---|---|
| `parent` | 0.00035 → 0.0120 | **identical** to the left column | 1 → 5 (rises for the fold) |
| `ema` τ=0.5 | 0.00038 → 0.0012 (**flat**) | 0.00038 → **0.0186** | 2.0 (the nominal window) |
| `periodic` every 2 | 0.00037 → 0.0008 (saw-tooth) | 0.00037 → **0.0100** | 1, 0, 1, 0 (the cadence) |

**🚨 THE MOVING REFERENCE IS RUN STATE AND MUST BE PERSISTED — the mirror of the parent's restart
rule, not an exception to it.** The parent is re-read from a PATH on every launch precisely so a
restart cannot re-anchor to a drifted policy. `ema`/`periodic` have no path — the reference is a
function of THIS run's own trajectory — so re-initialising it on every launcher restart would reset
the trust region to fold start every few hours, silently, while still reading as ON. It is therefore
written as a **`<checkpoint>_anchor_ref.pt` SIBLING** at every site that records a resumable
checkpoint (the periodic callback, the SIGUSR1 forced save, the SIGTERM abort save, and both final
saves) and restored from the sibling of this launch's `--model`.

- **Beside the checkpoint rather than at the run root, for consistency.** A restart rewinds the
  policy to a checkpoint; a run-level file would hold whatever the reference was when the process
  died, i.e. AHEAD of the weights it is a trust region for.
- **A restore is REFUSED unless the blob's `run_dir`, resolved `parent_path`, `ref` mode and schema
  all match this launch** — the FORK GUARD: a fork off a fold's `final_model.zip` would otherwise
  inherit that fold's average as its own starting reference, which is a different run's trajectory
  wearing this run's name. (Verified live: a relaunch that resolved into a fresh run dir REFUSED the
  sibling by name and re-initialised from the parent.)
- **Every refusal, and a missing sibling, is stated on the startup line**, and the wording splits the
  two meanings of an absent file: on a RESTART (`--model` inside this run dir, the `--fork-lr`
  predicate) it reads *"EXPECTED one … the trust region has been RESET to fold start"*; on a fork's
  first launch it reads *"initialising from the PARENT (a fork's first launch)"*.
- Verified live across a real process restart: `RESTORED from …/checkpoint_16384_steps_anchor_ref.pt
  (saved at 16384 steps, 5 rollouts since its last refresh)` — weights AND the periodic cadence.

**THE MOVING REFERENCE IS A SECOND, INDEPENDENT LOAD OF THE PARENT — never a `deepcopy` of the live
student**, and that is a correctness decision rather than thrift. The extractor carries a per-forward
`ExtractorStashes` full of NON-LEAF tensors, which `deepcopy` refuses outright; and
`--compile-trainer` patches the BOUND `fe.forward` as an INSTANCE attribute, which `deepcopy` treats
as **ATOMIC** — so the copy's `forward` would still be closed over the LIVE extractor and every
"frozen reference" logit would silently be the student's own, reading a KL of exactly **0** forever
while every meter looked healthy. A second `load_parent(path)` has neither problem, is arch-identical
by construction, and starts at exactly the weights all three modes are supposed to start at.
`assert_reference_matches_student` then RAISES (→ `FATAL_CONFIG`) unless the two share one
`state_dict`, because a `polyak_update_` that quietly skipped the keys it could not find would report
a trust region against a partly-frozen reference.

**Class: training-runtime, exactly like `--distill-anchor-coef`.** None of the three flags reaches
the extractor, scales a weight shape, or belongs in `check_compatible`; all three carry an argparse
default of `None` and are `_resolve`d in `main/train/config.py`, so an unset flag lands on the
byte-identical default in one place and a flagless resume keeps the arm it was launched as. Refusals:
a reference knob with no live anchor, `--distill-anchor-ema-tau` outside `[0, 1]` (it is a convex
weight — outside that it EXTRAPOLATES away from the student and would still train and still read as
ON), and a negative refresh cadence. `_distill_anchor_ref` and `_distill_anchor_ref_writer` join the
parent in `_excluded_save_params` — the second is the CALLBACK, which back-references the model and
SB3's `Logger` (a `_contextvars.Context`), so pickling it would break every save in the run.

**THE FIRST CELL USES `parent`, monitor-or-fold** — the default, byte-identical to the arm the
licensing probe motivated. `ema`/`periodic` are the follow-on arms, and the pre-registered reading is
the one this section opened with: run them only if the fixed anchor is measured to suppress a GIFT
(off-slice change that is *good*), and read `collateral_kl_vs_parent` — not `collateral_kl` — against
`teacher_agreement_on_slice` in every one of them.

#### SOURCE-SEPARATED anchoring — `--distill-anchor-mode grad_project` (+ `--distill-anchor-proj-samples`)

`gen3_distill_grad_project_v1` — **the one thing an OUTPUT anchor structurally cannot do.** The
2026-09-01 gift/decay pair (ledger: *"v8's GIFT IS A TRANSIENT HUMP"*, *"WHAT v8's LAST 2.5M
UNDID"*) measured that a fold does two things in two directions:

| | what it is | direction | worth (untaught) |
|---|---|---|---|
| **the GIFT** | an early off-slice HABIT change, PPO-driven | ORTHOGONAL to the teachers' fingerprint (cos 0.14) | **+5 to +10pp**, 92% intact at +15M |
| **the LEAK** | the taught content arriving on untaught boards through shared weights | PARALLEL to it (cos +0.559, perm p 0.0015, ~⅓ amplitude) | **−5.66pp [−12.1, −0.2]** |

At the OUTPUT there is nothing left to tell apart — both are "the off-slice policy moved away from
the parent" — so `--distill-anchor-coef` at a fixed reference taxes the gift exactly as hard as the
leak. That is the ledger's design consequence (a), and it is the reason `ema`/`periodic` exist. **At
UPDATE time they are separable, because they have different SOURCES**: the total update is
`g_ppo + g_distill`, and this tree already computes those separately (the `_ntg.add("distill", …)`
tags the per-term noise sampler reads). The gift is PPO's; the leak is the distill term's.

**THE MECHANISM.** Every optimizer step, sample `m` OFF-SLICE rows of the micro-batch, take
`∇_θ log π_θ(a* | s)` for each (`a*` = the student's own argmax over the LEGAL set — no label, no
reference forward), orthonormalise them into a basis `Q`, and step with

```
g_total = g_ppo + P⊥ g_distill,      P⊥ g = g − Σ_j q_j ⟨q_j, g⟩
```

To first order the distill term then moves those off-slice log-probs by **zero**, while its
component along every direction that only moves TAUGHT states survives in full. **PPO's gradient is
never read, never projected, never scaled** — pinned by a test that recomputes
`g_ppo + P⊥ g_distill` independently and compares `.grad` to it.

**PRECEDENT:** Orthogonal Gradient Descent (Farajtabar et al. 2020, AISTATS) and Gradient Projection
Memory (Saha et al. 2021, ICLR), at a different seam — those project the new-TASK gradient off a
banked memory of old-task directions; here the two "tasks" are two TERMS of one loss at one
timestep, and the basis is rebuilt from the live minibatch every step rather than banked.

| scalar | reads |
|---|---|
| `distill/proj_removed_frac` | `‖g − P⊥g‖² / ‖g‖²` — the LEAK's share of the distill gradient, by this estimate |
| `distill/proj_rank` | constraints that SURVIVED Gram-Schmidt (≤ m); this is what was actually projected along |
| `distill/proj_constraint_rows` | off-slice rows sampled (= `min(m, #off-slice)`) |
| `distill/proj_ms` | wall-clock per micro-batch — read it against `train/train_ms` |
| `distill/collateral_kl_vs_parent` | **the experiment's readout**, and it is ON in this mode by construction (below) |

**IT IS A MODE, NOT A ROW SET.** `grad_project`'s OUTPUT half is `off_slice`'s, so
`--distill-anchor-coef 0` is projection-only and a positive coefficient **COMPOSES** an off-slice
output anchor on top — deliberately supported, because the projection is per-step and FIRST-ORDER
(the constraint set is resampled every step; curvature carries the policy off the tangent plane) and
the output anchor is what bounds the ACCUMULATED displacement. Any `--distill-anchor-ref` still
applies to that output half. **`grad_project` registers `DistillAnchorCallback` on its own** — even
at coefficient 0 and with no `--distill-anchor-monitor` — for two reasons: that callback is the only
site that sets `distill_anchor_mode` on the model, and the frozen parent it loads is what makes
`collateral_kl_vs_parent` exist. An experiment whose readout is optional is an experiment nobody
reads. (`resolve_config`'s `_anchor_wanted` and the registration condition in `main/train/callbacks.py`
are pinned to agree.)

**PER MICRO-BATCH, NOT PER ACCUMULATION GROUP**, and at the default `--grad-accum-steps 1` the two
are identical. (1) The `m` constraint vectors are full-parameter-sized, so holding them across a
group multiplies peak memory by `accum` — and `--grad-accum-steps` exists to CUT the memory peak.
(2) Per-micro needs one call after `backward()`; per-group would need an apply before
`clip_grad_norm_` at both step sites plus a reset on the KL early-stop discard, and `ppo.py` sits
~1.8k lines under a hard 2,000 gate. (3) `Σᵢ P⊥ᵢ gᵢ` removes at most what `P⊥_∪ (Σᵢ gᵢ)` would, so it
is the conservative one.

🚨 **THE COST IS REAL — MEASURE IT, DO NOT ASSUME IT.** The `m` constraint backwards run over a graph
built from only the `m` sampled rows (the obs are SLICED before the forward), so each is ~`m/B` of a
full backward *in FLOPs* — but at small `m` on CPU the extractor is dispatch-bound and the FLOP
argument does not survive contact with a clock. Measured on the build smoke (CPU `--debug`,
`--n-steps 512 --batch-size 128`, `m = 16`, real 2501-dim extractor, box carrying a live fleet):
`distill/proj_ms` **426–644 ms per micro-batch**, `train/train_ms` **12.9–19.1 s** against the
monitor-only arm's **4.2–5.9 s** — i.e. the projection was **~55–70% of `train()`**, ~2.5–3× the
step. The share should fall as `--batch-size` rises and on a GPU (per-row numerator, per-batch
denominator), **but that has not been measured**; read `proj_ms` against `train/train_ms` on your own
arm.

⚠️ **`proj_removed_frac` MEASURED 0.75–0.89, AND THAT IS THE FINDING RATHER THAN A BUG.** A random
vector's projection onto a random 16-dim subspace of a ~2M-dim space keeps ~1e-5 of its energy;
removing ~80% means the distill gradient and the off-slice behaviour gradients share their dominant
directions almost entirely — the "shared weights carry taught content onto untaught boards"
mechanism, seen directly at the update, and the same fact from the other side as M4's fingerprint
cosine. It also states the method's ceiling: **where a direction BOTH teaches and leaks, a
first-order projection cannot keep the teaching**, and at `m = 16` most of the teacher term's
magnitude went with the leak. `distill/proj_rank` came out at ~15.4 of 16, so the sampled directions
are near-independent — lowering `m` trades removal for coverage, it does not de-duplicate anything.

**THE BUILD SMOKE'S TWO SERIES (n = 1, A SMOKE, NOT A RESULT).** Same 3k teacher and 3k fold parent,
two 6k folds at `--distill-coef 0.2 --distill-anchor-monitor`, `distill/collateral_kl_vs_parent`
read off `tb/`. **The projected arm's collateral was HIGHER, not lower:**

| step | `grad_project` | `off_slice` (monitor-only, no projection) |
|---:|---:|---:|
| 4096 | 0.00342 | 0.00376 |
| 4608 | 0.00700 | 0.00673 |
| 5120 | 0.01388 | 0.00951 |
| 5632 | 0.02733 | 0.00976 |
| 6144 | **0.03373** | **0.01004** |

Two separate processes, different dice, five points, no CI, 6k CPU steps — this settles nothing and
is recorded so nobody has to re-derive it. The parsimonious reading is the `removed_frac` one: with
~80% of the distill gradient removed the fold pulls far less overall (`distill/kl` stayed HIGHER in
the projected arm — 0.027–0.035 vs 0.021–0.026, i.e. LESS absorbed), so PPO dominates the off-slice
motion, and `collateral_kl_vs_parent` counts PPO's own off-slice displacement — **the GIFT** — as
collateral just like the leak. That is the meter's known limitation, not a new one: it is an
output-displacement meter, and this whole mode exists because output displacement does not separate
the two. **A real verdict needs the untaught-team win-rate meter, paired arms and a CI**, not this.

**Class: training-runtime.** Neither flag reaches the extractor, scales a weight shape, or belongs in
`check_compatible`; both carry an argparse default of `None` and are `_resolve`d in
`main/train/config.py`. Refusals: `--distill-anchor-proj-samples` outside `grad_project`, or below 1;
and `grad_project` still requires `--distill-coef > 0` like every other anchor mode (the slice IS the
`distill_mask` obs key). Where it lives: `instrumented_ppo/distill_grad_project.py` (all of the
mechanism), a four-line seam in `ppo.py` (construct · `_dgp.add` on the three TEACHER terms and
**not** the anchor term · `before_backward` · `after_backward`), `distill_anchor_proj_samples` pushed
onto the model by `DistillAnchorCallback` beside the mode. The `autograd.grad` helper is SHARED with
`PerTermNoiseSampler` (`noise_scale_terms.term_gradient`) rather than forked — the projection's
correctness argument is that it operates on the same object the meters report. `--compile-trainer` is
not a new risk: `retain_graph` backwards through the compiled extractor already happen for the
noise-scale and grad-balance probes. Gate:
`src/agents/training/instrumented_ppo_distill_grad_project_test.py` (34 tests — the projection
algebra, `.grad == g_ppo + P⊥ g_distill` recomputed independently, the accum scaling, off-is-free,
and the first-order claim measured end to end: **100.0%** reduction of the fold's off-slice log-prob
movement with the optimizer swapped for plain SGD, **95.1%** on the real Adam + `clip_grad_norm_`
path — Adam rescales per coordinate, so a projection of the GRADIENT is not exactly a projection of
the UPDATE).

#### THE FOLD STOP RULE (`--distill-stop`) + the DUAL-ASCENT anchor coefficient (`--distill-anchor-target-kl`)

`gen3_distill_stop_rule_v1` — **a fold has an OPTIMAL LENGTH, and the two meters above are what
says when it has passed.** The 2026-09-01 pair (ledger: *"v8's GIFT IS A TRANSIENT HUMP"*, *"WHAT
v8's LAST 2.5M UNDID"*) measured v8's untaught-team gain peaking at **+9.67pp [+6.79, +12.50]**
around +12.5M and falling to **+4.98pp** by +15.04M, with distillation still running at
`--distill-coef 1.0` against teachers it had already absorbed. Nothing was unlearned — the gift is
**92% intact at cosine 0.864** — the decline is the LEAK: the taught content continuing to arrive on
untaught boards, parallel to the teachers' own fingerprint (cos +0.559, perm p 0.0015), costing
**−5.66pp [−12.1, −0.2]** there while costing nothing on taught teams. Design consequence (c) of
that entry names the live signal, and this is the mechanism that acts on it.

**THE TWO SIGNALS AND WHY BOTH.**

| detector | reads | fires when |
|---|---|---|
| **PLATEAU** | `distill/teacher_agreement_on_slice`, EMA α=0.2 | `ema[t] − ema[t−W] < --distill-stop-eps` (default **0.005**, ABSOLUTE, in top-1 agreement-rate units). SIGNED, so a FALLING agreement is a plateau too — it is not absorbing either |
| **RISE** | `distill/collateral_kl_vs_parent`, the **RAW** readings | OLS slope over the last `W+1` readings is `> 0` AND `> --distill-stop-kl-slope × se(slope)` (default **2.0** — a one-sided t-test) |

`--distill-stop-window` (default **8** rollouts, minimum 2) is `W` for both.

Rising collateral ALONE is an ordinary fold in progress — leak and teaching arrive together, and
paying collateral for content is the trade the fold exists to make. A plateaued agreement ALONE is a
fold that has merely finished absorbing. It is the **conjunction** — displacement still accumulating
with nothing left to absorb — that is R3-SELF's content-free regime seen from the inside, and the
AND-gate must hold for `--distill-stop-persist` consecutive rollouts (default **3**) before firing.
A rollout that breaks either half resets the count; a rollout where either meter does not read is
SILENCE — the count neither advances nor resets, `rank_tripwire`'s rule verbatim.

🚨 **THE RISE THRESHOLD IS A t-MULTIPLE, NOT NATS PER ROLLOUT.** Collateral KL's absolute scale
moves by two orders of magnitude across configs (the anchor's own build smoke read 0.00035 early and
0.034 late), so no absolute slope could be quoted in a help string and still be right on the next
arm. A t-statistic asks the only question that transfers: *is this rising by more than its own
wobble?*

🚨 **AND THE TREND IS FIT ON THE RAW SERIES, NOT ITS EMA — a measured correction, not a
preference.** An EMA is a low-pass filter, so consecutive points are strongly autocorrelated and an
OLS fit through it has residuals far smaller than the series' own noise; its standard error
understates the uncertainty by a large factor. Fitting the EMA therefore reported a SIGNIFICANT
positive trend on **white noise** — caught while building this: a zero-mean wobble (sd 0.004 around
a level 0.01) passed `t > 2` at a 6-rollout window, i.e. the detector would have fired on a fold that
was not drifting at all. The PLATEAU half keeps its EMA, because it compares two LEVELS and
autocorrelation does not bias a level. Both halves are pinned, including a test that reproduces the
EMA-fit false-positive rate so nobody tidies the fit back onto the smoothed series.

**THE THREE ACTIONS**, in increasing order of what they take away — all after the same fire:

| `--distill-stop` | does |
|---|---|
| `off` (**default**) | registers no callback at all; byte-identical |
| `warn` | one launcher event + `distill/stop_signal` latched to 1. Nothing about the run changes |
| `anneal` | plus `--distill-coef ×= --distill-stop-anneal-factor` (default **0.7**) every subsequent rollout, to a floor of 0. The fold winds DOWN over ~a dozen rollouts rather than vanishing between two, so nothing about the loss landscape moves discontinuously under an optimizer carrying momentum |
| `abort` | plus stopping `learn()` at the next step — `_on_step` returns False, exactly `--rank-tripwire abort`'s channel, so the run's normal end-of-learn save happens and the process exits **COMPLETE**, not CRASH; the launcher does not restart-loop |

**The anneal's floor is exactly 0, and it SNAPS.** A geometric decay never reaches zero, and a
`distill_coef` of 1e-12 still pays a full teacher forward per minibatch per teacher for a term that
changes nothing — so the decay snaps to exactly 0.0 once below `1e-6` of the coefficient in force at
the fire (~39 rollouts at 0.7). At exactly 0 `train()`'s `distill_on` predicate goes False: the
teacher forwards stop and `distill/teacher_agreement_on_slice` stops existing — the fold really is
over. The ANCHOR's meters keep reading, because `collateral_kl_vs_parent` and `on_slice_kl` depend
on the frozen parent and the `distill_mask` obs key, neither of which the coefficient gates. Pinned
end to end through the real `train()`.

**It REQUIRES the anchor monitor** (`--distill-anchor-coef > 0`, `--distill-anchor-monitor`, or
`--distill-anchor-mode grad_project`) — the rise half reads `collateral_kl_vs_parent`, which exists
only when the frozen parent is attached, and without it the AND-gate could never close while the
flag still read as ON. `resolve_config` refuses rather than shipping that silent no-op.

**THE DUAL — `--distill-anchor-target-kl` (default 0 = off).** The anchor's coefficient is a number
nobody can tune; this turns it into a **constraint with a readable budget**. Once per rollout:

```
kl_ema <- alpha*kl + (1-alpha)*kl_ema                  (alpha = 0.2, half-life ~3 rollouts)
coef   <- clip( coef * exp(eta * (kl_ema/target - 1)),  --distill-anchor-coef-{min,max} )
```

`eta` is `--distill-anchor-dual-lr` (default **0.1**), the clamps default to `0.0` and **10× the
starting `--distill-anchor-coef`**. It is gradient ascent on the Lagrange multiplier of
`minimize L(θ) s.t. KL ≤ target`, taken in LOG-coefficient space so the multiplier stays positive by
construction and a correction is proportional to the RATIO of the violation — which matters because
collateral KL spans two orders of magnitude and a fixed additive step would be a different
controller at each of them. Precedent: PPO-penalty's adaptive β (Schulman et al. 2017 §4) and MPO's
Lagrangian dual (Abdolmaleki et al. 2018). `distill/anchor_coef` is recorded **every rollout
whenever the anchor is attached** — a flat line under a static coefficient, on purpose, so a dual
arm and a static arm carry the same series (the `anchor_loss`-as-a-measured-zero rule);
`distill/anchor_dual_kl_ema` and `distill/anchor_dual_clamped` ride beside it.

**WHICH METER THE DUAL BUDGETS, and it depends on the reference.** A dual variable must be attached
to a quantity its own lever can MOVE, or it winds up against a clamp and sits there while still
reading as a live controller:

| `--distill-anchor-ref` | the dual reads | because |
|---|---|---|
| **`parent`** (default) | `distill/collateral_kl_vs_parent` | the ACCUMULATED-DISPLACEMENT meter — the quantity the untaught robbery is made of, and under a fixed reference exactly what the anchor loss penalises |
| `ema` / `periodic` | `distill/anchor_kl` | under a moving reference the anchor DELIBERATELY does not resist parent-displacement — that is what lets v8's +5.4pp GIFT through — so a dual budgeted on it could never satisfy its constraint |

`collateral_kl_vs_parent` is still logged in every mode; the choice is only about which number the
dual **acts** on.

**EVERY ROLLOUT, NO COOLDOWN — and that is a difference in KIND from the LR ladder.** The KL-driven
lr controller is bang-bang (a fixed multiplicative step whenever the EMA leaves a band), so it
COMPOUNDS while the EMA lags and its 7-rollout cooldown is what stops the overshoot. This is an
INTEGRATOR: the step shrinks to nothing as the constraint is met, and eta alone sets the response
timescale (at 0.1 a sustained 2× overshoot moves the coefficient +10.5%/rollout, ~7 rollouts to
double). Adding a cooldown to an integrator inserts DEAD TIME, which is the classic cause of the
oscillation a cooldown is meant to prevent. The EMA is the only smoothing this loop needs.
`--distill-anchor-coef 0` is **refused** with a target-kl: the update is multiplicative, so 0 is a
fixed point and the controller would run forever without moving anything.

🚨 **BOTH ARE RESTART STATE, and this is the mirror of the anchor's own restart rule.** A launcher
run restarts every few hours and **forwards the ORIGINAL argv**, so without persistence: a dual that
had climbed to 5× its starting coefficient would silently reset to 1× at every restart; a detector
would need its whole 8-rollout window again each time and might never fire at a 3h cadence, while
reading as ON throughout; and an explicit `--distill-coef 0.3` would re-install itself at full
strength over a completed anneal. So the dual's `(coef, kl_ema, n)` and the detector's two EMAs,
both histories, the hold count, the latch and the annealed coefficient are written into the
**checkpoint sidecar** — the same place `handoff_lr` and `grad_accum_steps` ride, via
`_model_hparams`, and written **only when the mechanism is live**, so an ordinary run's sidecar is
byte-for-byte what it always was. `_on_training_start` restores them and **RE-APPLIES the annealed
coefficient over the argv's** (only when the persisted value is LOWER — an operator who deliberately
raised it between restarts is not overruled by a stale wind-down). A restart after an `abort` refuses
at the first step rather than collecting one more rollout.

| scalar | reads |
|---|---|
| `distill/anchor_coef` | the coefficient in force this rollout (flat under a static one) |
| `distill/anchor_dual_kl_ema` · `distill/anchor_dual_clamped` | the dual's smoothed budget reading, and whether the last update hit a clamp |
| `distill/stop_state` | 0 armed · 1 plateau · 2 plateau+rise · 3 FIRED |
| `distill/stop_signal` | 0/1, latched at the fire |
| `distill/stop_rollouts_since_fire` | how long the run has been past its own stop point |

🚨 **`warn` IS THE DEFAULT FOR A FOLD; `anneal`/`abort` STAY OPT-IN UNTIL THE THREE-DOSE CELL'S
CURVES SIZE THE WINDOW.** `--distill-stop-window 8` and `--distill-stop-eps 0.005` are derived from
nothing but the shape of the v8 curve at a cadence this tree has never run a fold at — so a
mis-sized rule that ABORTS is a new way to lose a training window, while a mis-sized rule that WARNS
is exactly the calibration data `anneal` and `abort` need, at zero risk. Hence the split: `warn`
arrives by default on every fold (`gen3_distill_instruments_default_v1` — the section head above),
and giving the rule teeth is still something an operator types.

**Where it lives.** `agents/training/distill_stop_callback.py` holds both PURE controllers
(`AnchorDualAscent`, `FoldStopDetector`, `ols_slope_and_se` — no SB3, no torch, no logging) and the
`DistillStopCallback` wrapper; the dual is driven from `DistillAnchorCallback._on_rollout_end`,
which already owns the coefficient and already runs once per `train()`. `ppo.py` is **untouched** —
both mechanisms act on `model.distill_anchor_coef` / `model.distill_coef`, which `train()` already
reads. Registration: `main/train/callbacks.py`. Gate:
`src/agents/training/distill_stop_callback_test.py` (the dual's sign, its convergence to
the target against a closed-loop plant, both clamps, each detector on planted series, the EMA-fit
false-positive reproduction, the AND-gate + persist + reset, all three actions, the sidecar
round-trip, the byte-identity/config-refusal set, and — §7 — the whole
`gen3_distill_instruments_default_v1` matrix read off `build_callbacks`'s REAL callback list:
teacherless and `--distill-coef 0` attach nothing, a fold attaches the anchor exactly once at
coefficient 0 with the stop rule armed in `warn`, both opt-outs win, a live coefficient attaches
exactly once, and an unresolvable parent warns instead of exiting).

**THE BUILD SMOKE (n = 1, A SMOKE, NOT A RESULT).** A 3k-step teacher and a separate 3k-step fold
parent, then a fold at `--distill-coef 0.3 --distill-anchor-coef 0.02 --distill-anchor-target-kl 0.01
--distill-stop warn --distill-stop-window 3 --distill-stop-persist 1` (CPU `--debug`,
`--n-steps 512 --batch-size 128`, box carrying a live fleet), every series read back off `tb/`:

| step | `anchor_coef` | `anchor_dual_kl_ema` | `teacher_agreement_on_slice` | `collateral_kl_vs_parent` | `stop_state` |
|---:|---:|---:|---:|---:|---:|
| 3584 | 0.020000 | 0.000000 | — | — | 0 |
| 4096 | 0.018711 | 0.003335 | 0.5000 | 0.003335 | 0 |
| 4608 | 0.017587 | 0.003809 | 0.4219 | 0.005704 | 0 |
| 5120 | 0.016616 | 0.004318 | 0.4703 | 0.006356 | 0 |
| 5632 | 0.015919 | 0.005715 | 0.4836 | 0.011300 | **3** |
| 6144 | 0.015353 | 0.006380 | 0.4139 | 0.009041 | **3** |
| 6656 | 0.014927 | 0.007185 | 0.4726 | 0.010406 | **3** |
| 7168 | 0.014678 | 0.008321 | 0.5014 | 0.012863 | **3** |
| 7680 | 0.014595 | 0.009429 | 0.5239 | 0.013860 | **3** |
| 8192 | 0.014604 | 0.010065 | 0.4828 | 0.012608 | **3** |
| 8704 | 0.014657 | 0.010360 | 0.5069 | 0.011541 | **3** |
| 9216 | 0.014811 | 0.011044 | 0.4398 | 0.013782 | **3** |
| 9728 | 0.015346 | 0.013549 | 0.4159 | 0.023570 | **3** |
| 10240 | 0.015884 | 0.013447 | 0.4005 | 0.013037 | **3** |
| 10752 | 0.016482 | 0.013699 | 0.4063 | 0.014705 | **3** |
| 11264 | 0.017075 | 0.013532 | 0.4777 | 0.012868 | **3** |
| 11776 | 0.017810 | 0.014212 | 0.7030 | 0.016929 | **3** |
| 12288 | 0.019662 | 0.019897 | 0.7609 | 0.042639 | **3** |
| 12800 | 0.022570 | 0.023791 | 0.6425 | 0.039367 | **3** |
| 13312 | 0.026510 | 0.026090 | 0.5170 | 0.035285 | **3** |
| 13824 | 0.032536 | 0.030482 | 0.4681 | 0.048049 | **3** |
| 14336 | 0.039793 | 0.030135 | 0.5612 | 0.028749 | **3** |
| 14848 | 0.047643 | 0.028005 | 0.4579 | 0.019482 | **3** |
| 15360 | 0.056711 | 0.027424 | 0.5194 | 0.025100 | **3** |

**Five things it shows, and one it does not.**

1. **THE DUAL'S SIGN IS RIGHT, AND ITS TRAJECTORY IS V-SHAPED — the constraint tracking its budget
   in both directions.** The collateral EMA opens far BELOW the 0.01 target, so the coefficient walks
   DOWN (0.02000 → 0.01459 over six rollouts, ~−4%/rollout, which is what η = 0.1 at that ratio
   predicts). It bottoms out at step 7680, exactly where the EMA is closing on the target, TURNS
   AROUND as the EMA crosses 0.01 at step 8192, and then climbs for the rest of the run as the
   collateral keeps rising — ending at 0.0567 against an EMA of 0.0274. An integrator settling on a
   budget and then defending it, live; **`distill/anchor_dual_clamped` reads 0 on every rollout**, so
   it did all of that in the interior and never against a clamp.
2. **`anchor_kl` EQUALS `collateral_kl_vs_parent` TO EVERY DIGIT** — the documented identity under
   `--distill-anchor-ref parent` + `--distill-anchor-mode off_slice`: one frozen forward, two names.
3. **THE AND-GATE CLOSES AT EXACTLY THE ROLLOUT THE ARITHMETIC ALLOWS.** The rise test needs
   `window + 1 = 4` readings, so 5632 is the first rollout on which it *can* fire; the agreement EMA
   had not improved over the window (0.500002 → 0.481977, i.e. −0.018 against an eps of +0.005 — a
   plateau under the SIGNED rule, which is what a FALLING agreement is), collateral had risen
   monotonically, and at `persist 1` `stop_state` went straight to 3 with `stop_signal` latched.
   `stop_rollouts_since_fire` then counts 1, 2, 3, … as it should, and the detector's frozen
   histories confirm the latch stopped its EMAs advancing.
4. **EVERY NEW SERIES EXISTS FROM THE FIRST ROLLOUT BOUNDARY** — `anchor_coef` is already present at
   step 3584, before any `train()` has run, which is the "a series a reader can compare across arms"
   rule the `anchor_loss`-as-a-measured-zero convention set.
5. **BOTH CONTROLLERS REACHED THE SIDECAR**, read straight back out of the run's `metadata.json`:
   `distill_anchor_dual_state = {"coef": 0.056711, "kl_ema": 0.027424, "n_readings": 23}` and
   `distill_stop_state` carrying `fired: true`, `rollouts_since_fire: 19`, both 4-entry histories,
   and `distill_coef_annealed: 0.3` — unchanged, correctly, because the mode was `warn`.

What it does **not** show is a fold. `distill/kl` ROSE across the window the rule fired on (0.0446
→ 0.0652), so this toy student was not absorbing its toy teacher at all — which makes the
plateau half trivially satisfied. The run is therefore a WIRING check (every series exists, carries
the right value, the state machine transitions where the arithmetic says it must, and the state
survives to disk), not a reading on whether the rule fires at the right MOMENT of a real fold.
**Sizing the window is the three-dose cell's job, and the default stays `off` until it has.**

### Advantage-gated / action-form distillation (`--distill-target` / `--distill-topk` / `--distill-gate` / `--distill-gate-tau` / `--distill-beta`) + the rank tripwire (`--rank-tripwire`)

`gen3_distill_target_gate_v1` (config **v103**; `designs/ai_v10/design_advantage_gated_distillation.md`
§3.1/§3.3/§4.1, the §7.1 v1 scope) — the five-arm record convicted the full-distribution KL's
*content*, and the target FORM is the one axis no arm ever manipulated. All seven knobs are the
`td_aux_coef` provenance genre (argparse `None` → `_resolve` → recorded on `ModelVersion`, never
gated); **every default is byte-identical to today** (SHA256-verified over seeded tiny runs).

- **`--distill-target {kl,action}`** (default `kl` = the untouched `_distill_loss` call). `action`
  dispatches the policy term to **`_gated_action_distill_loss`**: the teacher's **top-K**
  probabilities renormalized over the legal set (`--distill-topk`, default 1 = pure argmax CE — one
  bit of ordering, no tail shape; `K ≥ n_actions` reproduces the KL to fp tolerance — the §7.3
  identity that makes the new path a superset, unit-pinned), AWR-weighted
  `w = clamp(exp(|Â|/--distill-beta), 20)` with `Â` the NORMALIZED minibatch advantage (the clip
  objective's own tensor; with `Â ≡ 0` the weighted mean IS the old masked mean, the other §7.3
  identity). K=1 with the weight reproduces `_searchteacher_loss`'s CE (also pinned). Requires
  `--distill-coef > 0`.
- **`--distill-gate {none,advantage}`** (default `none` = every on-pin row, exactly the KL's rows —
  arm G1). `advantage` keeps a row only when **the teacher's argmax disagrees with the SAMPLED
  action AND `Â(s,a) < -τ`** (`--distill-gate-tau`, normalized-advantage units): on such a row PPO
  pushes probability off `a` and the CE toward `a_T ≠ a` pushes the same way — objective agreement
  by construction, from the same number. An empty gate returns `None`, never a NaN (pinned
  end-to-end: τ=1e9 is byte-identical to no distillation). Requires `--distill-target action`.
  §4.3 liveness under `distill/`: `gated_frac` · `n_gated` (**0 is a reading**, the gate found
  nothing) · `gate_agree_rate` (student argmax == teacher argmax ON GATED ROWS) ·
  `mean_gate_adv` — read beside `grad/distill_share`, the §6.2 dose meter (G2's coefficient is set
  by SHARE, never by eye: the gated row count is ~10–20× smaller, so a healthy G2 at an unmatched
  dose is uninterpretable).
- **`--rank-tripwire {off,warn,abort}`** (default **warn**) + **`--rank-tripwire-drop`** (default
  0.20) — §4.1 verbatim, `agents/training/rank_tripwire.py` (`RankTripwireCallback`, registered in
  `main/train/callbacks.py` unless `off`). **No fold runs blind again**: the five failed arms'
  `rank/policy_pr` collapse (21.87 → 12.5–13.6, 38–43%) was on an instrument already running and
  read five days late. The callback re-reads the EXISTING `rank/policy_pr` scalar out of
  `logger.name_to_value` at each rollout boundary (no new probe, no forward): EMA (half-life 10
  train() calls) vs the run's own baseline (median of readings [5, 25), logged as
  `rank/policy_pr_baseline`); **WARN** at `ema < (1−drop/2)·base` ×3 consecutive (launcher event
  via `main.launcher.ipc.emit` + `rank/policy_pr_ratio`); **TRIP** at `< (1−drop)·base` ×3 (loud
  event + `rank/tripwire_fired = 1` latched; under `abort` the callback returns False from
  `_on_step`, so `learn()` stops cleanly and the normal final save runs). A missing reading is
  **"no reading"** — `rank/tripwire_no_reading`, counters frozen (not reset), never a trip and
  never an all-clear. 20% is calibrated: fires on all five known-bad arms, on no known-good
  control; 20–38% is the margin. Pure diagnostic — no loss, no grad; `abort` changes *when*
  training ends, never what a step computes. State machine pinned in `rank_tripwire_test.py`;
  provenance in `agents/model/distill_target_gate_provenance_test.py`.

Tests: `instrumented_ppo_test.py::test_distill_*` (policy KL: identical→0, masking, illegal-mask, None-guard,
grad-student-only, reuse-bit-identical, multi-teacher averaging) + `::test_value_distill_*` (equal→0, masking,
None-guard, PopArt-frame scaling, grad-student-only) + `::test_value_feat_distill_*` (aligned→0 + scale-free,
masking→cosine-distance, None-guards, grad-student-only).

## Search-as-teacher (`--search-teacher`, `teacher/` package)

Selective **Expert Iteration** — the offline-teacher plateau-breaker (design:
`designs/ai_v6/design_search_teacher.md`). Each cycle, **search + rollout-confirm the worst loss
craters** of recent eval traces and distil the VERIFIED-better action into the policy via an
**advantage-weighted CE aux loss (AWR)**. Off by default (`--search-teacher` absent / coef 0 ⇒
byte-identical). The "expert" is the prober's `better_line` beam + the rollout-confirm tiers
(`src/main/prober/`); this wires them into training. Package `src/agents/training/teacher/`:

- **`selection.py`** (`select_candidates`, Phase 0, model-free) — the two-stage funnel:
  `ProbeSession.scan` ranks the worst-ΔV loss craters → `falsifier.falsify_battle` gates to *reducible
  MISTAKEs* (not aleatoric LUCK — don't teach against dice) → expand to the crater **±window** (the
  cause is usually 1–2 turns BEFORE the value crater). Ranks by |δ|, caps at the budget.
- **`opponent_resolver.py`** (`resolve_opponent`) — the EXACT opponent: a `sentinel_<i>` trace → its
  `models/<run>/snapshots/snapshot_<step>.zip` (the positional index→step map is in
  `metadata.json:latest_eval.pool.sentinels[i].snapshot`, valid only for the latest cycle — which is
  what the teacher runs on); a bot → reproducible from its name; anything else → **`'unresolved'` →
  SKIPPED, never approximated** (distilling "A* beats a proxy" is a soundness failure, not a degrade).
- **`produce.py`** (`produce_correction`) — the 3-tier strictly-better gate: SEARCH (`session.better_line`
  with `interior_opponent='ckpt'`, the exact opp) → CONFIRM (rollout-to-end vs the same exact opp,
  Wilson CI) → GATE (keep only if the Wilson LOWER bound beats the played loss rate). Distils the
  **CONFIRMED** win-rate improvement (`confirmed − played`), never the critic's optimistic backed-up
  value (the Spore 95%-vs-62% lesson). Staleness re-verify: if the frozen trainee already argmaxes A*,
  skip (`already_known`).
- **`buffer.py`** (`Correction`, `CorrectionBuffer`) — a bounded recency RING of corrections, sampled
  (with its own forward) on each rollout minibatch inside `train()`. **STANDALONE, not the rollout buffer** — the searched states are
  off-policy (older eval traces), so they must never enter GAE / the clip objective. Lives on
  `model._correction_buffer`.
- **`callback.py`** (`SearchTeacherCallback`) + **`src/main/search_teacher_worker.py`** — the
  non-blocking driver mirrors the eval cadence: freeze the trainee, spawn frozen-snapshot worker
  subprocesses (own POKE_LOOP, spare cores — the live trunk mutates, so a thread is unsafe; isolation
  is why eval uses subprocesses too), each runs the search + confirm over a candidate slice (ONE warm
  `SearchSession` reused → the Node spawn is amortized), publishes a shard (obs `.npz` + scalars
  `.json`); the parent polls and fills the buffer. Skip-while-running, watchdog, crash-logged.
- **SUPPLY+POOL mode (`--teacher-persistent`)** — `teacher/generate.py` +
  `src/main/search_teacher_persistent_worker.py`. The per-cycle mode reads eval traces (a trickle every
  ~2M steps); the persistent mode is a LONG-LIVED worker pool that GENERATES its own fresh losses (the
  frozen trainee vs sampled current opponents — the recent pool snapshots + bots — recorded via the
  eval forensic path `begin_forensic_cycle` + `run_local_battles`) and searches them CONTINUOUSLY,
  dripping corrections into the buffer instead of a 2M-step burst. The parent RE-FREEZES the snapshot
  every `--teacher-refresh-steps` (default 500k, written to a polled `control.json`) so long-lived
  workers track the moving policy, and `_ingest`s correction shards incrementally each `_on_step`.
  Because the worker CHOSE the opponent, the exact-opponent is KNOWN directly (no sentinel-resolution
  fragility); `falsify_gate=False` here (supply is plentiful → the CONFIRM is the gate). Never touches
  the training hot path (a frozen-snapshot side activity, like eval). Validated end-to-end: one worker
  published 8 verified-better corrections from self-generated battles in ~150 s. Flags:
  `--teacher-persistent`, `--teacher-refresh-steps`, `--teacher-gen-battles`.
  - **Lifecycle hardening** (a long-lived, multi-process pool must self-heal — an adversarial review
    surfaced these): the parent `_reap_and_respawn`s a crashed worker on a step-backoff (so a dead
    worker can't silently drain the pool to zero — `teacher/workers_alive`/`worker_respawns_total`);
    snapshot pruning keeps the latest **three** (numeric `_version_key`, not lexical — `v10 > v9`) and
    the worker re-checks `os.path.exists` before every snapshot/opponent load + wraps both in try/except
    (a pruned/corrupt file SKIPS the iteration, never crashes); `_spawn_persistent` wipes stale shards +
    `gen_*` dirs from a prior crash/restart so a fresh pool never double-ingests; `_ingest` CONSUMES
    (deletes) a shard BEFORE buffering it (a delete failure DROPS it rather than re-globbing it into a
    duplicate); the worker's per-iteration `ProbeSession` is a context manager (drops its cached models)
    and the warm `SearchSession` recycles every `recycle_every` (Node V8-heap backstop; the launcher's
    3 h restart owns the rest). **The `_correction_buffer` is `_excluded_save_params` from the SB3 save**
    — it holds a `threading.Lock` that cloudpickle can't serialize (it would crash `model.save()` at the
    pre-train roundtrip smoke for EVERY `--search-teacher` run), and it's transient scaffolding like the
    rollout buffer (re-created empty on resume; keeps checkpoints small).

**The AWR aux loss** (`InstrumentedMaskablePPO._searchteacher_loss` + the `train()` fold): `coef ·
advantage-weighted CE(π(·|s), A*)` over a minibatch sampled from `_correction_buffer` with its OWN
policy forward (`get_distribution`); weight `w = clamp(exp(advantage/β), w_clip)`. The advantage is the
CONFIRMED win-rate improvement (NOT a critic advantage — the soundness point). The shared-trunk pull
rides `grad/searchteacher_share` / `_policy_cosine` (the live "is the teacher fighting the actor"
signal). `teacher/*` metrics: `agree_rate` (π ↔ A*, should RISE), `mean_adv`, `mean_w`, `loss`, `n`,
`buffer_size`, `corrections_per_cycle`, `yield`, `mean_confirmed_dwin`.

**On-policy self-distillation (OPD) — the KL upgrade of AWR (`--opd-coef`).** AWR distils only the
single verified-better action A*; OPD upgrades the distillation TARGET to the FULL improved distribution
**π'** via `opd_coef · KL(π' ‖ π_student)` (`InstrumentedMaskablePPO._opd_loss` + its own `train()` fold,
modelled EXACTLY on the AWR fold). π' is the softmax over LEGAL actions of the beam's per-action
**backed-up** values `(v(a) − max_legal_v) / opd_beta`, with a COMPLETED-Q floor (min legal value) for a
legal-but-unsearched slot and 0 on illegal slots — built worker-side in `produce.py` (`_build_pi_target`,
only when `build_pi_target`, so no cost off) and carried on the `Correction` as a NEW `pi_target [11]`
field (appended LAST, default None → an AWR-only run is backward-compatible). It travels the worker shard
`.npz` (like obs/mask, a NaN row = None) and `CorrectionBuffer.to_tensors` stacks it (all-present → a
tensor; **any-None → the key is None** so the KL None-guards — never a partial batch). The OPD fold
samples the **SAME** `_correction_buffer` (its own `get_distribution` forward), so a Correction carries
BOTH targets and a run can **A/B AWR vs KL** by which coef is set. `opd/*` metrics: `kl` (should FALL),
`agree_rate` (student ↔ π' mode, should RISE), `pi_target_entropy` (π' sharpness), `n`; the shared-trunk
pull rides `grad/opd_share` / `_policy_cosine`. **Training-only** (0 = byte-identical, NOT in
ModelVersion / `check_compatible` / any `check_*` → both A/B arms resume a pre-OPD checkpoint with zero
FATAL risk; coefs `_resolve`-inherited on a flagless resume). **Requires `--search-teacher`** (it fills
the buffer + its workers build π'; a `parser.error` guards `--opd-coef>0` without it).

**Why NOT value-only:** the search VALUE is the *improved-policy* value V^π*(s); regressing the PPO
critic (which must predict V^π for GAE) toward it biases advantages. So the signal is the **policy**
(AWR); the off-policy value term is wired but `--search-teacher-value-coef 0` by default (the
joint-ExIt A/B). **All training-only** (no `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` bump; coefs
`_resolve`-inherited on a flagless resume, operational knobs forwarded by the launcher). **Honesty
gate:** the search *finding* a better line ≠ it *helping* — validate `eval/td_resid_tail` /
calibration / ELO on a `coef=0` A/B. (The old "~⅔ of grind losses are matchup-lost / UNCOACHABLE"
caveat is **RETRACTED** — model-judged recoverability is circular; treat those losses as headroom.)

| Flag | default | role |
|---|---|---|
| `--search-teacher` | off | master enable (constructs the callback + buffer) |
| `--search-teacher-coef` | `0.0` | AWR policy CE weight (0 = byte-identical) |
| `--search-teacher-value-coef` | `0.0` | off-policy value term (OFF — soundness) |
| `--search-teacher-beta` | `1.0` | AWR temperature β |
| `--opd-coef` | `0.0` | OPD KL(π' ‖ π_student) weight (0 = byte-identical; requires `--search-teacher`) |
| `--opd-beta` | `1.0` | OPD softmax temperature β for π' |
| `--teacher-search-budget` | `200` | candidates searched per cycle |
| `--teacher-confirm-rollouts` | `8` | Monte-Carlo confirm games (the CI gate) |
| `--teacher-search-workers` | `3` | worker subprocesses per cycle |
| `--teacher-search-freq` | `0` | steps between cycles (0 = eval freq) |

**Sim engine (`impl`, no flag of its own).** Every child the teacher spawns — the generation
battles (`teacher/generate.py` → `run_local_battles`), the searches (`SearchSession`) and the
replay/re-roll driver (`ProbeSession`) — takes its engine from `SearchTeacherCallback(impl=…)`,
which `train_rl_agent` sources from the existing **`args.bridge_impl`** (so there is no new
user-facing flag; `"node"` when `--use-bridge` is off, which is the historical behavior). It rides
each worker's config JSON as an `"impl"` key. This closed a real silent gap:
`teacher/generate.py`'s `run_local_battles` call had **no** `impl=`, so on a `--use-bridge=rust`
run its battles would have been generated on node regardless.

**`--use-bridge=rust` + `--search-teacher` now RUNS** — the old hard `parser.error` is deleted
(`gen3_rust_search_driver_v1` / `gen3_rust_replay_driver_v1`: one `search_driver` binary serves both
offline verb families). Each LEG is gated on rust — `better_line` node≡rust candidate V (an
obs-level bit-identity claim), `search_clone_parity` (clone ≡ `reroll_many` at the obs), and the
counterfactual confirm leg — but the COMPOSITION is not: **no full multi-cycle teacher run has been
done end-to-end on rust.** Treat the first one as an experiment and fall back to `--use-bridge=node`
if a cycle misbehaves. That guard's OLD stated reason — the record's `input_log` being
replay-equivalent rather than byte-identical — was **wrong and is retracted**: no consumer reads the
committed-choice lines, so do not re-derive a plan from it. See `src/utils/bridge/README.md` →
*Offline driver transport* for the seam and the full gate table.

**Tests** (`src/agents/training/teacher/*_test.py`): `buffer_test` (ring/sample/stack), `awr_loss_test`
(AWR math, masking, grad), `opponent_resolver_test` (bot/sentinel/unresolved, tmp metadata),
`produce_test` (the 3-tier gate with a fake session), `selection_test` (the funnel with a fake
ProbeSession + monkeypatched falsify), `callback_test` (shard→buffer collect + crash-graceful); plus
`instrumented_ppo_test.py::test_search_teacher_*` (the AWR fold in a real `train()` moves the policy
toward A*; off-by-default no-op). **OPD tests:** `instrumented_ppo_test.py::test_opd_*` (the `_opd_loss`
KL — 0 at the fixed point / >0 otherwise / None-guards / illegal-action masking — plus the real-`train()`
fold moving the policy toward π', off-byte-identical even with a populated buffer, and the AWR-only
π'-less buffer being skipped), `teacher/buffer_test` (`pi_target` roundtrip: all-present → tensor,
any-None → None), `teacher/produce_test::test_pi_target_*` (π' sums to 1 over legal / 0 illegal / peaks
A* / temperature flattens / completed-Q floor). End-to-end pipeline (selection → exact-opp search →
confirm → gate → Correction) validated against a real run.

### The WIN-PROB ONE-PLY teacher (`--search-teacher-mode winprob_oneply`, ai_v12 routes 2+3)

**`--search-teacher-mode` defaults to `crater` — everything above — and that default is
byte-identical to the behaviour before the flag existed. Nothing has run `winprob_oneply`; no arm is
registered.** Design:
[`designs/ai_v12/design_winprob_behavior_coupling.md`](../../../designs/ai_v12/design_winprob_behavior_coupling.md).

A new **SUPPLY** of corrections on this exact seam, not a new pipeline. It produces the SAME
`Correction` record, so the shard format, `CorrectionBuffer`, `_searchteacher_loss` and
`--search-teacher-coef` are all untouched and cannot tell the two modes apart. Only the SELECTION and
PRODUCTION halves are swapped, and the dispatch lives in ONE place (`teacher/modes.py`) because there
are three call sites and a mode string validated in three places will eventually mean three things.

| | `crater` (default) | `winprob_oneply` |
|---|---|---|
| asks | *where did the model lose the most value, and is there a strictly better LINE?* | *at a decision the head calls CONTESTED, does a one-ply read prefer another action by a margin that survives confirmation?* |
| selection | `select_candidates` — value craters, falsify-gated to reducible mistakes, ±window | `select_winprob_candidates` — the H rule, **model-free** off the trace's recorded `win_probs` / `action_mask` |
| production | `produce_correction` — a depth-2 beam over the **critic**, Wilson-gated | `produce_winprob_correction` — one-ply **win-prob** ranking → margin floor → paired rollouts |
| battles used | LOSSES only | **every outcome** — a whiff in a won game is still a whiff, and the head's self-referential labels are exactly why it never noticed |

The pipeline, which is the design doc's "3 filters → 2 transplants" as code:

1. **CONTESTED gate** — `n_legal ≥ 2` AND `|P(win|s) − 0.5| < --winprob-teacher-band` (default
   `0.15`). **Imported from `main.search_dividend.defensive.gate`**, not re-typed: two definitions of
   "contested" that could drift apart while both looked right is a failure this tree has paid for,
   and the teacher's band IS `DefensiveConfig.wp_margin`. A decision with no recorded win-prob (NaN)
   is never contested and is **never imputed** — one we cannot judge is one we do not teach from.
2. **ONE-PLY read** — `ProbeSession.lookahead` re-rolls the turn under each legal action (opponent
   plays its RECORDED move), materializes the successor through the real encoder, reads the heads.
   We take the **win-prob** read, not V: the critic estimates shaped return in PopArt units, and
   probe G measured the win-prob head beating the played action on exactly this job. A candidate with
   no win-prob read is **dropped, never scored from the critic** — a fall-back would silently run a
   different teacher under the same flag (the confusion `defensive.check_leaf` exists to prevent).
3. **MARGIN gate** — `--winprob-teacher-margin` (default `0.02`), against the **PLAYED** action, not
   the runner-up: the target exists to move probability OFF what the policy did.
4. **CONFIRMATION** — `--teacher-confirm-rollouts` (the **existing** flag, default 8) paired
   `replay_counterfactual` rollouts to a terminal for A\* and for the played action. A rollout
   contains the opponent response the one-ply leaf structurally lacks. The test is **asymmetric on
   purpose** — A\*'s Wilson LOWER bound against the played action's POINT rate — because the failure
   it catches is a flattering estimate of the challenger.

⚠️ **STEP 4 IS A REQUIREMENT, NOT A REFINEMENT — the WINNER'S CURSE.** Defensive-search iter 2
(`designs/research_state/measurements/defensive_search_iter2_2026-08-29.md`) un-throttled its
allocator, produced **13× more evidence-certified overrules (1.8% → 5.82%)** and landed the win rate
on **0.5003 [0.4803, 0.5203] — the point estimate IS the null**. CRN pairing removes dice noise *and*
the shared offset, so what a separation procedure certifies is the leaf's residual **differential**
bias (RMS 0.122, larger than most true gaps) as much as signal. **Statistical separation of a biased
reader is not correctness**, and unlike route 1's PBRS a distillation target has **no invariance
shield** — a wrong target simply trains the policy to be wrong. `--teacher-confirm-rollouts 0` exists
only because the design doc's **E2** needs an undisciplined control arm to demonstrate this.

The counter-evidence that keeps the mode alive: **probe K** re-judged iter 2's 3,531 overrules under
opponent-MARGINALIZED ground truth and found **+0.0474 [+0.0216, +0.0730] per decision — REAL**. The
overrules were right; the per-decision → per-episode TRANSFER failed (+4.7pp × ~2.2 overrules/game
bought +0.0003). A **training** target changes the policy everywhere the network generalizes, not
only at the 2.2 decisions per game where a searcher intervened — which is why the response to probe K
is route 2 rather than a fourth iteration of route 3 as an inference lever.

**Why `--winprob-teacher-margin` defaults to 0.02 and not 0.122.** 0.122 is the *measured* leaf-bias
RMS, and running there collapses target volume by roughly an order of magnitude before any arm has
asked whether it should. E4 is the arm that measures the volume/quality trade; E2 runs at the working
default. ⚠️ If the head's differential bias is ever fixed at source (the empowerment program's
contrastive marginalized labels), **this default and E4's whole premise need re-measuring** — they
are keyed to a bias that would no longer exist.

**What was reused from `search_dividend/` and what was not.** `defensive.gate` + `DefensiveConfig`:
imported. `defensive.verdict` / `resolve_action`: NOT — they answer "which action do I PLAY", and the
teacher answers "is this a target". `racing.Racer` and the budget/deadline machinery: NOT — they are
the *allocator*, racing arms against a wall clock inside a battle in flight, and the teacher works
offline from a recorded reconstruction with no clock to race. `playoff.PlayoffRunner`: NOT — it needs
a live `SearchEngine` and a shared `Deadline`; the confirmation goes through
`ProbeSession.replay_counterfactual`, the same offline primitive `produce_correction` already uses.
The residual duplication is the paired-margin arithmetic, a handful of lines, and it is deliberate.

**Flags** (all OPERATIONAL — re-pass on resume, like `--search-teacher` itself; not `_resolve`d, not
on `ModelVersion`, recorded in `metadata.json`'s `cli_args` like the rest of this family):
`--search-teacher-mode {crater,winprob_oneply}` (default `crater`), `--winprob-teacher-band` (0.15),
`--winprob-teacher-margin` (0.02). The confirm count is the **existing** `--teacher-confirm-rollouts`
— adding a second spelling for one number is how a flag surface rots.

**Config gates** (the only gates there are): `winprob_oneply` without `--search-teacher` is refused
(no teacher would run at all); without `--win-prob-mode read_only|shaping` it is refused (the ranking
IS the head, and falling back to the critic would run a different teacher under the same flag); the
band must be in `(0, 0.5]` and the margin in `[0, 1)`. An unknown mode string **raises** at callback
construction rather than falling back to `crater` — and a worker config with no `mode` key defaults
to `crater`, so an older parent's config still runs exactly as it did.

**Tests.** `teacher/winprob_oneply_test.py` (40): every gate as a pure function (contested / ranking /
margin / Wilson / paired confirmation, including the synthetic winner's-curse rejection and the
asymmetry of the test); the mode seam (default, unknown-mode raise, both dispatch pairs, the two
margins staying separate parameters, both workers' `crater` fall-back, callback-time validation); the
consumer contract (a winprob `Correction` runs through the real `_searchteacher_loss`); crater-path
argument identity; and all five config gates.

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
