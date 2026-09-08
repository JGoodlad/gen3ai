# CLAUDE.md — Training (`src/agents/training/`)

Callbacks, reward manager, episode/turn tracking, stall detection, and the bot-eval pipeline.
**How to launch training** (commands, flags) lives in the root `CLAUDE.md` → Training /
Launcher; this file documents the subsystems' internal design. The `TurnDelta` fold and the
LiveView/TurnView/LegalActions read-models it consumes are documented in
`src/agents/battle/CLAUDE.md`. The obs-build performance gate is in
`src/agents/observation/CLAUDE.md`.

## TensorBoard export census — every scalar, and its CURRENCY

> 🔒 **This section is PINNED to this file by `tb_relevance_test.py::test_the_census_table_carries_an_era_column`, which asserts `gen3_tb_relevance_v1` appears in
> `src/agents/training/CLAUDE.md`. Do NOT move it to `designs/training/` — that breaks the gate.
> (Learned the hard way in the 2026-09-07 doc split.)**


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

**Counts, measured 2026-09-06** — 153 static `logger.record(` sites across `src/agents/training`,
`src/main/train`, `src/main/eval_worker.py` and `src/main/elo.py`; 185 distinct tags observed
across three CPU smokes (a plain run, a `--win-prob-mode read_only` run, and a `--use-popart` run).
The two do not match and should not: one site can emit a whole dict (`f"reward/{k}"`), and many
sites are flag-gated off in any one run. **Recount before quoting** — `tmp_census.py`'s recipe is
`grep -rn "logger.record(" src/agents/training src/main/train src/main/eval_worker.py` for the
sites and an `EventAccumulator` walk of a run's `tb/` for the tags.

| group | sites | tags seen | cadence | currency | **era** (`--critic winprob`) | computed in |
|---|---:|---:|---|---|---|---|
| `reward/` | 1 | 46 | **per rollout** | RAW REWARD | 14 emitted, of which **6 NOISE (gated)** + 5 REDUNDANT — a 1-term composition has nothing to apportion | `reward_term_callback` ← `reward_term_stats` |
| `train/` | 53 | 23 | per rollout (`train()`) | MIXED — see per-tag below | LIVE, but `return_*` / `value_loss` / `explained_variance` **change currency to P(win)**; `scaffolding_*` **NOISE (gated)** | `instrumented_ppo/ppo.py`, `grad_balance`, `run_io` |
| `win_prob/` | 5 | 42 (+10 under `--critic winprob`) | per rollout | PROBABILITY | **the era's core group.** The `critic_*` ten are LIVE and primary; the 19 `contested`/`material` tags are **NOISE** on a spread-free margin and are ALL gated (see below) | `ppo.py` ← `value_terms`, `calibration`, `scaffolding.reliability_table` |
| `eval/` | 35 | — | **per EVAL CYCLE** | win rate / ELO / reward | LIVE — except **13 `mean_reward_*` REDUNDANT** (byte-identical to their `win_rate_*` twin under the indicator) | `eval_callback`, `selfplay_callback` |
| `eval_final/` | 2 | 10 | once, at run end | win rate | LIVE | `main/train/final_eval.py` |
| `signal/` | 4 | 12 | per rollout | NORMALIZED (adv) / probability (outcome) / rate (draw) | LIVE — `draw_rate` is **promoted to a PRIMARY endpoint** (the G7 kill condition) | `signal_metrics`, `signal_callback` |
| `popart/` | 5 | 5 | per rollout | raw (μ,σ) + unitless (norm) | CONDITIONAL — **structurally silent**: PopArt is REFUSED under `winprob` | `ppo.py` |
| `grad/` | (dynamic) | 16 | per rollout | unitless shares | LIVE — `win_prob_*` **NOISE (gated)**: it IS the value term, and counting it twice deflated every share | `grad_balance` |
| `rank/` | 6 | 18 | per rollout | unitless | CONDITIONAL (needs the rank probe); `tripwire_no_reading` correctly reports its own blindness | `rank_tripwire`, `rank_metrics` |
| `belief/` | 1 | 8 | per rollout | accuracy / CE | LIVE (unchanged by the critic mode) | `belief_bank` |
| `distill/` | 7 | — | per rollout | KL / MSE / rate | CONDITIONAL — silent, no teacher | `distill_terms`, `distill_anchor*`, `distill_stop_callback` |
| `cf/` | 5 | — | per rollout | probability + counts | CONDITIONAL — silent, no `--cf-records` | `cf_terms`, `cf_label_buffer` |
| `teacher/` · `opd/` | 11 | — | per cycle / rollout | CE / KL | CONDITIONAL — silent | `teacher/callback`, `ppo.py` |
| `team_pfsp/` · `hparams/` · `capacity/` · `defent/` · `baitent/` · `value_dist/` · `td_aux/` · `q_winprob/` | 20 | — | per rollout | see each section | CONDITIONAL — all silent; `value_dist/` is **REFUSED** by the mode, the rest are flag-off | their own callbacks |

### ERA RELEVANCE — a tag whose SOURCE is absent is not emitted (`gen3_tb_relevance_v1`)

**Measured 2026-09-06 over `models/ai_v12_01_winprob_critic`'s first hours: 216 tags emitted.** The
`--critic winprob` era changes no tag NAME but removes the SOURCE behind several, and the recorders
kept publishing — as flat constants and byte-identical duplicates that a reader cannot tell from a
measurement. Every classification below is from that arm's own tfevents, not from reading code.

| class | on the arm | means |
|---|---:|---|
| **LIVE** | 166 | meaningful as published |
| **NOISE** | **31** | emitted, content-free *here* — a constant, a tautology, or a copy created by the era. All 31 are now GATED |
| **REDUNDANT** | 19 | an exact duplicate of another tag by identical formula or affine invariance |
| **CONDITIONAL** | (48 more observed on other configs, + ~14 whole families) | correctly SILENT — their flag is off, or the mode refuses them |
| **DEAD** | **0** | nothing in the recorder set is unreachable; the v75 latent-belief and v88 `V_pub` purges left no orphans |

**The NOISE tags are GATED** — the gate is on the SOURCE, never on the value, so a shaped run is
byte-identical (verified: 172 tags, empty before/after diff):

| gated | why it was content-free | gate lives in |
|---|---|---|
| `train/scaffolding_{gauge,rho,n}` | `V = sigmoid(win_prob_logit)`, so ρ ≡ 1.0 and the gauge ≡ 5.5e-13. A rank gauge between a quantity and **itself** | `scaffolding._same_ordering` — identical rank vectors ⇒ no keys |
| `grad/win_prob_{share,norm_shared,policy_cosine}` | the critic loss IS the win-prob BCE — the SAME tensor object, passed as `value_term` *and* as `aux_terms["win_prob"]` | `grad_balance_metrics` skips an aux term that `is value_term` |
| `win_prob/{brier,acc}_contested` · `contested_{frac,label_mean}` · `brier_material` · `skill_vs_material` | **NO LONGER NOISE — the SOURCE was repaired 2026-09-06.** `win_margin` was identically 0.0 on the whole win-prob era (a MATERIAL-potential by-product whose compute the composition gated away), so `contested_frac` ≡ 1.0, every `*_contested` ≡ its pooled twin, `brier_material` ≡ 0.25 and `skill_vs_material` collapsed to `1 − 4·brier`. `gen3_obs_margin_unconditional_v1` computes Φ_mat unconditionally, so the margin now spreads and the six tags are LIVE — see *WHAT THE SHORT CIRCUIT MUST NEVER SKIP* below. ⚠️ A run started before that commit carries the degenerate series | `value_terms._win_prob_loss` treats a **spread-free** margin as absent — kept, as the consumer-side guard against any future flat margin |
| `reward/{bias_refund,class_refund}_{mean,abs_mean,abs_share}` | the refund is the BIAS class's accumulate-and-refund MECHANISM; with no bias term it is structurally 0.0 | `reward_term_stats._has_bias` |

Plus `eval/{win_rate,mean_reward,mean_ep_len}_vs_pool` and `eval/sentinel_monotonicity`, which fell
back to a confident **0.0** / a perfect **1.0** when no sentinel had been measured yet — the first
two cycles of the live arm published exactly that beside a `pool_snapshot_count` of 1.
`selfplay_callback` now leaves a GAP until a sentinel actually reports (the in-process values keep
their 0.0 default, because `_check_promotion` and the ELO fit consume them; only the EXPORT is gated).

🚨 **The `grad/` one was not merely cosmetic — it was a GIGO defect.** The duplicated norm sat in the
shared denominator too, so on that arm **every `grad/*_share` was deflated by the critic's own pull**:
the published norms are policy 0.4555, value 0.1380, win_prob 0.1380 (the same number), hp_type
0.0723, move_latent 0.0039, and the published `policy_share` 0.5639 is `0.4555 / 0.8077` — a
denominator carrying 0.1380 twice. Over a de-duplicated denominator the true shares are **policy
0.680 · value 0.206 · aux 0.114**. The shares summed to 1.0 throughout, which is why nothing caught it.

**The last 13 are GATED too** — `win_prob/contested_{ece,mce,rel_gap_b0…b9,rel_n}`, which come
through `calibration.contested_mask`. It used to return `None` only when the margin **key** was
missing, not when the margin had no **spread**, so a flat column made `|const| < tau` all-ones and
the family emitted as byte-identical copies of its pooled twins. It now returns `None` on a
spread-free (or empty) margin, MIRRORING `value_terms._win_prob_loss`'s own `max − min > 0`
predicate — the two consumers stratify on the same column, so a run in which one publishes the
split and the other does not would be worse than either answer alone (a source-read test pins that
the two predicates cannot drift apart). `_CalibrationAccumulator.metrics()` already returned `{}`
for an unobserved accumulator, so the whole family now disappears rather than duplicating.

**REDUNDANT (19) — kept, deliberately, and named so nobody measures them twice:**

* **`eval/mean_reward_*` ≡ `eval/win_rate_*`, all 13 of them.** At `--victory-value 1.0
  --draw-penalty 0 --terminal-indicator` the episode reward IS the win indicator, so the two families
  are byte-identical on every opponent (verified to the last bit on the live arm). They are NOT
  redundant on a `shaped` run and the TUI reads both, so neither is dropped — but quoting both as
  "two agreeing signals" would be quoting one number twice.
* `reward/class_terminal_*` ≡ `reward/win_loss_*` and `reward/total_{mean,abs_mean}` ≡ the same —
  a class rollup over ONE term, and a total over one term. `*_abs_share` is then a constant 1.0: a
  share partition of a single term.
* `train/nonbot_fraction` ≡ `train/selfplay_fraction` whenever `--stable-opponents` is unused
  (`nonbot = pool + stable`). Kept because it separates the moment a stable opponent is added, and a
  curve that appears mid-run is worse than a duplicate.

**Three tags are the same quantity in three WINDOWS and are not duplicates**: `train/return_mean`
(this rollout's returns), `rollout/ep_rew_mean` (SB3's 100-episode deque) and
`signal/outcome_win_rate` (a 200-episode window). Under the indicator all three read a win rate;
disagreement between them is a window effect, never a defect.

### What to watch on a WIN-PROB run — the 28-tag dashboard

Grouped by the question each answers, **with its currency**. Everything in the first two blocks is in
PROBABILITY units, which is the era's whole point: the critic, the returns and the head finally share
one scale. Read the run's `🎯 [CRITIC]` startup line first — a `shaped` run's `train/*` value tags are
not comparable with these.

**Is it getting stronger?** *(win rate / ELO)*
1. `eval/elo` **+ `eval/elo_ci`** — anchored Bradley-Terry, the only cross-run number. Quote the run-END `snapshot_ladder/ladder.json`, never a mid-run node (the newest BT node is systematically inflated).
2. `eval/win_rate_vs_bots` — saturates, so read it as an ALARM: a fall is a regression.
3. `eval/win_rate_vs_pool` — the promotion gate pins it near 0.50; leaving 0.50 is the news.
4. `signal/outcome_win_rate_bots` · `_pool` — the same, REALIZED, per rollout instead of per cycle.
5. `rollout/ep_rew_mean` — under the indicator this IS the training win rate (see the window note above).

**Is the critic HONEST?** *(probability — the era's central claim, and the §4.3 gate's inputs)*
6. **`win_prob/critic_resolution`** — Murphy RES, **HIGHER is better**. The G1 PRIMARY meter; `main.critic_gate` reads its bar from the committed baseline artifact.
7. `win_prob/critic_reliability` — Murphy REL, LOWER is better. Miscalibration only.
8. `win_prob/critic_brier` · `critic_skill` — the DEPLOYED value's proper score, and its skill over the base rate.
9. `win_prob/critic_uncertainty` · `critic_base_rate` — the decomposition's context; a skill score without them is unreadable.
10. `win_prob/critic_decomp_residual` — must sit at ≈0, or REL−RES+UNC is not a decomposition of that Brier.
11. `win_prob/ece` · `win_prob/mce` — the HEAD's own calibration: the count-weighted average, and the worst READABLE bin.
12. `win_prob/rel_gap_b0`…`b9` — the SHAPE of the miscalibration. A NaN renders as a hole and means an under-populated bin, never a zero error.
13. `win_prob/start_gap` — the PAIRED episode-start read. **Positive = optimistic at the opening board.**
14. `train/explained_variance` — now EV in P(win) units (the same number in either currency — see below).
15. `win_prob/coverage` — the fraction of rows carrying a label. A fall here invalidates every line above it.

**Is the OPTIMIZATION healthy?** *(unitless)*
16. `train/approx_kl` — the KL controller's input.
17. `train/learning_rate` · **`train/dose_rate`** — the step size, and the quantity that predicts collateral.
18. `train/clip_fraction` — how much of the surrogate is at the clip bound.
19. `train/grad_norm`.
20. **`grad/value_policy_logratio`** — value-vs-policy pull on the shared trunk, aux-independent. Now that the BCE IS the critic, this is the `--vf-coef` gauge.
21. `grad/policy_share` · `value_share` · `aux_share` — the trunk partition (corrected this pass; see the GIGO note).
22. `train/noise_scale_ratio_policy` — read BESIDE `train/noise_scale_ratio`, never alone.
23. `signal/adv_raw_mean` **as a ratio to** `signal/adv_raw_std` — a systematically mis-centred critic; the no-harm watch.

**The G7 KILL condition, and the GIGO guards** *(the two that must read an exact number)*
24. **`rollout/ep_len_mean`** — a PRIMARY endpoint, not a monitored one. A `[0,1]` critic cannot represent "a timeout is worse than a loss", so stalling is this era's registered failure mode.
25. **`signal/draw_rate`** — the same failure as a rate.
26. **`reward/untracked_abs_mean` must read exactly 0.0.** Anything else means the startup composition census and the reward folds disagree.
27. **`train/return_abs_max` must read exactly 1.0.** A departure means the reward stream is not the win indicator the mode's `V = P(win)` identity rests on.
28. `time/fps` — the run is still moving.

**What to read when the contested split is ABSENT:** on a run whose margin has no spread the whole
`contested_*` family is now gated off rather than duplicated, so the contested-vs-blowout question
is answered offline by `python -m main.scaffolding_gauge --reliability --reliability-reweight`
(which stratifies bot vs pool sentinel, where the head's bias FLIPS SIGN) and by
`main.critic_gate`. ⚠️ A run started before `gen3_obs_margin_unconditional_v1` carries the
degenerate margin AND, if it also predates the gate, the duplicate tags — read them as absent.

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
the startup line. Under the `shaped`-critic default composition that is 10 terms (1 terminal +
7 PBRS + 1 bias + the refund mechanism) → 46 tags; under `--no-all-shaping-pbrs` it is 28 terms →
~100 tags. ⚠️ **The production run is neither**: `--no-hand-shaping --terminal-indicator` is ONE
terminal term, so it emits the smallest tag set of the three. Bounded
by the REGISTRY, never per-team.

**Transport: an `env_method` PULL, not an info-dict thread.** The reward is computed in the env
WORKER, and under `--async-rollout` the callback's step locals arrive wave-batched with no way to
recover which buffer row a step landed on — the same reason `TeamWinRateCallback` uses this seam.
`AsyncSubprocVecEnv.env_method` is drain-safe, so one seam covers both collectors, and
`RewardTermAccumulator.drain()` zeroes the window so a double pull cannot double-count. ALWAYS ON,
no flag: the accumulator folds only the ACTIVE terms — 9 of 35 under the `shaped` default, and
**1 of 35 under the production composition** (`--no-hand-shaping --terminal-indicator`: 1 TERMINAL,
0 PBRS, 0 BIAS). Read `metadata.json`'s `reward_composition`, never a remembered count.

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
| `scaffolding_gauge` · `scaffolding_rho` · `scaffolding_n` | the shaped critic vs the win-prob head — **DEGENERATE under `--critic winprob`**, where the two readouts are one head | unitless (rank) |
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
| `ppo.py` | `InstrumentedMaskablePPO` + `train()` — the vendored upstream override and **the whole fold sequence**, plus `train_step_source()` (below) |
| `train_setup.py` | **the pre-loop half of `train()`** — `_align_opp_intent_labels`, `_resolve_fold_flags` (→ `FoldFlags`: which terms this call folds, plus the counterfactual buffer's one per-rollout poll) and `_train_probe_setup` (→ `ProbeSetup`: the once-per-call diagnostics, PopArt's advance, the two gradient samplers). Both containers are `NamedTuple`s carrying the SAME names the fold uses, so `train()` unpacks them back into the locals the loop was written against |
| `metrics_export.py` | **the ~400-line `self.logger.record` tail** — diagnostics, no gradient, one method per TB prefix group, each taking the accumulators this call filled |
| `rollout_probes.py` | `collect_rollouts`, the entropy-boost schedule (`_annealed_entropy_boost` and its two accessors) and `_winprob_start_metrics` — per-ROLLOUT work that is not part of the fold at all |
| `hparams.py` | every after-construction knob `train_rl_agent` sets (`value_tail_weight`, the belief/intent/cf coefficients, `grad_accum_steps`, …) with the rationale comment each carries, plus `_excluded_save_params` |
| `noise_scale.py` | the McCandlish gradient-noise-scale estimator + the rate-limited NSR advisor + `noise_ratio_sample`, the read seam `--adaptive-batch` steers by |
| `noise_scale_terms.py` | the PER-LOSS-TERM half of it — is the total reading the POLICY gradient's, or the dense aux heads'? |
| `distill_terms.py` | search-teacher AWR · OPD · the exploiter-distillation family (policy KL — or the top-K/action-CE form with the advantage gate, `_gated_action_distill_loss` — value MSE, the FitNets hint) |
| `value_terms.py` | the win-prob BCE · the value-dist HL-Gauss CE · `_value_loss_from_se` |
| `aux_terms.py` | the `belief_bank` / `td_aux` / `cf_terms` delegates |
| `constants.py` | `_VALUE_TAIL_FRAC` · `_WIN_CONTESTED_TAU` · `_NOISE_SCALE_EMA_DECAY` |

**THE FOLD SEQUENCE is deliberately NOT split**, and the reason is the contract below. `train()` is
~1,220 lines in one module — of which the epoch loop is ~1,020 — because the ORDER the terms are
folded in is straight-line source order, and that is only checkable by reading while it stays one
straight line. What DID move out is everything AROUND the sequence: the pre-loop setup
(`train_setup`) and the metrics export (`metrics_export`), neither of which folds a term, plus the
per-rollout probes (`rollout_probes`), which `train()` does not call at all. `ppo.py` is **1,331
lines** — its floor with the loop intact is ~1,200, so the file-size ratchet's 1,000-line TARGET is
unreachable here without splitting the sequence, which is the thing that must not happen.

🚨 **A SOURCE-LEVEL PIN THAT SAYS "in `train()`" SHOULD READ `ppo.train_step_source()`** — `train()`
concatenated with the three setup methods and the seven export methods. The fold, its setup and its
export are ONE train step; which of the three modules a given line sits in is a decomposition
detail, and a pin that depends on it breaks on a move that changed nothing. Five test files read it
(`instrumented_ppo_winprob_critic_test`, `vf_coef_scale_readout_test`,
`instrumented_ppo_noise_scale_terms_test`, `winprob_pbrs_test`, and the hub contract's own
docstring). **The fold's own ORDERING pins stay on `inspect.getsource(train)`**, where
straight-line source order is the thing being checked. The same rule holds for a MONKEYPATCH: a
test that stubs a diagnostic must patch the module that now owns the call
(`train_setup.advantage_density_metrics` / `train_setup.shared_trunk_parameters` /
`metrics_export.live_gauge_metrics`), because patching `ppo` would silently stub nothing and the
byte-identity test would then compare two identical arms and pass. Per minibatch:

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

The upstream-drift hash check (`_verify_upstream_unchanged` + `_EXPECTED_UPSTREAM_TRAIN_HASH`)
stays in the HUB on purpose: `instrumented_ppo_test` patches that global on the module object it
imports, so moving it into a submodule would have left the patch reaching a different global than
the function reads — a test that still passes, for the wrong reason.

## Reward redesign — registry + PBRS + the no-progress clock (`reward_manager.py`, `progress_clock.py`)

> **Where the reward lives — six modules, one import path.** `reward_manager.py` re-exports every
> public name the other five declare, so `from agents.training.reward_manager import RewardConfig`
> (or `SE_SWITCH_BONUS`, or `reward_class_composition`) resolves exactly as it always did. What is
> re-exported is stated at each hub in that file.
>
> | Module | Holds | Lines |
> |---|---|---|
> | `reward_manager.py` | `Gen3RewardManager`: its state + lifecycle, the CURRENT-BOARD accessors, the class-level applications (`_apply_progress_clock` / `_apply_bias_drops` / `_apply_pbrs_suppression` / `_fold_bias_refund`) and **`process_turn_reward` — the fold SEQUENCE, deliberately not split** | 808 |
> | `reward_bias_terms.py` | `RewardBiasTerms` — every `_compute_*` producing one additive BIAS field. A MIXIN (it reads the manager's cross-turn state) | 533 |
> | `reward_config.py` | The DECLARATIONS: `RewardClass`, `RewardConfig`, `RewardBreakdown` (+ `_REGISTRY`, the reward's source of truth) and `SWITCH_BIAS_DROP_FAMILY` | 445 |
> | `reward_potentials.py` | `RewardPotentials` — the Φ potentials, `_pbrs_step`, `_hand_pbrs_on` and the eight `_fold_*_pbrs`. Also a MIXIN | 357 |
> | `reward_composition.py` | The stateless, config-duck-typed per-class CENSUS + its one-line render | 253 |
> | `reward_weights.py` | Every tunable MAGNITUDE — weights, bonuses, thresholds, clamps | 174 |
>
> Plus `reward_verify.py` (the `GEN3AI_REWARD_VERIFY=1` shadow twin) and `progress_clock.py` (the
> no-progress clock the reward READS). Changing a value in `reward_weights.py` is a RETRAIN-class
> change, not a knob.
>
> 🚨 **THE SEQUENCE IS NOT SPLIT, AND THAT IS THE DESIGN** (the rule `instrumented_ppo/ppo.py`
> keeps for its minibatch fold, `ccd08003`). The per-term math moved out on 2026-09-07 — 1,990
> lines, ten short of the size gate's hard bound, into the table above with the reward sequence
> **byte-identical** (2,802 decisions × 39 fields × 6 compositions, sha256
> `9463dc24…`). The ORDER `process_turn_reward` folds those terms in is a CONTRACT and stays one
> straight line there.
>
> 🚨 **A PATCH TARGET FOLLOWS THE SYMBOL.** `_encode_incoming_block` is read in
> `reward_potentials`, not `reward_manager`; a stub naming the old module would stub NOTHING and
> the test would assert about the real code path. `src/test_stub_vacuity_gate_test.py` fails that
> rather than letting it pass — do not silence it with a re-export that exists only to keep a
> stale target alive.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/reward.md`](../../../designs/training/reward.md).**

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

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/reward.md`](../../../designs/training/reward.md).**

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

## THE BASELINE REGISTRY (`baselines.py` · `designs/baselines.json` · `python -m main.baselines`)

**A baseline is the thing a result is read AGAINST, and this module is the ONE accessor over the
named set.** `gen3_baselines_registry_v1` (2026-09-06) — before it, "production" was a hand-copied
JSON nothing consumes at launch, THIS package's untaught meter kept its fixed opponent as a string
literal (`DEFAULT_OPPONENT = "ai_v9_29_rev1_0823/snapshots/…"`), the famine comparator and its floor
lived in one ledger entry, and the curated TensorBoard set was decided by asking. Torch-free and
offline: it reads JSON, and only `resolve()` touches `models/`.

**Every entry is EXPLICIT** — a `.zip`, a `.json`, or an `@step`, never a bare run directory — so
`gen3_last_snapshot_resolution_v1`'s last-snapshot rule cannot move what a name points at while its
run keeps training. `resolve()` therefore always lands on the `explicit_zip` / `explicit_step` rung,
and that is asserted rather than assumed.

```python
from agents.training import baselines
baselines.get("v9_fold_parent").spec        # "ai_v9_59_R2ACTION_0827/final_model.zip"
baselines.resolve("v9_fold_parent")         # through fixed_opponent_pool.resolve_model_ref
baselines.describe("famine_comparator")     # the line every consumer prints
baselines.get("famine_comparator").floor_elo   # 38.0 — the bar travels with its comparator
baselines.protected_files()                 # {run: [rel path]} — the grooming keep-list
```

**Consumers in this package and its CLIs, all accepting a NAME wherever they accept a ref:**
`main.untaught_meter`'s `--opponent` / `--config` (their literals are GONE — the engine exposes
`default_opponent()` / `default_config()` and `resolve_ref` expands a name), `--baseline` and
`--control` through the same path; `main.critic_gate --parent` and its new
`--famine-comparator` (default the `famine_comparator` baseline, whose `floor_elo` is the kill
floor — and **an absent DEFAULT comparator is recorded as NOT READ rather than refusing the whole
read**, since `models/` is not committed and one endpoint of five must not take the other four down;
an explicit one still refuses); `main.elo`'s positional run dir; `main.tb_curate`, which unions the registry's `tb_curated`
list into the curated logdir. **Each prints `baseline <name> = <run>@<step> (set <date>, <ledger
title>)`** — a reader must never have to recognise a path.

🚨 **A NEW OPPONENT IS A RE-MEASUREMENT, NOT A RENAME.** Untaught-meter levels are not comparable
across opponents, so re-pointing `untaught_meter_opponent` invalidates every banked level measured
against the old one. That is exactly why it is a registry entry with a `set_by` ledger title rather
than a constant somebody can edit: `python -m main.baselines set <name> <run>/<file>.zip --reason
"<ledger entry title>"` rewrites the entry with a freshly computed sha/commit/version and PRINTS the
ledger line to append. It never edits the ledger — append-only, and the WHY is the one field no tool
can author.

**Validation is a test in the routine suite** (`src/main/baselines_test.py`, unmarked): every named
file exists, every sha matches, `config_version` / `arch_signature` are re-read from the run's own
`model_config.json`, and the `production` entry's declared CONSTRUCTION matches
`designs/production_config.json`. Archive-backed checks skip through `main_models_dir()`; the
structural half runs everywhere. `designs/research_state/measurements/archive_grooming_tiers.py`
reads `protected_files()` so a registry-named checkpoint survives every retention tier.

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

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/eval_and_rating.md`](../../../designs/training/eval_and_rating.md).**

## Self-play opponents (`--self-play`, gated behind pathology hunting)

When `--self-play` is set, `SelfPlayCallback` replaces `PerOpponentEvalCallback` and the
training opponents become frozen snapshots of the agent itself, drawn from a directory-backed
`SnapshotPool` (`snapshot_pool.py`; state reconstructed from `<run_dir>/snapshots/` on every
restart — no manifest). Design lives in `designs/ai_v5/`. Key behaviors:

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md).**

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

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md).**

## Exploiter mode (`--exploiter`, `MaskableAgentWrapper._exploiter_player`)

A clean opponent-mix front-end for the league **exploiter** role: train a dedicated agent against
ONE fixed foreign model as the **sole opponent every episode** — to surface (and then patch, by
folding the exploiter back as a stable opponent / pool member) the non-robustness a *self-play* Nash
can't see. It needs **no `--self-play` / `--stable-opponents` / share fiddling** — point `--exploiter`
at the target and it's the only opponent.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/exploiter_and_distillation.md`](../../../designs/training/exploiter_and_distillation.md).**

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

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/eval_and_rating.md`](../../../designs/training/eval_and_rating.md)
and [`designs/training/self_play_and_pool.md`](../../../designs/training/self_play_and_pool.md).**

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

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/compile_flags.md`](../../../designs/training/compile_flags.md).**

## Compiled GPU trainer (`--compile-trainer`, DEFAULT ON for cuda)

`torch.compile`s the LEARNER's feature extractor — the CUDA forward **and backward** the PPO step
runs. The other half of the pair above, and the larger of the two.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/compile_flags.md`](../../../designs/training/compile_flags.md).**

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

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/telemetry_scalars.md`](../../../designs/training/telemetry_scalars.md).**

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

### THE ACTOR-ONLY FROZEN POTENTIAL — `--win-prob-pbrs-frozen` (`gen3_frozen_phi_actor_only_v1`)

**The value loss is the ONE thing this flag does not touch, and that is the whole construction.**
`agents/training/frozen_phi.py` reads φ = σ(logit) from a FROZEN checkpoint's win-prob head and adds
`γφ(s′) − φ(s)` to the stream that feeds the POLICY's advantages — writing **only**
`rollout_buffer.advantages`. `rewards` and `returns` are restored to what the collector produced (by
ASSIGNMENT from a snapshot; `(a+b)−b` is not `a` in float32), so the critic keeps regressing the
unshaped terminal indicator and every consumer of the value target — the scalar-MSE diagnostic
`train/value_loss`, `train/explained_variance`, `value_scale_metrics` — reads the UNSHAPED return.
Under this critic the real value loss is the head's BCE against `win_target`, which never reads
`rewards` at all, so `V ≡ P(win|s)` is preserved **bit-for-bit** with or without the flag.

**A potential added to the REWARD could not be**, and that is why the rung was held rather than
because of any doubt about the invariance: the shaped return telescopes to `1{win} − φ(s)`, negative
wherever the frozen head was optimistic about a lost game, and a sigmoid cannot output a negative
number — so the critic would be fitted to a target outside its own range and the identity the search
leaf, the calibration gate and `--vf-coef`'s meaning all rest on would be false by a known function.
Restricting the term to the advantage keeps the Ng shield (φ is FIXED, so the theorem holds
**exactly**) and, at λ = 1, makes the shaped advantage the unshaped one minus `−coef·φ(s)` — a
state-dependent BASELINE, i.e. zero-bias for a policy gradient. `φ(terminal) := 0` both makes the sum
telescope and stops the potential leaking the outcome the terminal state just revealed; the
conventions come from `winprob_pbrs.successor_potential`, IMPORTED, so the two shaping paths cannot
drift on the one convention the theorem rests on. **The coefficient is exactly 1.0, DERIVED** (φ is
already one unit of V per unit of V) and PRINTED at startup, never a knob.

**BOTH SEAMS LIVE IN `frozen_phi.py`**, in the `distill_anchor.py` shape: `shape_after_rollout` at
`collect_rollouts` — the ONE point both rollout loops pass through, so the async collector is covered
by construction — and `record_metrics` in `train()`. `ppo.py` carries one call each, because it sits
AT the file-size ratchet's 2,000-line hard bound. The frozen network rides `_winprob_phi_source`, the
attribute the shaped ladder's `--win-prob-pbrs-source` already uses (the two are mutually exclusive
by refusal), so `phi_model` and `_excluded_save_params` need no second name and the foreign weights
are never pickled into our checkpoint.

⚠️ **Read `pbrs/frozen_phi_mean` FIRST, and it must be FLAT** — φ is a fixed function of state, so a
mean that wanders like a live head's means the frozen source is not the thing being read (the
runbook §4.2 check: frozen `0.403 → 0.391 → 0.391` against live-φ's `0.680 → 0.347 → 0.216`).
`pbrs/frozen_phi_episode_dose` prices the shaping against one win, and
`signal/adv_shaped_minus_unshaped_mean` is the telescoping term — at λ = 1 on complete episodes
exactly `−coef ×` the φ mean, so the pair is a one-glance audit that the terminal convention holds on
real episodes. **The cost is stated with the benefit**: the frozen head's biases are now inside every
advantage, and §4.1's baseline measured this class of head starved of RESOLUTION — so a FROZEN-φ arm
that beats SPARSE has measured *this head's* separation, not dense credit in general.

Training-only (config **v110**, recorded and `_resolve`-inherited) — and the inheritance is
load-bearing here in a way it is not for a coefficient: the flag is **boolean by PRESENCE**, so
"not typed" and "off" are the same argv, and a launcher restart re-invoking the original command
would otherwise turn a FROZEN-φ arm into the SPARSE arm mid-run under the same run name. Refused
under `--critic shaped` (there φ and V are in different units and the dose is a real question — use
`--win-prob-pbrs-coef` / `--win-prob-pbrs-source`). Gate: `frozen_phi_test.py`.

### 🚨 THE 250-TURN CAP IS A TERMINAL, NOT A TRUNCATION — and it was the other way round (B6)

**THIS ENV NEVER TRUNCATES IN THE SB3 SENSE, AND THAT IS THE WHOLE FINDING.**
`PokeEnv.calc_term_trunc` (`poke_env/environment/env.py:951`) sets EITHER flag only when
`battle.finished`; it then splits a finished battle by *how* it finished — exactly one side wiped ⇒
`terminated`, anything else ⇒ `truncated`. "Anything else" is the **250-turn cap forfeit** and a
genuine **tie**. Both are OUTCOMES. Nothing here is a time limit that interrupted an episode
mid-flight, which is the one thing `TimeLimit.truncated` is supposed to mean.

MEASURED 2026-09-06 on a real bridge battle with `StallConfig.threshold` lowered to 6:
`gen3_env.action_to_order` returns a `ForfeitBattleOrder` at `turn >= threshold` (`== MAX_TURNS` in
production), Showdown answers `|win|<opponent>`, and the boundary reported `finished=True`,
`won=False`, **six mons alive a side**, `terminated=False, truncated=True`.

That flag then becomes `info["TimeLimit.truncated"]` in `DummyVecEnv`/`SubprocVecEnv`, and
`MaskablePPO.collect_rollouts` (`sb3_contrib/ppo_mask/ppo_mask.py:251-260`, mirrored by our
`async_vec_env.collect_rollouts_async:216-219`) does `rewards[idx] += gamma * V(s_last)`. Under
`--critic winprob` the terminal reward is the win indicator — **0** for a cap loss — and γ is
**1**, so the last step's target becomes `0 + 1.0·V(s_last) = V(s_last)`: **a TD error of
identically zero.** The timeout leaves the loss entirely, and a policy that stalls to the cap is
taught nothing about it — while G7's stall rate is a KILL CONDITION on that arm, i.e. the gate
would have been reading a signal the critic never received. Verified by revert: the row reads
exactly `V`.

**`agents/training/wrappers.resolve_episode_end` is the fix** — under `winprob` only, a
`trunc and not term` end is re-labelled TERMINAL. It sits at the END of
`MaskableAgentWrapper.step`, so every outcome consumer above it still sees the flags the sim
produced, and it is the ONE seam upstream of BOTH rollout loops (which is why it is here and not in
`Gen3Env.calc_term_trunc`). The mode arrives as `MaskableAgentWrapper(critic=…)` from
`env_factory`. Pinned by `winprob_truncation_test.py`, including the SB3 composition through a real
`collect_rollouts`.

**`shaped` is UNCHANGED and byte-identical, and what it does is worth stating rather than leaving
implicit:** a cap forfeit and a tie both bootstrap `0.9999·V(s_last)` on top of a terminal reward
that already paid `--draw-penalty`. Two wrongs that partly cancel — the shaped terminal
double-counts the ending while the bootstrap removes it — and re-deriving that composition is a
`shaped`-era question this change does not reopen.

**Two adjacent paths are CLEAN, checked rather than assumed.** A launcher SIGTERM
(`lifecycle.abort_training`) saves a checkpoint and `os._exit`s, so the partially-collected rollout
buffer is discarded and never reaches an update — SB3 does not persist it. The eval
reset-mid-battle forfeit lives in `PokeEnv.reset`, which returns only an observation and fills no
rollout buffer. Neither can leak a truncation into the loss.

⚠️ **CORRECTION to a comment that shipped with the mode: the cap forfeit is a LOSS, not a draw.**
`won_by` (`poke_env/battle/abstract_battle.py:1619-1626`) sets `_won = False` on `|win|<opponent>`;
`won is None` is `|tie|` ALONE. The label is 0.0 either way so nothing downstream moved, but
`signal/draw_rate` counts **ties and not timeouts** — the series that watches the cap is
`signal/stall_rate` / mean episode length, which is why those and not `draw_rate` are G7's kill
condition.

### `--vf-coef` multiplies a BCE now, and the first NON-DEGENERATE update SAYS what that is worth

`--vf-coef` means a different quantity under each critic while keeping its name: under `shaped` an
MSE on a PopArt-normalised shaped return (O(100) unnormalised, on a ±30 scale); under `winprob` the
win-prob head's **BCE against a Bernoulli outcome**, which is `ln 2 ≈ 0.693` per sample at
initialisation and falls. The 0.5 default was tuned against the first and carries no information
about the second, and the normalisation that made the two comparable is refused here.

So `calibration.announce_vf_coef_scale` prints, ONCE, two numbers that answer different questions:
the **raw BCE** (a statement about the HEAD — `ln 2` is chance, well below it means the head
already calls lopsided games) and the **ratio of the value term's shared-trunk gradient norm to
the policy term's** (the statement about the COEFFICIENT — the value norm scales linearly in
`vf_coef`, so the ratio IS the factor to divide it by, `≫1` cut / `≪1` raise).

🚨 **IT USED TO DIVIDE THE VALUE TERM BY `|policy loss|`, AND THAT DENOMINATOR IS DEGENERATE BY
CONSTRUCTION.** On epoch 1 the clipped surrogate has `ratio ≡ 1` and sits at its stationary point,
so `|policy loss| ≈ 0`. The live arm `ai_v12_01_winprob_critic` printed `value term = 0.5 × BCE
0.1330 = 0.0665 against |policy loss| 0.0004 → **165×**` on rollout 1 — and its own gradient series
reads UNREADABLE at rollout 1 (`grad/policy_norm_shared` exactly 0.0 against a value norm of 7.53),
**91×** at rollout 2 and **4.6×** by rollout 17. 165× was not a noisy estimate of 4.6×; it was a
number about the epoch, and a ratio of LOSS MAGNITUDES was the wrong quantity anyway — what
competes on a shared trunk is the gradient.

**The norms are READ from `grad_balance_metrics`, never recomputed.** That read-only
`autograd.grad(retain_graph=True)` probe already runs per-term on every `train()`, so the printed
ratio is exactly `10 ** grad/value_policy_logratio` and no second backward pass is run for a
banner. **The threshold is on the NORM, not the epoch** (`MIN_POLICY_GRAD_NORM`, 1e-6): the probe
samples ONE minibatch per `train()` and it is in epoch 1 by construction, so there is no later
epoch to wait for inside a reading — and epoch 1 is not degenerate in general, since the policy
GRADIENT at `ratio ≡ 1` is `A·∇log π ≠ 0`. It does NOT latch on an update it could not read (no
scorable label, no grad probe, or a policy norm under the floor — the next rollout tries again),
and `_vf_scale_announced` is in `_excluded_save_params`, so a launcher restart re-prints it beside
the startup `[CRITIC]` banner. It is a PRINT, not a scalar: past the first reading the right
instrument is `grad/value_policy_logratio` (0 = balanced). Reading rule: design §5.4.

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

### A FORK INHERITS ITS PARENT'S CURVES (`tb_inherit.py`, `gen3_tb_inherit_v1`)

**Bookkeeping over two facts this tree already had, not a new mechanism.** (1) TensorBoard merges
every `events.out.tfevents.*` in ONE run directory into one series per tag, ordered by step — not a
claim: `ai_v8_03_zarch_control_0718` carries **29** event files (one per launcher restart) in a
single `tb/` and renders as one curve. (2) A fork's global step CONTINUES the parent's
(`reset_num_timesteps=False`), measured on that same run — its `tb/` opens at step **148,401,356**,
precisely the `fork_step` its `lineage` block records. So the parent's curve occupies `[0, fork_step]`
and the fork's occupies `[fork_step, …]`: two halves of one line, apart only because they live in two
directories. `<fork>/tb/` now also gets a TRUNCATED copy of the parent's scalar events at steps
**≤ `fork_step`**, and the fork's TensorBoard reads from step 0.

**TRUNCATION is not optional.** The parent usually trained PAST the fork (`ai_v8_01` reached 170.6M
having been forked at 148.4M); copying its tail would draw parent-only progress inside the fork's own
step range, where it reads as the fork's.

**FORK-OF-A-FORK composes for free.** The parent's `tb/` already holds its own inherited prefix, so
reading the parent's WHOLE directory and truncating again yields grandparent `[0, parent_fork_step]`
+ parent `[parent_fork_step, fork_step]`. One rule per link.

🚨 **THE VALUE FORM IS PART OF THE COPY, AND GETTING IT WRONG IS INVISIBLE.** A TB scalar has two
on-disk spellings — the classic `simple_value` field and a rank-0 tensor tagged with the `scalars`
plugin — and **`EventFileLoader` MIGRATES the first into the second as it reads**. Every writer in
this tree emits `simple_value`, so copying what that loader returns writes the migrated form, which
`EventAccumulator` files under `tensors` rather than `scalars`: the events are in the file, the
provenance is right, the byte count is right, and **the scalars dashboard shows nothing**. Measured
while building this — the fork's prefix read back as 124 `tensors` tags and **0** `scalars` tags,
beside its own 124 `scalars`. `scalar_prefix` therefore reads with **`LegacyEventFileLoader`** (raw),
and the test writes its synthetic parent in `simple_value` form and reads back through
`EventAccumulator.Scalars` — the accessor the dashboard uses — because a test that writes and reads
the same migrated form cannot see it. Two guards fail on revert.

**IDEMPOTENCY is load-bearing.** `<fork>/tb/INHERITED_FROM.json` is the key: if it exists, no-op. A
launcher restart that still names the parent as `--model` (which happens before the fork has written
its own checkpoint) re-enters the fork path and would otherwise append a SECOND copy of the prefix —
the same series twice, which renders as a saw-tooth rather than an error.

**The seam is the LINEAGE seam.** `inherit_from_lineage(model_dir, lineage_block, enabled=…)` is
called in `main/train/model_build.py` immediately after the block is recorded and BEFORE
`_attach_run_tb_logger`. The block being non-None **is already the fork decision**
(`build_lineage` returns None on a same-run restart, via `fork_lr.is_same_run_checkpoint`), and the
parent + `fork_step` are read out of it — so the curve a fork inherits and the parent it claims
cannot disagree, and nothing here owns a second answer to "is this a fork?" (pinned by an AST test).
It **never raises**: a cosmetic convenience must not be able to kill a launch, so a failure returns a
reason and the run simply starts its chart at `fork_step`.

**SCALARS ONLY** — histograms/images/audio and every non-`scalars` plugin are skipped. Measured
2026-09-06: every value in `models/*/tb/` is a scalar, so the filter drops nothing today; it exists
so adding a histogram later cannot silently multiply the copy cost. `tb/` is 262 MB across 217 runs;
a prefix is a few hundred KB.

**`--no-tb-inherit`** opts out. Worth it for a large sibling FLEET under an UNCURATED logdir — 8
exploiters off one target then draw 8 identical prefixes in every chart (under `main.tb_curate` that
is exactly what you want). Training-runtime class: reaches no extractor, scales no loss, changes no
weight shape ⇒ no `ARCH_SIGNATURE` bump, not on `ModelVersion`, not in `check_compatible`, not in
`flag_registry`; it lands in `cli_args` and the launcher forwards it verbatim.

**Existing forks are NOT backfilled.** `python -m main.tb_inherit --list` censuses them (**137**
missing a prefix as of 2026-09-06); `--backfill <run>… | --all` writes one, **dry-run unless
`--apply`**, `--show` prints a run's provenance. ⚠️ **105 of the 137 name a DERIVED parent** — every
`lineage` block on disk was written by `main.lineage --backfill`, i.e. by REGEXING `--model` out of a
recorded shell command — and the derivation can be wrong: `ai_v8_01_zarch_film_0717` records
`role="fresh", fork_step=0` while its own `tb/` opens at step **148,401,356**, which is arithmetically
impossible for a fresh run (it was built by a hand-written `tmp/fork_zarch_v8.py` the regex cannot
read). Every row prints that flag and the census prints the caveat.

Tests: `tb_inherit_test.py` (22) — truncation at the exact boundary; the one-continuous-series claim
read through `EventAccumulator.Scalars`; the on-disk value-form preservation (both spellings) and its
revert-catcher; the non-scalar skip; provenance content + sha; the second call being a no-op and
`force` replacing rather than appending; dry-run touching nothing; fork-of-a-fork truncating the
grandparent prefix; the seam's restart/fresh/disabled/missing-parent no-ops; the census incl. the
`derived` flag; and the torch-free + no-second-fork-predicate contracts.

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
  PopArt — when it is on — normalizes the shaped returns (the only order that keeps the value loss in
  the units of the stream being optimized). ⚠️ This whole `--win-prob-pbrs-*` family is REFUSED under
  `--critic winprob`, so the path described here is a `shaped`-critic path throughout.
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
It holds `wilson_ci`, `spearman`, `cluster_bootstrap_ci` / `cluster_bootstrap_diff_ci`,
`sd_true_excess` and the `MIN_CELL_N` / `MIN_SUBCELL_N` sample-size floors below which that
spread is not reported — pure NumPy in, floats out: no labels, no battles, no checkpoints, no
filesystem, no torch, no RNG except an explicitly seeded bootstrap. That is the admission rule; a helper that
has to know what a *decision* or a *bias map* is belongs beside the instrument that owns the
concept. The floors are the one pair of CONSTANTS admitted, and for the estimator's own reason:
`cf_audit.resolution_cells` and `cf_audit_twin.twin_resolution_read` bin different things and
must refuse at the same n, and two copies of a floor is two thresholds that drift.
They were lifted out of `cf_audit.py` on 2026-09-06 (the file-size ratchet's first cut of
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

**Three modules, one instrument.** `cf_audit.py` owns the frame, the sampler, the label
schema, the bias map and the CLI; two readouts live beside it because they are the parts that
need nothing else the module knows. **`cf_audit_render.py`** takes a finished bias map and
returns markdown — no statistic, no file, and its two formatting rules are the ABSENT-vs-ZERO
ones (a checkpoint with no evidential head renders a NOTE, a flat width renders `n/a`, because
a row of zeros makes "no head" and "no uncertainty" read identically). **`cf_audit_twin.py`**
holds the twin-head paired read, `twin_resolution_read`, `shadow_read` and `attach_twin_heads`
— it takes labelled rows and, for the last one, a session. Both were extracted on 2026-09-06
(the ratchet's second cut, 1279 → 918 lines, under the 1,000 TARGET), and `cf_audit`
re-imports every name, so `from agents.training.cf_audit import render_markdown` still
resolves. **The extraction-parity golden below is the evidence for both moves** — it calls
them through those re-exports and was not regenerated.

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/cf_grounding.md`](../../../designs/training/cf_grounding.md).**

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

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/cf_grounding.md`](../../../designs/training/cf_grounding.md).**

## The STALL-TAIL HARVEST + head-repair pipeline (`main.harvest` → `winprob_finetune` → `main.harvest_meter`)

An **offline, three-stage pipeline** that manufactures win-probability labels for the population
probe O convicted, fits the win-prob head on them with the trunk frozen, and measures the result
against a battle-level holdout. It is the ai_v12 head-repair backbone and it writes nothing into
`models/` — artifacts land under `utils.paths.harvest_dir()` (repo-root `harvest/`, gitignored,
`$GEN3AI_HARVEST_DIR` overrides).

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/stall_tail_harvest.md`](../../../designs/training/stall_tail_harvest.md).**

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

**Full detail — every flag, gate, measurement and hazard — is in [`designs/training/exploiter_and_distillation.md`](../../../designs/training/exploiter_and_distillation.md).**

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
   We take the **win-prob** read, not V: under `--critic shaped` the critic estimates shaped return
   in PopArt units (under `winprob` the two are the same readout, so the choice is free), and
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
