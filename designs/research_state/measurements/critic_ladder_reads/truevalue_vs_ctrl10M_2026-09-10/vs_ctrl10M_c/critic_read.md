# CRITIC READ — `ai_v12_14_ladder_truevalue` vs `ai_v12_16_ladder_ctrl10M_c`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v5 at 2026-09-10T10:59:57. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_14_ladder_truevalue` | `step_10000032` | 628 | 0.2% | 150/150 (100.0%) | no |
| control | `ai_v12_16_ladder_ctrl10M_c` | `step_10000032` | 653 | 0.4% | 149/150 (99.3%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |
| control | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_14_ladder_truevalue` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_16_ladder_ctrl10M_c` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_14_ladder_truevalue` | **40/39/1** | 628 | 480 / 146 / 2 | 12 |
| control | `ai_v12_16_ladder_ctrl10M_c` | **40/39/2** | 653 | 480 / 168 / 5 | 12 |

**Frames:** both sides carry the SAME realized cap (40/39/1) — the frames are already equalised and nothing is subsampled

> ✅ **SYMMETRIC.** Both sides carry the same realized cap, so every conditioning row is read AS TRACED and this report is what the tool printed before quota matching existed. Residual per-opponent differences below the cap are the two runs' own outcome mixes, not a selection asymmetry — matching them would shrink both frames for no power gain.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json` — 68 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0326 | +0.0177 | **+0.0149** | [+0.0000, +0.0308] | **NOT DETECTED** (CI does not clear the floor 0.0102) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0878 | -0.0110 | **+0.0988** | [+0.0370, +0.1598] | **NOT DETECTED** (CI does not clear the floor 0.0558) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.1128 | -0.0267 | **+0.1395** | [-0.0058, +0.2896] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2782 | +0.1467 | **+0.1315** | [+0.0129, +0.2790] | **NOT DETECTED** (CI does not clear the floor 0.0642) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0000 | +0.0896 | **-0.0896** | [-0.1458, +0.0310] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | as traced (frames equal) | -0.0104 | -0.0004 | **-0.0100** | [-0.0417, +0.0201] | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.1666 | +1.2400 | **-0.0734** | [-0.3168, +0.1788] | **WITHIN FLOOR** (|delta| <= floor 0.2012) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0366 | +0.0459 | **-0.0093** | [-0.0270, +0.0073] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0127) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0326 | +0.0177 | **+0.0149** | [+0.0000, +0.0308] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0102) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0373 | +0.0596 | **-0.0223** | [-0.0461, +0.0050] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0058 | +0.0004 | **+0.0054** | [+0.0011, +0.0114] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0084) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0021 | +0.0051 | **-0.0030** | [-0.0086, +0.0025] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0178 | +0.0110 | **+0.0069** | [-0.0107, +0.0289] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0215) |
| ece | `all` | `capture-rate (gauge)` | +0.0646 | +0.0125 | **+0.0521** | [+0.0132, +0.0754] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0383) |
| ece | `bot` | `capture-rate (gauge)` | +0.0270 | +0.0549 | **-0.0279** | [-0.0537, +0.0123] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `pool` | `capture-rate (gauge)` | +0.1244 | +0.0949 | **+0.0295** | [-0.0390, +0.0985] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0770) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2134 | +0.2817 | **-0.0683** | [-0.1377, +0.0103] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2782 | +0.1467 | **+0.1315** | [+0.0129, +0.2790] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0642) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.1067 | +0.2144 | **-0.1077** | [-0.2183, +0.0140] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1448) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.1483 | +0.0621 | **+0.0862** | [+0.0429, +0.1300] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0758) |
| bias V - p_hat | `ALL` | `pop` | +0.0745 | -0.0032 | **+0.0777** | [+0.0457, +0.1105] | 4000 | **DETECTED** (CI clears the floor +0.0390) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | +0.0400 | -0.0272 | **+0.0673** | [+0.0399, +0.0954] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0696) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0943 | +0.0404 | **+0.0539** | [+0.0053, +0.1029] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0599) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0286 | -0.0220 | **+0.0505** | [+0.0123, +0.0893] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0499) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | +0.0017 | -0.0463 | **+0.0480** | [+0.0134, +0.0819] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0940) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.1588 | +0.0747 | **+0.0841** | [+0.0310, +0.1354] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0785) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0808 | +0.0077 | **+0.0731** | [+0.0332, +0.1133] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0421) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0423 | -0.0204 | **+0.0626** | [+0.0313, +0.0952] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0518) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.2038 | +0.0746 | **+0.1292** | [+0.0341, +0.2234] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0979) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.1230 | +0.0046 | **+0.1184** | [+0.0475, +0.1874] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0487) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0878 | -0.0110 | **+0.0988** | [+0.0370, +0.1598] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0558) |
| bias V - p_hat | `bot` | `raw` | +0.1204 | +0.0593 | **+0.0612** | [+0.0039, +0.1212] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0667) |
| bias V - p_hat | `bot` | `pop` | +0.0555 | -0.0122 | **+0.0677** | [+0.0283, +0.1077] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0657) |
| bias V - p_hat ⭐ | `bot` | `ipw` | +0.0220 | -0.0368 | **+0.0588** | [+0.0262, +0.0918] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0579) |
| bias V - p_hat | `pool` | `raw` | +0.1834 | +0.0655 | **+0.1179** | [+0.0538, +0.1776] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1500) |
| bias V - p_hat | `pool` | `pop` | +0.1196 | +0.0083 | **+0.1113** | [+0.0548, +0.1627] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.1049) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0930 | -0.0109 | **+0.1039** | [+0.0499, +0.1551] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0945) |
| Murphy resolution | `ALL` | `raw` | +0.0305 | +0.0285 | **+0.0020** | [-0.0122, +0.0163] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0079) |
| Murphy resolution | `ALL` | `pop` | +0.0418 | +0.0440 | **-0.0022** | [-0.0225, +0.0190] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0210) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0268 | +0.0315 | **-0.0047** | [-0.0200, +0.0110] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0091) |
| Murphy skill_score | `ALL` | `raw` | +0.0390 | +0.1250 | **-0.0860** | [-0.1722, +0.0049] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `pop` | +0.2205 | +0.2689 | **-0.0485** | [-0.1482, +0.0568] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2075 | +0.2372 | **-0.0296** | [-0.1280, +0.0764] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0496) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1523 | +0.1484 | **+0.0039** | [-0.0687, +0.0753] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0345) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2591 | +0.2724 | **-0.0133** | [-0.1163, +0.0912] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0365) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2199 | +0.2544 | **-0.0345** | [-0.1325, +0.0676] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0551) |
| Murphy reliability | `ALL` | `raw` | +0.0255 | +0.0058 | **+0.0197** | [+0.0096, +0.0310] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| Murphy reliability | `ALL` | `pop` | +0.0082 | +0.0010 | **+0.0072** | [+0.0020, +0.0136] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0077) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0031 | +0.0021 | **+0.0010** | [-0.0030, +0.0056] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0017) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.1128 | -0.0267 | **+0.1395** | [-0.0058, +0.2896] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | +0.0774 | -0.0104 | **+0.0878** | [-0.0241, +0.2039] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0886) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.1059 | +0.0273 | **+0.0786** | [-0.0329, +0.1863] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0835) |

⭐ = the REGISTERED estimand for that quantity. The others are the same statistic under a different selection correction, printed so the registered number is never the only one on the page.

### the three weightings

| name | what it corrects |
|---|---|
| `raw` | unweighted over the cf_audit draw — the estimand arm A's committed +0.3089 was computed under, kept for comparability |
| `pop` | cf_audit's stratified draw recombined at the frame's own (decile, outcome) mass — corrects the SAMPLER against the trace tree |
| `ipw` | `pop` times 1/capture_rate(opponent, outcome) from the cycle's eval_manifest — RULE OF EVIDENCE 17, correcting the loss-enriched TREE against the eval population |

The gate rows carry no weighting column because the scaffolding gauge applies the capture-rate correction itself, from the same `eval_manifest.json` rates — there is no unweighted variant of a gate row. And the anchor (label-trust) arm is a CENSUS of the bot battles it draws from and its estimand is the REPLAY DRIVER's fidelity, not a property of the eval population — capture-rate reweighting does NOT apply to it and none is applied.

## 3. CONDITIONING — does the head know WHO it is playing and WHOSE TEAM it holds?

Promoted 2026-09-09 from two committed measurements — `measurements/winprob_mixture_diagnostic_2026-09-09/` (the between-opponent SPREAD IDENTITY and the bias-on-Elo slope) and `measurements/winprob_probe_read_2026-09-09/` (the own-team leave-one-battle-out win-rate target, battle-grouped folds, HT reweighting) — and computed here by `main.ops.conditioning_meters`, which both measurement directories can import.

> 🚨 **The spread ratio's target is 1.0.** For ANY calibrated critic `E[V | opponent] = E[y | opponent]` exactly, so the between-opponent spread of `V` must EQUAL the between-opponent spread of the outcome. A head emitting one marginal win probability regardless of opponent reads near ZERO. The identity needs no strength axis, which is why it is the primary row and the Elo slope the secondary one.

> 🚨 **A SPREAD RATIO CAN SIT BELOW ITS OWN INTERVAL, for TWO independent reasons, and neither is a defect in the bootstrap.** (1) The noise-corrected ratio is **CLAMPED** — each side's sampling variance is subtracted and a negative result floored at zero, a biased non-monotone operator — so a point estimate of 0.000 routinely carries a CI like [0.21, 0.53] (`winprob_head_refit_2026-09-09` §12 hazard 1). (2) A **resampled between-group variance is UPWARD BIASED**, so even the unclamped `ratio_raw`'s draws can centre above its point (the mixture diagnostic reports the same shape on its η² rows). **THE INTERVAL IS THE READ, and a 0.000 is not "no spread".** The unclamped, uncorrected `ratio_raw` is printed beside the clamped one as the monotone companion — never as a substitute. Both effects hit the arm and the control alike, so the DELTA is the quantity least disturbed by either; the per-run points below are the ones to read with the caveat in hand.

**arm — `ai_v12_14_ladder_truevalue` @ `step_10000032`**: 19640 states / 626 battles / 12 opponents / 201 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 2 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

**control — `ai_v12_16_ladder_ctrl10M_c` @ `step_10000032`**: 20617 states / 648 battles / 12 opponents / 198 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 5 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0000 | +0.0896 | **-0.0896** | [-0.1458, +0.0310] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | as traced (frames equal) | +0.0632 | +0.1154 | **-0.0523** | [-0.0914, +0.0361] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1016 | -0.1277 | **+0.0261** | [-0.0091, +0.0642] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0391) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.3209 | +0.5072 | **-0.1863** | [-0.3587, -0.0154] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3898) |
| the same ratio UNCORRECTED and unclamped | `all states` | as traced (frames equal) | +0.3235 | +0.5046 | **-0.1810** | [-0.3416, -0.0240] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0690 | -0.0691 | **+0.0002** | [-0.0402, +0.0377] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0480) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0212 | +0.0224 | **-0.0012** | [-0.0119, +0.0096] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | as traced (frames equal) | -0.0104 | -0.0004 | **-0.0100** | [-0.0417, +0.0201] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | as traced (frames equal) | +0.0653 | +0.0722 | **-0.0069** | [-0.0759, +0.0604] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0312) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | as traced (frames equal) | +0.6223 | +0.5203 | **+0.1020** | [+0.0349, +0.1656] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0773) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | as traced (frames equal) | +0.0359 | +0.0649 | **-0.0289** | [-0.0466, -0.0135] | 2000 | **DETECTED** (CI clears the floor +0.0071) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | as traced (frames equal) | +0.0280 | +0.0328 | **-0.0048** | [-0.0208, +0.0074] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0146) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | as traced (frames equal) | +0.2563 | +0.4044 | **-0.1482** | [-0.2247, +0.0169] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1848) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | as traced (frames equal) | +0.2294 | +0.3171 | **-0.0877** | [-0.1336, +0.0054] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1740) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | as traced (frames equal) | +0.0187 | +0.0862 | **-0.0676** | [-0.1548, +0.0291] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | as traced (frames equal) | -0.0290 | -0.0866 | **+0.0576** | [-0.0397, +0.1451] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.1666 | +1.2400 | **-0.0734** | [-0.3168, +0.1788] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2012) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | -0.6861 | -0.0602 | **-0.6259** | [-1.1112, -0.1496] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.7584) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.7973 | +0.9786 | **-0.1814** | [-0.5963, +0.2422] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2172) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.6890 | +0.4728 | **+0.2162** | [-0.4196, +0.8315] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.6845) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | as traced (frames equal) | +1.1107 | +1.0707 | **+0.0401** | [-0.2942, +0.4039] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3705) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.0993 | +1.2143 | **-0.1150** | [-0.4198, +0.1848] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1683) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | -0.5325 | -0.0344 | **-0.4981** | [-1.0536, +0.0586] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.8902) |

### (A) CONDITIONING or (B) SUBSTITUTION — the within/between decomposition of the own-team decode

A critic whose `V` decodes its OWN TEAM can be doing either of two things, and the own-team R² row alone cannot tell them apart: **(A)** it CONDITIONS on its team — teams differ in strength, so knowing which one it holds predicts better, and inside a team it still discriminates by the board; or **(B)** it SUBSTITUTES team identity for board state — right on average per team, blind INSIDE one. The separation is the classic between/within decomposition of a forecast's skill, with the own team as the cell:

| | within-team resolution | between-team spread | own-team R² t1 − late |
|---|---|---|---|
| **(A) conditioning** | not lower, ideally higher | UP | NEGATIVE — the board takes over as the game unfolds |
| **(B) substitution** | **DOWN** | UP | ~0 or POSITIVE — team identity still carries the forecast late |
| neither | flat | flat | flat |

**The OWN-TEAM CELL CENSUS** — the cells these rows are computed on (a team is a cell only with >= 4 battles; the strata are 5 quantiles of the team's leave-one-battle-out win rate, cut at equal battle mass):

| role | cells | teams seen | battles in cells | states | median battles/cell | median states/cell | min–max battles/cell |
|---|---|---|---|---|---|---|---|
| arm · team | 47 | 201 | 416 | 832 | 9.0 | 18.0 | 4–22 |
| arm · stratum | 5 | 201 | 416 | 832 | 83.0 | 166.0 | 78–90 |
| arm · between-team spread | 47 | 201 | 416 | — | 9.0 | — | — |
| control · team | 50 | 198 | 433 | 866 | 8.0 | 16.0 | 4–23 |
| control · stratum | 5 | 198 | 433 | 866 | 87.0 | 174.0 | 81–91 |
| control · between-team spread | 50 | 198 | 433 | — | 8.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | as traced (frames equal) | +0.0359 | +0.0649 | **-0.0289** | [-0.0466, -0.0135] | **DETECTED** (CI clears the floor +0.0071) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | as traced (frames equal) | +0.0280 | +0.0328 | **-0.0048** | [-0.0208, +0.0074] | **WITHIN FLOOR** (|delta| <= floor 0.0146) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | as traced (frames equal) | +0.2563 | +0.4044 | **-0.1482** | [-0.2247, +0.0169] | **WITHIN FLOOR** (|delta| <= floor 0.1848) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | as traced (frames equal) | -0.0104 | -0.0004 | **-0.0100** | [-0.0417, +0.0201] | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | as traced (frames equal) | +0.0187 | +0.0862 | **-0.0676** | [-0.1548, +0.0291] | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | as traced (frames equal) | -0.0290 | -0.0866 | **+0.0576** | [-0.0397, +0.1451] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

The two own-team R² rows are printed with them: the contrast is their difference, and it cannot be read without seeing which end moved.

**Reading of the three signs:** neither as posed — BOTH components are DOWN: less dispersion BETWEEN teams and less discrimination INSIDE one. Note that the spread ratio is an AMPLITUDE while the own-team R^2 is an ALIGNMENT (a monotone decode, invariant to scale), so a head can ORDER its teams better while emitting a smaller between-team spread — read the two together, never the R^2 alone.

> 🚨 **The between-team spread is an AMPLITUDE; the own-team R² is an ALIGNMENT.** The spread ratio compares `sd(mean V per team)` with `sd(that team's win rate)` — how far apart the head's per-team opinions are. The own-team R² is an out-of-fold MONOTONE decode, invariant to scale — whether those opinions are in the right ORDER. The two can move in opposite directions (a head that orders its teams correctly but under-disperses reads R² UP and spread DOWN), so the sign table above is read with both rows in hand and the R² row is never read alone.

> 🚨 **The per-TEAM cells are small and the coarse row is the check on them.** With ~719 teams in the pool a few-thousand-battle frame leaves a handful of episodes per team, and a binned resolution inside a cell that size is largely the binning's own noise — which is *positively* biased, so a small per-team number is evidence of neither reading. The STRATUM row is the same estimator on cells hundreds of episodes deep. **Where the two disagree, believe the stratum row and say so.**

> 🚨 **`cond.own_team_r2.t1_minus_late` is PROVISIONAL and is never labelled DETECTED.** NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED.

### The CALIBRATION SLOPE — is `V` correctly DISPERSED, or SHRUNK toward the base rate?

Regress the realized outcome on the forecast, in the forecast's own logit — a weighted logistic regression `logit P(y=1) = a + b · logit(V)`. `(a, b)` is the classic Cox (1958) recalibration pair: **`a` is calibration-in-the-large** (0 when the average forecast is right) and **`b` is the calibration slope** (1 when the forecast is correctly dispersed).

| slope `b` | what it says | what would fix it |
|---|---|---|
| **> 1** | **UNDER-dispersed — SHRUNK.** Where the head says 0.7 the realized rate is *above* 0.7, and where it says 0.3 it is *below*. | stretch the predictions AWAY from the base rate |
| ≈ 1 | correctly dispersed | nothing |
| < 1 | OVER-dispersed — the opinions are more extreme than the evidence behind them | shrink the predictions TOWARD the base rate |

**Why it is the sharp test of a bootstrapped target.** A λ-return (or any bootstrapped) target blends the critic's own `V` into the label, so the thing being fitted is compressed toward the base rate relative to a raw 0/1 outcome. Fitting a compressed target IS a shrinkage estimator: it can improve the rank ORDER of what is emitted while reducing its AMPLITUDE. The own-team R² row is a monotone out-of-fold decode and is invariant to scale, so it cannot see that at all; the spread ratio sees it mixed with everything else. The slope sees it directly, and a shrunk head reads `b > 1`.

**The LEVER ARM these slopes are fitted on** — the slope's SE scales as `1/sd(logit V)`, so the side with the more compressed `V` is handed the wider interval by the effect under test:

| role | frame | states | battles | mean V | **sd(V)** | **sd(logit V)** | central 2.5–97.5% of V | clipped share |
|---|---|---|---|---|---|---|---|---|
| arm · all | `all` | 1252 | 626 | 0.9050 | **0.1381** | **1.5009** | [0.503, 0.996] | 0.0000 |
| arm · t1_3 | `t1_3` | 1252 | 626 | 0.8310 | **0.0844** | **0.6535** | [0.636, 0.954] | 0.0000 |
| arm · common support | `common support` | 1167 | 620 | 0.9192 | **0.0890** | **1.2592** | [0.658, 0.995] | 0.0000 |
| control · all | `all` | 1296 | 648 | 0.8254 | **0.1849** | **1.6208** | [0.274, 0.996] | 0.0000 |
| control · t1_3 | `t1_3` | 1296 | 648 | 0.7827 | **0.0901** | **0.5742** | [0.590, 0.935] | 0.0000 |
| control · common support | `common support` | 1148 | 628 | 0.8576 | **0.1200** | **1.2684** | [0.553, 0.994] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.5026, 0.9959]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.1666 | +1.2400 | **-0.0734** | [-0.3168, +0.1788] | 0.2012 | **WITHIN FLOOR** (|delta| <= floor 0.2012) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | -0.6861 | -0.0602 | **-0.6259** | [-1.1112, -0.1496] | 0.7584 | **WITHIN FLOOR** (|delta| <= floor 0.7584) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.7973 | +0.9786 | **-0.1814** | [-0.5963, +0.2422] | 0.2172 | **WITHIN FLOOR** (|delta| <= floor 0.2172) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.6890 | +0.4728 | **+0.2162** | [-0.4196, +0.8315] | 0.6845 | **WITHIN FLOOR** (|delta| <= floor 0.6845) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | as traced (frames equal) | +1.1107 | +1.0707 | **+0.0401** | [-0.2942, +0.4039] | 0.3705 | **WITHIN FLOOR** (|delta| <= floor 0.3705) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.0993 | +1.2143 | **-0.1150** | [-0.4198, +0.1848] | 0.1683 | **WITHIN FLOOR** (|delta| <= floor 0.1683) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | -0.5325 | -0.0344 | **-0.4981** | [-1.0536, +0.0586] | 0.8902 | **WITHIN FLOOR** (|delta| <= floor 0.8902) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0000 | [+0.0000, +0.1345] | +0.0896 | [+0.0628, +0.1649] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0632 | [+0.0610, +0.1597] | +0.1154 | [+0.0948, +0.1786] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1016 | [-0.1259, -0.0724] | -0.1277 | [-0.1540, -0.1006] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.3209 | [+0.2314, +0.4551] | +0.5072 | [+0.4035, +0.6454] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.3235 | [+0.2389, +0.4454] | +0.5046 | [+0.4057, +0.6337] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0690 | [-0.0977, -0.0445] | -0.0691 | [-0.0996, -0.0425] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0212 | [+0.0139, +0.0281] | +0.0224 | [+0.0143, +0.0307] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | -0.0104 | [-0.0394, +0.0131] | -0.0004 | [-0.0174, +0.0117] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0653 | [+0.0156, +0.1106] | +0.0722 | [+0.0226, +0.1166] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.6223 | [+0.5756, +0.6661] | +0.5203 | [+0.4722, +0.5718] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0359 | [+0.0267, +0.0481] | +0.0649 | [+0.0550, +0.0805] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0280 | [+0.0214, +0.0403] | +0.0328 | [+0.0270, +0.0476] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.2563 | [+0.1839, +0.2826] | +0.4044 | [+0.2325, +0.4536] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.2294 | [+0.1774, +0.2491] | +0.3171 | [+0.2180, +0.3384] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0187 | [-0.0317, +0.0614] | +0.0862 | [+0.0010, +0.1599] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | -0.0290 | [-0.0805, +0.0211] | -0.0866 | [-0.1609, -0.0024] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.1666 | [+1.0168, +1.3342] | +1.2400 | [+1.0818, +1.4276] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | -0.6861 | [-1.0696, -0.3339] | -0.0602 | [-0.3440, +0.2153] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.7973 | [+0.5162, +1.1095] | +0.9786 | [+0.6843, +1.2969] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.6890 | [+0.2162, +1.1539] | +0.4728 | [+0.0586, +0.8777] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.1107 | [+0.8973, +1.4149] | +1.0707 | [+0.8841, +1.3086] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.0993 | [+0.9344, +1.2863] | +1.2143 | [+1.0187, +1.4470] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | -0.5325 | [-0.9692, -0.1214] | -0.0344 | [-0.3933, +0.3069] |

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_14_ladder_truevalue` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0366 [+0.0264, +0.0492] | 0.0618 | -0.0252 | +0.2134 [+0.1559, +0.2593] | ❌ | ❌ | ❌ | ✅ |
| `bot` | yes | 0.0326 [+0.0202, +0.0486] | 0.0337 | -0.0011 | +0.2782 [+0.2093, +0.3650] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0373 [+0.0238, +0.0584] | 0.0711 | -0.0338 | +0.1067 [+0.0150, +0.1844] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 370 battles / 6400 rollouts · Brier 0.0967 = REL 0.0031 − RES 0.0268 + UNC 0.1220 + WBV 0.0007 (resid -2.32e-03) · base rate 0.8577 · **resolution is 22.0% of the base-rate cap** · corr(turn,V) -0.1038 vs corr(turn,MC) -0.2166

**control — `ai_v12_16_ladder_ctrl10M_c` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0459 [+0.0353, +0.0598] | 0.0618 | -0.0159 | +0.2817 [+0.2338, +0.3308] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0177 [+0.0113, +0.0255] | 0.0337 | -0.0159 | +0.1467 [-0.0075, +0.2393] | ❌ | ❌ | ❌ | ❌ |
| `pool` | yes | 0.0596 [+0.0453, +0.0775] | 0.0711 | -0.0114 | +0.2144 [+0.1224, +0.2921] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 379 battles / 6400 rollouts · Brier 0.0946 = REL 0.0021 − RES 0.0315 + UNC 0.1240 + WBV 0.0008 (resid -8.41e-04) · base rate 0.8550 · **resolution is 25.4% of the base-rate cap** · corr(turn,V) -0.0648 vs corr(turn,MC) -0.0381

## 5. THE LEDGER LINE

```
ai_v12_14_ladder_truevalue vs ai_v12_16_ladder_ctrl10M_c at 10M: G1 bot Δ +0.0149 [+0.0000, +0.0308] NOT DETECTED · identity bias late Δ +0.0988 [+0.0370, +0.1598] NOT DETECTED · turn-contrast Δ +0.1395 [-0.0058, +0.2896] NOT DETECTED · spread ratio t1-3 Δ -0.0896 [-0.1458, +0.0310] NOT DETECTED · own-team R2 t1 Δ -0.0100 [-0.0417, +0.0201] WITHIN FLOOR · calib slope Δ -0.0734 [-0.3168, +0.1788] WITHIN FLOOR
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_14_ladder_truevalue --control ai_v12_16_ladder_ctrl10M_c --step 10000032 --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M_c

# arm — ai_v12_14_ladder_truevalue  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_16_ladder_ctrl10M_c  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_14_ladder_truevalue` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_14_ladder_truevalue/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M_c/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M_c/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
