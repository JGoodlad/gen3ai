# CRITIC READ — `ai_v12_15_ladder_ctrl10M_b` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v5 at 2026-09-10T09:38:48. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_15_ladder_ctrl10M_b` | `step_10000032` | 660 | 0.7% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 204 | 0.5% | 140/140 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |
| control | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_15_ladder_ctrl10M_b` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_15_ladder_ctrl10M_b` | **40/38/4** | 660 | 480 / 172 / 8 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **8/12/3** | 204 | 96 / 102 / 6 | 12 |

**Frames:** the realized per-opponent caps differ — arm 40/38/4, control 8/12/3 (traced W/L/D per opponent, counted on disk); arm cut to 8/12/3

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm side is subsampled to caps **8/12/3** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> 🚨 **NO REPLICATE FLOOR — deltas are NOT DETECTED unless their CI clears zero, and never DETECTED against a floor.** The ladder's replicate floor is the control-vs-control difference and does not exist until a second control replicate does. Every DETECTED below is a detection against ZERO and carries that qualifier.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0290 | +0.0187 | **+0.0102** | [-0.0083, +0.0288] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0360 | -0.0198 | **+0.0558** | [-0.0188, +0.1335] | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.0787 | -0.0940 | **+0.0153** | [-0.2329, +0.2390] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2015 | +0.1372 | **+0.0642** | [-0.0772, +0.2030] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0871 | +0.1152 | **-0.0281** | [-0.4982, +0.0919] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/3 | -0.0360 | -0.0237 | **-0.0123** | [-0.0852, +0.0671] | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.3131 | +1.4412 | **-0.1281** | [-0.5295, +0.1973] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0545 | +0.0419 | **+0.0127** | [-0.0152, +0.0339] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0290 | +0.0187 | **+0.0102** | [-0.0083, +0.0288] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0793 | +0.0720 | **+0.0073** | [-0.0366, +0.0422] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0106 | +0.0022 | **+0.0084** | [-0.0006, +0.0149] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0029 | +0.0033 | **-0.0004** | [-0.0106, +0.0057] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0240 | +0.0025 | **+0.0215** | [-0.0010, +0.0380] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `all` | `capture-rate (gauge)` | +0.0734 | +0.0351 | **+0.0383** | [-0.0092, +0.0667] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `bot` | `capture-rate (gauge)` | +0.0393 | +0.0425 | **-0.0032** | [-0.0635, +0.0352] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `pool` | `capture-rate (gauge)` | +0.1177 | +0.0407 | **+0.0770** | [-0.0203, +0.1173] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2608 | +0.2668 | **-0.0060** | [-0.1310, +0.1322] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2015 | +0.1372 | **+0.0642** | [-0.0772, +0.2030] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2682 | +0.3592 | **-0.0910** | [-0.2310, +0.0958] | 400 | **NOT DETECTED** (CI covers zero) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.1675 | +0.0917 | **+0.0758** | [+0.0236, +0.1261] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `ALL` | `pop` | +0.0697 | +0.0306 | **+0.0390** | [-0.0009, +0.0795] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | +0.0277 | -0.0418 | **+0.0696** | [+0.0359, +0.1011] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.1586 | +0.0987 | **+0.0599** | [-0.0015, +0.1161] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0634 | +0.0134 | **+0.0499** | [-0.0032, +0.1016] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | +0.0237 | -0.0703 | **+0.0940** | [+0.0379, +0.1485] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.1675 | +0.0889 | **+0.0785** | [+0.0244, +0.1336] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0669 | +0.0498 | **+0.0171** | [-0.0262, +0.0628] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0232 | -0.0286 | **+0.0518** | [+0.0175, +0.0853] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.1797 | +0.0818 | **+0.0979** | [-0.0524, +0.2350] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0784 | +0.0297 | **+0.0487** | [-0.0591, +0.1549] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0360 | -0.0198 | **+0.0558** | [-0.0188, +0.1335] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `bot` | `raw` | +0.1591 | +0.1259 | **+0.0332** | [-0.0358, +0.1033] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `bot` | `pop` | +0.0542 | +0.0535 | **+0.0007** | [-0.0487, +0.0525] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `bot` | `ipw` | +0.0123 | -0.0456 | **+0.0579** | [+0.0211, +0.0951] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `pool` | `raw` | +0.1777 | +0.0277 | **+0.1500** | [+0.0751, +0.2199] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `pool` | `pop` | +0.0915 | -0.0134 | **+0.1049** | [+0.0309, +0.1769] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0563 | -0.0382 | **+0.0945** | [+0.0313, +0.1518] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| Murphy resolution | `ALL` | `raw` | +0.0424 | +0.0364 | **+0.0060** | [-0.0117, +0.0221] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0574 | +0.0650 | **-0.0076** | [-0.0302, +0.0138] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0390 | +0.0300 | **+0.0091** | [-0.0094, +0.0247] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.0407 | +0.1172 | **-0.0765** | [-0.1848, +0.0401] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `pop` | +0.2791 | +0.2892 | **-0.0102** | [-0.1047, +0.0892] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2792 | +0.2296 | **+0.0496** | [-0.0640, +0.1785] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1951 | +0.1606 | **+0.0345** | [-0.0474, +0.1072] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.3300 | +0.2935 | **+0.0365** | [-0.0632, +0.1287] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.3033 | +0.2482 | **+0.0551** | [-0.0594, +0.1508] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0360 | +0.0113 | **+0.0247** | [+0.0086, +0.0391] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| Murphy reliability | `ALL` | `pop` | +0.0099 | +0.0022 | **+0.0077** | [+0.0001, +0.0135] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0036 | +0.0038 | **-0.0002** | [-0.0073, +0.0043] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.0787 | -0.0940 | **+0.0153** | [-0.2329, +0.2390] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0995 | -0.0990 | **-0.0005** | [-0.2026, +0.1885] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0831 | +0.0004 | **-0.0835** | [-0.3085, +0.1483] | 4000 | **NOT DETECTED** (CI covers zero) |

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

**arm — `ai_v12_15_ladder_ctrl10M_b` @ `step_10000032`**: 20943 states / 652 battles / 12 opponents / 194 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 8 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 6080 states / 198 battles / 12 opponents / 76 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 6 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0871 | +0.1152 | **-0.0281** | [-0.4982, +0.0919] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/3 | +0.2006 | +0.3333 | **-0.1327** | [-0.4276, +0.0098] | 2000 | **NOT DETECTED** (CI covers zero) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0945 | -0.0886 | **-0.0059** | [-0.0705, +0.0199] | 2000 | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.3997 | +0.7895 | **-0.3898** | [-0.7951, -0.1195] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 8/12/3 | +0.4201 | +0.8053 | **-0.3852** | [-0.7240, -0.0879] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0621 | -0.0211 | **-0.0410** | [-0.0903, -0.0025] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0251 | +0.0098 | **+0.0153** | [+0.0033, +0.0286] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 8/12/3 | -0.0360 | -0.0237 | **-0.0123** | [-0.0852, +0.0671] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 8/12/3 | -0.0380 | -0.0098 | **-0.0282** | [-0.1927, +0.0516] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 8/12/3 | +0.5285 | +0.4619 | **+0.0666** | [-0.0717, +0.2003] | 2000 | **NOT DETECTED** (CI covers zero) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 8/12/3 | +0.0768 | +0.0839 | **-0.0071** | [-0.0508, +0.0311] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 8/12/3 | +0.0673 | +0.0819 | **-0.0146** | [-0.0526, +0.0392] | 2000 | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 8/12/3 | +0.2124 | +0.3973 | **-0.1848** | [-0.3148, -0.0614] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/3 | +0.1833 | +0.3573 | **-0.1740** | [-0.2669, -0.0755] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 8/12/3 | -0.0827 | -0.0507 | **-0.0320** | [-0.3062, +0.3866] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 8/12/3 | +0.0650 | +0.0270 | **+0.0380** | [-0.3527, +0.5669] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.3131 | +1.4412 | **-0.1281** | [-0.5295, +0.1973] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | -0.7457 | +0.0127 | **-0.7584** | [-1.3931, -0.0785] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.8027 | +0.7614 | **+0.0412** | [-0.5019, +0.5072] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.3169 | +1.0014 | **-0.6845** | [-1.3932, +0.1310] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 8/12/3 | +1.3452 | +1.5075 | **-0.1624** | [-1.2983, +1.3816] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.2875 | +1.3071 | **-0.0196** | [-0.4428, +0.3479] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | -0.6965 | +0.1937 | **-0.8902** | [-1.5835, -0.1544] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |

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
| arm · team | 53 | 194 | 461 | 922 | 8.0 | 16.0 | 4–21 |
| arm · stratum | 5 | 194 | 461 | 922 | 94.0 | 188.0 | 88–95 |
| arm · between-team spread | 53 | 194 | 461 | — | 8.0 | — | — |
| control · team | 15 | 76 | 104 | 208 | 5.0 | 10.0 | 4–17 |
| control · stratum | 5 | 76 | 104 | 208 | 20.0 | 40.0 | 15–27 |
| control · between-team spread | 15 | 76 | 104 | — | 5.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 8/12/3 | +0.0768 | +0.0839 | **-0.0071** | [-0.0508, +0.0311] | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 8/12/3 | +0.0673 | +0.0819 | **-0.0146** | [-0.0526, +0.0392] | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 8/12/3 | +0.2124 | +0.3973 | **-0.1848** | [-0.3148, -0.0614] | **DETECTED** (vs ZERO — NO FLOOR) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/3 | -0.0360 | -0.0237 | **-0.0123** | [-0.0852, +0.0671] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 8/12/3 | -0.0827 | -0.0507 | **-0.0320** | [-0.3062, +0.3866] | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 8/12/3 | +0.0650 | +0.0270 | **+0.0380** | [-0.3527, +0.5669] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 1304 | 652 | 0.8741 | **0.1537** | **1.4000** | [0.349, 0.995] | 0.0000 |
| arm · t1_3 | `t1_3` | 1304 | 652 | 0.8542 | **0.0707** | **0.5730** | [0.678, 0.953] | 0.0000 |
| arm · common support | `common support` | 1221 | 645 | 0.8884 | **0.1115** | **1.1638** | [0.553, 0.993] | 0.0000 |
| control · all | `all` | 396 | 198 | 0.8288 | **0.1865** | **1.7004** | [0.308, 0.997] | 0.0000 |
| control · t1_3 | `t1_3` | 396 | 198 | 0.7697 | **0.1184** | **0.6933** | [0.437, 0.932] | 0.0000 |
| control · common support | `common support` | 337 | 184 | 0.8432 | **0.1460** | **1.3983** | [0.443, 0.994] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.3492, 0.9953]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.3131 | +1.4412 | **-0.1281** | [-0.5295, +0.1973] | — | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | -0.7457 | +0.0127 | **-0.7584** | [-1.3931, -0.0785] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.8027 | +0.7614 | **+0.0412** | [-0.5019, +0.5072] | — | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.3169 | +1.0014 | **-0.6845** | [-1.3932, +0.1310] | — | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 8/12/3 | +1.3452 | +1.5075 | **-0.1624** | [-1.2983, +1.3816] | — | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.2875 | +1.3071 | **-0.0196** | [-0.4428, +0.3479] | — | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | -0.6965 | +0.1937 | **-0.8902** | [-1.5835, -0.1544] | — | **DETECTED** (vs ZERO — NO FLOOR) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | +0.2006 [+0.1057, +0.2782] | +0.3333 | **-0.1327** | [-0.4276, +0.0098] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | +0.1847 [+0.1502, +0.2516] | +0.3333 | **-0.1486** | [-0.4582, -0.0384] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1139 | +0.3333 | **-0.2193** | [-0.5328, -0.1271] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | +0.4201 [+0.3431, +0.4631] | +0.8053 | **-0.3852** | [-0.7240, -0.0879] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | +0.4056 [+0.3705, +0.4753] | +0.8053 | **-0.3997** | [-0.7540, -0.0970] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.3947 | +0.8053 | **-0.4106** | [-0.7772, -0.1644] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | -0.0360 [-0.1145, +0.3179] | -0.0237 | **-0.0123** | [-0.0852, +0.0671] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | -0.0341 [-0.0857, +0.2751] | -0.0237 | **-0.0104** | [-0.1087, +0.0723] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0095 | -0.0237 | **+0.0331** | [+0.0035, +0.1198] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | -0.0380 [-0.1085, +0.0433] | -0.0098 | **-0.0282** | [-0.1927, +0.0516] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | -0.0283 [-0.0792, +0.0797] | -0.0098 | **-0.0185** | [-0.1397, +0.0545] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0424 | -0.0098 | **+0.0522** | [-0.0081, +0.1324] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | +0.5285 [+0.4009, +0.6004] | +0.4619 | **+0.0666** | [-0.0717, +0.2003] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | +0.5874 [+0.5120, +0.6345] | +0.4619 | **+0.1255** | [-0.0026, +0.2512] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.6057 | +0.4619 | **+0.1438** | [+0.0385, +0.2494] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | +0.0768 [+0.0550, +0.0966] | +0.0839 | **-0.0071** | [-0.0508, +0.0311] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | +0.0776 [+0.0534, +0.0956] | +0.0839 | **-0.0062** | [-0.0413, +0.0385] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0618 | +0.0839 | **-0.0221** | [-0.0393, +0.0220] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | +0.0673 [+0.0452, +0.0853] | +0.0819 | **-0.0146** | [-0.0526, +0.0392] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | +0.0631 [+0.0403, +0.0873] | +0.0819 | **-0.0188** | [-0.0637, +0.0299] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0433 | +0.0819 | **-0.0386** | [-0.0729, +0.0002] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | +0.2124 [+0.1556, +0.4400] | +0.3973 | **-0.1848** | [-0.3148, -0.0614] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | +0.2056 [+0.1479, +0.3963] | +0.3973 | **-0.1916** | [-0.3238, -0.0906] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.2626 | +0.3973 | **-0.1347** | [-0.2529, -0.0103] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | +0.1833 [+0.1414, +0.2381] | +0.3573 | **-0.1740** | [-0.2669, -0.0755] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | +0.1759 [+0.1406, +0.2965] | +0.3573 | **-0.1814** | [-0.2795, -0.1029] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.2216 | +0.3573 | **-0.1357** | [-0.2207, -0.0439] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | -0.0827 [-0.2041, +0.2319] | -0.0507 | **-0.0320** | [-0.3062, +0.3866] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | -0.0815 [-0.2115, +0.1641] | -0.0507 | **-0.0308** | [-0.2651, +0.3765] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0561 | -0.0507 | **+0.1068** | [+0.0385, +0.5612] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | +0.0650 [-0.2803, +0.4170] | +0.0270 | **+0.0380** | [-0.3527, +0.5669] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | +0.0918 [-0.1175, +0.3370] | +0.0270 | **+0.0648** | [-0.3267, +0.2910] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | -0.0466 | +0.0270 | **-0.0737** | [-0.4714, +0.0044] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 8/12/3 (213 battles, 89 decoder battles, 21 seeds) | +1.3452 [+1.0142, +1.7393] | +1.5075 | **-0.1624** | [-1.2983, +1.3816] | **NOT DETECTED** (CI covers zero) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 9/14/3 (235 battles, 106 decoder battles, 21 seeds) | +1.5188 [+1.1491, +2.0537] | +1.5075 | **+0.0113** | [-1.1381, +1.1122] | **NOT DETECTED** (CI covers zero) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.2377 | +1.5075 | **-0.2698** | [-1.3598, +0.1687] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0871 | [+0.0614, +0.1739] | +0.1152 | [+0.0000, +0.6098] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1139 | [+0.0905, +0.1852] | +0.3333 | [+0.2703, +0.6582] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0945 | [-0.1222, -0.0718] | -0.0886 | [-0.1059, -0.0353] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.3997 | [+0.2927, +0.5473] | +0.7895 | [+0.5725, +1.1703] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.3947 | [+0.2967, +0.5230] | +0.8053 | [+0.6030, +1.1480] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0621 | [-0.0923, -0.0388] | -0.0211 | [-0.0534, +0.0144] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0251 | [+0.0174, +0.0330] | +0.0098 | [-0.0006, +0.0193] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0095 | [-0.0155, +0.0305] | -0.0237 | [-0.1122, -0.0067] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0424 | [+0.0101, +0.0716] | -0.0098 | [-0.0867, +0.0421] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.6057 | [+0.5626, +0.6477] | +0.4619 | [+0.3675, +0.5582] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0618 | [+0.0510, +0.0737] | +0.0839 | [+0.0413, +0.1005] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0433 | [+0.0357, +0.0567] | +0.0819 | [+0.0483, +0.1175] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.2626 | [+0.1710, +0.2813] | +0.3973 | [+0.2479, +0.4615] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.2216 | [+0.1628, +0.2300] | +0.3573 | [+0.2470, +0.4098] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0561 | [-0.0191, +0.1195] | -0.0507 | [-0.4948, -0.0286] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | -0.0466 | [-0.1168, +0.0294] | +0.0270 | [-0.0119, +0.4322] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.3131 | [+1.1345, +1.5362] | +1.4412 | [+1.1870, +1.7946] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | -0.7457 | [-1.1568, -0.3945] | +0.0127 | [-0.5732, +0.4663] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.8027 | [+0.5567, +1.0696] | +0.7614 | [+0.3759, +1.2507] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.3169 | [-0.1712, +0.7897] | +1.0014 | [+0.3395, +1.5465] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.2377 | [+1.0336, +1.5369] | +1.5075 | [+1.1775, +2.6126] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.2875 | [+1.1014, +1.5175] | +1.3071 | [+1.0195, +1.6871] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | -0.6965 | [-1.1228, -0.3088] | +0.1937 | [-0.4630, +0.7289] |

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_15_ladder_ctrl10M_b` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0545 [+0.0412, +0.0680] | 0.0618 | -0.0073 | +0.2608 [+0.2118, +0.3048] | ❌ | ❌ | ❌ | ✅ |
| `bot` | yes | 0.0290 [+0.0182, +0.0454] | 0.0337 | -0.0047 | +0.2015 [+0.1342, +0.2626] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0793 [+0.0608, +0.0976] | 0.0711 | +0.0082 | +0.2682 [+0.1854, +0.3287] | ❌ | ❌ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 381 battles / 6400 rollouts · Brier 0.0928 = REL 0.0036 − RES 0.0390 + UNC 0.1287 + WBV 0.0008 (resid -1.30e-03) · base rate 0.8482 · **resolution is 30.3% of the base-rate cap** · corr(turn,V) -0.1643 vs corr(turn,MC) -0.0856

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 185 battles / 6400 rollouts · Brier 0.0930 = REL 0.0038 − RES 0.0300 + UNC 0.1208 + WBV 0.0008 (resid -2.34e-03) · base rate 0.8595 · **resolution is 24.8% of the base-rate cap** · corr(turn,V) -0.1227 vs corr(turn,MC) -0.0286

## 5. THE LEDGER LINE

```
ai_v12_15_ladder_ctrl10M_b vs ai_v12_11_ladder_ctrl10M at 10M: G1 bot Δ +0.0102 [-0.0083, +0.0288] NOT DETECTED · identity bias late Δ +0.0558 [-0.0188, +0.1335] NOT DETECTED · turn-contrast Δ +0.0153 [-0.2329, +0.2390] NOT DETECTED · spread ratio t1-3 Δ -0.0281 [-0.4982, +0.0919] NOT DETECTED · own-team R2 t1 Δ -0.0123 [-0.0852, +0.0671] NOT DETECTED [QUOTA-MATCHED] · calib slope Δ -0.1281 [-0.5295, +0.1973] NOT DETECTED
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_15_ladder_ctrl10M_b --control ai_v12_11_ladder_ctrl10M --step 10000032 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_floor_v5

# arm — ai_v12_15_ladder_ctrl10M_b  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_floor_v5/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_floor_v5/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
