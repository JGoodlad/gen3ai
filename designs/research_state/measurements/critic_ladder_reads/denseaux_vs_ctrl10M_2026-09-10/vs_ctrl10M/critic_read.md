# CRITIC READ — `ai_v12_20_ladder_denseaux` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v5 at 2026-09-10T19:01:47. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_20_ladder_denseaux` | `step_10000032` | 629 | 0.7% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 204 | 0.5% | 140/140 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |
| control | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_20_ladder_denseaux` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_20_ladder_denseaux` | **40/33/3** | 629 | 480 / 141 / 8 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **8/12/3** | 204 | 96 / 102 / 6 | 12 |

**Frames:** the realized per-opponent caps differ — arm 40/33/3, control 8/12/3 (traced W/L/D per opponent, counted on disk); arm cut to 8/12/3

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm side is subsampled to caps **8/12/3** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json` — 68 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0259 | +0.0187 | **+0.0072** | [-0.0094, +0.0199] | **WITHIN FLOOR** (|delta| <= floor 0.0102) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0098 | -0.0198 | **+0.0296** | [-0.0448, +0.1014] | **WITHIN FLOOR** (|delta| <= floor 0.0558) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.0312 | -0.0940 | **+0.1253** | [-0.1321, +0.3530] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2174 | +0.1372 | **+0.0802** | [-0.0708, +0.2243] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.1511 | +0.1152 | **+0.0359** | [-0.4417, +0.1500] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/3 | -0.0611 | -0.0237 | **-0.0374** | [-0.1560, +0.0721] | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.4509 | +1.4412 | **+0.0097** | [-0.4037, +0.3630] | **WITHIN FLOOR** (|delta| <= floor 0.2012) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0424 | +0.0419 | **+0.0006** | [-0.0262, +0.0205] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0127) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0259 | +0.0187 | **+0.0072** | [-0.0094, +0.0199] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0102) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0609 | +0.0720 | **-0.0111** | [-0.0537, +0.0272] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0123) |
| reliability | `all` | `capture-rate (gauge)` | +0.0003 | +0.0022 | **-0.0020** | [-0.0086, +0.0001] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0084) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0016 | +0.0033 | **-0.0017** | [-0.0106, +0.0024] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0018) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0029 | +0.0025 | **+0.0005** | [-0.0144, +0.0082] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0215) |
| ece | `all` | `capture-rate (gauge)` | +0.0076 | +0.0351 | **-0.0275** | [-0.0558, +0.0022] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0383) |
| ece | `bot` | `capture-rate (gauge)` | +0.0269 | +0.0425 | **-0.0156** | [-0.0514, +0.0209] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `pool` | `capture-rate (gauge)` | +0.0410 | +0.0407 | **+0.0002** | [-0.0641, +0.0423] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0770) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2839 | +0.2668 | **+0.0171** | [-0.1227, +0.1514] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2174 | +0.1372 | **+0.0802** | [-0.0708, +0.2243] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.3052 | +0.3592 | **-0.0540** | [-0.2017, +0.1157] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1448) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0574 | +0.0917 | **-0.0343** | [-0.0839, +0.0130] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0758) |
| bias V - p_hat | `ALL` | `pop` | +0.0063 | +0.0306 | **-0.0243** | [-0.0636, +0.0149] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0390) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0212 | -0.0418 | **+0.0207** | [-0.0120, +0.0532] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0696) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0177 | +0.0987 | **-0.0810** | [-0.1372, -0.0252] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0599) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0338 | +0.0134 | **-0.0472** | [-0.1000, +0.0026] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0499) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0605 | -0.0703 | **+0.0098** | [-0.0470, +0.0636] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0940) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0598 | +0.0889 | **-0.0291** | [-0.0798, +0.0202] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0785) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0137 | +0.0498 | **-0.0361** | [-0.0796, +0.0081] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0421) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0135 | -0.0286 | **+0.0151** | [-0.0195, +0.0477] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0518) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0965 | +0.0818 | **+0.0147** | [-0.1254, +0.1452] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0979) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0361 | +0.0297 | **+0.0064** | [-0.0968, +0.1091] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0487) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0098 | -0.0198 | **+0.0296** | [-0.0448, +0.1014] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0558) |
| bias V - p_hat | `bot` | `raw` | +0.0429 | +0.1259 | **-0.0830** | [-0.1446, -0.0211] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0667) |
| bias V - p_hat | `bot` | `pop` | -0.0224 | +0.0535 | **-0.0759** | [-0.1233, -0.0289] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0657) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0490 | -0.0456 | **-0.0035** | [-0.0391, +0.0313] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0579) |
| bias V - p_hat | `pool` | `raw` | +0.0758 | +0.0277 | **+0.0482** | [-0.0290, +0.1197] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1500) |
| bias V - p_hat | `pool` | `pop` | +0.0449 | -0.0134 | **+0.0583** | [-0.0198, +0.1325] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1049) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0251 | -0.0382 | **+0.0632** | [-0.0026, +0.1202] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0945) |
| Murphy resolution | `ALL` | `raw` | +0.0339 | +0.0364 | **-0.0025** | [-0.0200, +0.0123] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0079) |
| Murphy resolution | `ALL` | `pop` | +0.0438 | +0.0650 | **-0.0212** | [-0.0440, -0.0008] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0210) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0279 | +0.0300 | **-0.0021** | [-0.0194, +0.0122] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0091) |
| Murphy skill_score | `ALL` | `raw` | +0.1582 | +0.1172 | **+0.0410** | [-0.0577, +0.1484] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| Murphy skill_score | `ALL` | `pop` | +0.2708 | +0.2892 | **-0.0185** | [-0.1252, +0.0848] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0203) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2202 | +0.2296 | **-0.0093** | [-0.1344, +0.1251] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0496) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1834 | +0.1606 | **+0.0228** | [-0.0597, +0.0928] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0345) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2757 | +0.2935 | **-0.0179** | [-0.1229, +0.0741] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0365) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2339 | +0.2482 | **-0.0143** | [-0.1343, +0.0763] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0551) |
| Murphy reliability | `ALL` | `raw` | +0.0057 | +0.0113 | **-0.0055** | [-0.0177, +0.0023] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| Murphy reliability | `ALL` | `pop` | +0.0014 | +0.0022 | **-0.0007** | [-0.0070, +0.0021] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0077) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0022 | +0.0038 | **-0.0017** | [-0.0087, +0.0026] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0017) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.0312 | -0.0940 | **+0.1253** | [-0.1321, +0.3530] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0124 | -0.0990 | **+0.0866** | [-0.1300, +0.2913] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0886) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0244 | +0.0004 | **+0.0240** | [-0.2046, +0.2669] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0835) |

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

**arm — `ai_v12_20_ladder_denseaux` @ `step_10000032`**: 20641 states / 621 battles / 12 opponents / 193 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 8 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 6080 states / 198 battles / 12 opponents / 76 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 6 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.1511 | +0.1152 | **+0.0359** | [-0.4417, +0.1500] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/3 | +0.2531 | +0.3333 | **-0.0802** | [-0.3874, +0.0878] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0830 | -0.0886 | **+0.0056** | [-0.0577, +0.0318] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0391) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.4473 | +0.7895 | **-0.3421** | [-0.7338, -0.0603] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3898) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 8/12/3 | +0.4687 | +0.8053 | **-0.3367** | [-0.6813, +0.0236] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0541 | -0.0211 | **-0.0330** | [-0.0770, +0.0068] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0480) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0169 | +0.0098 | **+0.0071** | [-0.0047, +0.0199] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 8/12/3 | -0.0611 | -0.0237 | **-0.0374** | [-0.1560, +0.0721] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 8/12/3 | -0.0283 | -0.0098 | **-0.0186** | [-0.1863, +0.1286] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0312) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 8/12/3 | +0.4833 | +0.4619 | **+0.0214** | [-0.1236, +0.1650] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0773) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 8/12/3 | +0.0699 | +0.0839 | **-0.0139** | [-0.0600, +0.0385] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 8/12/3 | +0.0663 | +0.0819 | **-0.0156** | [-0.0829, +0.0314] | 2000 | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 8/12/3 | +0.2716 | +0.3973 | **-0.1256** | [-0.2379, +0.0078] | 1999 | **WITHIN FLOOR** (|delta| <= floor 0.1848) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/3 | +0.3177 | +0.3573 | **-0.0397** | [-0.1871, +0.0645] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1740) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 8/12/3 | -0.0943 | -0.0507 | **-0.0436** | [-4.5529, +0.3810] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 8/12/3 | +0.0495 | +0.0270 | **+0.0224** | [-0.3847, +2.4824] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.4509 | +1.4412 | **+0.0097** | [-0.4037, +0.3630] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2012) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | -0.0083 | +0.0127 | **-0.0210** | [-0.5979, +0.6363] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.7584) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +1.4336 | +0.7614 | **+0.6722** | [+0.0645, +1.2524] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.2172) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.4493 | +1.0014 | **-0.5521** | [-1.2008, +0.2233] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.6845) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 8/12/3 | +1.4318 | +1.5075 | **-0.0757** | [-1.3046, +3.0210] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3705) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.4257 | +1.3574 | **+0.0683** | [-0.3721, +0.4508] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1683) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | +0.0265 | +0.1211 | **-0.0947** | [-0.7033, +0.6176] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.8902) |

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
| arm · team | 52 | 193 | 420 | 840 | 7.5 | 15.0 | 4–21 |
| arm · stratum | 5 | 193 | 420 | 840 | 83.0 | 166.0 | 81–88 |
| arm · between-team spread | 52 | 193 | 420 | — | 7.5 | — | — |
| control · team | 15 | 76 | 104 | 208 | 5.0 | 10.0 | 4–17 |
| control · stratum | 5 | 76 | 104 | 208 | 20.0 | 40.0 | 15–27 |
| control · between-team spread | 15 | 76 | 104 | — | 5.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 8/12/3 | +0.0699 | +0.0839 | **-0.0139** | [-0.0600, +0.0385] | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 8/12/3 | +0.0663 | +0.0819 | **-0.0156** | [-0.0829, +0.0314] | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 8/12/3 | +0.2716 | +0.3973 | **-0.1256** | [-0.2379, +0.0078] | **WITHIN FLOOR** (|delta| <= floor 0.1848) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/3 | -0.0611 | -0.0237 | **-0.0374** | [-0.1560, +0.0721] | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 8/12/3 | -0.0943 | -0.0507 | **-0.0436** | [-4.5529, +0.3810] | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 8/12/3 | +0.0495 | +0.0270 | **+0.0224** | [-0.3847, +2.4824] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 1242 | 621 | 0.8351 | **0.1822** | **1.6348** | [0.226, 0.997] | 0.0000 |
| arm · t1_3 | `t1_3` | 1242 | 621 | 0.7552 | **0.0854** | **0.4882** | [0.566, 0.902] | 0.0000 |
| arm · common support | `common support` | 1150 | 614 | 0.8526 | **0.1365** | **1.3538** | [0.468, 0.994] | 0.0000 |
| control · all | `all` | 396 | 198 | 0.8288 | **0.1865** | **1.7004** | [0.308, 0.997] | 0.0000 |
| control · t1_3 | `t1_3` | 396 | 198 | 0.7697 | **0.1184** | **0.6933** | [0.437, 0.932] | 0.0000 |
| control · common support | `common support` | 347 | 188 | 0.8411 | **0.1507** | **1.4274** | [0.436, 0.994] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.3080, 0.9965]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.4509 | +1.4412 | **+0.0097** | [-0.4037, +0.3630] | 0.2012 | **WITHIN FLOOR** (|delta| <= floor 0.2012) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | -0.0083 | +0.0127 | **-0.0210** | [-0.5979, +0.6363] | 0.7584 | **WITHIN FLOOR** (|delta| <= floor 0.7584) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +1.4336 | +0.7614 | **+0.6722** | [+0.0645, +1.2524] | 0.2172 | **NOT DETECTED** (CI does not clear the floor 0.2172) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.4493 | +1.0014 | **-0.5521** | [-1.2008, +0.2233] | 0.6845 | **WITHIN FLOOR** (|delta| <= floor 0.6845) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 8/12/3 | +1.4318 | +1.5075 | **-0.0757** | [-1.3046, +3.0210] | 0.3705 | **WITHIN FLOOR** (|delta| <= floor 0.3705) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.4257 | +1.3574 | **+0.0683** | [-0.3721, +0.4508] | 0.1683 | **WITHIN FLOOR** (|delta| <= floor 0.1683) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +0.0265 | +0.1211 | **-0.0947** | [-0.7033, +0.6176] | 0.8902 | **WITHIN FLOOR** (|delta| <= floor 0.8902) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | +0.2531 [+0.1918, +0.3909] | +0.3333 | **-0.0802** | [-0.3874, +0.0878] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | +0.2348 [+0.1662, +0.2732] | +0.3333 | **-0.0985** | [-0.3971, +0.0505] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1750 | +0.3333 | **-0.1582** | [-0.4813, -0.0572] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | +0.4687 [+0.3871, +0.5534] | +0.8053 | **-0.3367** | [-0.6813, +0.0236] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | +0.4816 [+0.4187, +0.5503] | +0.8053 | **-0.3237** | [-0.6773, +0.0065] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.4503 | +0.8053 | **-0.3550** | [-0.7076, -0.0934] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | -0.0611 [-0.1924, +0.1472] | -0.0237 | **-0.0374** | [-0.1560, +0.0721] | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| `cond.own_team_r2.t1` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | -0.0185 [-0.0790, +0.0698] | -0.0237 | **+0.0052** | [-0.1215, +0.0958] | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0595 | -0.0237 | **+0.0832** | [+0.0254, +0.1797] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | -0.0283 [-0.1910, +0.1160] | -0.0098 | **-0.0186** | [-0.1863, +0.1286] | **WITHIN FLOOR** (|delta| <= floor 0.0312) |
| `cond.own_team_r2.all` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | -0.0145 [-0.0676, +0.1187] | -0.0098 | **-0.0047** | [-0.1050, +0.0733] | **WITHIN FLOOR** (|delta| <= floor 0.0312) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0361 | -0.0098 | **+0.0458** | [-0.0218, +0.1317] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | +0.4833 [+0.3668, +0.6645] | +0.4619 | **+0.0214** | [-0.1236, +0.1650] | **WITHIN FLOOR** (|delta| <= floor 0.0773) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | +0.4162 [+0.3408, +0.5299] | +0.4619 | **-0.0457** | [-0.1702, +0.0812] | **WITHIN FLOOR** (|delta| <= floor 0.0773) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5198 | +0.4619 | **+0.0579** | [-0.0551, +0.1674] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | +0.0699 [+0.0393, +0.1097] | +0.0839 | **-0.0139** | [-0.0600, +0.0385] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | +0.0705 [+0.0541, +0.0890] | +0.0839 | **-0.0134** | [-0.0477, +0.0335] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0455 | +0.0839 | **-0.0384** | [-0.0561, +0.0070] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | +0.0663 [+0.0378, +0.0932] | +0.0819 | **-0.0156** | [-0.0829, +0.0314] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | +0.0594 [+0.0414, +0.0743] | +0.0819 | **-0.0225** | [-0.0601, +0.0261] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0325 | +0.0819 | **-0.0494** | [-0.0830, -0.0107] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | +0.2716 [+0.2161, +0.3990] | +0.3973 | **-0.1256** | [-0.2379, +0.0078] | **WITHIN FLOOR** (|delta| <= floor 0.1848) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | +0.3939 [+0.2343, +2.7930] | +0.3973 | **-0.0034** | [-0.2311, +0.0810] | **WITHIN FLOOR** (|delta| <= floor 0.1848) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.4930 | +0.3973 | **+0.0957** | [-0.1255, +0.3188] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | +0.3177 [+0.2115, +0.5739] | +0.3573 | **-0.0397** | [-0.1871, +0.0645] | **WITHIN FLOOR** (|delta| <= floor 0.1740) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | +0.3191 [+0.2257, +0.5126] | +0.3573 | **-0.0382** | [-0.1711, +0.0661] | **WITHIN FLOOR** (|delta| <= floor 0.1740) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.3892 | +0.3573 | **+0.0319** | [-0.1058, +0.1417] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | -0.0943 [-0.2966, +0.0663] | -0.0507 | **-0.0436** | [-4.5529, +0.3810] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | -0.0588 [-0.1495, +0.0860] | -0.0507 | **-0.0080** | [-0.1143, +0.4076] | **WITHIN FLOOR** (|delta| <= floor 0.0405) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | -0.0082 | -0.0507 | **+0.0425** | [-0.0242, +0.4878] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | +0.0495 [-0.1389, +0.3980] | +0.0270 | **+0.0224** | [-0.3847, +2.4824] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | +0.0596 [-0.1238, +0.1401] | +0.0270 | **+0.0325** | [-0.3468, +0.5986] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0678 | +0.0270 | **+0.0407** | [-0.3516, +0.1252] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 8/12/3 (202 battles, 57 decoder battles, 21 seeds) | +1.4318 [+0.7490, +2.2778] | +1.5075 | **-0.0757** | [-1.3046, +3.0210] | **WITHIN FLOOR** (|delta| <= floor 0.3705) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 12/18/3 (263 battles, 104 decoder battles, 21 seeds) | +1.4255 [+1.1808, +2.0461] | +1.5075 | **-0.0820** | [-1.2539, +0.7361] | **WITHIN FLOOR** (|delta| <= floor 0.3705) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.4291 | +1.5075 | **-0.0784** | [-1.2453, +0.3783] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.1511 | [+0.1004, +0.2651] | +0.1152 | [+0.0000, +0.6098] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1750 | [+0.1308, +0.2733] | +0.3333 | [+0.2703, +0.6582] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0830 | [-0.1094, -0.0589] | -0.0886 | [-0.1059, -0.0353] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.4473 | [+0.3280, +0.6469] | +0.7895 | [+0.5725, +1.1703] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.4503 | [+0.3406, +0.6267] | +0.8053 | [+0.6030, +1.1480] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0541 | [-0.0825, -0.0289] | -0.0211 | [-0.0534, +0.0144] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0169 | [+0.0091, +0.0244] | +0.0098 | [-0.0006, +0.0193] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0595 | [+0.0004, +0.1085] | -0.0237 | [-0.1122, -0.0067] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0361 | [-0.0050, +0.0794] | -0.0098 | [-0.0867, +0.0421] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5198 | [+0.4709, +0.5674] | +0.4619 | [+0.3675, +0.5582] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0455 | [+0.0362, +0.0584] | +0.0839 | [+0.0413, +0.1005] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0325 | [+0.0258, +0.0449] | +0.0819 | [+0.0483, +0.1175] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.4930 | [+0.2713, +0.6274] | +0.3973 | [+0.2479, +0.4615] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.3892 | [+0.2571, +0.4301] | +0.3573 | [+0.2470, +0.4098] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | -0.0082 | [-0.0795, +0.0621] | -0.0507 | [-0.4948, -0.0286] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0678 | [-0.0152, +0.1440] | +0.0270 | [-0.0119, +0.4322] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.4509 | [+1.2674, +1.6711] | +1.4412 | [+1.1870, +1.7946] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | -0.0083 | [-0.3348, +0.2899] | +0.0127 | [-0.5732, +0.4663] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +1.4336 | [+1.0584, +1.8814] | +0.7614 | [+0.3759, +1.2507] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.4493 | [+0.0078, +0.8687] | +1.0014 | [+0.3395, +1.5465] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.4291 | [+1.1955, +1.7680] | +1.5075 | [+1.1775, +2.6126] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.4257 | [+1.2260, +1.6624] | +1.3574 | [+1.0707, +1.7319] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | +0.0265 | [-0.3197, +0.3450] | +0.1211 | [-0.5033, +0.6315] |

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_20_ladder_denseaux` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": true, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0424 [+0.0312, +0.0556] | 0.0618 | -0.0194 | +0.2839 [+0.2190, +0.3380] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0259 [+0.0175, +0.0368] | 0.0337 | -0.0077 | +0.2174 [+0.1210, +0.2968] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0609 [+0.0430, +0.0848] | 0.0711 | -0.0102 | +0.3052 [+0.2238, +0.3835] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 385 battles / 6400 rollouts · Brier 0.0931 = REL 0.0022 − RES 0.0279 + UNC 0.1194 + WBV 0.0008 (resid -1.35e-03) · base rate 0.8614 · **resolution is 23.4% of the base-rate cap** · corr(turn,V) -0.0404 vs corr(turn,MC) -0.0716

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 185 battles / 6400 rollouts · Brier 0.0930 = REL 0.0038 − RES 0.0300 + UNC 0.1208 + WBV 0.0008 (resid -2.34e-03) · base rate 0.8595 · **resolution is 24.8% of the base-rate cap** · corr(turn,V) -0.1227 vs corr(turn,MC) -0.0286

## 5. THE LEDGER LINE

```
ai_v12_20_ladder_denseaux vs ai_v12_11_ladder_ctrl10M at 10M: G1 bot Δ +0.0072 [-0.0094, +0.0199] WITHIN FLOOR · identity bias late Δ +0.0296 [-0.0448, +0.1014] WITHIN FLOOR · turn-contrast Δ +0.1253 [-0.1321, +0.3530] NOT DETECTED · spread ratio t1-3 Δ +0.0359 [-0.4417, +0.1500] NOT DETECTED · own-team R2 t1 Δ -0.0374 [-0.1560, +0.0721] WITHIN FLOOR [QUOTA-MATCHED] · calib slope Δ +0.0097 [-0.4037, +0.3630] WITHIN FLOOR
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_20_ladder_denseaux --control ai_v12_11_ladder_ctrl10M --step 10000032 --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_dense10M_vs_ctrl10M

# arm — ai_v12_20_ladder_denseaux
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_20_ladder_denseaux --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_dense10M_vs_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_20_ladder_denseaux --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/read_dense10M_vs_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/read_dense10M_vs_ctrl10M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_20_ladder_denseaux` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_20_ladder_denseaux/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_dense10M_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_dense10M_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_dense10M_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_dense10M_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_dense10M_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
