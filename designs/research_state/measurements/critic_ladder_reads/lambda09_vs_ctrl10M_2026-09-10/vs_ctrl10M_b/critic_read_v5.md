# CRITIC READ — `ai_v12_19_ladder_lambda09` vs `ai_v12_15_ladder_ctrl10M_b`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v5 at 2026-09-10T09:46:57. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | `step_10000032` | 700 | 0.5% | 150/150 (100.0%) | no |
| control | `ai_v12_15_ladder_ctrl10M_b` | `step_10000032` | 660 | 0.7% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | LIVE cycle: 13 opponents (9 scripted bots + 4 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |
| control | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_19_ladder_lambda09` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_15_ladder_ctrl10M_b` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | **40/37/2** | 700 | 520 / 174 / 6 | 13 |
| control | `ai_v12_15_ladder_ctrl10M_b` | **40/38/4** | 660 | 480 / 172 / 8 | 12 |

**Frames:** the realized per-opponent caps differ — arm 40/37/2, control 40/38/4 (traced W/L/D per opponent, counted on disk); control cut to 40/37/2

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **40/37/2** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json` — 68 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0380 | +0.0290 | **+0.0090** | [-0.0094, +0.0266] | **WITHIN FLOOR** (|delta| <= floor 0.0102) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0123 | +0.0360 | **-0.0237** | [-0.0736, +0.0273] | **WITHIN FLOOR** (|delta| <= floor 0.0558) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.0484 | -0.0787 | **+0.0303** | [-0.1359, +0.1865] | **WITHIN FLOOR** (|delta| <= floor 0.0673) |
| skill · `bot` | — | +0.2738 | +0.2015 | **+0.0723** | [-0.0107, +0.1545] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0548 | +0.0871 | **-0.0324** | [-0.1115, +0.0680] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 40/37/2 | +0.1420 | -0.0041 | **+0.1460** | [+0.0638, +0.2279] | **DETECTED** (CI clears the floor +0.0389) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.5793 | +1.3131 | **+0.2662** | [-0.0333, +0.5692] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0597 | +0.0545 | **+0.0052** | [-0.0134, +0.0234] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0127) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0380 | +0.0290 | **+0.0090** | [-0.0094, +0.0266] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0102) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0826 | +0.0793 | **+0.0033** | [-0.0230, +0.0320] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0123) |
| reliability | `all` | `capture-rate (gauge)` | +0.0083 | +0.0106 | **-0.0023** | [-0.0111, +0.0070] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0084) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0029 | +0.0029 | **-0.0000** | [-0.0068, +0.0065] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0018) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0169 | +0.0240 | **-0.0071** | [-0.0283, +0.0134] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0215) |
| ece | `all` | `capture-rate (gauge)` | +0.0552 | +0.0734 | **-0.0182** | [-0.0546, +0.0207] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0383) |
| ece | `bot` | `capture-rate (gauge)` | +0.0325 | +0.0393 | **-0.0068** | [-0.0405, +0.0334] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0124) |
| ece | `pool` | `capture-rate (gauge)` | +0.0832 | +0.1177 | **-0.0345** | [-0.0971, +0.0310] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0770) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3259 | +0.2608 | **+0.0651** | [+0.0085, +0.1283] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0149) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2738 | +0.2015 | **+0.0723** | [-0.0107, +0.1545] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.3494 | +0.2682 | **+0.0811** | [-0.0011, +0.1761] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1448) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.1037 | +0.1675 | **-0.0639** | [-0.1115, -0.0180] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0758) |
| bias V - p_hat | `ALL` | `pop` | +0.0308 | +0.0697 | **-0.0388** | [-0.0714, -0.0065] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0390) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0081 | +0.0277 | **-0.0359** | [-0.0607, -0.0110] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0696) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0831 | +0.1586 | **-0.0755** | [-0.1287, -0.0193] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0599) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0021 | +0.0634 | **-0.0613** | [-0.1024, -0.0198] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0499) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0314 | +0.0237 | **-0.0552** | [-0.0926, -0.0193] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0940) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.1167 | +0.1675 | **-0.0508** | [-0.1113, +0.0086] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0785) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0386 | +0.0669 | **-0.0284** | [-0.0698, +0.0137] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0421) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0004 | +0.0232 | **-0.0228** | [-0.0561, +0.0096] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0518) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.1178 | +0.1797 | **-0.0619** | [-0.1566, +0.0337] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0979) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0577 | +0.0784 | **-0.0207** | [-0.0847, +0.0451] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0487) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0123 | +0.0360 | **-0.0237** | [-0.0736, +0.0273] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0558) |
| bias V - p_hat | `bot` | `raw` | +0.1072 | +0.1591 | **-0.0519** | [-0.1216, +0.0152] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0667) |
| bias V - p_hat | `bot` | `pop` | +0.0133 | +0.0542 | **-0.0409** | [-0.0836, +0.0008] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0657) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0264 | +0.0123 | **-0.0387** | [-0.0678, -0.0086] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0579) |
| bias V - p_hat | `pool` | `raw` | +0.0994 | +0.1777 | **-0.0783** | [-0.1353, -0.0166] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1500) |
| bias V - p_hat | `pool` | `pop` | +0.0535 | +0.0915 | **-0.0380** | [-0.0829, +0.0088] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1049) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0226 | +0.0563 | **-0.0337** | [-0.0739, +0.0078] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0945) |
| Murphy resolution | `ALL` | `raw` | +0.0463 | +0.0424 | **+0.0039** | [-0.0113, +0.0188] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0079) |
| Murphy resolution | `ALL` | `pop` | +0.0627 | +0.0574 | **+0.0053** | [-0.0154, +0.0266] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0210) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0384 | +0.0390 | **-0.0006** | [-0.0162, +0.0154] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0091) |
| Murphy skill_score | `ALL` | `raw` | +0.1605 | +0.0407 | **+0.1198** | [+0.0344, +0.2106] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0765) |
| Murphy skill_score | `ALL` | `pop` | +0.3565 | +0.2791 | **+0.0774** | [-0.0034, +0.1605] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.3194 | +0.2792 | **+0.0402** | [-0.0422, +0.1223] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0496) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.2294 | +0.1951 | **+0.0342** | [-0.0363, +0.1048] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0345) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.3820 | +0.3300 | **+0.0520** | [-0.0415, +0.1438] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.3329 | +0.3033 | **+0.0296** | [-0.0628, +0.1232] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0551) |
| Murphy reliability | `ALL` | `raw` | +0.0158 | +0.0360 | **-0.0202** | [-0.0355, -0.0050] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| Murphy reliability | `ALL` | `pop` | +0.0052 | +0.0099 | **-0.0048** | [-0.0116, +0.0016] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0077) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0021 | +0.0036 | **-0.0016** | [-0.0052, +0.0015] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0017) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.0484 | -0.0787 | **+0.0303** | [-0.1359, +0.1865] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0673) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0116 | -0.0995 | **+0.0879** | [-0.0214, +0.1935] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0886) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0264 | -0.0831 | **+0.1095** | [-0.0033, +0.2253] | 4000 | **NOT DETECTED** (CI covers zero) |

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

**arm — `ai_v12_19_ladder_lambda09` @ `step_10000032`**: 20287 states / 694 battles / 13 opponents / 202 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 6 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (5 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

**control — `ai_v12_15_ladder_ctrl10M_b` @ `step_10000032`**: 20943 states / 652 battles / 12 opponents / 194 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 8 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0548 | +0.0871 | **-0.0324** | [-0.1115, +0.0680] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 40/37/2 | +0.0966 | +0.1140 | **-0.0174** | [-0.0828, +0.0577] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0764 | -0.0945 | **+0.0181** | [-0.0177, +0.0515] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0391) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.4694 | +0.3997 | **+0.0697** | [-0.1340, +0.2970] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3898) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 40/37/2 | +0.4601 | +0.3937 | **+0.0664** | [-0.1155, +0.2527] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0429 | -0.0621 | **+0.0193** | [-0.0187, +0.0536] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0480) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0181 | +0.0251 | **-0.0069** | [-0.0176, +0.0035] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 40/37/2 | +0.1420 | -0.0041 | **+0.1460** | [+0.0638, +0.2279] | 2000 | **DETECTED** (CI clears the floor +0.0389) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 40/37/2 | +0.1619 | +0.0414 | **+0.1205** | [+0.0466, +0.1912] | 2000 | **DETECTED** (CI clears the floor +0.0312) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 40/37/2 | +0.4714 | +0.6016 | **-0.1302** | [-0.1951, -0.0620] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0773) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 40/37/2 | +0.0455 | +0.0569 | **-0.0114** | [-0.0274, +0.0032] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 40/37/2 | +0.0355 | +0.0412 | **-0.0057** | [-0.0213, +0.0089] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0146) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 40/37/2 | +0.1631 | +0.2620 | **-0.0989** | [-0.1399, -0.0119] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1848) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 40/37/2 | +0.1521 | +0.2212 | **-0.0691** | [-0.0953, -0.0149] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1740) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 40/37/2 | +0.2262 | +0.0699 | **+0.1563** | [+0.0214, +0.2881] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0405) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 40/37/2 | -0.0842 | -0.0744 | **-0.0098** | [-0.1640, +0.1448] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.5793 | +1.3131 | **+0.2662** | [-0.0333, +0.5692] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | -0.7606 | -0.7457 | **-0.0149** | [-0.5856, +0.5524] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.7584) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +1.4263 | +0.8027 | **+0.6237** | [+0.1548, +1.1184] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.2172) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | -0.5136 | +0.3169 | **-0.8306** | [-1.6719, -0.0165] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.6845) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 40/37/2 | +1.4287 | +1.2392 | **+0.1895** | [-0.2205, +0.6190] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3705) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.5713 | +1.2875 | **+0.2838** | [-0.0423, +0.6012] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | -0.7473 | -0.6965 | **-0.0507** | [-0.6631, +0.5562] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.8902) |

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
| arm · team | 55 | 202 | 467 | 934 | 6.0 | 12.0 | 4–22 |
| arm · stratum | 5 | 202 | 467 | 934 | 94.0 | 188.0 | 86–98 |
| arm · between-team spread | 55 | 202 | 467 | — | 6.0 | — | — |
| control · team | 53 | 194 | 461 | 922 | 8.0 | 16.0 | 4–21 |
| control · stratum | 5 | 194 | 461 | 922 | 94.0 | 188.0 | 88–95 |
| control · between-team spread | 53 | 194 | 461 | — | 8.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 40/37/2 | +0.0455 | +0.0569 | **-0.0114** | [-0.0274, +0.0032] | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 40/37/2 | +0.0355 | +0.0412 | **-0.0057** | [-0.0213, +0.0089] | **WITHIN FLOOR** (|delta| <= floor 0.0146) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 40/37/2 | +0.1631 | +0.2620 | **-0.0989** | [-0.1399, -0.0119] | **WITHIN FLOOR** (|delta| <= floor 0.1848) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 40/37/2 | +0.1420 | -0.0041 | **+0.1460** | [+0.0638, +0.2279] | **DETECTED** (CI clears the floor +0.0389) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 40/37/2 | +0.2262 | +0.0699 | **+0.1563** | [+0.0214, +0.2881] | **NOT DETECTED** (CI does not clear the floor 0.0405) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 40/37/2 | -0.0842 | -0.0744 | **-0.0098** | [-0.1640, +0.1448] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 1388 | 694 | 0.8600 | **0.1669** | **1.5466** | [0.282, 0.997] | 0.0009 |
| arm · t1_3 | `t1_3` | 1388 | 694 | 0.8451 | **0.0586** | **0.4423** | [0.697, 0.936] | 0.0000 |
| arm · common support | `common support` | 1251 | 670 | 0.8773 | **0.1114** | **1.1741** | [0.545, 0.993] | 0.0000 |
| control · all | `all` | 1304 | 652 | 0.8741 | **0.1537** | **1.4000** | [0.349, 0.995] | 0.0000 |
| control · t1_3 | `t1_3` | 1304 | 652 | 0.8542 | **0.0707** | **0.5730** | [0.678, 0.953] | 0.0000 |
| control · common support | `common support` | 1221 | 645 | 0.8884 | **0.1115** | **1.1638** | [0.553, 0.993] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.3492, 0.9953]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.5793 | +1.3131 | **+0.2662** | [-0.0333, +0.5692] | 0.2012 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | -0.7606 | -0.7457 | **-0.0149** | [-0.5856, +0.5524] | 0.7584 | **WITHIN FLOOR** (|delta| <= floor 0.7584) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +1.4263 | +0.8027 | **+0.6237** | [+0.1548, +1.1184] | 0.2172 | **NOT DETECTED** (CI does not clear the floor 0.2172) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | -0.5136 | +0.3169 | **-0.8306** | [-1.6719, -0.0165] | 0.6845 | **NOT DETECTED** (CI does not clear the floor 0.6845) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 40/37/2 | +1.4287 | +1.2392 | **+0.1895** | [-0.2205, +0.6190] | 0.3705 | **WITHIN FLOOR** (|delta| <= floor 0.3705) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.5713 | +1.2875 | **+0.2838** | [-0.0423, +0.6012] | 0.1683 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | -0.7473 | -0.6965 | **-0.0507** | [-0.6631, +0.5562] | 0.8902 | **WITHIN FLOOR** (|delta| <= floor 0.8902) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +0.0966 [+0.1126, +0.1146] | +0.1140 | **-0.0174** | [-0.0828, +0.0577] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +0.0966 [+0.1139, +0.1139] | +0.1139 | **-0.0173** | [-0.0825, +0.0567] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.0966 | +0.1139 | **-0.0173** | [-0.0825, +0.0567] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +0.4601 [+0.3896, +0.3989] | +0.3937 | **+0.0664** | [-0.1155, +0.2527] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +0.4601 [+0.3947, +0.3947] | +0.3947 | **+0.0654** | [-0.1171, +0.2620] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.4601 | +0.3947 | **+0.0654** | [-0.1171, +0.2620] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +0.1420 [-0.0102, +0.0050] | -0.0041 | **+0.1460** | [+0.0638, +0.2279] | **DETECTED** (CI clears the floor +0.0389) |
| `cond.own_team_r2.t1` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +0.1420 [+0.0095, +0.0095] | +0.0095 | **+0.1325** | [+0.0516, +0.2150] | **DETECTED** (CI clears the floor +0.0389) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.1420 | +0.0095 | **+0.1325** | [+0.0516, +0.2150] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +0.1619 [+0.0368, +0.0457] | +0.0414 | **+0.1205** | [+0.0466, +0.1912] | **DETECTED** (CI clears the floor +0.0312) |
| `cond.own_team_r2.all` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +0.1619 [+0.0424, +0.0424] | +0.0424 | **+0.1195** | [+0.0469, +0.1906] | **DETECTED** (CI clears the floor +0.0312) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.1619 | +0.0424 | **+0.1195** | [+0.0469, +0.1906] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +0.4714 [+0.5987, +0.6063] | +0.6016 | **-0.1302** | [-0.1951, -0.0620] | **NOT DETECTED** (CI does not clear the floor 0.0773) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +0.4714 [+0.6057, +0.6057] | +0.6057 | **-0.1344** | [-0.2009, -0.0698] | **NOT DETECTED** (CI does not clear the floor 0.0773) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.4714 | +0.6057 | **-0.1344** | [-0.2009, -0.0698] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +0.0455 [+0.0555, +0.0604] | +0.0569 | **-0.0114** | [-0.0274, +0.0032] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +0.0455 [+0.0618, +0.0618] | +0.0618 | **-0.0162** | [-0.0324, +0.0001] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0455 | +0.0618 | **-0.0162** | [-0.0324, +0.0001] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +0.0355 [+0.0388, +0.0433] | +0.0412 | **-0.0057** | [-0.0213, +0.0089] | **WITHIN FLOOR** (|delta| <= floor 0.0146) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +0.0355 [+0.0433, +0.0433] | +0.0433 | **-0.0078** | [-0.0233, +0.0057] | **WITHIN FLOOR** (|delta| <= floor 0.0146) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0355 | +0.0433 | **-0.0078** | [-0.0233, +0.0057] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +0.1631 [+0.2610, +0.2640] | +0.2620 | **-0.0989** | [-0.1399, -0.0119] | **WITHIN FLOOR** (|delta| <= floor 0.1848) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +0.1631 [+0.2626, +0.2626] | +0.2626 | **-0.0994** | [-0.1362, -0.0101] | **WITHIN FLOOR** (|delta| <= floor 0.1848) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.1631 | +0.2626 | **-0.0994** | [-0.1362, -0.0101] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +0.1521 [+0.2208, +0.2222] | +0.2212 | **-0.0691** | [-0.0953, -0.0149] | **WITHIN FLOOR** (|delta| <= floor 0.1740) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +0.1521 [+0.2216, +0.2216] | +0.2216 | **-0.0695** | [-0.0942, -0.0141] | **WITHIN FLOOR** (|delta| <= floor 0.1740) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1521 | +0.2216 | **-0.0695** | [-0.0942, -0.0141] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +0.2262 [+0.0579, +0.0872] | +0.0699 | **+0.1563** | [+0.0214, +0.2881] | **NOT DETECTED** (CI does not clear the floor 0.0405) |
| `cond.own_team_r2.late` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +0.2262 [+0.0561, +0.0561] | +0.0561 | **+0.1701** | [+0.0348, +0.2970] | **NOT DETECTED** (CI does not clear the floor 0.0405) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.2262 | +0.0561 | **+0.1701** | [+0.0348, +0.2970] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | -0.0842 [-0.0910, -0.0553] | -0.0744 | **-0.0098** | [-0.1640, +0.1448] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | -0.0842 [-0.0466, -0.0466] | -0.0466 | **-0.0375** | [-0.1985, +0.1190] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | -0.0842 | -0.0466 | **-0.0375** | [-0.1985, +0.1190] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 40/37/2 (651 battles, 460 decoder battles, 21 seeds) | +1.4287 [+1.2059, +1.2649] | +1.2392 | **+0.1895** | [-0.2205, +0.6190] | **WITHIN FLOOR** (|delta| <= floor 0.3705) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 45/42/2 (652 battles, 461 decoder battles, 21 seeds) | +1.4287 [+1.2377, +1.2377] | +1.2377 | **+0.1910** | [-0.2115, +0.6233] | **WITHIN FLOOR** (|delta| <= floor 0.3705) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.4287 | +1.2377 | **+0.1910** | [-0.2115, +0.6233] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0548 | [+0.0228, +0.1640] | +0.0871 | [+0.0614, +0.1739] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0966 | [+0.0742, +0.1748] | +0.1139 | [+0.0905, +0.1852] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0764 | [-0.1036, -0.0565] | -0.0945 | [-0.1222, -0.0718] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.4694 | [+0.3145, +0.6675] | +0.3997 | [+0.2927, +0.5473] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.4601 | [+0.3250, +0.6238] | +0.3947 | [+0.2967, +0.5230] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0429 | [-0.0726, -0.0226] | -0.0621 | [-0.0923, -0.0388] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0181 | [+0.0111, +0.0255] | +0.0251 | [+0.0174, +0.0330] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.1420 | [+0.0609, +0.2204] | +0.0095 | [-0.0155, +0.0305] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.1619 | [+0.0929, +0.2259] | +0.0424 | [+0.0101, +0.0716] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.4714 | [+0.4227, +0.5179] | +0.6057 | [+0.5626, +0.6477] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0455 | [+0.0354, +0.0576] | +0.0618 | [+0.0510, +0.0737] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0355 | [+0.0275, +0.0485] | +0.0433 | [+0.0357, +0.0567] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.1631 | [+0.1253, +0.1823] | +0.2626 | [+0.1710, +0.2813] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1521 | [+0.1231, +0.1639] | +0.2216 | [+0.1628, +0.2300] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.2262 | [+0.1070, +0.3277] | +0.0561 | [-0.0191, +0.1195] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | -0.0842 | [-0.2228, +0.0591] | -0.0466 | [-0.1168, +0.0294] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.5793 | [+1.3781, +1.8312] | +1.3131 | [+1.1345, +1.5362] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | -0.7606 | [-1.1974, -0.3652] | -0.7457 | [-1.1568, -0.3945] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +1.4263 | [+1.0375, +1.8603] | +0.8027 | [+0.5567, +1.0696] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | -0.5136 | [-1.2204, +0.1434] | +0.3169 | [-0.1712, +0.7897] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.4287 | [+1.1483, +1.8233] | +1.2377 | [+1.0336, +1.5369] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.5713 | [+1.3341, +1.8383] | +1.2875 | [+1.1014, +1.5175] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | -0.7473 | [-1.2132, -0.3015] | -0.6965 | [-1.1228, -0.3088] |

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_19_ladder_lambda09` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 9}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0597 [+0.0478, +0.0727] | 0.0618 | -0.0021 | +0.3259 [+0.2885, +0.3667] | ❌ | ❌ | ❌ | ✅ |
| `bot` | yes | 0.0380 [+0.0273, +0.0515] | 0.0337 | +0.0044 | +0.2738 [+0.2223, +0.3240] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0826 [+0.0638, +0.1072] | 0.0711 | +0.0115 | +0.3494 [+0.2946, +0.4073] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 412 battles / 6400 rollouts · Brier 0.0786 = REL 0.0021 − RES 0.0384 + UNC 0.1155 + WBV 0.0008 (resid -1.34e-03) · base rate 0.8668 · **resolution is 33.3% of the base-rate cap** · corr(turn,V) -0.1078 vs corr(turn,MC) -0.0594

**control — `ai_v12_15_ladder_ctrl10M_b` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0545 [+0.0412, +0.0680] | 0.0618 | -0.0073 | +0.2608 [+0.2118, +0.3048] | ❌ | ❌ | ❌ | ✅ |
| `bot` | yes | 0.0290 [+0.0182, +0.0454] | 0.0337 | -0.0047 | +0.2015 [+0.1342, +0.2626] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0793 [+0.0608, +0.0976] | 0.0711 | +0.0082 | +0.2682 [+0.1854, +0.3287] | ❌ | ❌ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 381 battles / 6400 rollouts · Brier 0.0928 = REL 0.0036 − RES 0.0390 + UNC 0.1287 + WBV 0.0008 (resid -1.30e-03) · base rate 0.8482 · **resolution is 30.3% of the base-rate cap** · corr(turn,V) -0.1643 vs corr(turn,MC) -0.0856

## 5. THE LEDGER LINE

```
ai_v12_19_ladder_lambda09 vs ai_v12_15_ladder_ctrl10M_b at 10M: G1 bot Δ +0.0090 [-0.0094, +0.0266] WITHIN FLOOR · identity bias late Δ -0.0237 [-0.0736, +0.0273] WITHIN FLOOR · turn-contrast Δ +0.0303 [-0.1359, +0.1865] WITHIN FLOOR · spread ratio t1-3 Δ -0.0324 [-0.1115, +0.0680] NOT DETECTED · own-team R2 t1 Δ +0.1460 [+0.0638, +0.2279] DETECTED [QUOTA-MATCHED] · calib slope Δ +0.2662 [-0.0333, +0.5692] NOT DETECTED
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_19_ladder_lambda09 --control ai_v12_15_ladder_ctrl10M_b --step 10000032 --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_v5_vs_ai_v12_15_ladder_ctrl10M_b

# arm — ai_v12_19_ladder_lambda09  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_15_ladder_ctrl10M_b  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_v5_vs_ai_v12_15_ladder_ctrl10M_b/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_v5_vs_ai_v12_15_ladder_ctrl10M_b/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
