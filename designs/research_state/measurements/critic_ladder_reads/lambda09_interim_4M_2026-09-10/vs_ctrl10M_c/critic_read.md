# CRITIC READ — `ai_v12_19_ladder_lambda09` vs `ai_v12_16_ladder_ctrl10M_c`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v3 at 2026-09-10T02:35:32. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | `step_4000032` | 576 | 0.7% | 150/150 (100.0%) | yes pid 737657,898663 |
| control | `ai_v12_16_ladder_ctrl10M_c` | `step_4000032` | 560 | 1.3% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | LIVE cycle: 10 opponents (9 scripted bots + 1 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |
| control | LIVE cycle: 9 opponents (9 scripted bots + 0 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_19_ladder_lambda09` → `step_4000032`: PINNED by --step 4000032
- **control** `ai_v12_16_ladder_ctrl10M_c` → `step_4000032`: PINNED by --step 4000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | **40/28/2** | 576 | 400 / 169 / 7 | 10 |
| control | `ai_v12_16_ladder_ctrl10M_c` | **40/29/4** | 560 | 360 / 188 / 12 | 9 |

**Frames:** the realized per-opponent caps differ — arm 40/28/2, control 40/29/4 (traced W/L/D per opponent, counted on disk); control cut to 40/28/2

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **40/28/2** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json` — 55 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0492 | +0.0617 | **-0.0125** | [-0.0317, +0.0098] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0678 | +0.0157 | **-0.0835** | [-0.1517, -0.0191] | **NOT DETECTED** (CI does not clear the floor 0.0558) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.1237 | +0.0795 | **-0.2033** | [-0.3278, -0.0851] | **DETECTED** (CI clears the floor +0.0673) |
| skill · `bot` | — | +0.2902 | +0.2924 | **-0.0022** | [-0.0848, +0.0847] | **WITHIN FLOOR** (|delta| <= floor 0.0642) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0768 | +0.1657 | **-0.0889** | [-0.2694, +0.1058] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 40/28/2 | +0.0804 | +0.0224 | **+0.0580** | [-0.0069, +0.1253] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0545 | +0.0617 | **-0.0073** | [-0.0295, +0.0137] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0127) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0492 | +0.0617 | **-0.0125** | [-0.0317, +0.0098] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0023 | +0.0012 | **+0.0011** | [-0.0021, +0.0035] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0084) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0013 | +0.0012 | **+0.0001** | [-0.0031, +0.0026] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0018) |
| ece | `all` | `capture-rate (gauge)` | +0.0345 | +0.0283 | **+0.0063** | [-0.0213, +0.0255] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0383) |
| ece | `bot` | `capture-rate (gauge)` | +0.0301 | +0.0283 | **+0.0018** | [-0.0268, +0.0222] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0124) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3052 | +0.2924 | **+0.0128** | [-0.0805, +0.0937] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0149) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2902 | +0.2924 | **-0.0022** | [-0.0848, +0.0847] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0642) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0589 | +0.0946 | **-0.0358** | [-0.0866, +0.0131] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0758) |
| bias V - p_hat | `ALL` | `pop` | -0.0107 | +0.0050 | **-0.0158** | [-0.0543, +0.0204] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0390) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0477 | -0.0389 | **-0.0088** | [-0.0421, +0.0225] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0696) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.1027 | +0.0824 | **+0.0203** | [-0.0431, +0.0826] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0599) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0225 | -0.0147 | **+0.0372** | [-0.0162, +0.0879] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0499) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0248 | -0.0608 | **+0.0361** | [-0.0118, +0.0809] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0940) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0563 | +0.0765 | **-0.0203** | [-0.0860, +0.0440] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0785) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0185 | -0.0010 | **-0.0175** | [-0.0654, +0.0293] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0421) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0558 | -0.0392 | **-0.0166** | [-0.0608, +0.0257] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0518) |
| bias V - p_hat | `late (turn>=25)` | `raw` | -0.0104 | +0.1435 | **-0.1539** | [-0.2472, -0.0489] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0979) |
| bias V - p_hat | `late (turn>=25)` | `pop` | -0.0478 | +0.0620 | **-0.1098** | [-0.1836, -0.0331] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0487) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0678 | +0.0157 | **-0.0835** | [-0.1517, -0.0191] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0558) |
| bias V - p_hat | `bot` | `raw` | +0.0859 | +0.0946 | **-0.0088** | [-0.0627, +0.0446] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0667) |
| bias V - p_hat | `bot` | `pop` | +0.0092 | +0.0050 | **+0.0041** | [-0.0359, +0.0431] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0657) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0362 | -0.0389 | **+0.0027** | [-0.0319, +0.0367] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0579) |
| Murphy resolution | `ALL` | `raw` | +0.0376 | +0.0459 | **-0.0083** | [-0.0243, +0.0074] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0538 | +0.0716 | **-0.0178** | [-0.0376, +0.0030] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0210) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0356 | +0.0543 | **-0.0187** | [-0.0361, -0.0014] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0091) |
| Murphy skill_score | `ALL` | `raw` | +0.1460 | +0.1372 | **+0.0088** | [-0.0875, +0.1082] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| Murphy skill_score | `ALL` | `pop` | +0.2878 | +0.3342 | **-0.0463** | [-0.1325, +0.0426] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2228 | +0.2991 | **-0.0763** | [-0.1743, +0.0252] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1799 | +0.1970 | **-0.0171** | [-0.0874, +0.0536] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0345) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2917 | +0.3327 | **-0.0410** | [-0.1245, +0.0479] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2519 | +0.3121 | **-0.0602** | [-0.1433, +0.0265] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0076 | +0.0158 | **-0.0083** | [-0.0207, +0.0022] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| Murphy reliability | `ALL` | `pop` | +0.0011 | +0.0010 | **+0.0000** | [-0.0032, +0.0028] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0077) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0041 | +0.0034 | **+0.0007** | [-0.0050, +0.0056] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0017) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.1237 | +0.0795 | **-0.2033** | [-0.3278, -0.0851] | 4000 | **DETECTED** (CI clears the floor +0.0673) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0884 | +0.0298 | **-0.1183** | [-0.2280, -0.0294] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0886) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0534 | +0.0499 | **-0.1033** | [-0.2127, -0.0119] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0835) |

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

**arm — `ai_v12_19_ladder_lambda09` @ `step_4000032`**: 16999 states / 569 battles / 10 opponents / 182 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 7 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (2 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

**control — `ai_v12_16_ladder_ctrl10M_c` @ `step_4000032`**: 14875 states / 548 battles / 9 opponents / 182 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 12 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors only — this cycle has no sentinel opponents

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0768 | +0.1657 | **-0.0889** | [-0.2694, +0.1058] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 40/28/2 | +0.1340 | +0.2271 | **-0.0930** | [-0.2270, +0.0302] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0770 | -0.0634 | **-0.0136** | [-0.0441, +0.0191] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0391) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.3643 | +0.6841 | **-0.3198** | [-0.5377, -0.0255] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3898) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 40/28/2 | +0.3884 | +0.6529 | **-0.2645** | [-0.4485, -0.0143] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0530 | -0.0240 | **-0.0290** | [-0.0611, +0.0045] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0480) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0261 | +0.0147 | **+0.0115** | [+0.0009, +0.0227] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 40/28/2 | +0.0804 | +0.0224 | **+0.0580** | [-0.0069, +0.1253] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 40/28/2 | +0.0448 | +0.0490 | **-0.0041** | [-0.0650, +0.0602] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0312) |

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 40/28/2 (547 battles, 344 decoder battles, 21 seeds) | +0.1340 [+0.2239, +0.2283] | +0.2271 | **-0.0930** | [-0.2270, +0.0302] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 40/28/2 (547 battles, 344 decoder battles, 21 seeds) | +0.1340 [+0.2239, +0.2283] | +0.2271 | **-0.0930** | [-0.2270, +0.0302] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1340 | +0.2267 | **-0.0927** | [-0.2219, +0.0247] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 40/28/2 (547 battles, 344 decoder battles, 21 seeds) | +0.3884 [+0.6525, +0.6535] | +0.6529 | **-0.2645** | [-0.4485, -0.0143] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 40/28/2 (547 battles, 344 decoder battles, 21 seeds) | +0.3884 [+0.6525, +0.6535] | +0.6529 | **-0.2645** | [-0.4485, -0.0143] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.3884 | +0.6529 | **-0.2645** | [-0.4514, -0.0259] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 40/28/2 (547 battles, 344 decoder battles, 21 seeds) | +0.0804 [+0.0157, +0.0231] | +0.0224 | **+0.0580** | [-0.0069, +0.1253] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | MATCHED · decoder 40/28/2 (547 battles, 344 decoder battles, 21 seeds) | +0.0804 [+0.0157, +0.0231] | +0.0224 | **+0.0580** | [-0.0069, +0.1253] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0804 | +0.0234 | **+0.0571** | [-0.0070, +0.1255] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 40/28/2 (547 battles, 344 decoder battles, 21 seeds) | +0.0448 [+0.0457, +0.0537] | +0.0490 | **-0.0041** | [-0.0650, +0.0602] | **WITHIN FLOOR** (|delta| <= floor 0.0312) |
| `cond.own_team_r2.all` | MATCHED · decoder 40/28/2 (547 battles, 344 decoder battles, 21 seeds) | +0.0448 [+0.0457, +0.0537] | +0.0490 | **-0.0041** | [-0.0650, +0.0602] | **WITHIN FLOOR** (|delta| <= floor 0.0312) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0448 | +0.0537 | **-0.0089** | [-0.0621, +0.0488] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0768 | [+0.0000, +0.2298] | +0.1657 | [+0.0305, +0.3715] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1340 | [+0.0990, +0.2392] | +0.2267 | [+0.1564, +0.3689] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0770 | [-0.1036, -0.0573] | -0.0634 | [-0.0912, -0.0457] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.3643 | [+0.2264, +0.5839] | +0.6841 | [+0.4971, +0.8677] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.3884 | [+0.2725, +0.5690] | +0.6529 | [+0.5019, +0.7970] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0530 | [-0.0831, -0.0307] | -0.0240 | [-0.0501, -0.0093] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0261 | [+0.0189, +0.0336] | +0.0147 | [+0.0069, +0.0222] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0804 | [+0.0180, +0.1383] | +0.0234 | [-0.0083, +0.0467] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0448 | [+0.0027, +0.0817] | +0.0537 | [+0.0110, +0.0915] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.6057 | [+0.5313, +0.6747] | — | — |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.opp_class_auc.t1` · control: this cycle's roster carries only ONE opponent class, so the class AUC is undefined. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_19_ladder_lambda09` @ step_4000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 3}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0545 [+0.0410, +0.0700] | 0.0618 | -0.0074 | +0.3052 [+0.2515, +0.3542] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0492 [+0.0379, +0.0644] | 0.0337 | +0.0156 | +0.2902 [+0.2311, +0.3383] | ✅ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0793 [+0.0382, +0.1206] | 0.0711 | +0.0082 | +0.3519 [+0.1660, +0.4837] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 385 battles / 6400 rollouts · Brier 0.1100 = REL 0.0041 − RES 0.0356 + UNC 0.1415 + WBV 0.0008 (resid -7.95e-04) · base rate 0.8294 · **resolution is 25.2% of the base-rate cap** · corr(turn,V) +0.0018 vs corr(turn,MC) +0.1255

**control — `ai_v12_16_ladder_ctrl10M_c` @ step_4000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0617 [+0.0479, +0.0772] | 0.0618 | -0.0001 | +0.2924 [+0.2248, +0.3534] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0617 [+0.0479, +0.0772] | 0.0337 | +0.0281 | +0.2924 [+0.2219, +0.3499] | ✅ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 351 battles / 6400 rollouts · Brier 0.1219 = REL 0.0034 − RES 0.0543 + UNC 0.1740 + WBV 0.0009 (resid -1.99e-03) · base rate 0.7757 · **resolution is 31.2% of the base-rate cap** · corr(turn,V) -0.0429 vs corr(turn,MC) -0.1224

## 5. THE LEDGER LINE

```
ai_v12_19_ladder_lambda09 vs ai_v12_16_ladder_ctrl10M_c at 4M: G1 bot Δ -0.0125 [-0.0317, +0.0098] NOT DETECTED · identity bias late Δ -0.0835 [-0.1517, -0.0191] NOT DETECTED · turn-contrast Δ -0.2033 [-0.3278, -0.0851] DETECTED · spread ratio t1-3 Δ -0.0889 [-0.2694, +0.1058] NOT DETECTED · own-team R2 t1 Δ +0.0580 [-0.0069, +0.1253] NOT DETECTED [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_19_ladder_lambda09 --control ai_v12_16_ladder_ctrl10M_c --step 4000032 --on-live use --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M_vs_c

# arm — ai_v12_19_ladder_lambda09  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_16_ladder_ctrl10M_c
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c --step 4000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/gate/critic_gate.md
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09/eval_traces/step_4000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c/eval_traces/step_4000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M_vs_c/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M_vs_c/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
