# CRITIC READ — `ai_v12_19_ladder_lambda09` vs `ai_v12_15_ladder_ctrl10M_b`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v3 at 2026-09-10T02:13:43. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | `step_4000032` | 576 | 0.7% | 150/150 (100.0%) | yes pid 737657,737727 |
| control | `ai_v12_15_ladder_ctrl10M_b` | `step_4000032` | 591 | 1.1% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | LIVE cycle: 10 opponents (9 scripted bots + 1 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |
| control | LIVE cycle: 9 opponents (9 scripted bots + 0 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_19_ladder_lambda09` → `step_4000032`: PINNED by --step 4000032
- **control** `ai_v12_15_ladder_ctrl10M_b` → `step_4000032`: PINNED by --step 4000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | **40/28/2** | 576 | 400 / 169 / 7 | 10 |
| control | `ai_v12_15_ladder_ctrl10M_b` | **40/37/3** | 591 | 360 / 221 / 10 | 9 |

**Frames:** the realized per-opponent caps differ — arm 40/28/2, control 40/37/3 (traced W/L/D per opponent, counted on disk); control cut to 40/28/2

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **40/28/2** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json` — 55 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0492 | +0.0694 | **-0.0202** | [-0.0377, +0.0016] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0678 | +0.0182 | **-0.0860** | [-0.1415, -0.0291] | **NOT DETECTED** (CI does not clear the floor 0.0558) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.1237 | -0.0416 | **-0.0822** | [-0.2394, +0.0498] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2902 | +0.3380 | **-0.0479** | [-0.1163, +0.0236] | **WITHIN FLOOR** (|delta| <= floor 0.0642) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0768 | +0.0000 | **+0.0768** | [-0.1749, +0.1771] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 40/28/2 | +0.0804 | +0.2013 | **-0.1209** | [-0.2145, -0.0253] | **NOT DETECTED** (CI does not clear the floor 0.0389) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0545 | +0.0694 | **-0.0150** | [-0.0356, +0.0044] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0492 | +0.0694 | **-0.0202** | [-0.0377, +0.0016] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0023 | +0.0005 | **+0.0017** | [-0.0016, +0.0045] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0084) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0013 | +0.0005 | **+0.0008** | [-0.0022, +0.0036] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0018) |
| ece | `all` | `capture-rate (gauge)` | +0.0345 | +0.0179 | **+0.0166** | [-0.0170, +0.0324] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0383) |
| ece | `bot` | `capture-rate (gauge)` | +0.0301 | +0.0179 | **+0.0122** | [-0.0242, +0.0330] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0124) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3052 | +0.3380 | **-0.0329** | [-0.1094, +0.0329] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2902 | +0.3380 | **-0.0479** | [-0.1163, +0.0236] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0642) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0589 | +0.1239 | **-0.0651** | [-0.1153, -0.0153] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0758) |
| bias V - p_hat | `ALL` | `pop` | -0.0107 | +0.0431 | **-0.0538** | [-0.0926, -0.0165] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0390) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0477 | +0.0020 | **-0.0497** | [-0.0845, -0.0162] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0696) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.1027 | +0.1633 | **-0.0606** | [-0.1267, +0.0038] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0225 | +0.0526 | **-0.0301** | [-0.0878, +0.0283] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0499) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0248 | +0.0026 | **-0.0274** | [-0.0828, +0.0263] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0940) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0563 | +0.0693 | **-0.0130** | [-0.0785, +0.0514] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0785) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0185 | +0.0212 | **-0.0397** | [-0.0934, +0.0143] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0421) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0558 | -0.0181 | **-0.0377** | [-0.0905, +0.0126] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0518) |
| bias V - p_hat | `late (turn>=25)` | `raw` | -0.0104 | +0.1294 | **-0.1397** | [-0.2326, -0.0450] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0979) |
| bias V - p_hat | `late (turn>=25)` | `pop` | -0.0478 | +0.0518 | **-0.0996** | [-0.1642, -0.0332] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0487) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0678 | +0.0182 | **-0.0860** | [-0.1415, -0.0291] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0558) |
| bias V - p_hat | `bot` | `raw` | +0.0859 | +0.1239 | **-0.0381** | [-0.0888, +0.0150] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0667) |
| bias V - p_hat | `bot` | `pop` | +0.0092 | +0.0431 | **-0.0339** | [-0.0733, +0.0069] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0657) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0362 | +0.0020 | **-0.0382** | [-0.0741, -0.0013] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0579) |
| Murphy resolution | `ALL` | `raw` | +0.0376 | +0.0470 | **-0.0095** | [-0.0294, +0.0085] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0538 | +0.0724 | **-0.0186** | [-0.0413, +0.0047] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0210) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0356 | +0.0566 | **-0.0210** | [-0.0418, -0.0018] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0091) |
| Murphy skill_score | `ALL` | `raw` | +0.1460 | +0.1242 | **+0.0218** | [-0.0873, +0.1357] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| Murphy skill_score | `ALL` | `pop` | +0.2878 | +0.3216 | **-0.0338** | [-0.1295, +0.0646] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2228 | +0.3014 | **-0.0786** | [-0.1814, +0.0272] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1799 | +0.1980 | **-0.0182** | [-0.1054, +0.0613] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0345) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2917 | +0.3253 | **-0.0336** | [-0.1317, +0.0622] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0365) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2519 | +0.2989 | **-0.0470** | [-0.1436, +0.0494] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0551) |
| Murphy reliability | `ALL` | `raw` | +0.0076 | +0.0195 | **-0.0119** | [-0.0250, -0.0012] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| Murphy reliability | `ALL` | `pop` | +0.0011 | +0.0021 | **-0.0011** | [-0.0058, +0.0018] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0077) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0041 | +0.0005 | **+0.0036** | [-0.0012, +0.0075] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.1237 | -0.0416 | **-0.0822** | [-0.2394, +0.0498] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0884 | -0.0238 | **-0.0647** | [-0.1701, +0.0320] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0886) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0534 | -0.0001 | **-0.0533** | [-0.1655, +0.0510] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0835) |

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

**control — `ai_v12_15_ladder_ctrl10M_b` @ `step_4000032`**: 19859 states / 581 battles / 9 opponents / 178 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 10 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors only — this cycle has no sentinel opponents

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0768 | +0.0000 | **+0.0768** | [-0.1749, +0.1771] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 40/28/2 | +0.1340 | +0.1481 | **-0.0141** | [-0.1452, +0.0877] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0770 | -0.1028 | **+0.0258** | [-0.0163, +0.0542] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0391) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.3643 | +0.5179 | **-0.1535** | [-0.3522, +0.1137] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3898) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 40/28/2 | +0.3884 | +0.5192 | **-0.1308** | [-0.3245, +0.0909] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0530 | -0.0496 | **-0.0034** | [-0.0369, +0.0346] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0480) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0261 | +0.0245 | **+0.0016** | [-0.0100, +0.0128] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 40/28/2 | +0.0804 | +0.2013 | **-0.1209** | [-0.2145, -0.0253] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0389) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 40/28/2 | +0.0448 | +0.1434 | **-0.0986** | [-0.1718, -0.0248] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0312) |

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 40/28/2 (569 battles, 410 decoder battles, 21 seeds) | +0.1340 [+0.1275, +0.1639] | +0.1481 | **-0.0141** | [-0.1452, +0.0877] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 40/28/2 (569 battles, 410 decoder battles, 21 seeds) | +0.1340 [+0.1275, +0.1639] | +0.1481 | **-0.0141** | [-0.1452, +0.0877] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1340 | +0.1447 | **-0.0107** | [-0.1343, +0.0890] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 40/28/2 (569 battles, 410 decoder battles, 21 seeds) | +0.3884 [+0.5027, +0.5368] | +0.5192 | **-0.1308** | [-0.3245, +0.0909] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 40/28/2 (569 battles, 410 decoder battles, 21 seeds) | +0.3884 [+0.5027, +0.5368] | +0.5192 | **-0.1308** | [-0.3245, +0.0909] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.3884 | +0.5169 | **-0.1285** | [-0.3007, +0.0950] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 40/28/2 (569 battles, 410 decoder battles, 21 seeds) | +0.0804 [+0.1759, +0.2256] | +0.2013 | **-0.1209** | [-0.2145, -0.0253] | **NOT DETECTED** (CI does not clear the floor 0.0389) |
| `cond.own_team_r2.t1` | MATCHED · decoder 40/28/2 (569 battles, 410 decoder battles, 21 seeds) | +0.0804 [+0.1759, +0.2256] | +0.2013 | **-0.1209** | [-0.2145, -0.0253] | **NOT DETECTED** (CI does not clear the floor 0.0389) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0804 | +0.2034 | **-0.1230** | [-0.2175, -0.0288] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 40/28/2 (569 battles, 410 decoder battles, 21 seeds) | +0.0448 [+0.1195, +0.1599] | +0.1434 | **-0.0986** | [-0.1718, -0.0248] | **NOT DETECTED** (CI does not clear the floor 0.0312) |
| `cond.own_team_r2.all` | MATCHED · decoder 40/28/2 (569 battles, 410 decoder battles, 21 seeds) | +0.0448 [+0.1195, +0.1599] | +0.1434 | **-0.0986** | [-0.1718, -0.0248] | **NOT DETECTED** (CI does not clear the floor 0.0312) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0448 | +0.1249 | **-0.0801** | [-0.1512, -0.0135] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0768 | [+0.0000, +0.2298] | +0.0000 | [+0.0000, +0.2695] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1340 | [+0.0990, +0.2392] | +0.1447 | [+0.1046, +0.2872] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0770 | [-0.1036, -0.0573] | -0.1028 | [-0.1254, -0.0713] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.3643 | [+0.2264, +0.5839] | +0.5179 | [+0.3748, +0.6878] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.3884 | [+0.2725, +0.5690] | +0.5169 | [+0.3913, +0.6632] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0530 | [-0.0831, -0.0307] | -0.0496 | [-0.0813, -0.0295] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0261 | [+0.0189, +0.0336] | +0.0245 | [+0.0160, +0.0335] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0804 | [+0.0180, +0.1383] | +0.2034 | [+0.1253, +0.2715] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0448 | [+0.0027, +0.0817] | +0.1249 | [+0.0636, +0.1794] |
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

**control — `ai_v12_15_ladder_ctrl10M_b` @ step_4000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0694 [+0.0565, +0.0838] | 0.0618 | +0.0076 | +0.3380 [+0.2757, +0.3889] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0694 [+0.0565, +0.0838] | 0.0337 | +0.0358 | +0.3380 [+0.2850, +0.3874] | ✅ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 359 battles / 6400 rollouts · Brier 0.1323 = REL 0.0005 − RES 0.0566 + UNC 0.1894 + WBV 0.0008 (resid -1.76e-03) · base rate 0.7462 · **resolution is 29.9% of the base-rate cap** · corr(turn,V) -0.0916 vs corr(turn,MC) -0.0501

## 5. THE LEDGER LINE

```
ai_v12_19_ladder_lambda09 vs ai_v12_15_ladder_ctrl10M_b at 4M: G1 bot Δ -0.0202 [-0.0377, +0.0016] NOT DETECTED · identity bias late Δ -0.0860 [-0.1415, -0.0291] NOT DETECTED · turn-contrast Δ -0.0822 [-0.2394, +0.0498] NOT DETECTED · spread ratio t1-3 Δ +0.0768 [-0.1749, +0.1771] NOT DETECTED · own-team R2 t1 Δ -0.1209 [-0.2145, -0.0253] NOT DETECTED [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_19_ladder_lambda09 --control ai_v12_15_ladder_ctrl10M_b --step 4000032 --on-live use --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M_vs_b

# arm — ai_v12_19_ladder_lambda09  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_15_ladder_ctrl10M_b
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b --step 4000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/gate/critic_gate.md
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09/eval_traces/step_4000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b/eval_traces/step_4000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_15_ladder_ctrl10M_b/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M_vs_b/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M_vs_b/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
