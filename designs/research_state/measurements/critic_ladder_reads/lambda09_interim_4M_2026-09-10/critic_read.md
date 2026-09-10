# CRITIC READ — `ai_v12_19_ladder_lambda09` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v3 at 2026-09-10T01:45:23. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | `step_4000032` | 576 | 0.7% | 150/150 (100.0%) | yes pid 737657,737727 |
| control | `ai_v12_11_ladder_ctrl10M` | `step_4000032` | 177 | 1.1% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | LIVE cycle: 10 opponents (9 scripted bots + 1 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |
| control | LIVE cycle: 9 opponents (9 scripted bots + 0 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_19_ladder_lambda09` → `step_4000032`: PINNED by --step 4000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_4000032`: PINNED by --step 4000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | **40/28/2** | 576 | 400 / 169 / 7 | 10 |
| control | `ai_v12_11_ladder_ctrl10M` | **8/12/3** | 177 | 72 / 95 / 10 | 9 |

**Frames:** the realized per-opponent caps differ — arm 40/28/2, control 8/12/3 (traced W/L/D per opponent, counted on disk); arm cut to 8/12/2

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm side is subsampled to caps **8/12/2** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json` — 55 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0492 | +0.0726 | **-0.0233** | [-0.0521, +0.0025] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0678 | -0.0047 | **-0.0631** | [-0.1302, -0.0100] | **NOT DETECTED** (CI does not clear the floor 0.0558) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.1237 | -0.0302 | **-0.0935** | [-0.2480, +0.0273] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2902 | +0.3455 | **-0.0553** | [-0.1537, +0.0615] | **WITHIN FLOOR** (|delta| <= floor 0.0642) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0768 | +0.0649 | **+0.0119** | [-0.4061, +0.1655] | **WITHIN FLOOR** (|delta| <= floor 0.0281) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/2 | -0.0831 | +0.0031 | **-0.0862** | [-1.8503, +0.1363] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0545 | +0.0726 | **-0.0181** | [-0.0496, +0.0084] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0492 | +0.0726 | **-0.0233** | [-0.0521, +0.0025] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0023 | +0.0010 | **+0.0013** | [-0.0063, +0.0033] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0084) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0013 | +0.0010 | **+0.0003** | [-0.0067, +0.0026] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0018) |
| ece | `all` | `capture-rate (gauge)` | +0.0345 | +0.0278 | **+0.0067** | [-0.0372, +0.0242] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0383) |
| ece | `bot` | `capture-rate (gauge)` | +0.0301 | +0.0278 | **+0.0023** | [-0.0384, +0.0208] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0124) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3052 | +0.3455 | **-0.0403** | [-0.1500, +0.0766] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2902 | +0.3455 | **-0.0553** | [-0.1537, +0.0615] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0642) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0589 | +0.0920 | **-0.0331** | [-0.0921, +0.0256] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0758) |
| bias V - p_hat | `ALL` | `pop` | -0.0107 | +0.0216 | **-0.0323** | [-0.0803, +0.0168] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0390) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0477 | -0.0558 | **+0.0081** | [-0.0296, +0.0449] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0696) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.1027 | +0.0741 | **+0.0286** | [-0.0460, +0.1012] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0599) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0225 | +0.0121 | **+0.0104** | [-0.0555, +0.0745] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0499) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0248 | -0.0904 | **+0.0656** | [-0.0042, +0.1286] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0940) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0563 | +0.0931 | **-0.0369** | [-0.1171, +0.0483] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0785) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0185 | +0.0148 | **-0.0334** | [-0.0954, +0.0292] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0421) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0558 | -0.0447 | **-0.0111** | [-0.0597, +0.0359] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0518) |
| bias V - p_hat | `late (turn>=25)` | `raw` | -0.0104 | +0.1199 | **-0.1302** | [-0.2485, -0.0095] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0979) |
| bias V - p_hat | `late (turn>=25)` | `pop` | -0.0478 | +0.0535 | **-0.1013** | [-0.1962, -0.0063] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0487) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0678 | -0.0047 | **-0.0631** | [-0.1302, -0.0100] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0558) |
| bias V - p_hat | `bot` | `raw` | +0.0859 | +0.0920 | **-0.0062** | [-0.0658, +0.0563] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0667) |
| bias V - p_hat | `bot` | `pop` | +0.0092 | +0.0216 | **-0.0124** | [-0.0627, +0.0387] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0657) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0362 | -0.0558 | **+0.0196** | [-0.0205, +0.0591] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0579) |
| Murphy resolution | `ALL` | `raw` | +0.0376 | +0.0489 | **-0.0113** | [-0.0317, +0.0069] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0538 | +0.0735 | **-0.0197** | [-0.0451, +0.0046] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0210) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0356 | +0.0573 | **-0.0217** | [-0.0424, -0.0019] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0091) |
| Murphy skill_score | `ALL` | `raw` | +0.1460 | +0.1226 | **+0.0234** | [-0.0921, +0.1415] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| Murphy skill_score | `ALL` | `pop` | +0.2878 | +0.2896 | **-0.0017** | [-0.1096, +0.1090] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0203) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2228 | +0.2976 | **-0.0748** | [-0.1885, +0.0447] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1799 | +0.1970 | **-0.0172** | [-0.1019, +0.0601] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0345) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2917 | +0.2987 | **-0.0070** | [-0.1118, +0.0910] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0365) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2519 | +0.3259 | **-0.0740** | [-0.1772, +0.0270] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0076 | +0.0189 | **-0.0113** | [-0.0285, +0.0011] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| Murphy reliability | `ALL` | `pop` | +0.0011 | +0.0028 | **-0.0017** | [-0.0090, +0.0012] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0077) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0041 | +0.0053 | **-0.0012** | [-0.0086, +0.0041] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0017) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.1237 | -0.0302 | **-0.0935** | [-0.2480, +0.0273] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0884 | -0.0143 | **-0.0741** | [-0.2093, +0.0336] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0886) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0534 | +0.0760 | **-0.1294** | [-0.2514, -0.0256] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0835) |

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

**control — `ai_v12_11_ladder_ctrl10M` @ `step_4000032`**: 5028 states / 167 battles / 9 opponents / 75 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 10 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors only — this cycle has no sentinel opponents

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0768 | +0.0649 | **+0.0119** | [-0.4061, +0.1655] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0281) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/2 | +0.2564 | +0.2855 | **-0.0291** | [-0.2628, +0.1782] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0770 | -0.0880 | **+0.0110** | [-0.0443, +0.0398] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0391) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.3643 | +0.6742 | **-0.3099** | [-0.7820, +0.0098] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3898) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 8/12/2 | +0.4480 | +0.7481 | **-0.3001** | [-0.6399, +0.1413] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0530 | -0.0306 | **-0.0224** | [-0.0779, +0.0161] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0480) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0261 | +0.0175 | **+0.0086** | [-0.0058, +0.0234] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 8/12/2 | -0.0831 | +0.0031 | **-0.0862** | [-1.8503, +0.1363] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 8/12/2 | -0.0922 | -0.0364 | **-0.0558** | [-0.4559, +0.2589] | 2000 | **NOT DETECTED** (CI covers zero) |

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 8/12/2 (185 battles, 33 decoder battles, 21 seeds) | +0.2564 [+0.1898, +0.3634] | +0.2855 | **-0.0291** | [-0.2628, +0.1782] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 11/16/2 (241 battles, 68 decoder battles, 21 seeds) | +0.2223 [+0.1286, +0.3248] | +0.2855 | **-0.0632** | [-0.3086, +0.1387] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1340 | +0.2855 | **-0.1515** | [-0.3895, -0.0194] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 8/12/2 (185 battles, 33 decoder battles, 21 seeds) | +0.4480 [+0.3179, +0.6314] | +0.7481 | **-0.3001** | [-0.6399, +0.1413] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 11/16/2 (241 battles, 68 decoder battles, 21 seeds) | +0.4676 [+0.3648, +0.5880] | +0.7481 | **-0.2806** | [-0.6662, +0.0581] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.3884 | +0.7481 | **-0.3597** | [-0.7477, -0.0917] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 8/12/2 (185 battles, 33 decoder battles, 21 seeds) | -0.0831 [-0.4789, +0.3075] | +0.0031 | **-0.0862** | [-1.8503, +0.1363] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | MATCHED · decoder 11/16/2 (241 battles, 68 decoder battles, 21 seeds) | -0.0372 [-0.1075, +0.2433] | +0.0031 | **-0.0403** | [-0.1738, +0.0806] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0804 | +0.0031 | **+0.0773** | [+0.0115, +0.1777] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 8/12/2 (185 battles, 33 decoder battles, 21 seeds) | -0.0922 [-0.3283, +0.1926] | -0.0364 | **-0.0558** | [-0.4559, +0.2589] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | MATCHED · decoder 11/16/2 (241 battles, 68 decoder battles, 21 seeds) | -0.0428 [-0.1129, +0.2698] | -0.0364 | **-0.0064** | [-0.1428, +0.1416] | **WITHIN FLOOR** (|delta| <= floor 0.0312) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0448 | -0.0364 | **+0.0813** | [+0.0096, +0.2064] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0768 | [+0.0000, +0.2298] | +0.0649 | [+0.0000, +0.5133] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1340 | [+0.0990, +0.2392] | +0.2855 | [+0.2010, +0.5351] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0770 | [-0.1036, -0.0573] | -0.0880 | [-0.1113, -0.0439] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.3643 | [+0.2264, +0.5839] | +0.6742 | [+0.4250, +1.1536] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.3884 | [+0.2725, +0.5690] | +0.7481 | [+0.5352, +1.1322] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0530 | [-0.0831, -0.0307] | -0.0306 | [-0.0651, +0.0135] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0261 | [+0.0189, +0.0336] | +0.0175 | [+0.0044, +0.0288] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0804 | [+0.0180, +0.1383] | +0.0031 | [-0.0793, +0.0380] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0448 | [+0.0027, +0.0817] | -0.0364 | [-0.1572, +0.0210] |
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

**control — `ai_v12_11_ladder_ctrl10M` @ step_4000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0726 [+0.0490, +0.1014] | 0.0618 | +0.0107 | +0.3455 [+0.2317, +0.4318] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0726 [+0.0490, +0.1014] | 0.0337 | +0.0389 | +0.3455 [+0.2495, +0.4300] | ✅ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 162 battles / 6400 rollouts · Brier 0.1236 = REL 0.0053 − RES 0.0573 + UNC 0.1759 + WBV 0.0008 (resid -1.08e-03) · base rate 0.7722 · **resolution is 32.6% of the base-rate cap** · corr(turn,V) -0.1528 vs corr(turn,MC) -0.1226

## 5. THE LEDGER LINE

```
ai_v12_19_ladder_lambda09 vs ai_v12_11_ladder_ctrl10M at 4M: G1 bot Δ -0.0233 [-0.0521, +0.0025] NOT DETECTED · identity bias late Δ -0.0631 [-0.1302, -0.0100] NOT DETECTED · turn-contrast Δ -0.0935 [-0.2480, +0.0273] NOT DETECTED · spread ratio t1-3 Δ +0.0119 [-0.4061, +0.1655] WITHIN FLOOR · own-team R2 t1 Δ -0.0862 [-1.8503, +0.1363] NOT DETECTED [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_19_ladder_lambda09 --control ai_v12_11_ladder_ctrl10M --step 4000032 --on-live use --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M

# arm — ai_v12_19_ladder_lambda09
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09 --step 4000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09 --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M --step 4000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate/critic_gate.md
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09/eval_traces/step_4000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_4000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda_4M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
