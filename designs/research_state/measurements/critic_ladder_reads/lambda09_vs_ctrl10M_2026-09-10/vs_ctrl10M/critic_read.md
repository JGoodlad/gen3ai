# CRITIC READ — `ai_v12_19_ladder_lambda09` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v3 at 2026-09-10T04:51:51. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | `step_10000032` | 700 | 0.5% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 204 | 0.5% | 140/140 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | LIVE cycle: 13 opponents (9 scripted bots + 4 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |
| control | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_19_ladder_lambda09` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | **40/37/2** | 700 | 520 / 174 / 6 | 13 |
| control | `ai_v12_11_ladder_ctrl10M` | **8/12/3** | 204 | 96 / 102 / 6 | 12 |

**Frames:** the realized per-opponent caps differ — arm 40/37/2, control 8/12/3 (traced W/L/D per opponent, counted on disk); arm cut to 8/12/2

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm side is subsampled to caps **8/12/2** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json` — 55 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0380 | +0.0187 | **+0.0193** | [-0.0009, +0.0354] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0123 | -0.0198 | **+0.0321** | [-0.0424, +0.1088] | **WITHIN FLOOR** (|delta| <= floor 0.0558) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.0484 | -0.0940 | **+0.0456** | [-0.1793, +0.2428] | **WITHIN FLOOR** (|delta| <= floor 0.0673) |
| skill · `bot` | — | +0.2738 | +0.1372 | **+0.1366** | [-0.0059, +0.2801] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0548 | +0.1152 | **-0.0605** | [-0.5219, +0.0738] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/2 | +0.1735 | -0.0237 | **+0.1971** | [-0.0123, +0.4450] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0597 | +0.0419 | **+0.0179** | [-0.0099, +0.0381] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0380 | +0.0187 | **+0.0193** | [-0.0009, +0.0354] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0826 | +0.0720 | **+0.0106** | [-0.0379, +0.0463] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0123) |
| reliability | `all` | `capture-rate (gauge)` | +0.0083 | +0.0022 | **+0.0061** | [-0.0017, +0.0117] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0084) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0029 | +0.0033 | **-0.0004** | [-0.0103, +0.0050] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0018) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0169 | +0.0025 | **+0.0144** | [-0.0033, +0.0259] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0215) |
| ece | `all` | `capture-rate (gauge)` | +0.0552 | +0.0351 | **+0.0201** | [-0.0210, +0.0420] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0383) |
| ece | `bot` | `capture-rate (gauge)` | +0.0325 | +0.0425 | **-0.0101** | [-0.0513, +0.0197] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0124) |
| ece | `pool` | `capture-rate (gauge)` | +0.0832 | +0.0407 | **+0.0425** | [-0.0325, +0.0764] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0770) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3259 | +0.2668 | **+0.0591** | [-0.0573, +0.1897] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2738 | +0.1372 | **+0.1366** | [-0.0059, +0.2801] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.3494 | +0.3592 | **-0.0099** | [-0.1457, +0.1716] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1448) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.1037 | +0.0917 | **+0.0120** | [-0.0401, +0.0616] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0758) |
| bias V - p_hat | `ALL` | `pop` | +0.0308 | +0.0306 | **+0.0002** | [-0.0398, +0.0402] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0390) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0081 | -0.0418 | **+0.0337** | [+0.0017, +0.0650] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0696) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0831 | +0.0987 | **-0.0156** | [-0.0761, +0.0444] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0599) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0021 | +0.0134 | **-0.0113** | [-0.0602, +0.0377] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0499) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0314 | -0.0703 | **+0.0388** | [-0.0150, +0.0894] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0940) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.1167 | +0.0889 | **+0.0277** | [-0.0275, +0.0859] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0785) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0386 | +0.0498 | **-0.0112** | [-0.0535, +0.0347] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0421) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0004 | -0.0286 | **+0.0290** | [-0.0043, +0.0625] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0518) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.1178 | +0.0818 | **+0.0360** | [-0.1018, +0.1678] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0979) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0577 | +0.0297 | **+0.0280** | [-0.0802, +0.1336] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0487) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0123 | -0.0198 | **+0.0321** | [-0.0424, +0.1088] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0558) |
| bias V - p_hat | `bot` | `raw` | +0.1072 | +0.1259 | **-0.0187** | [-0.0844, +0.0471] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0667) |
| bias V - p_hat | `bot` | `pop` | +0.0133 | +0.0535 | **-0.0402** | [-0.0901, +0.0100] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0657) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0264 | -0.0456 | **+0.0192** | [-0.0167, +0.0534] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0579) |
| bias V - p_hat | `pool` | `raw` | +0.0994 | +0.0277 | **+0.0717** | [-0.0050, +0.1483] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1500) |
| bias V - p_hat | `pool` | `pop` | +0.0535 | -0.0134 | **+0.0669** | [-0.0053, +0.1405] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1049) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0226 | -0.0382 | **+0.0608** | [+0.0002, +0.1165] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0945) |
| Murphy resolution | `ALL` | `raw` | +0.0463 | +0.0364 | **+0.0099** | [-0.0080, +0.0264] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0627 | +0.0650 | **-0.0024** | [-0.0263, +0.0205] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0210) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0384 | +0.0300 | **+0.0085** | [-0.0098, +0.0240] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0091) |
| Murphy skill_score | `ALL` | `raw` | +0.1605 | +0.1172 | **+0.0434** | [-0.0562, +0.1524] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| Murphy skill_score | `ALL` | `pop` | +0.3565 | +0.2892 | **+0.0673** | [-0.0270, +0.1638] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.3194 | +0.2296 | **+0.0898** | [-0.0260, +0.2152] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.2294 | +0.1606 | **+0.0688** | [-0.0125, +0.1418] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.3820 | +0.2935 | **+0.0885** | [-0.0112, +0.1809] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.3329 | +0.2482 | **+0.0847** | [-0.0283, +0.1817] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0158 | +0.0113 | **+0.0045** | [-0.0095, +0.0154] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| Murphy reliability | `ALL` | `pop` | +0.0052 | +0.0022 | **+0.0030** | [-0.0040, +0.0071] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0077) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0021 | +0.0038 | **-0.0017** | [-0.0086, +0.0019] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.0484 | -0.0940 | **+0.0456** | [-0.1793, +0.2428] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0673) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0116 | -0.0990 | **+0.0874** | [-0.1053, +0.2606] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0886) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0264 | +0.0004 | **+0.0260** | [-0.1767, +0.2416] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0835) |

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

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 6080 states / 198 battles / 12 opponents / 76 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 6 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0548 | +0.1152 | **-0.0605** | [-0.5219, +0.0738] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/2 | +0.1831 | +0.3333 | **-0.1502** | [-0.4513, -0.0249] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0764 | -0.0886 | **+0.0122** | [-0.0516, +0.0347] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0391) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.4694 | +0.7895 | **-0.3201** | [-0.7382, -0.0319] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3898) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 8/12/2 | +0.5166 | +0.8053 | **-0.2887** | [-0.6436, +0.0771] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0429 | -0.0211 | **-0.0218** | [-0.0700, +0.0152] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0480) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0181 | +0.0098 | **+0.0084** | [-0.0033, +0.0208] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 8/12/2 | +0.1735 | -0.0237 | **+0.1971** | [-0.0123, +0.4450] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 8/12/2 | +0.1392 | -0.0098 | **+0.1490** | [-0.0137, +0.3081] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 8/12/2 | +0.4484 | +0.4619 | **-0.0135** | [-0.1453, +0.1246] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0773) |

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 8/12/2 (230 battles, 79 decoder battles, 21 seeds) | +0.1831 [+0.1241, +0.2317] | +0.3333 | **-0.1502** | [-0.4513, -0.0249] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 8/12/2 (230 battles, 79 decoder battles, 21 seeds) | +0.1831 [+0.1241, +0.2317] | +0.3333 | **-0.1502** | [-0.4513, -0.0249] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.0966 | +0.3333 | **-0.2367** | [-0.5447, -0.1401] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 8/12/2 (230 battles, 79 decoder battles, 21 seeds) | +0.5166 [+0.4546, +0.6293] | +0.8053 | **-0.2887** | [-0.6436, +0.0771] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 8/12/2 (230 battles, 79 decoder battles, 21 seeds) | +0.5166 [+0.4546, +0.6293] | +0.8053 | **-0.2887** | [-0.6436, +0.0771] | **WITHIN FLOOR** (|delta| <= floor 0.3852) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.4601 | +0.8053 | **-0.3452** | [-0.7107, -0.0872] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 8/12/2 (230 battles, 79 decoder battles, 21 seeds) | +0.1735 [-0.0336, +0.3117] | -0.0237 | **+0.1971** | [-0.0123, +0.4450] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | MATCHED · decoder 8/12/2 (230 battles, 79 decoder battles, 21 seeds) | +0.1735 [-0.0336, +0.3117] | -0.0237 | **+0.1971** | [-0.0123, +0.4450] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.1420 | -0.0237 | **+0.1657** | [+0.0899, +0.2835] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 8/12/2 (230 battles, 79 decoder battles, 21 seeds) | +0.1392 [-0.0131, +0.3181] | -0.0098 | **+0.1490** | [-0.0137, +0.3081] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | MATCHED · decoder 8/12/2 (230 battles, 79 decoder battles, 21 seeds) | +0.1392 [-0.0131, +0.3181] | -0.0098 | **+0.1490** | [-0.0137, +0.3081] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.1619 | -0.0098 | **+0.1717** | [+0.0900, +0.2694] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 8/12/2 (230 battles, 79 decoder battles, 21 seeds) | +0.4484 [+0.3677, +0.6006] | +0.4619 | **-0.0135** | [-0.1453, +0.1246] | **WITHIN FLOOR** (|delta| <= floor 0.0773) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 8/12/2 (230 battles, 79 decoder battles, 21 seeds) | +0.4484 [+0.3677, +0.6006] | +0.4619 | **-0.0135** | [-0.1453, +0.1246] | **WITHIN FLOOR** (|delta| <= floor 0.0773) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.4714 | +0.4619 | **+0.0095** | [-0.0982, +0.1138] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0548 | [+0.0228, +0.1640] | +0.1152 | [+0.0000, +0.6098] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0966 | [+0.0742, +0.1748] | +0.3333 | [+0.2703, +0.6582] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0764 | [-0.1036, -0.0565] | -0.0886 | [-0.1059, -0.0353] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.4694 | [+0.3145, +0.6675] | +0.7895 | [+0.5725, +1.1703] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.4601 | [+0.3250, +0.6238] | +0.8053 | [+0.6030, +1.1480] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0429 | [-0.0726, -0.0226] | -0.0211 | [-0.0534, +0.0144] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0181 | [+0.0111, +0.0255] | +0.0098 | [-0.0006, +0.0193] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.1420 | [+0.0609, +0.2204] | -0.0237 | [-0.1122, -0.0067] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.1619 | [+0.0929, +0.2259] | -0.0098 | [-0.0867, +0.0421] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.4714 | [+0.4227, +0.5179] | +0.4619 | [+0.3675, +0.5582] |

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

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 185 battles / 6400 rollouts · Brier 0.0930 = REL 0.0038 − RES 0.0300 + UNC 0.1208 + WBV 0.0008 (resid -2.34e-03) · base rate 0.8595 · **resolution is 24.8% of the base-rate cap** · corr(turn,V) -0.1227 vs corr(turn,MC) -0.0286

## 5. THE LEDGER LINE

```
ai_v12_19_ladder_lambda09 vs ai_v12_11_ladder_ctrl10M at 10M: G1 bot Δ +0.0193 [-0.0009, +0.0354] NOT DETECTED · identity bias late Δ +0.0321 [-0.0424, +0.1088] WITHIN FLOOR · turn-contrast Δ +0.0456 [-0.1793, +0.2428] WITHIN FLOOR · spread ratio t1-3 Δ -0.0605 [-0.5219, +0.0738] NOT DETECTED · own-team R2 t1 Δ +0.1971 [-0.0123, +0.4450] NOT DETECTED [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_19_ladder_lambda09 --control ai_v12_11_ladder_ctrl10M --step 10000032 --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M

# arm — ai_v12_19_ladder_lambda09
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09 --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09 --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate/critic_gate.md
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_lambda10M_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
