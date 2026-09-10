# CRITIC READ — `ai_v12_14_ladder_truevalue` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v4 at 2026-09-10T10:56:58. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_14_ladder_truevalue` | `step_10000032` | 628 | 0.2% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 204 | 0.5% | 140/140 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |
| control | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_14_ladder_truevalue` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_14_ladder_truevalue` | **40/39/1** | 628 | 480 / 146 / 2 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **8/12/3** | 204 | 96 / 102 / 6 | 12 |

**Frames:** the realized per-opponent caps differ — arm 40/39/1, control 8/12/3 (traced W/L/D per opponent, counted on disk); arm cut to 8/12/1

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm side is subsampled to caps **8/12/1** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json` — 68 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0326 | +0.0187 | **+0.0138** | [-0.0044, +0.0315] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0878 | -0.0198 | **+0.1076** | [+0.0255, +0.1875] | **NOT DETECTED** (CI does not clear the floor 0.0558) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.1128 | -0.0940 | **+0.2068** | [-0.0250, +0.4162] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2782 | +0.1372 | **+0.1410** | [-0.0036, +0.2996] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0000 | +0.1152 | **-0.1152** | [-0.5582, +0.0384] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/1 | -0.0179 | -0.0237 | **+0.0058** | [-0.3069, +0.1382] | **WITHIN FLOOR** (|delta| <= floor 0.0389) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0366 | +0.0419 | **-0.0052** | [-0.0334, +0.0147] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0127) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0326 | +0.0187 | **+0.0138** | [-0.0044, +0.0315] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0373 | +0.0720 | **-0.0347** | [-0.0752, +0.0025] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0058 | +0.0022 | **+0.0036** | [-0.0047, +0.0096] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0084) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0021 | +0.0033 | **-0.0012** | [-0.0105, +0.0026] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0018) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0178 | +0.0025 | **+0.0154** | [-0.0025, +0.0336] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0215) |
| ece | `all` | `capture-rate (gauge)` | +0.0646 | +0.0351 | **+0.0295** | [-0.0192, +0.0575] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0383) |
| ece | `bot` | `capture-rate (gauge)` | +0.0270 | +0.0425 | **-0.0155** | [-0.0636, +0.0141] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `pool` | `capture-rate (gauge)` | +0.1244 | +0.0407 | **+0.0836** | [-0.0067, +0.1254] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2134 | +0.2668 | **-0.0534** | [-0.1673, +0.0823] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2782 | +0.1372 | **+0.1410** | [-0.0036, +0.2996] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.1067 | +0.3592 | **-0.2526** | [-0.4061, -0.0679] | 400 | **NOT DETECTED** (CI does not clear the floor 0.1448) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.1483 | +0.0917 | **+0.0566** | [+0.0044, +0.1062] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0758) |
| bias V - p_hat | `ALL` | `pop` | +0.0745 | +0.0306 | **+0.0439** | [+0.0041, +0.0853] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0390) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | +0.0400 | -0.0418 | **+0.0819** | [+0.0477, +0.1159] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0696) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0943 | +0.0987 | **-0.0044** | [-0.0612, +0.0518] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0599) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0286 | +0.0134 | **+0.0151** | [-0.0349, +0.0639] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0499) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | +0.0017 | -0.0703 | **+0.0720** | [+0.0163, +0.1240] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0940) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.1588 | +0.0889 | **+0.0699** | [+0.0189, +0.1204] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0785) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0808 | +0.0498 | **+0.0310** | [-0.0123, +0.0758] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0421) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0423 | -0.0286 | **+0.0708** | [+0.0373, +0.1034] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0518) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.2038 | +0.0818 | **+0.1220** | [-0.0275, +0.2573] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.1230 | +0.0297 | **+0.0934** | [-0.0202, +0.2005] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0878 | -0.0198 | **+0.1076** | [+0.0255, +0.1875] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0558) |
| bias V - p_hat | `bot` | `raw` | +0.1204 | +0.1259 | **-0.0055** | [-0.0731, +0.0604] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0667) |
| bias V - p_hat | `bot` | `pop` | +0.0555 | +0.0535 | **+0.0020** | [-0.0487, +0.0526] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0657) |
| bias V - p_hat ⭐ | `bot` | `ipw` | +0.0220 | -0.0456 | **+0.0676** | [+0.0298, +0.1042] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0579) |
| bias V - p_hat | `pool` | `raw` | +0.1834 | +0.0277 | **+0.1558** | [+0.0819, +0.2247] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.1500) |
| bias V - p_hat | `pool` | `pop` | +0.1196 | -0.0134 | **+0.1330** | [+0.0567, +0.2073] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.1049) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0930 | -0.0382 | **+0.1312** | [+0.0658, +0.1929] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0945) |
| Murphy resolution | `ALL` | `raw` | +0.0305 | +0.0364 | **-0.0059** | [-0.0225, +0.0095] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0079) |
| Murphy resolution | `ALL` | `pop` | +0.0418 | +0.0650 | **-0.0232** | [-0.0454, -0.0013] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0210) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0268 | +0.0300 | **-0.0031** | [-0.0209, +0.0122] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0091) |
| Murphy skill_score | `ALL` | `raw` | +0.0390 | +0.1172 | **-0.0781** | [-0.1808, +0.0343] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `pop` | +0.2205 | +0.2892 | **-0.0688** | [-0.1721, +0.0375] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2075 | +0.2296 | **-0.0220** | [-0.1425, +0.1074] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0496) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1523 | +0.1606 | **-0.0082** | [-0.0844, +0.0652] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0345) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2591 | +0.2935 | **-0.0344** | [-0.1379, +0.0644] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0365) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2199 | +0.2482 | **-0.0283** | [-0.1472, +0.0688] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0551) |
| Murphy reliability | `ALL` | `raw` | +0.0255 | +0.0113 | **+0.0142** | [+0.0001, +0.0264] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| Murphy reliability | `ALL` | `pop` | +0.0082 | +0.0022 | **+0.0060** | [-0.0012, +0.0122] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0077) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0031 | +0.0038 | **-0.0007** | [-0.0079, +0.0046] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0017) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.1128 | -0.0940 | **+0.2068** | [-0.0250, +0.4162] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | +0.0774 | -0.0990 | **+0.1764** | [-0.0276, +0.3612] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.1059 | +0.0004 | **+0.1055** | [-0.1082, +0.3231] | 4000 | **NOT DETECTED** (CI covers zero) |

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

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 6080 states / 198 battles / 12 opponents / 76 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 6 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0000 | +0.1152 | **-0.1152** | [-0.5582, +0.0384] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/1 | +0.1821 | +0.3333 | **-0.1512** | [-0.4381, +0.0249] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1016 | -0.0886 | **-0.0130** | [-0.0733, +0.0156] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0391) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.3209 | +0.7895 | **-0.4685** | [-0.8499, -0.2043] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.3898) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 8/12/1 | +0.3465 | +0.8053 | **-0.4588** | [-0.8015, -0.1711] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.3852) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0690 | -0.0211 | **-0.0479** | [-0.0940, -0.0084] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0480) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0212 | +0.0098 | **+0.0114** | [-0.0006, +0.0235] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 8/12/1 | -0.0179 | -0.0237 | **+0.0058** | [-0.3069, +0.1382] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 8/12/1 | -0.0184 | -0.0098 | **-0.0086** | [-0.1456, +0.1307] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0312) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 8/12/1 | +0.5601 | +0.4619 | **+0.0982** | [-0.0352, +0.2334] | 2000 | **NOT DETECTED** (CI covers zero) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 8/12/1 | +0.0573 | +0.0839 | **-0.0266** | [-0.0710, +0.0122] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 8/12/1 | +0.0448 | +0.0819 | **-0.0371** | [-0.0857, +0.0063] | 2000 | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 8/12/1 | +0.1792 | +0.3973 | **-0.2181** | [-0.2845, -0.0675] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.1848) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/1 | +0.1745 | +0.3573 | **-0.1828** | [-0.2405, -0.0718] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.1740) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 8/12/1 | -0.1314 | -0.0507 | **-0.0807** | [-0.4423, +0.3324] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 8/12/1 | +0.1154 | +0.0270 | **+0.0884** | [-0.3049, +0.6543] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| control · team | 15 | 76 | 104 | 208 | 5.0 | 10.0 | 4–17 |
| control · stratum | 5 | 76 | 104 | 208 | 20.0 | 40.0 | 15–27 |
| control · between-team spread | 15 | 76 | 104 | — | 5.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 8/12/1 | +0.0573 | +0.0839 | **-0.0266** | [-0.0710, +0.0122] | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 8/12/1 | +0.0448 | +0.0819 | **-0.0371** | [-0.0857, +0.0063] | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 8/12/1 | +0.1792 | +0.3973 | **-0.2181** | [-0.2845, -0.0675] | **NOT DETECTED** (CI does not clear the floor 0.1848) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/1 | -0.0179 | -0.0237 | **+0.0058** | [-0.3069, +0.1382] | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 8/12/1 | -0.1314 | -0.0507 | **-0.0807** | [-0.4423, +0.3324] | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 8/12/1 | +0.1154 | +0.0270 | **+0.0884** | [-0.3049, +0.6543] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

The two own-team R² rows are printed with them: the contrast is their difference, and it cannot be read without seeing which end moved.

**Reading of the three signs:** neither as posed — BOTH components are DOWN: less dispersion BETWEEN teams and less discrimination INSIDE one. Note that the spread ratio is an AMPLITUDE while the own-team R^2 is an ALIGNMENT (a monotone decode, invariant to scale), so a head can ORDER its teams better while emitting a smaller between-team spread — read the two together, never the R^2 alone.

> 🚨 **The between-team spread is an AMPLITUDE; the own-team R² is an ALIGNMENT.** The spread ratio compares `sd(mean V per team)` with `sd(that team's win rate)` — how far apart the head's per-team opinions are. The own-team R² is an out-of-fold MONOTONE decode, invariant to scale — whether those opinions are in the right ORDER. The two can move in opposite directions (a head that orders its teams correctly but under-disperses reads R² UP and spread DOWN), so the sign table above is read with both rows in hand and the R² row is never read alone.

> 🚨 **The per-TEAM cells are small and the coarse row is the check on them.** With ~719 teams in the pool a few-thousand-battle frame leaves a handful of episodes per team, and a binned resolution inside a cell that size is largely the binning's own noise — which is *positively* biased, so a small per-team number is evidence of neither reading. The STRATUM row is the same estimator on cells hundreds of episodes deep. **Where the two disagree, believe the stratum row and say so.**

> 🚨 **`cond.own_team_r2.t1_minus_late` is PROVISIONAL and is never labelled DETECTED.** NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | +0.1821 [+0.1129, +0.2549] | +0.3333 | **-0.1512** | [-0.4381, +0.0249] | **WITHIN FLOOR** (|delta| <= floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | +0.1521 [+0.0921, +0.1928] | +0.3333 | **-0.1812** | [-0.4608, -0.0343] | **NOT DETECTED** (CI does not clear the floor 0.1556) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.0632 | +0.3333 | **-0.2701** | [-0.5567, -0.1581] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | +0.3465 [+0.2970, +0.4441] | +0.8053 | **-0.4588** | [-0.8015, -0.1711] | **NOT DETECTED** (CI does not clear the floor 0.3852) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | +0.3470 [+0.2849, +0.3992] | +0.8053 | **-0.4583** | [-0.8007, -0.1634] | **NOT DETECTED** (CI does not clear the floor 0.3852) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.3235 | +0.8053 | **-0.4818** | [-0.8212, -0.2362] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | -0.0179 [-0.0944, +0.2926] | -0.0237 | **+0.0058** | [-0.3069, +0.1382] | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| `cond.own_team_r2.t1` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | -0.0193 [-0.0828, +0.1839] | -0.0237 | **+0.0044** | [-0.0719, +0.0862] | **WITHIN FLOOR** (|delta| <= floor 0.0389) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | -0.0104 | -0.0237 | **+0.0133** | [-0.0184, +0.1032] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | -0.0184 [-0.1586, +0.1872] | -0.0098 | **-0.0086** | [-0.1456, +0.1307] | **WITHIN FLOOR** (|delta| <= floor 0.0312) |
| `cond.own_team_r2.all` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | +0.0011 [-0.0543, +0.1303] | -0.0098 | **+0.0109** | [-0.0776, +0.0954] | **WITHIN FLOOR** (|delta| <= floor 0.0312) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0653 | -0.0098 | **+0.0751** | [+0.0053, +0.1652] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | +0.5601 [+0.4101, +0.6646] | +0.4619 | **+0.0982** | [-0.0352, +0.2334] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | +0.6063 [+0.4773, +0.6781] | +0.4619 | **+0.1444** | [+0.0165, +0.2706] | **NOT DETECTED** (CI does not clear the floor 0.0773) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.6223 | +0.4619 | **+0.1604** | [+0.0516, +0.2667] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | +0.0573 [+0.0339, +0.0779] | +0.0839 | **-0.0266** | [-0.0710, +0.0122] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | +0.0551 [+0.0390, +0.0779] | +0.0839 | **-0.0288** | [-0.0609, +0.0148] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0359 | +0.0839 | **-0.0479** | [-0.0653, -0.0033] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | +0.0448 [+0.0226, +0.0640] | +0.0819 | **-0.0371** | [-0.0857, +0.0063] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | +0.0480 [+0.0267, +0.0726] | +0.0819 | **-0.0340** | [-0.0811, +0.0131] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0280 | +0.0819 | **-0.0540** | [-0.0887, -0.0159] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | +0.1792 [+0.1466, +0.2486] | +0.3973 | **-0.2181** | [-0.2845, -0.0675] | **NOT DETECTED** (CI does not clear the floor 0.1848) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | +0.2078 [+0.1605, +0.3079] | +0.3973 | **-0.1895** | [-0.2909, -0.0521] | **NOT DETECTED** (CI does not clear the floor 0.1848) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.2563 | +0.3973 | **-0.1410** | [-0.2450, -0.0034] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | +0.1745 [+0.1457, +0.2220] | +0.3573 | **-0.1828** | [-0.2405, -0.0718] | **NOT DETECTED** (CI does not clear the floor 0.1740) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | +0.1908 [+0.1578, +0.2434] | +0.3573 | **-0.1665** | [-0.2444, -0.0644] | **WITHIN FLOOR** (|delta| <= floor 0.1740) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.2294 | +0.3573 | **-0.1279** | [-0.2071, -0.0253] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | -0.1314 [-0.3196, +0.1970] | -0.0507 | **-0.0807** | [-0.4423, +0.3324] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | -0.0571 [-0.2423, +0.1060] | -0.0507 | **-0.0064** | [-0.2104, +0.4275] | **WITHIN FLOOR** (|delta| <= floor 0.0405) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0187 | -0.0507 | **+0.0694** | [+0.0218, +0.5191] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 8/12/1 (201 battles, 72 decoder battles, 21 seeds) | +0.1154 [-0.2780, +0.5154] | +0.0270 | **+0.0884** | [-0.3049, +0.6543] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 11/16/1 (246 battles, 105 decoder battles, 21 seeds) | +0.0620 [-0.1517, +0.3053] | +0.0270 | **+0.0350** | [-0.3486, +0.1646] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | -0.0290 | +0.0270 | **-0.0561** | [-0.4584, +0.0066] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0000 | [+0.0000, +0.1345] | +0.1152 | [+0.0000, +0.6098] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0632 | [+0.0610, +0.1597] | +0.3333 | [+0.2703, +0.6582] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1016 | [-0.1259, -0.0724] | -0.0886 | [-0.1059, -0.0353] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.3209 | [+0.2314, +0.4551] | +0.7895 | [+0.5725, +1.1703] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.3235 | [+0.2389, +0.4454] | +0.8053 | [+0.6030, +1.1480] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0690 | [-0.0977, -0.0445] | -0.0211 | [-0.0534, +0.0144] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0212 | [+0.0139, +0.0281] | +0.0098 | [-0.0006, +0.0193] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | -0.0104 | [-0.0394, +0.0131] | -0.0237 | [-0.1122, -0.0067] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0653 | [+0.0156, +0.1106] | -0.0098 | [-0.0867, +0.0421] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.6223 | [+0.5756, +0.6661] | +0.4619 | [+0.3675, +0.5582] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0359 | [+0.0267, +0.0481] | +0.0839 | [+0.0413, +0.1005] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0280 | [+0.0214, +0.0403] | +0.0819 | [+0.0483, +0.1175] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.2563 | [+0.1839, +0.2826] | +0.3973 | [+0.2479, +0.4615] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.2294 | [+0.1774, +0.2491] | +0.3573 | [+0.2470, +0.4098] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0187 | [-0.0317, +0.0614] | -0.0507 | [-0.4948, -0.0286] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | -0.0290 | [-0.0805, +0.0211] | +0.0270 | [-0.0119, +0.4322] |

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

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 185 battles / 6400 rollouts · Brier 0.0930 = REL 0.0038 − RES 0.0300 + UNC 0.1208 + WBV 0.0008 (resid -2.34e-03) · base rate 0.8595 · **resolution is 24.8% of the base-rate cap** · corr(turn,V) -0.1227 vs corr(turn,MC) -0.0286

## 5. THE LEDGER LINE

```
ai_v12_14_ladder_truevalue vs ai_v12_11_ladder_ctrl10M at 10M: G1 bot Δ +0.0138 [-0.0044, +0.0315] NOT DETECTED · identity bias late Δ +0.1076 [+0.0255, +0.1875] NOT DETECTED · turn-contrast Δ +0.2068 [-0.0250, +0.4162] NOT DETECTED · spread ratio t1-3 Δ -0.1152 [-0.5582, +0.0384] NOT DETECTED · own-team R2 t1 Δ +0.0058 [-0.3069, +0.1382] WITHIN FLOOR [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_14_ladder_truevalue --control ai_v12_11_ladder_ctrl10M --step 10000032 --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/replicate_floor_10M.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M

# arm — ai_v12_14_ladder_truevalue
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_14_ladder_truevalue --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_14_ladder_truevalue --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_14_ladder_truevalue` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_14_ladder_truevalue/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_ttv10M_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
