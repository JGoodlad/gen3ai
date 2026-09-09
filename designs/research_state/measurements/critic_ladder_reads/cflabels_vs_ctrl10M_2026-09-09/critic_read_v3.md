# CRITIC READ — `ai_v12_12_ladder_cflabels` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v3 at 2026-09-09T10:29:34. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_12_ladder_cflabels` | `step_10000032` | 632 | 0.4% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 204 | 0.5% | 140/140 (100.0%) | no |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_12_ladder_cflabels` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_12_ladder_cflabels` | **40/35/2** | 632 | 480 / 147 / 5 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **8/12/3** | 204 | 96 / 102 / 6 | 12 |

**Frames:** the realized per-opponent caps differ — arm 40/35/2, control 8/12/3 (traced W/L/D per opponent, counted on disk); arm cut to 8/12/2

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm side is subsampled to caps **8/12/2** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> 🚨 **NO REPLICATE FLOOR — deltas are NOT DETECTED unless their CI clears zero, and never DETECTED against a floor.** The ladder's replicate floor is the control-vs-control difference and does not exist until a second control replicate does. Every DETECTED below is a detection against ZERO and carries that qualifier.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0242 | +0.0187 | **+0.0054** | [-0.0121, +0.0191] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0182 | -0.0198 | **+0.0380** | [-0.0382, +0.1122] | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.0988 | -0.0940 | **+0.1928** | [-0.0549, +0.4084] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.1960 | +0.1372 | **+0.0588** | [-0.0912, +0.2310] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0000 | +0.1152 | **-0.1152** | [-0.5230, +0.0879] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/2 | -0.0160 | -0.0237 | **+0.0077** | [-0.1353, +0.0816] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0401 | +0.0419 | **-0.0018** | [-0.0264, +0.0169] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0242 | +0.0187 | **+0.0054** | [-0.0121, +0.0191] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0570 | +0.0720 | **-0.0150** | [-0.0588, +0.0179] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0014 | +0.0022 | **-0.0008** | [-0.0073, +0.0027] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0009 | +0.0033 | **-0.0024** | [-0.0111, +0.0019] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0053 | +0.0025 | **+0.0028** | [-0.0109, +0.0108] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `all` | `capture-rate (gauge)` | +0.0347 | +0.0351 | **-0.0004** | [-0.0447, +0.0301] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `bot` | `capture-rate (gauge)` | +0.0276 | +0.0425 | **-0.0149** | [-0.0562, +0.0244] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `pool` | `capture-rate (gauge)` | +0.0613 | +0.0407 | **+0.0205** | [-0.0468, +0.0617] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2533 | +0.2668 | **-0.0135** | [-0.1313, +0.1246] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.1960 | +0.1372 | **+0.0588** | [-0.0912, +0.2310] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2673 | +0.3592 | **-0.0919** | [-0.2317, +0.0867] | 400 | **NOT DETECTED** (CI covers zero) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0734 | +0.0917 | **-0.0183** | [-0.0668, +0.0292] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `ALL` | `pop` | +0.0225 | +0.0306 | **-0.0081** | [-0.0468, +0.0305] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0016 | -0.0418 | **+0.0403** | [+0.0074, +0.0712] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0244 | +0.0987 | **-0.0743** | [-0.1309, -0.0172] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0178 | +0.0134 | **-0.0312** | [-0.0823, +0.0173] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0429 | -0.0703 | **+0.0274** | [-0.0280, +0.0803] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0976 | +0.0889 | **+0.0087** | [-0.0410, +0.0584] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0485 | +0.0498 | **-0.0013** | [-0.0463, +0.0437] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0229 | -0.0286 | **+0.0514** | [+0.0146, +0.0863] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.1055 | +0.0818 | **+0.0236** | [-0.1145, +0.1530] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0375 | +0.0297 | **+0.0078** | [-0.0970, +0.1117] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0182 | -0.0198 | **+0.0380** | [-0.0382, +0.1122] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `bot` | `raw` | +0.0816 | +0.1259 | **-0.0444** | [-0.1058, +0.0172] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `bot` | `pop` | +0.0162 | +0.0535 | **-0.0373** | [-0.0861, +0.0103] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0129 | -0.0456 | **+0.0327** | [-0.0037, +0.0685] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `pool` | `raw` | +0.0610 | +0.0277 | **+0.0333** | [-0.0400, +0.1016] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `pool` | `pop` | +0.0274 | -0.0134 | **+0.0407** | [-0.0315, +0.1099] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0198 | -0.0382 | **+0.0580** | [-0.0041, +0.1106] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `raw` | +0.0270 | +0.0364 | **-0.0094** | [-0.0259, +0.0058] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0419 | +0.0650 | **-0.0232** | [-0.0449, -0.0021] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0269 | +0.0300 | **-0.0031** | [-0.0197, +0.0113] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1277 | +0.1172 | **+0.0105** | [-0.0894, +0.1215] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `pop` | +0.2743 | +0.2892 | **-0.0149** | [-0.1190, +0.0906] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2335 | +0.2296 | **+0.0040** | [-0.1189, +0.1363] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1491 | +0.1606 | **-0.0115** | [-0.0885, +0.0623] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2693 | +0.2935 | **-0.0242** | [-0.1252, +0.0732] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2305 | +0.2482 | **-0.0177** | [-0.1308, +0.0811] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0063 | +0.0113 | **-0.0050** | [-0.0168, +0.0027] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `pop` | +0.0010 | +0.0022 | **-0.0012** | [-0.0072, +0.0017] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0009 | +0.0038 | **-0.0029** | [-0.0096, +0.0009] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.0988 | -0.0940 | **+0.1928** | [-0.0549, +0.4084] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | +0.0462 | -0.0990 | **+0.1452** | [-0.0548, +0.3224] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0669 | +0.0004 | **+0.0666** | [-0.1440, +0.2901] | 4000 | **NOT DETECTED** (CI covers zero) |

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

**arm — `ai_v12_12_ladder_cflabels` @ `step_10000032`**: 18739 states / 627 battles / 12 opponents / 183 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 5 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 6080 states / 198 battles / 12 opponents / 76 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 6 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0000 | +0.1152 | **-0.1152** | [-0.5230, +0.0879] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/2 | +0.2332 | +0.3333 | **-0.1001** | [-0.3947, +0.0885] | 2000 | **NOT DETECTED** (CI covers zero) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1026 | -0.0886 | **-0.0140** | [-0.0701, +0.0189] | 2000 | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5149 | +0.7895 | **-0.2746** | [-0.6919, +0.0168] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 8/12/2 | +0.5419 | +0.8053 | **-0.2634** | [-0.6307, +0.1166] | 2000 | **NOT DETECTED** (CI covers zero) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0498 | -0.0211 | **-0.0287** | [-0.0762, +0.0104] | 2000 | **NOT DETECTED** (CI covers zero) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0191 | +0.0098 | **+0.0093** | [-0.0026, +0.0222] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 8/12/2 | -0.0160 | -0.0237 | **+0.0077** | [-0.1353, +0.0816] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 8/12/2 | -0.0380 | -0.0098 | **-0.0283** | [-0.3414, +0.0904] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 8/12/2 | +0.5492 | +0.4619 | **+0.0873** | [-0.0616, +0.2356] | 2000 | **NOT DETECTED** (CI covers zero) |

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 8/12/2 (197 battles, 62 decoder battles, 21 seeds) | +0.2332 [+0.1460, +0.2965] | +0.3333 | **-0.1001** | [-0.3947, +0.0885] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 11/16/2 (247 battles, 103 decoder battles, 21 seeds) | +0.1999 [+0.1415, +0.2546] | +0.3333 | **-0.1334** | [-0.4245, +0.0138] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1077 | +0.3333 | **-0.2256** | [-0.5187, -0.1046] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 8/12/2 (197 battles, 62 decoder battles, 21 seeds) | +0.5419 [+0.4518, +0.6102] | +0.8053 | **-0.2634** | [-0.6307, +0.1166] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 11/16/2 (247 battles, 103 decoder battles, 21 seeds) | +0.5178 [+0.4112, +0.6008] | +0.8053 | **-0.2875** | [-0.6464, +0.0354] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.5108 | +0.8053 | **-0.2945** | [-0.6714, -0.0316] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 8/12/2 (197 battles, 62 decoder battles, 21 seeds) | -0.0160 [-0.1010, +0.2530] | -0.0237 | **+0.0077** | [-0.1353, +0.0816] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | MATCHED · decoder 11/16/2 (247 battles, 103 decoder battles, 21 seeds) | +0.0050 [-0.0841, +0.2822] | -0.0237 | **+0.0287** | [-0.0808, +0.1327] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0604 | -0.0237 | **+0.0841** | [+0.0323, +0.1825] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 8/12/2 (197 battles, 62 decoder battles, 21 seeds) | -0.0380 [-0.1845, +0.1214] | -0.0098 | **-0.0283** | [-0.3414, +0.0904] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | MATCHED · decoder 11/16/2 (247 battles, 103 decoder battles, 21 seeds) | -0.0028 [-0.0504, +0.1325] | -0.0098 | **+0.0070** | [-0.0800, +0.0824] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0837 | -0.0098 | **+0.0935** | [+0.0252, +0.1836] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 8/12/2 (197 battles, 62 decoder battles, 21 seeds) | +0.5492 [+0.4085, +0.6243] | +0.4619 | **+0.0873** | [-0.0616, +0.2356] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 11/16/2 (247 battles, 103 decoder battles, 21 seeds) | +0.5522 [+0.4012, +0.6478] | +0.4619 | **+0.0903** | [-0.0487, +0.2219] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5552 | +0.4619 | **+0.0933** | [-0.0209, +0.2022] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0000 | [+0.0000, +0.1913] | +0.1152 | [+0.0000, +0.6098] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1077 | [+0.0916, +0.2179] | +0.3333 | [+0.2703, +0.6582] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1026 | [-0.1232, -0.0700] | -0.0886 | [-0.1059, -0.0353] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5149 | [+0.3773, +0.7051] | +0.7895 | [+0.5725, +1.1703] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.5108 | [+0.3836, +0.6821] | +0.8053 | [+0.6030, +1.1480] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0498 | [-0.0799, -0.0244] | -0.0211 | [-0.0534, +0.0144] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0191 | [+0.0111, +0.0267] | +0.0098 | [-0.0006, +0.0193] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0604 | [+0.0079, +0.1085] | -0.0237 | [-0.1122, -0.0067] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0837 | [+0.0399, +0.1280] | -0.0098 | [-0.0867, +0.0421] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5552 | [+0.5047, +0.6016] | +0.4619 | [+0.3675, +0.5582] |

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_12_ladder_cflabels` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0401 [+0.0317, +0.0508] | 0.0618 | -0.0218 | +0.2533 [+0.2002, +0.3035] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0242 [+0.0168, +0.0337] | 0.0337 | -0.0095 | +0.1960 [+0.1216, +0.2749] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0570 [+0.0427, +0.0756] | 0.0711 | -0.0141 | +0.2673 [+0.1812, +0.3384] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 381 battles / 6400 rollouts · Brier 0.0895 = REL 0.0009 − RES 0.0269 + UNC 0.1168 + WBV 0.0008 (resid -2.03e-03) · base rate 0.8650 · **resolution is 23.0% of the base-rate cap** · corr(turn,V) +0.0111 vs corr(turn,MC) -0.0877

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 185 battles / 6400 rollouts · Brier 0.0930 = REL 0.0038 − RES 0.0300 + UNC 0.1208 + WBV 0.0008 (resid -2.34e-03) · base rate 0.8595 · **resolution is 24.8% of the base-rate cap** · corr(turn,V) -0.1227 vs corr(turn,MC) -0.0286

## 5. THE LEDGER LINE

```
ai_v12_12_ladder_cflabels vs ai_v12_11_ladder_ctrl10M at 10M: G1 bot Δ +0.0054 [-0.0121, +0.0191] NOT DETECTED · identity bias late Δ +0.0380 [-0.0382, +0.1122] NOT DETECTED · turn-contrast Δ +0.1928 [-0.0549, +0.4084] NOT DETECTED · spread ratio t1-3 Δ -0.1152 [-0.5230, +0.0879] NOT DETECTED · own-team R2 t1 Δ +0.0077 [-0.1353, +0.0816] NOT DETECTED [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_12_ladder_cflabels --control ai_v12_11_ladder_ctrl10M --step 10000032 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/reads_v3/cflabels

# arm — ai_v12_12_ladder_cflabels  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_12_ladder_cflabels` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_12_ladder_cflabels/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_cflabels/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_cflabels/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_cflabels/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/reads_v3/cflabels/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/reads_v3/cflabels/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
