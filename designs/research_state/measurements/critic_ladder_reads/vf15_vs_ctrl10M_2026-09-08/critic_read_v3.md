# CRITIC READ — `ai_v12_10_ladder_vf15` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v3 at 2026-09-09T10:28:01. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_10_ladder_vf15` | `step_10000032` | 205 | 0.7% | 141/142 (99.3%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 204 | 0.5% | 140/140 (100.0%) | no |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_10_ladder_vf15` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_10_ladder_vf15` | **8/12/2** | 205 | 96 / 101 / 8 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **8/12/3** | 204 | 96 / 102 / 6 | 12 |

**Frames:** both sides carry the SAME realized cap (8/12/2) — the frames are already equalised and nothing is subsampled

> ✅ **SYMMETRIC.** Both sides carry the same realized cap, so every conditioning row is read AS TRACED and this report is what the tool printed before quota matching existed. Residual per-opponent differences below the cap are the two runs' own outcome mixes, not a selection asymmetry — matching them would shrink both frames for no power gain.

> 🚨 **NO REPLICATE FLOOR — deltas are NOT DETECTED unless their CI clears zero, and never DETECTED against a floor.** The ladder's replicate floor is the control-vs-control difference and does not exist until a second control replicate does. Every DETECTED below is a detection against ZERO and carries that qualifier.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0327 | +0.0187 | **+0.0140** | [-0.0080, +0.0389] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0288 | -0.0198 | **+0.0486** | [-0.0305, +0.1311] | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.0694 | -0.0940 | **+0.1634** | [-0.0807, +0.3733] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2591 | +0.1372 | **+0.1219** | [-0.0681, +0.2941] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.2515 | +0.1152 | **+0.1362** | [-0.3152, +0.4826] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | as traced (frames equal) | -0.0466 | -0.0237 | **-0.0229** | [-0.2312, +0.0917] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0507 | +0.0419 | **+0.0088** | [-0.0207, +0.0384] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0327 | +0.0187 | **+0.0140** | [-0.0080, +0.0389] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.1094 | +0.0720 | **+0.0374** | [-0.0253, +0.0861] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0047 | +0.0022 | **+0.0024** | [-0.0047, +0.0137] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0013 | +0.0033 | **-0.0020** | [-0.0100, +0.0066] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0358 | +0.0025 | **+0.0333** | [+0.0065, +0.0682] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |
| ece | `all` | `capture-rate (gauge)` | +0.0337 | +0.0351 | **-0.0014** | [-0.0399, +0.0370] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `bot` | `capture-rate (gauge)` | +0.0143 | +0.0425 | **-0.0282** | [-0.0598, +0.0175] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `pool` | `capture-rate (gauge)` | +0.1008 | +0.0407 | **+0.0601** | [-0.0224, +0.1156] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3277 | +0.2668 | **+0.0609** | [-0.0772, +0.2024] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2591 | +0.1372 | **+0.1219** | [-0.0681, +0.2941] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.4133 | +0.3592 | **+0.0541** | [-0.1163, +0.2317] | 400 | **NOT DETECTED** (CI covers zero) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.1761 | +0.0917 | **+0.0845** | [+0.0261, +0.1425] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `ALL` | `pop` | +0.0986 | +0.0306 | **+0.0680** | [+0.0206, +0.1184] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | +0.0065 | -0.0418 | **+0.0484** | [+0.0139, +0.0815] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.1197 | +0.0987 | **+0.0211** | [-0.0407, +0.0853] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0576 | +0.0134 | **+0.0442** | [-0.0111, +0.0997] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0134 | -0.0703 | **+0.0568** | [-0.0023, +0.1133] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.1954 | +0.0889 | **+0.1064** | [+0.0331, +0.1759] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.1057 | +0.0498 | **+0.0560** | [-0.0030, +0.1158] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0161 | -0.0286 | **+0.0447** | [+0.0112, +0.0829] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.2314 | +0.0818 | **+0.1496** | [-0.0016, +0.2857] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.1408 | +0.0297 | **+0.1111** | [-0.0120, +0.2287] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0288 | -0.0198 | **+0.0486** | [-0.0305, +0.1311] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `bot` | `raw` | +0.2178 | +0.1259 | **+0.0918** | [+0.0191, +0.1585] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `bot` | `pop` | +0.1291 | +0.0535 | **+0.0756** | [+0.0147, +0.1340] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `bot` | `ipw` | +0.0101 | -0.0456 | **+0.0557** | [+0.0178, +0.0943] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `pool` | `raw` | +0.1037 | +0.0277 | **+0.0760** | [-0.0058, +0.1546] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `pool` | `pop` | +0.0598 | -0.0134 | **+0.0731** | [-0.0043, +0.1506] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0047 | -0.0382 | **+0.0428** | [-0.0168, +0.0993] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `raw` | +0.0368 | +0.0364 | **+0.0004** | [-0.0173, +0.0159] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0673 | +0.0650 | **+0.0023** | [-0.0199, +0.0228] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0343 | +0.0300 | **+0.0043** | [-0.0148, +0.0204] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.0045 | +0.1172 | **-0.1126** | [-0.2294, +0.0073] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `pop` | +0.2353 | +0.2892 | **-0.0539** | [-0.1561, +0.0470] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2924 | +0.2296 | **+0.0628** | [-0.0596, +0.1892] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1570 | +0.1606 | **-0.0036** | [-0.0815, +0.0653] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.3021 | +0.2935 | **+0.0086** | [-0.0851, +0.0932] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.3048 | +0.2482 | **+0.0566** | [-0.0588, +0.1524] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0367 | +0.0113 | **+0.0254** | [+0.0064, +0.0445] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| Murphy reliability | `ALL` | `pop` | +0.0153 | +0.0022 | **+0.0131** | [+0.0030, +0.0246] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0014 | +0.0038 | **-0.0024** | [-0.0093, +0.0019] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.0694 | -0.0940 | **+0.1634** | [-0.0807, +0.3733] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | +0.0080 | -0.0990 | **+0.1070** | [-0.1035, +0.2964] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0164 | +0.0004 | **+0.0161** | [-0.2028, +0.2402] | 4000 | **NOT DETECTED** (CI covers zero) |

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

**arm — `ai_v12_10_ladder_vf15` @ `step_10000032`**: 6161 states / 197 battles / 12 opponents / 91 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 8 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 6080 states / 198 battles / 12 opponents / 76 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 6 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.2515 | +0.1152 | **+0.1362** | [-0.3152, +0.4826] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | as traced (frames equal) | +0.3942 | +0.3333 | **+0.0609** | [-0.2650, +0.3167] | 2000 | **NOT DETECTED** (CI covers zero) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0389 | -0.0886 | **+0.0497** | [-0.0115, +0.0789] | 2000 | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5345 | +0.7895 | **-0.2549** | [-0.6704, +0.4634] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `all states` | as traced (frames equal) | +0.6293 | +0.8053 | **-0.1760** | [-0.5588, +0.3676] | 2000 | **NOT DETECTED** (CI covers zero) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0242 | -0.0211 | **-0.0031** | [-0.0463, +0.0440] | 2000 | **NOT DETECTED** (CI covers zero) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0142 | +0.0098 | **+0.0044** | [-0.0090, +0.0168] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | as traced (frames equal) | -0.0466 | -0.0237 | **-0.0229** | [-0.2312, +0.0917] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | as traced (frames equal) | -0.0093 | -0.0098 | **+0.0005** | [-0.1135, +0.1285] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | as traced (frames equal) | +0.5963 | +0.4619 | **+0.1344** | [-0.0214, +0.2857] | 2000 | **NOT DETECTED** (CI covers zero) |

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.2515 | [+0.1058, +0.7417] | +0.1152 | [+0.0000, +0.6098] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.3942 | [+0.2723, +0.7052] | +0.3333 | [+0.2703, +0.6582] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0389 | [-0.0654, -0.0105] | -0.0886 | [-0.1059, -0.0353] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5345 | [+0.3006, +1.2392] | +0.7895 | [+0.5725, +1.1703] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.6293 | [+0.4144, +1.1567] | +0.8053 | [+0.6030, +1.1480] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0242 | [-0.0524, +0.0103] | -0.0211 | [-0.0534, +0.0144] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0142 | [+0.0050, +0.0222] | +0.0098 | [-0.0006, +0.0193] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | -0.0466 | [-0.2693, +0.0356] | -0.0237 | [-0.1122, -0.0067] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | -0.0093 | [-0.1084, +0.0969] | -0.0098 | [-0.0867, +0.0421] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5963 | [+0.4756, +0.7195] | +0.4619 | [+0.3675, +0.5582] |

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_10_ladder_vf15` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0507 [+0.0334, +0.0770] | 0.0618 | -0.0112 | +0.3277 [+0.2500, +0.3918] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0327 [+0.0203, +0.0583] | 0.0337 | -0.0009 | +0.2591 [+0.1340, +0.3623] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.1094 [+0.0606, +0.1517] | 0.0711 | +0.0383 | +0.4133 [+0.2852, +0.4998] | ❌ | ❌ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 178 battles / 6400 rollouts · Brier 0.0796 = REL 0.0014 − RES 0.0343 + UNC 0.1125 + WBV 0.0008 (resid -8.50e-04) · base rate 0.8708 · **resolution is 30.5% of the base-rate cap** · corr(turn,V) -0.1905 vs corr(turn,MC) -0.2598

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 185 battles / 6400 rollouts · Brier 0.0930 = REL 0.0038 − RES 0.0300 + UNC 0.1208 + WBV 0.0008 (resid -2.34e-03) · base rate 0.8595 · **resolution is 24.8% of the base-rate cap** · corr(turn,V) -0.1227 vs corr(turn,MC) -0.0286

## 5. THE LEDGER LINE

```
ai_v12_10_ladder_vf15 vs ai_v12_11_ladder_ctrl10M at 10M: G1 bot Δ +0.0140 [-0.0080, +0.0389] NOT DETECTED · identity bias late Δ +0.0486 [-0.0305, +0.1311] NOT DETECTED · turn-contrast Δ +0.1634 [-0.0807, +0.3733] NOT DETECTED · spread ratio t1-3 Δ +0.1362 [-0.3152, +0.4826] NOT DETECTED · own-team R2 t1 Δ -0.0229 [-0.2312, +0.0917] NOT DETECTED
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_10_ladder_vf15 --control ai_v12_11_ladder_ctrl10M --step 10000032 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/reads_v3/vf15

# arm — ai_v12_10_ladder_vf15  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_10_ladder_vf15` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_10_ladder_vf15/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/reads_v3/vf15/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/reads_v3/vf15/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
