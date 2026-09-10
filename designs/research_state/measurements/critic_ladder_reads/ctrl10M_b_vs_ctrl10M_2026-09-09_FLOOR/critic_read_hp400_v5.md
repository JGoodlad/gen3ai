# CRITIC READ — `ai_v12_15_ladder_ctrl10M_b` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v5 at 2026-09-10T11:15:42. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_15_ladder_ctrl10M_b` | `step_10000032` | 4800 | 0.6% | 149/150 (99.3%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 4800 | 0.5% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 400 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260909, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_15_ladder_ctrl10M_b` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_15_ladder_ctrl10M_b` | **398/186/10** | 4800 | 4032 / 740 / 28 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **400/188/5** | 4800 | 4143 / 634 / 23 | 12 |

**Frames:** the realized per-opponent caps differ — arm 398/186/10, control 400/188/5 (traced W/L/D per opponent, counted on disk); control cut to 398/186/5

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **398/186/5** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> 🚨 **NO REPLICATE FLOOR — deltas are NOT DETECTED unless their CI clears zero, and never DETECTED against a floor.** The ladder's replicate floor is the control-vs-control difference and does not exist until a second control replicate does. Every DETECTED below is a detection against ZERO and carries that qualifier.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0207 | +0.0202 | **+0.0004** | [-0.0070, +0.0073] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0716 | -0.0308 | **+0.1024** | [+0.0503, +0.1553] | **DETECTED** (vs ZERO — NO FLOOR) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.0316 | +0.0124 | **-0.0440** | [-0.2229, +0.1430] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.1962 | +0.1636 | **+0.0327** | [-0.0352, +0.0978] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.1110 | +0.1774 | **-0.0664** | [-0.1069, -0.0307] | **DETECTED** (vs ZERO — NO FLOOR) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 398/186/5 | +0.0224 | +0.0369 | **-0.0146** | [-0.0289, +0.0002] | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.2633 | +1.0719 | **+0.1914** | [+0.0882, +0.2960] | **DETECTED** (vs ZERO — NO FLOOR) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0394 | +0.0395 | **-0.0001** | [-0.0073, +0.0074] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0207 | +0.0202 | **+0.0004** | [-0.0070, +0.0073] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0507 | +0.0544 | **-0.0038** | [-0.0171, +0.0101] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0049 | +0.0013 | **+0.0037** | [+0.0013, +0.0061] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0004 | +0.0036 | **-0.0032** | [-0.0058, -0.0014] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0269 | +0.0022 | **+0.0247** | [+0.0153, +0.0349] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |
| ece | `all` | `capture-rate (gauge)` | +0.0579 | +0.0206 | **+0.0373** | [+0.0193, +0.0514] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |
| ece | `bot` | `capture-rate (gauge)` | +0.0107 | +0.0394 | **-0.0286** | [-0.0420, -0.0119] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |
| ece | `pool` | `capture-rate (gauge)` | +0.1550 | +0.0379 | **+0.1171** | [+0.0800, +0.1447] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2228 | +0.2535 | **-0.0307** | [-0.0681, +0.0093] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.1962 | +0.1636 | **+0.0327** | [-0.0352, +0.0978] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.1122 | +0.2446 | **-0.1324** | [-0.2107, -0.0575] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.1084 | +0.0055 | **+0.1029** | [+0.0732, +0.1327] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `ALL` | `pop` | +0.0344 | -0.0391 | **+0.0735** | [+0.0512, +0.0955] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | +0.0344 | -0.0391 | **+0.0735** | [+0.0512, +0.0955] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0890 | -0.0196 | **+0.1085** | [+0.0649, +0.1525] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0137 | -0.0675 | **+0.0812** | [+0.0488, +0.1135] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | +0.0137 | -0.0675 | **+0.0812** | [+0.0488, +0.1135] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0849 | +0.0143 | **+0.0707** | [+0.0297, +0.1123] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0273 | -0.0228 | **+0.0501** | [+0.0198, +0.0809] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0273 | -0.0228 | **+0.0501** | [+0.0198, +0.0809] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.1675 | +0.0241 | **+0.1434** | [+0.0742, +0.2125] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0716 | -0.0308 | **+0.1024** | [+0.0503, +0.1553] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0716 | -0.0308 | **+0.1024** | [+0.0503, +0.1553] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `bot` | `raw` | +0.0606 | -0.0033 | **+0.0639** | [+0.0278, +0.0999] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `bot` | `pop` | -0.0024 | -0.0473 | **+0.0449** | [+0.0219, +0.0672] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0024 | -0.0473 | **+0.0449** | [+0.0219, +0.0672] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `pool` | `raw` | +0.1803 | +0.0210 | **+0.1594** | [+0.1055, +0.2123] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `pool` | `pop` | +0.1026 | -0.0230 | **+0.1256** | [+0.0788, +0.1749] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.1026 | -0.0230 | **+0.1256** | [+0.0788, +0.1749] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| Murphy resolution | `ALL` | `raw` | +0.0292 | +0.0263 | **+0.0029** | [-0.0089, +0.0150] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0357 | +0.0315 | **+0.0042** | [-0.0104, +0.0190] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0357 | +0.0315 | **+0.0042** | [-0.0104, +0.0190] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.0971 | +0.1567 | **-0.0596** | [-0.1320, +0.0163] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `pop` | +0.2512 | +0.2213 | **+0.0299** | [-0.0602, +0.1314] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2512 | +0.2213 | **+0.0299** | [-0.0602, +0.1314] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1570 | +0.1683 | **-0.0113** | [-0.0784, +0.0513] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2528 | +0.2448 | **+0.0080** | [-0.0785, +0.0939] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2528 | +0.2448 | **+0.0080** | [-0.0785, +0.0939] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0142 | +0.0031 | **+0.0111** | [+0.0045, +0.0184] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| Murphy reliability | `ALL` | `pop` | +0.0022 | +0.0036 | **-0.0013** | [-0.0060, +0.0024] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0022 | +0.0036 | **-0.0013** | [-0.0060, +0.0024] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.0316 | +0.0124 | **-0.0440** | [-0.2229, +0.1430] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0719 | -0.0212 | **-0.0507** | [-0.1970, +0.1065] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0719 | -0.0212 | **-0.0507** | [-0.1970, +0.1065] | 4000 | **NOT DETECTED** (CI covers zero) |

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

**arm — `ai_v12_15_ladder_ctrl10M_b` @ `step_10000032`**: 144313 states / 4772 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 28 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 147339 states / 4777 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 23 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.1110 | +0.1774 | **-0.0664** | [-0.1069, -0.0307] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 398/186/5 | +0.1124 | +0.1806 | **-0.0682** | [-0.1062, -0.0331] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1134 | -0.0987 | **-0.0147** | [-0.0339, +0.0040] | 2000 | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.4277 | +0.6680 | **-0.2403** | [-0.3488, -0.1419] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 398/186/5 | +0.4256 | +0.6641 | **-0.2385** | [-0.3384, -0.1455] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0730 | -0.0398 | **-0.0332** | [-0.0529, -0.0129] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 398/186/5 | +0.0224 | +0.0369 | **-0.0146** | [-0.0289, +0.0002] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 398/186/5 | +0.0124 | +0.0044 | **+0.0081** | [+0.0012, +0.0151] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 398/186/5 | +0.5107 | +0.5177 | **-0.0070** | [-0.0337, +0.0201] | 2000 | **NOT DETECTED** (CI covers zero) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 398/186/5 | +0.0535 | +0.0591 | **-0.0056** | [-0.0108, +0.0009] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 398/186/5 | +0.0299 | +0.0312 | **-0.0013** | [-0.0061, +0.0037] | 2000 | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 398/186/5 | +0.6922 | +2.0892 | **-1.3970** | [-0.3571, -0.1546] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 398/186/5 | +0.3260 | +0.5730 | **-0.2470** | [-0.2298, -0.1427] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 398/186/5 | +0.0056 | -0.0002 | **+0.0057** | [-0.0043, +0.0153] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 398/186/5 | +0.0168 | +0.0371 | **-0.0202** | [-0.0368, -0.0032] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.2633 | +1.0719 | **+0.1914** | [+0.0882, +0.2960] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | -0.7106 | +0.4257 | **-1.1363** | [-1.3259, -0.9484] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.7103 | +0.8379 | **-0.1276** | [-0.2972, +0.0449] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.3998 | +0.9122 | **-0.5124** | [-0.7845, -0.2386] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 398/186/5 | +1.1838 | +1.0330 | **+0.1507** | [+0.0389, +0.2539] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.2467 | +1.0255 | **+0.2212** | [+0.0970, +0.3401] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | -0.6791 | +0.4900 | **-1.1692** | [-1.3858, -0.9530] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |

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
| arm · team | 534 | 601 | 4614 | 9228 | 7.0 | 14.0 | 4–51 |
| arm · stratum | 5 | 601 | 4614 | 9228 | 923.0 | 1846.0 | 916–931 |
| arm · between-team spread | 534 | 601 | 4614 | — | 7.0 | — | — |
| control · team | 535 | 601 | 4622 | 9244 | 7.0 | 14.0 | 4–51 |
| control · stratum | 5 | 601 | 4622 | 9244 | 926.0 | 1852.0 | 905–937 |
| control · between-team spread | 535 | 601 | 4622 | — | 7.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 398/186/5 | +0.0535 | +0.0591 | **-0.0056** | [-0.0108, +0.0009] | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 398/186/5 | +0.0299 | +0.0312 | **-0.0013** | [-0.0061, +0.0037] | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 398/186/5 | +0.6922 | +2.0892 | **-1.3970** | [-0.3571, -0.1546] | **DETECTED** (vs ZERO — NO FLOOR) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 398/186/5 | +0.0224 | +0.0369 | **-0.0146** | [-0.0289, +0.0002] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 398/186/5 | +0.0056 | -0.0002 | **+0.0057** | [-0.0043, +0.0153] | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 398/186/5 | +0.0168 | +0.0371 | **-0.0202** | [-0.0368, -0.0032] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 9544 | 4772 | 0.8716 | **0.1528** | **1.4015** | [0.368, 0.995] | 0.0000 |
| arm · t1_3 | `t1_3` | 9544 | 4772 | 0.8567 | **0.0717** | **0.5978** | [0.685, 0.958] | 0.0000 |
| arm · common support | `common support` | 9066 | 4723 | 0.8859 | **0.1096** | **1.1639** | [0.564, 0.993] | 0.0000 |
| control · all | `all` | 9554 | 4777 | 0.8200 | **0.2003** | **1.7478** | [0.199, 0.996] | 0.0000 |
| control · t1_3 | `t1_3` | 9554 | 4777 | 0.7641 | **0.1120** | **0.6466** | [0.477, 0.932] | 0.0000 |
| control · common support | `common support` | 8622 | 4626 | 0.8455 | **0.1414** | **1.3667** | [0.471, 0.993] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.3677, 0.9948]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.2633 | +1.0719 | **+0.1914** | [+0.0882, +0.2960] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | -0.7106 | +0.4257 | **-1.1363** | [-1.3259, -0.9484] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.7103 | +0.8379 | **-0.1276** | [-0.2972, +0.0449] | — | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.3998 | +0.9122 | **-0.5124** | [-0.7845, -0.2386] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 398/186/5 | +1.1838 | +1.0330 | **+0.1507** | [+0.0389, +0.2539] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.2467 | +1.0255 | **+0.2212** | [+0.0970, +0.3401] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | -0.6791 | +0.4900 | **-1.1692** | [-1.3858, -0.9530] | — | **DETECTED** (vs ZERO — NO FLOOR) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.1124 [+0.1794, +0.1818] | +0.1806 | **-0.0682** | [-0.1062, -0.0331] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.1124 [+0.1794, +0.1818] | +0.1806 | **-0.0682** | [-0.1062, -0.0331] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1124 | +0.1804 | **-0.0680** | [-0.1074, -0.0333] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.4256 [+0.6622, +0.6666] | +0.6641 | **-0.2385** | [-0.3384, -0.1455] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.4256 [+0.6622, +0.6666] | +0.6641 | **-0.2385** | [-0.3384, -0.1455] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.4256 | +0.6644 | **-0.2388** | [-0.3450, -0.1424] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0224 [+0.0359, +0.0382] | +0.0369 | **-0.0146** | [-0.0289, +0.0002] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0224 [+0.0359, +0.0382] | +0.0369 | **-0.0146** | [-0.0289, +0.0002] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0224 | +0.0376 | **-0.0152** | [-0.0302, -0.0007] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0124 [+0.0038, +0.0055] | +0.0044 | **+0.0081** | [+0.0012, +0.0151] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.own_team_r2.all` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0124 [+0.0038, +0.0055] | +0.0044 | **+0.0081** | [+0.0012, +0.0151] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0124 | +0.0048 | **+0.0077** | [+0.0005, +0.0152] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.5107 [+0.5166, +0.5202] | +0.5177 | **-0.0070** | [-0.0337, +0.0201] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.5107 [+0.5166, +0.5202] | +0.5177 | **-0.0070** | [-0.0337, +0.0201] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5107 | +0.5137 | **-0.0030** | [-0.0290, +0.0242] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0535 [+0.0584, +0.0601] | +0.0591 | **-0.0056** | [-0.0108, +0.0009] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0535 [+0.0584, +0.0601] | +0.0591 | **-0.0056** | [-0.0108, +0.0009] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0535 | +0.0595 | **-0.0060** | [-0.0113, +0.0002] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0299 [+0.0305, +0.0317] | +0.0312 | **-0.0013** | [-0.0061, +0.0037] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0299 [+0.0305, +0.0317] | +0.0312 | **-0.0013** | [-0.0061, +0.0037] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0299 | +0.0305 | **-0.0006** | [-0.0057, +0.0041] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.6922 [+1.9488, +2.9623] | +2.0892 | **-1.3970** | [-0.3571, -0.1546] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.6922 [+1.9488, +2.9623] | +2.0892 | **-1.3970** | [-0.3571, -0.1546] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.6922 | +2.0616 | **-1.3694** | [-0.3603, -0.1592] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.3260 [+0.5717, +0.5831] | +0.5730 | **-0.2470** | [-0.2298, -0.1427] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.3260 [+0.5717, +0.5831] | +0.5730 | **-0.2470** | [-0.2298, -0.1427] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.3260 | +0.5732 | **-0.2472** | [-0.2296, -0.1445] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0056 [-0.0029, +0.0020] | -0.0002 | **+0.0057** | [-0.0043, +0.0153] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0056 [-0.0029, +0.0020] | -0.0002 | **+0.0057** | [-0.0043, +0.0153] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0056 | +0.0028 | **+0.0028** | [-0.0070, +0.0128] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0168 [+0.0346, +0.0401] | +0.0371 | **-0.0202** | [-0.0368, -0.0032] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +0.0168 [+0.0346, +0.0401] | +0.0371 | **-0.0202** | [-0.0368, -0.0032] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0168 | +0.0348 | **-0.0180** | [-0.0351, -0.0003] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +1.1838 [+1.0210, +1.0444] | +1.0330 | **+0.1507** | [+0.0389, +0.2539] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 398/186/5 (4773 battles, 4618 decoder battles, 21 seeds) | +1.1838 [+1.0210, +1.0444] | +1.0330 | **+0.1507** | [+0.0389, +0.2539] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.1838 | +1.0067 | **+0.1770** | [+0.0711, +0.2833] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.1110 | [+0.0948, +0.1333] | +0.1774 | [+0.1491, +0.2145] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1124 | [+0.0965, +0.1340] | +0.1804 | [+0.1527, +0.2168] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1134 | [-0.1269, -0.1012] | -0.0987 | [-0.1130, -0.0855] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.4277 | [+0.3765, +0.4821] | +0.6680 | [+0.5832, +0.7618] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.4256 | [+0.3753, +0.4787] | +0.6644 | [+0.5812, +0.7564] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0730 | [-0.0867, -0.0609] | -0.0398 | [-0.0549, -0.0259] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0224 | [+0.0133, +0.0310] | +0.0376 | [+0.0259, +0.0493] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0124 | [+0.0060, +0.0185] | +0.0048 | [+0.0003, +0.0090] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5107 | [+0.4927, +0.5301] | +0.5137 | [+0.4959, +0.5325] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0535 | [+0.0495, +0.0570] | +0.0595 | [+0.0543, +0.0629] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0299 | [+0.0272, +0.0340] | +0.0305 | [+0.0277, +0.0347] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.6922 | [+0.2892, +0.3751] | +2.0616 | [+0.5026, +0.6841] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.3260 | [+0.2382, +0.2774] | +0.5732 | [+0.4093, +0.4831] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0056 | [-0.0034, +0.0132] | +0.0028 | [-0.0038, +0.0079] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0168 | [+0.0059, +0.0291] | +0.0348 | [+0.0229, +0.0473] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.2633 | [+1.1888, +1.3413] | +1.0719 | [+1.0083, +1.1389] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | -0.7106 | [-0.8652, -0.5604] | +0.4257 | [+0.3162, +0.5278] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.7103 | [+0.5913, +0.8343] | +0.8379 | [+0.7197, +0.9623] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.3998 | [+0.1843, +0.6208] | +0.9122 | [+0.7647, +1.0621] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.1838 | [+1.1050, +1.2694] | +1.0067 | [+0.9405, +1.0788] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.2467 | [+1.1640, +1.3312] | +1.0255 | [+0.9397, +1.1139] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | -0.6791 | [-0.8490, -0.5137] | +0.4900 | [+0.3553, +0.6204] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

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

Identity, capture-rate reweighted: 800 labels / 677 battles / 6400 rollouts · Brier 0.1058 = REL 0.0022 − RES 0.0357 + UNC 0.1413 + WBV 0.0008 (resid -2.80e-03) · base rate 0.8298 · **resolution is 25.3% of the base-rate cap** · corr(turn,V) -0.2201 vs corr(turn,MC) -0.1885

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 678 battles / 6400 rollouts · Brier 0.1002 = REL 0.0036 − RES 0.0315 + UNC 0.1287 + WBV 0.0008 (resid -1.32e-03) · base rate 0.8482 · **resolution is 24.5% of the base-rate cap** · corr(turn,V) -0.1044 vs corr(turn,MC) -0.1168

## 5. THE LEDGER LINE

```
ai_v12_15_ladder_ctrl10M_b vs ai_v12_11_ladder_ctrl10M at 10M [OFFLINE-GENERATED: 400 games x 9+3 opponents, full capture]: G1 bot Δ +0.0004 [-0.0070, +0.0073] NOT DETECTED · identity bias late Δ +0.1024 [+0.0503, +0.1553] DETECTED · turn-contrast Δ -0.0440 [-0.2229, +0.1430] NOT DETECTED · spread ratio t1-3 Δ -0.0664 [-0.1069, -0.0307] DETECTED · own-team R2 t1 Δ -0.0146 [-0.0289, +0.0002] NOT DETECTED [QUOTA-MATCHED] · calib slope Δ +0.1914 [+0.0882, +0.2960] DETECTED
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_15_ladder_ctrl10M_b --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_15_ladder_ctrl10M_b --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_11_ladder_ctrl10M --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/v5_floor400 --nice 15

# arm — ai_v12_15_ladder_ctrl10M_b  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_15_ladder_ctrl10M_b/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_15_ladder_ctrl10M_b_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_15_ladder_ctrl10M_b_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_15_ladder_ctrl10M_b_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/v5_floor400/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/v5_floor400/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
