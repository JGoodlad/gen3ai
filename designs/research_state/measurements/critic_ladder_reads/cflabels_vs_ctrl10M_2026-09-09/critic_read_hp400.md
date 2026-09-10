# CRITIC READ — `ai_v12_12_ladder_cflabels` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v3 at 2026-09-09T19:24:56. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_12_ladder_cflabels` | `step_10000032` | 4800 | 0.6% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 4800 | 0.5% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 400 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260909, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_12_ladder_cflabels` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_12_ladder_cflabels` | **399/145/6** | 4800 | 4149 / 622 / 29 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **400/188/5** | 4800 | 4143 / 634 / 23 | 12 |

**Frames:** the realized per-opponent caps differ — arm 399/145/6, control 400/188/5 (traced W/L/D per opponent, counted on disk); control cut to 399/145/5

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **399/145/5** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> 🚨 **NO REPLICATE FLOOR — deltas are NOT DETECTED unless their CI clears zero, and never DETECTED against a floor.** The ladder's replicate floor is the control-vs-control difference and does not exist until a second control replicate does. Every DETECTED below is a detection against ZERO and carries that qualifier.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0216 | +0.0202 | **+0.0014** | [-0.0056, +0.0090] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0156 | -0.0308 | **+0.0464** | [-0.0049, +0.0968] | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.0133 | +0.0124 | **+0.0008** | [-0.1523, +0.1544] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2052 | +0.1636 | **+0.0416** | [-0.0244, +0.1178] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.1521 | +0.1774 | **-0.0253** | [-0.0711, +0.0220] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 399/145/5 | +0.0534 | +0.0370 | **+0.0164** | [-0.0011, +0.0341] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0372 | +0.0395 | **-0.0023** | [-0.0108, +0.0047] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0216 | +0.0202 | **+0.0014** | [-0.0056, +0.0090] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0523 | +0.0544 | **-0.0021** | [-0.0169, +0.0136] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0002 | +0.0013 | **-0.0011** | [-0.0024, -0.0001] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0012 | +0.0036 | **-0.0024** | [-0.0049, -0.0004] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0035 | +0.0022 | **+0.0013** | [-0.0024, +0.0058] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `all` | `capture-rate (gauge)` | +0.0141 | +0.0206 | **-0.0064** | [-0.0189, +0.0077] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `bot` | `capture-rate (gauge)` | +0.0253 | +0.0394 | **-0.0141** | [-0.0303, +0.0012] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `pool` | `capture-rate (gauge)` | +0.0542 | +0.0379 | **+0.0163** | [-0.0137, +0.0445] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2616 | +0.2535 | **+0.0081** | [-0.0421, +0.0520] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2052 | +0.1636 | **+0.0416** | [-0.0244, +0.1178] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2408 | +0.2446 | **-0.0038** | [-0.0780, +0.0725] | 400 | **NOT DETECTED** (CI covers zero) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0454 | +0.0055 | **+0.0399** | [+0.0121, +0.0671] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `ALL` | `pop` | -0.0043 | -0.0391 | **+0.0348** | [+0.0125, +0.0556] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0043 | -0.0391 | **+0.0348** | [+0.0125, +0.0556] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0268 | -0.0196 | **+0.0464** | [+0.0015, +0.0909] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0214 | -0.0675 | **+0.0460** | [+0.0114, +0.0824] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0214 | -0.0675 | **+0.0460** | [+0.0114, +0.0824] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0538 | +0.0143 | **+0.0395** | [-0.0017, +0.0793] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0000 | -0.0228 | **+0.0228** | [-0.0057, +0.0520] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0000 | -0.0228 | **+0.0228** | [-0.0057, +0.0520] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0569 | +0.0241 | **+0.0328** | [-0.0279, +0.0931] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0156 | -0.0308 | **+0.0464** | [-0.0049, +0.0968] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0156 | -0.0308 | **+0.0464** | [-0.0049, +0.0968] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `bot` | `raw` | +0.0363 | -0.0033 | **+0.0396** | [+0.0061, +0.0733] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `bot` | `pop` | -0.0117 | -0.0473 | **+0.0356** | [+0.0128, +0.0581] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0117 | -0.0473 | **+0.0356** | [+0.0128, +0.0581] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `pool` | `raw` | +0.0618 | +0.0210 | **+0.0409** | [-0.0071, +0.0904] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `pool` | `pop` | +0.0128 | -0.0230 | **+0.0358** | [-0.0097, +0.0827] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0128 | -0.0230 | **+0.0358** | [-0.0097, +0.0827] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `raw` | +0.0275 | +0.0263 | **+0.0011** | [-0.0105, +0.0128] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0307 | +0.0315 | **-0.0008** | [-0.0146, +0.0130] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0307 | +0.0315 | **-0.0008** | [-0.0146, +0.0130] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1538 | +0.1567 | **-0.0029** | [-0.0794, +0.0776] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `pop` | +0.2391 | +0.2213 | **+0.0179** | [-0.0853, +0.1296] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2391 | +0.2213 | **+0.0179** | [-0.0853, +0.1296] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1719 | +0.1683 | **+0.0037** | [-0.0629, +0.0711] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2413 | +0.2448 | **-0.0035** | [-0.0909, +0.0887] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2413 | +0.2448 | **-0.0035** | [-0.0909, +0.0887] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0042 | +0.0031 | **+0.0011** | [-0.0033, +0.0052] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `pop` | +0.0008 | +0.0036 | **-0.0028** | [-0.0071, +0.0009] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0008 | +0.0036 | **-0.0028** | [-0.0071, +0.0009] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.0133 | +0.0124 | **+0.0008** | [-0.1523, +0.1544] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | +0.0101 | -0.0212 | **+0.0313** | [-0.1160, +0.1650] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0101 | -0.0212 | **+0.0313** | [-0.1160, +0.1650] | 4000 | **NOT DETECTED** (CI covers zero) |

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

**arm — `ai_v12_12_ladder_cflabels` @ `step_10000032`**: 150285 states / 4771 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 29 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 147339 states / 4777 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 23 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.1521 | +0.1774 | **-0.0253** | [-0.0711, +0.0220] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 399/145/5 | +0.1562 | +0.1823 | **-0.0262** | [-0.0713, +0.0203] | 2000 | **NOT DETECTED** (CI covers zero) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0800 | -0.0987 | **+0.0187** | [+0.0001, +0.0375] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.6136 | +0.6680 | **-0.0544** | [-0.1860, +0.0734] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 399/145/5 | +0.6083 | +0.6642 | **-0.0560** | [-0.1816, +0.0695] | 2000 | **NOT DETECTED** (CI covers zero) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0365 | -0.0398 | **+0.0034** | [-0.0161, +0.0228] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 399/145/5 | +0.0534 | +0.0370 | **+0.0164** | [-0.0011, +0.0341] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 399/145/5 | +0.0077 | +0.0058 | **+0.0019** | [-0.0049, +0.0091] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 399/145/5 | +0.5080 | +0.5186 | **-0.0106** | [-0.0376, +0.0158] | 2000 | **NOT DETECTED** (CI covers zero) |

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 399/145/5 (4733 battles, 4570 decoder battles, 21 seeds) | +0.1562 [+0.1776, +0.1872] | +0.1823 | **-0.0262** | [-0.0713, +0.0203] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 499/181/5 (4770 battles, 4615 decoder battles, 21 seeds) | +0.1562 [+0.1788, +0.1825] | +0.1801 | **-0.0240** | [-0.0688, +0.0227] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1562 | +0.1804 | **-0.0243** | [-0.0691, +0.0212] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 399/145/5 (4733 battles, 4570 decoder battles, 21 seeds) | +0.6083 [+0.6574, +0.6729] | +0.6642 | **-0.0560** | [-0.1816, +0.0695] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 499/181/5 (4770 battles, 4615 decoder battles, 21 seeds) | +0.6083 [+0.6610, +0.6683] | +0.6637 | **-0.0554** | [-0.1814, +0.0697] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.6083 | +0.6644 | **-0.0562** | [-0.1849, +0.0680] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 399/145/5 (4733 battles, 4570 decoder battles, 21 seeds) | +0.0534 [+0.0339, +0.0442] | +0.0370 | **+0.0164** | [-0.0011, +0.0341] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | MATCHED · decoder 499/181/5 (4770 battles, 4615 decoder battles, 21 seeds) | +0.0534 [+0.0368, +0.0393] | +0.0379 | **+0.0155** | [-0.0010, +0.0329] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0534 | +0.0376 | **+0.0158** | [-0.0014, +0.0336] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 399/145/5 (4733 battles, 4570 decoder battles, 21 seeds) | +0.0077 [+0.0035, +0.0087] | +0.0058 | **+0.0019** | [-0.0049, +0.0091] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | MATCHED · decoder 499/181/5 (4770 battles, 4615 decoder battles, 21 seeds) | +0.0077 [+0.0037, +0.0064] | +0.0047 | **+0.0030** | [-0.0032, +0.0095] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0077 | +0.0048 | **+0.0029** | [-0.0035, +0.0097] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 399/145/5 (4733 battles, 4570 decoder battles, 21 seeds) | +0.5080 [+0.5146, +0.5231] | +0.5186 | **-0.0106** | [-0.0376, +0.0158] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 499/181/5 (4770 battles, 4615 decoder battles, 21 seeds) | +0.5080 [+0.5175, +0.5209] | +0.5185 | **-0.0105** | [-0.0374, +0.0153] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5080 | +0.5137 | **-0.0056** | [-0.0330, +0.0202] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.1521 | [+0.1234, +0.1916] | +0.1774 | [+0.1491, +0.2145] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1562 | [+0.1283, +0.1942] | +0.1804 | [+0.1527, +0.2168] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0800 | [-0.0933, -0.0691] | -0.0987 | [-0.1130, -0.0855] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.6136 | [+0.5239, +0.7133] | +0.6680 | [+0.5832, +0.7618] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.6083 | [+0.5211, +0.7049] | +0.6644 | [+0.5812, +0.7564] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0365 | [-0.0509, -0.0245] | -0.0398 | [-0.0549, -0.0259] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0534 | [+0.0407, +0.0664] | +0.0376 | [+0.0259, +0.0493] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0077 | [+0.0028, +0.0126] | +0.0048 | [+0.0003, +0.0090] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5080 | [+0.4888, +0.5265] | +0.5137 | [+0.4959, +0.5325] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

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

Identity, capture-rate reweighted: 800 labels / 690 battles / 6400 rollouts · Brier 0.0969 = REL 0.0008 − RES 0.0307 + UNC 0.1273 + WBV 0.0008 (resid -1.30e-03) · base rate 0.8503 · **resolution is 24.1% of the base-rate cap** · corr(turn,V) -0.0509 vs corr(turn,MC) -0.0642

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 678 battles / 6400 rollouts · Brier 0.1002 = REL 0.0036 − RES 0.0315 + UNC 0.1287 + WBV 0.0008 (resid -1.32e-03) · base rate 0.8482 · **resolution is 24.5% of the base-rate cap** · corr(turn,V) -0.1044 vs corr(turn,MC) -0.1168

## 5. THE LEDGER LINE

```
ai_v12_12_ladder_cflabels vs ai_v12_11_ladder_ctrl10M at 10M [OFFLINE-GENERATED: 400 games x 9+3 opponents, full capture]: G1 bot Δ +0.0014 [-0.0056, +0.0090] NOT DETECTED · identity bias late Δ +0.0464 [-0.0049, +0.0968] NOT DETECTED · turn-contrast Δ +0.0008 [-0.1523, +0.1544] NOT DETECTED · spread ratio t1-3 Δ -0.0253 [-0.0711, +0.0220] NOT DETECTED · own-team R2 t1 Δ +0.0164 [-0.0011, +0.0341] NOT DETECTED [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_12_ladder_cflabels --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_12_ladder_cflabels --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_11_ladder_ctrl10M --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M --nice 15 --ledger-line

# arm — ai_v12_12_ladder_cflabels
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_12_ladder_cflabels --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_12_ladder_cflabels --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_12_ladder_cflabels` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_12_ladder_cflabels/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
