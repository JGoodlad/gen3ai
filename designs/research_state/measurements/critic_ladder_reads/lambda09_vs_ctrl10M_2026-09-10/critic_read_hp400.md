# CRITIC READ — `ai_v12_19_ladder_lambda09` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v3 at 2026-09-10T08:03:27. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | `step_10000032` | 4800 | 0.3% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 4800 | 0.5% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 400 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260909, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_19_ladder_lambda09` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | **397/191/4** | 4800 | 4140 / 646 / 14 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **400/188/5** | 4800 | 4143 / 634 / 23 | 12 |

**Frames:** the realized per-opponent caps differ — arm 397/191/4, control 400/188/5 (traced W/L/D per opponent, counted on disk); arm and control cut to 397/188/4

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm and control side is subsampled to caps **397/188/4** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json` — 54 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0225 | +0.0202 | **+0.0023** | [-0.0044, +0.0091] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0184 | -0.0308 | **+0.0124** | [-0.0365, +0.0639] | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.1121 | +0.0124 | **-0.1246** | [-0.2787, +0.0314] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2307 | +0.1636 | **+0.0671** | [+0.0049, +0.1395] | **NOT DETECTED** (CI does not clear the floor 0.0327) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0538 | +0.1774 | **-0.1236** | [-0.1607, -0.0895] | **DETECTED** (CI clears the floor +0.0664) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 397/188/4 | +0.0359 | +0.0386 | **-0.0027** | [-0.0193, +0.0135] | **WITHIN FLOOR** (|delta| <= floor 0.0146) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0495 | +0.0395 | **+0.0100** | [+0.0018, +0.0183] | 400 | **DETECTED** (CI clears the floor +0.0001) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0225 | +0.0202 | **+0.0023** | [-0.0044, +0.0091] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0745 | +0.0544 | **+0.0201** | [+0.0048, +0.0330] | 400 | **DETECTED** (CI clears the floor +0.0038) |
| reliability | `all` | `capture-rate (gauge)` | +0.0014 | +0.0013 | **+0.0001** | [-0.0013, +0.0019] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0037) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0007 | +0.0036 | **-0.0029** | [-0.0052, -0.0011] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0032) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0082 | +0.0022 | **+0.0060** | [+0.0007, +0.0129] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| ece | `all` | `capture-rate (gauge)` | +0.0265 | +0.0206 | **+0.0060** | [-0.0071, +0.0187] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0373) |
| ece | `bot` | `capture-rate (gauge)` | +0.0244 | +0.0394 | **-0.0150** | [-0.0300, +0.0003] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0286) |
| ece | `pool` | `capture-rate (gauge)` | +0.0774 | +0.0379 | **+0.0395** | [+0.0061, +0.0673] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1171) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3244 | +0.2535 | **+0.0709** | [+0.0311, +0.1142] | 400 | **DETECTED** (CI clears the floor +0.0307) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2307 | +0.1636 | **+0.0671** | [+0.0049, +0.1395] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0327) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.3040 | +0.2446 | **+0.0594** | [-0.0024, +0.1247] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1324) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0425 | +0.0055 | **+0.0369** | [+0.0090, +0.0654] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1029) |
| bias V - p_hat | `ALL` | `pop` | -0.0070 | -0.0391 | **+0.0321** | [+0.0105, +0.0529] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0735) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0070 | -0.0391 | **+0.0321** | [+0.0105, +0.0529] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0735) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0544 | -0.0196 | **+0.0740** | [+0.0303, +0.1168] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1085) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0035 | -0.0675 | **+0.0640** | [+0.0301, +0.0986] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0812) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0035 | -0.0675 | **+0.0640** | [+0.0301, +0.0986] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0812) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0394 | +0.0143 | **+0.0252** | [-0.0154, +0.0623] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0707) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0037 | -0.0228 | **+0.0191** | [-0.0100, +0.0466] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0501) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0037 | -0.0228 | **+0.0191** | [-0.0100, +0.0466] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0501) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0298 | +0.0241 | **+0.0057** | [-0.0566, +0.0681] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1434) |
| bias V - p_hat | `late (turn>=25)` | `pop` | -0.0184 | -0.0308 | **+0.0124** | [-0.0365, +0.0639] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0184 | -0.0308 | **+0.0124** | [-0.0365, +0.0639] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| bias V - p_hat | `bot` | `raw` | +0.0097 | -0.0033 | **+0.0130** | [-0.0195, +0.0450] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0639) |
| bias V - p_hat | `bot` | `pop` | -0.0313 | -0.0473 | **+0.0161** | [-0.0057, +0.0371] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0313 | -0.0473 | **+0.0161** | [-0.0057, +0.0371] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| bias V - p_hat | `pool` | `raw` | +0.0957 | +0.0210 | **+0.0747** | [+0.0250, +0.1239] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1594) |
| bias V - p_hat | `pool` | `pop` | +0.0422 | -0.0230 | **+0.0652** | [+0.0194, +0.1119] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1256) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0422 | -0.0230 | **+0.0652** | [+0.0194, +0.1119] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1256) |
| Murphy resolution | `ALL` | `raw` | +0.0367 | +0.0263 | **+0.0104** | [-0.0015, +0.0211] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0425 | +0.0315 | **+0.0110** | [-0.0032, +0.0249] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0425 | +0.0315 | **+0.0110** | [-0.0032, +0.0249] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.2036 | +0.1567 | **+0.0469** | [-0.0232, +0.1211] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0596) |
| Murphy skill_score | `ALL` | `pop` | +0.3131 | +0.2213 | **+0.0919** | [+0.0027, +0.1900] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0299) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.3131 | +0.2213 | **+0.0919** | [+0.0027, +0.1900] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0299) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.2200 | +0.1683 | **+0.0517** | [-0.0144, +0.1109] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.3156 | +0.2448 | **+0.0709** | [-0.0133, +0.1530] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.3156 | +0.2448 | **+0.0709** | [-0.0133, +0.1530] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0038 | +0.0031 | **+0.0007** | [-0.0039, +0.0048] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0111) |
| Murphy reliability | `ALL` | `pop` | +0.0011 | +0.0036 | **-0.0025** | [-0.0070, +0.0007] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0011 | +0.0036 | **-0.0025** | [-0.0070, +0.0007] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.1121 | +0.0124 | **-0.1246** | [-0.2787, +0.0314] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.1102 | -0.0212 | **-0.0889** | [-0.2199, +0.0426] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.1102 | -0.0212 | **-0.0889** | [-0.2199, +0.0426] | 4000 | **NOT DETECTED** (CI covers zero) |

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

**arm — `ai_v12_19_ladder_lambda09` @ `step_10000032`**: 143812 states / 4786 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 14 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 147339 states / 4777 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 23 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0538 | +0.1774 | **-0.1236** | [-0.1607, -0.0895] | 2000 | **DETECTED** (CI clears the floor +0.0664) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 397/188/4 | +0.0570 | +0.1803 | **-0.1233** | [-0.1597, -0.0905] | 2000 | **DETECTED** (CI clears the floor +0.0682) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1218 | -0.0987 | **-0.0231** | [-0.0415, -0.0041] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0147) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5511 | +0.6680 | **-0.1169** | [-0.2258, -0.0081] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2403) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 397/188/4 | +0.5486 | +0.6644 | **-0.1159** | [-0.2277, -0.0114] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0578 | -0.0398 | **-0.0180** | [-0.0377, +0.0019] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0332) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 397/188/4 | +0.0359 | +0.0386 | **-0.0027** | [-0.0193, +0.0135] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0146) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 397/188/4 | +0.0042 | +0.0062 | **-0.0020** | [-0.0077, +0.0035] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 397/188/4 | +0.5082 | +0.5201 | **-0.0119** | [-0.0402, +0.0146] | 2000 | **NOT DETECTED** (CI covers zero) |

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 397/188/4 (4783 battles, 4625 decoder battles, 21 seeds) | +0.0570 [+0.0564, +0.0577] | +0.1803 | **-0.1233** | [-0.1597, -0.0905] | **DETECTED** (CI clears the floor +0.0682) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 397/188/4 (4783 battles, 4625 decoder battles, 21 seeds) | +0.0570 [+0.0564, +0.0577] | +0.1804 | **-0.1235** | [-0.1591, -0.0906] | **DETECTED** (CI clears the floor +0.0682) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.0572 | +0.1804 | **-0.1233** | [-0.1592, -0.0903] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 397/188/4 (4783 battles, 4625 decoder battles, 21 seeds) | +0.5486 [+0.5465, +0.5499] | +0.6644 | **-0.1159** | [-0.2277, -0.0114] | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 397/188/4 (4783 battles, 4625 decoder battles, 21 seeds) | +0.5486 [+0.5465, +0.5499] | +0.6644 | **-0.1159** | [-0.2281, -0.0078] | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.5487 | +0.6644 | **-0.1157** | [-0.2226, -0.0084] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 397/188/4 (4783 battles, 4625 decoder battles, 21 seeds) | +0.0359 [+0.0343, +0.0380] | +0.0386 | **-0.0027** | [-0.0193, +0.0135] | **WITHIN FLOOR** (|delta| <= floor 0.0146) |
| `cond.own_team_r2.t1` | MATCHED · decoder 397/188/4 (4783 battles, 4625 decoder battles, 21 seeds) | +0.0359 [+0.0343, +0.0380] | +0.0376 | **-0.0017** | [-0.0181, +0.0144] | **WITHIN FLOOR** (|delta| <= floor 0.0146) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0368 | +0.0376 | **-0.0007** | [-0.0169, +0.0155] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 397/188/4 (4783 battles, 4625 decoder battles, 21 seeds) | +0.0042 [+0.0036, +0.0048] | +0.0062 | **-0.0020** | [-0.0077, +0.0035] | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| `cond.own_team_r2.all` | MATCHED · decoder 397/188/4 (4783 battles, 4625 decoder battles, 21 seeds) | +0.0042 [+0.0036, +0.0048] | +0.0048 | **-0.0006** | [-0.0066, +0.0047] | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0044 | +0.0048 | **-0.0004** | [-0.0063, +0.0054] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 397/188/4 (4783 battles, 4625 decoder battles, 21 seeds) | +0.5082 [+0.5044, +0.5107] | +0.5201 | **-0.0119** | [-0.0402, +0.0146] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 397/188/4 (4783 battles, 4625 decoder battles, 21 seeds) | +0.5082 [+0.5044, +0.5107] | +0.5137 | **-0.0055** | [-0.0317, +0.0201] | **WITHIN FLOOR** (|delta| <= floor 0.0070) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5062 | +0.5137 | **-0.0075** | [-0.0342, +0.0185] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0538 | [+0.0424, +0.0717] | +0.1774 | [+0.1491, +0.2145] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0572 | [+0.0465, +0.0742] | +0.1804 | [+0.1527, +0.2168] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1218 | [-0.1360, -0.1088] | -0.0987 | [-0.1130, -0.0855] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5511 | [+0.4851, +0.6230] | +0.6680 | [+0.5832, +0.7618] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.5487 | [+0.4836, +0.6187] | +0.6644 | [+0.5812, +0.7564] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0578 | [-0.0730, -0.0445] | -0.0398 | [-0.0549, -0.0259] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0368 | [+0.0255, +0.0479] | +0.0376 | [+0.0259, +0.0493] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0044 | [+0.0003, +0.0078] | +0.0048 | [+0.0003, +0.0090] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5062 | [+0.4880, +0.5251] | +0.5137 | [+0.4959, +0.5325] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

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

Identity, capture-rate reweighted: 800 labels / 689 battles / 6400 rollouts · Brier 0.0925 = REL 0.0011 − RES 0.0425 + UNC 0.1347 + WBV 0.0008 (resid -1.54e-03) · base rate 0.8395 · **resolution is 31.6% of the base-rate cap** · corr(turn,V) -0.1911 vs corr(turn,MC) -0.0790

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 678 battles / 6400 rollouts · Brier 0.1002 = REL 0.0036 − RES 0.0315 + UNC 0.1287 + WBV 0.0008 (resid -1.32e-03) · base rate 0.8482 · **resolution is 24.5% of the base-rate cap** · corr(turn,V) -0.1044 vs corr(turn,MC) -0.1168

## 5. THE LEDGER LINE

```
ai_v12_19_ladder_lambda09 vs ai_v12_11_ladder_ctrl10M at 10M [OFFLINE-GENERATED: 400 games x 9+3 opponents, full capture]: G1 bot Δ +0.0023 [-0.0044, +0.0091] NOT DETECTED · identity bias late Δ +0.0124 [-0.0365, +0.0639] WITHIN FLOOR · turn-contrast Δ -0.1246 [-0.2787, +0.0314] NOT DETECTED · spread ratio t1-3 Δ -0.1236 [-0.1607, -0.0895] DETECTED · own-team R2 t1 Δ -0.0027 [-0.0193, +0.0135] WITHIN FLOOR [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_19_ladder_lambda09 --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_19_ladder_lambda09 --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_11_ladder_ctrl10M --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M --nice 15 --ledger-line

# arm — ai_v12_19_ladder_lambda09
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_19_ladder_lambda09 --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09 --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_19_ladder_lambda09/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
