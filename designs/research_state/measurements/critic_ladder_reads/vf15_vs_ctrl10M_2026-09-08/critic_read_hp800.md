# CRITIC READ — `ai_v12_10_ladder_vf15` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v3 at 2026-09-10T01:08:26. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_10_ladder_vf15` | `step_10000032` | 9600 | 0.7% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 9600 | 0.4% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 800 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260910, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_10_ladder_vf15` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_10_ladder_vf15` | **797/269/14** | 9600 | 8386 / 1151 / 63 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **797/368/9** | 9600 | 8252 / 1305 / 43 | 12 |

**Frames:** the realized per-opponent caps differ — arm 797/269/14, control 797/368/9 (traced W/L/D per opponent, counted on disk); control cut to 797/269/9

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **797/269/9** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/hp800_floor.json` — 54 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0238 | +0.0170 | **+0.0068** | [+0.0015, +0.0125] | **NOT DETECTED** (CI does not clear the floor 0.0036) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0029 | -0.0189 | **+0.0218** | [-0.0173, +0.0616] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.0502 | +0.0386 | **-0.0888** | [-0.2171, +0.0408] | **WITHIN FLOOR** (|delta| <= floor 0.1082) |
| skill · `bot` | — | +0.2152 | +0.1237 | **+0.0915** | [+0.0321, +0.1574] | **WITHIN FLOOR** (|delta| <= floor 0.1004) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.1652 | +0.1454 | **+0.0198** | [-0.0151, +0.0574] | **WITHIN FLOOR** (|delta| <= floor 0.0426) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 797/269/9 | +0.0732 | +0.0440 | **+0.0291** | [+0.0147, +0.0437] | **DETECTED** (CI clears the floor +0.0037) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0363 | +0.0386 | **-0.0023** | [-0.0086, +0.0035] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0238 | +0.0170 | **+0.0068** | [+0.0015, +0.0125] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0036) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0565 | +0.0614 | **-0.0048** | [-0.0165, +0.0081] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0086) |
| reliability | `all` | `capture-rate (gauge)` | +0.0007 | +0.0010 | **-0.0002** | [-0.0011, +0.0006] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0025 | +0.0053 | **-0.0027** | [-0.0048, -0.0009] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0019 | +0.0019 | **-0.0000** | [-0.0022, +0.0021] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0167) |
| ece | `all` | `capture-rate (gauge)` | +0.0145 | +0.0237 | **-0.0092** | [-0.0160, +0.0009] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0122) |
| ece | `bot` | `capture-rate (gauge)` | +0.0323 | +0.0445 | **-0.0122** | [-0.0245, -0.0026] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0297) |
| ece | `pool` | `capture-rate (gauge)` | +0.0371 | +0.0403 | **-0.0031** | [-0.0265, +0.0169] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0822) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2750 | +0.2598 | **+0.0152** | [-0.0224, +0.0521] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2152 | +0.1237 | **+0.0915** | [+0.0321, +0.1574] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1004) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2948 | +0.2742 | **+0.0206** | [-0.0346, +0.0718] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1108) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0389 | +0.0164 | **+0.0225** | [-0.0037, +0.0482] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0829) |
| bias V - p_hat | `ALL` | `pop` | -0.0063 | -0.0363 | **+0.0300** | [+0.0099, +0.0498] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0063 | -0.0363 | **+0.0300** | [+0.0099, +0.0498] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0307 | -0.0107 | **+0.0414** | [+0.0024, +0.0824] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0948) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0198 | -0.0648 | **+0.0450** | [+0.0123, +0.0774] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0198 | -0.0648 | **+0.0450** | [+0.0123, +0.0774] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0430 | +0.0229 | **+0.0201** | [-0.0174, +0.0615] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0748) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0053 | -0.0195 | **+0.0248** | [-0.0035, +0.0545] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0556) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0053 | -0.0195 | **+0.0248** | [-0.0035, +0.0545] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0556) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0447 | +0.0411 | **+0.0036** | [-0.0487, +0.0573] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0821) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0029 | -0.0189 | **+0.0218** | [-0.0173, +0.0616] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0029 | -0.0189 | **+0.0218** | [-0.0173, +0.0616] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat | `bot` | `raw` | +0.0239 | +0.0029 | **+0.0211** | [-0.0107, +0.0514] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0575) |
| bias V - p_hat | `bot` | `pop` | -0.0212 | -0.0441 | **+0.0230** | [+0.0008, +0.0437] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0417) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0212 | -0.0441 | **+0.0230** | [+0.0008, +0.0437] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0417) |
| bias V - p_hat | `pool` | `raw` | +0.0677 | +0.0436 | **+0.0241** | [-0.0227, +0.0711] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1164) |
| bias V - p_hat | `pool` | `pop` | +0.0263 | -0.0190 | **+0.0453** | [+0.0046, +0.0852] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0263 | -0.0190 | **+0.0453** | [+0.0046, +0.0852] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| Murphy resolution | `ALL` | `raw` | +0.0338 | +0.0274 | **+0.0064** | [-0.0054, +0.0181] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0070) |
| Murphy resolution | `ALL` | `pop` | +0.0386 | +0.0359 | **+0.0027** | [-0.0127, +0.0176] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0386 | +0.0359 | **+0.0027** | [-0.0127, +0.0176] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1882 | +0.1626 | **+0.0256** | [-0.0497, +0.1012] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0370) |
| Murphy skill_score | `ALL` | `pop` | +0.2922 | +0.2601 | **+0.0321** | [-0.0712, +0.1364] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2922 | +0.2601 | **+0.0321** | [-0.0712, +0.1364] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.2147 | +0.1739 | **+0.0408** | [-0.0250, +0.1057] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.3011 | +0.2834 | **+0.0177** | [-0.0722, +0.1070] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.3011 | +0.2834 | **+0.0177** | [-0.0722, +0.1070] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0052 | +0.0024 | **+0.0028** | [-0.0011, +0.0069] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0119) |
| Murphy reliability | `ALL` | `pop` | +0.0015 | +0.0030 | **-0.0015** | [-0.0051, +0.0016] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0015 | +0.0030 | **-0.0015** | [-0.0051, +0.0016] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.0502 | +0.0386 | **-0.0888** | [-0.2171, +0.0408] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1082) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0346 | +0.0222 | **-0.0567** | [-0.1584, +0.0476] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0984) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0346 | +0.0222 | **-0.0567** | [-0.1584, +0.0476] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0984) |

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

**arm — `ai_v12_10_ladder_vf15` @ `step_10000032`**: 282553 states / 9537 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 63 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 291604 states / 9557 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 43 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.1652 | +0.1454 | **+0.0198** | [-0.0151, +0.0574] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0426) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 797/269/9 | +0.1681 | +0.1474 | **+0.0207** | [-0.0148, +0.0573] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0448) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0661 | -0.1018 | **+0.0357** | [+0.0227, +0.0484] | 2000 | **DETECTED** (CI clears the floor +0.0024) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5691 | +0.6237 | **-0.0546** | [-0.1454, +0.0443] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1973) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 797/269/9 | +0.5665 | +0.6217 | **-0.0552** | [-0.1497, +0.0381] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0341 | -0.0448 | **+0.0107** | [-0.0028, +0.0245] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0218) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 797/269/9 | +0.0732 | +0.0440 | **+0.0291** | [+0.0147, +0.0437] | 2000 | **DETECTED** (CI clears the floor +0.0037) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 797/269/9 | +0.0104 | +0.0108 | **-0.0004** | [-0.0058, +0.0050] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0013) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 797/269/9 | +0.5073 | +0.5002 | **+0.0071** | [-0.0124, +0.0272] | 2000 | **NOT DETECTED** (CI covers zero) |

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 797/269/9 (9458 battles, 9452 decoder battles, 21 seeds) | +0.1681 [+0.1436, +0.1516] | +0.1474 | **+0.0207** | [-0.0148, +0.0573] | **WITHIN FLOOR** (|delta| <= floor 0.0448) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 996/336/9 (9525 battles, 9519 decoder battles, 21 seeds) | +0.1681 [+0.1462, +0.1497] | +0.1478 | **+0.0203** | [-0.0142, +0.0551] | **WITHIN FLOOR** (|delta| <= floor 0.0448) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1681 | +0.1477 | **+0.0204** | [-0.0137, +0.0570] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 797/269/9 (9458 battles, 9452 decoder battles, 21 seeds) | +0.5665 [+0.6137, +0.6301] | +0.6217 | **-0.0552** | [-0.1497, +0.0381] | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 996/336/9 (9525 battles, 9519 decoder battles, 21 seeds) | +0.5665 [+0.6194, +0.6276] | +0.6215 | **-0.0550** | [-0.1543, +0.0384] | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.5665 | +0.6221 | **-0.0556** | [-0.1449, +0.0419] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 797/269/9 (9458 battles, 9452 decoder battles, 21 seeds) | +0.0732 [+0.0396, +0.0498] | +0.0440 | **+0.0291** | [+0.0147, +0.0437] | **DETECTED** (CI clears the floor +0.0037) |
| `cond.own_team_r2.t1` | MATCHED · decoder 996/336/9 (9525 battles, 9519 decoder battles, 21 seeds) | +0.0732 [+0.0429, +0.0489] | +0.0465 | **+0.0267** | [+0.0117, +0.0414] | **DETECTED** (CI clears the floor +0.0037) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0732 | +0.0475 | **+0.0257** | [+0.0109, +0.0402] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 797/269/9 (9458 battles, 9452 decoder battles, 21 seeds) | +0.0104 [+0.0085, +0.0123] | +0.0108 | **-0.0004** | [-0.0058, +0.0050] | **WITHIN FLOOR** (|delta| <= floor 0.0013) |
| `cond.own_team_r2.all` | MATCHED · decoder 996/336/9 (9525 battles, 9519 decoder battles, 21 seeds) | +0.0104 [+0.0093, +0.0114] | +0.0103 | **+0.0001** | [-0.0054, +0.0057] | **WITHIN FLOOR** (|delta| <= floor 0.0013) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0104 | +0.0105 | **-0.0001** | [-0.0054, +0.0057] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 797/269/9 (9458 battles, 9452 decoder battles, 21 seeds) | +0.5073 [+0.4896, +0.5028] | +0.5002 | **+0.0071** | [-0.0124, +0.0272] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 996/336/9 (9525 battles, 9519 decoder battles, 21 seeds) | +0.5073 [+0.4865, +0.5007] | +0.4955 | **+0.0118** | [-0.0068, +0.0311] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5073 | +0.5014 | **+0.0060** | [-0.0112, +0.0256] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.1652 | [+0.1392, +0.1985] | +0.1454 | [+0.1265, +0.1696] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1681 | [+0.1425, +0.2006] | +0.1477 | [+0.1291, +0.1717] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0661 | [-0.0758, -0.0577] | -0.1018 | [-0.1114, -0.0924] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5691 | [+0.4943, +0.6466] | +0.6237 | [+0.5653, +0.6832] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.5665 | [+0.4931, +0.6421] | +0.6221 | [+0.5643, +0.6808] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0341 | [-0.0443, -0.0255] | -0.0448 | [-0.0552, -0.0353] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0732 | [+0.0612, +0.0845] | +0.0475 | [+0.0380, +0.0566] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0104 | [+0.0063, +0.0143] | +0.0105 | [+0.0065, +0.0143] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5073 | [+0.4947, +0.5205] | +0.5014 | [+0.4883, +0.5145] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

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

Identity, capture-rate reweighted: 800 labels / 728 battles / 6400 rollouts · Brier 0.0908 = REL 0.0015 − RES 0.0386 + UNC 0.1283 + WBV 0.0008 (resid -1.22e-03) · base rate 0.8488 · **resolution is 30.1% of the base-rate cap** · corr(turn,V) -0.1714 vs corr(turn,MC) -0.1213

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 748 battles / 6400 rollouts · Brier 0.0938 = REL 0.0030 − RES 0.0359 + UNC 0.1267 + WBV 0.0008 (resid -8.83e-04) · base rate 0.8511 · **resolution is 28.3% of the base-rate cap** · corr(turn,V) -0.0229 vs corr(turn,MC) -0.0615

## 5. THE LEDGER LINE

```
ai_v12_10_ladder_vf15 vs ai_v12_11_ladder_ctrl10M at 10M [OFFLINE-GENERATED: 800 games x 9+3 opponents, full capture]: G1 bot Δ +0.0068 [+0.0015, +0.0125] NOT DETECTED · identity bias late Δ +0.0218 [-0.0173, +0.0616] WITHIN FLOOR · turn-contrast Δ -0.0888 [-0.2171, +0.0408] WITHIN FLOOR · spread ratio t1-3 Δ +0.0198 [-0.0151, +0.0574] WITHIN FLOOR · own-team R2 t1 Δ +0.0291 [+0.0147, +0.0437] DETECTED [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_10_ladder_vf15 --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_10_ladder_vf15 --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_11_ladder_ctrl10M --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/hp800_floor.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_10_ladder_vf15_vs_ctrl10M --nice 15 --ledger-line

# arm — ai_v12_10_ladder_vf15
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_10_ladder_vf15 --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_10_ladder_vf15_vs_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_10_ladder_vf15 --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_10_ladder_vf15_vs_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_10_ladder_vf15_vs_ctrl10M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_10_ladder_vf15` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_10_ladder_vf15/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_10_ladder_vf15_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_10_ladder_vf15_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_10_ladder_vf15_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_11_ladder_ctrl10M__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_11_ladder_ctrl10M__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_11_ladder_ctrl10M__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_10_ladder_vf15_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_10_ladder_vf15_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
