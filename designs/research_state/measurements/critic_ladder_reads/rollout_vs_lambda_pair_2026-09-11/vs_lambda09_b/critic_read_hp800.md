# CRITIC READ — `ai_v12_23_ladder_rollout` vs `ai_v12_21_ladder_lambda09_b`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v6 at 2026-09-11T15:47:28. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

**SELF-PLAY CROSSING** — arm 2,162,688 · control 4,128,768 (first step carrying any `*_pool` scalar; the PROMOTION scalar itself lands one eval→rollout lag earlier, at 2,000,016 / 4,000,032). Descriptive — no floor, no verdict.

> 🚨 **CROSSING MISMATCH: arm 2,162,688 vs control 4,128,768.** The two runs seeded self-play at different steps, so the rows below compare heads fitted against opponent distributions that differ over the span between them. The first crossing is a coin flip on the SEED path — `SELF_PLAY_START = 0.55` (`--self-play-start-wr`) against `win_rate_vs_bots`, inside the 2M-bots replicate floor — so this is a property of the DRAW, not of the lever under test. ⚠️ It is a RAMP, not a step: the early crosser's extra span runs at a self-play fraction rising from ~0.03, not at 0.9, so the size of the mismatch is the integral of `selfplay_fraction` over the span and not the span itself. This line SELECTS NOTHING — the crossing is POST-TREATMENT and may be a mediator (ledger 2026-09-11 CORRECTION).

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_23_ladder_rollout` | `step_10000032` | 9600 | 0.4% | 150/150 (100.0%) | no |
| control | `ai_v12_21_ladder_lambda09_b` | `step_10000032` | 9600 | 0.5% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 800 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260910, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_23_ladder_rollout` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_21_ladder_lambda09_b` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_23_ladder_rollout` | **794/316/8** | 9600 | 8291 / 1267 / 42 | 12 |
| control | `ai_v12_21_ladder_lambda09_b` | **795/371/8** | 9600 | 8142 / 1407 / 51 | 12 |

**Frames:** the realized per-opponent caps differ — arm 794/316/8, control 795/371/8 (traced W/L/D per opponent, counted on disk); control cut to 794/316/8

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **794/316/8** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/hp800_floor_v6.json` — 74 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0230 | +0.0253 | **-0.0024** | [-0.0075, +0.0034] | **WITHIN FLOOR** (|delta| <= floor 0.0074) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0103 | -0.0246 | **+0.0348** | [-0.0098, +0.0814] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.0717 | -0.0679 | **-0.0038** | [-0.1419, +0.1370] | **WITHIN FLOOR** (|delta| <= floor 0.1082) |
| skill · `bot` | — | +0.2195 | +0.1967 | **+0.0228** | [-0.0206, +0.0740] | **WITHIN FLOOR** (|delta| <= floor 0.1004) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0702 | +0.0518 | **+0.0184** | [+0.0007, +0.0364] | **WITHIN FLOOR** (|delta| <= floor 0.0533) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 794/316/8 | +0.0464 | +0.0686 | **-0.0222** | [-0.0370, -0.0073] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.3416 | +1.2568 | **+0.0848** | [-0.0007, +0.1694] | **WITHIN FLOOR** (|delta| <= floor 0.1085) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0390 | +0.0432 | **-0.0042** | [-0.0098, +0.0014] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0230 | +0.0253 | **-0.0024** | [-0.0075, +0.0034] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0074) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0568 | +0.0523 | **+0.0044** | [-0.0056, +0.0136] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0141) |
| reliability | `all` | `capture-rate (gauge)` | +0.0016 | +0.0006 | **+0.0010** | [+0.0001, +0.0020] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0004 | +0.0049 | **-0.0045** | [-0.0065, -0.0031] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0123 | +0.0062 | **+0.0062** | [-0.0001, +0.0127] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0167) |
| ece | `all` | `capture-rate (gauge)` | +0.0240 | +0.0221 | **+0.0019** | [-0.0079, +0.0128] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0144) |
| ece | `bot` | `capture-rate (gauge)` | +0.0142 | +0.0633 | **-0.0491** | [-0.0601, -0.0385] | 400 | **DETECTED** (CI clears the floor +0.0297) |
| ece | `pool` | `capture-rate (gauge)` | +0.0918 | +0.0667 | **+0.0250** | [-0.0075, +0.0515] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0822) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2582 | +0.2681 | **-0.0099** | [-0.0403, +0.0165] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2195 | +0.1967 | **+0.0228** | [-0.0206, +0.0740] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1004) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2161 | +0.2041 | **+0.0119** | [-0.0353, +0.0568] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1108) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0615 | -0.0004 | **+0.0620** | [+0.0352, +0.0881] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0829) |
| bias V - p_hat | `ALL` | `pop` | +0.0010 | -0.0416 | **+0.0426** | [+0.0213, +0.0635] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | +0.0010 | -0.0416 | **+0.0426** | [+0.0213, +0.0635] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0374 | +0.0057 | **+0.0318** | [-0.0069, +0.0731] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0948) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0229 | -0.0456 | **+0.0227** | [-0.0094, +0.0557] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0229 | -0.0456 | **+0.0227** | [-0.0094, +0.0557] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0742 | -0.0187 | **+0.0929** | [+0.0568, +0.1295] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0748) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0183 | -0.0458 | **+0.0641** | [+0.0359, +0.0932] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0556) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0183 | -0.0458 | **+0.0641** | [+0.0359, +0.0932] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0556) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0761 | +0.0162 | **+0.0599** | [+0.0027, +0.1186] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0821) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0103 | -0.0246 | **+0.0348** | [-0.0098, +0.0814] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0103 | -0.0246 | **+0.0348** | [-0.0098, +0.0814] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat | `bot` | `raw` | +0.0390 | -0.0216 | **+0.0606** | [+0.0294, +0.0920] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0575) |
| bias V - p_hat | `bot` | `pop` | -0.0179 | -0.0640 | **+0.0461** | [+0.0251, +0.0682] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0417) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0179 | -0.0640 | **+0.0461** | [+0.0251, +0.0682] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0417) |
| bias V - p_hat | `pool` | `raw` | +0.0970 | +0.0340 | **+0.0630** | [+0.0210, +0.1058] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1164) |
| bias V - p_hat | `pool` | `pop` | +0.0434 | -0.0004 | **+0.0437** | [+0.0041, +0.0843] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0434 | -0.0004 | **+0.0437** | [+0.0041, +0.0843] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| Murphy resolution | `ALL` | `raw` | +0.0305 | +0.0311 | **-0.0006** | [-0.0124, +0.0106] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0070) |
| Murphy resolution | `ALL` | `pop` | +0.0315 | +0.0383 | **-0.0068** | [-0.0224, +0.0081] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0315 | +0.0383 | **-0.0068** | [-0.0224, +0.0081] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1531 | +0.1878 | **-0.0348** | [-0.0988, +0.0270] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0370) |
| Murphy skill_score | `ALL` | `pop` | +0.2531 | +0.2636 | **-0.0104** | [-0.1010, +0.0803] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2531 | +0.2636 | **-0.0104** | [-0.1010, +0.0803] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1821 | +0.1907 | **-0.0086** | [-0.0716, +0.0499] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0188) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2506 | +0.2745 | **-0.0239** | [-0.1150, +0.0652] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2506 | +0.2745 | **-0.0239** | [-0.1150, +0.0652] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0065 | +0.0015 | **+0.0050** | [+0.0009, +0.0094] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0119) |
| Murphy reliability | `ALL` | `pop` | +0.0006 | +0.0022 | **-0.0016** | [-0.0044, +0.0009] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0022) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0006 | +0.0022 | **-0.0016** | [-0.0044, +0.0009] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0022) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.0717 | -0.0679 | **-0.0038** | [-0.1419, +0.1370] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1082) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0927 | -0.0668 | **-0.0259** | [-0.1492, +0.1002] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0984) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0927 | -0.0668 | **-0.0259** | [-0.1492, +0.1002] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0984) |

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

**arm — `ai_v12_23_ladder_rollout` @ `step_10000032`**: 286865 states / 9558 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 42 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_21_ladder_lambda09_b` @ `step_10000032`**: 300133 states / 9549 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 51 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0702 | +0.0518 | **+0.0184** | [+0.0007, +0.0364] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0533) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 794/316/8 | +0.0730 | +0.0542 | **+0.0188** | [+0.0000, +0.0360] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1034 | -0.1192 | **+0.0158** | [+0.0028, +0.0287] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0071) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.4685 | +0.5070 | **-0.0385** | [-0.1032, +0.0255] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1973) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 794/316/8 | +0.4671 | +0.5051 | **-0.0379** | [-0.1038, +0.0213] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0591 | -0.0620 | **+0.0029** | [-0.0105, +0.0165] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0218) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 4-10` | as traced | +0.2743 | +0.2415 | **+0.0329** | [-0.0040, +0.0691] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2179) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 11-24` | as traced | +0.5285 | +0.5384 | **-0.0099** | [-0.0844, +0.0640] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1920) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 794/316/8 | +0.0464 | +0.0686 | **-0.0222** | [-0.0370, -0.0073] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 794/316/8 | +0.0074 | +0.0082 | **-0.0008** | [-0.0058, +0.0039] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0025) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 794/316/8 | +0.4995 | +0.4930 | **+0.0065** | [-0.0128, +0.0254] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1-3` | MATCHED 794/316/8 | +0.5520 | +0.5591 | **-0.0071** | [-0.0243, +0.0094] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 4-10` | MATCHED 794/316/8 | +0.6875 | +0.6938 | **-0.0063** | [-0.0208, +0.0085] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0219) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 1-3` | MATCHED 794/316/8 | +0.0140 | +0.0150 | **-0.0010** | [-0.0039, +0.0017] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 4-10` | MATCHED 794/316/8 | +0.1089 | +0.0867 | **+0.0222** | [+0.0090, +0.0339] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 11-24` | MATCHED 794/316/8 | +0.1966 | +0.1631 | **+0.0335** | [+0.0109, +0.0539] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 794/316/8 | +0.0495 | +0.0576 | **-0.0080** | [-0.0126, -0.0042] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0044) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 794/316/8 | +0.0306 | +0.0343 | **-0.0037** | [-0.0072, -0.0003] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0021) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 794/316/8 | +0.9433 | +0.9382 | **+0.0051** | [-0.0519, +0.0981] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 794/316/8 | +0.4606 | +0.4390 | **+0.0216** | [-0.0134, +0.0522] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2507) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 794/316/8 | +0.0057 | +0.0048 | **+0.0009** | [-0.0064, +0.0078] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0012) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 794/316/8 | +0.0407 | +0.0636 | **-0.0228** | [-0.0390, -0.0068] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.3416 | +1.2568 | **+0.0848** | [-0.0007, +0.1694] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1085) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | -0.3915 | +0.2103 | **-0.6019** | [-0.7414, -0.4606] | 2000 | **DETECTED** (CI clears the floor +0.2642) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.9573 | +1.0389 | **-0.0816** | [-0.2475, +0.0771] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1802) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.6045 | +0.3380 | **+0.2665** | [+0.0512, +0.4945] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3514) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 794/316/8 | +1.2828 | +1.1980 | **+0.0847** | [+0.0021, +0.1691] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.3584 | +1.3958 | **-0.0374** | [-0.1352, +0.0607] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1195) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | -0.4197 | +0.0229 | **-0.4426** | [-0.5980, -0.2896] | 2000 | **DETECTED** (CI clears the floor +0.2778) |

### HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY — `V` against its opponent-decodable part

| side | window | ratio of `V` | opponent-decodable part | `V` / decodable |
|---|---|---|---|---|
| arm `ai_v12_23_ladder_rollout` | `t1_3` | +0.0702 | +0.0140 | 5.006 |
| arm `ai_v12_23_ladder_rollout` | `t4_10` | +0.2743 | +0.1089 | 2.519 |
| arm `ai_v12_23_ladder_rollout` | `t11_24` | +0.5285 | +0.1966 | 2.688 |
| control `ai_v12_21_ladder_lambda09_b` | `t1_3` | +0.0518 | +0.0150 | 3.467 |
| control `ai_v12_21_ladder_lambda09_b` | `t4_10` | +0.2415 | +0.0869 | 2.779 |
| control `ai_v12_21_ladder_lambda09_b` | `t11_24` | +0.5384 | +0.1629 | 3.305 |

`V_opt(s) = Σ_o q(o | V(s)) · p_o` — an out-of-fold, battle-grouped, binned posterior over the pinned opponent roster from this side's own recorded `V`, times each opponent's manifest win rate, put through the SAME noise-corrected between-opponent spread the row above it uses. It is the between-opponent spread a head conditioning ONLY on which opponent its own output reveals would emit.

> 🚨 **IT IS NOT AN UPPER BOUND, AND `V` ROUTINELY EXCEEDS IT.** `E[p_o | V]` is a per-state CONDITIONAL MEAN, and a conditional mean ATTENUATES: the between-cell spread of a shrinking transform of `V` is smaller than the between-cell spread of `V` itself. So the excess in the last column is not a paradox — it is between-opponent spread riding on **BOARD STATE** that differs by opponent (you are ahead by turn 12 against a weak bot, and `V` says so without recognising the bot) rather than on opponent IDENTITY. Read the two columns as a DECOMPOSITION; never quote the decodable part as "the maximum".

> ⚠️ **Decoded from the head's OUTPUT, not its information set.** The block does no model forward, so the posterior is `q(o | V)` and not `q(o | value_pooled)`: this is a LOWER bound on how much opponent identity the head's FEATURES carry. **Descriptive: never a pass/fail bar, and the delta between two sides is never DETECTED.**

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
| arm · team | 600 | 602 | 9552 | 19104 | 13.0 | 26.0 | 4–94 |
| arm · stratum | 5 | 602 | 9552 | 19104 | 1908.0 | 3816.0 | 1895–1929 |
| arm · between-team spread | 600 | 602 | 9552 | — | 13.0 | — | — |
| control · team | 600 | 602 | 9543 | 19086 | 13.0 | 26.0 | 4–94 |
| control · stratum | 5 | 602 | 9543 | 19086 | 1909.0 | 3818.0 | 1890–1923 |
| control · between-team spread | 600 | 602 | 9543 | — | 13.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 794/316/8 | +0.0495 | +0.0576 | **-0.0080** | [-0.0126, -0.0042] | **NOT DETECTED** (CI does not clear the floor 0.0044) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 794/316/8 | +0.0306 | +0.0343 | **-0.0037** | [-0.0072, -0.0003] | **NOT DETECTED** (CI does not clear the floor 0.0021) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 794/316/8 | +0.9433 | +0.9382 | **+0.0051** | [-0.0519, +0.0981] | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 794/316/8 | +0.0464 | +0.0686 | **-0.0222** | [-0.0370, -0.0073] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 794/316/8 | +0.0057 | +0.0048 | **+0.0009** | [-0.0064, +0.0078] | **WITHIN FLOOR** (|delta| <= floor 0.0012) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 794/316/8 | +0.0407 | +0.0636 | **-0.0228** | [-0.0390, -0.0068] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

The two own-team R² rows are printed with them: the contrast is their difference, and it cannot be read without seeing which end moved.

**Reading of the three signs:** (B) SUBSTITUTION — the between-team spread is UP while the WITHIN-team discrimination is DOWN: the head is right about which team it holds and worse inside one.

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
| arm · all | `all` | 19116 | 9558 | 0.8598 | **0.1590** | **1.4917** | [0.357, 0.996] | 0.0001 |
| arm · t1_3 | `t1_3` | 19116 | 9558 | 0.7934 | **0.0792** | **0.5147** | [0.618, 0.927] | 0.0000 |
| arm · common support | `common support` | 17232 | 9364 | 0.8675 | **0.1166** | **1.1146** | [0.546, 0.989] | 0.0000 |
| control · all | `all` | 19098 | 9549 | 0.8023 | **0.1937** | **1.4416** | [0.171, 0.991] | 0.0001 |
| control · t1_3 | `t1_3` | 19098 | 9549 | 0.7989 | **0.0740** | **0.4677** | [0.621, 0.917] | 0.0000 |
| control · common support | `common support` | 17687 | 9383 | 0.8304 | **0.1309** | **1.0619** | [0.473, 0.986] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.3575, 0.9914]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.3416 | +1.2568 | **+0.0848** | [-0.0007, +0.1694] | 0.1085 | **WITHIN FLOOR** (|delta| <= floor 0.1085) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | -0.3915 | +0.2103 | **-0.6019** | [-0.7414, -0.4606] | 0.2642 | **DETECTED** (CI clears the floor +0.2642) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.9573 | +1.0389 | **-0.0816** | [-0.2475, +0.0771] | 0.1802 | **WITHIN FLOOR** (|delta| <= floor 0.1802) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.6045 | +0.3380 | **+0.2665** | [+0.0512, +0.4945] | 0.3514 | **WITHIN FLOOR** (|delta| <= floor 0.3514) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 794/316/8 | +1.2828 | +1.1980 | **+0.0847** | [+0.0021, +0.1691] | 0.1363 | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.3584 | +1.3958 | **-0.0374** | [-0.1352, +0.0607] | 0.1195 | **WITHIN FLOOR** (|delta| <= floor 0.1195) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | -0.4197 | +0.0229 | **-0.4426** | [-0.5980, -0.2896] | 0.2778 | **DETECTED** (CI clears the floor +0.2778) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.0730 [+0.0523, +0.0564] | +0.0542 | **+0.0188** | [+0.0000, +0.0360] | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.0730 [+0.0547, +0.0547] | +0.0547 | **+0.0183** | [+0.0014, +0.0356] | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.0730 | +0.0547 | **+0.0183** | [+0.0014, +0.0356] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.4671 [+0.5023, +0.5107] | +0.5051 | **-0.0379** | [-0.1038, +0.0213] | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.4671 [+0.5061, +0.5061] | +0.5061 | **-0.0390** | [-0.1031, +0.0243] | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.4671 | +0.5061 | **-0.0390** | [-0.1031, +0.0243] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.0464 [+0.0642, +0.0741] | +0.0686 | **-0.0222** | [-0.0370, -0.0073] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| `cond.own_team_r2.t1` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.0464 [+0.0711, +0.0711] | +0.0711 | **-0.0247** | [-0.0397, -0.0104] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0464 | +0.0711 | **-0.0247** | [-0.0397, -0.0104] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.0074 [+0.0066, +0.0098] | +0.0082 | **-0.0008** | [-0.0058, +0.0039] | **WITHIN FLOOR** (|delta| <= floor 0.0025) |
| `cond.own_team_r2.all` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.0074 [+0.0088, +0.0088] | +0.0088 | **-0.0014** | [-0.0063, +0.0034] | **WITHIN FLOOR** (|delta| <= floor 0.0025) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0074 | +0.0088 | **-0.0014** | [-0.0063, +0.0034] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.4995 [+0.4892, +0.4970] | +0.4930 | **+0.0065** | [-0.0128, +0.0254] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.4995 [+0.4842, +0.4842] | +0.4842 | **+0.0153** | [-0.0038, +0.0339] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.4995 | +0.4842 | **+0.0153** | [-0.0038, +0.0339] | *no label — not a reading* |
| `cond.opp_class_auc.t1_3` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.5520 [+0.5581, +0.5606] | +0.5591 | **-0.0071** | [-0.0243, +0.0094] | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| `cond.opp_class_auc.t1_3` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.5520 [+0.5601, +0.5601] | +0.5601 | **-0.0081** | [-0.0245, +0.0084] | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| `cond.opp_class_auc.t1_3` | UNMATCHED (as traced) | +0.5520 | +0.5601 | **-0.0081** | [-0.0245, +0.0084] | *no label — not a reading* |
| `cond.opp_class_auc.t4_10` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.6875 [+0.6907, +0.6954] | +0.6938 | **-0.0063** | [-0.0208, +0.0085] | **WITHIN FLOOR** (|delta| <= floor 0.0219) |
| `cond.opp_class_auc.t4_10` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.6875 [+0.6919, +0.6919] | +0.6919 | **-0.0044** | [-0.0196, +0.0108] | **WITHIN FLOOR** (|delta| <= floor 0.0219) |
| `cond.opp_class_auc.t4_10` | UNMATCHED (as traced) | +0.6875 | +0.6919 | **-0.0044** | [-0.0196, +0.0108] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.0140 [+0.0146, +0.0153] | +0.0150 | **-0.0010** | [-0.0039, +0.0017] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.0140 [+0.0150, +0.0150] | +0.0150 | **-0.0009** | [-0.0035, +0.0018] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | UNMATCHED (as traced) | +0.0140 | +0.0150 | **-0.0009** | [-0.0035, +0.0018] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.1089 [+0.0860, +0.0876] | +0.0867 | **+0.0222** | [+0.0090, +0.0339] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.1089 [+0.0869, +0.0869] | +0.0869 | **+0.0220** | [+0.0096, +0.0345] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | UNMATCHED (as traced) | +0.1089 | +0.0869 | **+0.0220** | [+0.0096, +0.0345] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.1966 [+0.1624, +0.1639] | +0.1631 | **+0.0335** | [+0.0109, +0.0539] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.1966 [+0.1629, +0.1629] | +0.1629 | **+0.0337** | [+0.0119, +0.0562] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | UNMATCHED (as traced) | +0.1966 | +0.1629 | **+0.0337** | [+0.0119, +0.0562] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.0495 [+0.0569, +0.0583] | +0.0576 | **-0.0080** | [-0.0126, -0.0042] | **NOT DETECTED** (CI does not clear the floor 0.0044) |
| `cond.within_team_resolution.all` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.0495 [+0.0580, +0.0580] | +0.0580 | **-0.0085** | [-0.0132, -0.0047] | **DETECTED** (CI clears the floor +0.0044) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0495 | +0.0580 | **-0.0085** | [-0.0132, -0.0047] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.0306 [+0.0337, +0.0347] | +0.0343 | **-0.0037** | [-0.0072, -0.0003] | **NOT DETECTED** (CI does not clear the floor 0.0021) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.0306 [+0.0343, +0.0343] | +0.0343 | **-0.0037** | [-0.0073, -0.0001] | **NOT DETECTED** (CI does not clear the floor 0.0021) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0306 | +0.0343 | **-0.0037** | [-0.0073, -0.0001] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.9433 [+0.8711, +1.0237] | +0.9382 | **+0.0051** | [-0.0519, +0.0981] | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.9433 [+0.9553, +0.9553] | +0.9553 | **-0.0121** | [-0.0619, +0.0929] | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.9433 | +0.9553 | **-0.0121** | [-0.0619, +0.0929] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.4606 [+0.4342, +0.4443] | +0.4390 | **+0.0216** | [-0.0134, +0.0522] | **WITHIN FLOOR** (|delta| <= floor 0.2507) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.4606 [+0.4448, +0.4448] | +0.4448 | **+0.0158** | [-0.0180, +0.0493] | **WITHIN FLOOR** (|delta| <= floor 0.2507) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.4606 | +0.4448 | **+0.0158** | [-0.0180, +0.0493] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.0057 [+0.0030, +0.0076] | +0.0048 | **+0.0009** | [-0.0064, +0.0078] | **WITHIN FLOOR** (|delta| <= floor 0.0012) |
| `cond.own_team_r2.late` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.0057 [+0.0045, +0.0045] | +0.0045 | **+0.0012** | [-0.0062, +0.0082] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0057 | +0.0045 | **+0.0012** | [-0.0062, +0.0082] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +0.0407 [+0.0582, +0.0689] | +0.0636 | **-0.0228** | [-0.0390, -0.0068] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +0.0407 [+0.0666, +0.0666] | +0.0666 | **-0.0259** | [-0.0412, -0.0105] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0407 | +0.0666 | **-0.0259** | [-0.0412, -0.0105] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 794/316/8 (9493 battles, 9487 decoder battles, 21 seeds) | +1.2828 [+1.1907, +1.2081] | +1.1980 | **+0.0847** | [+0.0021, +0.1691] | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 992/395/8 (9549 battles, 9543 decoder battles, 21 seeds) | +1.2828 [+1.2180, +1.2180] | +1.2180 | **+0.0648** | [-0.0209, +0.1510] | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.2828 | +1.2180 | **+0.0648** | [-0.0209, +0.1510] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0702 | [+0.0594, +0.0866] | +0.0518 | [+0.0432, +0.0661] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0730 | [+0.0626, +0.0888] | +0.0547 | [+0.0465, +0.0683] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1034 | [-0.1130, -0.0951] | -0.1192 | [-0.1291, -0.1102] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.4685 | [+0.4243, +0.5141] | +0.5070 | [+0.4628, +0.5516] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.4671 | [+0.4234, +0.5121] | +0.5061 | [+0.4622, +0.5502] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0591 | [-0.0693, -0.0508] | -0.0620 | [-0.0723, -0.0530] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 4-10` | +0.2743 | [+0.2480, +0.3021] | +0.2415 | [+0.2187, +0.2676] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 11-24` | +0.5285 | [+0.4766, +0.5825] | +0.5384 | [+0.4918, +0.5885] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0464 | [+0.0373, +0.0549] | +0.0711 | [+0.0594, +0.0827] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0074 | [+0.0041, +0.0106] | +0.0088 | [+0.0050, +0.0122] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.4995 | [+0.4865, +0.5130] | +0.4842 | [+0.4708, +0.4980] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1-3` | +0.5520 | [+0.5408, +0.5632] | +0.5601 | [+0.5482, +0.5710] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 4-10` | +0.6875 | [+0.6772, +0.6979] | +0.6919 | [+0.6814, +0.7017] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 1-3` | +0.0140 | [+0.0125, +0.0161] | +0.0150 | [+0.0133, +0.0172] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 4-10` | +0.1089 | [+0.0992, +0.1187] | +0.0869 | [+0.0794, +0.0945] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 11-24` | +0.1966 | [+0.1796, +0.2138] | +0.1629 | [+0.1503, +0.1766] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0495 | [+0.0497, +0.0554] | +0.0580 | [+0.0585, +0.0645] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0306 | [+0.0283, +0.0334] | +0.0343 | [+0.0319, +0.0374] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.9433 | [+0.4119, +0.5277] | +0.9553 | [+0.3996, +0.5068] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.4606 | [+0.3343, +0.3846] | +0.4448 | [+0.3212, +0.3669] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0057 | [-0.0003, +0.0109] | +0.0045 | [-0.0002, +0.0089] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0407 | [+0.0309, +0.0506] | +0.0666 | [+0.0551, +0.0789] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.3416 | [+1.2766, +1.4100] | +1.2568 | [+1.2032, +1.3163] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | -0.3915 | [-0.5002, -0.2863] | +0.2103 | [+0.1189, +0.3008] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.9573 | [+0.8492, +1.0777] | +1.0389 | [+0.9286, +1.1588] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.6045 | [+0.4507, +0.7533] | +0.3380 | [+0.1788, +0.4880] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.2828 | [+1.2171, +1.3513] | +1.2180 | [+1.1639, +1.2785] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.3584 | [+1.2868, +1.4344] | +1.3958 | [+1.3320, +1.4647] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | -0.4197 | [-0.5400, -0.3048] | +0.0229 | [-0.0791, +0.1211] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_23_ladder_rollout` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": true, "n_gated_rows": 9}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0277 [+0.0202, +0.0382] | 0.0618 | -0.0342 | +0.1964 [+0.1468, +0.2410] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0217 [+0.0140, +0.0333] | 0.0337 | -0.0119 | +0.2027 [+0.1340, +0.2649] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0303 [+0.0195, +0.0471] | 0.0711 | -0.0408 | +0.1684 [+0.0877, +0.2305] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 743 battles / 6400 rollouts · Brier 0.0938 = REL 0.0006 − RES 0.0315 + UNC 0.1256 + WBV 0.0008 (resid -1.69e-03) · base rate 0.8527 · **resolution is 25.1% of the base-rate cap** · corr(turn,V) -0.1588 vs corr(turn,MC) -0.0872

**control — `ai_v12_21_ladder_lambda09_b` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0461 [+0.0335, +0.0644] | 0.0618 | -0.0157 | +0.2862 [+0.2070, +0.3557] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0239 [+0.0161, +0.0362] | 0.0337 | -0.0098 | +0.1803 [+0.0186, +0.2870] | ❌ | ❌ | ❌ | ✅ |
| `pool` | yes | 0.0570 [+0.0398, +0.0809] | 0.0711 | -0.0141 | +0.2806 [+0.1926, +0.3509] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 742 battles / 6400 rollouts · Brier 0.1027 = REL 0.0022 − RES 0.0383 + UNC 0.1394 + WBV 0.0008 (resid -1.42e-03) · base rate 0.8326 · **resolution is 27.4% of the base-rate cap** · corr(turn,V) -0.2052 vs corr(turn,MC) -0.1374

## 5. THE LEDGER LINE

```
ai_v12_23_ladder_rollout vs ai_v12_21_ladder_lambda09_b at 10M [OFFLINE-GENERATED: 800 games x 9+3 opponents, full capture]: G1 bot Δ -0.0024 [-0.0075, +0.0034] WITHIN FLOOR · identity bias late Δ +0.0348 [-0.0098, +0.0814] WITHIN FLOOR · turn-contrast Δ -0.0038 [-0.1419, +0.1370] WITHIN FLOOR · spread ratio t1-3 Δ +0.0184 [+0.0007, +0.0364] WITHIN FLOOR · own-team R2 t1 Δ -0.0222 [-0.0370, -0.0073] NOT DETECTED [QUOTA-MATCHED] · calib slope Δ +0.0848 [-0.0007, +0.1694] WITHIN FLOOR
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_23_ladder_rollout --control ai_v12_21_ladder_lambda09_b --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_23_ladder_rollout --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_21_ladder_lambda09_b --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/hp800_floor_v6.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_23_ladder_rollout_vs_lambda09_b --nice 15 --ledger-line

# arm — ai_v12_23_ladder_rollout  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_21_ladder_lambda09_b  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_23_ladder_rollout` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_23_ladder_rollout/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_23_ladder_rollout_vs_lambda09/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_23_ladder_rollout_vs_lambda09/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_23_ladder_rollout_vs_lambda09/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_21_ladder_lambda09_b` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_21_ladder_lambda09_b/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_21_ladder_lambda09_b_vs_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_21_ladder_lambda09_b_vs_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_21_ladder_lambda09_b_vs_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_23_ladder_rollout_vs_lambda09_b/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_23_ladder_rollout_vs_lambda09_b/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
