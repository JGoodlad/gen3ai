# CRITIC READ — `ai_v12_24_ladder_strata_b` vs `ai_v12_16_ladder_ctrl10M_c`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v6 at 2026-09-12T01:02:52. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

**SELF-PLAY CROSSING** — arm 4,128,768 · control 4,128,768 (first step carrying any `*_pool` scalar; the PROMOTION scalar itself lands one eval→rollout lag earlier, at 4,000,032 / 4,000,032). Descriptive — no floor, no verdict.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_24_ladder_strata_b` | `step_10000032` | 9600 | 0.4% | 150/150 (100.0%) | no |
| control | `ai_v12_16_ladder_ctrl10M_c` | `step_10000032` | 9600 | 0.6% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 800 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260910, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_24_ladder_strata_b` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_16_ladder_ctrl10M_c` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_24_ladder_strata_b` | **798/382/9** | 9600 | 8015 / 1551 / 34 | 12 |
| control | `ai_v12_16_ladder_ctrl10M_c` | **797/330/8** | 9600 | 8164 / 1382 / 54 | 12 |

**Frames:** the realized per-opponent caps differ — arm 798/382/9, control 797/330/8 (traced W/L/D per opponent, counted on disk); arm cut to 797/330/8

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm side is subsampled to caps **797/330/8** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/hp800_floor_v6.json` — 74 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0313 | +0.0244 | **+0.0068** | [+0.0000, +0.0133] | **WITHIN FLOOR** (|delta| <= floor 0.0074) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0717 | +0.0138 | **-0.0854** | [-0.1326, -0.0370] | **NOT DETECTED** (CI does not clear the floor 0.0681) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.1681 | +0.0715 | **-0.2396** | [-0.3839, -0.0952] | **NOT DETECTED** (CI does not clear the floor 0.1082) |
| skill · `bot` | — | +0.1995 | +0.2153 | **-0.0158** | [-0.0743, +0.0382] | **WITHIN FLOOR** (|delta| <= floor 0.1004) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.1163 | +0.0921 | **+0.0242** | [+0.0005, +0.0499] | **WITHIN FLOOR** (|delta| <= floor 0.0533) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 797/330/8 | +0.0484 | +0.0358 | **+0.0126** | [+0.0005, +0.0251] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.0349 | +1.1780 | **-0.1432** | [-0.2151, -0.0762] | **NOT DETECTED** (CI does not clear the floor 0.1085) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0483 | +0.0389 | **+0.0094** | [+0.0034, +0.0153] | 400 | **DETECTED** (CI clears the floor +0.0003) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0313 | +0.0244 | **+0.0068** | [+0.0000, +0.0133] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0074) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0625 | +0.0473 | **+0.0153** | [+0.0052, +0.0249] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0141) |
| reliability | `all` | `capture-rate (gauge)` | +0.0026 | +0.0001 | **+0.0025** | [+0.0015, +0.0036] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0021) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0082 | +0.0017 | **+0.0065** | [+0.0041, +0.0090] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0047) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0016 | +0.0075 | **-0.0059** | [-0.0108, -0.0023] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0167) |
| ece | `all` | `capture-rate (gauge)` | +0.0421 | +0.0093 | **+0.0327** | [+0.0205, +0.0408] | 400 | **DETECTED** (CI clears the floor +0.0144) |
| ece | `bot` | `capture-rate (gauge)` | +0.0700 | +0.0301 | **+0.0400** | [+0.0278, +0.0510] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0297) |
| ece | `pool` | `capture-rate (gauge)` | +0.0349 | +0.0845 | **-0.0496** | [-0.0744, -0.0227] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0822) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2832 | +0.2547 | **+0.0285** | [-0.0062, +0.0608] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.1995 | +0.2153 | **-0.0158** | [-0.0743, +0.0382] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1004) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2678 | +0.1820 | **+0.0858** | [+0.0407, +0.1351] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1108) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | -0.0080 | +0.0370 | **-0.0451** | [-0.0735, -0.0181] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0829) |
| bias V - p_hat | `ALL` | `pop` | -0.0581 | -0.0147 | **-0.0434** | [-0.0656, -0.0222] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0581 | -0.0147 | **-0.0434** | [-0.0656, -0.0222] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0049 | -0.0241 | **+0.0289** | [-0.0124, +0.0686] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0948) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0528 | -0.0621 | **+0.0093** | [-0.0245, +0.0421] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0528 | -0.0621 | **+0.0093** | [-0.0245, +0.0421] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| bias V - p_hat | `mid (11-24)` | `raw` | -0.0110 | +0.0584 | **-0.0694** | [-0.1082, -0.0310] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0748) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0495 | +0.0021 | **-0.0516** | [-0.0821, -0.0208] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0556) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0495 | +0.0021 | **-0.0516** | [-0.0821, -0.0208] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0556) |
| bias V - p_hat | `late (turn>=25)` | `raw` | -0.0203 | +0.0786 | **-0.0989** | [-0.1602, -0.0378] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0821) |
| bias V - p_hat | `late (turn>=25)` | `pop` | -0.0717 | +0.0138 | **-0.0854** | [-0.1326, -0.0370] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0681) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0717 | +0.0138 | **-0.0854** | [-0.1326, -0.0370] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0681) |
| bias V - p_hat | `bot` | `raw` | -0.0291 | -0.0001 | **-0.0290** | [-0.0621, +0.0047] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0575) |
| bias V - p_hat | `bot` | `pop` | -0.0771 | -0.0392 | **-0.0379** | [-0.0633, -0.0129] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0417) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0771 | -0.0392 | **-0.0379** | [-0.0633, -0.0129] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0417) |
| bias V - p_hat | `pool` | `raw` | +0.0338 | +0.1020 | **-0.0682** | [-0.1153, -0.0222] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1164) |
| bias V - p_hat | `pool` | `pop` | -0.0187 | +0.0421 | **-0.0608** | [-0.1056, -0.0161] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| bias V - p_hat ⭐ | `pool` | `ipw` | -0.0187 | +0.0421 | **-0.0608** | [-0.1056, -0.0161] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| Murphy resolution | `ALL` | `raw` | +0.0298 | +0.0284 | **+0.0013** | [-0.0097, +0.0135] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0070) |
| Murphy resolution | `ALL` | `pop` | +0.0370 | +0.0351 | **+0.0018** | [-0.0126, +0.0173] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0370 | +0.0351 | **+0.0018** | [-0.0126, +0.0173] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1524 | +0.1643 | **-0.0118** | [-0.0861, +0.0655] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0370) |
| Murphy skill_score | `ALL` | `pop` | +0.2240 | +0.2643 | **-0.0403** | [-0.1406, +0.0639] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2240 | +0.2643 | **-0.0403** | [-0.1406, +0.0639] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1756 | +0.1731 | **+0.0025** | [-0.0567, +0.0684] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0188) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2592 | +0.2657 | **-0.0065** | [-0.0897, +0.0819] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0177) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2592 | +0.2657 | **-0.0065** | [-0.0897, +0.0819] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0177) |
| Murphy reliability | `ALL` | `raw` | +0.0055 | +0.0027 | **+0.0027** | [-0.0014, +0.0076] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0119) |
| Murphy reliability | `ALL` | `pop` | +0.0059 | +0.0008 | **+0.0051** | [+0.0013, +0.0100] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0022) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0059 | +0.0008 | **+0.0051** | [+0.0013, +0.0100] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0022) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.1681 | +0.0715 | **-0.2396** | [-0.3839, -0.0952] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.1082) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.1466 | +0.0302 | **-0.1768** | [-0.3022, -0.0478] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0984) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.1466 | +0.0302 | **-0.1768** | [-0.3022, -0.0478] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0984) |

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

**arm — `ai_v12_24_ladder_strata_b` @ `step_10000032`**: 298999 states / 9566 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 34 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_16_ladder_ctrl10M_c` @ `step_10000032`**: 297960 states / 9546 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 54 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.1163 | +0.0921 | **+0.0242** | [+0.0005, +0.0499] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0533) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 797/330/8 | +0.1194 | +0.0942 | **+0.0252** | [+0.0027, +0.0510] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1101 | -0.1088 | **-0.0013** | [-0.0149, +0.0125] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0071) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.6091 | +0.4980 | **+0.1110** | [+0.0361, +0.1833] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1973) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 797/330/8 | +0.6077 | +0.4970 | **+0.1107** | [+0.0400, +0.1830] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0487 | -0.0602 | **+0.0115** | [-0.0029, +0.0257] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0218) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 4-10` | as traced | +0.3829 | +0.3463 | **+0.0366** | [-0.0165, +0.0862] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2179) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 11-24` | as traced | +0.7015 | +0.5946 | **+0.1070** | [+0.0196, +0.1922] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1920) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 797/330/8 | +0.0484 | +0.0358 | **+0.0126** | [+0.0005, +0.0251] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 797/330/8 | +0.0129 | +0.0075 | **+0.0054** | [+0.0002, +0.0106] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0025) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 797/330/8 | +0.5004 | +0.4992 | **+0.0012** | [-0.0180, +0.0194] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0040) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1-3` | MATCHED 797/330/8 | +0.5812 | +0.5676 | **+0.0136** | [-0.0030, +0.0304] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 4-10` | MATCHED 797/330/8 | +0.6804 | +0.6878 | **-0.0075** | [-0.0225, +0.0073] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0219) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 1-3` | MATCHED 797/330/8 | +0.0160 | +0.0137 | **+0.0023** | [-0.0005, +0.0054] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 4-10` | MATCHED 797/330/8 | +0.0874 | +0.0958 | **-0.0084** | [-0.0203, +0.0033] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 11-24` | MATCHED 797/330/8 | +0.1683 | +0.1707 | **-0.0023** | [-0.0224, +0.0182] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 797/330/8 | +0.0654 | +0.0557 | **+0.0097** | [+0.0057, +0.0143] | 2000 | **DETECTED** (CI clears the floor +0.0044) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 797/330/8 | +0.0384 | +0.0335 | **+0.0048** | [+0.0011, +0.0087] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0021) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 797/330/8 | +1.3873 | +1.1225 | **+0.2648** | [+0.0795, +0.2735] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 797/330/8 | +0.6744 | +0.4973 | **+0.1771** | [+0.0946, +0.1804] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2507) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 797/330/8 | +0.0046 | +0.0036 | **+0.0010** | [-0.0056, +0.0077] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0012) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 797/330/8 | +0.0438 | +0.0322 | **+0.0116** | [-0.0018, +0.0252] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.0349 | +1.1780 | **-0.1432** | [-0.2151, -0.0762] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.1085) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | +0.4728 | +0.0829 | **+0.3899** | [+0.2787, +0.5011] | 2000 | **DETECTED** (CI clears the floor +0.2642) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.8141 | +0.9126 | **-0.0985** | [-0.2255, +0.0266] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1802) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.5904 | +0.6313 | **-0.0409** | [-0.2098, +0.1259] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3514) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 797/330/8 | +0.9981 | +1.1301 | **-0.1320** | [-0.2049, -0.0640] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.0972 | +1.1750 | **-0.0778** | [-0.1574, +0.0048] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1195) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | +0.3938 | +0.0877 | **+0.3061** | [+0.1793, +0.4295] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.2778) |

### HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY — `V` against its opponent-decodable part

| side | window | ratio of `V` | opponent-decodable part | `V` / decodable |
|---|---|---|---|---|
| arm `ai_v12_24_ladder_strata_b` | `t1_3` | +0.1163 | +0.0159 | 7.310 |
| arm `ai_v12_24_ladder_strata_b` | `t4_10` | +0.3829 | +0.0873 | 4.386 |
| arm `ai_v12_24_ladder_strata_b` | `t11_24` | +0.7015 | +0.1683 | 4.169 |
| control `ai_v12_16_ladder_ctrl10M_c` | `t1_3` | +0.0921 | +0.0137 | 6.741 |
| control `ai_v12_16_ladder_ctrl10M_c` | `t4_10` | +0.3463 | +0.0958 | 3.614 |
| control `ai_v12_16_ladder_ctrl10M_c` | `t11_24` | +0.5946 | +0.1707 | 3.484 |

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
| arm · team | 600 | 602 | 9560 | 19120 | 13.0 | 26.0 | 4–93 |
| arm · stratum | 5 | 602 | 9560 | 19120 | 1910.0 | 3820.0 | 1893–1925 |
| arm · between-team spread | 600 | 602 | 9560 | — | 13.0 | — | — |
| control · team | 600 | 602 | 9540 | 19080 | 13.0 | 26.0 | 4–94 |
| control · stratum | 5 | 602 | 9540 | 19080 | 1911.0 | 3822.0 | 1895–1915 |
| control · between-team spread | 600 | 602 | 9540 | — | 13.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 797/330/8 | +0.0654 | +0.0557 | **+0.0097** | [+0.0057, +0.0143] | **DETECTED** (CI clears the floor +0.0044) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 797/330/8 | +0.0384 | +0.0335 | **+0.0048** | [+0.0011, +0.0087] | **NOT DETECTED** (CI does not clear the floor 0.0021) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 797/330/8 | +1.3873 | +1.1225 | **+0.2648** | [+0.0795, +0.2735] | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 797/330/8 | +0.0484 | +0.0358 | **+0.0126** | [+0.0005, +0.0251] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 797/330/8 | +0.0046 | +0.0036 | **+0.0010** | [-0.0056, +0.0077] | **WITHIN FLOOR** (|delta| <= floor 0.0012) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 797/330/8 | +0.0438 | +0.0322 | **+0.0116** | [-0.0018, +0.0252] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

The two own-team R² rows are printed with them: the contrast is their difference, and it cannot be read without seeing which end moved.

**Reading of the three signs:** (A) CONDITIONING — the between-team spread is UP and the within-team discrimination is not lower: the head has ADDED team information without spending board information.

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
| arm · all | `all` | 19132 | 9566 | 0.7844 | **0.2249** | **1.7802** | [0.100, 0.996] | 0.0000 |
| arm · t1_3 | `t1_3` | 19132 | 9566 | 0.7847 | **0.1151** | **0.7081** | [0.507, 0.942] | 0.0000 |
| arm · common support | `common support` | 17748 | 9390 | 0.8133 | **0.1692** | **1.3766** | [0.359, 0.993] | 0.0000 |
| control · all | `all` | 19092 | 9546 | 0.8267 | **0.1838** | **1.5943** | [0.247, 0.996] | 0.0005 |
| control · t1_3 | `t1_3` | 19092 | 9546 | 0.7802 | **0.0860** | **0.5309** | [0.583, 0.923] | 0.0000 |
| control · common support | `common support` | 18080 | 9447 | 0.8400 | **0.1477** | **1.3325** | [0.422, 0.993] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.2469, 0.9956]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.0349 | +1.1780 | **-0.1432** | [-0.2151, -0.0762] | 0.1085 | **NOT DETECTED** (CI does not clear the floor 0.1085) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | +0.4728 | +0.0829 | **+0.3899** | [+0.2787, +0.5011] | 0.2642 | **DETECTED** (CI clears the floor +0.2642) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.8141 | +0.9126 | **-0.0985** | [-0.2255, +0.0266] | 0.1802 | **WITHIN FLOOR** (|delta| <= floor 0.1802) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.5904 | +0.6313 | **-0.0409** | [-0.2098, +0.1259] | 0.3514 | **WITHIN FLOOR** (|delta| <= floor 0.3514) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 797/330/8 | +0.9981 | +1.1301 | **-0.1320** | [-0.2049, -0.0640] | 0.1363 | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.0972 | +1.1750 | **-0.0778** | [-0.1574, +0.0048] | 0.1195 | **WITHIN FLOOR** (|delta| <= floor 0.1195) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +0.3938 | +0.0877 | **+0.3061** | [+0.1793, +0.4295] | 0.2778 | **NOT DETECTED** (CI does not clear the floor 0.2778) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.1194 [+0.1165, +0.1217] | +0.0942 | **+0.0252** | [+0.0027, +0.0510] | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.1194 [+0.1179, +0.1205] | +0.0942 | **+0.0252** | [+0.0016, +0.0504] | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1194 | +0.0942 | **+0.0252** | [+0.0019, +0.0504] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.6077 [+0.6020, +0.6115] | +0.4970 | **+0.1107** | [+0.0400, +0.1830] | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.6078 [+0.6052, +0.6099] | +0.4970 | **+0.1108** | [+0.0410, +0.1827] | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.6078 | +0.4970 | **+0.1108** | [+0.0366, +0.1824] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.0484 [+0.0459, +0.0513] | +0.0358 | **+0.0126** | [+0.0005, +0.0251] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| `cond.own_team_r2.t1` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.0500 [+0.0486, +0.0518] | +0.0358 | **+0.0142** | [+0.0018, +0.0265] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0503 | +0.0358 | **+0.0145** | [+0.0014, +0.0268] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.0129 [+0.0116, +0.0137] | +0.0075 | **+0.0054** | [+0.0002, +0.0106] | **NOT DETECTED** (CI does not clear the floor 0.0025) |
| `cond.own_team_r2.all` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.0134 [+0.0126, +0.0142] | +0.0075 | **+0.0059** | [+0.0004, +0.0117] | **NOT DETECTED** (CI does not clear the floor 0.0025) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0123 | +0.0075 | **+0.0049** | [-0.0005, +0.0100] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.5004 [+0.4946, +0.5055] | +0.4992 | **+0.0012** | [-0.0180, +0.0194] | **WITHIN FLOOR** (|delta| <= floor 0.0040) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.5019 [+0.5000, +0.5048] | +0.4992 | **+0.0027** | [-0.0157, +0.0215] | **WITHIN FLOOR** (|delta| <= floor 0.0040) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5044 | +0.4992 | **+0.0052** | [-0.0141, +0.0246] | *no label — not a reading* |
| `cond.opp_class_auc.t1_3` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.5812 [+0.5804, +0.5826] | +0.5676 | **+0.0136** | [-0.0030, +0.0304] | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| `cond.opp_class_auc.t1_3` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.5799 [+0.5788, +0.5808] | +0.5676 | **+0.0123** | [-0.0039, +0.0288] | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| `cond.opp_class_auc.t1_3` | UNMATCHED (as traced) | +0.5807 | +0.5676 | **+0.0131** | [-0.0039, +0.0296] | *no label — not a reading* |
| `cond.opp_class_auc.t4_10` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.6804 [+0.6782, +0.6811] | +0.6878 | **-0.0075** | [-0.0225, +0.0073] | **WITHIN FLOOR** (|delta| <= floor 0.0219) |
| `cond.opp_class_auc.t4_10` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.6775 [+0.6759, +0.6781] | +0.6878 | **-0.0103** | [-0.0250, +0.0039] | **WITHIN FLOOR** (|delta| <= floor 0.0219) |
| `cond.opp_class_auc.t4_10` | UNMATCHED (as traced) | +0.6826 | +0.6878 | **-0.0052** | [-0.0199, +0.0094] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.0160 [+0.0154, +0.0165] | +0.0137 | **+0.0023** | [-0.0005, +0.0054] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.0159 [+0.0157, +0.0163] | +0.0137 | **+0.0022** | [-0.0005, +0.0054] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | UNMATCHED (as traced) | +0.0159 | +0.0137 | **+0.0022** | [-0.0007, +0.0052] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.0874 [+0.0867, +0.0888] | +0.0958 | **-0.0084** | [-0.0203, +0.0033] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.0876 [+0.0871, +0.0880] | +0.0958 | **-0.0082** | [-0.0199, +0.0036] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | UNMATCHED (as traced) | +0.0873 | +0.0958 | **-0.0085** | [-0.0206, +0.0026] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.1683 [+0.1673, +0.1691] | +0.1707 | **-0.0023** | [-0.0224, +0.0182] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.1685 [+0.1681, +0.1691] | +0.1707 | **-0.0022** | [-0.0225, +0.0178] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | UNMATCHED (as traced) | +0.1683 | +0.1707 | **-0.0024** | [-0.0231, +0.0174] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.0654 [+0.0643, +0.0662] | +0.0557 | **+0.0097** | [+0.0057, +0.0143] | **DETECTED** (CI clears the floor +0.0044) |
| `cond.within_team_resolution.all` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.0654 [+0.0650, +0.0660] | +0.0557 | **+0.0097** | [+0.0058, +0.0144] | **DETECTED** (CI clears the floor +0.0044) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0651 | +0.0557 | **+0.0095** | [+0.0057, +0.0144] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.0384 [+0.0376, +0.0388] | +0.0335 | **+0.0048** | [+0.0011, +0.0087] | **NOT DETECTED** (CI does not clear the floor 0.0021) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.0370 [+0.0366, +0.0372] | +0.0335 | **+0.0035** | [-0.0003, +0.0072] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0369 | +0.0335 | **+0.0034** | [-0.0003, +0.0074] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +1.3873 [+1.3221, +1.4717] | +1.1225 | **+0.2648** | [+0.0795, +0.2735] | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +1.3935 [+1.3724, +1.4299] | +1.1225 | **+0.2710** | [+0.0912, +0.2840] | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +1.3909 | +1.1225 | **+0.2685** | [+0.0897, +0.2841] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.6744 [+0.6676, +0.6830] | +0.4973 | **+0.1771** | [+0.0946, +0.1804] | **WITHIN FLOOR** (|delta| <= floor 0.2507) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.6789 [+0.6770, +0.6825] | +0.4973 | **+0.1816** | [+0.0991, +0.1831] | **WITHIN FLOOR** (|delta| <= floor 0.2507) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.6799 | +0.4973 | **+0.1826** | [+0.1011, +0.1845] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.0046 [+0.0023, +0.0062] | +0.0036 | **+0.0010** | [-0.0056, +0.0077] | **WITHIN FLOOR** (|delta| <= floor 0.0012) |
| `cond.own_team_r2.late` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.0045 [+0.0016, +0.0055] | +0.0036 | **+0.0009** | [-0.0055, +0.0075] | **WITHIN FLOOR** (|delta| <= floor 0.0012) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0052 | +0.0036 | **+0.0016** | [-0.0051, +0.0086] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.0438 [+0.0400, +0.0476] | +0.0322 | **+0.0116** | [-0.0018, +0.0252] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +0.0454 [+0.0441, +0.0485] | +0.0322 | **+0.0133** | [-0.0001, +0.0268] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0450 | +0.0322 | **+0.0129** | [-0.0008, +0.0261] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 797/330/8 (9513 battles, 9507 decoder battles, 21 seeds) | +0.9981 [+0.9899, +1.0040] | +1.1301 | **-0.1320** | [-0.2049, -0.0640] | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 897/371/8 (9555 battles, 9549 decoder battles, 21 seeds) | +1.0038 [+0.9984, +1.0081] | +1.1301 | **-0.1263** | [-0.1963, -0.0594] | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +0.9990 | +1.1301 | **-0.1311** | [-0.2035, -0.0618] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.1163 | [+0.0997, +0.1397] | +0.0921 | [+0.0799, +0.1094] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1194 | [+0.1034, +0.1422] | +0.0942 | [+0.0823, +0.1112] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1101 | [-0.1195, -0.1006] | -0.1088 | [-0.1182, -0.0997] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.6091 | [+0.5560, +0.6655] | +0.4980 | [+0.4536, +0.5461] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.6078 | [+0.5553, +0.6637] | +0.4970 | [+0.4529, +0.5446] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0487 | [-0.0591, -0.0390] | -0.0602 | [-0.0706, -0.0507] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 4-10` | +0.3829 | [+0.3482, +0.4227] | +0.3463 | [+0.3145, +0.3833] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 11-24` | +0.7015 | [+0.6418, +0.7674] | +0.5946 | [+0.5414, +0.6538] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0503 | [+0.0412, +0.0593] | +0.0358 | [+0.0275, +0.0440] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0123 | [+0.0081, +0.0162] | +0.0075 | [+0.0040, +0.0105] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5044 | [+0.4912, +0.5174] | +0.4992 | [+0.4864, +0.5122] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1-3` | +0.5807 | [+0.5690, +0.5931] | +0.5676 | [+0.5560, +0.5793] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 4-10` | +0.6826 | [+0.6721, +0.6938] | +0.6878 | [+0.6773, +0.6985] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 1-3` | +0.0159 | [+0.0141, +0.0186] | +0.0137 | [+0.0120, +0.0158] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 4-10` | +0.0873 | [+0.0800, +0.0955] | +0.0958 | [+0.0875, +0.1052] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 11-24` | +0.1683 | [+0.1544, +0.1826] | +0.1707 | [+0.1566, +0.1859] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0651 | [+0.0659, +0.0722] | +0.0557 | [+0.0559, +0.0621] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0369 | [+0.0346, +0.0402] | +0.0335 | [+0.0312, +0.0365] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +1.3909 | [+0.6119, +0.7677] | +1.1225 | [+0.4449, +0.5652] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.6799 | [+0.4930, +0.5619] | +0.4973 | [+0.3587, +0.4108] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0052 | [+0.0000, +0.0103] | +0.0036 | [-0.0013, +0.0079] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0450 | [+0.0355, +0.0553] | +0.0322 | [+0.0231, +0.0415] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.0349 | [+0.9929, +1.0803] | +1.1780 | [+1.1272, +1.2313] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | +0.4728 | [+0.3991, +0.5459] | +0.0829 | [-0.0049, +0.1676] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.8141 | [+0.7425, +0.8907] | +0.9126 | [+0.8125, +1.0166] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.5904 | [+0.4832, +0.6888] | +0.6313 | [+0.5012, +0.7642] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +0.9990 | [+0.9550, +1.0443] | +1.1301 | [+1.0799, +1.1831] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.0972 | [+1.0477, +1.1519] | +1.1750 | [+1.1162, +1.2351] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | +0.3938 | [+0.3098, +0.4735] | +0.0877 | [-0.0079, +0.1780] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_24_ladder_strata_b` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": true, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0490 [+0.0377, +0.0607] | 0.0618 | -0.0128 | +0.2530 [+0.1843, +0.3111] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0392 [+0.0287, +0.0561] | 0.0337 | +0.0055 | +0.2336 [+0.1537, +0.3026] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0577 [+0.0391, +0.0776] | 0.0711 | -0.0134 | +0.2482 [+0.1272, +0.3382] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 735 battles / 6400 rollouts · Brier 0.1107 = REL 0.0059 − RES 0.0370 + UNC 0.1426 + WBV 0.0008 (resid -1.66e-03) · base rate 0.8277 · **resolution is 25.9% of the base-rate cap** · corr(turn,V) -0.3237 vs corr(turn,MC) -0.1556

**control — `ai_v12_16_ladder_ctrl10M_c` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0459 [+0.0353, +0.0598] | 0.0618 | -0.0159 | +0.2817 [+0.2338, +0.3308] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0177 [+0.0113, +0.0255] | 0.0337 | -0.0159 | +0.1467 [-0.0075, +0.2393] | ❌ | ❌ | ❌ | ❌ |
| `pool` | yes | 0.0596 [+0.0453, +0.0775] | 0.0711 | -0.0114 | +0.2144 [+0.1224, +0.2921] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 736 battles / 6400 rollouts · Brier 0.0973 = REL 0.0008 − RES 0.0351 + UNC 0.1323 + WBV 0.0008 (resid -1.40e-03) · base rate 0.8431 · **resolution is 26.6% of the base-rate cap** · corr(turn,V) -0.0833 vs corr(turn,MC) -0.1548

## 5. THE LEDGER LINE

```
ai_v12_24_ladder_strata_b vs ai_v12_16_ladder_ctrl10M_c at 10M [OFFLINE-GENERATED: 800 games x 9+3 opponents, full capture]: G1 bot Δ +0.0068 [+0.0000, +0.0133] WITHIN FLOOR · identity bias late Δ -0.0854 [-0.1326, -0.0370] NOT DETECTED · turn-contrast Δ -0.2396 [-0.3839, -0.0952] NOT DETECTED · spread ratio t1-3 Δ +0.0242 [+0.0005, +0.0499] WITHIN FLOOR · own-team R2 t1 Δ +0.0126 [+0.0005, +0.0251] NOT DETECTED [QUOTA-MATCHED] · calib slope Δ -0.1432 [-0.2151, -0.0762] NOT DETECTED
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_24_ladder_strata_b --control ai_v12_16_ladder_ctrl10M_c --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_24_ladder_strata_b --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_16_ladder_ctrl10M_c --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/hp800_floor_v6.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_24_ladder_strata_b_vs_ctrl10M_c --nice 15 --ledger-line

# arm — ai_v12_24_ladder_strata_b  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_16_ladder_ctrl10M_c  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_24_ladder_strata_b` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_24_ladder_strata_b/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_24_ladder_strata_b_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_24_ladder_strata_b_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_24_ladder_strata_b_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_16_ladder_ctrl10M_c/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_24_ladder_strata_b_vs_ctrl10M_c/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_24_ladder_strata_b_vs_ctrl10M_c/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
