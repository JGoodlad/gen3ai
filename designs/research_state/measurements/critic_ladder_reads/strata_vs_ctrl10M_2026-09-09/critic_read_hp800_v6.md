# CRITIC READ — `ai_v12_17_ladder_strata` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v6 at 2026-09-11T12:13:32. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

**SELF-PLAY CROSSING** — arm 4,128,768 · control 4,128,768 (first step carrying any `*_pool` scalar; the PROMOTION scalar itself lands one eval→rollout lag earlier, at 4,000,032 / 4,000,032). Descriptive — no floor, no verdict.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_17_ladder_strata` | `step_10000032` | 9600 | 0.6% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 9600 | 0.4% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 800 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260910, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_17_ladder_strata` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_17_ladder_strata` | **797/350/10** | 9600 | 8190 / 1350 / 60 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **797/368/9** | 9600 | 8252 / 1305 / 43 | 12 |

**Frames:** the realized per-opponent caps differ — arm 797/350/10, control 797/368/9 (traced W/L/D per opponent, counted on disk); control cut to 797/350/9

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **797/350/9** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/hp800_floor_v6.json` — 74 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0225 | +0.0170 | **+0.0055** | [+0.0006, +0.0106] | **WITHIN FLOOR** (|delta| <= floor 0.0074) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0059 | -0.0189 | **+0.0248** | [-0.0177, +0.0691] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.0194 | +0.0386 | **-0.0192** | [-0.1391, +0.0979] | **WITHIN FLOOR** (|delta| <= floor 0.1082) |
| skill · `bot` | — | +0.2059 | +0.1237 | **+0.0822** | [+0.0242, +0.1429] | **WITHIN FLOOR** (|delta| <= floor 0.1004) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.1500 | +0.1454 | **+0.0046** | [-0.0252, +0.0305] | **WITHIN FLOOR** (|delta| <= floor 0.0533) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 797/350/9 | +0.0422 | +0.0473 | **-0.0051** | [-0.0177, +0.0075] | **WITHIN FLOOR** (|delta| <= floor 0.0109) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.0504 | +1.0695 | **-0.0191** | [-0.0889, +0.0522] | **WITHIN FLOOR** (|delta| <= floor 0.1085) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0412 | +0.0386 | **+0.0026** | [-0.0032, +0.0082] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0225 | +0.0170 | **+0.0055** | [+0.0006, +0.0106] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0074) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0576 | +0.0614 | **-0.0037** | [-0.0129, +0.0061] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0141) |
| reliability | `all` | `capture-rate (gauge)` | +0.0003 | +0.0010 | **-0.0007** | [-0.0014, -0.0000] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0017 | +0.0053 | **-0.0035** | [-0.0056, -0.0020] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0022 | +0.0019 | **+0.0002** | [-0.0023, +0.0023] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0167) |
| ece | `all` | `capture-rate (gauge)` | +0.0102 | +0.0237 | **-0.0135** | [-0.0206, -0.0040] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0144) |
| ece | `bot` | `capture-rate (gauge)` | +0.0239 | +0.0445 | **-0.0206** | [-0.0306, -0.0114] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0297) |
| ece | `pool` | `capture-rate (gauge)` | +0.0423 | +0.0403 | **+0.0021** | [-0.0213, +0.0238] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0822) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2790 | +0.2598 | **+0.0192** | [-0.0164, +0.0535] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2059 | +0.1237 | **+0.0822** | [+0.0242, +0.1429] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1004) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2590 | +0.2742 | **-0.0153** | [-0.0524, +0.0325] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1108) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0286 | +0.0164 | **+0.0123** | [-0.0150, +0.0389] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0829) |
| bias V - p_hat | `ALL` | `pop` | -0.0213 | -0.0363 | **+0.0150** | [-0.0056, +0.0361] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0213 | -0.0363 | **+0.0150** | [-0.0056, +0.0361] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0244 | -0.0107 | **+0.0352** | [-0.0070, +0.0774] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0948) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0432 | -0.0648 | **+0.0216** | [-0.0117, +0.0543] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0432 | -0.0648 | **+0.0216** | [-0.0117, +0.0543] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0151 | +0.0229 | **-0.0078** | [-0.0466, +0.0317] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0748) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0242 | -0.0195 | **-0.0047** | [-0.0325, +0.0247] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0556) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0242 | -0.0195 | **-0.0047** | [-0.0325, +0.0247] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0556) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0478 | +0.0411 | **+0.0067** | [-0.0456, +0.0602] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0821) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0059 | -0.0189 | **+0.0248** | [-0.0177, +0.0691] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0059 | -0.0189 | **+0.0248** | [-0.0177, +0.0691] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat | `bot` | `raw` | +0.0109 | +0.0029 | **+0.0081** | [-0.0236, +0.0392] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0575) |
| bias V - p_hat | `bot` | `pop` | -0.0346 | -0.0441 | **+0.0096** | [-0.0143, +0.0327] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0417) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0346 | -0.0441 | **+0.0096** | [-0.0143, +0.0327] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0417) |
| bias V - p_hat | `pool` | `raw` | +0.0595 | +0.0436 | **+0.0160** | [-0.0322, +0.0625] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1164) |
| bias V - p_hat | `pool` | `pop` | -0.0026 | -0.0190 | **+0.0164** | [-0.0271, +0.0606] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| bias V - p_hat ⭐ | `pool` | `ipw` | -0.0026 | -0.0190 | **+0.0164** | [-0.0271, +0.0606] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| Murphy resolution | `ALL` | `raw` | +0.0323 | +0.0274 | **+0.0049** | [-0.0068, +0.0166] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0070) |
| Murphy resolution | `ALL` | `pop` | +0.0390 | +0.0359 | **+0.0031** | [-0.0121, +0.0183] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0390 | +0.0359 | **+0.0031** | [-0.0121, +0.0183] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1766 | +0.1626 | **+0.0140** | [-0.0608, +0.0903] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0370) |
| Murphy skill_score | `ALL` | `pop` | +0.2714 | +0.2601 | **+0.0113** | [-0.0888, +0.1148] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2714 | +0.2601 | **+0.0113** | [-0.0888, +0.1148] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1945 | +0.1739 | **+0.0206** | [-0.0450, +0.0860] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2863 | +0.2834 | **+0.0030** | [-0.0868, +0.0906] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0177) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2863 | +0.2834 | **+0.0030** | [-0.0868, +0.0906] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0177) |
| Murphy reliability | `ALL` | `raw` | +0.0036 | +0.0024 | **+0.0012** | [-0.0025, +0.0047] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0119) |
| Murphy reliability | `ALL` | `pop` | +0.0019 | +0.0030 | **-0.0011** | [-0.0050, +0.0022] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0022) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0019 | +0.0030 | **-0.0011** | [-0.0050, +0.0022] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0022) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.0194 | +0.0386 | **-0.0192** | [-0.1391, +0.0979] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1082) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | +0.0548 | +0.0222 | **+0.0327** | [-0.0689, +0.1323] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0984) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0548 | +0.0222 | **+0.0327** | [-0.0689, +0.1323] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0984) |

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

**arm — `ai_v12_17_ladder_strata` @ `step_10000032`**: 312673 states / 9540 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 60 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 291604 states / 9557 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 43 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.1500 | +0.1454 | **+0.0046** | [-0.0252, +0.0305] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0533) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 797/350/9 | +0.1506 | +0.1478 | **+0.0028** | [-0.0269, +0.0290] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0960 | -0.1018 | **+0.0058** | [-0.0078, +0.0188] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0071) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.6725 | +0.6237 | **+0.0488** | [-0.0358, +0.1371] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1973) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 797/350/9 | +0.6703 | +0.6224 | **+0.0479** | [-0.0357, +0.1331] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0370 | -0.0448 | **+0.0078** | [-0.0062, +0.0221] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0218) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 4-10` | as traced | +0.5305 | +0.5080 | **+0.0225** | [-0.0450, +0.0919] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2179) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 11-24` | as traced | +0.7546 | +0.7039 | **+0.0507** | [-0.0477, +0.1488] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1920) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 797/350/9 | +0.0422 | +0.0473 | **-0.0051** | [-0.0177, +0.0075] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0109) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 797/350/9 | +0.0075 | +0.0094 | **-0.0019** | [-0.0067, +0.0029] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0025) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 797/350/9 | +0.4838 | +0.5013 | **-0.0175** | [-0.0362, +0.0004] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1-3` | MATCHED 797/350/9 | +0.6206 | +0.5806 | **+0.0400** | [+0.0245, +0.0563] | 2000 | **DETECTED** (CI clears the floor +0.0153) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 4-10` | MATCHED 797/350/9 | +0.7599 | +0.7103 | **+0.0496** | [+0.0357, +0.0640] | 2000 | **DETECTED** (CI clears the floor +0.0219) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 1-3` | MATCHED 797/350/9 | +0.0298 | +0.0179 | **+0.0119** | [+0.0076, +0.0162] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 4-10` | MATCHED 797/350/9 | +0.1480 | +0.1266 | **+0.0214** | [+0.0039, +0.0394] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 11-24` | MATCHED 797/350/9 | +0.1854 | +0.1867 | **-0.0013** | [-0.0254, +0.0217] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 797/350/9 | +0.0582 | +0.0539 | **+0.0043** | [-0.0006, +0.0081] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 797/350/9 | +0.0336 | +0.0313 | **+0.0022** | [-0.0015, +0.0063] | 2000 | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 797/350/9 | +0.9394 | +1.2573 | **-0.3179** | [-0.3572, -0.1674] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 797/350/9 | +0.4211 | +0.6778 | **-0.2567** | [-0.2483, -0.1636] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.2507) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 797/350/9 | +0.0020 | +0.0047 | **-0.0027** | [-0.0083, +0.0026] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 797/350/9 | +0.0402 | +0.0429 | **-0.0027** | [-0.0155, +0.0105] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.0504 | +1.0695 | **-0.0191** | [-0.0889, +0.0522] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1085) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | +0.2140 | +0.3471 | **-0.1330** | [-0.2471, -0.0190] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2642) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.8244 | +0.7324 | **+0.0919** | [-0.0289, +0.2109] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1802) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.3598 | +0.9827 | **-0.6229** | [-0.8175, -0.4284] | 2000 | **DETECTED** (CI clears the floor +0.3514) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 797/350/9 | +1.0104 | +0.9941 | **+0.0163** | [-0.0539, +0.0856] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.0586 | +1.0542 | **+0.0044** | [-0.0775, +0.0859] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1195) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | +0.1992 | +0.3683 | **-0.1691** | [-0.2960, -0.0399] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2778) |

### HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY — `V` against its opponent-decodable part

| side | window | ratio of `V` | opponent-decodable part | `V` / decodable |
|---|---|---|---|---|
| arm `ai_v12_17_ladder_strata` | `t1_3` | +0.1500 | +0.0298 | 5.028 |
| arm `ai_v12_17_ladder_strata` | `t4_10` | +0.5305 | +0.1480 | 3.585 |
| arm `ai_v12_17_ladder_strata` | `t11_24` | +0.7546 | +0.1854 | 4.070 |
| control `ai_v12_11_ladder_ctrl10M` | `t1_3` | +0.1454 | +0.0181 | 8.045 |
| control `ai_v12_11_ladder_ctrl10M` | `t4_10` | +0.5080 | +0.1264 | 4.019 |
| control `ai_v12_11_ladder_ctrl10M` | `t11_24` | +0.7039 | +0.1871 | 3.763 |

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
| arm · team | 600 | 602 | 9534 | 19068 | 13.0 | 26.0 | 4–94 |
| arm · stratum | 5 | 602 | 9534 | 19068 | 1906.0 | 3812.0 | 1892–1925 |
| arm · between-team spread | 600 | 602 | 9534 | — | 13.0 | — | — |
| control · team | 600 | 602 | 9551 | 19102 | 13.0 | 26.0 | 4–93 |
| control · stratum | 5 | 602 | 9551 | 19102 | 1910.0 | 3820.0 | 1898–1919 |
| control · between-team spread | 600 | 602 | 9551 | — | 13.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 797/350/9 | +0.0582 | +0.0539 | **+0.0043** | [-0.0006, +0.0081] | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 797/350/9 | +0.0336 | +0.0313 | **+0.0022** | [-0.0015, +0.0063] | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 797/350/9 | +0.9394 | +1.2573 | **-0.3179** | [-0.3572, -0.1674] | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 797/350/9 | +0.0422 | +0.0473 | **-0.0051** | [-0.0177, +0.0075] | **WITHIN FLOOR** (|delta| <= floor 0.0109) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 797/350/9 | +0.0020 | +0.0047 | **-0.0027** | [-0.0083, +0.0026] | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 797/350/9 | +0.0402 | +0.0429 | **-0.0027** | [-0.0155, +0.0105] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

The two own-team R² rows are printed with them: the contrast is their difference, and it cannot be read without seeing which end moved.

**Reading of the three signs:** neither cleanly — the WITHIN-team discrimination is up without a between-team spread to go with it, which is a board-reading gain, not a conditioning one.

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
| arm · all | `all` | 19080 | 9540 | 0.8339 | **0.2034** | **1.8900** | [0.163, 0.998] | 0.0069 |
| arm · t1_3 | `t1_3` | 19080 | 9540 | 0.8497 | **0.0805** | **0.6216** | [0.647, 0.956] | 0.0000 |
| arm · common support | `common support` | 17335 | 9352 | 0.8486 | **0.1568** | **1.4267** | [0.376, 0.995] | 0.0000 |
| control · all | `all` | 19114 | 9557 | 0.8236 | **0.1990** | **1.7456** | [0.214, 0.996] | 0.0001 |
| control · t1_3 | `t1_3` | 19114 | 9557 | 0.7656 | **0.1128** | **0.6542** | [0.480, 0.931] | 0.0000 |
| control · common support | `common support` | 18158 | 9431 | 0.8379 | **0.1643** | **1.4944** | [0.366, 0.995] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.2144, 0.9965]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.0504 | +1.0695 | **-0.0191** | [-0.0889, +0.0522] | 0.1085 | **WITHIN FLOOR** (|delta| <= floor 0.1085) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | +0.2140 | +0.3471 | **-0.1330** | [-0.2471, -0.0190] | 0.2642 | **WITHIN FLOOR** (|delta| <= floor 0.2642) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.8244 | +0.7324 | **+0.0919** | [-0.0289, +0.2109] | 0.1802 | **WITHIN FLOOR** (|delta| <= floor 0.1802) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.3598 | +0.9827 | **-0.6229** | [-0.8175, -0.4284] | 0.3514 | **DETECTED** (CI clears the floor +0.3514) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 797/350/9 | +1.0104 | +0.9941 | **+0.0163** | [-0.0539, +0.0856] | 0.1363 | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.0586 | +1.0542 | **+0.0044** | [-0.0775, +0.0859] | 0.1195 | **WITHIN FLOOR** (|delta| <= floor 0.1195) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +0.1992 | +0.3683 | **-0.1691** | [-0.2960, -0.0399] | 0.2778 | **WITHIN FLOOR** (|delta| <= floor 0.2778) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.1506 [+0.1465, +0.1498] | +0.1478 | **+0.0028** | [-0.0269, +0.0290] | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.1506 [+0.1465, +0.1498] | +0.1478 | **+0.0028** | [-0.0269, +0.0290] | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1506 | +0.1477 | **+0.0029** | [-0.0264, +0.0285] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.6703 [+0.6188, +0.6254] | +0.6224 | **+0.0479** | [-0.0357, +0.1331] | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.6703 [+0.6188, +0.6254] | +0.6224 | **+0.0479** | [-0.0357, +0.1331] | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.6703 | +0.6221 | **+0.0482** | [-0.0356, +0.1353] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0422 [+0.0454, +0.0491] | +0.0473 | **-0.0051** | [-0.0177, +0.0075] | **WITHIN FLOOR** (|delta| <= floor 0.0109) |
| `cond.own_team_r2.t1` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0422 [+0.0454, +0.0491] | +0.0473 | **-0.0051** | [-0.0177, +0.0075] | **WITHIN FLOOR** (|delta| <= floor 0.0109) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0422 | +0.0475 | **-0.0052** | [-0.0178, +0.0071] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0075 [+0.0089, +0.0105] | +0.0094 | **-0.0019** | [-0.0067, +0.0029] | **WITHIN FLOOR** (|delta| <= floor 0.0025) |
| `cond.own_team_r2.all` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0075 [+0.0089, +0.0105] | +0.0094 | **-0.0019** | [-0.0067, +0.0029] | **WITHIN FLOOR** (|delta| <= floor 0.0025) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0075 | +0.0105 | **-0.0030** | [-0.0083, +0.0021] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.4838 [+0.4994, +0.5048] | +0.5013 | **-0.0175** | [-0.0362, +0.0004] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.4838 [+0.4994, +0.5048] | +0.5013 | **-0.0175** | [-0.0362, +0.0004] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.4838 | +0.5014 | **-0.0175** | [-0.0358, +0.0014] | *no label — not a reading* |
| `cond.opp_class_auc.t1_3` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.6206 [+0.5792, +0.5820] | +0.5806 | **+0.0400** | [+0.0245, +0.0563] | **DETECTED** (CI clears the floor +0.0153) |
| `cond.opp_class_auc.t1_3` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.6206 [+0.5792, +0.5820] | +0.5806 | **+0.0400** | [+0.0245, +0.0563] | **DETECTED** (CI clears the floor +0.0153) |
| `cond.opp_class_auc.t1_3` | UNMATCHED (as traced) | +0.6206 | +0.5796 | **+0.0410** | [+0.0244, +0.0577] | *no label — not a reading* |
| `cond.opp_class_auc.t4_10` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.7599 [+0.7094, +0.7111] | +0.7103 | **+0.0496** | [+0.0357, +0.0640] | **DETECTED** (CI clears the floor +0.0219) |
| `cond.opp_class_auc.t4_10` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.7599 [+0.7094, +0.7111] | +0.7103 | **+0.0496** | [+0.0357, +0.0640] | **DETECTED** (CI clears the floor +0.0219) |
| `cond.opp_class_auc.t4_10` | UNMATCHED (as traced) | +0.7599 | +0.7086 | **+0.0513** | [+0.0377, +0.0657] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0298 [+0.0176, +0.0185] | +0.0179 | **+0.0119** | [+0.0076, +0.0162] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0298 [+0.0176, +0.0185] | +0.0179 | **+0.0119** | [+0.0076, +0.0162] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | UNMATCHED (as traced) | +0.0298 | +0.0181 | **+0.0118** | [+0.0074, +0.0160] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.1480 [+0.1255, +0.1283] | +0.1266 | **+0.0214** | [+0.0039, +0.0394] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.1480 [+0.1255, +0.1283] | +0.1266 | **+0.0214** | [+0.0039, +0.0394] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | UNMATCHED (as traced) | +0.1480 | +0.1264 | **+0.0216** | [+0.0046, +0.0392] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.1854 [+0.1857, +0.1875] | +0.1867 | **-0.0013** | [-0.0254, +0.0217] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.1854 [+0.1857, +0.1875] | +0.1867 | **-0.0013** | [-0.0254, +0.0217] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | UNMATCHED (as traced) | +0.1854 | +0.1871 | **-0.0017** | [-0.0248, +0.0222] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0582 [+0.0535, +0.0542] | +0.0539 | **+0.0043** | [-0.0006, +0.0081] | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| `cond.within_team_resolution.all` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0582 [+0.0535, +0.0542] | +0.0539 | **+0.0043** | [-0.0006, +0.0081] | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0582 | +0.0531 | **+0.0051** | [+0.0001, +0.0087] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0336 [+0.0310, +0.0318] | +0.0313 | **+0.0022** | [-0.0015, +0.0063] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0336 [+0.0310, +0.0318] | +0.0313 | **+0.0022** | [-0.0015, +0.0063] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0336 | +0.0308 | **+0.0028** | [-0.0009, +0.0066] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.9394 [+1.2172, +1.2991] | +1.2573 | **-0.3179** | [-0.3572, -0.1674] | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.9394 [+1.2172, +1.2991] | +1.2573 | **-0.3179** | [-0.3572, -0.1674] | **WITHIN FLOOR** (|delta| <= floor 0.4324) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.9394 | +1.2523 | **-0.3129** | [-0.3523, -0.1709] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.4211 [+0.6729, +0.6825] | +0.6778 | **-0.2567** | [-0.2483, -0.1636] | **NOT DETECTED** (CI does not clear the floor 0.2507) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.4211 [+0.6729, +0.6825] | +0.6778 | **-0.2567** | [-0.2483, -0.1636] | **NOT DETECTED** (CI does not clear the floor 0.2507) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.4211 | +0.6791 | **-0.2580** | [-0.2497, -0.1633] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0020 [+0.0031, +0.0065] | +0.0047 | **-0.0027** | [-0.0083, +0.0026] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0020 [+0.0031, +0.0065] | +0.0047 | **-0.0027** | [-0.0083, +0.0026] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0020 | +0.0050 | **-0.0031** | [-0.0082, +0.0020] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0402 [+0.0401, +0.0445] | +0.0429 | **-0.0027** | [-0.0155, +0.0105] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +0.0402 [+0.0401, +0.0445] | +0.0429 | **-0.0027** | [-0.0155, +0.0105] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0402 | +0.0424 | **-0.0022** | [-0.0151, +0.0103] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +1.0104 [+0.9867, +1.0035] | +0.9941 | **+0.0163** | [-0.0539, +0.0856] | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 797/350/9 (9539 battles, 9533 decoder battles, 21 seeds) | +1.0104 [+0.9867, +1.0035] | +0.9941 | **+0.0163** | [-0.0539, +0.0856] | **WITHIN FLOOR** (|delta| <= floor 0.1363) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.0104 | +1.0200 | **-0.0096** | [-0.0810, +0.0634] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.1500 | [+0.1340, +0.1689] | +0.1454 | [+0.1265, +0.1696] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1506 | [+0.1348, +0.1692] | +0.1477 | [+0.1291, +0.1717] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0960 | [-0.1058, -0.0870] | -0.1018 | [-0.1114, -0.0924] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.6725 | [+0.6111, +0.7402] | +0.6237 | [+0.5653, +0.6832] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.6703 | [+0.6095, +0.7370] | +0.6221 | [+0.5643, +0.6808] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0370 | [-0.0470, -0.0272] | -0.0448 | [-0.0552, -0.0353] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 4-10` | +0.5305 | [+0.4832, +0.5839] | +0.5080 | [+0.4626, +0.5569] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 11-24` | +0.7546 | [+0.6856, +0.8311] | +0.7039 | [+0.6391, +0.7725] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0422 | [+0.0337, +0.0501] | +0.0475 | [+0.0380, +0.0566] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0075 | [+0.0038, +0.0107] | +0.0105 | [+0.0065, +0.0143] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.4838 | [+0.4708, +0.4974] | +0.5014 | [+0.4883, +0.5145] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1-3` | +0.6206 | [+0.6099, +0.6318] | +0.5796 | [+0.5675, +0.5917] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 4-10` | +0.7599 | [+0.7503, +0.7695] | +0.7086 | [+0.6987, +0.7188] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 1-3` | +0.0298 | [+0.0267, +0.0335] | +0.0181 | [+0.0159, +0.0207] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 4-10` | +0.1480 | [+0.1353, +0.1620] | +0.1264 | [+0.1157, +0.1380] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 11-24` | +0.1854 | [+0.1694, +0.2036] | +0.1871 | [+0.1713, +0.2033] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0582 | [+0.0582, +0.0646] | +0.0531 | [+0.0541, +0.0601] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0336 | [+0.0312, +0.0368] | +0.0308 | [+0.0285, +0.0338] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.9394 | [+0.3773, +0.4742] | +1.2523 | [+0.6046, +0.7674] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.4211 | [+0.3071, +0.3520] | +0.6791 | [+0.4983, +0.5740] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0020 | [-0.0014, +0.0047] | +0.0050 | [+0.0006, +0.0091] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0402 | [+0.0320, +0.0484] | +0.0424 | [+0.0327, +0.0523] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.0504 | [+1.0046, +1.1017] | +1.0695 | [+1.0223, +1.1229] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | +0.2140 | [+0.1283, +0.2987] | +0.3471 | [+0.2648, +0.4224] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.8244 | [+0.7427, +0.9103] | +0.7324 | [+0.6476, +0.8189] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.3598 | [+0.2014, +0.5095] | +0.9827 | [+0.8779, +1.0856] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.0104 | [+0.9642, +1.0631] | +1.0200 | [+0.9727, +1.0739] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.0586 | [+1.0052, +1.1159] | +1.0542 | [+0.9996, +1.1146] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | +0.1992 | [+0.1047, +0.2877] | +0.3683 | [+0.2755, +0.4524] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_17_ladder_strata` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0483 [+0.0348, +0.0631] | 0.0618 | -0.0135 | +0.3085 [+0.2331, +0.3633] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0239 [+0.0168, +0.0334] | 0.0337 | -0.0098 | +0.2260 [+0.1499, +0.2848] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0656 [+0.0431, +0.0923] | 0.0711 | -0.0055 | +0.2840 [+0.1694, +0.3722] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 735 battles / 6400 rollouts · Brier 0.0993 = REL 0.0019 − RES 0.0390 + UNC 0.1363 + WBV 0.0008 (resid -6.81e-04) · base rate 0.8371 · **resolution is 28.6% of the base-rate cap** · corr(turn,V) -0.1125 vs corr(turn,MC) -0.1319

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 748 battles / 6400 rollouts · Brier 0.0938 = REL 0.0030 − RES 0.0359 + UNC 0.1267 + WBV 0.0008 (resid -8.83e-04) · base rate 0.8511 · **resolution is 28.3% of the base-rate cap** · corr(turn,V) -0.0229 vs corr(turn,MC) -0.0615

## 5. THE LEDGER LINE

```
ai_v12_17_ladder_strata vs ai_v12_11_ladder_ctrl10M at 10M [OFFLINE-GENERATED: 800 games x 9+3 opponents, full capture]: G1 bot Δ +0.0055 [+0.0006, +0.0106] WITHIN FLOOR · identity bias late Δ +0.0248 [-0.0177, +0.0691] WITHIN FLOOR · turn-contrast Δ -0.0192 [-0.1391, +0.0979] WITHIN FLOOR · spread ratio t1-3 Δ +0.0046 [-0.0252, +0.0305] WITHIN FLOOR · own-team R2 t1 Δ -0.0051 [-0.0177, +0.0075] WITHIN FLOOR [QUOTA-MATCHED] · calib slope Δ -0.0191 [-0.0889, +0.0522] WITHIN FLOOR
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_17_ladder_strata --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_17_ladder_strata --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_11_ladder_ctrl10M --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/hp800_floor_v6.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_17_ladder_strata_vs_ctrl10M_v6 --nice 15 --ledger-line

# arm — ai_v12_17_ladder_strata  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_17_ladder_strata` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_17_ladder_strata/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_17_ladder_strata_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_17_ladder_strata_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_17_ladder_strata_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_11_ladder_ctrl10M__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_11_ladder_ctrl10M__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_11_ladder_ctrl10M__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_17_ladder_strata_vs_ctrl10M_v6/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_17_ladder_strata_vs_ctrl10M_v6/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
