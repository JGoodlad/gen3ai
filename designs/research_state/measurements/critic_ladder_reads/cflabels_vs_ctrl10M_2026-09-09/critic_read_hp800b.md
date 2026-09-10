# CRITIC READ — `ai_v12_12_ladder_cflabels` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v5 at 2026-09-10T16:43:21. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_12_ladder_cflabels` | `step_10000032` | 9600 | 0.7% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 9600 | 0.4% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 800 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260911, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_12_ladder_cflabels` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_12_ladder_cflabels` | **793/319/13** | 9600 | 8278 / 1257 / 65 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **797/366/10** | 9600 | 8280 / 1277 / 43 | 12 |

**Frames:** the realized per-opponent caps differ — arm 793/319/13, control 797/366/10 (traced W/L/D per opponent, counted on disk); control cut to 793/319/10

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **793/319/10** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/hp800b_floor.json` — 67 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0214 | +0.0188 | **+0.0026** | [-0.0026, +0.0072] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0001 | -0.0355 | **+0.0356** | [-0.0067, +0.0797] | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.0642 | +0.0584 | **+0.0058** | [-0.1234, +0.1378] | **WITHIN FLOOR** (|delta| <= floor 0.0143) |
| skill · `bot` | — | +0.1972 | +0.1472 | **+0.0500** | [-0.0054, +0.1115] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.1122 | +0.1535 | **-0.0413** | [-0.0721, -0.0114] | **WITHIN FLOOR** (|delta| <= floor 0.0671) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 793/319/10 | +0.0486 | +0.0602 | **-0.0116** | [-0.0254, +0.0029] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.0118 | +1.0668 | **-0.0549** | [-0.1240, +0.0160] | **WITHIN FLOOR** (|delta| <= floor 0.2001) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0368 | +0.0391 | **-0.0023** | [-0.0085, +0.0038] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0024) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0214 | +0.0188 | **+0.0026** | [-0.0026, +0.0072] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0495 | +0.0574 | **-0.0079** | [-0.0182, +0.0034] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0005 | +0.0013 | **-0.0008** | [-0.0016, +0.0000] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0026) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0011 | +0.0053 | **-0.0042** | [-0.0060, -0.0028] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0053) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0057 | +0.0016 | **+0.0041** | [+0.0010, +0.0074] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0232) |
| ece | `all` | `capture-rate (gauge)` | +0.0199 | +0.0243 | **-0.0044** | [-0.0145, +0.0071] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| ece | `bot` | `capture-rate (gauge)` | +0.0259 | +0.0493 | **-0.0235** | [-0.0338, -0.0139] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| ece | `pool` | `capture-rate (gauge)` | +0.0689 | +0.0342 | **+0.0347** | [+0.0101, +0.0547] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1097) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2477 | +0.2650 | **-0.0173** | [-0.0546, +0.0220] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0454) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.1972 | +0.1472 | **+0.0500** | [-0.0054, +0.1115] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2112 | +0.2596 | **-0.0484** | [-0.0991, +0.0074] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1381) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0408 | +0.0069 | **+0.0339** | [+0.0066, +0.0607] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0934) |
| bias V - p_hat | `ALL` | `pop` | -0.0094 | -0.0408 | **+0.0314** | [+0.0104, +0.0525] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0744) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0094 | -0.0408 | **+0.0314** | [+0.0104, +0.0525] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0744) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0065 | -0.0325 | **+0.0390** | [+0.0018, +0.0757] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1037) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0371 | -0.0674 | **+0.0303** | [-0.0017, +0.0617] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0793) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0371 | -0.0674 | **+0.0303** | [-0.0017, +0.0617] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0793) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0493 | +0.0294 | **+0.0199** | [-0.0211, +0.0606] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0642) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0077 | -0.0195 | **+0.0272** | [-0.0041, +0.0590] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0393) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0077 | -0.0195 | **+0.0272** | [-0.0041, +0.0590] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0393) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0714 | +0.0255 | **+0.0459** | [-0.0131, +0.1034] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1252) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0001 | -0.0355 | **+0.0356** | [-0.0067, +0.0797] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0001 | -0.0355 | **+0.0356** | [-0.0067, +0.0797] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| bias V - p_hat | `bot` | `raw` | +0.0328 | -0.0065 | **+0.0393** | [+0.0051, +0.0747] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0666) |
| bias V - p_hat | `bot` | `pop` | -0.0119 | -0.0497 | **+0.0378** | [+0.0145, +0.0621] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0567) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0119 | -0.0497 | **+0.0378** | [+0.0145, +0.0621] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0567) |
| bias V - p_hat | `pool` | `raw` | +0.0533 | +0.0310 | **+0.0223** | [-0.0200, +0.0642] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1390) |
| bias V - p_hat | `pool` | `pop` | -0.0022 | -0.0143 | **+0.0122** | [-0.0265, +0.0502] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1161) |
| bias V - p_hat ⭐ | `pool` | `ipw` | -0.0022 | -0.0143 | **+0.0122** | [-0.0265, +0.0502] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1161) |
| Murphy resolution | `ALL` | `raw` | +0.0234 | +0.0268 | **-0.0035** | [-0.0143, +0.0074] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0291 | +0.0323 | **-0.0032** | [-0.0175, +0.0112] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0291 | +0.0323 | **-0.0032** | [-0.0175, +0.0112] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1274 | +0.1550 | **-0.0275** | [-0.1024, +0.0469] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0538) |
| Murphy skill_score | `ALL` | `pop` | +0.2193 | +0.2297 | **-0.0104** | [-0.1221, +0.1009] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0162) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2193 | +0.2297 | **-0.0104** | [-0.1221, +0.1009] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0162) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1478 | +0.1727 | **-0.0249** | [-0.0873, +0.0377] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2281 | +0.2540 | **-0.0259** | [-0.1144, +0.0624] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2281 | +0.2540 | **-0.0259** | [-0.1144, +0.0624] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0043 | +0.0041 | **+0.0003** | [-0.0039, +0.0045] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0088) |
| Murphy reliability | `ALL` | `pop` | +0.0018 | +0.0039 | **-0.0021** | [-0.0066, +0.0025] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0018 | +0.0039 | **-0.0021** | [-0.0066, +0.0025] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.0642 | +0.0584 | **+0.0058** | [-0.1234, +0.1378] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0143) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | +0.0081 | +0.0135 | **-0.0054** | [-0.1299, +0.1237] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0327) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0081 | +0.0135 | **-0.0054** | [-0.1299, +0.1237] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0327) |

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

**arm — `ai_v12_12_ladder_cflabels` @ `step_10000032`**: 297389 states / 9535 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 65 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 293683 states / 9557 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 43 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.1122 | +0.1535 | **-0.0413** | [-0.0721, -0.0114] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0671) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 793/319/10 | +0.1152 | +0.1559 | **-0.0407** | [-0.0707, -0.0116] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0902 | -0.0973 | **+0.0072** | [-0.0062, +0.0205] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0147) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5836 | +0.6340 | **-0.0505** | [-0.1376, +0.0389] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2399) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 793/319/10 | +0.5815 | +0.6326 | **-0.0510** | [-0.1361, +0.0317] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0423 | -0.0421 | **-0.0002** | [-0.0139, +0.0138] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0322) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 793/319/10 | +0.0486 | +0.0602 | **-0.0116** | [-0.0254, +0.0029] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 793/319/10 | +0.0092 | +0.0107 | **-0.0015** | [-0.0068, +0.0037] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 793/319/10 | +0.4892 | +0.4942 | **-0.0050** | [-0.0236, +0.0150] | 2000 | **NOT DETECTED** (CI covers zero) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 793/319/10 | +0.0504 | +0.0530 | **-0.0026** | [-0.0070, +0.0015] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0034) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 793/319/10 | +0.0288 | +0.0299 | **-0.0011** | [-0.0049, +0.0025] | 2000 | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 793/319/10 | +0.9706 | +1.6838 | **-0.7131** | [-0.2666, -0.0487] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.7010) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 793/319/10 | +0.5522 | +0.7064 | **-0.1541** | [-0.1494, -0.0560] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 793/319/10 | +0.0063 | +0.0041 | **+0.0023** | [-0.0042, +0.0091] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 793/319/10 | +0.0423 | +0.0561 | **-0.0138** | [-0.0297, +0.0016] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.0118 | +1.0668 | **-0.0549** | [-0.1240, +0.0160] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2001) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | +0.2097 | +0.3808 | **-0.1711** | [-0.2812, -0.0594] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.0247) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.7885 | +0.7445 | **+0.0440** | [-0.0799, +0.1709] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0887) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.8590 | +0.9909 | **-0.1320** | [-0.2988, +0.0323] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.4031) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 793/319/10 | +0.9618 | +1.0091 | **-0.0474** | [-0.1145, +0.0219] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.0136 | +1.0385 | **-0.0250** | [-0.1025, +0.0548] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2481) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | +0.2047 | +0.4206 | **-0.2159** | [-0.3401, -0.0927] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.0981) |

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
| arm · team | 601 | 602 | 9532 | 19064 | 13.0 | 26.0 | 4–88 |
| arm · stratum | 5 | 602 | 9532 | 19064 | 1911.0 | 3822.0 | 1898–1912 |
| arm · between-team spread | 601 | 602 | 9532 | — | 13.0 | — | — |
| control · team | 601 | 602 | 9554 | 19108 | 13.0 | 26.0 | 4–88 |
| control · stratum | 5 | 602 | 9554 | 19108 | 1912.0 | 3824.0 | 1902–1917 |
| control · between-team spread | 601 | 602 | 9554 | — | 13.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 793/319/10 | +0.0504 | +0.0530 | **-0.0026** | [-0.0070, +0.0015] | **WITHIN FLOOR** (|delta| <= floor 0.0034) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 793/319/10 | +0.0288 | +0.0299 | **-0.0011** | [-0.0049, +0.0025] | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 793/319/10 | +0.9706 | +1.6838 | **-0.7131** | [-0.2666, -0.0487] | **NOT DETECTED** (CI does not clear the floor 0.7010) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 793/319/10 | +0.0486 | +0.0602 | **-0.0116** | [-0.0254, +0.0029] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 793/319/10 | +0.0063 | +0.0041 | **+0.0023** | [-0.0042, +0.0091] | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 793/319/10 | +0.0423 | +0.0561 | **-0.0138** | [-0.0297, +0.0016] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 19070 | 9535 | 0.8486 | **0.1896** | **1.8470** | [0.235, 0.998] | 0.0037 |
| arm · t1_3 | `t1_3` | 19070 | 9535 | 0.7868 | **0.0976** | **0.6261** | [0.551, 0.944] | 0.0000 |
| arm · common support | `common support` | 17546 | 9346 | 0.8595 | **0.1516** | **1.5050** | [0.412, 0.995] | 0.0000 |
| control · all | `all` | 19114 | 9557 | 0.8240 | **0.1990** | **1.7526** | [0.208, 0.997] | 0.0000 |
| control · t1_3 | `t1_3` | 19114 | 9557 | 0.7666 | **0.1129** | **0.6593** | [0.489, 0.932] | 0.0000 |
| control · common support | `common support` | 18094 | 9435 | 0.8406 | **0.1599** | **1.4896** | [0.393, 0.995] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.2349, 0.9965]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.0118 | +1.0668 | **-0.0549** | [-0.1240, +0.0160] | 0.2001 | **WITHIN FLOOR** (|delta| <= floor 0.2001) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | +0.2097 | +0.3808 | **-0.1711** | [-0.2812, -0.0594] | 1.0247 | **WITHIN FLOOR** (|delta| <= floor 1.0247) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.7885 | +0.7445 | **+0.0440** | [-0.0799, +0.1709] | 0.0887 | **WITHIN FLOOR** (|delta| <= floor 0.0887) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.8590 | +0.9909 | **-0.1320** | [-0.2988, +0.0323] | 0.4031 | **WITHIN FLOOR** (|delta| <= floor 0.4031) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 793/319/10 | +0.9618 | +1.0091 | **-0.0474** | [-0.1145, +0.0219] | 0.1729 | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.0136 | +1.0385 | **-0.0250** | [-0.1025, +0.0548] | 0.2481 | **WITHIN FLOOR** (|delta| <= floor 0.2481) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +0.2047 | +0.4206 | **-0.2159** | [-0.3401, -0.0927] | 1.0981 | **WITHIN FLOOR** (|delta| <= floor 1.0981) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.1152 [+0.1532, +0.1579] | +0.1559 | **-0.0407** | [-0.0707, -0.0116] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.1152 [+0.1545, +0.1566] | +0.1557 | **-0.0405** | [-0.0687, -0.0122] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1152 | +0.1557 | **-0.0406** | [-0.0703, -0.0112] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.5815 [+0.6279, +0.6405] | +0.6326 | **-0.0510** | [-0.1361, +0.0317] | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.5815 [+0.6297, +0.6339] | +0.6319 | **-0.0504** | [-0.1399, +0.0384] | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.5815 | +0.6323 | **-0.0508** | [-0.1368, +0.0371] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.0486 [+0.0563, +0.0643] | +0.0602 | **-0.0116** | [-0.0254, +0.0029] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| `cond.own_team_r2.t1` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.0486 [+0.0591, +0.0629] | +0.0608 | **-0.0122** | [-0.0263, +0.0026] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0486 | +0.0621 | **-0.0135** | [-0.0285, +0.0005] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.0092 [+0.0098, +0.0121] | +0.0107 | **-0.0015** | [-0.0068, +0.0037] | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| `cond.own_team_r2.all` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.0092 [+0.0104, +0.0119] | +0.0114 | **-0.0022** | [-0.0078, +0.0032] | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0092 | +0.0118 | **-0.0026** | [-0.0083, +0.0028] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.4892 [+0.4907, +0.4967] | +0.4942 | **-0.0050** | [-0.0236, +0.0150] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.4892 [+0.4926, +0.4988] | +0.4951 | **-0.0058** | [-0.0250, +0.0138] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.4892 | +0.4887 | **+0.0005** | [-0.0182, +0.0196] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.0504 [+0.0524, +0.0536] | +0.0530 | **-0.0026** | [-0.0070, +0.0015] | **WITHIN FLOOR** (|delta| <= floor 0.0034) |
| `cond.within_team_resolution.all` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.0504 [+0.0527, +0.0534] | +0.0530 | **-0.0026** | [-0.0070, +0.0013] | **WITHIN FLOOR** (|delta| <= floor 0.0034) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0504 | +0.0530 | **-0.0026** | [-0.0069, +0.0012] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.0288 [+0.0296, +0.0305] | +0.0299 | **-0.0011** | [-0.0049, +0.0025] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.0288 [+0.0289, +0.0296] | +0.0293 | **-0.0004** | [-0.0039, +0.0031] | **WITHIN FLOOR** (|delta| <= floor 0.0010) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0288 | +0.0292 | **-0.0003** | [-0.0038, +0.0032] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.9706 [+1.5323, +1.8104] | +1.6838 | **-0.7131** | [-0.2666, -0.0487] | **NOT DETECTED** (CI does not clear the floor 0.7010) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.9706 [+1.6470, +1.7448] | +1.6968 | **-0.7262** | [-0.2707, -0.0464] | **NOT DETECTED** (CI does not clear the floor 0.7010) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.9706 | +1.7001 | **-0.7294** | [-0.2773, -0.0501] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.5522 [+0.6966, +0.7122] | +0.7064 | **-0.1541** | [-0.1494, -0.0560] | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.5522 [+0.7115, +0.7163] | +0.7137 | **-0.1615** | [-0.1534, -0.0547] | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.5522 | +0.7151 | **-0.1629** | [-0.1557, -0.0575] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.0063 [+0.0017, +0.0067] | +0.0041 | **+0.0023** | [-0.0042, +0.0091] | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| `cond.own_team_r2.late` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.0063 [+0.0027, +0.0075] | +0.0052 | **+0.0011** | [-0.0061, +0.0085] | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0063 | +0.0060 | **+0.0003** | [-0.0074, +0.0076] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.0423 [+0.0505, +0.0599] | +0.0561 | **-0.0138** | [-0.0297, +0.0016] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.0423 [+0.0529, +0.0591] | +0.0560 | **-0.0138** | [-0.0294, +0.0025] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0423 | +0.0561 | **-0.0138** | [-0.0300, +0.0018] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 793/319/10 (9506 battles, 9503 decoder battles, 21 seeds) | +0.9618 [+0.9977, +1.0195] | +1.0091 | **-0.0474** | [-0.1145, +0.0219] | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 892/359/10 (9550 battles, 9547 decoder battles, 21 seeds) | +0.9618 [+0.9962, +1.0060] | +1.0019 | **-0.0402** | [-0.1090, +0.0317] | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +0.9618 | +1.0078 | **-0.0460** | [-0.1147, +0.0251] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.1122 | [+0.0949, +0.1353] | +0.1535 | [+0.1343, +0.1792] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1152 | [+0.0984, +0.1379] | +0.1557 | [+0.1370, +0.1811] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0902 | [-0.0999, -0.0810] | -0.0973 | [-0.1076, -0.0881] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5836 | [+0.5198, +0.6535] | +0.6340 | [+0.5749, +0.6984] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.5815 | [+0.5187, +0.6502] | +0.6323 | [+0.5737, +0.6958] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0423 | [-0.0531, -0.0325] | -0.0421 | [-0.0529, -0.0324] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0486 | [+0.0392, +0.0578] | +0.0621 | [+0.0517, +0.0730] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0092 | [+0.0053, +0.0128] | +0.0118 | [+0.0074, +0.0159] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.4892 | [+0.4765, +0.5037] | +0.4887 | [+0.4750, +0.5022] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0504 | [+0.0507, +0.0565] | +0.0530 | [+0.0535, +0.0594] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0288 | [+0.0266, +0.0317] | +0.0292 | [+0.0269, +0.0320] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.9706 | [+0.4954, +0.6308] | +1.7001 | [+0.6368, +0.8152] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.5522 | [+0.4102, +0.4721] | +0.7151 | [+0.5106, +0.5848] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0063 | [+0.0007, +0.0113] | +0.0060 | [+0.0006, +0.0109] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0423 | [+0.0322, +0.0531] | +0.0561 | [+0.0456, +0.0676] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.0118 | [+0.9643, +1.0618] | +1.0668 | [+1.0179, +1.1195] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | +0.2097 | [+0.1263, +0.2942] | +0.3808 | [+0.3027, +0.4604] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.7885 | [+0.6957, +0.8825] | +0.7445 | [+0.6597, +0.8303] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.8590 | [+0.7369, +0.9857] | +0.9909 | [+0.8862, +1.0995] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +0.9618 | [+0.9142, +1.0143] | +1.0078 | [+0.9620, +1.0592] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.0136 | [+0.9610, +1.0704] | +1.0385 | [+0.9860, +1.0974] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | +0.2047 | [+0.1131, +0.2965] | +0.4206 | [+0.3341, +0.5073] |

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

Identity, capture-rate reweighted: 800 labels / 738 battles / 6400 rollouts · Brier 0.0997 = REL 0.0018 − RES 0.0291 + UNC 0.1277 + WBV 0.0008 (resid -1.44e-03) · base rate 0.8497 · **resolution is 22.8% of the base-rate cap** · corr(turn,V) -0.0927 vs corr(turn,MC) -0.1568

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 741 battles / 6400 rollouts · Brier 0.0980 = REL 0.0039 − RES 0.0323 + UNC 0.1272 + WBV 0.0008 (resid -1.61e-03) · base rate 0.8504 · **resolution is 25.4% of the base-rate cap** · corr(turn,V) -0.0506 vs corr(turn,MC) -0.1089

## 5. THE LEDGER LINE

```
ai_v12_12_ladder_cflabels vs ai_v12_11_ladder_ctrl10M at 10M [OFFLINE-GENERATED: 800 games x 9+3 opponents, full capture]: G1 bot Δ +0.0026 [-0.0026, +0.0072] NOT DETECTED · identity bias late Δ +0.0356 [-0.0067, +0.0797] WITHIN FLOOR · turn-contrast Δ +0.0058 [-0.1234, +0.1378] WITHIN FLOOR · spread ratio t1-3 Δ -0.0413 [-0.0721, -0.0114] WITHIN FLOOR · own-team R2 t1 Δ -0.0116 [-0.0254, +0.0029] WITHIN FLOOR [QUOTA-MATCHED] · calib slope Δ -0.0549 [-0.1240, +0.0160] WITHIN FLOOR
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_12_ladder_cflabels --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_12_ladder_cflabels --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_11_ladder_ctrl10M --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/hp800b_floor.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M --nice 15 --ledger-line

# arm — ai_v12_12_ladder_cflabels
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_12_ladder_cflabels --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_12_ladder_cflabels --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_12_ladder_cflabels` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_12_ladder_cflabels/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_11_ladder_ctrl10M__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_11_ladder_ctrl10M__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_11_ladder_ctrl10M__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_12_ladder_cflabels_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
