# CRITIC READ — `ai_v12_28_ladder_ent05` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v6 at 2026-09-12T20:56:15. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

**SELF-PLAY CROSSING** — arm 4,128,768 · control 4,128,768 (first step carrying any `*_pool` scalar; the PROMOTION scalar itself lands one eval→rollout lag earlier, at 4,000,032 / 4,000,032). Descriptive — no floor, no verdict.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_28_ladder_ent05` | `step_10000032` | 9600 | 0.3% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 9600 | 0.4% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 800 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260910, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_28_ladder_ent05` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_28_ladder_ent05` | **799/284/6** | 9600 | 8310 / 1261 / 29 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **797/368/9** | 9600 | 8252 / 1305 / 43 | 12 |

**Frames:** the realized per-opponent caps differ — arm 799/284/6, control 797/368/9 (traced W/L/D per opponent, counted on disk); arm and control cut to 797/284/6

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm and control side is subsampled to caps **797/284/6** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/hp800_floor_v6.json` — 74 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0209 | +0.0170 | **+0.0039** | [-0.0012, +0.0092] | **WITHIN FLOOR** (|delta| <= floor 0.0074) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0785 | -0.0189 | **-0.0596** | [-0.1088, -0.0100] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.0867 | +0.0386 | **-0.1253** | [-0.2860, +0.0360] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.1612 | +0.1237 | **+0.0375** | [-0.0203, +0.1027] | **WITHIN FLOOR** (|delta| <= floor 0.1004) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.1675 | +0.1454 | **+0.0221** | [-0.0108, +0.0522] | **WITHIN FLOOR** (|delta| <= floor 0.0533) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 797/284/6 | +0.0594 | +0.0445 | **+0.0149** | [+0.0006, +0.0290] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.2863 | +1.0695 | **+0.2168** | [+0.1405, +0.2958] | **DETECTED** (CI clears the floor +0.1085) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0385 | +0.0386 | **-0.0001** | [-0.0054, +0.0062] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0003) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0209 | +0.0170 | **+0.0039** | [-0.0012, +0.0092] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0074) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0543 | +0.0614 | **-0.0070** | [-0.0160, +0.0032] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0141) |
| reliability | `all` | `capture-rate (gauge)` | +0.0018 | +0.0010 | **+0.0008** | [-0.0002, +0.0019] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0054 | +0.0053 | **+0.0001** | [-0.0023, +0.0022] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0005 | +0.0019 | **-0.0014** | [-0.0033, +0.0003] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0167) |
| ece | `all` | `capture-rate (gauge)` | +0.0413 | +0.0237 | **+0.0176** | [+0.0066, +0.0278] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0144) |
| ece | `bot` | `capture-rate (gauge)` | +0.0646 | +0.0445 | **+0.0201** | [+0.0071, +0.0298] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0297) |
| ece | `pool` | `capture-rate (gauge)` | +0.0208 | +0.0403 | **-0.0195** | [-0.0379, +0.0019] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0822) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2580 | +0.2598 | **-0.0018** | [-0.0348, +0.0364] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0071) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.1612 | +0.1237 | **+0.0375** | [-0.0203, +0.1027] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1004) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2557 | +0.2742 | **-0.0185** | [-0.0581, +0.0252] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1108) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | -0.0288 | +0.0164 | **-0.0452** | [-0.0726, -0.0167] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0829) |
| bias V - p_hat | `ALL` | `pop` | -0.0708 | -0.0363 | **-0.0345** | [-0.0550, -0.0128] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0708 | -0.0363 | **-0.0345** | [-0.0550, -0.0128] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat | `early (turn<=10)` | `raw` | -0.0277 | -0.0107 | **-0.0170** | [-0.0547, +0.0229] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0948) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0757 | -0.0648 | **-0.0109** | [-0.0417, +0.0200] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0757 | -0.0648 | **-0.0109** | [-0.0417, +0.0200] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0765) |
| bias V - p_hat | `mid (11-24)` | `raw` | -0.0250 | +0.0229 | **-0.0479** | [-0.0855, -0.0096] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0748) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0606 | -0.0195 | **-0.0411** | [-0.0689, -0.0131] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0556) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0606 | -0.0195 | **-0.0411** | [-0.0689, -0.0131] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0556) |
| bias V - p_hat | `late (turn>=25)` | `raw` | -0.0363 | +0.0411 | **-0.0773** | [-0.1347, -0.0187] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0821) |
| bias V - p_hat | `late (turn>=25)` | `pop` | -0.0785 | -0.0189 | **-0.0596** | [-0.1088, -0.0100] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0785 | -0.0189 | **-0.0596** | [-0.1088, -0.0100] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| bias V - p_hat | `bot` | `raw` | -0.0088 | +0.0029 | **-0.0117** | [-0.0466, +0.0228] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0575) |
| bias V - p_hat | `bot` | `pop` | -0.0665 | -0.0441 | **-0.0223** | [-0.0455, +0.0016] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0417) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0665 | -0.0441 | **-0.0223** | [-0.0455, +0.0016] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0417) |
| bias V - p_hat | `pool` | `raw` | -0.0619 | +0.0436 | **-0.1054** | [-0.1503, -0.0614] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1164) |
| bias V - p_hat | `pool` | `pop` | -0.0824 | -0.0190 | **-0.0634** | [-0.1025, -0.0231] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| bias V - p_hat ⭐ | `pool` | `ipw` | -0.0824 | -0.0190 | **-0.0634** | [-0.1025, -0.0231] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| Murphy resolution | `ALL` | `raw` | +0.0225 | +0.0274 | **-0.0050** | [-0.0163, +0.0062] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0070) |
| Murphy resolution | `ALL` | `pop` | +0.0292 | +0.0359 | **-0.0067** | [-0.0214, +0.0080] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0292 | +0.0359 | **-0.0067** | [-0.0214, +0.0080] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1314 | +0.1626 | **-0.0312** | [-0.1135, +0.0495] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0370) |
| Murphy skill_score | `ALL` | `pop` | +0.1852 | +0.2601 | **-0.0749** | [-0.1939, +0.0385] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.1852 | +0.2601 | **-0.0749** | [-0.1939, +0.0385] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1560 | +0.1739 | **-0.0178** | [-0.0841, +0.0509] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0188) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2473 | +0.2834 | **-0.0361** | [-0.1274, +0.0564] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2473 | +0.2834 | **-0.0361** | [-0.1274, +0.0564] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0032 | +0.0024 | **+0.0008** | [-0.0027, +0.0046] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0119) |
| Murphy reliability | `ALL` | `pop` | +0.0067 | +0.0030 | **+0.0037** | [-0.0007, +0.0083] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0067 | +0.0030 | **+0.0037** | [-0.0007, +0.0083] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.0867 | +0.0386 | **-0.1253** | [-0.2860, +0.0360] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0908 | +0.0222 | **-0.1129** | [-0.2426, +0.0166] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0908 | +0.0222 | **-0.1129** | [-0.2426, +0.0166] | 4000 | **NOT DETECTED** (CI covers zero) |

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

**arm — `ai_v12_28_ladder_ent05` @ `step_10000032`**: 287195 states / 9571 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 29 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 291604 states / 9557 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 43 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.1675 | +0.1454 | **+0.0221** | [-0.0108, +0.0522] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0533) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 797/284/6 | +0.1679 | +0.1475 | **+0.0204** | [-0.0107, +0.0472] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0827 | -0.1018 | **+0.0191** | [+0.0056, +0.0320] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0071) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.6305 | +0.6237 | **+0.0068** | [-0.0833, +0.0954] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1973) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 797/284/6 | +0.6278 | +0.6223 | **+0.0056** | [-0.0805, +0.0903] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0367 | -0.0448 | **+0.0081** | [-0.0060, +0.0218] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0218) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 4-10` | as traced | +0.4540 | +0.5080 | **-0.0540** | [-0.1195, +0.0116] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2179) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 11-24` | as traced | +0.7094 | +0.7039 | **+0.0055** | [-0.0959, +0.1072] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1920) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 797/284/6 | +0.0594 | +0.0445 | **+0.0149** | [+0.0006, +0.0290] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 797/284/6 | +0.0107 | +0.0096 | **+0.0011** | [-0.0049, +0.0068] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0025) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 797/284/6 | +0.5016 | +0.5026 | **-0.0011** | [-0.0205, +0.0186] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0040) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1-3` | MATCHED 797/284/6 | +0.6141 | +0.5807 | **+0.0334** | [+0.0168, +0.0486] | 2000 | **DETECTED** (CI clears the floor +0.0153) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 4-10` | MATCHED 797/284/6 | +0.7348 | +0.7109 | **+0.0239** | [+0.0094, +0.0379] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0219) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 1-3` | MATCHED 797/284/6 | +0.0287 | +0.0180 | **+0.0107** | [+0.0065, +0.0145] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 4-10` | MATCHED 797/284/6 | +0.1342 | +0.1263 | **+0.0079** | [-0.0103, +0.0247] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 11-24` | MATCHED 797/284/6 | +0.1674 | +0.1868 | **-0.0194** | [-0.0429, +0.0032] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 797/284/6 | +0.0522 | +0.0533 | **-0.0011** | [-0.0056, +0.0027] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 797/284/6 | +0.0308 | +0.0306 | **+0.0002** | [-0.0034, +0.0039] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 797/284/6 | +0.8097 | +1.2709 | **-0.4612** | [-0.3204, -0.1349] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.4324) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 797/284/6 | +0.4484 | +0.6710 | **-0.2226** | [-0.2165, -0.1293] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2507) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 797/284/6 | +0.0089 | +0.0034 | **+0.0054** | [-0.0026, +0.0135] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 797/284/6 | +0.0505 | +0.0420 | **+0.0086** | [-0.0071, +0.0241] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.2863 | +1.0695 | **+0.2168** | [+0.1405, +0.2958] | 2000 | **DETECTED** (CI clears the floor +0.1085) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | +0.2458 | +0.3471 | **-0.1013** | [-0.2235, +0.0241] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2642) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.9995 | +0.7324 | **+0.2670** | [+0.1303, +0.4144] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.1802) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.6639 | +0.9827 | **-0.3187** | [-0.4998, -0.1517] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.3514) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 797/284/6 | +1.2318 | +0.9991 | **+0.2327** | [+0.1544, +0.3174] | 2000 | **DETECTED** (CI clears the floor +0.1363) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.3242 | +1.0555 | **+0.2688** | [+0.1800, +0.3548] | 2000 | **DETECTED** (CI clears the floor +0.1195) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | +0.1961 | +0.3672 | **-0.1710** | [-0.3008, -0.0352] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2778) |

### HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY — `V` against its opponent-decodable part

| side | window | ratio of `V` | opponent-decodable part | `V` / decodable |
|---|---|---|---|---|
| arm `ai_v12_28_ladder_ent05` | `t1_3` | +0.1675 | +0.0285 | 5.873 |
| arm `ai_v12_28_ladder_ent05` | `t4_10` | +0.4540 | +0.1339 | 3.390 |
| arm `ai_v12_28_ladder_ent05` | `t11_24` | +0.7094 | +0.1676 | 4.232 |
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
| arm · team | 600 | 602 | 9565 | 19130 | 13.0 | 26.0 | 4–94 |
| arm · stratum | 5 | 602 | 9565 | 19130 | 1915.0 | 3830.0 | 1906–1918 |
| arm · between-team spread | 600 | 602 | 9565 | — | 13.0 | — | — |
| control · team | 600 | 602 | 9551 | 19102 | 13.0 | 26.0 | 4–93 |
| control · stratum | 5 | 602 | 9551 | 19102 | 1910.0 | 3820.0 | 1898–1919 |
| control · between-team spread | 600 | 602 | 9551 | — | 13.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 797/284/6 | +0.0522 | +0.0533 | **-0.0011** | [-0.0056, +0.0027] | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 797/284/6 | +0.0308 | +0.0306 | **+0.0002** | [-0.0034, +0.0039] | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 797/284/6 | +0.8097 | +1.2709 | **-0.4612** | [-0.3204, -0.1349] | **NOT DETECTED** (CI does not clear the floor 0.4324) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 797/284/6 | +0.0594 | +0.0445 | **+0.0149** | [+0.0006, +0.0290] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 797/284/6 | +0.0089 | +0.0034 | **+0.0054** | [-0.0026, +0.0135] | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 797/284/6 | +0.0505 | +0.0420 | **+0.0086** | [-0.0071, +0.0241] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 19142 | 9571 | 0.8123 | **0.1833** | **1.3734** | [0.215, 0.989] | 0.0000 |
| arm · t1_3 | `t1_3` | 19142 | 9571 | 0.7760 | **0.0799** | **0.4750** | [0.598, 0.907] | 0.0000 |
| arm · common support | `common support` | 18184 | 9492 | 0.8263 | **0.1442** | **1.0997** | [0.409, 0.984] | 0.0000 |
| control · all | `all` | 19114 | 9557 | 0.8236 | **0.1990** | **1.7456** | [0.214, 0.996] | 0.0001 |
| control · t1_3 | `t1_3` | 19114 | 9557 | 0.7656 | **0.1128** | **0.6542** | [0.480, 0.931] | 0.0000 |
| control · common support | `common support` | 16400 | 9128 | 0.8213 | **0.1644** | **1.2514** | [0.353, 0.986] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.2150, 0.9886]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.2863 | +1.0695 | **+0.2168** | [+0.1405, +0.2958] | 0.1085 | **DETECTED** (CI clears the floor +0.1085) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | +0.2458 | +0.3471 | **-0.1013** | [-0.2235, +0.0241] | 0.2642 | **WITHIN FLOOR** (|delta| <= floor 0.2642) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.9995 | +0.7324 | **+0.2670** | [+0.1303, +0.4144] | 0.1802 | **NOT DETECTED** (CI does not clear the floor 0.1802) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.6639 | +0.9827 | **-0.3187** | [-0.4998, -0.1517] | 0.3514 | **WITHIN FLOOR** (|delta| <= floor 0.3514) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 797/284/6 | +1.2318 | +0.9991 | **+0.2327** | [+0.1544, +0.3174] | 0.1363 | **DETECTED** (CI clears the floor +0.1363) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.3242 | +1.0555 | **+0.2688** | [+0.1800, +0.3548] | 0.1195 | **DETECTED** (CI clears the floor +0.1195) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +0.1961 | +0.3672 | **-0.1710** | [-0.3008, -0.0352] | 0.2778 | **WITHIN FLOOR** (|delta| <= floor 0.2778) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.1679 [+0.1676, +0.1681] | +0.1475 | **+0.0204** | [-0.0107, +0.0472] | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.1679 [+0.1676, +0.1681] | +0.1477 | **+0.0202** | [-0.0121, +0.0468] | **WITHIN FLOOR** (|delta| <= floor 0.0535) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1679 | +0.1477 | **+0.0202** | [-0.0120, +0.0499] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.6278 [+0.6278, +0.6280] | +0.6223 | **+0.0056** | [-0.0805, +0.0903] | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.6278 [+0.6278, +0.6280] | +0.6221 | **+0.0057** | [-0.0831, +0.0856] | **WITHIN FLOOR** (|delta| <= floor 0.1969) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.6278 | +0.6221 | **+0.0057** | [-0.0833, +0.0936] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0594 [+0.0592, +0.0596] | +0.0445 | **+0.0149** | [+0.0006, +0.0290] | **NOT DETECTED** (CI does not clear the floor 0.0109) |
| `cond.own_team_r2.t1` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0594 [+0.0592, +0.0596] | +0.0475 | **+0.0120** | [-0.0023, +0.0261] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0597 | +0.0475 | **+0.0122** | [-0.0016, +0.0263] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0107 [+0.0104, +0.0109] | +0.0096 | **+0.0011** | [-0.0049, +0.0068] | **WITHIN FLOOR** (|delta| <= floor 0.0025) |
| `cond.own_team_r2.all` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0107 [+0.0104, +0.0109] | +0.0105 | **+0.0002** | [-0.0052, +0.0056] | **WITHIN FLOOR** (|delta| <= floor 0.0025) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0129 | +0.0105 | **+0.0024** | [-0.0036, +0.0081] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.5016 [+0.5011, +0.5024] | +0.5026 | **-0.0011** | [-0.0205, +0.0186] | **WITHIN FLOOR** (|delta| <= floor 0.0040) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.5016 [+0.5011, +0.5024] | +0.5014 | **+0.0002** | [-0.0185, +0.0196] | **WITHIN FLOOR** (|delta| <= floor 0.0040) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5010 | +0.5014 | **-0.0004** | [-0.0182, +0.0175] | *no label — not a reading* |
| `cond.opp_class_auc.t1_3` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.6141 [+0.6139, +0.6153] | +0.5807 | **+0.0334** | [+0.0168, +0.0486] | **DETECTED** (CI clears the floor +0.0153) |
| `cond.opp_class_auc.t1_3` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.6141 [+0.6139, +0.6153] | +0.5796 | **+0.0346** | [+0.0181, +0.0504] | **DETECTED** (CI clears the floor +0.0153) |
| `cond.opp_class_auc.t1_3` | UNMATCHED (as traced) | +0.6090 | +0.5796 | **+0.0294** | [+0.0124, +0.0472] | *no label — not a reading* |
| `cond.opp_class_auc.t4_10` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.7348 [+0.7342, +0.7352] | +0.7109 | **+0.0239** | [+0.0094, +0.0379] | **NOT DETECTED** (CI does not clear the floor 0.0219) |
| `cond.opp_class_auc.t4_10` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.7348 [+0.7342, +0.7352] | +0.7086 | **+0.0262** | [+0.0122, +0.0403] | **NOT DETECTED** (CI does not clear the floor 0.0219) |
| `cond.opp_class_auc.t4_10` | UNMATCHED (as traced) | +0.7331 | +0.7086 | **+0.0245** | [+0.0103, +0.0389] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0287 [+0.0286, +0.0287] | +0.0180 | **+0.0107** | [+0.0065, +0.0145] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0287 [+0.0286, +0.0287] | +0.0181 | **+0.0106** | [+0.0064, +0.0144] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | UNMATCHED (as traced) | +0.0285 | +0.0181 | **+0.0104** | [+0.0063, +0.0147] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.1342 [+0.1341, +0.1343] | +0.1263 | **+0.0079** | [-0.0103, +0.0247] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.1342 [+0.1341, +0.1343] | +0.1264 | **+0.0078** | [-0.0105, +0.0236] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | UNMATCHED (as traced) | +0.1339 | +0.1264 | **+0.0075** | [-0.0095, +0.0246] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.1674 [+0.1673, +0.1676] | +0.1868 | **-0.0194** | [-0.0429, +0.0032] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.1674 [+0.1673, +0.1676] | +0.1871 | **-0.0197** | [-0.0432, +0.0013] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | UNMATCHED (as traced) | +0.1676 | +0.1871 | **-0.0194** | [-0.0422, +0.0026] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0522 [+0.0521, +0.0523] | +0.0533 | **-0.0011** | [-0.0056, +0.0027] | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| `cond.within_team_resolution.all` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0522 [+0.0521, +0.0523] | +0.0531 | **-0.0009** | [-0.0063, +0.0023] | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0513 | +0.0531 | **-0.0018** | [-0.0069, +0.0015] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0308 [+0.0308, +0.0309] | +0.0306 | **+0.0002** | [-0.0034, +0.0039] | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0308 [+0.0308, +0.0309] | +0.0308 | **+0.0001** | [-0.0037, +0.0038] | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0305 | +0.0308 | **-0.0002** | [-0.0040, +0.0032] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.8097 [+0.8074, +0.8106] | +1.2709 | **-0.4612** | [-0.3204, -0.1349] | **NOT DETECTED** (CI does not clear the floor 0.4324) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.8097 [+0.8074, +0.8106] | +1.2523 | **-0.4426** | [-0.3309, -0.1397] | **NOT DETECTED** (CI does not clear the floor 0.4324) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.8094 | +1.2523 | **-0.4429** | [-0.3290, -0.1401] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.4484 [+0.4479, +0.4485] | +0.6710 | **-0.2226** | [-0.2165, -0.1293] | **WITHIN FLOOR** (|delta| <= floor 0.2507) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.4484 [+0.4479, +0.4485] | +0.6791 | **-0.2307** | [-0.2273, -0.1366] | **WITHIN FLOOR** (|delta| <= floor 0.2507) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.4483 | +0.6791 | **-0.2307** | [-0.2244, -0.1366] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0089 [+0.0078, +0.0097] | +0.0034 | **+0.0054** | [-0.0026, +0.0135] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0089 [+0.0078, +0.0097] | +0.0050 | **+0.0038** | [-0.0047, +0.0120] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0097 | +0.0050 | **+0.0046** | [-0.0038, +0.0129] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0505 [+0.0496, +0.0518] | +0.0420 | **+0.0086** | [-0.0071, +0.0241] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +0.0505 [+0.0496, +0.0518] | +0.0424 | **+0.0081** | [-0.0082, +0.0238] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0500 | +0.0424 | **+0.0076** | [-0.0076, +0.0235] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +1.2318 [+1.2278, +1.2341] | +0.9991 | **+0.2327** | [+0.1544, +0.3174] | **DETECTED** (CI clears the floor +0.1363) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 797/284/6 (9569 battles, 9563 decoder battles, 21 seeds) | +1.2318 [+1.2278, +1.2341] | +1.0200 | **+0.2118** | [+0.1365, +0.2958] | **DETECTED** (CI clears the floor +0.1363) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.2448 | +1.0200 | **+0.2249** | [+0.1483, +0.3040] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.1675 | [+0.1498, +0.1894] | +0.1454 | [+0.1265, +0.1696] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1679 | [+0.1505, +0.1896] | +0.1477 | [+0.1291, +0.1717] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0827 | [-0.0917, -0.0744] | -0.1018 | [-0.1114, -0.0924] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.6305 | [+0.5689, +0.6945] | +0.6237 | [+0.5653, +0.6832] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.6278 | [+0.5671, +0.6908] | +0.6221 | [+0.5643, +0.6808] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0367 | [-0.0464, -0.0279] | -0.0448 | [-0.0552, -0.0353] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 4-10` | +0.4540 | [+0.4125, +0.5010] | +0.5080 | [+0.4626, +0.5569] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 11-24` | +0.7094 | [+0.6388, +0.7840] | +0.7039 | [+0.6391, +0.7725] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0597 | [+0.0489, +0.0699] | +0.0475 | [+0.0380, +0.0566] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0129 | [+0.0082, +0.0172] | +0.0105 | [+0.0065, +0.0143] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5010 | [+0.4874, +0.5146] | +0.5014 | [+0.4883, +0.5145] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1-3` | +0.6090 | [+0.5981, +0.6207] | +0.5796 | [+0.5675, +0.5917] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 4-10` | +0.7331 | [+0.7234, +0.7429] | +0.7086 | [+0.6987, +0.7188] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 1-3` | +0.0285 | [+0.0257, +0.0321] | +0.0181 | [+0.0159, +0.0207] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 4-10` | +0.1339 | [+0.1221, +0.1472] | +0.1264 | [+0.1157, +0.1380] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 11-24` | +0.1676 | [+0.1521, +0.1842] | +0.1871 | [+0.1713, +0.2033] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0513 | [+0.0516, +0.0574] | +0.0531 | [+0.0541, +0.0601] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0305 | [+0.0284, +0.0333] | +0.0308 | [+0.0285, +0.0338] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.8094 | [+0.4034, +0.5042] | +1.2523 | [+0.6046, +0.7674] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.4483 | [+0.3317, +0.3791] | +0.6791 | [+0.4983, +0.5740] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0097 | [+0.0023, +0.0164] | +0.0050 | [+0.0006, +0.0091] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0500 | [+0.0385, +0.0621] | +0.0424 | [+0.0327, +0.0523] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.2863 | [+1.2303, +1.3462] | +1.0695 | [+1.0223, +1.1229] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | +0.2458 | [+0.1545, +0.3381] | +0.3471 | [+0.2648, +0.4224] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.9995 | [+0.8958, +1.1157] | +0.7324 | [+0.6476, +0.8189] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.6639 | [+0.5188, +0.7950] | +0.9827 | [+0.8779, +1.0856] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.2448 | [+1.1886, +1.3101] | +1.0200 | [+0.9727, +1.0739] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.3242 | [+1.2625, +1.3907] | +1.0555 | [+0.9990, +1.1167] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | +0.1961 | [+0.1016, +0.2955] | +0.3672 | [+0.2754, +0.4525] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_28_ladder_ent05` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0716 [+0.0411, +0.1034] | 0.0618 | +0.0098 | +0.3872 [+0.2713, +0.4885] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0579 [+0.0314, +0.0872] | 0.0337 | +0.0243 | +0.3678 [+0.2408, +0.4725] | ❌ | ❌ | ❌ | ✅ |
| `pool` | yes | 0.0863 [+0.0412, +0.1441] | 0.0711 | +0.0152 | +0.3738 [+0.1096, +0.5196] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 738 battles / 6400 rollouts · Brier 0.0964 = REL 0.0067 − RES 0.0292 + UNC 0.1183 + WBV 0.0008 (resid -1.00e-04) · base rate 0.8630 · **resolution is 24.7% of the base-rate cap** · corr(turn,V) -0.2132 vs corr(turn,MC) -0.1266

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 748 battles / 6400 rollouts · Brier 0.0938 = REL 0.0030 − RES 0.0359 + UNC 0.1267 + WBV 0.0008 (resid -8.83e-04) · base rate 0.8511 · **resolution is 28.3% of the base-rate cap** · corr(turn,V) -0.0229 vs corr(turn,MC) -0.0615

## 5. THE LEDGER LINE

```
ai_v12_28_ladder_ent05 vs ai_v12_11_ladder_ctrl10M at 10M [OFFLINE-GENERATED: 800 games x 9+3 opponents, full capture]: G1 bot Δ +0.0039 [-0.0012, +0.0092] WITHIN FLOOR · identity bias late Δ -0.0596 [-0.1088, -0.0100] WITHIN FLOOR · turn-contrast Δ -0.1253 [-0.2860, +0.0360] NOT DETECTED · spread ratio t1-3 Δ +0.0221 [-0.0108, +0.0522] WITHIN FLOOR · own-team R2 t1 Δ +0.0149 [+0.0006, +0.0290] NOT DETECTED [QUOTA-MATCHED] · calib slope Δ +0.2168 [+0.1405, +0.2958] DETECTED
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_28_ladder_ent05 --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_28_ladder_ent05 --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_11_ladder_ctrl10M --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/hp800_floor_v6.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_28_ladder_ent05_vs_ctrl10M --nice 15 --ledger-line

# arm — ai_v12_28_ladder_ent05
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_28_ladder_ent05 --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_28_ladder_ent05 --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_28_ladder_ent05` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_28_ladder_ent05/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_11_ladder_ctrl10M__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_11_ladder_ctrl10M__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_11_ladder_ctrl10M__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
