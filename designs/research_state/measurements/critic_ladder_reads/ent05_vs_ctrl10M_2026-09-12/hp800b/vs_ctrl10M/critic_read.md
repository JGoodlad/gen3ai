# CRITIC READ — `ai_v12_28_ladder_ent05` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v6 at 2026-09-12T22:23:33. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

**SELF-PLAY CROSSING** — arm 4,128,768 · control 4,128,768 (first step carrying any `*_pool` scalar; the PROMOTION scalar itself lands one eval→rollout lag earlier, at 4,000,032 / 4,000,032). Descriptive — no floor, no verdict.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_28_ladder_ent05` | `step_10000032` | 9600 | 0.4% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 9600 | 0.4% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 800 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260911, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_28_ladder_ent05` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_28_ladder_ent05` | **794/324/12** | 9600 | 8255 / 1308 / 37 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **797/366/10** | 9600 | 8280 / 1277 / 43 | 12 |

**Frames:** the realized per-opponent caps differ — arm 794/324/12, control 797/366/10 (traced W/L/D per opponent, counted on disk); control cut to 794/324/10

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **794/324/10** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/hp800b_floor_v6.json` — 74 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0302 | +0.0188 | **+0.0114** | [+0.0052, +0.0172] | **DETECTED** (CI clears the floor +0.0051) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0842 | -0.0355 | **-0.0487** | [-0.0890, -0.0078] | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.1954 | +0.0584 | **-0.2538** | [-0.3814, -0.1274] | **DETECTED** (CI clears the floor +0.0164) |
| skill · `bot` | — | +0.2429 | +0.1472 | **+0.0957** | [+0.0362, +0.1568] | **NOT DETECTED** (CI does not clear the floor 0.0733) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.1355 | +0.1535 | **-0.0180** | [-0.0480, +0.0091] | **WITHIN FLOOR** (|delta| <= floor 0.0671) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 794/324/10 | +0.0586 | +0.0594 | **-0.0008** | [-0.0155, +0.0140] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.3433 | +1.0668 | **+0.2765** | [+0.1956, +0.3583] | **NOT DETECTED** (CI does not clear the floor 0.2001) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0420 | +0.0391 | **+0.0029** | [-0.0034, +0.0089] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0302 | +0.0188 | **+0.0114** | [+0.0052, +0.0172] | 400 | **DETECTED** (CI clears the floor +0.0051) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0503 | +0.0574 | **-0.0071** | [-0.0176, +0.0036] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0017 | +0.0013 | **+0.0004** | [-0.0006, +0.0014] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0026) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0037 | +0.0053 | **-0.0016** | [-0.0035, +0.0001] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0053) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0006 | +0.0016 | **-0.0010** | [-0.0029, +0.0005] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0232) |
| ece | `all` | `capture-rate (gauge)` | +0.0393 | +0.0243 | **+0.0150** | [+0.0035, +0.0255] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| ece | `bot` | `capture-rate (gauge)` | +0.0551 | +0.0493 | **+0.0057** | [-0.0051, +0.0162] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| ece | `pool` | `capture-rate (gauge)` | +0.0188 | +0.0342 | **-0.0154** | [-0.0328, +0.0029] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1097) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2751 | +0.2650 | **+0.0101** | [-0.0267, +0.0468] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0454) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2429 | +0.1472 | **+0.0957** | [+0.0362, +0.1568] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0733) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2440 | +0.2596 | **-0.0157** | [-0.0659, +0.0362] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1381) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | -0.0240 | +0.0069 | **-0.0309** | [-0.0564, -0.0056] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0934) |
| bias V - p_hat | `ALL` | `pop` | -0.0629 | -0.0408 | **-0.0222** | [-0.0420, -0.0019] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0744) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0629 | -0.0408 | **-0.0222** | [-0.0420, -0.0019] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0744) |
| bias V - p_hat | `early (turn<=10)` | `raw` | -0.0097 | -0.0325 | **+0.0228** | [-0.0170, +0.0607] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1037) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0538 | -0.0674 | **+0.0136** | [-0.0210, +0.0471] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0793) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0538 | -0.0674 | **+0.0136** | [-0.0210, +0.0471] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0793) |
| bias V - p_hat | `mid (11-24)` | `raw` | -0.0155 | +0.0294 | **-0.0450** | [-0.0849, -0.0066] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0642) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0540 | -0.0195 | **-0.0346** | [-0.0646, -0.0048] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0393) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0540 | -0.0195 | **-0.0346** | [-0.0646, -0.0048] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0393) |
| bias V - p_hat | `late (turn>=25)` | `raw` | -0.0564 | +0.0255 | **-0.0819** | [-0.1348, -0.0284] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1252) |
| bias V - p_hat | `late (turn>=25)` | `pop` | -0.0842 | -0.0355 | **-0.0487** | [-0.0890, -0.0078] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0842 | -0.0355 | **-0.0487** | [-0.0890, -0.0078] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| bias V - p_hat | `bot` | `raw` | -0.0120 | -0.0065 | **-0.0055** | [-0.0387, +0.0267] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0666) |
| bias V - p_hat | `bot` | `pop` | -0.0651 | -0.0497 | **-0.0154** | [-0.0392, +0.0081] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0567) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0651 | -0.0497 | **-0.0154** | [-0.0392, +0.0081] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0567) |
| bias V - p_hat | `pool` | `raw` | -0.0450 | +0.0310 | **-0.0760** | [-0.1157, -0.0380] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1390) |
| bias V - p_hat | `pool` | `pop` | -0.0657 | -0.0143 | **-0.0514** | [-0.0896, -0.0146] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1161) |
| bias V - p_hat ⭐ | `pool` | `ipw` | -0.0657 | -0.0143 | **-0.0514** | [-0.0896, -0.0146] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1161) |
| Murphy resolution | `ALL` | `raw` | +0.0274 | +0.0268 | **+0.0005** | [-0.0104, +0.0107] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0022) |
| Murphy resolution | `ALL` | `pop` | +0.0353 | +0.0323 | **+0.0030** | [-0.0114, +0.0167] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0353 | +0.0323 | **+0.0030** | [-0.0114, +0.0167] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1749 | +0.1550 | **+0.0199** | [-0.0569, +0.0948] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0538) |
| Murphy skill_score | `ALL` | `pop` | +0.2414 | +0.2297 | **+0.0116** | [-0.0954, +0.1196] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0165) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2414 | +0.2297 | **+0.0116** | [-0.0954, +0.1196] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0165) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1833 | +0.1727 | **+0.0106** | [-0.0534, +0.0722] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0125) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2791 | +0.2540 | **+0.0251** | [-0.0621, +0.1050] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2791 | +0.2540 | **+0.0251** | [-0.0621, +0.1050] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0024 | +0.0041 | **-0.0016** | [-0.0059, +0.0024] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0088) |
| Murphy reliability | `ALL` | `pop` | +0.0056 | +0.0039 | **+0.0017** | [-0.0036, +0.0064] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0029) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0056 | +0.0039 | **+0.0017** | [-0.0036, +0.0064] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0029) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.1954 | +0.0584 | **-0.2538** | [-0.3814, -0.1274] | 4000 | **DETECTED** (CI clears the floor +0.0164) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.1589 | +0.0135 | **-0.1724** | [-0.2937, -0.0462] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0487) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.1589 | +0.0135 | **-0.1724** | [-0.2937, -0.0462] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0487) |

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

**arm — `ai_v12_28_ladder_ent05` @ `step_10000032`**: 290887 states / 9563 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 37 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 293683 states / 9557 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 43 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.1355 | +0.1535 | **-0.0180** | [-0.0480, +0.0091] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0671) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 794/324/10 | +0.1364 | +0.1558 | **-0.0194** | [-0.0498, +0.0069] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0925 | -0.0973 | **+0.0049** | [-0.0086, +0.0182] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0147) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5945 | +0.6340 | **-0.0396** | [-0.1195, +0.0441] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2399) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 794/324/10 | +0.5924 | +0.6306 | **-0.0383** | [-0.1258, +0.0488] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0434 | -0.0421 | **-0.0013** | [-0.0146, +0.0125] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0322) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 4-10` | as traced | +0.4013 | +0.5359 | **-0.1346** | [-0.1954, -0.0718] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2833) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 11-24` | as traced | +0.6697 | +0.7383 | **-0.0686** | [-0.1627, +0.0332] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2669) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 794/324/10 | +0.0586 | +0.0594 | **-0.0008** | [-0.0155, +0.0140] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 794/324/10 | +0.0088 | +0.0112 | **-0.0024** | [-0.0073, +0.0029] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 794/324/10 | +0.5070 | +0.4949 | **+0.0121** | [-0.0057, +0.0314] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0227) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1-3` | MATCHED 794/324/10 | +0.5953 | +0.5821 | **+0.0132** | [-0.0031, +0.0293] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 4-10` | MATCHED 794/324/10 | +0.7272 | +0.7130 | **+0.0143** | [-0.0007, +0.0287] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 1-3` | MATCHED 794/324/10 | +0.0220 | +0.0180 | **+0.0040** | [+0.0004, +0.0074] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 4-10` | MATCHED 794/324/10 | +0.1243 | +0.1249 | **-0.0006** | [-0.0176, +0.0163] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 11-24` | MATCHED 794/324/10 | +0.1693 | +0.1841 | **-0.0148** | [-0.0394, +0.0083] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 794/324/10 | +0.0542 | +0.0524 | **+0.0018** | [-0.0028, +0.0055] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0034) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 794/324/10 | +0.0326 | +0.0296 | **+0.0030** | [-0.0005, +0.0065] | 2000 | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 794/324/10 | +1.1186 | +1.6704 | **-0.5518** | [-0.3459, -0.1437] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.7010) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 794/324/10 | +0.4668 | +0.7077 | **-0.2409** | [-0.2291, -0.1399] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 794/324/10 | +0.0087 | +0.0057 | **+0.0030** | [-0.0049, +0.0108] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 794/324/10 | +0.0499 | +0.0531 | **-0.0032** | [-0.0195, +0.0126] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.3433 | +1.0668 | **+0.2765** | [+0.1956, +0.3583] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.2001) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | +0.1394 | +0.3808 | **-0.2415** | [-0.3652, -0.1141] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.0247) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.9974 | +0.7445 | **+0.2529** | [+0.1098, +0.3941] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.1915) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.6156 | +0.9909 | **-0.3754** | [-0.5476, -0.1965] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.4031) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 794/324/10 | +1.2826 | +1.0012 | **+0.2814** | [+0.2009, +0.3606] | 2000 | **DETECTED** (CI clears the floor +0.1729) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.3796 | +1.0346 | **+0.3450** | [+0.2564, +0.4323] | 2000 | **DETECTED** (CI clears the floor +0.2481) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | +0.0920 | +0.4219 | **-0.3299** | [-0.4581, -0.1985] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.0981) |

### HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY — `V` against its opponent-decodable part

| side | window | ratio of `V` | opponent-decodable part | `V` / decodable |
|---|---|---|---|---|
| arm `ai_v12_28_ladder_ent05` | `t1_3` | +0.1355 | +0.0220 | 6.172 |
| arm `ai_v12_28_ladder_ent05` | `t4_10` | +0.4013 | +0.1243 | 3.229 |
| arm `ai_v12_28_ladder_ent05` | `t11_24` | +0.6697 | +0.1693 | 3.955 |
| control `ai_v12_11_ladder_ctrl10M` | `t1_3` | +0.1535 | +0.0181 | 8.481 |
| control `ai_v12_11_ladder_ctrl10M` | `t4_10` | +0.5359 | +0.1251 | 4.284 |
| control `ai_v12_11_ladder_ctrl10M` | `t11_24` | +0.7383 | +0.1844 | 4.004 |

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
| arm · team | 601 | 602 | 9560 | 19120 | 13.0 | 26.0 | 4–88 |
| arm · stratum | 5 | 602 | 9560 | 19120 | 1919.0 | 3838.0 | 1873–1928 |
| arm · between-team spread | 601 | 602 | 9560 | — | 13.0 | — | — |
| control · team | 601 | 602 | 9554 | 19108 | 13.0 | 26.0 | 4–88 |
| control · stratum | 5 | 602 | 9554 | 19108 | 1912.0 | 3824.0 | 1902–1917 |
| control · between-team spread | 601 | 602 | 9554 | — | 13.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 794/324/10 | +0.0542 | +0.0524 | **+0.0018** | [-0.0028, +0.0055] | **WITHIN FLOOR** (|delta| <= floor 0.0034) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 794/324/10 | +0.0326 | +0.0296 | **+0.0030** | [-0.0005, +0.0065] | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 794/324/10 | +1.1186 | +1.6704 | **-0.5518** | [-0.3459, -0.1437] | **WITHIN FLOOR** (|delta| <= floor 0.7010) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 794/324/10 | +0.0586 | +0.0594 | **-0.0008** | [-0.0155, +0.0140] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 794/324/10 | +0.0087 | +0.0057 | **+0.0030** | [-0.0049, +0.0108] | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 794/324/10 | +0.0499 | +0.0531 | **-0.0032** | [-0.0195, +0.0126] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 19126 | 9563 | 0.8113 | **0.1850** | **1.3834** | [0.205, 0.989] | 0.0000 |
| arm · t1_3 | `t1_3` | 19126 | 9563 | 0.7771 | **0.0796** | **0.4741** | [0.596, 0.905] | 0.0000 |
| arm · common support | `common support` | 18163 | 9471 | 0.8258 | **0.1450** | **1.1004** | [0.402, 0.984] | 0.0000 |
| control · all | `all` | 19114 | 9557 | 0.8240 | **0.1990** | **1.7526** | [0.208, 0.997] | 0.0000 |
| control · t1_3 | `t1_3` | 19114 | 9557 | 0.7666 | **0.1129** | **0.6593** | [0.489, 0.932] | 0.0000 |
| control · common support | `common support` | 16405 | 9155 | 0.8219 | **0.1639** | **1.2597** | [0.353, 0.986] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.2082, 0.9888]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.3433 | +1.0668 | **+0.2765** | [+0.1956, +0.3583] | 0.2001 | **NOT DETECTED** (CI does not clear the floor 0.2001) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | +0.1394 | +0.3808 | **-0.2415** | [-0.3652, -0.1141] | 1.0247 | **WITHIN FLOOR** (|delta| <= floor 1.0247) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.9974 | +0.7445 | **+0.2529** | [+0.1098, +0.3941] | 0.1915 | **NOT DETECTED** (CI does not clear the floor 0.1915) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.6156 | +0.9909 | **-0.3754** | [-0.5476, -0.1965] | 0.4031 | **WITHIN FLOOR** (|delta| <= floor 0.4031) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 794/324/10 | +1.2826 | +1.0012 | **+0.2814** | [+0.2009, +0.3606] | 0.1729 | **DETECTED** (CI clears the floor +0.1729) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.3796 | +1.0346 | **+0.3450** | [+0.2564, +0.4323] | 0.2481 | **DETECTED** (CI clears the floor +0.2481) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +0.0920 | +0.4219 | **-0.3299** | [-0.4581, -0.1985] | 1.0981 | **WITHIN FLOOR** (|delta| <= floor 1.0981) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.1364 [+0.1537, +0.1581] | +0.1558 | **-0.0194** | [-0.0498, +0.0069] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.1364 [+0.1557, +0.1557] | +0.1557 | **-0.0193** | [-0.0490, +0.0075] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1364 | +0.1557 | **-0.0193** | [-0.0490, +0.0075] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.5924 [+0.6271, +0.6367] | +0.6306 | **-0.0383** | [-0.1258, +0.0488] | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.5924 [+0.6323, +0.6323] | +0.6323 | **-0.0399** | [-0.1187, +0.0425] | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.5924 | +0.6323 | **-0.0399** | [-0.1187, +0.0425] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.0586 [+0.0571, +0.0641] | +0.0594 | **-0.0008** | [-0.0155, +0.0140] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| `cond.own_team_r2.t1` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.0586 [+0.0621, +0.0621] | +0.0621 | **-0.0035** | [-0.0185, +0.0105] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0586 | +0.0621 | **-0.0035** | [-0.0185, +0.0105] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.0088 [+0.0103, +0.0122] | +0.0112 | **-0.0024** | [-0.0073, +0.0029] | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| `cond.own_team_r2.all` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.0088 [+0.0118, +0.0118] | +0.0118 | **-0.0031** | [-0.0083, +0.0026] | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0088 | +0.0118 | **-0.0031** | [-0.0083, +0.0026] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.5070 [+0.4927, +0.5008] | +0.4949 | **+0.0121** | [-0.0057, +0.0314] | **WITHIN FLOOR** (|delta| <= floor 0.0227) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.5070 [+0.4887, +0.4887] | +0.4887 | **+0.0183** | [-0.0015, +0.0381] | **WITHIN FLOOR** (|delta| <= floor 0.0227) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5070 | +0.4887 | **+0.0183** | [-0.0015, +0.0381] | *no label — not a reading* |
| `cond.opp_class_auc.t1_3` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.5953 [+0.5807, +0.5835] | +0.5821 | **+0.0132** | [-0.0031, +0.0293] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1_3` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.5953 [+0.5808, +0.5808] | +0.5808 | **+0.0145** | [-0.0028, +0.0309] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1_3` | UNMATCHED (as traced) | +0.5953 | +0.5808 | **+0.0145** | [-0.0028, +0.0309] | *no label — not a reading* |
| `cond.opp_class_auc.t4_10` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.7272 [+0.7117, +0.7148] | +0.7130 | **+0.0143** | [-0.0007, +0.0287] | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| `cond.opp_class_auc.t4_10` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.7272 [+0.7131, +0.7131] | +0.7131 | **+0.0141** | [-0.0005, +0.0284] | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| `cond.opp_class_auc.t4_10` | UNMATCHED (as traced) | +0.7272 | +0.7131 | **+0.0141** | [-0.0005, +0.0284] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.0220 [+0.0173, +0.0185] | +0.0180 | **+0.0040** | [+0.0004, +0.0074] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.0220 [+0.0181, +0.0181] | +0.0181 | **+0.0039** | [+0.0001, +0.0073] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | UNMATCHED (as traced) | +0.0220 | +0.0181 | **+0.0039** | [+0.0001, +0.0073] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.1243 [+0.1230, +0.1266] | +0.1249 | **-0.0006** | [-0.0176, +0.0163] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.1243 [+0.1251, +0.1251] | +0.1251 | **-0.0008** | [-0.0165, +0.0154] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | UNMATCHED (as traced) | +0.1243 | +0.1251 | **-0.0008** | [-0.0165, +0.0154] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.1693 [+0.1829, +0.1858] | +0.1841 | **-0.0148** | [-0.0394, +0.0083] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.1693 [+0.1844, +0.1844] | +0.1844 | **-0.0151** | [-0.0377, +0.0079] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | UNMATCHED (as traced) | +0.1693 | +0.1844 | **-0.0151** | [-0.0377, +0.0079] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.0542 [+0.0519, +0.0529] | +0.0524 | **+0.0018** | [-0.0028, +0.0055] | **WITHIN FLOOR** (|delta| <= floor 0.0034) |
| `cond.within_team_resolution.all` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.0542 [+0.0530, +0.0530] | +0.0530 | **+0.0012** | [-0.0035, +0.0049] | **WITHIN FLOOR** (|delta| <= floor 0.0034) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0542 | +0.0530 | **+0.0012** | [-0.0035, +0.0049] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.0326 [+0.0294, +0.0300] | +0.0296 | **+0.0030** | [-0.0005, +0.0065] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.0326 [+0.0292, +0.0292] | +0.0292 | **+0.0035** | [-0.0001, +0.0071] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0326 | +0.0292 | **+0.0035** | [-0.0001, +0.0071] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +1.1186 [+1.5775, +1.7711] | +1.6704 | **-0.5518** | [-0.3459, -0.1437] | **WITHIN FLOOR** (|delta| <= floor 0.7010) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +1.1186 [+1.7001, +1.7001] | +1.7001 | **-0.5815** | [-0.3591, -0.1474] | **WITHIN FLOOR** (|delta| <= floor 0.7010) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +1.1186 | +1.7001 | **-0.5815** | [-0.3591, -0.1474] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.4668 [+0.7001, +0.7145] | +0.7077 | **-0.2409** | [-0.2291, -0.1399] | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.4668 [+0.7151, +0.7151] | +0.7151 | **-0.2484** | [-0.2356, -0.1452] | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.4668 | +0.7151 | **-0.2484** | [-0.2356, -0.1452] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.0087 [+0.0042, +0.0077] | +0.0057 | **+0.0030** | [-0.0049, +0.0108] | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| `cond.own_team_r2.late` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.0087 [+0.0060, +0.0060] | +0.0060 | **+0.0027** | [-0.0049, +0.0102] | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0087 | +0.0060 | **+0.0027** | [-0.0049, +0.0102] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +0.0499 [+0.0511, +0.0597] | +0.0531 | **-0.0032** | [-0.0195, +0.0126] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +0.0499 [+0.0561, +0.0561] | +0.0561 | **-0.0061** | [-0.0226, +0.0091] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0499 | +0.0561 | **-0.0061** | [-0.0226, +0.0091] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 794/324/10 (9512 battles, 9509 decoder battles, 21 seeds) | +1.2826 [+0.9931, +1.0058] | +1.0012 | **+0.2814** | [+0.2009, +0.3606] | **DETECTED** (CI clears the floor +0.1729) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 992/405/10 (9557 battles, 9554 decoder battles, 21 seeds) | +1.2826 [+1.0078, +1.0078] | +1.0078 | **+0.2748** | [+0.1923, +0.3544] | **DETECTED** (CI clears the floor +0.1729) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.2826 | +1.0078 | **+0.2748** | [+0.1923, +0.3544] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.1355 | [+0.1202, +0.1538] | +0.1535 | [+0.1343, +0.1792] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1364 | [+0.1213, +0.1544] | +0.1557 | [+0.1370, +0.1811] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0925 | [-0.1017, -0.0837] | -0.0973 | [-0.1076, -0.0881] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5945 | [+0.5394, +0.6582] | +0.6340 | [+0.5749, +0.6984] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.5924 | [+0.5379, +0.6550] | +0.6323 | [+0.5737, +0.6958] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0434 | [-0.0533, -0.0338] | -0.0421 | [-0.0529, -0.0324] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 4-10` | +0.4013 | [+0.3651, +0.4423] | +0.5359 | [+0.4886, +0.5887] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 11-24` | +0.6697 | [+0.6065, +0.7417] | +0.7383 | [+0.6690, +0.8124] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0586 | [+0.0483, +0.0680] | +0.0621 | [+0.0517, +0.0730] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0088 | [+0.0054, +0.0120] | +0.0118 | [+0.0074, +0.0159] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5070 | [+0.4938, +0.5205] | +0.4887 | [+0.4750, +0.5022] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1-3` | +0.5953 | [+0.5841, +0.6066] | +0.5808 | [+0.5688, +0.5930] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 4-10` | +0.7272 | [+0.7172, +0.7370] | +0.7131 | [+0.7027, +0.7231] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 1-3` | +0.0220 | [+0.0197, +0.0246] | +0.0181 | [+0.0160, +0.0208] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 4-10` | +0.1243 | [+0.1134, +0.1364] | +0.1251 | [+0.1142, +0.1371] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 11-24` | +0.1693 | [+0.1544, +0.1860] | +0.1844 | [+0.1686, +0.2018] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0542 | [+0.0542, +0.0600] | +0.0530 | [+0.0535, +0.0594] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0326 | [+0.0304, +0.0357] | +0.0292 | [+0.0269, +0.0320] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +1.1186 | [+0.4170, +0.5304] | +1.7001 | [+0.6368, +0.8152] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.4668 | [+0.3337, +0.3827] | +0.7151 | [+0.5106, +0.5848] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0087 | [+0.0027, +0.0142] | +0.0060 | [+0.0006, +0.0109] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0499 | [+0.0385, +0.0607] | +0.0561 | [+0.0456, +0.0676] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.3433 | [+1.2838, +1.4094] | +1.0668 | [+1.0179, +1.1195] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | +0.1394 | [+0.0485, +0.2347] | +0.3808 | [+0.3027, +0.4604] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.9974 | [+0.8880, +1.1157] | +0.7445 | [+0.6597, +0.8303] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.6156 | [+0.4770, +0.7581] | +0.9909 | [+0.8862, +1.0995] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.2826 | [+1.2207, +1.3497] | +1.0078 | [+0.9620, +1.0592] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.3796 | [+1.3155, +1.4504] | +1.0346 | [+0.9813, +1.0935] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | +0.0920 | [-0.0051, +0.1926] | +0.4219 | [+0.3358, +0.5085] |

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

Identity, capture-rate reweighted: 800 labels / 748 battles / 6400 rollouts · Brier 0.0960 = REL 0.0056 − RES 0.0353 + UNC 0.1266 + WBV 0.0008 (resid -1.62e-03) · base rate 0.8513 · **resolution is 27.9% of the base-rate cap** · corr(turn,V) -0.2728 vs corr(turn,MC) -0.0774

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 741 battles / 6400 rollouts · Brier 0.0980 = REL 0.0039 − RES 0.0323 + UNC 0.1272 + WBV 0.0008 (resid -1.61e-03) · base rate 0.8504 · **resolution is 25.4% of the base-rate cap** · corr(turn,V) -0.0506 vs corr(turn,MC) -0.1089

## 5. THE LEDGER LINE

```
ai_v12_28_ladder_ent05 vs ai_v12_11_ladder_ctrl10M at 10M [OFFLINE-GENERATED: 800 games x 9+3 opponents, full capture]: G1 bot Δ +0.0114 [+0.0052, +0.0172] DETECTED · identity bias late Δ -0.0487 [-0.0890, -0.0078] WITHIN FLOOR · turn-contrast Δ -0.2538 [-0.3814, -0.1274] DETECTED · spread ratio t1-3 Δ -0.0180 [-0.0480, +0.0091] WITHIN FLOOR · own-team R2 t1 Δ -0.0008 [-0.0155, +0.0140] WITHIN FLOOR [QUOTA-MATCHED] · calib slope Δ +0.2765 [+0.1956, +0.3583] NOT DETECTED
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_28_ladder_ent05 --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_28_ladder_ent05 --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_11_ladder_ctrl10M --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/hp800b_floor_v6.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_28_ladder_ent05_vs_ctrl10M --nice 15 --ledger-line

# arm — ai_v12_28_ladder_ent05
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_28_ladder_ent05 --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_28_ladder_ent05 --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_28_ladder_ent05` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_28_ladder_ent05/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_11_ladder_ctrl10M__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_11_ladder_ctrl10M__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_11_ladder_ctrl10M__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_28_ladder_ent05_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
