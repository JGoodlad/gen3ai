# CRITIC READ — `ai_v12_29_ladder_vf025` vs `ai_v12_15_ladder_ctrl10M_b`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v6 at 2026-09-13T02:52:50. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

**SELF-PLAY CROSSING** — arm 4,128,768 · control 4,128,768 (first step carrying any `*_pool` scalar; the PROMOTION scalar itself lands one eval→rollout lag earlier, at 4,000,032 / 4,000,032). Descriptive — no floor, no verdict.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_29_ladder_vf025` | `step_10000032` | 9600 | 0.4% | 150/150 (100.0%) | no |
| control | `ai_v12_15_ladder_ctrl10M_b` | `step_10000032` | 9600 | 0.6% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 800 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260911, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_29_ladder_vf025` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_15_ladder_ctrl10M_b` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_29_ladder_vf025` | **796/285/9** | 9600 | 8409 / 1149 / 42 | 12 |
| control | `ai_v12_15_ladder_ctrl10M_b` | **798/347/10** | 9600 | 8176 / 1366 / 58 | 12 |

**Frames:** the realized per-opponent caps differ — arm 796/285/9, control 798/347/10 (traced W/L/D per opponent, counted on disk); control cut to 796/285/9

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **796/285/9** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/hp800b_floor_v6.json` — 74 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0244 | +0.0184 | **+0.0060** | [-0.0005, +0.0122] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0125 | +0.0845 | **-0.0970** | [-0.1430, -0.0498] | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.1812 | +0.0441 | **-0.2253** | [-0.3422, -0.1065] | **DETECTED** (CI clears the floor +0.0164) |
| skill · `bot` | — | +0.2392 | +0.1867 | **+0.0524** | [+0.0035, +0.0991] | **WITHIN FLOOR** (|delta| <= floor 0.0733) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0917 | +0.0864 | **+0.0053** | [-0.0156, +0.0310] | **WITHIN FLOOR** (|delta| <= floor 0.0671) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 796/285/9 | +0.0742 | +0.0253 | **+0.0490** | [+0.0365, +0.0622] | **DETECTED** (CI clears the floor +0.0359) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.2977 | +1.2669 | **+0.0308** | [-0.0588, +0.1174] | **WITHIN FLOOR** (|delta| <= floor 0.2001) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0370 | +0.0367 | **+0.0003** | [-0.0065, +0.0065] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0024) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0244 | +0.0184 | **+0.0060** | [-0.0005, +0.0122] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0522 | +0.0507 | **+0.0014** | [-0.0096, +0.0131] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0067) |
| reliability | `all` | `capture-rate (gauge)` | +0.0003 | +0.0039 | **-0.0036** | [-0.0054, -0.0023] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0026) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0007 | +0.0001 | **+0.0007** | [+0.0003, +0.0013] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0053) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0044 | +0.0248 | **-0.0203** | [-0.0285, -0.0132] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0232) |
| ece | `all` | `capture-rate (gauge)` | +0.0093 | +0.0488 | **-0.0395** | [-0.0491, -0.0285] | 400 | **DETECTED** (CI clears the floor +0.0245) |
| ece | `bot` | `capture-rate (gauge)` | +0.0217 | +0.0045 | **+0.0172** | [+0.0079, +0.0250] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| ece | `pool` | `capture-rate (gauge)` | +0.0550 | +0.1439 | **-0.0889** | [-0.1161, -0.0619] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1097) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2728 | +0.2196 | **+0.0532** | [+0.0185, +0.0863] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0454) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2392 | +0.1867 | **+0.0524** | [+0.0035, +0.0991] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0733) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2444 | +0.1215 | **+0.1229** | [+0.0746, +0.1796] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1381) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0382 | +0.1003 | **-0.0621** | [-0.0898, -0.0344] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0934) |
| bias V - p_hat | `ALL` | `pop` | -0.0115 | +0.0337 | **-0.0452** | [-0.0660, -0.0242] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0744) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0115 | +0.0337 | **-0.0452** | [-0.0660, -0.0242] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0744) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0420 | +0.0712 | **-0.0292** | [-0.0707, +0.0126] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1037) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0131 | +0.0120 | **-0.0251** | [-0.0568, +0.0070] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0793) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0131 | +0.0120 | **-0.0251** | [-0.0568, +0.0070] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0793) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0438 | +0.0936 | **-0.0498** | [-0.0923, -0.0077] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0642) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0078 | +0.0199 | **-0.0276** | [-0.0590, +0.0024] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0393) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0078 | +0.0199 | **-0.0276** | [-0.0590, +0.0024] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0393) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0241 | +0.1507 | **-0.1266** | [-0.1840, -0.0683] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.1252) |
| bias V - p_hat | `late (turn>=25)` | `pop` | -0.0125 | +0.0845 | **-0.0970** | [-0.1430, -0.0498] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0125 | +0.0845 | **-0.0970** | [-0.1430, -0.0498] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| bias V - p_hat | `bot` | `raw` | +0.0243 | +0.0600 | **-0.0358** | [-0.0706, -0.0032] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0666) |
| bias V - p_hat | `bot` | `pop` | -0.0224 | +0.0070 | **-0.0294** | [-0.0517, -0.0080] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0567) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0224 | +0.0070 | **-0.0294** | [-0.0517, -0.0080] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0567) |
| bias V - p_hat | `pool` | `raw` | +0.0641 | +0.1700 | **-0.1059** | [-0.1539, -0.0597] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1390) |
| bias V - p_hat | `pool` | `pop` | +0.0161 | +0.1018 | **-0.0857** | [-0.1288, -0.0422] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1161) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0161 | +0.1018 | **-0.0857** | [-0.1288, -0.0422] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1161) |
| Murphy resolution | `ALL` | `raw` | +0.0310 | +0.0290 | **+0.0020** | [-0.0096, +0.0141] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0022) |
| Murphy resolution | `ALL` | `pop` | +0.0360 | +0.0353 | **+0.0007** | [-0.0143, +0.0164] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0030) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0360 | +0.0353 | **+0.0007** | [-0.0143, +0.0164] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0030) |
| Murphy skill_score | `ALL` | `raw` | +0.1925 | +0.1011 | **+0.0913** | [+0.0297, +0.1540] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0538) |
| Murphy skill_score | `ALL` | `pop` | +0.2910 | +0.2459 | **+0.0451** | [-0.0406, +0.1313] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2910 | +0.2459 | **+0.0451** | [-0.0406, +0.1313] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.2004 | +0.1602 | **+0.0402** | [-0.0214, +0.1078] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2969 | +0.2547 | **+0.0422** | [-0.0478, +0.1384] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2969 | +0.2547 | **+0.0422** | [-0.0478, +0.1384] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0031 | +0.0129 | **-0.0097** | [-0.0161, -0.0038] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0088) |
| Murphy reliability | `ALL` | `pop` | +0.0015 | +0.0026 | **-0.0011** | [-0.0047, +0.0026] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0029) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0015 | +0.0026 | **-0.0011** | [-0.0047, +0.0026] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0029) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.1812 | +0.0441 | **-0.2253** | [-0.3422, -0.1065] | 4000 | **DETECTED** (CI clears the floor +0.0164) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.1337 | +0.0462 | **-0.1800** | [-0.3000, -0.0619] | 4000 | **DETECTED** (CI clears the floor +0.0487) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.1337 | +0.0462 | **-0.1800** | [-0.3000, -0.0619] | 4000 | **DETECTED** (CI clears the floor +0.0487) |

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

**arm — `ai_v12_29_ladder_vf025` @ `step_10000032`**: 281590 states / 9558 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 42 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_15_ladder_ctrl10M_b` @ `step_10000032`**: 285588 states / 9542 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 58 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0917 | +0.0864 | **+0.0053** | [-0.0156, +0.0310] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0671) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 796/285/9 | +0.0946 | +0.0878 | **+0.0068** | [-0.0134, +0.0305] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.0808 | -0.1120 | **+0.0312** | [+0.0178, +0.0439] | 2000 | **DETECTED** (CI clears the floor +0.0147) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.4904 | +0.3941 | **+0.0963** | [+0.0277, +0.1665] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2399) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 796/285/9 | +0.4884 | +0.3934 | **+0.0950** | [+0.0264, +0.1645] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0454 | -0.0743 | **+0.0289** | [+0.0155, +0.0421] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0322) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 4-10` | as traced | +0.3318 | +0.2526 | **+0.0792** | [+0.0328, +0.1281] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2833) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 11-24` | as traced | +0.5402 | +0.4714 | **+0.0688** | [-0.0126, +0.1512] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2669) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 796/285/9 | +0.0742 | +0.0253 | **+0.0490** | [+0.0365, +0.0622] | 2000 | **DETECTED** (CI clears the floor +0.0359) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 796/285/9 | +0.0109 | +0.0068 | **+0.0040** | [-0.0016, +0.0095] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 796/285/9 | +0.5097 | +0.4921 | **+0.0176** | [-0.0012, +0.0358] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0227) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1-3` | MATCHED 796/285/9 | +0.5541 | +0.5839 | **-0.0298** | [-0.0463, -0.0130] | 2000 | **DETECTED** (CI clears the floor +0.0091) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 4-10` | MATCHED 796/285/9 | +0.6788 | +0.6928 | **-0.0140** | [-0.0295, +0.0012] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 1-3` | MATCHED 796/285/9 | +0.0088 | +0.0187 | **-0.0099** | [-0.0126, -0.0071] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 4-10` | MATCHED 796/285/9 | +0.0772 | +0.1025 | **-0.0254** | [-0.0383, -0.0133] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 11-24` | MATCHED 796/285/9 | +0.1116 | +0.1613 | **-0.0498** | [-0.0680, -0.0314] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 796/285/9 | +0.0447 | +0.0486 | **-0.0039** | [-0.0077, +0.0003] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 796/285/9 | +0.0271 | +0.0296 | **-0.0026** | [-0.0062, +0.0009] | 2000 | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 796/285/9 | +1.2800 | +0.9727 | **+0.3073** | [+0.0063, +0.1723] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.7010) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 796/285/9 | +0.5116 | +0.4227 | **+0.0889** | [+0.0288, +0.1011] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 796/285/9 | +0.0052 | +0.0002 | **+0.0050** | [-0.0008, +0.0109] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 796/285/9 | +0.0691 | +0.0255 | **+0.0436** | [+0.0303, +0.0577] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.2977 | +1.2669 | **+0.0308** | [-0.0588, +0.1174] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2001) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | -0.2651 | -0.6439 | **+0.3788** | [+0.2160, +0.5367] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.0247) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.9822 | +0.6558 | **+0.3264** | [+0.1851, +0.4688] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.1915) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.4737 | +0.5878 | **-0.1141** | [-0.3521, +0.1166] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.4031) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 796/285/9 | +1.2352 | +1.2205 | **+0.0147** | [-0.0765, +0.1018] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.3393 | +1.2688 | **+0.0705** | [-0.0273, +0.1692] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2481) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | -0.3378 | -0.6486 | **+0.3108** | [+0.1331, +0.4916] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.0981) |

### HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY — `V` against its opponent-decodable part

| side | window | ratio of `V` | opponent-decodable part | `V` / decodable |
|---|---|---|---|---|
| arm `ai_v12_29_ladder_vf025` | `t1_3` | +0.0917 | +0.0088 | 10.430 |
| arm `ai_v12_29_ladder_vf025` | `t4_10` | +0.3318 | +0.0772 | 4.300 |
| arm `ai_v12_29_ladder_vf025` | `t11_24` | +0.5402 | +0.1116 | 4.842 |
| control `ai_v12_15_ladder_ctrl10M_b` | `t1_3` | +0.0864 | +0.0186 | 4.644 |
| control `ai_v12_15_ladder_ctrl10M_b` | `t4_10` | +0.2526 | +0.1026 | 2.461 |
| control `ai_v12_15_ladder_ctrl10M_b` | `t11_24` | +0.4714 | +0.1619 | 2.911 |

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
| arm · team | 601 | 602 | 9555 | 19110 | 13.0 | 26.0 | 4–88 |
| arm · stratum | 5 | 602 | 9555 | 19110 | 1912.0 | 3824.0 | 1903–1916 |
| arm · between-team spread | 601 | 602 | 9555 | — | 13.0 | — | — |
| control · team | 601 | 602 | 9539 | 19078 | 13.0 | 26.0 | 4–88 |
| control · stratum | 5 | 602 | 9539 | 19078 | 1909.0 | 3818.0 | 1902–1912 |
| control · between-team spread | 601 | 602 | 9539 | — | 13.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 796/285/9 | +0.0447 | +0.0486 | **-0.0039** | [-0.0077, +0.0003] | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 796/285/9 | +0.0271 | +0.0296 | **-0.0026** | [-0.0062, +0.0009] | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 796/285/9 | +1.2800 | +0.9727 | **+0.3073** | [+0.0063, +0.1723] | **WITHIN FLOOR** (|delta| <= floor 0.7010) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 796/285/9 | +0.0742 | +0.0253 | **+0.0490** | [+0.0365, +0.0622] | **DETECTED** (CI clears the floor +0.0359) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 796/285/9 | +0.0052 | +0.0002 | **+0.0050** | [-0.0008, +0.0109] | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 796/285/9 | +0.0691 | +0.0255 | **+0.0436** | [+0.0303, +0.0577] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 19116 | 9558 | 0.8653 | **0.1550** | **1.3620** | [0.368, 0.994] | 0.0000 |
| arm · t1_3 | `t1_3` | 19116 | 9558 | 0.8262 | **0.0782** | **0.5614** | [0.638, 0.944] | 0.0000 |
| arm · common support | `common support` | 18094 | 9445 | 0.8814 | **0.1078** | **1.1080** | [0.567, 0.991] | 0.0000 |
| control · all | `all` | 19084 | 9542 | 0.8759 | **0.1479** | **1.3897** | [0.400, 0.995] | 0.0000 |
| control · t1_3 | `t1_3` | 19084 | 9542 | 0.8574 | **0.0710** | **0.5907** | [0.685, 0.959] | 0.0000 |
| control · common support | `common support` | 17851 | 9412 | 0.8881 | **0.1051** | **1.1157** | [0.571, 0.991] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.3997, 0.9937]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.2977 | +1.2669 | **+0.0308** | [-0.0588, +0.1174] | 0.2001 | **WITHIN FLOOR** (|delta| <= floor 0.2001) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | -0.2651 | -0.6439 | **+0.3788** | [+0.2160, +0.5367] | 1.0247 | **WITHIN FLOOR** (|delta| <= floor 1.0247) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.9822 | +0.6558 | **+0.3264** | [+0.1851, +0.4688] | 0.1915 | **NOT DETECTED** (CI does not clear the floor 0.1915) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.4737 | +0.5878 | **-0.1141** | [-0.3521, +0.1166] | 0.4031 | **WITHIN FLOOR** (|delta| <= floor 0.4031) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 796/285/9 | +1.2352 | +1.2205 | **+0.0147** | [-0.0765, +0.1018] | 0.1729 | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.3393 | +1.2688 | **+0.0705** | [-0.0273, +0.1692] | 0.2481 | **WITHIN FLOOR** (|delta| <= floor 0.2481) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | -0.3378 | -0.6486 | **+0.3108** | [+0.1331, +0.4916] | 1.0981 | **WITHIN FLOOR** (|delta| <= floor 1.0981) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.0946 [+0.0860, +0.0899] | +0.0878 | **+0.0068** | [-0.0134, +0.0305] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.0946 [+0.0877, +0.0877] | +0.0877 | **+0.0069** | [-0.0137, +0.0318] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.0946 | +0.0877 | **+0.0069** | [-0.0137, +0.0318] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.4884 [+0.3895, +0.3953] | +0.3934 | **+0.0950** | [+0.0264, +0.1645] | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.4884 [+0.3933, +0.3933] | +0.3933 | **+0.0950** | [+0.0273, +0.1641] | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.4884 | +0.3933 | **+0.0950** | [+0.0273, +0.1641] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.0742 [+0.0232, +0.0285] | +0.0253 | **+0.0490** | [+0.0365, +0.0622] | **DETECTED** (CI clears the floor +0.0359) |
| `cond.own_team_r2.t1` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.0742 [+0.0256, +0.0256] | +0.0256 | **+0.0486** | [+0.0361, +0.0615] | **DETECTED** (CI clears the floor +0.0359) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0742 | +0.0256 | **+0.0486** | [+0.0361, +0.0615] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.0109 [+0.0058, +0.0080] | +0.0068 | **+0.0040** | [-0.0016, +0.0095] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.0109 [+0.0067, +0.0067] | +0.0067 | **+0.0042** | [-0.0009, +0.0094] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0109 | +0.0067 | **+0.0042** | [-0.0009, +0.0094] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.5097 [+0.4856, +0.4974] | +0.4921 | **+0.0176** | [-0.0012, +0.0358] | **WITHIN FLOOR** (|delta| <= floor 0.0227) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.5097 [+0.4820, +0.4820] | +0.4820 | **+0.0277** | [+0.0087, +0.0467] | **NOT DETECTED** (CI does not clear the floor 0.0227) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5097 | +0.4820 | **+0.0277** | [+0.0087, +0.0467] | *no label — not a reading* |
| `cond.opp_class_auc.t1_3` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.5541 [+0.5819, +0.5853] | +0.5839 | **-0.0298** | [-0.0463, -0.0130] | **DETECTED** (CI clears the floor +0.0091) |
| `cond.opp_class_auc.t1_3` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.5541 [+0.5819, +0.5819] | +0.5819 | **-0.0277** | [-0.0430, -0.0108] | **DETECTED** (CI clears the floor +0.0091) |
| `cond.opp_class_auc.t1_3` | UNMATCHED (as traced) | +0.5541 | +0.5819 | **-0.0277** | [-0.0430, -0.0108] | *no label — not a reading* |
| `cond.opp_class_auc.t4_10` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.6788 [+0.6909, +0.6941] | +0.6928 | **-0.0140** | [-0.0295, +0.0012] | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| `cond.opp_class_auc.t4_10` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.6788 [+0.6931, +0.6931] | +0.6931 | **-0.0143** | [-0.0289, +0.0017] | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| `cond.opp_class_auc.t4_10` | UNMATCHED (as traced) | +0.6788 | +0.6931 | **-0.0143** | [-0.0289, +0.0017] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.0088 [+0.0179, +0.0191] | +0.0187 | **-0.0099** | [-0.0126, -0.0071] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.0088 [+0.0186, +0.0186] | +0.0186 | **-0.0098** | [-0.0127, -0.0069] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | UNMATCHED (as traced) | +0.0088 | +0.0186 | **-0.0098** | [-0.0127, -0.0069] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.0772 [+0.1010, +0.1043] | +0.1025 | **-0.0254** | [-0.0383, -0.0133] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.0772 [+0.1026, +0.1026] | +0.1026 | **-0.0255** | [-0.0385, -0.0125] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | UNMATCHED (as traced) | +0.0772 | +0.1026 | **-0.0255** | [-0.0385, -0.0125] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.1116 [+0.1605, +0.1629] | +0.1613 | **-0.0498** | [-0.0680, -0.0314] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.1116 [+0.1619, +0.1619] | +0.1619 | **-0.0504** | [-0.0690, -0.0322] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | UNMATCHED (as traced) | +0.1116 | +0.1619 | **-0.0504** | [-0.0690, -0.0322] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.0447 [+0.0478, +0.0493] | +0.0486 | **-0.0039** | [-0.0077, +0.0003] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.0447 [+0.0473, +0.0473] | +0.0473 | **-0.0026** | [-0.0066, +0.0011] | **WITHIN FLOOR** (|delta| <= floor 0.0034) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0447 | +0.0473 | **-0.0026** | [-0.0066, +0.0011] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.0271 [+0.0290, +0.0300] | +0.0296 | **-0.0026** | [-0.0062, +0.0009] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.0271 [+0.0292, +0.0292] | +0.0292 | **-0.0022** | [-0.0054, +0.0014] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0271 | +0.0292 | **-0.0022** | [-0.0054, +0.0014] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +1.2800 [+0.8771, +1.0673] | +0.9727 | **+0.3073** | [+0.0063, +0.1723] | **WITHIN FLOOR** (|delta| <= floor 0.7010) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +1.2800 [+0.9968, +0.9968] | +0.9968 | **+0.2832** | [-0.0007, +0.1665] | **WITHIN FLOOR** (|delta| <= floor 0.7010) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +1.2800 | +0.9968 | **+0.2832** | [-0.0007, +0.1665] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.5116 [+0.4165, +0.4280] | +0.4227 | **+0.0889** | [+0.0288, +0.1011] | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.5116 [+0.4291, +0.4291] | +0.4291 | **+0.0824** | [+0.0253, +0.0990] | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.5116 | +0.4291 | **+0.0824** | [+0.0253, +0.0990] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.0052 [-0.0019, +0.0019] | +0.0002 | **+0.0050** | [-0.0008, +0.0109] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.0052 [+0.0013, +0.0013] | +0.0013 | **+0.0039** | [-0.0019, +0.0099] | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0052 | +0.0013 | **+0.0039** | [-0.0019, +0.0099] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +0.0691 [+0.0224, +0.0282] | +0.0255 | **+0.0436** | [+0.0303, +0.0577] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +0.0691 [+0.0243, +0.0243] | +0.0243 | **+0.0447** | [+0.0315, +0.0581] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0691 | +0.0243 | **+0.0447** | [+0.0315, +0.0581] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 796/285/9 (9478 battles, 9475 decoder battles, 21 seeds) | +1.2352 [+1.2042, +1.2299] | +1.2205 | **+0.0147** | [-0.0765, +0.1018] | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 995/356/9 (9542 battles, 9539 decoder battles, 21 seeds) | +1.2352 [+1.2091, +1.2091] | +1.2091 | **+0.0261** | [-0.0624, +0.1148] | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.2352 | +1.2091 | **+0.0261** | [-0.0624, +0.1148] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0917 | [+0.0765, +0.1145] | +0.0864 | [+0.0754, +0.1009] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0946 | [+0.0798, +0.1168] | +0.0877 | [+0.0769, +0.1019] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.0808 | [-0.0901, -0.0727] | -0.1120 | [-0.1217, -0.1031] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.4904 | [+0.4335, +0.5503] | +0.3941 | [+0.3585, +0.4324] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.4884 | [+0.4324, +0.5472] | +0.3933 | [+0.3580, +0.4313] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0454 | [-0.0551, -0.0368] | -0.0743 | [-0.0843, -0.0652] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 4-10` | +0.3318 | [+0.2941, +0.3730] | +0.2526 | [+0.2289, +0.2790] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 11-24` | +0.5402 | [+0.4766, +0.6088] | +0.4714 | [+0.4266, +0.5195] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0742 | [+0.0634, +0.0848] | +0.0256 | [+0.0188, +0.0320] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0109 | [+0.0066, +0.0151] | +0.0067 | [+0.0038, +0.0096] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5097 | [+0.4960, +0.5233] | +0.4820 | [+0.4679, +0.4954] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1-3` | +0.5541 | [+0.5429, +0.5662] | +0.5819 | [+0.5702, +0.5934] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 4-10` | +0.6788 | [+0.6685, +0.6893] | +0.6931 | [+0.6816, +0.7038] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 1-3` | +0.0088 | [+0.0076, +0.0107] | +0.0186 | [+0.0166, +0.0213] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 4-10` | +0.0772 | [+0.0692, +0.0858] | +0.1026 | [+0.0936, +0.1127] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 11-24` | +0.1116 | [+0.0998, +0.1240] | +0.1619 | [+0.1487, +0.1758] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0447 | [+0.0444, +0.0500] | +0.0473 | [+0.0471, +0.0526] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0271 | [+0.0250, +0.0300] | +0.0292 | [+0.0271, +0.0320] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +1.2800 | [+0.4541, +0.5849] | +0.9968 | [+0.3830, +0.4892] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.5116 | [+0.3627, +0.4182] | +0.4291 | [+0.3052, +0.3512] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0052 | [-0.0003, +0.0103] | +0.0013 | [-0.0015, +0.0033] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0691 | [+0.0578, +0.0808] | +0.0243 | [+0.0177, +0.0312] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.2977 | [+1.2367, +1.3613] | +1.2669 | [+1.2114, +1.3271] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | -0.2651 | [-0.3761, -0.1518] | -0.6439 | [-0.7551, -0.5298] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.9822 | [+0.8715, +1.0958] | +0.6558 | [+0.5658, +0.7460] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.4737 | [+0.3103, +0.6410] | +0.5878 | [+0.4183, +0.7526] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.2352 | [+1.1734, +1.3032] | +1.2091 | [+1.1535, +1.2732] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.3393 | [+1.2688, +1.4130] | +1.2688 | [+1.2027, +1.3357] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | -0.3378 | [-0.4613, -0.2082] | -0.6486 | [-0.7764, -0.5264] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_29_ladder_vf025` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0551 [+0.0198, +0.1031] | 0.0618 | -0.0067 | +0.3365 [+0.0325, +0.5232] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0180 [+0.0099, +0.0330] | 0.0337 | -0.0157 | +0.1631 [+0.0225, +0.2518] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0809 [+0.0257, +0.1647] | 0.0711 | +0.0099 | +0.3453 [-0.0835, +0.5395] | ❌ | ✅ | ✅ | ❌ |

Identity, capture-rate reweighted: 800 labels / 739 battles / 6400 rollouts · Brier 0.0859 = REL 0.0015 − RES 0.0360 + UNC 0.1212 + WBV 0.0007 (resid -1.56e-03) · base rate 0.8589 · **resolution is 29.7% of the base-rate cap** · corr(turn,V) -0.2707 vs corr(turn,MC) -0.0895

**control — `ai_v12_15_ladder_ctrl10M_b` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0545 [+0.0412, +0.0680] | 0.0618 | -0.0073 | +0.2608 [+0.2118, +0.3048] | ❌ | ❌ | ❌ | ✅ |
| `bot` | yes | 0.0290 [+0.0182, +0.0454] | 0.0337 | -0.0047 | +0.2015 [+0.1342, +0.2626] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0793 [+0.0608, +0.0976] | 0.0711 | +0.0082 | +0.2682 [+0.1854, +0.3287] | ❌ | ❌ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 721 battles / 6400 rollouts · Brier 0.1046 = REL 0.0026 − RES 0.0353 + UNC 0.1387 + WBV 0.0008 (resid -2.23e-03) · base rate 0.8336 · **resolution is 25.5% of the base-rate cap** · corr(turn,V) -0.1141 vs corr(turn,MC) -0.1582

## 5. THE LEDGER LINE

```
ai_v12_29_ladder_vf025 vs ai_v12_15_ladder_ctrl10M_b at 10M [OFFLINE-GENERATED: 800 games x 9+3 opponents, full capture]: G1 bot Δ +0.0060 [-0.0005, +0.0122] NOT DETECTED · identity bias late Δ -0.0970 [-0.1430, -0.0498] WITHIN FLOOR · turn-contrast Δ -0.2253 [-0.3422, -0.1065] DETECTED · spread ratio t1-3 Δ +0.0053 [-0.0156, +0.0310] WITHIN FLOOR · own-team R2 t1 Δ +0.0490 [+0.0365, +0.0622] DETECTED [QUOTA-MATCHED] · calib slope Δ +0.0308 [-0.0588, +0.1174] WITHIN FLOOR
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_29_ladder_vf025 --control ai_v12_15_ladder_ctrl10M_b --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_29_ladder_vf025 --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_15_ladder_ctrl10M_b --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/hp800b_floor_v6.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_29_ladder_vf025_vs_ctrl10M_b --nice 15 --ledger-line

# arm — ai_v12_29_ladder_vf025  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_15_ladder_ctrl10M_b  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_29_ladder_vf025` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_29_ladder_vf025/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_29_ladder_vf025_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_29_ladder_vf025_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_29_ladder_vf025_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/ai_v12_15_ladder_ctrl10M_b/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_15_ladder_ctrl10M_b_vs_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_15_ladder_ctrl10M_b_vs_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_15_ladder_ctrl10M_b_vs_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_29_ladder_vf025_vs_ctrl10M_b/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads/ai_v12_29_ladder_vf025_vs_ctrl10M_b/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
