# CRITIC READ — `ai_v13_04_flywheel_winprob_b` vs `ai_v13_02_flywheel_winprob`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v6 at 2026-09-18T14:45:17. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

**SELF-PLAY CROSSING** — arm 4,128,768 · control 4,128,768 (first step carrying any `*_pool` scalar; the PROMOTION scalar itself lands one eval→rollout lag earlier, at 4,000,032 / 4,000,032). Descriptive — no floor, no verdict.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v13_04_flywheel_winprob_b` | `step_74000016` | 9600 | 0.5% | 150/150 (100.0%) | no |
| control | `ai_v13_02_flywheel_winprob` | `step_74000016` | 9600 | 0.4% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 800 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 800 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260910, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v13_04_flywheel_winprob_b` → `step_74000016`: PINNED by --step 74000016
- **control** `ai_v13_02_flywheel_winprob` → `step_74000016`: PINNED by --step 74000016

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v13_04_flywheel_winprob_b` | **800/399/8** | 9600 | 7878 / 1672 / 50 | 12 |
| control | `ai_v13_02_flywheel_winprob` | **799/392/7** | 9600 | 7872 / 1685 / 43 | 12 |

**Frames:** the realized per-opponent caps differ — arm 800/399/8, control 799/392/7 (traced W/L/D per opponent, counted on disk); arm cut to 799/392/7

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm side is subsampled to caps **799/392/7** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/floors_v6_max_imported.json` — 74 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0276 | +0.0217 | **+0.0058** | [-0.0007, +0.0122] | **WITHIN FLOOR** (|delta| <= floor 0.0074) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0026 | -0.0907 | **+0.0933** | [+0.0509, +0.1368] | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.1758 | -0.0236 | **+0.1994** | [+0.0775, +0.3180] | **NOT DETECTED** (CI does not clear the floor 0.1082) |
| skill · `bot` | — | +0.1572 | -0.0172 | **+0.1744** | [+0.0888, +0.2631] | **NOT DETECTED** (CI does not clear the floor 0.1004) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0785 | +0.0607 | **+0.0178** | [-0.0007, +0.0366] | **WITHIN FLOOR** (|delta| <= floor 0.0671) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 799/392/7 | +0.0224 | +0.0345 | **-0.0121** | [-0.0228, -0.0014] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.2836 | +1.3504 | **-0.0668** | [-0.1474, +0.0186] | **WITHIN FLOOR** (|delta| <= floor 0.2001) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0593 | +0.0593 | **+0.0000** | [-0.0064, +0.0061] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0024) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0276 | +0.0217 | **+0.0058** | [-0.0007, +0.0122] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0074) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0516 | +0.0634 | **-0.0118** | [-0.0213, -0.0021] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0141) |
| reliability | `all` | `capture-rate (gauge)` | +0.0001 | +0.0051 | **-0.0050** | [-0.0063, -0.0038] | 400 | **DETECTED** (CI clears the floor +0.0026) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0110 | +0.0234 | **-0.0124** | [-0.0161, -0.0091] | 400 | **DETECTED** (CI clears the floor +0.0053) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0156 | +0.0040 | **+0.0116** | [+0.0053, +0.0182] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0232) |
| ece | `all` | `capture-rate (gauge)` | +0.0101 | +0.0688 | **-0.0587** | [-0.0669, -0.0455] | 400 | **DETECTED** (CI clears the floor +0.0245) |
| ece | `bot` | `capture-rate (gauge)` | +0.0870 | +0.1338 | **-0.0467** | [-0.0584, -0.0356] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0449) |
| ece | `pool` | `capture-rate (gauge)` | +0.1125 | +0.0528 | **+0.0597** | [+0.0300, +0.0850] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1097) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3021 | +0.2980 | **+0.0041** | [-0.0294, +0.0352] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0454) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.1572 | -0.0172 | **+0.1744** | [+0.0888, +0.2631] | 400 | **NOT DETECTED** (CI does not clear the floor 0.1004) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.1464 | +0.2412 | **-0.0948** | [-0.1519, -0.0414] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1381) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | -0.0200 | -0.0813 | **+0.0613** | [+0.0364, +0.0859] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0934) |
| bias V - p_hat | `ALL` | `pop` | -0.0624 | -0.1101 | **+0.0477** | [+0.0258, +0.0690] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0744) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0624 | -0.1101 | **+0.0477** | [+0.0258, +0.0690] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0744) |
| bias V - p_hat | `early (turn<=10)` | `raw` | -0.0732 | -0.0939 | **+0.0208** | [-0.0188, +0.0609] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1037) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.1092 | -0.1239 | **+0.0147** | [-0.0228, +0.0517] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0793) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.1092 | -0.1239 | **+0.0147** | [-0.0228, +0.0517] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0793) |
| bias V - p_hat | `mid (11-24)` | `raw` | -0.0461 | -0.0908 | **+0.0446** | [+0.0101, +0.0794] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0748) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0763 | -0.1156 | **+0.0393** | [+0.0068, +0.0729] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0556) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0763 | -0.1156 | **+0.0393** | [+0.0068, +0.0729] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0556) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0571 | -0.0573 | **+0.1144** | [+0.0646, +0.1670] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1252) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0026 | -0.0907 | **+0.0933** | [+0.0509, +0.1368] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0026 | -0.0907 | **+0.0933** | [+0.0509, +0.1368] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1201) |
| bias V - p_hat | `bot` | `raw` | -0.0536 | -0.1097 | **+0.0561** | [+0.0273, +0.0852] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0666) |
| bias V - p_hat | `bot` | `pop` | -0.0862 | -0.1316 | **+0.0453** | [+0.0208, +0.0698] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0567) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0862 | -0.1316 | **+0.0453** | [+0.0208, +0.0698] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0567) |
| bias V - p_hat | `pool` | `raw` | +0.0273 | -0.0329 | **+0.0602** | [+0.0151, +0.1064] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1390) |
| bias V - p_hat | `pool` | `pop` | -0.0273 | -0.0759 | **+0.0485** | [+0.0049, +0.0923] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| bias V - p_hat ⭐ | `pool` | `ipw` | -0.0273 | -0.0759 | **+0.0485** | [+0.0049, +0.0923] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1196) |
| Murphy resolution | `ALL` | `raw` | +0.0400 | +0.0366 | **+0.0034** | [-0.0091, +0.0160] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0070) |
| Murphy resolution | `ALL` | `pop` | +0.0529 | +0.0453 | **+0.0076** | [-0.0078, +0.0226] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0529 | +0.0453 | **+0.0076** | [-0.0078, +0.0226] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.2162 | +0.1660 | **+0.0502** | [-0.0257, +0.1308] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0538) |
| Murphy skill_score | `ALL` | `pop` | +0.3014 | +0.2014 | **+0.1000** | [+0.0044, +0.1980] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0165) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.3014 | +0.2014 | **+0.1000** | [+0.0044, +0.1980] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0165) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.2262 | +0.2226 | **+0.0036** | [-0.0613, +0.0701] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0188) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.3322 | +0.2945 | **+0.0377** | [-0.0371, +0.1131] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.3322 | +0.2945 | **+0.0377** | [-0.0371, +0.1131] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0020 | +0.0099 | **-0.0080** | [-0.0127, -0.0036] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0119) |
| Murphy reliability | `ALL` | `pop` | +0.0048 | +0.0146 | **-0.0098** | [-0.0154, -0.0044] | 4000 | **DETECTED** (CI clears the floor +0.0029) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0048 | +0.0146 | **-0.0098** | [-0.0154, -0.0044] | 4000 | **DETECTED** (CI clears the floor +0.0029) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.1758 | -0.0236 | **+0.1994** | [+0.0775, +0.3180] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.1082) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | +0.1149 | -0.0304 | **+0.1453** | [+0.0395, +0.2474] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0984) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.1149 | -0.0304 | **+0.1453** | [+0.0395, +0.2474] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0984) |

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

**arm — `ai_v13_04_flywheel_winprob_b` @ `step_74000016`**: 312348 states / 9550 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 50 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v13_02_flywheel_winprob` @ `step_74000016`**: 312880 states / 9557 battles / 12 opponents / 602 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 43 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0785 | +0.0607 | **+0.0178** | [-0.0007, +0.0366] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0671) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 799/392/7 | +0.0812 | +0.0639 | **+0.0173** | [-0.0009, +0.0349] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1579 | -0.1631 | **+0.0052** | [-0.0075, +0.0182] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0147) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5413 | +0.5466 | **-0.0053** | [-0.0532, +0.0453] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2399) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 799/392/7 | +0.5408 | +0.5461 | **-0.0053** | [-0.0572, +0.0457] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0786 | -0.0787 | **+0.0001** | [-0.0128, +0.0140] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0322) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 4-10` | as traced | +0.3402 | +0.3301 | **+0.0100** | [-0.0225, +0.0436] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2833) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 11-24` | as traced | +0.6538 | +0.6608 | **-0.0070** | [-0.0656, +0.0540] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2669) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 799/392/7 | +0.0224 | +0.0345 | **-0.0121** | [-0.0228, -0.0014] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 799/392/7 | +0.0062 | +0.0039 | **+0.0023** | [-0.0015, +0.0062] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 799/392/7 | +0.4917 | +0.4948 | **-0.0031** | [-0.0226, +0.0155] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0227) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1-3` | MATCHED 799/392/7 | +0.5698 | +0.5605 | **+0.0093** | [-0.0080, +0.0270] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 4-10` | MATCHED 799/392/7 | +0.7445 | +0.7431 | **+0.0014** | [-0.0131, +0.0150] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 1-3` | MATCHED 799/392/7 | +0.0122 | +0.0095 | **+0.0027** | [-0.0001, +0.0054] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 4-10` | MATCHED 799/392/7 | +0.1363 | +0.1316 | **+0.0047** | [-0.0084, +0.0175] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 11-24` | MATCHED 799/392/7 | +0.2748 | +0.2492 | **+0.0256** | [+0.0026, +0.0483] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 799/392/7 | +0.0690 | +0.0729 | **-0.0039** | [-0.0079, +0.0011] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 799/392/7 | +0.0439 | +0.0444 | **-0.0005** | [-0.0045, +0.0034] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 799/392/7 | +1.3547 | +2.4869 | **-1.1323** | [-0.1773, +0.0583] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 799/392/7 | +0.6735 | +0.7328 | **-0.0593** | [-0.0666, +0.0252] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 799/392/7 | +0.0033 | -0.0011 | **+0.0044** | [+0.0003, +0.0086] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 799/392/7 | +0.0192 | +0.0356 | **-0.0164** | [-0.0276, -0.0050] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.2836 | +1.3504 | **-0.0668** | [-0.1474, +0.0186] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2001) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | +0.3761 | +0.5804 | **-0.2043** | [-0.3037, -0.1084] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.0247) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.6691 | +0.7333 | **-0.0642** | [-0.1883, +0.0572] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1915) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +1.0306 | +1.0153 | **+0.0153** | [-0.0913, +0.1229] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.4031) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 799/392/7 | +1.2538 | +1.3077 | **-0.0539** | [-0.1342, +0.0286] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.2786 | +1.4103 | **-0.1316** | [-0.2167, -0.0448] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2481) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | +0.3801 | +0.5363 | **-0.1562** | [-0.2583, -0.0564] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.0981) |

### HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY — `V` against its opponent-decodable part

| side | window | ratio of `V` | opponent-decodable part | `V` / decodable |
|---|---|---|---|---|
| arm `ai_v13_04_flywheel_winprob_b` | `t1_3` | +0.0785 | +0.0119 | 6.594 |
| arm `ai_v13_04_flywheel_winprob_b` | `t4_10` | +0.3402 | +0.1361 | 2.499 |
| arm `ai_v13_04_flywheel_winprob_b` | `t11_24` | +0.6538 | +0.2745 | 2.381 |
| control `ai_v13_02_flywheel_winprob` | `t1_3` | +0.0607 | +0.0095 | 6.388 |
| control `ai_v13_02_flywheel_winprob` | `t4_10` | +0.3301 | +0.1316 | 2.509 |
| control `ai_v13_02_flywheel_winprob` | `t11_24` | +0.6608 | +0.2492 | 2.652 |

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
| arm · team | 600 | 602 | 9544 | 19088 | 13.0 | 26.0 | 4–93 |
| arm · stratum | 5 | 602 | 9544 | 19088 | 1902.0 | 3804.0 | 1893–1930 |
| arm · between-team spread | 600 | 602 | 9544 | — | 13.0 | — | — |
| control · team | 600 | 602 | 9551 | 19102 | 13.0 | 26.0 | 4–94 |
| control · stratum | 5 | 602 | 9551 | 19102 | 1910.0 | 3820.0 | 1898–1931 |
| control · between-team spread | 600 | 602 | 9551 | — | 13.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 799/392/7 | +0.0690 | +0.0729 | **-0.0039** | [-0.0079, +0.0011] | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 799/392/7 | +0.0439 | +0.0444 | **-0.0005** | [-0.0045, +0.0034] | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 799/392/7 | +1.3547 | +2.4869 | **-1.1323** | [-0.1773, +0.0583] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 799/392/7 | +0.0224 | +0.0345 | **-0.0121** | [-0.0228, -0.0014] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 799/392/7 | +0.0033 | -0.0011 | **+0.0044** | [+0.0003, +0.0086] | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 799/392/7 | +0.0192 | +0.0356 | **-0.0164** | [-0.0276, -0.0050] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 19100 | 9550 | 0.7559 | **0.2187** | **1.5868** | [0.123, 0.992] | 0.0000 |
| arm · t1_3 | `t1_3` | 19100 | 9550 | 0.6873 | **0.1151** | **0.5774** | [0.438, 0.887] | 0.0000 |
| arm · common support | `common support` | 17665 | 9404 | 0.7620 | **0.1879** | **1.2238** | [0.246, 0.982] | 0.0000 |
| control · all | `all` | 19114 | 9557 | 0.7251 | **0.2227** | **1.5739** | [0.087, 0.987] | 0.0008 |
| control · t1_3 | `t1_3` | 19114 | 9557 | 0.6766 | **0.1100** | **0.5243** | [0.425, 0.856] | 0.0000 |
| control · common support | `common support` | 18021 | 9428 | 0.7411 | **0.1864** | **1.1738** | [0.252, 0.978] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.1227, 0.9874]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.2836 | +1.3504 | **-0.0668** | [-0.1474, +0.0186] | 0.2001 | **WITHIN FLOOR** (|delta| <= floor 0.2001) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | +0.3761 | +0.5804 | **-0.2043** | [-0.3037, -0.1084] | 1.0247 | **WITHIN FLOOR** (|delta| <= floor 1.0247) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.6691 | +0.7333 | **-0.0642** | [-0.1883, +0.0572] | 0.1915 | **WITHIN FLOOR** (|delta| <= floor 0.1915) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +1.0306 | +1.0153 | **+0.0153** | [-0.0913, +0.1229] | 0.4031 | **WITHIN FLOOR** (|delta| <= floor 0.4031) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 799/392/7 | +1.2538 | +1.3077 | **-0.0539** | [-0.1342, +0.0286] | 0.1729 | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.2786 | +1.4103 | **-0.1316** | [-0.2167, -0.0448] | 0.2481 | **WITHIN FLOOR** (|delta| <= floor 0.2481) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +0.3801 | +0.5363 | **-0.1562** | [-0.2583, -0.0564] | 1.0981 | **WITHIN FLOOR** (|delta| <= floor 1.0981) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.0812 [+0.0806, +0.0818] | +0.0639 | **+0.0173** | [-0.0009, +0.0349] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.0811 [+0.0811, +0.0811] | +0.0639 | **+0.0172** | [-0.0006, +0.0353] | **WITHIN FLOOR** (|delta| <= floor 0.0681) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.0811 | +0.0639 | **+0.0172** | [-0.0006, +0.0353] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.5408 [+0.5399, +0.5418] | +0.5461 | **-0.0053** | [-0.0572, +0.0457] | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.5407 [+0.5407, +0.5407] | +0.5461 | **-0.0054** | [-0.0529, +0.0450] | **WITHIN FLOOR** (|delta| <= floor 0.2383) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.5407 | +0.5461 | **-0.0054** | [-0.0529, +0.0450] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.0224 [+0.0212, +0.0234] | +0.0345 | **-0.0121** | [-0.0228, -0.0014] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| `cond.own_team_r2.t1` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.0221 [+0.0221, +0.0221] | +0.0345 | **-0.0124** | [-0.0227, -0.0017] | **WITHIN FLOOR** (|delta| <= floor 0.0359) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0221 | +0.0345 | **-0.0124** | [-0.0227, -0.0017] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.0062 [+0.0057, +0.0068] | +0.0039 | **+0.0023** | [-0.0015, +0.0062] | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| `cond.own_team_r2.all` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.0067 [+0.0067, +0.0067] | +0.0039 | **+0.0028** | [-0.0013, +0.0070] | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0067 | +0.0039 | **+0.0028** | [-0.0013, +0.0070] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.4917 [+0.4873, +0.4938] | +0.4948 | **-0.0031** | [-0.0226, +0.0155] | **WITHIN FLOOR** (|delta| <= floor 0.0227) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.4927 [+0.4927, +0.4927] | +0.4948 | **-0.0021** | [-0.0209, +0.0171] | **WITHIN FLOOR** (|delta| <= floor 0.0227) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.4927 | +0.4948 | **-0.0021** | [-0.0209, +0.0171] | *no label — not a reading* |
| `cond.opp_class_auc.t1_3` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.5698 [+0.5686, +0.5709] | +0.5605 | **+0.0093** | [-0.0080, +0.0270] | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| `cond.opp_class_auc.t1_3` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.5688 [+0.5688, +0.5688] | +0.5605 | **+0.0083** | [-0.0081, +0.0258] | **WITHIN FLOOR** (|delta| <= floor 0.0153) |
| `cond.opp_class_auc.t1_3` | UNMATCHED (as traced) | +0.5688 | +0.5605 | **+0.0083** | [-0.0081, +0.0258] | *no label — not a reading* |
| `cond.opp_class_auc.t4_10` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.7445 [+0.7435, +0.7451] | +0.7431 | **+0.0014** | [-0.0131, +0.0150] | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| `cond.opp_class_auc.t4_10` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.7467 [+0.7467, +0.7467] | +0.7431 | **+0.0036** | [-0.0099, +0.0181] | **WITHIN FLOOR** (|delta| <= floor 0.0245) |
| `cond.opp_class_auc.t4_10` | UNMATCHED (as traced) | +0.7467 | +0.7431 | **+0.0036** | [-0.0099, +0.0181] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.0122 [+0.0119, +0.0125] | +0.0095 | **+0.0027** | [-0.0001, +0.0054] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.0119 [+0.0119, +0.0119] | +0.0095 | **+0.0024** | [-0.0001, +0.0050] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | UNMATCHED (as traced) | +0.0119 | +0.0095 | **+0.0024** | [-0.0001, +0.0050] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.1363 [+0.1357, +0.1370] | +0.1316 | **+0.0047** | [-0.0084, +0.0175] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.1361 [+0.1361, +0.1361] | +0.1316 | **+0.0045** | [-0.0076, +0.0175] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | UNMATCHED (as traced) | +0.1361 | +0.1316 | **+0.0045** | [-0.0076, +0.0175] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.2748 [+0.2743, +0.2751] | +0.2492 | **+0.0256** | [+0.0026, +0.0483] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.2745 [+0.2745, +0.2745] | +0.2492 | **+0.0254** | [+0.0039, +0.0476] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | UNMATCHED (as traced) | +0.2745 | +0.2492 | **+0.0254** | [+0.0039, +0.0476] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.0690 [+0.0687, +0.0697] | +0.0729 | **-0.0039** | [-0.0079, +0.0011] | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| `cond.within_team_resolution.all` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.0707 [+0.0707, +0.0707] | +0.0729 | **-0.0022** | [-0.0068, +0.0022] | **WITHIN FLOOR** (|delta| <= floor 0.0044) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0707 | +0.0729 | **-0.0022** | [-0.0068, +0.0022] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.0439 [+0.0436, +0.0443] | +0.0444 | **-0.0005** | [-0.0045, +0.0034] | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.0446 [+0.0446, +0.0446] | +0.0444 | **+0.0002** | [-0.0039, +0.0041] | **WITHIN FLOOR** (|delta| <= floor 0.0021) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0446 | +0.0444 | **+0.0002** | [-0.0039, +0.0041] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +1.3547 [+1.3396, +1.3758] | +2.4869 | **-1.1323** | [-0.1773, +0.0583] | **NOT DETECTED** (CI covers zero) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +1.3582 [+1.3582, +1.3582] | +2.4869 | **-1.1287** | [-0.1800, +0.0575] | **NOT DETECTED** (CI covers zero) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +1.3582 | +2.4869 | **-1.1287** | [-0.1800, +0.0575] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.6735 [+0.6720, +0.6756] | +0.7328 | **-0.0593** | [-0.0666, +0.0252] | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.6747 [+0.6747, +0.6747] | +0.7328 | **-0.0581** | [-0.0639, +0.0266] | **WITHIN FLOOR** (|delta| <= floor 0.2824) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.6747 | +0.7328 | **-0.0581** | [-0.0639, +0.0266] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.0033 [+0.0026, +0.0040] | -0.0011 | **+0.0044** | [+0.0003, +0.0086] | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| `cond.own_team_r2.late` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.0031 [+0.0031, +0.0031] | -0.0011 | **+0.0042** | [+0.0005, +0.0080] | **WITHIN FLOOR** (|delta| <= floor 0.0047) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0031 | -0.0011 | **+0.0042** | [+0.0005, +0.0080] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +0.0192 [+0.0174, +0.0207] | +0.0356 | **-0.0164** | [-0.0276, -0.0050] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +0.0190 [+0.0190, +0.0190] | +0.0356 | **-0.0166** | [-0.0273, -0.0052] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0190 | +0.0356 | **-0.0166** | [-0.0273, -0.0052] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 799/392/7 (9542 battles, 9536 decoder battles, 21 seeds) | +1.2538 [+1.2439, +1.2584] | +1.3077 | **-0.0539** | [-0.1342, +0.0286] | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 899/441/7 (9550 battles, 9544 decoder battles, 21 seeds) | +1.2526 [+1.2526, +1.2526] | +1.3077 | **-0.0551** | [-0.1377, +0.0319] | **WITHIN FLOOR** (|delta| <= floor 0.1729) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.2526 | +1.3077 | **-0.0551** | [-0.1377, +0.0319] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0785 | [+0.0679, +0.0947] | +0.0607 | [+0.0506, +0.0764] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0811 | [+0.0708, +0.0970] | +0.0639 | [+0.0542, +0.0790] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1579 | [-0.1665, -0.1487] | -0.1631 | [-0.1722, -0.1533] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5413 | [+0.5069, +0.5769] | +0.5466 | [+0.5132, +0.5831] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.5407 | [+0.5065, +0.5761] | +0.5461 | [+0.5129, +0.5824] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0786 | [-0.0881, -0.0691] | -0.0787 | [-0.0883, -0.0692] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 4-10` | +0.3402 | [+0.3177, +0.3660] | +0.3301 | [+0.3069, +0.3540] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 11-24` | +0.6538 | [+0.6128, +0.6986] | +0.6608 | [+0.6203, +0.7040] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0221 | [+0.0162, +0.0280] | +0.0345 | [+0.0253, +0.0433] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0067 | [+0.0032, +0.0098] | +0.0039 | [+0.0013, +0.0063] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.4927 | [+0.4795, +0.5061] | +0.4948 | [+0.4816, +0.5088] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1-3` | +0.5688 | [+0.5570, +0.5808] | +0.5605 | [+0.5481, +0.5726] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 4-10` | +0.7467 | [+0.7368, +0.7566] | +0.7431 | [+0.7326, +0.7528] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 1-3` | +0.0119 | [+0.0105, +0.0142] | +0.0095 | [+0.0081, +0.0117] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 4-10` | +0.1361 | [+0.1274, +0.1463] | +0.1316 | [+0.1229, +0.1406] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 11-24` | +0.2745 | [+0.2590, +0.2916] | +0.2492 | [+0.2351, +0.2642] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0707 | [+0.0717, +0.0780] | +0.0729 | [+0.0739, +0.0803] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0446 | [+0.0421, +0.0478] | +0.0444 | [+0.0420, +0.0477] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +1.3582 | [+0.6109, +0.7530] | +2.4869 | [+0.6538, +0.8353] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.6747 | [+0.4932, +0.5545] | +0.7328 | [+0.5090, +0.5765] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0031 | [-0.0007, +0.0064] | -0.0011 | [-0.0026, -0.0003] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0190 | [+0.0126, +0.0259] | +0.0356 | [+0.0266, +0.0449] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.2836 | [+1.2300, +1.3417] | +1.3504 | [+1.2876, +1.4111] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | +0.3761 | [+0.3077, +0.4387] | +0.5804 | [+0.5151, +0.6499] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.6691 | [+0.5823, +0.7508] | +0.7333 | [+0.6470, +0.8250] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +1.0306 | [+0.9570, +1.1088] | +1.0153 | [+0.9382, +1.0921] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.2526 | [+1.1961, +1.3117] | +1.3077 | [+1.2472, +1.3691] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.2786 | [+1.2208, +1.3400] | +1.4103 | [+1.3491, +1.4725] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | +0.3801 | [+0.3088, +0.4457] | +0.5363 | [+0.4703, +0.6042] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v13_04_flywheel_winprob_b` @ step_74000016** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 40}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0546 [+0.0370, +0.0797] | 0.0618 | -0.0073 | +0.2358 [+0.0559, +0.3698] | ❌ | ❌ | ❌ | ✅ |
| `bot` | yes | 0.0234 [+0.0146, +0.0373] | 0.0337 | -0.0103 | +0.1234 [-0.0855, +0.2830] | ❌ | ❌ | ❌ | ❌ |
| `pool` | yes | 0.0472 [+0.0348, +0.0744] | 0.0711 | -0.0239 | +0.1441 [-0.1111, +0.2917] | ❌ | ✅ | ✅ | ❌ |

Identity, capture-rate reweighted: 800 labels / 722 battles / 6400 rollouts · Brier 0.1113 = REL 0.0048 − RES 0.0529 + UNC 0.1593 + WBV 0.0008 (resid -7.51e-04) · base rate 0.8011 · **resolution is 33.2% of the base-rate cap** · corr(turn,V) -0.0796 vs corr(turn,MC) -0.2553

**control — `ai_v13_02_flywheel_winprob` @ step_74000016** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 40}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0590 [+0.0395, +0.0801] | 0.0618 | -0.0028 | +0.2820 [+0.1502, +0.3888] | ❌ | ❌ | ❌ | ✅ |
| `bot` | yes | 0.0175 [+0.0105, +0.0297] | 0.0337 | -0.0161 | -0.1964 [-0.6414, +0.1534] | ❌ | ❌ | ❌ | ❌ |
| `pool` | yes | 0.0646 [+0.0417, +0.0967] | 0.0711 | -0.0065 | +0.2660 [+0.1014, +0.3685] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 741 battles / 6400 rollouts · Brier 0.1229 = REL 0.0146 − RES 0.0453 + UNC 0.1539 + WBV 0.0008 (resid -1.12e-03) · base rate 0.8099 · **resolution is 29.4% of the base-rate cap** · corr(turn,V) -0.2097 vs corr(turn,MC) -0.1861

## 5. THE LEDGER LINE

```
ai_v13_04_flywheel_winprob_b vs ai_v13_02_flywheel_winprob at 74M [OFFLINE-GENERATED: 800 games x 9+3 opponents, full capture]: G1 bot Δ +0.0058 [-0.0007, +0.0122] WITHIN FLOOR · identity bias late Δ +0.0933 [+0.0509, +0.1368] WITHIN FLOOR · turn-contrast Δ +0.1994 [+0.0775, +0.3180] NOT DETECTED · spread ratio t1-3 Δ +0.0178 [-0.0007, +0.0366] WITHIN FLOOR · own-team R2 t1 Δ -0.0121 [-0.0228, -0.0014] WITHIN FLOOR [QUOTA-MATCHED] · calib slope Δ -0.0668 [-0.1474, +0.0186] WITHIN FLOOR
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v13_04_flywheel_winprob_b --control ai_v13_02_flywheel_winprob --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/traces/draw1/eval_traces/step_74000016 --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/traces/draw1/eval_traces/step_74000016 --step 74000016 --v-column win_probs --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/floors_v6_max_imported.json --nice 15 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/read_draw1_Wb_vs_W

# arm — ai_v13_04_flywheel_winprob_b
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/traces/draw1 --step 74000016 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/read_draw1_Wb_vs_W/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v13_04_flywheel_winprob_b --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/read_draw1_Wb_vs_W/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/read_draw1_Wb_vs_W/gate/critic_gate.md
# control — ai_v13_02_flywheel_winprob
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/traces/draw1 --step 74000016 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/ai_v13_02_flywheel_winprob__offline/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v13_02_flywheel_winprob --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/ai_v13_02_flywheel_winprob__offline/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/ai_v13_02_flywheel_winprob__offline/gate/critic_gate.md
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v13_04_flywheel_winprob_b` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/traces/draw1/eval_traces/step_74000016` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/read_draw1_Wb_vs_W/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/read_draw1_Wb_vs_W/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/read_draw1_Wb_vs_W/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v13_02_flywheel_winprob` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/traces/draw1/eval_traces/step_74000016` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/ai_v13_02_flywheel_winprob__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/ai_v13_02_flywheel_winprob__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/ai_v13_02_flywheel_winprob__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/read_draw1_Wb_vs_W/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/critic/read_draw1_Wb_vs_W/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
