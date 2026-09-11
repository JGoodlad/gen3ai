# CRITIC READ — `ai_v12_21_ladder_lambda09_b` vs `ai_v12_16_ladder_ctrl10M_c`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v6 at 2026-09-11T04:41:01. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

**SELF-PLAY CROSSING** — arm 4,128,768 · control 4,128,768 (first step carrying any `*_pool` scalar; the PROMOTION scalar itself lands one eval→rollout lag earlier, at 4,000,032 / 4,000,032). Descriptive — no floor, no verdict.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_21_ladder_lambda09_b` | `step_10000032` | 4800 | 0.4% | 150/150 (100.0%) | no |
| control | `ai_v12_16_ladder_ctrl10M_c` | `step_10000032` | 4800 | 0.5% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 400 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260909, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_21_ladder_lambda09_b` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_16_ladder_ctrl10M_c` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_21_ladder_lambda09_b` | **400/180/3** | 4800 | 4038 / 743 / 19 | 12 |
| control | `ai_v12_16_ladder_ctrl10M_c` | **398/177/5** | 4800 | 4068 / 706 / 26 | 12 |

**Frames:** the realized per-opponent caps differ — arm 400/180/3, control 398/177/5 (traced W/L/D per opponent, counted on disk); arm cut to 398/177/3

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm side is subsampled to caps **398/177/3** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json` — 60 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0226 | +0.0178 | **+0.0048** | [-0.0010, +0.0098] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0602 | +0.0395 | **-0.0996** | [-0.1441, -0.0549] | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.0232 | +0.0052 | **+0.0181** | [-0.1433, +0.1706] | **WITHIN FLOOR** (|delta| <= floor 0.0440) |
| skill · `bot` | — | +0.1584 | +0.1479 | **+0.0105** | [-0.0568, +0.0685] | **WITHIN FLOOR** (|delta| <= floor 0.0327) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0603 | +0.1283 | **-0.0680** | [-0.0958, -0.0403] | **NOT DETECTED** (CI does not clear the floor 0.0664) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 398/177/3 | +0.0483 | +0.0216 | **+0.0268** | [+0.0107, +0.0429] | **NOT DETECTED** (CI does not clear the floor 0.0164) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.2257 | +1.1723 | **+0.0534** | [-0.0516, +0.1663] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0443 | +0.0384 | **+0.0058** | [-0.0013, +0.0128] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0226 | +0.0178 | **+0.0048** | [-0.0010, +0.0098] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0560 | +0.0562 | **-0.0002** | [-0.0111, +0.0113] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| reliability | `all` | `capture-rate (gauge)` | +0.0002 | +0.0001 | **+0.0001** | [-0.0002, +0.0006] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0037) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0048 | +0.0039 | **+0.0009** | [-0.0020, +0.0042] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0032) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0082 | +0.0101 | **-0.0019** | [-0.0104, +0.0070] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| ece | `all` | `capture-rate (gauge)` | +0.0098 | +0.0063 | **+0.0035** | [-0.0039, +0.0140] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0373) |
| ece | `bot` | `capture-rate (gauge)` | +0.0564 | +0.0432 | **+0.0132** | [-0.0071, +0.0311] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0286) |
| ece | `pool` | `capture-rate (gauge)` | +0.0801 | +0.0951 | **-0.0150** | [-0.0557, +0.0281] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1171) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2559 | +0.2542 | **+0.0017** | [-0.0353, +0.0387] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0307) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.1584 | +0.1479 | **+0.0105** | [-0.0568, +0.0685] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0327) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2058 | +0.2056 | **+0.0002** | [-0.0550, +0.0590] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1324) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | -0.0223 | +0.0554 | **-0.0778** | [-0.1077, -0.0478] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1029) |
| bias V - p_hat | `ALL` | `pop` | -0.0652 | -0.0039 | **-0.0614** | [-0.0831, -0.0394] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0735) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0652 | -0.0039 | **-0.0614** | [-0.0831, -0.0394] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0735) |
| bias V - p_hat | `early (turn<=10)` | `raw` | -0.0296 | +0.0115 | **-0.0410** | [-0.0828, +0.0004] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1085) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0711 | -0.0461 | **-0.0249** | [-0.0572, +0.0071] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0812) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0711 | -0.0461 | **-0.0249** | [-0.0572, +0.0071] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0812) |
| bias V - p_hat | `mid (11-24)` | `raw` | -0.0250 | +0.0552 | **-0.0802** | [-0.1222, -0.0389] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0707) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0636 | +0.0039 | **-0.0675** | [-0.1005, -0.0363] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0501) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0636 | +0.0039 | **-0.0675** | [-0.1005, -0.0363] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0501) |
| bias V - p_hat | `late (turn>=25)` | `raw` | -0.0100 | +0.1067 | **-0.1168** | [-0.1754, -0.0593] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1434) |
| bias V - p_hat | `late (turn>=25)` | `pop` | -0.0602 | +0.0395 | **-0.0996** | [-0.1441, -0.0549] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0602 | +0.0395 | **-0.0996** | [-0.1441, -0.0549] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| bias V - p_hat | `bot` | `raw` | -0.0519 | +0.0042 | **-0.0561** | [-0.0902, -0.0220] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0639) |
| bias V - p_hat | `bot` | `pop` | -0.0813 | -0.0382 | **-0.0431** | [-0.0666, -0.0190] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0813 | -0.0382 | **-0.0431** | [-0.0666, -0.0190] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| bias V - p_hat | `pool` | `raw` | +0.0244 | +0.1500 | **-0.1256** | [-0.1775, -0.0761] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1594) |
| bias V - p_hat | `pool` | `pop` | -0.0202 | +0.0777 | **-0.0979** | [-0.1445, -0.0522] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1256) |
| bias V - p_hat ⭐ | `pool` | `ipw` | -0.0202 | +0.0777 | **-0.0979** | [-0.1445, -0.0522] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1256) |
| Murphy resolution | `ALL` | `raw` | +0.0288 | +0.0261 | **+0.0027** | [-0.0093, +0.0146] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0029) |
| Murphy resolution | `ALL` | `pop` | +0.0350 | +0.0332 | **+0.0018** | [-0.0144, +0.0173] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0042) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0350 | +0.0332 | **+0.0018** | [-0.0144, +0.0173] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0042) |
| Murphy skill_score | `ALL` | `raw` | +0.1811 | +0.1223 | **+0.0589** | [-0.0159, +0.1334] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0596) |
| Murphy skill_score | `ALL` | `pop` | +0.2400 | +0.2353 | **+0.0047** | [-0.1033, +0.1101] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0299) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2400 | +0.2353 | **+0.0047** | [-0.1033, +0.1101] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0299) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1927 | +0.1477 | **+0.0450** | [-0.0204, +0.1108] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2825 | +0.2339 | **+0.0486** | [-0.0446, +0.1408] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2825 | +0.2339 | **+0.0486** | [-0.0446, +0.1408] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0029 | +0.0058 | **-0.0030** | [-0.0082, +0.0013] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0111) |
| Murphy reliability | `ALL` | `pop` | +0.0057 | +0.0005 | **+0.0052** | [+0.0016, +0.0088] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0031) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0057 | +0.0005 | **+0.0052** | [+0.0016, +0.0088] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0031) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.0232 | +0.0052 | **+0.0181** | [-0.1433, +0.1706] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0440) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0335 | -0.0014 | **-0.0322** | [-0.1507, +0.0803] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0507) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0335 | -0.0014 | **-0.0322** | [-0.1507, +0.0803] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0507) |

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

**arm — `ai_v12_21_ladder_lambda09_b` @ `step_10000032`**: 150912 states / 4781 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 19 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_16_ladder_ctrl10M_c` @ `step_10000032`**: 148990 states / 4774 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 26 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0603 | +0.1283 | **-0.0680** | [-0.0958, -0.0403] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0664) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 398/177/3 | +0.0642 | +0.1305 | **-0.0663** | [-0.0927, -0.0386] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0682) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1232 | -0.1073 | **-0.0159** | [-0.0339, +0.0027] | 2000 | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5045 | +0.5066 | **-0.0021** | [-0.0925, +0.0826] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2403) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 398/177/3 | +0.5029 | +0.5046 | **-0.0017** | [-0.0877, +0.0847] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0650 | -0.0607 | **-0.0043** | [-0.0234, +0.0146] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0332) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 4-10` | as traced | +0.2451 | +0.3875 | **-0.1424** | [-0.2060, -0.0813] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 11-24` | as traced | +0.5169 | +0.6023 | **-0.0853** | [-0.1899, +0.0113] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 398/177/3 | +0.0483 | +0.0216 | **+0.0268** | [+0.0107, +0.0429] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0164) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 398/177/3 | +0.0078 | +0.0048 | **+0.0030** | [-0.0034, +0.0097] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 398/177/3 | +0.5035 | +0.5080 | **-0.0045** | [-0.0316, +0.0221] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0118) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1-3` | MATCHED 398/177/3 | +0.5677 | +0.5924 | **-0.0247** | [-0.0472, -0.0013] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 4-10` | MATCHED 398/177/3 | +0.6991 | +0.7041 | **-0.0051** | [-0.0272, +0.0157] | 2000 | **NOT DETECTED** (CI covers zero) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 1-3` | MATCHED 398/177/3 | +0.0146 | +0.0239 | **-0.0093** | [-0.0141, -0.0039] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 4-10` | MATCHED 398/177/3 | +0.0932 | +0.1154 | **-0.0221** | [-0.0402, -0.0042] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 11-24` | MATCHED 398/177/3 | +0.1691 | +0.1774 | **-0.0083** | [-0.0357, +0.0197] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 398/177/3 | +0.0636 | +0.0630 | **+0.0006** | [-0.0060, +0.0061] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 398/177/3 | +0.0345 | +0.0337 | **+0.0008** | [-0.0044, +0.0062] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0028) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 398/177/3 | +0.6277 | +1.1807 | **-0.5530** | [-0.1598, -0.0155] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 398/177/3 | +0.3132 | +0.3943 | **-0.0811** | [-0.0891, -0.0282] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 398/177/3 | +0.0155 | -0.0049 | **+0.0204** | [+0.0087, +0.0323] | 2000 | **DETECTED** (CI clears the floor +0.0057) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 398/177/3 | +0.0324 | +0.0264 | **+0.0060** | [-0.0136, +0.0247] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.2257 | +1.1723 | **+0.0534** | [-0.0516, +0.1663] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | +0.1960 | +0.0662 | **+0.1298** | [-0.0385, +0.3050] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +1.1265 | +1.1093 | **+0.0172** | [-0.2100, +0.2358] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.1581 | +0.3827 | **-0.2247** | [-0.5163, +0.0742] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 398/177/3 | +1.1020 | +1.1233 | **-0.0213** | [-0.1357, +0.0989] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.2767 | +1.1704 | **+0.1063** | [-0.0104, +0.2317] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | +0.1265 | +0.0705 | **+0.0560** | [-0.1355, +0.2507] | 2000 | **NOT DETECTED** (CI covers zero) |

### HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY — `V` against its opponent-decodable part

| side | window | ratio of `V` | opponent-decodable part | `V` / decodable |
|---|---|---|---|---|
| arm `ai_v12_21_ladder_lambda09_b` | `t1_3` | +0.0603 | +0.0151 | 3.994 |
| arm `ai_v12_21_ladder_lambda09_b` | `t4_10` | +0.2451 | +0.0932 | 2.630 |
| arm `ai_v12_21_ladder_lambda09_b` | `t11_24` | +0.5169 | +0.1691 | 3.057 |
| control `ai_v12_16_ladder_ctrl10M_c` | `t1_3` | +0.1283 | +0.0239 | 5.362 |
| control `ai_v12_16_ladder_ctrl10M_c` | `t4_10` | +0.3875 | +0.1154 | 3.358 |
| control `ai_v12_16_ladder_ctrl10M_c` | `t11_24` | +0.6023 | +0.1774 | 3.394 |

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
| arm · team | 535 | 601 | 4627 | 9254 | 7.0 | 14.0 | 4–51 |
| arm · stratum | 5 | 601 | 4627 | 9254 | 924.0 | 1848.0 | 924–930 |
| arm · between-team spread | 535 | 601 | 4627 | — | 7.0 | — | — |
| control · team | 534 | 601 | 4615 | 9230 | 7.0 | 14.0 | 4–50 |
| control · stratum | 5 | 601 | 4615 | 9230 | 923.0 | 1846.0 | 920–926 |
| control · between-team spread | 534 | 601 | 4615 | — | 7.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 398/177/3 | +0.0636 | +0.0630 | **+0.0006** | [-0.0060, +0.0061] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 398/177/3 | +0.0345 | +0.0337 | **+0.0008** | [-0.0044, +0.0062] | **WITHIN FLOOR** (|delta| <= floor 0.0028) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 398/177/3 | +0.6277 | +1.1807 | **-0.5530** | [-0.1598, -0.0155] | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 398/177/3 | +0.0483 | +0.0216 | **+0.0268** | [+0.0107, +0.0429] | **NOT DETECTED** (CI does not clear the floor 0.0164) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 398/177/3 | +0.0155 | -0.0049 | **+0.0204** | [+0.0087, +0.0323] | **DETECTED** (CI clears the floor +0.0057) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 398/177/3 | +0.0324 | +0.0264 | **+0.0060** | [-0.0136, +0.0247] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 9562 | 4781 | 0.7980 | **0.1975** | **1.4485** | [0.157, 0.990] | 0.0001 |
| arm · t1_3 | `t1_3` | 9562 | 4781 | 0.7990 | **0.0715** | **0.4578** | [0.634, 0.918] | 0.0000 |
| arm · common support | `common support` | 8975 | 4724 | 0.8194 | **0.1470** | **1.0949** | [0.399, 0.984] | 0.0000 |
| control · all | `all` | 9548 | 4774 | 0.8258 | **0.1874** | **1.6002** | [0.239, 0.996] | 0.0001 |
| control · t1_3 | `t1_3` | 9548 | 4774 | 0.7795 | **0.0872** | **0.5336** | [0.573, 0.923] | 0.0000 |
| control · common support | `common support` | 8641 | 4662 | 0.8325 | **0.1498** | **1.2054** | [0.407, 0.987] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.2386, 0.9901]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.2257 | +1.1723 | **+0.0534** | [-0.0516, +0.1663] | — | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | +0.1960 | +0.0662 | **+0.1298** | [-0.0385, +0.3050] | — | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +1.1265 | +1.1093 | **+0.0172** | [-0.2100, +0.2358] | — | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.1581 | +0.3827 | **-0.2247** | [-0.5163, +0.0742] | — | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 398/177/3 | +1.1020 | +1.1233 | **-0.0213** | [-0.1357, +0.0989] | — | **NOT DETECTED** (CI covers zero) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.2767 | +1.1704 | **+0.1063** | [-0.0104, +0.2317] | — | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +0.1265 | +0.0705 | **+0.0560** | [-0.1355, +0.2507] | — | **NOT DETECTED** (CI covers zero) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0642 [+0.0633, +0.0650] | +0.1305 | **-0.0663** | [-0.0927, -0.0386] | **WITHIN FLOOR** (|delta| <= floor 0.0682) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0642 [+0.0633, +0.0650] | +0.1305 | **-0.0663** | [-0.0927, -0.0386] | **WITHIN FLOOR** (|delta| <= floor 0.0682) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.0642 | +0.1305 | **-0.0663** | [-0.0932, -0.0395] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.5029 [+0.5011, +0.5044] | +0.5046 | **-0.0017** | [-0.0877, +0.0847] | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.5029 [+0.5011, +0.5044] | +0.5046 | **-0.0017** | [-0.0877, +0.0847] | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.5029 | +0.5046 | **-0.0017** | [-0.0905, +0.0817] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0483 [+0.0463, +0.0493] | +0.0216 | **+0.0268** | [+0.0107, +0.0429] | **NOT DETECTED** (CI does not clear the floor 0.0164) |
| `cond.own_team_r2.t1` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0483 [+0.0463, +0.0493] | +0.0216 | **+0.0268** | [+0.0107, +0.0429] | **NOT DETECTED** (CI does not clear the floor 0.0164) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0486 | +0.0216 | **+0.0270** | [+0.0101, +0.0430] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0078 [+0.0054, +0.0096] | +0.0048 | **+0.0030** | [-0.0034, +0.0097] | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| `cond.own_team_r2.all` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0078 [+0.0054, +0.0096] | +0.0048 | **+0.0030** | [-0.0034, +0.0097] | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0050 | +0.0048 | **+0.0002** | [-0.0054, +0.0062] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.5035 [+0.5015, +0.5056] | +0.5080 | **-0.0045** | [-0.0316, +0.0221] | **WITHIN FLOOR** (|delta| <= floor 0.0118) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.5035 [+0.5015, +0.5056] | +0.5080 | **-0.0045** | [-0.0316, +0.0221] | **WITHIN FLOOR** (|delta| <= floor 0.0118) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5123 | +0.5080 | **+0.0043** | [-0.0220, +0.0301] | *no label — not a reading* |
| `cond.opp_class_auc.t1_3` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.5677 [+0.5666, +0.5684] | +0.5924 | **-0.0247** | [-0.0472, -0.0013] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.opp_class_auc.t1_3` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.5677 [+0.5666, +0.5684] | +0.5924 | **-0.0247** | [-0.0472, -0.0013] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.opp_class_auc.t1_3` | UNMATCHED (as traced) | +0.5707 | +0.5924 | **-0.0217** | [-0.0439, +0.0006] | *no label — not a reading* |
| `cond.opp_class_auc.t4_10` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.6991 [+0.6983, +0.7004] | +0.7041 | **-0.0051** | [-0.0272, +0.0157] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t4_10` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.6991 [+0.6983, +0.7004] | +0.7041 | **-0.0051** | [-0.0272, +0.0157] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t4_10` | UNMATCHED (as traced) | +0.7034 | +0.7041 | **-0.0007** | [-0.0220, +0.0204] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0146 [+0.0142, +0.0149] | +0.0239 | **-0.0093** | [-0.0141, -0.0039] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0146 [+0.0142, +0.0149] | +0.0239 | **-0.0093** | [-0.0141, -0.0039] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | UNMATCHED (as traced) | +0.0151 | +0.0239 | **-0.0088** | [-0.0139, -0.0039] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0932 [+0.0928, +0.0936] | +0.1154 | **-0.0221** | [-0.0402, -0.0042] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0932 [+0.0928, +0.0936] | +0.1154 | **-0.0221** | [-0.0402, -0.0042] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | UNMATCHED (as traced) | +0.0932 | +0.1154 | **-0.0222** | [-0.0409, -0.0041] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.1691 [+0.1685, +0.1699] | +0.1774 | **-0.0083** | [-0.0357, +0.0197] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.1691 [+0.1685, +0.1699] | +0.1774 | **-0.0083** | [-0.0357, +0.0197] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | UNMATCHED (as traced) | +0.1691 | +0.1774 | **-0.0083** | [-0.0364, +0.0176] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0636 [+0.0630, +0.0648] | +0.0630 | **+0.0006** | [-0.0060, +0.0061] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| `cond.within_team_resolution.all` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0636 [+0.0630, +0.0648] | +0.0630 | **+0.0006** | [-0.0060, +0.0061] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0645 | +0.0630 | **+0.0015** | [-0.0051, +0.0072] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0345 [+0.0339, +0.0349] | +0.0337 | **+0.0008** | [-0.0044, +0.0062] | **WITHIN FLOOR** (|delta| <= floor 0.0028) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0345 [+0.0339, +0.0349] | +0.0337 | **+0.0008** | [-0.0044, +0.0062] | **WITHIN FLOOR** (|delta| <= floor 0.0028) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0347 | +0.0337 | **+0.0010** | [-0.0042, +0.0064] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.6277 [+0.6194, +0.6398] | +1.1807 | **-0.5530** | [-0.1598, -0.0155] | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.6277 [+0.6194, +0.6398] | +1.1807 | **-0.5530** | [-0.1598, -0.0155] | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.6300 | +1.1807 | **-0.5507** | [-0.1568, -0.0134] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.3132 [+0.3125, +0.3141] | +0.3943 | **-0.0811** | [-0.0891, -0.0282] | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.3132 [+0.3125, +0.3141] | +0.3943 | **-0.0811** | [-0.0891, -0.0282] | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.3135 | +0.3943 | **-0.0808** | [-0.0877, -0.0290] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0155 [+0.0128, +0.0196] | -0.0049 | **+0.0204** | [+0.0087, +0.0323] | **DETECTED** (CI clears the floor +0.0057) |
| `cond.own_team_r2.late` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0155 [+0.0128, +0.0196] | -0.0049 | **+0.0204** | [+0.0087, +0.0323] | **DETECTED** (CI clears the floor +0.0057) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0145 | -0.0049 | **+0.0194** | [+0.0081, +0.0313] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0324 [+0.0273, +0.0352] | +0.0264 | **+0.0060** | [-0.0136, +0.0247] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +0.0324 [+0.0273, +0.0352] | +0.0264 | **+0.0060** | [-0.0136, +0.0247] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0341 | +0.0264 | **+0.0076** | [-0.0121, +0.0255] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +1.1020 [+1.0917, +1.1099] | +1.1233 | **-0.0213** | [-0.1357, +0.0989] | **NOT DETECTED** (CI covers zero) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 398/177/3 (4776 battles, 4622 decoder battles, 21 seeds) | +1.1020 [+1.0917, +1.1099] | +1.1233 | **-0.0213** | [-0.1357, +0.0989] | **NOT DETECTED** (CI covers zero) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.1672 | +1.1233 | **+0.0439** | [-0.0699, +0.1650] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0603 | [+0.0489, +0.0803] | +0.1283 | [+0.1092, +0.1556] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0642 | [+0.0537, +0.0832] | +0.1305 | [+0.1118, +0.1572] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1232 | [-0.1367, -0.1106] | -0.1073 | [-0.1207, -0.0947] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5045 | [+0.4485, +0.5671] | +0.5066 | [+0.4446, +0.5753] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.5029 | [+0.4479, +0.5644] | +0.5046 | [+0.4437, +0.5718] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0650 | [-0.0794, -0.0521] | -0.0607 | [-0.0753, -0.0479] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 4-10` | +0.2451 | [+0.2154, +0.2830] | +0.3875 | [+0.3399, +0.4431] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 11-24` | +0.5169 | [+0.4589, +0.5906] | +0.6023 | [+0.5283, +0.6859] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0486 | [+0.0341, +0.0623] | +0.0216 | [+0.0123, +0.0296] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0050 | [+0.0007, +0.0090] | +0.0048 | [+0.0002, +0.0086] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5123 | [+0.4938, +0.5313] | +0.5080 | [+0.4891, +0.5270] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1-3` | +0.5707 | [+0.5537, +0.5879] | +0.5924 | [+0.5766, +0.6091] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 4-10` | +0.7034 | [+0.6886, +0.7182] | +0.7041 | [+0.6889, +0.7193] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 1-3` | +0.0151 | [+0.0126, +0.0188] | +0.0239 | [+0.0205, +0.0287] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 4-10` | +0.0932 | [+0.0831, +0.1055] | +0.1154 | [+0.1019, +0.1306] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 11-24` | +0.1691 | [+0.1518, +0.1887] | +0.1774 | [+0.1577, +0.1993] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0645 | [+0.0591, +0.0676] | +0.0630 | [+0.0579, +0.0668] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0347 | [+0.0317, +0.0391] | +0.0337 | [+0.0305, +0.0382] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.6300 | [+0.2797, +0.3622] | +1.1807 | [+0.3477, +0.4630] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.3135 | [+0.2347, +0.2715] | +0.3943 | [+0.2884, +0.3351] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0145 | [+0.0031, +0.0257] | -0.0049 | [-0.0086, -0.0028] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0341 | [+0.0177, +0.0500] | +0.0264 | [+0.0174, +0.0356] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.2257 | [+1.1546, +1.3047] | +1.1723 | [+1.0969, +1.2499] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | +0.1960 | [+0.0722, +0.3175] | +0.0662 | [-0.0508, +0.1880] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +1.1265 | [+0.9645, +1.2908] | +1.1093 | [+0.9642, +1.2613] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.1581 | [-0.0583, +0.3835] | +0.3827 | [+0.2005, +0.5678] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.1672 | [+1.0896, +1.2562] | +1.1233 | [+1.0415, +1.2086] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.2767 | [+1.1957, +1.3636] | +1.1704 | [+1.0870, +1.2558] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | +0.1265 | [-0.0034, +0.2585] | +0.0705 | [-0.0563, +0.2020] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_21_ladder_lambda09_b` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0461 [+0.0335, +0.0644] | 0.0618 | -0.0157 | +0.2862 [+0.2070, +0.3557] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0239 [+0.0161, +0.0362] | 0.0337 | -0.0098 | +0.1803 [+0.0186, +0.2870] | ❌ | ❌ | ❌ | ✅ |
| `pool` | yes | 0.0570 [+0.0398, +0.0809] | 0.0711 | -0.0141 | +0.2806 [+0.1926, +0.3509] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 693 battles / 6400 rollouts · Brier 0.0940 = REL 0.0057 − RES 0.0350 + UNC 0.1237 + WBV 0.0007 (resid -1.17e-03) · base rate 0.8553 · **resolution is 28.3% of the base-rate cap** · corr(turn,V) -0.1602 vs corr(turn,MC) -0.1834

**control — `ai_v12_16_ladder_ctrl10M_c` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0459 [+0.0353, +0.0598] | 0.0618 | -0.0159 | +0.2817 [+0.2338, +0.3308] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0177 [+0.0113, +0.0255] | 0.0337 | -0.0159 | +0.1467 [-0.0075, +0.2393] | ❌ | ❌ | ❌ | ❌ |
| `pool` | yes | 0.0596 [+0.0453, +0.0775] | 0.0711 | -0.0114 | +0.2144 [+0.1224, +0.2921] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 673 battles / 6400 rollouts · Brier 0.1085 = REL 0.0005 − RES 0.0332 + UNC 0.1418 + WBV 0.0008 (resid -1.43e-03) · base rate 0.8289 · **resolution is 23.4% of the base-rate cap** · corr(turn,V) -0.1023 vs corr(turn,MC) -0.1075

## 5. THE LEDGER LINE

```
ai_v12_21_ladder_lambda09_b vs ai_v12_16_ladder_ctrl10M_c at 10M [OFFLINE-GENERATED: 400 games x 9+3 opponents, full capture]: G1 bot Δ +0.0048 [-0.0010, +0.0098] NOT DETECTED · identity bias late Δ -0.0996 [-0.1441, -0.0549] WITHIN FLOOR · turn-contrast Δ +0.0181 [-0.1433, +0.1706] WITHIN FLOOR · spread ratio t1-3 Δ -0.0680 [-0.0958, -0.0403] NOT DETECTED · own-team R2 t1 Δ +0.0268 [+0.0107, +0.0429] NOT DETECTED [QUOTA-MATCHED] · calib slope Δ +0.0534 [-0.0516, +0.1663] NOT DETECTED
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_21_ladder_lambda09_b --control ai_v12_16_ladder_ctrl10M_c --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_21_ladder_lambda09_b --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_16_ladder_ctrl10M_c --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_21_ladder_lambda09_b_vs_ctrl10M_c --nice 15 --ledger-line

# arm — ai_v12_21_ladder_lambda09_b  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_16_ladder_ctrl10M_c  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_21_ladder_lambda09_b` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_21_ladder_lambda09_b/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_21_ladder_lambda09_b_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_21_ladder_lambda09_b_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_21_ladder_lambda09_b_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_16_ladder_ctrl10M_c/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_21_ladder_lambda09_b_vs_ctrl10M_c/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_21_ladder_lambda09_b_vs_ctrl10M_c/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
