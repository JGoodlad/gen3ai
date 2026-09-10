# CRITIC READ — `ai_v12_16_ladder_ctrl10M_c` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v4 at 2026-09-10T10:54:08. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_16_ladder_ctrl10M_c` | `step_10000032` | 4800 | 0.5% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 4800 | 0.5% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 400 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260909, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_16_ladder_ctrl10M_c` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_16_ladder_ctrl10M_c` | **398/177/5** | 4800 | 4068 / 706 / 26 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **400/188/5** | 4800 | 4143 / 634 / 23 | 12 |

**Frames:** the realized per-opponent caps differ — arm 398/177/5, control 400/188/5 (traced W/L/D per opponent, counted on disk); control cut to 398/177/5

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **398/177/5** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json` — 60 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0178 | +0.0202 | **-0.0024** | [-0.0090, +0.0046] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0395 | -0.0308 | **+0.0703** | [+0.0204, +0.1221] | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | +0.0052 | +0.0124 | **-0.0073** | [-0.1684, +0.1552] | **WITHIN FLOOR** (|delta| <= floor 0.0440) |
| skill · `bot` | — | +0.1479 | +0.1636 | **-0.0157** | [-0.0877, +0.0579] | **WITHIN FLOOR** (|delta| <= floor 0.0327) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.1283 | +0.1774 | **-0.0491** | [-0.0894, -0.0127] | **WITHIN FLOOR** (|delta| <= floor 0.0664) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 398/177/5 | +0.0216 | +0.0380 | **-0.0164** | [-0.0306, -0.0027] | **NOT DETECTED** (CI does not clear the floor 0.0146) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0384 | +0.0395 | **-0.0011** | [-0.0085, +0.0058] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0178 | +0.0202 | **-0.0024** | [-0.0090, +0.0046] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0562 | +0.0544 | **+0.0018** | [-0.0107, +0.0152] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0038) |
| reliability | `all` | `capture-rate (gauge)` | +0.0001 | +0.0013 | **-0.0012** | [-0.0025, -0.0002] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0037) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0039 | +0.0036 | **+0.0002** | [-0.0028, +0.0030] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0032) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0101 | +0.0022 | **+0.0079** | [+0.0018, +0.0151] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| ece | `all` | `capture-rate (gauge)` | +0.0063 | +0.0206 | **-0.0143** | [-0.0242, -0.0003] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0373) |
| ece | `bot` | `capture-rate (gauge)` | +0.0432 | +0.0394 | **+0.0038** | [-0.0139, +0.0215] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0286) |
| ece | `pool` | `capture-rate (gauge)` | +0.0951 | +0.0379 | **+0.0572** | [+0.0163, +0.0891] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1171) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2542 | +0.2535 | **+0.0006** | [-0.0386, +0.0453] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0307) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.1479 | +0.1636 | **-0.0157** | [-0.0877, +0.0579] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0327) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2056 | +0.2446 | **-0.0390** | [-0.0998, +0.0230] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1324) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0554 | +0.0055 | **+0.0499** | [+0.0182, +0.0820] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1029) |
| bias V - p_hat | `ALL` | `pop` | -0.0039 | -0.0391 | **+0.0353** | [+0.0127, +0.0588] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0735) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0039 | -0.0391 | **+0.0353** | [+0.0127, +0.0588] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0735) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0115 | -0.0196 | **+0.0310** | [-0.0156, +0.0771] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1085) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0461 | -0.0675 | **+0.0213** | [-0.0117, +0.0554] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0812) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0461 | -0.0675 | **+0.0213** | [-0.0117, +0.0554] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0812) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0552 | +0.0143 | **+0.0409** | [-0.0011, +0.0838] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0707) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0039 | -0.0228 | **+0.0267** | [-0.0035, +0.0593] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0501) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0039 | -0.0228 | **+0.0267** | [-0.0035, +0.0593] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0501) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.1067 | +0.0241 | **+0.0826** | [+0.0177, +0.1501] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1434) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0395 | -0.0308 | **+0.0703** | [+0.0204, +0.1221] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0395 | -0.0308 | **+0.0703** | [+0.0204, +0.1221] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| bias V - p_hat | `bot` | `raw` | +0.0042 | -0.0033 | **+0.0075** | [-0.0289, +0.0430] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0639) |
| bias V - p_hat | `bot` | `pop` | -0.0382 | -0.0473 | **+0.0091** | [-0.0147, +0.0316] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0382 | -0.0473 | **+0.0091** | [-0.0147, +0.0316] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| bias V - p_hat | `pool` | `raw` | +0.1500 | +0.0210 | **+0.1291** | [+0.0745, +0.1878] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1594) |
| bias V - p_hat | `pool` | `pop` | +0.0777 | -0.0230 | **+0.1007** | [+0.0510, +0.1510] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1256) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0777 | -0.0230 | **+0.1007** | [+0.0510, +0.1510] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1256) |
| Murphy resolution | `ALL` | `raw` | +0.0261 | +0.0263 | **-0.0002** | [-0.0118, +0.0119] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0029) |
| Murphy resolution | `ALL` | `pop` | +0.0332 | +0.0315 | **+0.0017** | [-0.0131, +0.0173] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0042) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0332 | +0.0315 | **+0.0017** | [-0.0131, +0.0173] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0042) |
| Murphy skill_score | `ALL` | `raw` | +0.1223 | +0.1567 | **-0.0345** | [-0.1119, +0.0499] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0596) |
| Murphy skill_score | `ALL` | `pop` | +0.2353 | +0.2213 | **+0.0141** | [-0.0902, +0.1234] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0299) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2353 | +0.2213 | **+0.0141** | [-0.0902, +0.1234] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0299) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1477 | +0.1683 | **-0.0205** | [-0.0864, +0.0451] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2339 | +0.2448 | **-0.0109** | [-0.1006, +0.0792] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2339 | +0.2448 | **-0.0109** | [-0.1006, +0.0792] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0058 | +0.0031 | **+0.0028** | [-0.0021, +0.0079] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0111) |
| Murphy reliability | `ALL` | `pop` | +0.0005 | +0.0036 | **-0.0031** | [-0.0072, +0.0004] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0005 | +0.0036 | **-0.0031** | [-0.0072, +0.0004] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.0052 | +0.0124 | **-0.0073** | [-0.1684, +0.1552] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0440) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0014 | -0.0212 | **+0.0199** | [-0.1183, +0.1603] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0507) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0014 | -0.0212 | **+0.0199** | [-0.1183, +0.1603] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0507) |

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

**arm — `ai_v12_16_ladder_ctrl10M_c` @ `step_10000032`**: 148990 states / 4774 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 26 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 147339 states / 4777 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 23 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.1283 | +0.1774 | **-0.0491** | [-0.0894, -0.0127] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0664) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 398/177/5 | +0.1305 | +0.1806 | **-0.0501** | [-0.0903, -0.0121] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0682) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1073 | -0.0987 | **-0.0086** | [-0.0276, +0.0091] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0147) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5066 | +0.6680 | **-0.1614** | [-0.2694, -0.0623] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2403) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 398/177/5 | +0.5046 | +0.6641 | **-0.1595** | [-0.2634, -0.0516] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0607 | -0.0398 | **-0.0209** | [-0.0405, -0.0024] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0332) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 398/177/5 | +0.0216 | +0.0380 | **-0.0164** | [-0.0306, -0.0027] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0146) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 398/177/5 | +0.0048 | +0.0049 | **-0.0001** | [-0.0057, +0.0056] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 398/177/5 | +0.5080 | +0.5197 | **-0.0118** | [-0.0385, +0.0152] | 2000 | **NOT DETECTED** (CI covers zero) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 398/177/5 | +0.0630 | +0.0595 | **+0.0035** | [-0.0022, +0.0095] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 398/177/5 | +0.0337 | +0.0308 | **+0.0028** | [-0.0022, +0.0079] | 2000 | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 398/177/5 | +1.1807 | +1.9932 | **-0.8124** | [-0.2891, -0.0777] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 398/177/5 | +0.3943 | +0.5705 | **-0.1762** | [-0.1774, -0.0921] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 398/177/5 | -0.0049 | +0.0006 | **-0.0055** | [-0.0123, +0.0016] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0057) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 398/177/5 | +0.0264 | +0.0368 | **-0.0103** | [-0.0257, +0.0046] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · team | 534 | 601 | 4615 | 9230 | 7.0 | 14.0 | 4–50 |
| arm · stratum | 5 | 601 | 4615 | 9230 | 923.0 | 1846.0 | 920–926 |
| arm · between-team spread | 534 | 601 | 4615 | — | 7.0 | — | — |
| control · team | 535 | 601 | 4622 | 9244 | 7.0 | 14.0 | 4–51 |
| control · stratum | 5 | 601 | 4622 | 9244 | 926.0 | 1852.0 | 905–937 |
| control · between-team spread | 535 | 601 | 4622 | — | 7.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 398/177/5 | +0.0630 | +0.0595 | **+0.0035** | [-0.0022, +0.0095] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 398/177/5 | +0.0337 | +0.0308 | **+0.0028** | [-0.0022, +0.0079] | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 398/177/5 | +1.1807 | +1.9932 | **-0.8124** | [-0.2891, -0.0777] | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 398/177/5 | +0.0216 | +0.0380 | **-0.0164** | [-0.0306, -0.0027] | **NOT DETECTED** (CI does not clear the floor 0.0146) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 398/177/5 | -0.0049 | +0.0006 | **-0.0055** | [-0.0123, +0.0016] | **WITHIN FLOOR** (|delta| <= floor 0.0057) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 398/177/5 | +0.0264 | +0.0368 | **-0.0103** | [-0.0257, +0.0046] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

The two own-team R² rows are printed with them: the contrast is their difference, and it cannot be read without seeing which end moved.

**Reading of the three signs:** neither cleanly — the WITHIN-team discrimination is up without a between-team spread to go with it, which is a board-reading gain, not a conditioning one.

> 🚨 **The between-team spread is an AMPLITUDE; the own-team R² is an ALIGNMENT.** The spread ratio compares `sd(mean V per team)` with `sd(that team's win rate)` — how far apart the head's per-team opinions are. The own-team R² is an out-of-fold MONOTONE decode, invariant to scale — whether those opinions are in the right ORDER. The two can move in opposite directions (a head that orders its teams correctly but under-disperses reads R² UP and spread DOWN), so the sign table above is read with both rows in hand and the R² row is never read alone.

> 🚨 **The per-TEAM cells are small and the coarse row is the check on them.** With ~719 teams in the pool a few-thousand-battle frame leaves a handful of episodes per team, and a binned resolution inside a cell that size is largely the binning's own noise — which is *positively* biased, so a small per-team number is evidence of neither reading. The STRATUM row is the same estimator on cells hundreds of episodes deep. **Where the two disagree, believe the stratum row and say so.**

> 🚨 **`cond.own_team_r2.t1_minus_late` is PROVISIONAL and is never labelled DETECTED.** NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | +0.1305 [+0.1778, +0.1826] | +0.1806 | **-0.0501** | [-0.0903, -0.0121] | **WITHIN FLOOR** (|delta| <= floor 0.0682) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.1305 [+0.1804, +0.1804] | +0.1804 | **-0.0500** | [-0.0894, -0.0143] | **WITHIN FLOOR** (|delta| <= floor 0.0682) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1305 | +0.1804 | **-0.0500** | [-0.0894, -0.0143] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | +0.5046 [+0.6593, +0.6682] | +0.6641 | **-0.1595** | [-0.2634, -0.0516] | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.5046 [+0.6644, +0.6644] | +0.6644 | **-0.1598** | [-0.2651, -0.0625] | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.5046 | +0.6644 | **-0.1598** | [-0.2651, -0.0625] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | +0.0216 [+0.0342, +0.0401] | +0.0380 | **-0.0164** | [-0.0306, -0.0027] | **NOT DETECTED** (CI does not clear the floor 0.0146) |
| `cond.own_team_r2.t1` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0216 [+0.0376, +0.0376] | +0.0376 | **-0.0160** | [-0.0311, -0.0013] | **NOT DETECTED** (CI does not clear the floor 0.0146) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0216 | +0.0376 | **-0.0160** | [-0.0311, -0.0013] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | +0.0048 [+0.0026, +0.0059] | +0.0049 | **-0.0001** | [-0.0057, +0.0056] | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| `cond.own_team_r2.all` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0048 [+0.0048, +0.0048] | +0.0048 | **+0.0000** | [-0.0062, +0.0057] | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0048 | +0.0048 | **+0.0000** | [-0.0062, +0.0057] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | +0.5080 [+0.5167, +0.5212] | +0.5197 | **-0.0118** | [-0.0385, +0.0152] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.5080 [+0.5137, +0.5137] | +0.5137 | **-0.0057** | [-0.0322, +0.0207] | **WITHIN FLOOR** (|delta| <= floor 0.0070) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5080 | +0.5137 | **-0.0057** | [-0.0322, +0.0207] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | +0.0630 [+0.0583, +0.0606] | +0.0595 | **+0.0035** | [-0.0022, +0.0095] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| `cond.within_team_resolution.all` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0630 [+0.0595, +0.0595] | +0.0595 | **+0.0035** | [-0.0030, +0.0095] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0630 | +0.0595 | **+0.0035** | [-0.0030, +0.0095] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | +0.0337 [+0.0302, +0.0321] | +0.0308 | **+0.0028** | [-0.0022, +0.0079] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0337 [+0.0305, +0.0305] | +0.0305 | **+0.0032** | [-0.0023, +0.0084] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0337 | +0.0305 | **+0.0032** | [-0.0023, +0.0084] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | +1.1807 [+1.8282, +2.6407] | +1.9932 | **-0.8124** | [-0.2891, -0.0777] | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | +1.1807 [+2.0616, +2.0616] | +2.0616 | **-0.8809** | [-0.2967, -0.0777] | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +1.1807 | +2.0616 | **-0.8809** | [-0.2967, -0.0777] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | +0.3943 [+0.5674, +0.5790] | +0.5705 | **-0.1762** | [-0.1774, -0.0921] | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.3943 [+0.5732, +0.5732] | +0.5732 | **-0.1789** | [-0.1788, -0.0892] | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.3943 | +0.5732 | **-0.1789** | [-0.1788, -0.0892] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | -0.0049 [-0.0014, +0.0042] | +0.0006 | **-0.0055** | [-0.0123, +0.0016] | **WITHIN FLOOR** (|delta| <= floor 0.0057) |
| `cond.own_team_r2.late` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | -0.0049 [+0.0028, +0.0028] | +0.0028 | **-0.0077** | [-0.0143, -0.0012] | **NOT DETECTED** (CI does not clear the floor 0.0057) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | -0.0049 | +0.0028 | **-0.0077** | [-0.0143, -0.0012] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 398/177/5 (4764 battles, 4606 decoder battles, 21 seeds) | +0.0264 [+0.0332, +0.0400] | +0.0368 | **-0.0103** | [-0.0257, +0.0046] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 448/199/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0264 [+0.0348, +0.0348] | +0.0348 | **-0.0083** | [-0.0237, +0.0071] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0264 | +0.0348 | **-0.0083** | [-0.0237, +0.0071] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.1283 | [+0.1092, +0.1556] | +0.1774 | [+0.1491, +0.2145] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1305 | [+0.1118, +0.1572] | +0.1804 | [+0.1527, +0.2168] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1073 | [-0.1207, -0.0947] | -0.0987 | [-0.1130, -0.0855] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5066 | [+0.4446, +0.5753] | +0.6680 | [+0.5832, +0.7618] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.5046 | [+0.4437, +0.5718] | +0.6644 | [+0.5812, +0.7564] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0607 | [-0.0753, -0.0479] | -0.0398 | [-0.0549, -0.0259] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0216 | [+0.0123, +0.0296] | +0.0376 | [+0.0259, +0.0493] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0048 | [+0.0002, +0.0086] | +0.0048 | [+0.0003, +0.0090] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5080 | [+0.4891, +0.5270] | +0.5137 | [+0.4959, +0.5325] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0630 | [+0.0579, +0.0668] | +0.0595 | [+0.0543, +0.0629] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0337 | [+0.0305, +0.0382] | +0.0305 | [+0.0277, +0.0347] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +1.1807 | [+0.3477, +0.4630] | +2.0616 | [+0.5026, +0.6841] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.3943 | [+0.2884, +0.3351] | +0.5732 | [+0.4093, +0.4831] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | -0.0049 | [-0.0086, -0.0028] | +0.0028 | [-0.0038, +0.0079] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0264 | [+0.0174, +0.0356] | +0.0348 | [+0.0229, +0.0473] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_16_ladder_ctrl10M_c` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0459 [+0.0353, +0.0598] | 0.0618 | -0.0159 | +0.2817 [+0.2338, +0.3308] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0177 [+0.0113, +0.0255] | 0.0337 | -0.0159 | +0.1467 [-0.0075, +0.2393] | ❌ | ❌ | ❌ | ❌ |
| `pool` | yes | 0.0596 [+0.0453, +0.0775] | 0.0711 | -0.0114 | +0.2144 [+0.1224, +0.2921] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 673 battles / 6400 rollouts · Brier 0.1085 = REL 0.0005 − RES 0.0332 + UNC 0.1418 + WBV 0.0008 (resid -1.43e-03) · base rate 0.8289 · **resolution is 23.4% of the base-rate cap** · corr(turn,V) -0.1023 vs corr(turn,MC) -0.1075

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 678 battles / 6400 rollouts · Brier 0.1002 = REL 0.0036 − RES 0.0315 + UNC 0.1287 + WBV 0.0008 (resid -1.32e-03) · base rate 0.8482 · **resolution is 24.5% of the base-rate cap** · corr(turn,V) -0.1044 vs corr(turn,MC) -0.1168

## 5. THE LEDGER LINE

```
ai_v12_16_ladder_ctrl10M_c vs ai_v12_11_ladder_ctrl10M at 10M [OFFLINE-GENERATED: 400 games x 9+3 opponents, full capture]: G1 bot Δ -0.0024 [-0.0090, +0.0046] NOT DETECTED · identity bias late Δ +0.0703 [+0.0204, +0.1221] WITHIN FLOOR · turn-contrast Δ -0.0073 [-0.1684, +0.1552] WITHIN FLOOR · spread ratio t1-3 Δ -0.0491 [-0.0894, -0.0127] WITHIN FLOOR · own-team R2 t1 Δ -0.0164 [-0.0306, -0.0027] NOT DETECTED [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_16_ladder_ctrl10M_c --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_16_ladder_ctrl10M_c --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_11_ladder_ctrl10M --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M --nice 15 --ledger-line

# arm — ai_v12_16_ladder_ctrl10M_c
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_16_ladder_ctrl10M_c --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_16_ladder_ctrl10M_c/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_16_ladder_ctrl10M_c_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
