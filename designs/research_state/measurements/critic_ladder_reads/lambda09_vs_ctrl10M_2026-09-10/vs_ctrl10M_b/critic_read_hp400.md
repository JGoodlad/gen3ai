# CRITIC READ — `ai_v12_19_ladder_lambda09` vs `ai_v12_15_ladder_ctrl10M_b`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v5 at 2026-09-10T12:57:00. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | `step_10000032` | 4800 | 0.3% | 150/150 (100.0%) | no |
| control | `ai_v12_15_ladder_ctrl10M_b` | `step_10000032` | 4800 | 0.6% | 149/150 (99.3%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 400 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260909, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_19_ladder_lambda09` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_15_ladder_ctrl10M_b` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_19_ladder_lambda09` | **397/191/4** | 4800 | 4140 / 646 / 14 | 12 |
| control | `ai_v12_15_ladder_ctrl10M_b` | **398/186/10** | 4800 | 4032 / 740 / 28 | 12 |

**Frames:** the realized per-opponent caps differ — arm 397/191/4, control 398/186/10 (traced W/L/D per opponent, counted on disk); arm and control cut to 397/186/4

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm and control side is subsampled to caps **397/186/4** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json` — 60 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0225 | +0.0207 | **+0.0018** | [-0.0053, +0.0082] | **WITHIN FLOOR** (|delta| <= floor 0.0024) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0184 | +0.0716 | **-0.0900** | [-0.1368, -0.0435] | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.1121 | -0.0316 | **-0.0806** | [-0.2477, +0.0754] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2307 | +0.1962 | **+0.0344** | [-0.0162, +0.0795] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0538 | +0.1110 | **-0.0572** | [-0.0804, -0.0315] | **WITHIN FLOOR** (|delta| <= floor 0.0664) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 397/186/4 | +0.0363 | +0.0229 | **+0.0134** | [-0.0005, +0.0275] | **WITHIN FLOOR** (|delta| <= floor 0.0164) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.3798 | +1.2633 | **+0.1165** | [-0.0023, +0.2384] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0495 | +0.0394 | **+0.0101** | [+0.0016, +0.0191] | 400 | **DETECTED** (CI clears the floor +0.0011) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0225 | +0.0207 | **+0.0018** | [-0.0053, +0.0082] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0024) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0745 | +0.0507 | **+0.0239** | [+0.0084, +0.0382] | 400 | **DETECTED** (CI clears the floor +0.0038) |
| reliability | `all` | `capture-rate (gauge)` | +0.0014 | +0.0049 | **-0.0035** | [-0.0061, -0.0010] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0037) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0007 | +0.0004 | **+0.0004** | [-0.0005, +0.0012] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0032) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0082 | +0.0269 | **-0.0187** | [-0.0298, -0.0079] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| ece | `all` | `capture-rate (gauge)` | +0.0265 | +0.0579 | **-0.0313** | [-0.0446, -0.0138] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0373) |
| ece | `bot` | `capture-rate (gauge)` | +0.0244 | +0.0107 | **+0.0137** | [-0.0004, +0.0249] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0286) |
| ece | `pool` | `capture-rate (gauge)` | +0.0774 | +0.1550 | **-0.0776** | [-0.1131, -0.0391] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1171) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3244 | +0.2228 | **+0.1016** | [+0.0683, +0.1354] | 400 | **DETECTED** (CI clears the floor +0.0307) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2307 | +0.1962 | **+0.0344** | [-0.0162, +0.0795] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.3040 | +0.1122 | **+0.1918** | [+0.1260, +0.2540] | 400 | **NOT DETECTED** (CI does not clear the floor 0.1324) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0425 | +0.1084 | **-0.0659** | [-0.0962, -0.0369] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1029) |
| bias V - p_hat | `ALL` | `pop` | -0.0070 | +0.0344 | **-0.0414** | [-0.0630, -0.0198] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0735) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0070 | +0.0344 | **-0.0414** | [-0.0630, -0.0198] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0735) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0544 | +0.0890 | **-0.0345** | [-0.0792, +0.0080] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1085) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0035 | +0.0137 | **-0.0172** | [-0.0513, +0.0152] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0812) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0035 | +0.0137 | **-0.0172** | [-0.0513, +0.0152] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0812) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0394 | +0.0849 | **-0.0455** | [-0.0862, -0.0056] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0707) |
| bias V - p_hat | `mid (11-24)` | `pop` | -0.0037 | +0.0273 | **-0.0310** | [-0.0611, -0.0010] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0501) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0037 | +0.0273 | **-0.0310** | [-0.0611, -0.0010] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0501) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0298 | +0.1675 | **-0.1377** | [-0.2031, -0.0731] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1434) |
| bias V - p_hat | `late (turn>=25)` | `pop` | -0.0184 | +0.0716 | **-0.0900** | [-0.1368, -0.0435] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0184 | +0.0716 | **-0.0900** | [-0.1368, -0.0435] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| bias V - p_hat | `bot` | `raw` | +0.0097 | +0.0606 | **-0.0510** | [-0.0850, -0.0166] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0639) |
| bias V - p_hat | `bot` | `pop` | -0.0313 | -0.0024 | **-0.0288** | [-0.0502, -0.0075] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0313 | -0.0024 | **-0.0288** | [-0.0502, -0.0075] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| bias V - p_hat | `pool` | `raw` | +0.0957 | +0.1803 | **-0.0847** | [-0.1361, -0.0345] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1594) |
| bias V - p_hat | `pool` | `pop` | +0.0422 | +0.1026 | **-0.0604** | [-0.1044, -0.0166] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1256) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0422 | +0.1026 | **-0.0604** | [-0.1044, -0.0166] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1256) |
| Murphy resolution | `ALL` | `raw` | +0.0367 | +0.0292 | **+0.0075** | [-0.0042, +0.0187] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0425 | +0.0357 | **+0.0068** | [-0.0087, +0.0223] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0425 | +0.0357 | **+0.0068** | [-0.0087, +0.0223] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.2036 | +0.0971 | **+0.1065** | [+0.0471, +0.1679] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0596) |
| Murphy skill_score | `ALL` | `pop` | +0.3131 | +0.2512 | **+0.0620** | [-0.0171, +0.1410] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.3131 | +0.2512 | **+0.0620** | [-0.0171, +0.1410] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.2200 | +0.1570 | **+0.0630** | [+0.0036, +0.1218] | 4000 | **NOT DETECTED** (CI does not clear the floor 0.0205) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.3156 | +0.2528 | **+0.0629** | [-0.0237, +0.1483] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.3156 | +0.2528 | **+0.0629** | [-0.0237, +0.1483] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0038 | +0.0142 | **-0.0104** | [-0.0175, -0.0038] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0111) |
| Murphy reliability | `ALL` | `pop` | +0.0011 | +0.0022 | **-0.0012** | [-0.0044, +0.0017] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0031) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0011 | +0.0022 | **-0.0012** | [-0.0044, +0.0017] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0031) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.1121 | -0.0316 | **-0.0806** | [-0.2477, +0.0754] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.1102 | -0.0719 | **-0.0383** | [-0.1745, +0.0868] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0507) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.1102 | -0.0719 | **-0.0383** | [-0.1745, +0.0868] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0507) |

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

**arm — `ai_v12_19_ladder_lambda09` @ `step_10000032`**: 143812 states / 4786 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 14 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_15_ladder_ctrl10M_b` @ `step_10000032`**: 144313 states / 4772 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 28 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0538 | +0.1110 | **-0.0572** | [-0.0804, -0.0315] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0664) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 397/186/4 | +0.0572 | +0.1125 | **-0.0553** | [-0.0780, -0.0311] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0682) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1218 | -0.1134 | **-0.0084** | [-0.0263, +0.0116] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0147) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5511 | +0.4277 | **+0.1234** | [+0.0390, +0.2157] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2403) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 397/186/4 | +0.5484 | +0.4256 | **+0.1228** | [+0.0345, +0.2086] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0578 | -0.0730 | **+0.0152** | [-0.0037, +0.0359] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0332) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 397/186/4 | +0.0363 | +0.0229 | **+0.0134** | [-0.0005, +0.0275] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0164) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 397/186/4 | +0.0047 | +0.0115 | **-0.0068** | [-0.0135, +0.0007] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 397/186/4 | +0.5087 | +0.5171 | **-0.0085** | [-0.0356, +0.0177] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0118) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 397/186/4 | +0.0608 | +0.0558 | **+0.0050** | [-0.0020, +0.0098] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 397/186/4 | +0.0364 | +0.0319 | **+0.0045** | [-0.0006, +0.0097] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 397/186/4 | +0.3162 | +0.3260 | **-0.0098** | [-0.0445, +0.0113] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 397/186/4 | -0.0022 | +0.0049 | **-0.0070** | [-0.0163, +0.0021] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 397/186/4 | +0.0381 | +0.0180 | **+0.0201** | [+0.0036, +0.0363] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.3798 | +1.2633 | **+0.1165** | [-0.0023, +0.2384] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | -0.3383 | -0.7106 | **+0.3723** | [+0.1633, +0.5930] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +1.1480 | +0.7103 | **+0.4377** | [+0.2322, +0.6500] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | -0.0229 | +0.3998 | **-0.4228** | [-0.7888, -0.0744] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 397/186/4 | +1.3318 | +1.1518 | **+0.1800** | [+0.0621, +0.3073] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.3933 | +1.2467 | **+0.1466** | [+0.0135, +0.2800] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | -0.3592 | -0.6791 | **+0.3200** | [+0.0895, +0.5689] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |

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
| arm · team | 534 | 601 | 4628 | 9256 | 7.0 | 14.0 | 4–51 |
| arm · stratum | 5 | 601 | 4628 | 9256 | 926.0 | 1852.0 | 921–928 |
| arm · between-team spread | 534 | 601 | 4628 | — | 7.0 | — | — |
| control · team | 534 | 601 | 4614 | 9228 | 7.0 | 14.0 | 4–51 |
| control · stratum | 5 | 601 | 4614 | 9228 | 923.0 | 1846.0 | 916–931 |
| control · between-team spread | 534 | 601 | 4614 | — | 7.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 397/186/4 | +0.0608 | +0.0558 | **+0.0050** | [-0.0020, +0.0098] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 397/186/4 | +0.0364 | +0.0319 | **+0.0045** | [-0.0006, +0.0097] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 397/186/4 | +0.0363 | +0.0229 | **+0.0134** | [-0.0005, +0.0275] | **WITHIN FLOOR** (|delta| <= floor 0.0164) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 397/186/4 | -0.0022 | +0.0049 | **-0.0070** | [-0.0163, +0.0021] | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 397/186/4 | +0.0381 | +0.0180 | **+0.0201** | [+0.0036, +0.0363] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 9572 | 4786 | 0.8506 | **0.1745** | **1.5553** | [0.254, 0.996] | 0.0003 |
| arm · t1_3 | `t1_3` | 9572 | 4786 | 0.8394 | **0.0618** | **0.4481** | [0.688, 0.931] | 0.0000 |
| arm · common support | `common support` | 8830 | 4692 | 0.8701 | **0.1193** | **1.2006** | [0.522, 0.992] | 0.0000 |
| control · all | `all` | 9544 | 4772 | 0.8716 | **0.1528** | **1.4015** | [0.368, 0.995] | 0.0000 |
| control · t1_3 | `t1_3` | 9544 | 4772 | 0.8567 | **0.0717** | **0.5978** | [0.685, 0.958] | 0.0000 |
| control · common support | `common support` | 9066 | 4723 | 0.8859 | **0.1096** | **1.1639** | [0.564, 0.993] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.3677, 0.9948]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.3798 | +1.2633 | **+0.1165** | [-0.0023, +0.2384] | — | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | -0.3383 | -0.7106 | **+0.3723** | [+0.1633, +0.5930] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +1.1480 | +0.7103 | **+0.4377** | [+0.2322, +0.6500] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | -0.0229 | +0.3998 | **-0.4228** | [-0.7888, -0.0744] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 397/186/4 | +1.3318 | +1.1518 | **+0.1800** | [+0.0621, +0.3073] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.3933 | +1.2467 | **+0.1466** | [+0.0135, +0.2800] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | -0.3592 | -0.6791 | **+0.3200** | [+0.0895, +0.5689] | — | **DETECTED** (vs ZERO — NO FLOOR) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0572 [+0.0556, +0.0581] | +0.1125 | **-0.0553** | [-0.0780, -0.0311] | **WITHIN FLOOR** (|delta| <= floor 0.0682) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0572 [+0.0556, +0.0581] | +0.1124 | **-0.0552** | [-0.0776, -0.0308] | **WITHIN FLOOR** (|delta| <= floor 0.0682) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.0572 | +0.1124 | **-0.0552** | [-0.0777, -0.0302] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.5484 [+0.5468, +0.5505] | +0.4256 | **+0.1228** | [+0.0345, +0.2086] | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.5484 [+0.5468, +0.5505] | +0.4256 | **+0.1228** | [+0.0424, +0.2069] | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.5487 | +0.4256 | **+0.1231** | [+0.0399, +0.2140] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0363 [+0.0328, +0.0384] | +0.0229 | **+0.0134** | [-0.0005, +0.0275] | **WITHIN FLOOR** (|delta| <= floor 0.0164) |
| `cond.own_team_r2.t1` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0363 [+0.0328, +0.0384] | +0.0224 | **+0.0139** | [-0.0001, +0.0280] | **WITHIN FLOOR** (|delta| <= floor 0.0164) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0368 | +0.0224 | **+0.0144** | [-0.0001, +0.0291] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0047 [+0.0024, +0.0058] | +0.0115 | **-0.0068** | [-0.0135, +0.0007] | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| `cond.own_team_r2.all` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0047 [+0.0024, +0.0058] | +0.0124 | **-0.0078** | [-0.0152, -0.0003] | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0044 | +0.0124 | **-0.0081** | [-0.0157, -0.0009] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.5087 [+0.5037, +0.5112] | +0.5171 | **-0.0085** | [-0.0356, +0.0177] | **WITHIN FLOOR** (|delta| <= floor 0.0118) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.5087 [+0.5037, +0.5112] | +0.5107 | **-0.0020** | [-0.0291, +0.0237] | **WITHIN FLOOR** (|delta| <= floor 0.0118) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5062 | +0.5107 | **-0.0045** | [-0.0314, +0.0212] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0608 [+0.0599, +0.0612] | +0.0558 | **+0.0050** | [-0.0020, +0.0098] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| `cond.within_team_resolution.all` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0608 [+0.0599, +0.0612] | +0.0535 | **+0.0073** | [+0.0001, +0.0113] | **NOT DETECTED** (CI does not clear the floor 0.0056) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0606 | +0.0535 | **+0.0072** | [-0.0001, +0.0113] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0364 [+0.0360, +0.0374] | +0.0319 | **+0.0045** | [-0.0006, +0.0097] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0364 [+0.0360, +0.0374] | +0.0299 | **+0.0064** | [+0.0012, +0.0114] | **NOT DETECTED** (CI does not clear the floor 0.0028) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0385 | +0.0299 | **+0.0086** | [+0.0035, +0.0139] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.3162 [+0.3153, +0.3183] | +0.3260 | **-0.0098** | [-0.0445, +0.0113] | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.3162 [+0.3153, +0.3183] | +0.3260 | **-0.0098** | [-0.0458, +0.0101] | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.3169 | +0.3260 | **-0.0091** | [-0.0464, +0.0113] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | -0.0022 [-0.0031, -0.0007] | +0.0049 | **-0.0070** | [-0.0163, +0.0021] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | -0.0022 [-0.0031, -0.0007] | +0.0056 | **-0.0077** | [-0.0165, +0.0007] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | -0.0039 | +0.0056 | **-0.0095** | [-0.0182, -0.0009] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0381 [+0.0344, +0.0403] | +0.0180 | **+0.0201** | [+0.0036, +0.0363] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +0.0381 [+0.0344, +0.0403] | +0.0168 | **+0.0213** | [+0.0049, +0.0375] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0408 | +0.0168 | **+0.0239** | [+0.0076, +0.0398] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +1.3318 [+1.3139, +1.3381] | +1.1518 | **+0.1800** | [+0.0621, +0.3073] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 397/186/4 (4781 battles, 4623 decoder battles, 21 seeds) | +1.3318 [+1.3139, +1.3381] | +1.1838 | **+0.1480** | [+0.0296, +0.2803] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.3310 | +1.1838 | **+0.1473** | [+0.0234, +0.2741] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0538 | [+0.0424, +0.0717] | +0.1110 | [+0.0948, +0.1333] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0572 | [+0.0465, +0.0742] | +0.1124 | [+0.0965, +0.1340] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1218 | [-0.1360, -0.1088] | -0.1134 | [-0.1269, -0.1012] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5511 | [+0.4851, +0.6230] | +0.4277 | [+0.3765, +0.4821] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.5487 | [+0.4836, +0.6187] | +0.4256 | [+0.3753, +0.4787] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0578 | [-0.0730, -0.0445] | -0.0730 | [-0.0867, -0.0609] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0368 | [+0.0255, +0.0479] | +0.0224 | [+0.0133, +0.0310] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0044 | [+0.0003, +0.0078] | +0.0124 | [+0.0060, +0.0185] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5062 | [+0.4880, +0.5251] | +0.5107 | [+0.4927, +0.5301] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0606 | [+0.0545, +0.0630] | +0.0535 | [+0.0495, +0.0570] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0385 | [+0.0352, +0.0428] | +0.0299 | [+0.0272, +0.0340] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | — | — | +0.6922 | [+0.2892, +0.3751] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.3169 | [+0.2210, +0.2623] | +0.3260 | [+0.2382, +0.2774] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | -0.0039 | [-0.0079, -0.0014] | +0.0056 | [-0.0034, +0.0132] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0408 | [+0.0295, +0.0524] | +0.0168 | [+0.0059, +0.0291] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.3798 | [+1.2977, +1.4676] | +1.2633 | [+1.1888, +1.3413] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | -0.3383 | [-0.4914, -0.1895] | -0.7106 | [-0.8652, -0.5604] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +1.1480 | [+0.9852, +1.3198] | +0.7103 | [+0.5913, +0.8343] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | -0.0229 | [-0.3119, +0.2463] | +0.3998 | [+0.1843, +0.6208] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.3310 | [+1.2470, +1.4295] | +1.1838 | [+1.1050, +1.2694] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.3933 | [+1.2977, +1.4957] | +1.2467 | [+1.1640, +1.3312] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | -0.3592 | [-0.5190, -0.1979] | -0.6791 | [-0.8490, -0.5137] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.team_spread_ratio.t1_3` · arm: the point estimate is undefined on this cycle (too few cells or a degenerate column).
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_19_ladder_lambda09` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 9}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0597 [+0.0478, +0.0727] | 0.0618 | -0.0021 | +0.3259 [+0.2885, +0.3667] | ❌ | ❌ | ❌ | ✅ |
| `bot` | yes | 0.0380 [+0.0273, +0.0515] | 0.0337 | +0.0044 | +0.2738 [+0.2223, +0.3240] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0826 [+0.0638, +0.1072] | 0.0711 | +0.0115 | +0.3494 [+0.2946, +0.4073] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 689 battles / 6400 rollouts · Brier 0.0925 = REL 0.0011 − RES 0.0425 + UNC 0.1347 + WBV 0.0008 (resid -1.54e-03) · base rate 0.8395 · **resolution is 31.6% of the base-rate cap** · corr(turn,V) -0.1911 vs corr(turn,MC) -0.0790

**control — `ai_v12_15_ladder_ctrl10M_b` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0545 [+0.0412, +0.0680] | 0.0618 | -0.0073 | +0.2608 [+0.2118, +0.3048] | ❌ | ❌ | ❌ | ✅ |
| `bot` | yes | 0.0290 [+0.0182, +0.0454] | 0.0337 | -0.0047 | +0.2015 [+0.1342, +0.2626] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0793 [+0.0608, +0.0976] | 0.0711 | +0.0082 | +0.2682 [+0.1854, +0.3287] | ❌ | ❌ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 677 battles / 6400 rollouts · Brier 0.1058 = REL 0.0022 − RES 0.0357 + UNC 0.1413 + WBV 0.0008 (resid -2.80e-03) · base rate 0.8298 · **resolution is 25.3% of the base-rate cap** · corr(turn,V) -0.2201 vs corr(turn,MC) -0.1885

## 5. THE LEDGER LINE

```
ai_v12_19_ladder_lambda09 vs ai_v12_15_ladder_ctrl10M_b at 10M [OFFLINE-GENERATED: 400 games x 9+3 opponents, full capture]: G1 bot Δ +0.0018 [-0.0053, +0.0082] WITHIN FLOOR · identity bias late Δ -0.0900 [-0.1368, -0.0435] WITHIN FLOOR · turn-contrast Δ -0.0806 [-0.2477, +0.0754] NOT DETECTED · spread ratio t1-3 Δ -0.0572 [-0.0804, -0.0315] WITHIN FLOOR · own-team R2 t1 Δ +0.0134 [-0.0005, +0.0275] WITHIN FLOOR [QUOTA-MATCHED] · calib slope Δ +0.1165 [-0.0023, +0.2384] NOT DETECTED
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_19_ladder_lambda09 --control ai_v12_15_ladder_ctrl10M_b --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_19_ladder_lambda09 --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_15_ladder_ctrl10M_b --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M_b --nice 15 --ledger-line

# arm — ai_v12_19_ladder_lambda09  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_15_ladder_ctrl10M_b  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_19_ladder_lambda09` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_19_ladder_lambda09/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_15_ladder_ctrl10M_b` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_15_ladder_ctrl10M_b/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_15_ladder_ctrl10M_b_vs_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_15_ladder_ctrl10M_b_vs_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_15_ladder_ctrl10M_b_vs_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M_b/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_19_ladder_lambda09_vs_ctrl10M_b/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
