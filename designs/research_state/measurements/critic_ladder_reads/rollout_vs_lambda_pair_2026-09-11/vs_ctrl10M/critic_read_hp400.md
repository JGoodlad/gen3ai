# CRITIC READ — `ai_v12_23_ladder_rollout` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v6 at 2026-09-11T12:10:01. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

**SELF-PLAY CROSSING** — arm 2,162,688 · control 4,128,768 (first step carrying any `*_pool` scalar; the PROMOTION scalar itself lands one eval→rollout lag earlier, at 2,000,016 / 4,000,032). Descriptive — no floor, no verdict.

> 🚨 **CROSSING MISMATCH: arm 2,162,688 vs control 4,128,768.** The two runs seeded self-play at different steps, so the rows below compare heads fitted against opponent distributions that differ over the span between them. The first crossing is a coin flip on the SEED path — `SELF_PLAY_START = 0.55` (`--self-play-start-wr`) against `win_rate_vs_bots`, inside the 2M-bots replicate floor — so this is a property of the DRAW, not of the lever under test. ⚠️ It is a RAMP, not a step: the early crosser's extra span runs at a self-play fraction rising from ~0.03, not at 0.9, so the size of the mismatch is the integral of `selfplay_fraction` over the span and not the span itself. This line SELECTS NOTHING — the crossing is POST-TREATMENT and may be a mediator (ledger 2026-09-11 CORRECTION).

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_23_ladder_rollout` | `step_10000032` | 4800 | 0.5% | 150/150 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 4800 | 0.5% | 150/150 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |
| control | OFFLINE-GENERATED cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 400 games, ALL battles traced (no capture quota); sentinels GREEDY, drawing the trainee's own team distribution |

> 🚨 **OFFLINE-GENERATED.** Both cycles were replayed from their saved checkpoints by `main.ops.eval_trace_gen` — 400 games per opponent against 3 pool sentinels, every battle traced. **These rows are NOT comparable with a live 100-game read and must never sit in one table beside it**: they are a different number of games, possibly a different number of opponent cells, and a random sample rather than the live outcome quota's loss-enriched slice. Report them as their own table, against their own floor pair.

> Reproducibility: seeded, concurrency 1 — the whole cycle is a pure function of (seed, plan); the worker count does not enter it (seed 20260909, 4 worker(s), concurrency 1).

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_23_ladder_rollout` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_23_ladder_rollout` | **399/169/6** | 4800 | 4126 / 651 / 23 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **400/188/5** | 4800 | 4143 / 634 / 23 | 12 |

**Frames:** the realized per-opponent caps differ — arm 399/169/6, control 400/188/5 (traced W/L/D per opponent, counted on disk); control cut to 399/169/5

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The control side is subsampled to caps **399/169/5** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> Floor: `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json` — 60 keyed magnitudes.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0236 | +0.0202 | **+0.0034** | [-0.0032, +0.0104] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | +0.0109 | -0.0308 | **+0.0417** | [-0.0103, +0.0937] | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.0496 | +0.0124 | **-0.0620** | [-0.2165, +0.0870] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.2218 | +0.1636 | **+0.0582** | [+0.0018, +0.1188] | **NOT DETECTED** (CI does not clear the floor 0.0327) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0800 | +0.1774 | **-0.0974** | [-0.1353, -0.0616] | **NOT DETECTED** (CI does not clear the floor 0.0664) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 399/169/5 | +0.0224 | +0.0364 | **-0.0141** | [-0.0286, +0.0002] | **WITHIN FLOOR** (|delta| <= floor 0.0164) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | as traced | +1.2970 | +1.0719 | **+0.2251** | [+0.1168, +0.3399] | **DETECTED** (vs ZERO — NO FLOOR) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0410 | +0.0395 | **+0.0015** | [-0.0061, +0.0090] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0236 | +0.0202 | **+0.0034** | [-0.0032, +0.0104] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0589 | +0.0544 | **+0.0044** | [-0.0094, +0.0178] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0020 | +0.0013 | **+0.0007** | [-0.0011, +0.0027] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0037) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0005 | +0.0036 | **-0.0031** | [-0.0055, -0.0013] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0032) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0120 | +0.0022 | **+0.0098** | [+0.0031, +0.0177] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0247) |
| ece | `all` | `capture-rate (gauge)` | +0.0290 | +0.0206 | **+0.0084** | [-0.0083, +0.0217] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0373) |
| ece | `bot` | `capture-rate (gauge)` | +0.0122 | +0.0394 | **-0.0271** | [-0.0406, -0.0121] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0286) |
| ece | `pool` | `capture-rate (gauge)` | +0.0894 | +0.0379 | **+0.0515** | [+0.0154, +0.0815] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1171) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2640 | +0.2535 | **+0.0104** | [-0.0276, +0.0490] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.0307) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2218 | +0.1636 | **+0.0582** | [+0.0018, +0.1188] | 400 | **NOT DETECTED** (CI does not clear the floor 0.0327) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2253 | +0.2446 | **-0.0193** | [-0.0785, +0.0457] | 400 | **WITHIN FLOOR** (|delta| <= floor 0.1324) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0624 | +0.0055 | **+0.0569** | [+0.0282, +0.0861] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1029) |
| bias V - p_hat | `ALL` | `pop` | -0.0014 | -0.0391 | **+0.0377** | [+0.0153, +0.0605] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0735) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0014 | -0.0391 | **+0.0377** | [+0.0153, +0.0605] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0735) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0648 | -0.0196 | **+0.0844** | [+0.0391, +0.1292] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1085) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0135 | -0.0675 | **+0.0540** | [+0.0195, +0.0911] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0812) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0135 | -0.0675 | **+0.0540** | [+0.0195, +0.0911] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0812) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0648 | +0.0143 | **+0.0505** | [+0.0091, +0.0886] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0707) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0057 | -0.0228 | **+0.0285** | [-0.0008, +0.0574] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0501) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0057 | -0.0228 | **+0.0285** | [-0.0008, +0.0574] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0501) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0556 | +0.0241 | **+0.0315** | [-0.0339, +0.0936] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1434) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0109 | -0.0308 | **+0.0417** | [-0.0103, +0.0937] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0109 | -0.0308 | **+0.0417** | [-0.0103, +0.0937] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1024) |
| bias V - p_hat | `bot` | `raw` | +0.0401 | -0.0033 | **+0.0433** | [+0.0084, +0.0777] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0639) |
| bias V - p_hat | `bot` | `pop` | -0.0197 | -0.0473 | **+0.0276** | [+0.0035, +0.0519] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0197 | -0.0473 | **+0.0276** | [+0.0035, +0.0519] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0449) |
| bias V - p_hat | `pool` | `raw` | +0.0995 | +0.0210 | **+0.0785** | [+0.0300, +0.1308] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1594) |
| bias V - p_hat | `pool` | `pop` | +0.0345 | -0.0230 | **+0.0575** | [+0.0121, +0.1052] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1256) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0345 | -0.0230 | **+0.0575** | [+0.0121, +0.1052] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.1256) |
| Murphy resolution | `ALL` | `raw` | +0.0308 | +0.0263 | **+0.0044** | [-0.0065, +0.0158] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0315 | +0.0315 | **-0.0000** | [-0.0133, +0.0137] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0042) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0315 | +0.0315 | **-0.0000** | [-0.0133, +0.0137] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0042) |
| Murphy skill_score | `ALL` | `raw` | +0.1402 | +0.1567 | **-0.0165** | [-0.0844, +0.0572] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0596) |
| Murphy skill_score | `ALL` | `pop` | +0.2406 | +0.2213 | **+0.0193** | [-0.0718, +0.1205] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0299) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2406 | +0.2213 | **+0.0193** | [-0.0718, +0.1205] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0299) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1803 | +0.1683 | **+0.0121** | [-0.0505, +0.0746] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0205) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2479 | +0.2448 | **+0.0031** | [-0.0780, +0.0884] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0109) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2479 | +0.2448 | **+0.0031** | [-0.0780, +0.0884] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0109) |
| Murphy reliability | `ALL` | `raw` | +0.0080 | +0.0031 | **+0.0050** | [-0.0004, +0.0103] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0111) |
| Murphy reliability | `ALL` | `pop` | +0.0011 | +0.0036 | **-0.0025** | [-0.0067, +0.0015] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0031) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0011 | +0.0036 | **-0.0025** | [-0.0067, +0.0015] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0031) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.0496 | +0.0124 | **-0.0620** | [-0.2165, +0.0870] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0145 | -0.0212 | **+0.0067** | [-0.1401, +0.1462] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0507) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | -0.0145 | -0.0212 | **+0.0067** | [-0.1401, +0.1462] | 4000 | **WITHIN FLOOR** (|delta| <= floor 0.0507) |

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

**arm — `ai_v12_23_ladder_rollout` @ `step_10000032`**: 142380 states / 4777 battles / 12 opponents / 600 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 23 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 147339 states / 4777 battles / 12 opponents / 601 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 23 draw/timeout battles excluded (no binary outcome) · strength axis: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0800 | +0.1774 | **-0.0974** | [-0.1353, -0.0616] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0664) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 399/169/5 | +0.0843 | +0.1799 | **-0.0956** | [-0.1330, -0.0594] | 2000 | **NOT DETECTED** (CI does not clear the floor 0.0682) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1056 | -0.0987 | **-0.0069** | [-0.0255, +0.0112] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0147) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.4926 | +0.6680 | **-0.1754** | [-0.2872, -0.0711] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2403) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 399/169/5 | +0.4896 | +0.6626 | **-0.1730** | [-0.2759, -0.0713] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0582 | -0.0398 | **-0.0184** | [-0.0382, +0.0007] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0332) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 4-10` | as traced | +0.2816 | +0.5451 | **-0.2635** | [-0.3473, -0.1901] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `turn 11-24` | as traced | +0.5217 | +0.7482 | **-0.2266** | [-0.3512, -0.1087] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 399/169/5 | +0.0224 | +0.0364 | **-0.0141** | [-0.0286, +0.0002] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0164) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 399/169/5 | +0.0031 | +0.0047 | **-0.0016** | [-0.0068, +0.0037] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 399/169/5 | +0.5020 | +0.5210 | **-0.0191** | [-0.0458, +0.0080] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1-3` | MATCHED 399/169/5 | +0.5579 | +0.6067 | **-0.0487** | [-0.0724, -0.0260] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 4-10` | MATCHED 399/169/5 | +0.6940 | +0.7161 | **-0.0222** | [-0.0421, -0.0014] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 1-3` | MATCHED 399/169/5 | +0.0155 | +0.0275 | **-0.0120** | [-0.0178, -0.0065] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 4-10` | MATCHED 399/169/5 | +0.1131 | +0.1350 | **-0.0219** | [-0.0439, -0.0002] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) | `turn 11-24` | MATCHED 399/169/5 | +0.2010 | +0.2011 | **-0.0002** | [-0.0342, +0.0346] | 2000 | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 399/169/5 | +0.0542 | +0.0591 | **-0.0049** | [-0.0098, +0.0019] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 399/169/5 | +0.0314 | +0.0310 | **+0.0003** | [-0.0048, +0.0053] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0028) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 399/169/5 | +1.0013 | +2.0094 | **-1.0082** | [-0.3017, -0.0871] | 2000 | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 399/169/5 | +0.3779 | +0.5675 | **-0.1897** | [-0.1883, -0.0998] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 399/169/5 | +0.0011 | +0.0003 | **+0.0007** | [-0.0072, +0.0089] | 2000 | **WITHIN FLOOR** (|delta| <= floor 0.0057) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 399/169/5 | +0.0213 | +0.0365 | **-0.0152** | [-0.0315, +0.0004] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) | `all states, <=2 per battle` | as traced | +1.2970 | +1.0719 | **+0.2251** | [+0.1168, +0.3399] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) | `all states, <=2 per battle` | as traced | -0.3850 | +0.4257 | **-0.8107** | [-0.9962, -0.6383] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) | `turn 1-3, <=2 per battle` | as traced | +0.9440 | +0.8379 | **+0.1061** | [-0.0871, +0.3117] | 2000 | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large — the INTERCEPT of that regression | `turn 1-3, <=2 per battle` | as traced | +0.5892 | +0.9122 | **-0.3230** | [-0.5838, -0.0564] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams | `all states, <=2 per battle, stratum fixed effects` | MATCHED 399/169/5 | +1.2351 | +1.0434 | **+0.1917** | [+0.0802, +0.3139] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them | `all states, <=2 per battle, common V window` | as traced | +1.2983 | +1.0281 | **+0.2702** | [+0.1449, +0.3926] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large on the COMMON SUPPORT | `all states, <=2 per battle, common V window` | as traced | -0.3885 | +0.4871 | **-0.8756** | [-1.0870, -0.6652] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |

### HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY — `V` against its opponent-decodable part

| side | window | ratio of `V` | opponent-decodable part | `V` / decodable |
|---|---|---|---|---|
| arm `ai_v12_23_ladder_rollout` | `t1_3` | +0.0800 | +0.0155 | 5.168 |
| arm `ai_v12_23_ladder_rollout` | `t4_10` | +0.2816 | +0.1131 | 2.490 |
| arm `ai_v12_23_ladder_rollout` | `t11_24` | +0.5217 | +0.2010 | 2.596 |
| control `ai_v12_11_ladder_ctrl10M` | `t1_3` | +0.1774 | +0.0268 | 6.625 |
| control `ai_v12_11_ladder_ctrl10M` | `t4_10` | +0.5451 | +0.1354 | 4.025 |
| control `ai_v12_11_ladder_ctrl10M` | `t11_24` | +0.7482 | +0.2009 | 3.725 |

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
| arm · team | 534 | 600 | 4620 | 9240 | 7.0 | 14.0 | 4–51 |
| arm · stratum | 5 | 600 | 4620 | 9240 | 922.0 | 1844.0 | 917–932 |
| arm · between-team spread | 534 | 600 | 4620 | — | 7.0 | — | — |
| control · team | 535 | 601 | 4622 | 9244 | 7.0 | 14.0 | 4–51 |
| control · stratum | 5 | 601 | 4622 | 9244 | 926.0 | 1852.0 | 905–937 |
| control · between-team spread | 535 | 601 | 4622 | — | 7.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 399/169/5 | +0.0542 | +0.0591 | **-0.0049** | [-0.0098, +0.0019] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 399/169/5 | +0.0314 | +0.0310 | **+0.0003** | [-0.0048, +0.0053] | **WITHIN FLOOR** (|delta| <= floor 0.0028) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 399/169/5 | +1.0013 | +2.0094 | **-1.0082** | [-0.3017, -0.0871] | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 399/169/5 | +0.0224 | +0.0364 | **-0.0141** | [-0.0286, +0.0002] | **WITHIN FLOOR** (|delta| <= floor 0.0164) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 399/169/5 | +0.0011 | +0.0003 | **+0.0007** | [-0.0072, +0.0089] | **WITHIN FLOOR** (|delta| <= floor 0.0057) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 399/169/5 | +0.0213 | +0.0365 | **-0.0152** | [-0.0315, +0.0004] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · all | `all` | 9554 | 4777 | 0.8606 | **0.1585** | **1.4853** | [0.360, 0.996] | 0.0000 |
| arm · t1_3 | `t1_3` | 9554 | 4777 | 0.7932 | **0.0789** | **0.5109** | [0.618, 0.925] | 0.0000 |
| arm · common support | `common support` | 9076 | 4740 | 0.8746 | **0.1169** | **1.2473** | [0.548, 0.994] | 0.0000 |
| control · all | `all` | 9554 | 4777 | 0.8200 | **0.2003** | **1.7478** | [0.199, 0.996] | 0.0000 |
| control · t1_3 | `t1_3` | 9554 | 4777 | 0.7641 | **0.1120** | **0.6466** | [0.477, 0.932] | 0.0000 |
| control · common support | `common support` | 8759 | 4649 | 0.8469 | **0.1426** | **1.4096** | [0.466, 0.994] | 0.0000 |

The COMMON-SUPPORT window is `V ∈ [0.3599, 0.9958]` — the intersection of the two sides' central 95% of `V`. Both sides are re-fitted inside it, so the lever arm cannot differ between them and a surviving slope difference is not the support.

| row | frame | arm | control | **Δ** | 95% CI | replicate floor | verdict |
|---|---|---|---|---|---|---|---|
| calibration SLOPE · `all states, <=2 per battle` | as traced | +1.2970 | +1.0719 | **+0.2251** | [+0.1168, +0.3399] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large · `all states, <=2 per battle` | as traced | -0.3850 | +0.4257 | **-0.8107** | [-0.9962, -0.6383] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE · `turn 1-3, <=2 per battle` | as traced | +0.9440 | +0.8379 | **+0.1061** | [-0.0871, +0.3117] | — | **NOT DETECTED** (CI covers zero) |
| calibration-in-the-large · `turn 1-3, <=2 per battle` | as traced | +0.5892 | +0.9122 | **-0.3230** | [-0.5838, -0.0564] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM · `all states, <=2 per battle, stratum fixed effects` | MATCHED 399/169/5 | +1.2351 | +1.0434 | **+0.1917** | [+0.0802, +0.3139] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration SLOPE on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | +1.2983 | +1.0281 | **+0.2702** | [+0.1449, +0.3926] | — | **DETECTED** (vs ZERO — NO FLOOR) |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | as traced | -0.3885 | +0.4871 | **-0.8756** | [-1.0870, -0.6652] | — | **DETECTED** (vs ZERO — NO FLOOR) |

> 🚨 **A WIDE REPLICATE FLOOR IS 'UNREADABLE AT THIS FRAME SIZE', NOT A NULL.** The floor column is the wider of the two control-vs-control draws for that row — total run-to-run variance between two identically-configured runs. Where it is larger than the arm's own delta the row says the frame cannot resolve the question; it does not say the effect is absent. A `—` means the floor file carries no entry for the row and no detection against a floor is possible.

> 🚨 **the calibration slope's standard error scales as 1/sd(logit V), so a head whose predictions are COMPRESSED gets a wider interval from the very effect under test — a conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the intersection of their central 95% of V, which removes the lever-arm difference by construction.**

> The **within-stratum** row is the same fit with a free intercept per own-team strength stratum. Shrinkage ACROSS teams and shrinkage INSIDE one are different statements: a head compressed only between strata moves the pooled row alone, while one compressed everywhere moves both. A stratum whose outcomes are all wins or all losses is DROPPED rather than fitted — its own dummy would diverge and take the shared slope's convergence with it.

> `V` is clipped into `[0.001, 0.999]` before the logit — `logit(0)` is not a number, and a forecast at 0.9999 is a leverage point worth several ordinary states on a logit x-axis. The clipped SHARE is in the lever-arm table above.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.0843 [+0.1769, +0.1835] | +0.1799 | **-0.0956** | [-0.1330, -0.0594] | **NOT DETECTED** (CI does not clear the floor 0.0682) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0843 [+0.1804, +0.1804] | +0.1804 | **-0.0962** | [-0.1331, -0.0615] | **NOT DETECTED** (CI does not clear the floor 0.0682) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.0843 | +0.1804 | **-0.0962** | [-0.1331, -0.0615] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.4896 [+0.6566, +0.6734] | +0.6626 | **-0.1730** | [-0.2759, -0.0713] | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.4896 [+0.6644, +0.6644] | +0.6644 | **-0.1748** | [-0.2845, -0.0731] | **WITHIN FLOOR** (|delta| <= floor 0.2385) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.4896 | +0.6644 | **-0.1748** | [-0.2845, -0.0731] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.0224 [+0.0334, +0.0425] | +0.0364 | **-0.0141** | [-0.0286, +0.0002] | **WITHIN FLOOR** (|delta| <= floor 0.0164) |
| `cond.own_team_r2.t1` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0224 [+0.0376, +0.0376] | +0.0376 | **-0.0152** | [-0.0306, -0.0007] | **WITHIN FLOOR** (|delta| <= floor 0.0164) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | +0.0224 | +0.0376 | **-0.0152** | [-0.0306, -0.0007] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.0031 [+0.0033, +0.0084] | +0.0047 | **-0.0016** | [-0.0068, +0.0037] | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| `cond.own_team_r2.all` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0031 [+0.0048, +0.0048] | +0.0048 | **-0.0017** | [-0.0074, +0.0037] | **WITHIN FLOOR** (|delta| <= floor 0.0081) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0031 | +0.0048 | **-0.0017** | [-0.0074, +0.0037] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.5020 [+0.5186, +0.5244] | +0.5210 | **-0.0191** | [-0.0458, +0.0080] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.5020 [+0.5137, +0.5137] | +0.5137 | **-0.0117** | [-0.0384, +0.0144] | **WITHIN FLOOR** (|delta| <= floor 0.0118) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5020 | +0.5137 | **-0.0117** | [-0.0384, +0.0144] | *no label — not a reading* |
| `cond.opp_class_auc.t1_3` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.5579 [+0.6048, +0.6089] | +0.6067 | **-0.0487** | [-0.0724, -0.0260] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.opp_class_auc.t1_3` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.5579 [+0.6005, +0.6005] | +0.6005 | **-0.0426** | [-0.0657, -0.0199] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.opp_class_auc.t1_3` | UNMATCHED (as traced) | +0.5579 | +0.6005 | **-0.0426** | [-0.0657, -0.0199] | *no label — not a reading* |
| `cond.opp_class_auc.t4_10` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.6940 [+0.7153, +0.7184] | +0.7161 | **-0.0222** | [-0.0421, -0.0014] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.opp_class_auc.t4_10` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.6940 [+0.7169, +0.7169] | +0.7169 | **-0.0229** | [-0.0431, -0.0028] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.opp_class_auc.t4_10` | UNMATCHED (as traced) | +0.6940 | +0.7169 | **-0.0229** | [-0.0431, -0.0028] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.0155 [+0.0265, +0.0287] | +0.0275 | **-0.0120** | [-0.0178, -0.0065] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0155 [+0.0268, +0.0268] | +0.0268 | **-0.0113** | [-0.0172, -0.0062] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t1_3` | UNMATCHED (as traced) | +0.0155 | +0.0268 | **-0.0113** | [-0.0172, -0.0062] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.1131 [+0.1327, +0.1390] | +0.1350 | **-0.0219** | [-0.0439, -0.0002] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.1131 [+0.1354, +0.1354] | +0.1354 | **-0.0223** | [-0.0443, -0.0014] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t4_10` | UNMATCHED (as traced) | +0.1131 | +0.1354 | **-0.0223** | [-0.0443, -0.0014] | *no label — not a reading* |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.2010 [+0.1992, +0.2049] | +0.2011 | **-0.0002** | [-0.0342, +0.0346] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.2010 [+0.2009, +0.2009] | +0.2009 | **+0.0001** | [-0.0358, +0.0336] | **PROVISIONAL — no floor; never DETECTED** (a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE `cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's 0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is between-opponent spread that rides on BOARD STATE correlated with the opponent rather than on opponent identity. A delta between two sides is a difference in how much opponent IDENTITY each side's output carries; it is reported, and it is never DETECTED) |
| `cond.spread_ratio_optimal.t11_24` | UNMATCHED (as traced) | +0.2010 | +0.2009 | **+0.0001** | [-0.0358, +0.0336] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.0542 [+0.0579, +0.0603] | +0.0591 | **-0.0049** | [-0.0098, +0.0019] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| `cond.within_team_resolution.all` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0542 [+0.0595, +0.0595] | +0.0595 | **-0.0052** | [-0.0107, +0.0012] | **WITHIN FLOOR** (|delta| <= floor 0.0056) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0542 | +0.0595 | **-0.0052** | [-0.0107, +0.0012] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.0314 [+0.0294, +0.0323] | +0.0310 | **+0.0003** | [-0.0048, +0.0053] | **WITHIN FLOOR** (|delta| <= floor 0.0028) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0314 [+0.0305, +0.0305] | +0.0305 | **+0.0009** | [-0.0042, +0.0056] | **WITHIN FLOOR** (|delta| <= floor 0.0028) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0314 | +0.0305 | **+0.0009** | [-0.0042, +0.0056] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +1.0013 [+1.7265, +3.1474] | +2.0094 | **-1.0082** | [-0.3017, -0.0871] | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +1.0013 [+2.0616, +2.0616] | +2.0616 | **-1.0603** | [-0.3082, -0.1006] | **WITHIN FLOOR** (|delta| <= floor 1.3970) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +1.0013 | +2.0616 | **-1.0603** | [-0.3082, -0.1006] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.3779 [+0.5629, +0.5822] | +0.5675 | **-0.1897** | [-0.1883, -0.0998] | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.3779 [+0.5732, +0.5732] | +0.5732 | **-0.1954** | [-0.1891, -0.1049] | **WITHIN FLOOR** (|delta| <= floor 0.2470) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.3779 | +0.5732 | **-0.1954** | [-0.1891, -0.1049] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.0011 [-0.0033, +0.0044] | +0.0003 | **+0.0007** | [-0.0072, +0.0089] | **WITHIN FLOOR** (|delta| <= floor 0.0057) |
| `cond.own_team_r2.late` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0011 [+0.0028, +0.0028] | +0.0028 | **-0.0017** | [-0.0100, +0.0064] | **WITHIN FLOOR** (|delta| <= floor 0.0057) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0011 | +0.0028 | **-0.0017** | [-0.0100, +0.0064] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +0.0213 [+0.0315, +0.0439] | +0.0365 | **-0.0152** | [-0.0315, +0.0004] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +0.0213 [+0.0348, +0.0348] | +0.0348 | **-0.0135** | [-0.0296, +0.0025] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | +0.0213 | +0.0348 | **-0.0135** | [-0.0296, +0.0025] | *no label — not a reading* |
| `cond.calibration_slope.within_stratum` | MATCHED · battle 399/169/5 (4757 battles, 4599 decoder battles, 21 seeds) | +1.2351 [+1.0234, +1.0559] | +1.0434 | **+0.1917** | [+0.0802, +0.3139] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.calibration_slope.within_stratum` | MATCHED · decoder 449/190/5 (4777 battles, 4622 decoder battles, 21 seeds) | +1.2351 [+1.0067, +1.0067] | +1.0067 | **+0.2284** | [+0.1164, +0.3455] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.calibration_slope.within_stratum` | UNMATCHED (as traced) | +1.2351 | +1.0067 | **+0.2284** | [+0.1164, +0.3455] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0800 | [+0.0636, +0.1054] | +0.1774 | [+0.1491, +0.2145] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.0843 | [+0.0686, +0.1087] | +0.1804 | [+0.1527, +0.2168] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1056 | [-0.1188, -0.0930] | -0.0987 | [-0.1130, -0.0855] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.4926 | [+0.4336, +0.5602] | +0.6680 | [+0.5832, +0.7618] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.4896 | [+0.4318, +0.5558] | +0.6644 | [+0.5812, +0.7564] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0582 | [-0.0723, -0.0458] | -0.0398 | [-0.0549, -0.0259] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 4-10` | +0.2816 | [+0.2455, +0.3212] | +0.5451 | [+0.4783, +0.6200] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 11-24` | +0.5217 | [+0.4582, +0.5973] | +0.7482 | [+0.6573, +0.8546] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | +0.0224 | [+0.0125, +0.0313] | +0.0376 | [+0.0259, +0.0493] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0031 | [-0.0006, +0.0061] | +0.0048 | [+0.0003, +0.0090] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5020 | [+0.4820, +0.5210] | +0.5137 | [+0.4959, +0.5325] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1-3` | +0.5579 | [+0.5414, +0.5734] | +0.6005 | [+0.5844, +0.6181] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 4-10` | +0.6940 | [+0.6796, +0.7077] | +0.7169 | [+0.7025, +0.7312] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 1-3` | +0.0155 | [+0.0131, +0.0189] | +0.0268 | [+0.0226, +0.0326] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 4-10` | +0.1131 | [+0.0999, +0.1284] | +0.1354 | [+0.1192, +0.1537] |
| OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above) · `turn 11-24` | +0.2010 | [+0.1780, +0.2266] | +0.2009 | [+0.1781, +0.2283] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0542 | [+0.0499, +0.0576] | +0.0595 | [+0.0543, +0.0629] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0314 | [+0.0284, +0.0355] | +0.0305 | [+0.0277, +0.0347] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +1.0013 | [+0.3326, +0.4442] | +2.0616 | [+0.5026, +0.6841] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.3779 | [+0.2746, +0.3212] | +0.5732 | [+0.4093, +0.4831] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0011 | [-0.0052, +0.0062] | +0.0028 | [-0.0038, +0.0079] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | +0.0213 | [+0.0112, +0.0323] | +0.0348 | [+0.0229, +0.0473] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) (1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed) · `all states, <=2 per battle` | +1.2970 | [+1.2186, +1.3860] | +1.0719 | [+1.0083, +1.1389] |
| calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated) · `all states, <=2 per battle` | -0.3850 | [-0.5341, -0.2479] | +0.4257 | [+0.3162, +0.5278] |
| calibration SLOPE — weighted logistic regression of the outcome on logit(V) · `turn 1-3, <=2 per battle` | +0.9440 | [+0.7962, +1.1121] | +0.8379 | [+0.7197, +0.9623] |
| calibration-in-the-large — the INTERCEPT of that regression · `turn 1-3, <=2 per battle` | +0.5892 | [+0.3646, +0.7944] | +0.9122 | [+0.7647, +1.0621] |
| calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the dispersion reading INSIDE a stratum rather than across teams · `all states, <=2 per battle, stratum fixed effects` | +1.2351 | [+1.1527, +1.3338] | +1.0067 | [+0.9405, +1.0788] |
| calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection of their central 95% of V, so the lever arm sd(logit V) cannot differ between them · `all states, <=2 per battle, common V window` | +1.2983 | [+1.2062, +1.3966] | +1.0281 | [+0.9434, +1.1179] |
| calibration-in-the-large on the COMMON SUPPORT · `all states, <=2 per battle, common V window` | -0.3885 | [-0.5542, -0.2298] | +0.4871 | [+0.3517, +0.6171] |

**Rows OMITTED, with the reason** — an unsupported meter is never emitted as a NaN that reads like a measurement:

- `cond.elo_slope` · arm: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.
- `cond.elo_slope` · control: 3 sentinel opponents and no eval_results.jsonl — the sentinel->snapshot map cannot be built, so a bot-only axis would compare two different populations. Row OMITTED.

⚠️ **Recorded `V`, one cycle.** every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model — no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles.

⚠️ **The rows above that are NOT frame-sensitive are read AS TRACED, and that is not an oversight.** The noise-corrected spread ratio, the spread delta, the Elo slope, every gate row and every identity row are weighted MEANS, differences of noise-corrected spreads, or an OLS over the pinned opponent roster. A weighted mean's expectation does not depend on how many states entered it given correct weights — which rule 17's capture-rate IPW supplies — so frame SIZE moves their variance and not their expectation, and equalising the frames would cost power without removing a bias.

⚠️ **No permutation null is run here.** The probe read's nulls on these very targets sit at R² ≈ 0.00 and AUC ≈ 0.50, so a near-zero own-team R² is a head that cannot be told from chance; but the DELTA is what this report licenses, and a per-run *detection* claim needs that measurement's own null, not this table.

## 4. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_23_ladder_rollout` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": true, "n_gated_rows": 9}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0277 [+0.0202, +0.0382] | 0.0618 | -0.0342 | +0.1964 [+0.1468, +0.2410] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0217 [+0.0140, +0.0333] | 0.0337 | -0.0119 | +0.2027 [+0.1340, +0.2649] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.0303 [+0.0195, +0.0471] | 0.0711 | -0.0408 | +0.1684 [+0.0877, +0.2305] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 685 battles / 6400 rollouts · Brier 0.0964 = REL 0.0011 − RES 0.0315 + UNC 0.1270 + WBV 0.0008 (resid -1.02e-03) · base rate 0.8507 · **resolution is 24.8% of the base-rate cap** · corr(turn,V) -0.1676 vs corr(turn,MC) -0.1180

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 678 battles / 6400 rollouts · Brier 0.1002 = REL 0.0036 − RES 0.0315 + UNC 0.1287 + WBV 0.0008 (resid -1.32e-03) · base rate 0.8482 · **resolution is 24.5% of the base-rate cap** · corr(turn,V) -0.1044 vs corr(turn,MC) -0.1168

## 5. THE LEDGER LINE

```
ai_v12_23_ladder_rollout vs ai_v12_11_ladder_ctrl10M at 10M [OFFLINE-GENERATED: 400 games x 9+3 opponents, full capture]: G1 bot Δ +0.0034 [-0.0032, +0.0104] NOT DETECTED · identity bias late Δ +0.0417 [-0.0103, +0.0937] WITHIN FLOOR · turn-contrast Δ -0.0620 [-0.2165, +0.0870] NOT DETECTED · spread ratio t1-3 Δ -0.0974 [-0.1353, -0.0616] NOT DETECTED · own-team R2 t1 Δ -0.0141 [-0.0286, +0.0002] WITHIN FLOOR [QUOTA-MATCHED] · calib slope Δ +0.2251 [+0.1168, +0.3399] DETECTED
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_23_ladder_rollout --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_23_ladder_rollout --control-traces /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_11_ladder_ctrl10M --floor-json /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json --out /home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_23_ladder_rollout_vs_ctrl10M --nice 15 --ledger-line

# arm — ai_v12_23_ladder_rollout  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_23_ladder_rollout` |
| arm trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_23_ladder_rollout/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_23_ladder_rollout_vs_lambda09/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_23_ladder_rollout_vs_lambda09/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_23_ladder_rollout_vs_lambda09/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_11_ladder_ctrl10M__offline/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_23_ladder_rollout_vs_ctrl10M/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/ai_v12_23_ladder_rollout_vs_ctrl10M/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
