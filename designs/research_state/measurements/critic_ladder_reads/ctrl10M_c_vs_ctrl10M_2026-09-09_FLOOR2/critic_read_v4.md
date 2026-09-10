# CRITIC READ — `ai_v12_16_ladder_ctrl10M_c` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v4 at 2026-09-10T07:36:30. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_16_ladder_ctrl10M_c` | `step_10000032` | 653 | 0.4% | 149/150 (99.3%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 204 | 0.5% | 140/140 (100.0%) | no |

**The POPULATION these numbers describe** — every conditioning and identity row is a statistic OF the traced frame, so the frame is part of the reading.

| role | population |
|---|---|
| arm | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |
| control | LIVE cycle: 12 opponents (9 scripted bots + 3 pool sentinels) x 100 games, traced under the per-opponent OUTCOME QUOTA (loss-enriched, not a random subsample) |

**Which cycle, and WHY** — a read that silently took the previous cycle is a read of a different model than the caller believes (backlog 2026-09-09: an arm whose launcher process was still alive was read one cycle back, and nothing said so). `--step N` pins it.

- **arm** `ai_v12_16_ladder_ctrl10M_c` → `step_10000032`: PINNED by --step 10000032
- **control** `ai_v12_11_ladder_ctrl10M` → `step_10000032`: PINNED by --step 10000032

### REALIZED capture profile — what is actually on disk, per opponent

| role | run | cap W/L/D per opponent | traced battles | W / L / D | opponents |
|---|---|---|---|---|---|
| arm | `ai_v12_16_ladder_ctrl10M_c` | **40/39/2** | 653 | 480 / 168 / 5 | 12 |
| control | `ai_v12_11_ladder_ctrl10M` | **8/12/3** | 204 | 96 / 102 / 6 | 12 |

**Frames:** the realized per-opponent caps differ — arm 40/39/2, control 8/12/3 (traced W/L/D per opponent, counted on disk); arm cut to 8/12/2

> ⚖️ **UNEQUAL FRAMES — MATCHED.** The arm side is subsampled to caps **8/12/2** per opponent over 21 seeded draws, with the capture rates RECOMPUTED for each subsample so rule 17 still holds on the view actually read. The FRAME-SENSITIVE rows' labels are decided on the MATCHED delta; the as-traced value is printed beside them marked UNMATCHED and is never labelled. Nothing is copied or written — the subsamples are in-memory battle lists.

> 🚨 **NO REPLICATE FLOOR — deltas are NOT DETECTED unless their CI clears zero, and never DETECTED against a floor.** The ladder's replicate floor is the control-vs-control difference and does not exist until a second control replicate does. Every DETECTED below is a detection against ZERO and carries that qualifier.

## SUMMARY — the headline deltas

| quantity | frame | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| resolution · `bot` | — | +0.0177 | +0.0187 | **-0.0010** | [-0.0158, +0.0106] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | — | -0.0110 | -0.0198 | **+0.0088** | [-0.0702, +0.0852] | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) · `ALL` | — | -0.0267 | -0.0940 | **+0.0673** | [-0.1615, +0.2725] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | — | +0.1467 | +0.1372 | **+0.0095** | [-0.1628, +0.1773] | **NOT DETECTED** (CI covers zero) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | as traced | +0.0896 | +0.1152 | **-0.0256** | [-0.5008, +0.0935] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/2 | -0.0626 | -0.0237 | **-0.0389** | [-0.1482, +0.0349] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock. The **spread ratio**'s target is ONE (for any calibrated critic the between-opponent spread of `V` equals that of the outcome), so a POSITIVE delta from a control below 1.0 is an improvement; the **own-team R²** and the **opponent-class AUC** are decodes FROM `V`, higher is better, with 0.0 and 0.5 the respective chance levels. The **within-team / within-stratum resolution** is DISCRIMINATION INSIDE a cell — higher is better, 0.0 is a forecast that says the same thing about every state of a team — and the **between-team spread ratio** is the own-team analogue of the opponent identity, target ONE. Their signs are read TOGETHER, in the (A)/(B) table of section 3.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0459 | +0.0419 | **+0.0040** | [-0.0227, +0.0244] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0177 | +0.0187 | **-0.0010** | [-0.0158, +0.0106] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0596 | +0.0720 | **-0.0123** | [-0.0548, +0.0213] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0004 | +0.0022 | **-0.0018** | [-0.0086, +0.0004] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0051 | +0.0033 | **+0.0018** | [-0.0076, +0.0071] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0110 | +0.0025 | **+0.0085** | [-0.0096, +0.0214] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `all` | `capture-rate (gauge)` | +0.0125 | +0.0351 | **-0.0226** | [-0.0542, +0.0033] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `bot` | `capture-rate (gauge)` | +0.0549 | +0.0425 | **+0.0124** | [-0.0389, +0.0394] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `pool` | `capture-rate (gauge)` | +0.0949 | +0.0407 | **+0.0542** | [-0.0268, +0.0963] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.2817 | +0.2668 | **+0.0149** | [-0.1055, +0.1537] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.1467 | +0.1372 | **+0.0095** | [-0.1628, +0.1773] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.2144 | +0.3592 | **-0.1448** | [-0.2929, +0.0354] | 400 | **NOT DETECTED** (CI covers zero) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0621 | +0.0917 | **-0.0296** | [-0.0806, +0.0185] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `ALL` | `pop` | -0.0032 | +0.0306 | **-0.0338** | [-0.0737, +0.0066] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0272 | -0.0418 | **+0.0146** | [-0.0201, +0.0487] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0404 | +0.0987 | **-0.0583** | [-0.1162, -0.0019] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0220 | +0.0134 | **-0.0354** | [-0.0864, +0.0138] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0463 | -0.0703 | **+0.0239** | [-0.0325, +0.0753] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.0747 | +0.0889 | **-0.0142** | [-0.0665, +0.0388] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0077 | +0.0498 | **-0.0421** | [-0.0874, +0.0030] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0204 | -0.0286 | **+0.0082** | [-0.0274, +0.0421] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0746 | +0.0818 | **-0.0073** | [-0.1502, +0.1245] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0046 | +0.0297 | **-0.0250** | [-0.1321, +0.0836] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0110 | -0.0198 | **+0.0088** | [-0.0702, +0.0852] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `bot` | `raw` | +0.0593 | +0.1259 | **-0.0667** | [-0.1276, -0.0061] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `bot` | `pop` | -0.0122 | +0.0535 | **-0.0657** | [-0.1146, -0.0178] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0368 | -0.0456 | **+0.0088** | [-0.0301, +0.0452] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `pool` | `raw` | +0.0655 | +0.0277 | **+0.0379** | [-0.0403, +0.1127] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `pool` | `pop` | +0.0083 | -0.0134 | **+0.0217** | [-0.0543, +0.0995] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `pool` | `ipw` | -0.0109 | -0.0382 | **+0.0273** | [-0.0368, +0.0879] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `raw` | +0.0285 | +0.0364 | **-0.0079** | [-0.0253, +0.0082] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0440 | +0.0650 | **-0.0210** | [-0.0434, +0.0008] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0315 | +0.0300 | **+0.0016** | [-0.0167, +0.0169] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1250 | +0.1172 | **+0.0079** | [-0.0943, +0.1209] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `pop` | +0.2689 | +0.2892 | **-0.0203** | [-0.1263, +0.0866] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2372 | +0.2296 | **+0.0076** | [-0.1206, +0.1423] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1484 | +0.1606 | **-0.0122** | [-0.0920, +0.0639] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.2724 | +0.2935 | **-0.0212** | [-0.1217, +0.0774] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2544 | +0.2482 | **+0.0062** | [-0.1096, +0.1045] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0058 | +0.0113 | **-0.0055** | [-0.0174, +0.0023] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `pop` | +0.0010 | +0.0022 | **-0.0012** | [-0.0071, +0.0024] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0021 | +0.0038 | **-0.0017** | [-0.0086, +0.0029] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.0267 | -0.0940 | **+0.0673** | [-0.1615, +0.2725] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0104 | -0.0990 | **+0.0886** | [-0.1120, +0.2704] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0273 | +0.0004 | **+0.0269** | [-0.1904, +0.2471] | 4000 | **NOT DETECTED** (CI covers zero) |

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

**arm — `ai_v12_16_ladder_ctrl10M_c` @ `step_10000032`**: 20617 states / 648 battles / 12 opponents / 198 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 5 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

**control — `ai_v12_11_ladder_ctrl10M` @ `step_10000032`**: 6080 states / 198 battles / 12 opponents / 76 trainee teams · manifest `selection_schema` 2 · `max|values − win_probs|` = 0.0 · 6 draw/timeout battles excluded (no binary outcome) · strength axis: bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder (4 nodes rated, anchored_to_bots=true), the positional sentinel map verified against the manifest's own win counts

| quantity | stratum | frame | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected ⭐ | `turn 1-3` | as traced | +0.0896 | +0.1152 | **-0.0256** | [-0.5008, +0.0935] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/2 | +0.1777 | +0.3333 | **-0.1556** | [-0.4376, +0.0174] | 2000 | **NOT DETECTED** (CI covers zero) |
| sd(V) - sd(outcome), noise-corrected | `turn 1-3` | as traced | -0.1277 | -0.0886 | **-0.0391** | [-0.1027, -0.0122] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected | `all states` | as traced | +0.5072 | +0.7895 | **-0.2823** | [-0.6801, -0.0198] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| the same ratio UNCORRECTED and unclamped | `all states` | MATCHED 8/12/2 | +0.5125 | +0.8053 | **-0.2928** | [-0.6536, +0.0060] | 2000 | **NOT DETECTED** (CI covers zero) |
| sd(V) - sd(outcome), noise-corrected | `all states` | as traced | -0.0691 | -0.0211 | **-0.0480** | [-0.0975, -0.0067] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo | `all states` | as traced | +0.0224 | +0.0098 | **+0.0126** | [-0.0002, +0.0260] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V ⭐ | `turn 1` | MATCHED 8/12/2 | -0.0626 | -0.0237 | **-0.0389** | [-0.1482, +0.0349] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V | `all states` | MATCHED 8/12/2 | -0.0410 | -0.0098 | **-0.0312** | [-0.1919, +0.0424] | 2000 | **NOT DETECTED** (CI covers zero) |
| opponent-CLASS (pool vs bot) AUC of V | `turn 1` | MATCHED 8/12/2 | +0.3846 | +0.4619 | **-0.0773** | [-0.2203, +0.0686] | 2000 | **NOT DETECTED** (CI covers zero) |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) | `all states, <=2 per battle` | MATCHED 8/12/2 | +0.0888 | +0.0839 | **+0.0050** | [-0.0567, +0.0419] | 2000 | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) | `all states, <=2 per battle` | MATCHED 8/12/2 | +0.0703 | +0.0819 | **-0.0116** | [-0.0639, +0.0398] | 2000 | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected | `turn 1-3` | MATCHED 8/12/2 | +0.2430 | +0.3973 | **-0.1542** | [-0.3038, +0.0857] | 1961 | **NOT DETECTED** (CI covers zero) |
| the same ratio UNCORRECTED and unclamped | `turn 1-3` | MATCHED 8/12/2 | +0.2178 | +0.3573 | **-0.1395** | [-0.2352, -0.0319] | 2000 | **DETECTED** (vs ZERO — NO FLOOR) |
| own-team leave-one-battle-out win-rate R^2 of V | `turn >= 25` | MATCHED 8/12/2 | -0.0912 | -0.0507 | **-0.0405** | [-0.5003, +0.3526] | 2000 | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late | `turn 1 - turn >= 25` | MATCHED 8/12/2 | +0.0210 | +0.0270 | **-0.0060** | [-0.3791, +0.2124] | 2000 | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

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
| arm · team | 50 | 198 | 433 | 866 | 8.0 | 16.0 | 4–23 |
| arm · stratum | 5 | 198 | 433 | 866 | 87.0 | 174.0 | 81–91 |
| arm · between-team spread | 50 | 198 | 433 | — | 8.0 | — | — |
| control · team | 15 | 76 | 104 | 208 | 5.0 | 10.0 | 4–17 |
| control · stratum | 5 | 76 | 104 | 208 | 20.0 | 40.0 | 15–27 |
| control · between-team spread | 15 | 76 | 104 | — | 5.0 | — | — |

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | MATCHED 8/12/2 | +0.0888 | +0.0839 | **+0.0050** | [-0.0567, +0.0419] | **NOT DETECTED** (CI covers zero) |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | MATCHED 8/12/2 | +0.0703 | +0.0819 | **-0.0116** | [-0.0639, +0.0398] | **NOT DETECTED** (CI covers zero) |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | MATCHED 8/12/2 | +0.2430 | +0.3973 | **-0.1542** | [-0.3038, +0.0857] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | MATCHED 8/12/2 | -0.0626 | -0.0237 | **-0.0389** | [-0.1482, +0.0349] | **NOT DETECTED** (CI covers zero) |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | MATCHED 8/12/2 | -0.0912 | -0.0507 | **-0.0405** | [-0.5003, +0.3526] | **NOT DETECTED** (CI covers zero) |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | MATCHED 8/12/2 | +0.0210 | +0.0270 | **-0.0060** | [-0.3791, +0.2124] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |

The two own-team R² rows are printed with them: the contrast is their difference, and it cannot be read without seeing which end moved.

**Reading of the three signs:** neither cleanly — the WITHIN-team discrimination is up without a between-team spread to go with it, which is a board-reading gain, not a conditioning one.

> 🚨 **The between-team spread is an AMPLITUDE; the own-team R² is an ALIGNMENT.** The spread ratio compares `sd(mean V per team)` with `sd(that team's win rate)` — how far apart the head's per-team opinions are. The own-team R² is an out-of-fold MONOTONE decode, invariant to scale — whether those opinions are in the right ORDER. The two can move in opposite directions (a head that orders its teams correctly but under-disperses reads R² UP and spread DOWN), so the sign table above is read with both rows in hand and the R² row is never read alone.

> 🚨 **The per-TEAM cells are small and the coarse row is the check on them.** With ~719 teams in the pool a few-thousand-battle frame leaves a handful of episodes per team, and a binned resolution inside a cell that size is largely the binning's own noise — which is *positively* biased, so a small per-team number is evidence of neither reading. The STRATUM row is the same estimator on cells hundreds of episodes deep. **Where the two disagree, believe the stratum row and say so.**

> 🚨 **`cond.own_team_r2.t1_minus_late` is PROVISIONAL and is never labelled DETECTED.** NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED.

### The FRAME-SENSITIVE rows, on the equalised frames

A row is matched because its ESTIMATOR moves with the size of the frame it is computed on, not because of anything in its name — the flag is declared on the meter (`main.ops.conditioning_meters.Meter.frame_sensitive`). Two mechanisms qualify: an out-of-fold score of a decoder **FIT** on the frame, and an **UNCORRECTED** second moment whose unsubtracted noise grows as the cells shrink. Everything else in the section above is a weighted mean, a difference of noise-corrected spreads, or an OLS over the pinned opponent roster — statistics whose expectation does not move with frame size given correct weights — and is read AS TRACED.

| row | frame | arm | control | **Δ** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| `cond.spread_ratio_raw.t1_3` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | +0.1777 [+0.1304, +0.2120] | +0.3333 | **-0.1556** | [-0.4376, +0.0174] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.t1_3` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | +0.1670 [+0.1375, +0.2302] | +0.3333 | **-0.1663** | [-0.4724, -0.0452] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.1154 | +0.3333 | **-0.2179** | [-0.5296, -0.1313] | *no label — not a reading* |
| `cond.spread_ratio_raw.all` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | +0.5125 [+0.4490, +0.5937] | +0.8053 | **-0.2928** | [-0.6536, +0.0060] | **NOT DETECTED** (CI covers zero) |
| `cond.spread_ratio_raw.all` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | +0.5279 [+0.4625, +0.5668] | +0.8053 | **-0.2774** | [-0.6345, -0.0019] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.spread_ratio_raw.all` | UNMATCHED (as traced) | +0.5046 | +0.8053 | **-0.3008** | [-0.6623, -0.0515] | *no label — not a reading* |
| `cond.own_team_r2.t1` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | -0.0626 [-0.1668, -0.0038] | -0.0237 | **-0.0389** | [-0.1482, +0.0349] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | -0.0240 [-0.1088, +0.0236] | -0.0237 | **-0.0003** | [-0.0745, +0.0767] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.t1` | UNMATCHED (as traced) | -0.0004 | -0.0237 | **+0.0233** | [-0.0005, +0.1079] | *no label — not a reading* |
| `cond.own_team_r2.all` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | -0.0410 [-0.1960, +0.0916] | -0.0098 | **-0.0312** | [-0.1919, +0.0424] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | -0.0166 [-0.0622, +0.0902] | -0.0098 | **-0.0068** | [-0.0968, +0.0702] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.all` | UNMATCHED (as traced) | +0.0722 | -0.0098 | **+0.0820** | [+0.0136, +0.1721] | *no label — not a reading* |
| `cond.opp_class_auc.t1` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | +0.3846 [+0.3318, +0.5416] | +0.4619 | **-0.0773** | [-0.2203, +0.0686] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | +0.4416 [+0.3009, +0.5772] | +0.4619 | **-0.0203** | [-0.1536, +0.1093] | **NOT DETECTED** (CI covers zero) |
| `cond.opp_class_auc.t1` | UNMATCHED (as traced) | +0.5203 | +0.4619 | **+0.0584** | [-0.0512, +0.1733] | *no label — not a reading* |
| `cond.within_team_resolution.all` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | +0.0888 [+0.0530, +0.1278] | +0.0839 | **+0.0050** | [-0.0567, +0.0419] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | +0.0926 [+0.0672, +0.1101] | +0.0839 | **+0.0088** | [-0.0404, +0.0504] | **NOT DETECTED** (CI covers zero) |
| `cond.within_team_resolution.all` | UNMATCHED (as traced) | +0.0649 | +0.0839 | **-0.0190** | [-0.0362, +0.0277] | *no label — not a reading* |
| `cond.within_stratum_resolution.all` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | +0.0703 [+0.0338, +0.1080] | +0.0819 | **-0.0116** | [-0.0639, +0.0398] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | +0.0717 [+0.0475, +0.0949] | +0.0819 | **-0.0102** | [-0.0462, +0.0446] | **NOT DETECTED** (CI covers zero) |
| `cond.within_stratum_resolution.all` | UNMATCHED (as traced) | +0.0328 | +0.0819 | **-0.0491** | [-0.0822, -0.0105] | *no label — not a reading* |
| `cond.team_spread_ratio.t1_3` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | +0.2430 [+0.1698, +0.2963] | +0.3973 | **-0.1542** | [-0.3038, +0.0857] | **NOT DETECTED** (CI covers zero) |
| `cond.team_spread_ratio.t1_3` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | +0.2425 [+0.1819, +0.7007] | +0.3973 | **-0.1548** | [-0.2839, +0.0269] | **NOT DETECTED** (CI covers zero) |
| `cond.team_spread_ratio.t1_3` | UNMATCHED (as traced) | +0.4044 | +0.3973 | **+0.0072** | [-0.1753, +0.1406] | *no label — not a reading* |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | +0.2178 [+0.1688, +0.3770] | +0.3573 | **-0.1395** | [-0.2352, -0.0319] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.team_spread_ratio_raw.t1_3` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | +0.2192 [+0.1720, +0.3039] | +0.3573 | **-0.1382** | [-0.2340, -0.0041] | **DETECTED** (vs ZERO — NO FLOOR) |
| `cond.team_spread_ratio_raw.t1_3` | UNMATCHED (as traced) | +0.3171 | +0.3573 | **-0.0402** | [-0.1552, +0.0465] | *no label — not a reading* |
| `cond.own_team_r2.late` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | -0.0912 [-0.3429, +0.2390] | -0.0507 | **-0.0405** | [-0.5003, +0.3526] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | -0.0288 [-0.1963, +0.1287] | -0.0507 | **+0.0219** | [-0.3811, +0.4659] | **NOT DETECTED** (CI covers zero) |
| `cond.own_team_r2.late` | UNMATCHED (as traced) | +0.0862 | -0.0507 | **+0.1370** | [+0.0618, +0.6003] | *no label — not a reading* |
| `cond.own_team_r2.t1_minus_late` | MATCHED · battle 8/12/2 (208 battles, 67 decoder battles, 21 seeds) | +0.0210 [-0.3355, +0.3035] | +0.0270 | **-0.0060** | [-0.3791, +0.2124] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | MATCHED · decoder 10/15/2 (243 battles, 100 decoder battles, 21 seeds) | +0.0147 [-0.1440, +0.1620] | +0.0270 | **-0.0124** | [-0.4246, +0.1405] | **PROVISIONAL — no floor; never DETECTED** (NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move either way is informative; a small one is not; and this row is never DETECTED) |
| `cond.own_team_r2.t1_minus_late` | UNMATCHED (as traced) | -0.0866 | +0.0270 | **-0.1137** | [-0.4976, -0.0311] | *no label — not a reading* |

The subsampled side's value is the **across-seed median** and the bracket beside it is the **2.5/97.5 across-seed spread**, i.e. how much the answer depends on WHICH battles the cut kept. The Δ's interval is the **median seed's own battle-clustered bootstrap** differenced against the other side's — the seed count is odd, so the median is an exact order statistic and the point and the interval describe the same draw. The label is decided on the BATTLE-matched rung; the DECODER-matched rung is printed beside it because `MIN_TEAM_BATTLES = 4` makes the own-team decoder's frame a nonlinear function of team diversity, so equal battle counts can leave the richer side's decoder with FEWER battles than the poorer side's (62 against 104 in the 2026-09-09 read) — battle-matching is then unfair to it.

Each run's OWN point and interval, so a delta is never the only number on the page:

| quantity | arm | 95% CI | control | 95% CI |
|---|---|---|---|---|
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `turn 1-3` | +0.0896 | [+0.0628, +0.1649] | +0.1152 | [+0.0000, +0.6098] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.1154 | [+0.0948, +0.1786] | +0.3333 | [+0.2703, +0.6582] |
| sd(V) - sd(outcome), noise-corrected · `turn 1-3` | -0.1277 | [-0.1540, -0.1006] | -0.0886 | [-0.1059, -0.0353] |
| between-opponent spread ratio sd(V)/sd(outcome), noise-corrected · `all states` | +0.5072 | [+0.4035, +0.6454] | +0.7895 | [+0.5725, +1.1703] |
| the same ratio UNCORRECTED and unclamped · `all states` | +0.5046 | [+0.4057, +0.6337] | +0.8053 | [+0.6030, +1.1480] |
| sd(V) - sd(outcome), noise-corrected · `all states` | -0.0691 | [-0.0996, -0.0425] | -0.0211 | [-0.0534, +0.0144] |
| slope of bias (V - true win rate) on opponent Elo, per 100 Elo · `all states` | +0.0224 | [+0.0143, +0.0307] | +0.0098 | [-0.0006, +0.0193] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn 1` | -0.0004 | [-0.0174, +0.0117] | -0.0237 | [-0.1122, -0.0067] |
| own-team leave-one-battle-out win-rate R^2 of V · `all states` | +0.0722 | [+0.0226, +0.1166] | -0.0098 | [-0.0867, +0.0421] |
| opponent-CLASS (pool vs bot) AUC of V · `turn 1` | +0.5203 | [+0.4722, +0.5718] | +0.4619 | [+0.3675, +0.5582] |
| WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, battle-weighted over teams) · `all states, <=2 per battle` | +0.0649 | [+0.0550, +0.0805] | +0.0839 | [+0.0413, +0.1005] |
| the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate) · `all states, <=2 per battle` | +0.0328 | [+0.0270, +0.0476] | +0.0819 | [+0.0483, +0.1175] |
| BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected · `turn 1-3` | +0.4044 | [+0.2325, +0.4536] | +0.3973 | [+0.2479, +0.4615] |
| the same ratio UNCORRECTED and unclamped · `turn 1-3` | +0.3171 | [+0.2180, +0.3384] | +0.3573 | [+0.2470, +0.4098] |
| own-team leave-one-battle-out win-rate R^2 of V · `turn >= 25` | +0.0862 | [+0.0010, +0.1599] | -0.0507 | [-0.4948, -0.0286] |
| own-team R^2 at turn 1 MINUS own-team R^2 late · `turn 1 - turn >= 25` | -0.0866 | [-0.1609, -0.0024] | +0.0270 | [-0.0119, +0.4322] |

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

Identity, capture-rate reweighted: 800 labels / 379 battles / 6400 rollouts · Brier 0.0946 = REL 0.0021 − RES 0.0315 + UNC 0.1240 + WBV 0.0008 (resid -8.41e-04) · base rate 0.8550 · **resolution is 25.4% of the base-rate cap** · corr(turn,V) -0.0648 vs corr(turn,MC) -0.0381

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 185 battles / 6400 rollouts · Brier 0.0930 = REL 0.0038 − RES 0.0300 + UNC 0.1208 + WBV 0.0008 (resid -2.34e-03) · base rate 0.8595 · **resolution is 24.8% of the base-rate cap** · corr(turn,V) -0.1227 vs corr(turn,MC) -0.0286

## 5. THE LEDGER LINE

```
ai_v12_16_ladder_ctrl10M_c vs ai_v12_11_ladder_ctrl10M at 10M: G1 bot Δ -0.0010 [-0.0158, +0.0106] NOT DETECTED · identity bias late Δ +0.0088 [-0.0702, +0.0852] NOT DETECTED · turn-contrast Δ +0.0673 [-0.1615, +0.2725] NOT DETECTED · spread ratio t1-3 Δ -0.0256 [-0.5008, +0.0935] NOT DETECTED · own-team R2 t1 Δ -0.0389 [-0.1482, +0.0349] NOT DETECTED [QUOTA-MATCHED]
```

## 6. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_16_ladder_ctrl10M_c --control ai_v12_11_ladder_ctrl10M --step 10000032 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_floor_c_v4

# arm — ai_v12_16_ladder_ctrl10M_c  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
# control — ai_v12_11_ladder_ctrl10M  [READOUT REUSED FROM CACHE]
  (no subprocess: the cached readout was reused)
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_16_ladder_ctrl10M_c/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_16_ladder_ctrl10M_c/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_floor_c_v4/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_floor_c_v4/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
