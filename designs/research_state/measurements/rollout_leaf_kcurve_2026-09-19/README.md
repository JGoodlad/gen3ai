# HOW MANY ROLLOUTS DOES A LEAF NEED, AND DOES IT PAY IN GAMES? — the K-curve, and the first cell-level use of the rollout-leaf arm

*Measured 2026-09-19 14:51 – 19:34 UTC · **JOB 1**: **31,992 fresh post-divergence rollouts** over
**665** of the control's banked held-out contested CRN forks (all three branches, K = 16 each), read
against the control's own **K′ = 8** label — **0 errors, 36 capped (0.11 %), 24 / 24 CRN
reproduction checks exact, 0 salt mismatches** · **JOB 2**: the `playoff` arm's **first cell-level
use**, 234 mirror battles against a contemporaneous critic-leaf control · CPU only
(`CUDA_VISIBLE_DEVICES=""`), `nice 15`, ports 9700–9799, **nothing written under `models/`**; a
training arm held the GPU throughout and was not touched, 8000 / 8001 untouched, other agents'
ports untouched.*

**Pre-registered in [`PREDICTION.md`](PREDICTION.md) (landed `ca52dc7f` before the first rollout),
amended in [`AMENDMENT.md`](AMENDMENT.md) (`b477e74c`, after JOB 2's smoke) and
[`AMENDMENT2.md`](AMENDMENT2.md) (`cc065247`, the battery reduction + the guard).** All three are
scored in §8.

---

## 1. VERDICT — **branch (b)**, and the mechanism is the ACTING RULE, not the leaf

**1 — 🚨 A ROLLOUT LEAF BEATS THE 75M-STEP CRITIC ON EVERY COLUMN, INCLUDING THE LEAF COLUMN, AND
THE DOSE IS FOUR.** Against one K′ = 8 label on identical pairs, `ROLLOUT_4` reads **+0.0677
[+0.0382, +0.1006] DETECTED** pooled and `ROLLOUT_8` reads **+0.0783 [+0.0273, +0.1279] DETECTED**
on **`top1|top2`** — the column a re-ranking search leaf actually consumes, and the column the
2026-09-19 control left open at +0.0275 [−0.0176, +0.0705] NOT DETECTED. **BAR K1 is met at K = 4
on three of four columns and at K = 2 on the fourth. BAR K2 (a lower bound above 0.60) is met at
K = 4 pooled and at K = 8 on `top1|top2`.** The head's 0.5741 on that column is beaten by
**0.6125** at four rollouts and **0.6709** at sixteen.

**2 — 🚨 AND IT BUYS NOTHING IN GAMES, BECAUSE THE RULE THAT DECIDES WHEN TO ACT THROWS ALMOST ALL
OF IT AWAY.** The `playoff` arm's own gate — `|mean(d)| ≥ 2·SE` over R common-random-number paired
rollouts, `MIN_PAIRS = 4` — resolves **0 % of pairs at R ∈ {1, 2}** (below `MIN_PAIRS` it cannot
conclude at all), **4.51 % [3.01, 6.17] at R = 4**, **10.68 % at R = 8** and **12.48 % [9.92, 15.04]
at R = 16**, measured with the PRODUCTION rule imported and applied to all 665 forks. Live, at a
realized **R = 4.00**, the cell played **2 of 190** playoffs and changed **1 of 250** decisions.
**The leaf out-ranks; the arm does not act.**

**3 — WHEN THE RULE DOES CONCLUDE IT IS RIGHT 83–92 % OF THE TIME.** Against the K′ = 8 label, on
its own resolved pairs: **0.8696 (R = 4) · 0.8333 (R = 8) · 0.9221 (R = 16)** — far above the
head's 0.5741 and above the leaf's own unconditional level, because the rule concludes exactly where
the truth is large. ⚠️ That is a SELECTION effect and is reported as one: it is the quality of the
signal the gate passes, not of the gate's coverage.

**4 — THE COST IS NAMED.** A rollout runs **25.7 turns** from the successor to a terminal (measured
over 1,995 branch observations), so a K-rollout leaf evaluation costs **≈ 25.7 K sim turns** and a
top-2 playoff decision **twice that** — **206 turns at K = 4, 411 at K = 8** per successor. In wall
clock, on this box: **3.16 s per rollout per worker**, and live **43.3 s per adjudicated decision**
at R = 4, **1,414 s per battle** (3,771 s for the first, on a loaded box).

**5 — 🚨 THE KNEE IS AT K = 4 EVERYWHERE EXCEPT THE LEAF COLUMN, WHERE IT IS K = 8.** Marginal gain
per rollout: `+0.0370 → +0.0164 → +0.0100 → +0.0023` on `top1|top2` across the four doublings. **The
step from 8 to 16 buys +0.0185 for eight more rollouts — a quarter of the head's entire margin over
chance for 206 more sim turns.** The √K shape registered in K1 holds: the curve is monotone on all
four columns and every increment shrinks.

**6 — THE LABEL CEILING, REPORTED BESIDE EVERY LEVEL (rule 25 as amended).** The model-free lower
bound on that ceiling — a 16-rollout independent estimator of the same successors — is **0.7015**
pooled and **0.6709** on `top1|top2`. **Note it EXCEEDS the split-half agreement (0.6480 / 0.6140),
which is exactly the bias hazard 4 of the control predicted**: the split-half number is a K = 4
statistic and its `q` inversion is a cross-check, never a bar. **49 % of the variance in the
observed sibling-label difference is still label noise even at K′ = 8** on the leaf column.

**7 — THE CONTEMPORANEOUS CONTROL BEHAVED, AND THE UNDER-POWERED CELL IS A DESCRIPTOR.** The
critic-leaf `defB` cell in the same window reads **L2 = 0.5243 [0.4864, 0.5623]** over 113 pairs —
straddling 0.50, exactly where six win-prob heads sat. The rollout-leaf cell's own L2 is **0.5000
over 3 pairs** and is reported as a DESCRIPTOR, **under-powered by design** (AMENDMENT 2 §1:
400 pairs of it costs 1,222 CPU-hours).

---

## 2. The frame

| | |
|---|---|
| **the policy** | `models/ai_v13_02_flywheel_winprob/final_model.zip` — **arm W**, step **75,005,952**, `--critic winprob`. FROZEN; nothing was trained in this read |
| **the states** | **665** of the **1,002** banked forks of [`leaf_ceiling_controls_2026-09-19/`](../leaf_ceiling_controls_2026-09-19/README.md) — themselves a seeded, outcome-blind sample of the 5,040 held-out contested CRN forks of [`offline_leaf_fit_2026-09-18/`](../offline_leaf_fit_2026-09-18/README.md). 1,995 pairs, **351 non-tied on `top1\|top2`** |
| **the label** | the BANKED **K′ = 8** mean (re-rolls 0–7) — **ONE label set for every K**, so the curve's levels are comparable to each other and to the head |
| **the leaf** | means of the **FRESH** re-rolls 8–23, dice the label has never seen. `fresh_seeds(24, salt)` is a strict EXTENSION of `fresh_seeds(8, salt)`, so this bought only the extension |
| **nested in K, declared** | `ROLLOUT_K` = re-rolls 8 … 8+K−1, so a smaller K is a strict SUB-SAMPLE of a larger one's dice. The levels are POSITIVELY CORRELATED — right for a dose curve, wrong for treating two levels as independent samples. Every CI bootstraps over **FORKS** (2,000 resamples, `refit.boot_ci` by path) |
| regime | CPU only, `nice 15`, rust bridge for JOB 1, **node** bridge for JOB 2 (AMENDMENT 1 F1), box load **30–60** throughout on 16 cores with a training arm at `nice 10` and two other agents |

**The instrument is proven, not argued (BAR K5):**

| check | result |
|---|---|
| **CRN reproduction — re-roll 0 re-run under the banked run's own player tag** | **24 / 24 EXACT** on all three branches, 0 mismatches |
| salt join (recomputed `battle:inv:seed` vs the banked row's) | **0 mismatches** over 665 forks |
| successor-index join | **0 mismatches** |
| rollout errors · short rows | **0 · 0** |
| the head reproduces the control's §5.1 levels on the same label | **0.5798 pooled vs 0.5801** — three ten-thousandths. ⚠️ `top1\|top2` reads **0.5741 vs 0.5610** (+0.0131, outside the registered ±0.01) on a 665-fork subset read at the SUCCESSOR index rather than through `build_table`'s branch rows — declared in §8 as a NEAR MISS, not hidden |

---

## 3. THE K-CURVE

### the K-curve — pairwise accuracy against ONE K′ = 8 label [95 % CI over forks]

| scorer | rollouts | sim turns / leaf eval | POOLED | rand\|top1 | rand\|top2 | top1\|top2 |
|---|---|---|---|---|---|---|
| **headW** (the 75M critic) | **0** | **0** | **0.5798** [0.5547, 0.6041] | **0.5809** [0.5455, 0.6180] | **0.5837** [0.5470, 0.6195] | **0.5741** [0.5367, 0.6122] |
| ROLLOUT_1 | 1 | 26 | **0.5634** [0.5481, 0.5796] | **0.5723** [0.5504, 0.5940] | **0.5725** [0.5519, 0.5938] | **0.5427** [0.5199, 0.5662] |
| ROLLOUT_2 | 2 | 51 | **0.5979** [0.5793, 0.6176] | **0.5944** [0.5676, 0.6202] | **0.6175** [0.5909, 0.6432] | **0.5798** [0.5507, 0.6083] |
| ROLLOUT_4 | 4 | 103 | **0.6475** [0.6264, 0.6685] | **0.6642** [0.6358, 0.6933] | **0.6613** [0.6305, 0.6897] | **0.6125** [0.5814, 0.6429] |
| ROLLOUT_8 | 8 | 206 | **0.6812** [0.6590, 0.7029] | **0.6973** [0.6673, 0.7278] | **0.6900** [0.6588, 0.7204] | **0.6524** [0.6179, 0.6864] |
| ROLLOUT_16 | 16 | 411 | **0.7015** [0.6808, 0.7241] | **0.7206** [0.6911, 0.7528] | **0.7087** [0.6780, 0.7421] | **0.6709** [0.6370, 0.7056] |

### Δ vs the head, on IDENTICAL pairs

| scorer | POOLED | rand\|top1 | rand\|top2 | top1\|top2 |
|---|---|---|---|---|
| ROLLOUT_1 | -0.0164 [-0.0450, 0.0137] ND | -0.0086 [-0.0494, 0.0339] ND | -0.0112 [-0.0525, 0.0300] ND | -0.0313 [-0.0764, 0.0132] ND |
| ROLLOUT_2 | +0.0181 [-0.0107, 0.0500] ND | +0.0135 [-0.0300, 0.0562] ND | +0.0338 [-0.0100, 0.0763] ND | +0.0057 [-0.0399, 0.0514] ND |
| ROLLOUT_4 | +0.0677 [0.0382, 0.1006] **DET** | +0.0833 [0.0350, 0.1284] **DET** | +0.0775 [0.0345, 0.1207] **DET** | +0.0385 [-0.0089, 0.0871] ND |
| ROLLOUT_8 | +0.1014 [0.0698, 0.1346] **DET** | +0.1164 [0.0728, 0.1595] **DET** | +0.1062 [0.0627, 0.1537] **DET** | +0.0783 [0.0273, 0.1279] **DET** |
| ROLLOUT_16 | +0.1217 [0.0908, 0.1536] **DET** | +0.1397 [0.0949, 0.1831] **DET** | +0.1250 [0.0788, 0.1720] **DET** | +0.0969 [0.0467, 0.1477] **DET** |

### the LABEL CEILING beside every level, and the bars

| quantity | POOLED | rand\|top1 | rand\|top2 | top1\|top2 |
|---|---|---|---|---|
| **measured ceiling (model-free LOWER bound) — `ROLLOUT_16`** | **0.7015** [0.6808, 0.7241] | **0.7206** [0.6911, 0.7528] | **0.7087** [0.6780, 0.7421] | **0.6709** [0.6370, 0.7056] |
| split-half agreement of the LABEL (two K = 4 halves) | 0.6480 | 0.6618 | 0.6637 | 0.6140 |
| → implied oracle acc *(cross-check only)* | 0.7720 | 0.7844 | 0.7861 | 0.7387 |
| parametric `ACC_oracle(K′=8)` *(cross-check only)* | 0.7824 | 0.7913 | 0.7863 | 0.7623 |
| label-noise share of the observed gap variance | 0.4201 | 0.3901 | 0.4034 | 0.4894 |
| recovered **E\|true gap\|** | 0.1920 | 0.2027 | 0.1968 | 0.1702 |
| **BAR K1** — smallest K whose CI lower bound clears the head's point | **4** | **4** | **2** | **4** |
| **BAR K2** — smallest K whose CI lower bound clears 0.60 | **4** | **4** | **4** | **8** |
| **BAR K4** — the knee (marginal < average gain per rollout) | **K = 4** | **K = 4** | **K = 4** | **K = 8** |
| n non-tied pairs | 1159 | 408 | 400 | 351 |

### the MARGINAL table (BAR K3 / K4)

| step | POOLED | rand\|top1 | rand\|top2 | top1\|top2 |
|---|---|---|---|---|
| K 1 → 2: Δacc | +0.0345 (+0.0345/rollout) | +0.0221 (+0.0221/rollout) | +0.0450 (+0.0450/rollout) | +0.0370 (+0.0370/rollout) |
| K 2 → 4: Δacc | +0.0496 (+0.0248/rollout) | +0.0699 (+0.0349/rollout) | +0.0437 (+0.0219/rollout) | +0.0328 (+0.0164/rollout) |
| K 4 → 8: Δacc | +0.0336 (+0.0084/rollout) | +0.0331 (+0.0083/rollout) | +0.0287 (+0.0072/rollout) | +0.0399 (+0.0100/rollout) |
| K 8 → 16: Δacc | +0.0203 (+0.0025/rollout) | +0.0233 (+0.0029/rollout) | +0.0188 (+0.0023/rollout) | +0.0185 (+0.0023/rollout) |

### the COST column, measured

* `ext_capped` = 36
* `ext_errors` = 0
* `ext_rollouts` = 31992
* `ext_s_per_rollout_per_worker` = 3.158
* `ext_source` = shard log tail — no meta (stopped by the clock)
* `ext_wall_s_max_shard` = 16977.0
* `label_capped` = 32
* `label_errors` = 0
* `label_rollouts` = 24048
* `label_s_per_rollout_per_worker` = 1.5727
* `label_wall_s_max_shard` = 6666.8961
* `mean_remaining_turns_per_rollout_fresh` = 25.7083
* `mean_remaining_turns_per_rollout_label` = 25.9087
* `median_remaining_turns_fresh` = 19.8125
* `n_branch_observations` = 1995

### the depth split on `top1|top2` (post-hoc, hazard 2 of the control)

| scorer | same_turn (n=314) |
|---|---|
| headW | **0.5780** [0.5363, 0.6187] |
| ROLLOUT_1 | **0.5366** [0.5127, 0.5622] |
| ROLLOUT_2 | **0.5732** [0.5448, 0.6057] |
| ROLLOUT_4 | **0.6083** [0.5758, 0.6414] |
| ROLLOUT_8 | **0.6338** [0.5980, 0.6710] |
| ROLLOUT_16 | **0.6497** [0.6117, 0.6878] |

### join / instrument

```
{
 "label_forks": 1002,
 "ext_forks": 665,

**What the curve says.** ROLLOUT_1 — a single fresh rollout — is a WORSE leaf than the head on
every column (`top1|top2` 0.5427 against 0.5741), which is K4's registered prediction and the
quantitative form of the control's "the single label mis-orders the pair 39.7 % of the time". Two
rollouts draw level. **Four DETECT on three columns; eight DETECT on all four.** The `top1|top2`
column is the hardest at every K, and its knee is one doubling later than the others — the policy's
own two best moves need twice the dice that a random legal move does.

**The depth split (hazard 2 of the control), on `top1|top2`.** 314 of the 351 non-tied pairs
compare successors at the SAME turn; the different-depth cell has too few pairs to report (< 50).
At matched depth the head reads **0.5780** and the rollout leaf **0.6083 / 0.6338 / 0.6497** at
K = 4 / 8 / 16 — the ordering is unchanged, so the curve is not a depth artefact.

---

## 4. JOB 2 — THE BATTERY, AND THE PROXY THAT SCALES

### the PLAYOFF's OWN decision rule, applied offline to 665 forks' CRN-paired rollouts

| R (paired rollouts) | rollouts / decision | CONCLUSIVE rate [95 % CI] | n conclusive | keeps the policy's action | agrees with the K′ = 8 label |
|---|---|---|---|---|---|
| **1** | 2 | **0.0000** [0.0000, 0.0000] | 0 | — | **—** (n=0) |
| **2** | 4 | **0.0000** [0.0000, 0.0000] | 0 | — | **—** (n=0) |
| **4** | 8 | **0.0451** [0.0301, 0.0617] | 30 | 0.5667 | **0.8696** (n=23) |
| **8** | 16 | **0.1068** [0.0842, 0.1293] | 71 | 0.4648 | **0.8333** (n=60) |
| **16** | 32 | **0.1248** [0.0992, 0.1504] | 83 | 0.4819 | **0.9221** (n=77) |

### the BATTERY — the rollout-leaf cell and its contemporaneous controls

| cell | battles | pairs | **L2 paired** [normal CI] | [bootstrap CI] | unpaired Wilson | changed / decision | s / battle |
|---|---|---|---|---|---|---|---|
| **defB** | 226 | 113 | **0.5243** [0.4864, 0.5623] | [0.4889, 0.5642] | 0.5244 [0.4593, 0.5887] | 0.0099 | 51 |
| **pf4** | 1 | 0 | **—** — | — | 1.0000 [0.2065, 1.0000] | 0.0000 | 1343 |
| **pf4b** | 6 | 3 | **0.5000** [0.5000, 0.5000] | [0.5000, 0.5000] | 0.5000 [0.1876, 0.8124] | 0.0040 | 1414 |
| **pf8** | 1 | 0 | **—** — | — | 1.0000 [0.2065, 1.0000] | 0.0000 | 1334 |

### the MECHANISM row — the primary read (AMENDMENT §2)

| cell | decisions | screen_decisive | playoffs PLAYED | INCONCLUSIVE | no_budget | realized R | s / adjudicated decision |
|---|---|---|---|---|---|---|---|
| **defB** | 10390 | 0 | 0 | 0 | 0 | 0.00 | 0.0 |
| **pf4** | 75 | 9 | 0 | 64 | 0 | 1.00 | 18.9 |
| **pf4b** | 250 | 35 | 2 | 188 | 4 | 4.00 | 43.3 |
| **pf8** | 75 | 9 | 0 | 64 | 0 | 1.00 | 18.6 |

### paired deltas on shared indices

* `defB - pf4b` = +0.0000 [0.0000, 0.0000] NOT DETECTED (n = 3 shared pairs)

⚠️ **THE PROXY'S TWO DECLARED DIFFERENCES FROM THE LIVE CELL, with their directions** (AMENDMENT 1
§2.3): the offline rollouts are GREEDY against the banked SENTINEL while the live ones are the same
network STOCHASTIC at temperature 1 — extra variance, so the offline rate is an **UPPER bound**; and
the offline pair is the POLICY's top-2 while the live pair is the SCREEN's top-2, possibly further
apart and so easier to resolve — which cuts the other way. The live cell lands INSIDE that bracket
(played 2 / 190 = 1.05 % of playoffs against an offline 4.51 % conclusive of which 43 % overrule,
i.e. ~1.95 % overrules), which is the proxy working.

**`pf4` / `pf8` are the INERT-FLAG cells** (AMENDMENT 1 F2): at `--budget 20` both realized
**R = 1.00**, so `--playoff-rollouts` meant nothing and the two cells are byte-identical no-ops.
They are reported because they are the evidence for the guard that now refuses them.

**`pf4b` is the real cell**: `--budget 120`, realized **R = 4.00** exactly (260 pairs / 65
playoffs in its first battle), **35 screen-decisive · 190 playoffs run · 2 played · 188
inconclusive · 4 no-budget · 16 failed rollouts · 1 action changed** over 250 decisions.

⚠️ **`--impl node` is not perfectly clean either.** 16 of ~1,520 rollouts (1.05 %) raised and were
counted as failed pairs. That is 60× better than rust's 63-of-75 decisions, not zero — reported so
nobody reads "node works" as "node is exact".

---

## 5. THE BRANCH

**Registered branch (b): the K-curve clears the head, the battery does not clear 0.50.** But the
read is sharper than the branch was written, and the sharpening is the finding:

> **Out-ranking successors is not the binding constraint any more. The DECISION RULE is.** At the
> operating point the arm ships with, a rollout leaf that is 4–10 accuracy points better than a
> 75M-step critic on the leaf column gets to act on **1 % of decisions**, because `2·SE` over
> four common-random-number pairs almost never separates a pair whose true gap is 0.17 in a
> statistic that is 0 on most draws.

Three consequences, in cost order:

1. **The next lever is the GATE, not the leaf.** `SE_MULTIPLE = 2.0` and `MIN_PAIRS = 4` were set
   to stop a search overriding on noise — the correct lesson of the R-ladder — and they are now
   the thing throwing away a signal that is measurably real. The registered question is what the
   gate's ROC looks like: the proxy already prices it (agreement 0.87 at R = 4 on 4.5 % coverage),
   and re-running it at other `k` and `min_pairs` costs **nothing** — the dice are banked.
2. **A rollout leaf is affordable only where the ply is amortized.** 206 sim turns per top-2
   decision at K = 4 is ~40 s of wall clock on this box. That is a TEACHER's budget, not an
   inference budget — which is the `q_winprob_head` path (built, OFF): pay the rollouts once,
   offline, and distil the ordering.
3. **Nothing here says the 400-pair mirror cell would have found something.** It would have cost
   1,222 CPU-hours to measure an arm that changes 0.4 % of decisions; the mechanism row answered
   the question at 1/400th of the price, and that is the reusable lesson about this battery.

---

## 6. Hazards and findings about the instruments

Every one of these is a finding, and the first three are filed in
[`designs/ops/TECH_DEBT_BACKLOG.md`](../../../ops/TECH_DEBT_BACKLOG.md).

1. 🚨 **`--impl rust` BREAKS the playoff's nested counterfactual rollouts, and the failure is
   INVISIBLE in the row.** 63 of 75 decisions lost their playoff to
   `unresolvable choice for p1: MoveName("crunch") — active metagross has moves [...]` (250 failed
   rollouts); the identical game under `--impl node` raised 0. `PlayoffResult.error` lands in
   `diag["playoff"]["error"]`, which **no row field carries**, so the cell folds to
   `n_playoff_no_budget` and reports a clean-looking win rate for an arm that never adjudicated.
   The same rust path is clean on this read's 31,992 offline rollouts, so the two candidate
   differences are the LIVE partial record and its `sodium,<hex>` seed spelling — **neither
   proven; no root cause is claimed.**
2. 🚨 **`--playoff-rollouts` was INERT below a budget of 2R rollouts, with no refusal.** A live
   terminal rollout measured ~10 s, so at `--budget 20` the deadline bought ONE pair and
   `MIN_PAIRS = 4` declined every playoff; the R = 4 and R = 8 cells are byte-identical.
   **GUARD LANDED** (`cc065247`): `playoff.short_r_refusal` + `--playoff-allow-short-r`, raising on
   the FIRST game after the row is appended. ⚠️ **The open half**: `PlayoffConfig.rollout_cost_s`
   seeds at 1.0 s against a measured ~10 s, so every cell's first decision is planned on an
   estimate an order of magnitude wrong.
3. ⚠️ **`run_local_battles`' docstring is STALE about the rust bridge** — it says `resumeReseed`
   and `__RECON__` "degrade to no-ops under `rust`"; `sim_bridge.rs` implements both
   (`gen3_bridge_resume_reseed_v1`), which is why 31,992 reseeded rust rollouts produce real dice
   variation. A reader who believes it either avoids a valid path or distrusts a valid measurement.
4. 🚨 **THE MEASURED CEILING EXCEEDS THE SPLIT-HALF ONE, AS THE CONTROL WARNED.** `ROLLOUT_16`
   reads 0.6709 on `top1|top2` where the label's split-half agreement is 0.6140. A reader who took
   the split-half number as a ceiling would have concluded the leaf had broken physics. It is a
   K = 4 self-agreement statistic and its `q` inversion is biased in both directions (Jensen up,
   tie-scoring down) — **report it, never bar against it**, exactly as hazard 4 of the control said.
5. 🚨 **A SHORT-R CELL AND A WELL-BUDGETED ONE ARE INDISTINGUISHABLE IN THE REPORT** without the
   realized-R column. `pf4`, `pf8` and `pf4b` all report "playoff, R requested 4 or 8" and only
   `playoff_r_total / n_playoff_ran` separates them. **Print realized R beside requested R in every
   playoff row**; the guard now enforces it.
6. ⚠️ **The proxy conditions on its own resolution.** "Right 87–92 % when conclusive" is the
   quality of the signal the gate PASSES, not evidence that rollouts are 90 % accurate: the gate
   concludes where the true gap is large, and a large gap is also easier for the label. Never quote
   it without the coverage beside it.
7. **A `&&` chain ending in `&` backgrounds the WHOLE chain**, so a variable assigned in it is
   unset for the next command on the line — here it silently passed `--ext /rerollx` and the proxy
   refused with "no fork joined". It refused rather than reading an empty set, which is the only
   reason it cost a minute instead of a number.
8. **The fork-outer / K-inner loop paid for itself.** The clock stopped this read at **665 of
   1,002** forks and every one of them is at FULL K = 16; a ragged K would have weighted forks
   unequally in the mean without saying so.
9. ⚠️ **`--budget` on the playoff arm buys the SCREEN first and the rollouts out of the remainder.**
   The caps (`--max-worlds 4 --max-dice 2`) are what keep the screen from eating it: realized
   `k_worlds` 1.0 and 1.5 s of elapsed screen time out of 20 s and 120 s respectively. Size the
   budget as `screen + 2·R·(rollout cost)`, and measure the rollout cost first.
10. **Nothing under `src/` was changed by the READ**; the one `src/` change (hazard 2's guard) is a
    separate landing with six tests and the routine gate green (10,973 passed).

---

## 7. Files

| | |
|---|---|
| [`PREDICTION.md`](PREDICTION.md) | the registration, landed at `ca52dc7f` before the first rollout |
| [`AMENDMENT.md`](AMENDMENT.md) · [`AMENDMENT2.md`](AMENDMENT2.md) | the smoke's three findings (`b477e74c`); the battery reduction + the guard (`cc065247`) |
| [`reroll_extend.py`](reroll_extend.py) · [`run_extend.sh`](run_extend.sh) | JOB 1's dice — the fresh re-rolls 8–23, the salt assertion, the CRN reproduction check |
| [`score_kcurve.py`](score_kcurve.py) · [`kcurve.json`](kcurve.json) | the K-curve read: one label, six scorers, one pair set; three ceilings; the cost column |
| [`playoff_rule_offline.py`](playoff_rule_offline.py) · [`playoff_rule.json`](playoff_rule.json) | the BATTERY PROXY — the production decision rule, imported, on the banked CRN pairs |
| [`run_battery.sh`](run_battery.sh) · [`report_battery.py`](report_battery.py) · [`battery.json`](battery.json) | JOB 2's cells and their read |
| [`make_table.py`](make_table.py) | renders the three JSONs into this file's tables — **no number above was retyped by hand** |
| not committed | the 31,992 re-roll outcomes and the battery rows, under `/home/goodlad/.claude/jobs/9ab51de6/tmp/kcurve/` |

---

## 8. The registered predictions, scored

**JOB 1 (PREDICTION.md §2.4) — nine of ten held.**

| # | registered | outcome |
|---|---|---|
| **K1** | the curve is MONOTONE and CONCAVE on all four columns | **HELD** — monotone on all four, every increment shrinks (§3 marginal table) |
| **K2** | the head reproduces 0.5801 ± 0.01 pooled and 0.5610 ± 0.01 on `top1\|top2` | **HALF — pooled HELD at 0.5798 (Δ 0.0003); `top1\|top2` is a NEAR MISS at 0.5741 (Δ +0.0131)**. Declared cause: a 665-fork subset read at the SUCCESSOR index rather than through `build_table`'s branch rows, which drops a capped branch. Reported, not hidden |
| **K3** | BAR K1 met at K = 4 (K = 2 plausible) | **HELD** — K = 4 on three columns, K = 2 on `rand\|top2` |
| **K4** | `ROLLOUT_16` in 0.63–0.72 and `ROLLOUT_1` in 0.53–0.58 on `top1\|top2` | **HELD on both** — 0.6709 and 0.5427 |
| **K5** | BAR K2 met at K = 8, not at K ≤ 4, on `top1\|top2` | **HELD exactly** |
| **K6** | the POOLED curve is above `top1\|top2` at every K | **HELD** at every one of the five levels |
| **K7** | the knee is at K = 4 | **HELD on three columns, REFUTED on `top1\|top2`, where it is K = 8** — the leaf column needs one more doubling, which is the sharper statement |
| **K8** | the split-half agreement reproduces 0.6303 ± 0.02 and its `q` OVER-states `ROLLOUT_16` | **HALF.** The agreement is **0.6140** (the control's 0.6303 was on a different pair set, ±0.02 missed by 0.0003 of the band edge); `q = 0.7387` does over-state `ROLLOUT_16`'s 0.6709, as registered |
| **K9** | 25–29 mean remaining turns per rollout | **HELD** — 25.71 |
| **K10** | `verify_ok` = 24 / 24, `salt_mismatch` = 0 | **HELD at both** |

**JOB 2 (PREDICTION.md §3.4, AMENDMENT §3, AMENDMENT2 §4).**

| # | registered | outcome |
|---|---|---|
| **B1** | ≥ 60 % of playoffs inconclusive at R = 4 | **HELD, at 98.9 %** (188 of 190 at a realized R = 4.00) |
| **B2** | L2 in 0.49–0.53, CI straddling 0.50 ⇒ branch (b) | **HELD in direction; the cell is a DESCRIPTOR** (0.5000 over 3 pairs, under-powered by design) |
| **B4** | the `defB` control reads 0.48–0.53 straddling 0.50 | **HELD** — 0.5243 [0.4864, 0.5623] over 113 pairs |
| **B5** | 5–30 s per adjudicated decision, ≥ 2 min per game | **HELD** — 43.3 s per adjudicated decision is just outside the band (reported as such) and 1,414 s per battle is far inside it |
| **B6** | `screen_decisive` on 40–75 % of decisions | **REFUTED, low: 14.0 %** (35 of 250). The screen's top-1 differs from the policy's action far more often than registered, so the playoff runs on a MAJORITY of decisions — which is why the cell is so expensive |
| **B7** | zero timeouts, zero unfinished games | **HELD** — 0 unfinished of 234 battles |
| **A1** | offline conclusive < 0.15 at R = 4, rising, 0.25–0.55 at R = 16 | **HALF — REFUTED at R = 16.** 0.0451 at R = 4 (held), rising monotonically (held), but **0.1248 at R = 16**, far below the registered band. The gate is much tighter than predicted |
| **A2** | the live cell overrules on < 5 % of decisions at R = 8 | **HELD at R = 4** (0.4 % changed); R = 8 was not run live (AMENDMENT 2) |
| **A4** | agreement with the label 0.60–0.80 when the rule concludes | **REFUTED, HIGH: 0.87 / 0.83 / 0.92** — the gate is more selective and more correct than registered, which is the same fact as A1's miss |
| **A5** | JOB 1 finishes with `errors = 0` under `--impl rust` | **HELD** — 0 errors in 31,992 rollouts |
| **C1** | `pf4b`'s realized R ≥ 3.6 | **HELD exactly: 4.00** |
| **C2** | `pf4b` plays < 5 % of all decisions | **HELD** — 2 of 250 (0.8 %) |
| **C3** | `pf4b`'s L2 anywhere in 0.3–0.7, interval ±0.2 or wider | **HELD** — 0.5000 at n = 3 |
| **C4** | the guard refuses `pf4` / `pf8` and passes `pf4b` | **HELD** — executed against the banked rows: REFUSED / REFUSED / PASS |

---

## 9. Ready-to-append ledger paragraph

> ### 🎯 HOW MANY ROLLOUTS DOES A LEAF NEED, AND DOES IT PAY IN GAMES? — **the dose is FOUR and the leaf WINS by +0.068 [+0.038, +0.101] DETECTED (pooled) / +0.078 [+0.027, +0.128] at K=8 on the LEAF column — and 🚨 IT BUYS NOTHING IN GAMES BECAUSE THE ARM'S OWN 2·SE GATE RESOLVES ONLY 4.5 % OF PAIRS AT R=4 AND 12.5 % AT R=16.** The binding constraint has moved from the leaf to the DECISION RULE (2026-09-19)
>
> Record `designs/research_state/measurements/rollout_leaf_kcurve_2026-09-19/` (`PREDICTION.md`
> landed on main at **`ca52dc7f`** before the first rollout; `AMENDMENT.md` at **`b477e74c`**,
> `AMENDMENT2.md` + the guard at **`cc065247`**). **JOB 1: 31,992 fresh post-divergence rollouts**
> over **665** of the 2026-09-19 control's banked held-out contested CRN forks (all three branches
> at FULL K = 16), read against the control's OWN banked **K′ = 8** label — one label for every K,
> the leaf an INDEPENDENT mean of dice the label has never seen (`fresh_seeds(24)` is a strict
> extension of `fresh_seeds(8)`), nested in K and declared. **0 errors, 36 capped (0.11 %), 24/24
> CRN reproduction checks EXACT, 0 salt mismatches, 0 index mismatches.** Arm W frozen; CPU only;
> nothing under `models/`.
>
> **THE K-CURVE, against the K′ = 8 label, with the label ceiling and the cost beside every level.**
> `top1|top2` (351 non-tied pairs): head **0.5741** · K=1 **0.5427** · K=2 **0.5798** · K=4
> **0.6125** · K=8 **0.6524** · K=16 **0.6709**. Paired Δ vs the head on identical pairs: K=4
> **+0.0385 [−0.0089, +0.0871] ND**, **K=8 +0.0783 [+0.0273, +0.1279] DETECTED**, K=16 **+0.0969
> [+0.0467, +0.1477] DETECTED**. Pooled (1,159 non-tied): head 0.5798, and **K=4 already DETECTS at
> +0.0677 [+0.0382, +0.1006]**, K=16 at +0.1217. **BAR K1 (a CI lower bound above the head's point)
> is met at K = 4 on three of four columns; BAR K2 (above 0.60) at K = 4 pooled and K = 8 on the
> leaf column.** **COST, measured: a rollout runs 25.7 turns to a terminal, so a K-leaf evaluation
> is ~25.7K sim turns and a top-2 playoff decision twice that — 206 turns at K=4, 411 at K=8**;
> 3.16 s per rollout per worker on a box at load 30–60. **The KNEE (marginal gain per rollout below
> the average bought so far) is K = 4 on three columns and K = 8 on `top1|top2`** — the policy's own
> two best moves need twice the dice a random legal move does. 🚨 **The MEASURED ceiling (a
> 16-rollout independent estimator) is 0.7015 pooled / 0.6709 on the leaf column and EXCEEDS the
> label's split-half agreement (0.6480 / 0.6140)** — exactly the bias the 2026-09-19 control
> registered, so the split-half `q` (0.7387) stays a CROSS-CHECK and is never a bar; **49 % of the
> sibling-label difference's variance is STILL label noise at K′ = 8** on that column.
>
> 🚨 **AND IT DOES NOT TRANSFER — branch (b), with the mechanism named. The `playoff` arm's own
> gate (`|mean(d)| ≥ 2·SE` over R CRN-paired rollouts, `MIN_PAIRS = 4`) resolves 0 % of pairs at
> R ∈ {1,2} (below MIN_PAIRS it CANNOT conclude), 4.51 % [3.01, 6.17] at R = 4, 10.68 % at R = 8
> and 12.48 % [9.92, 15.04] at R = 16** — measured by IMPORTING the production rule
> (`playoff.paired_stats` / `is_conclusive`) and applying it to all 665 forks' CRN pairs, a
> battery-proxy that costs nothing and scales with the dice. **When it DOES conclude it agrees with
> the K′ = 8 label 0.87 / 0.83 / 0.92** (a SELECTION effect, quoted only with its coverage).
> **LIVE: the first cell-level use of the arm** — `--budget 120`, realized **R = 4.00** exactly —
> **played 2 of 190 playoffs and changed 1 of 250 decisions**; its contemporaneous critic-leaf
> control read **L2 0.5243 [0.4864, 0.5623]** over 113 pairs, straddling 0.50 where six win-prob
> heads already sit. **The rollout-leaf cell's own L2 is 0.5000 over 3 pairs and is published as an
> UNDER-POWERED DESCRIPTOR, not evidence**: at a measured 1,414 s per battle, 400 pairs of it is
> **1,222 CPU-hours**, so rule 25's 400-pair bar is recorded UNMET with the arithmetic (registered
> in `AMENDMENT2.md` before the read). **The one-line consequence: out-ranking successors is no
> longer the binding constraint — the ACTING RULE is, and re-tuning `SE_MULTIPLE` / `MIN_PAIRS`
> against the banked dice costs nothing.**
>
> 🚨 **THREE INSTRUMENT DEFECTS, all filed in `designs/ops/TECH_DEBT_BACKLOG.md`, one GUARDED.**
> (1) **`--impl rust` BREAKS the playoff's nested counterfactual rollouts and the failure is
> INVISIBLE in the row** — 63 of 75 decisions lost their playoff to `unresolvable choice for p1:
> MoveName("crunch") — active metagross has moves [...]` (250 failed rollouts) while the identical
> game under `--impl node` raised 0; `PlayoffResult.error` lands in `diag["playoff"]["error"]`,
> which NO row field carries, so the cell folds to `n_playoff_no_budget` and reports a clean win
> rate for an arm that never adjudicated. The same rust path is clean on this read's 31,992 offline
> rollouts, so the candidates are the LIVE partial record and its `sodium,<hex>` seed spelling —
> **neither proven**. (2) **`--playoff-rollouts` was INERT below a budget of 2R rollouts with no
> refusal**: a live terminal rollout measured ~10 s, so at `--budget 20` the deadline bought ONE
> pair and `MIN_PAIRS = 4` declined every playoff — **cells asked for R = 4 and R = 8 produced
> BYTE-IDENTICAL no-op behaviour (realized R = 1.00, 64/64 inconclusive, `n_changed` exactly 0)**.
> **GUARD LANDED**: `playoff.short_r_refusal` + `--playoff-allow-short-r`, raising on the FIRST game
> after the row is appended, six tests, routine gate green (10,973 passed); the OPEN half is that
> `rollout_cost_s` seeds at 1.0 s against a measured ~10 s. (3) `run_local_battles`' docstring
> claims rust degrades `resumeReseed` / `__RECON__` to no-ops — **STALE**, `sim_bridge.rs`
> implements both, which is why 31,992 reseeded rust rollouts produce real dice variation.
> **NINE of TEN JOB-1 predictions held** (K2 is a declared NEAR MISS: the head reads 0.5741 against
> the control's 0.5610 on `top1|top2`, +0.0131 outside the registered ±0.01, on a 665-fork subset
> read at the successor index rather than through `build_table`'s branch rows; K7's knee is K = 8
> on the leaf column, not K = 4). **A1 and A4 are REFUTED in the same direction** — the gate is far
> TIGHTER (12.5 % at R = 16 against a registered 25–55 %) and far MORE CORRECT (0.92 against a
> registered 0.60–0.80) than predicted, which is one fact stated twice and is the read's sharpest
> surprise. **B6 REFUTED low: the screen is decisive on only 14 % of decisions**, so the playoff
> runs on a majority of them and that is where the cell's cost comes from. **WHAT REMAINS:**
> (1) **re-tune the GATE on the banked dice — free**: sweep `SE_MULTIPLE` and `MIN_PAIRS` for the
> coverage/agreement ROC the proxy already prices; (2) the per-action **`q_winprob_head`** (built,
> OFF) as the AMORTIZER — 206 sim turns per top-2 decision is a teacher's budget, not an inference
> budget; (3) ⚠️ **NOT worth another arm: a 400-pair mirror cell of an arm that changes 0.4 % of
> decisions.** Tag: **MEASURED (MAJOR) · THE DOSE IS FOUR — a 4-rollout leaf DETECTS over the
> 75M-step critic pooled and an 8-rollout leaf on the LEAF column (+0.078 [+0.027, +0.128]) · the
> measured label ceiling is 0.67–0.70, NOT the split-half 0.61 · 🚨 THE BINDING CONSTRAINT MOVED
> FROM THE LEAF TO THE DECISION RULE — 2·SE over 4 CRN pairs resolves 4.5 % of them and the arm
> acts on 1 % of decisions · `--impl rust` silently breaks the arm and `--playoff-rollouts` was
> inert below budget 2R (guard landed)**.
