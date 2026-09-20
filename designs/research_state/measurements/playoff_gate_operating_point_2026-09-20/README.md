# WHERE IS THE PLAYOFF GATE'S OPERATING POINT? — the acting rule swept on banked dice, and tested in games

*Measured 2026-09-19 20:20 – 2026-09-20 02:29 UTC · **JOB A**, free: the production decision rule
IMPORTED and swept over R × SE-multiple × MIN_PAIRS on the **665 banked forks** of
[`rollout_leaf_kcurve_2026-09-19/`](../rollout_leaf_kcurve_2026-09-19/README.md) — K = 16 fresh CRN
re-rolls per branch, read against the **DISJOINT** banked K′ = 8 label, **no new rollout paid for**
· **JOB B**: a **965-battle** live mirror battery at the point JOB A picks — 80 playoff pairs, and
**200 pairs each of TWO contemporaneous controls including, for the first time in this battery, the
unsearched `base` NULL**. CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, `--impl node`, ~28
CPU-hours, **nothing written under `models/`**; the GPU held `ai_v13_09_wcont` throughout and was
not touched, 8000 / 8001 untouched, no process this read did not start was signalled.*

**Pre-registered in [`PREDICTION.md`](PREDICTION.md), landed on main at `48743b41` before the first
number existed. JOB A landed at `1bdd233e` before JOB B was read.** All 21 predictions are scored in
§8.

---

## 1. VERDICT — **branch (b)**, and the sign is the finding

**1 — 🚨 THE SHIPPED GATE IS AT THE WRONG END OF ITS OWN CURVE, AND BY A FACTOR OF SIX.** At
`SE_MULTIPLE = 2.0`, `MIN_PAIRS = 4`, R = 4 the expected value gain is **+0.00254 win-prob units per
decision [−0.00038, +0.00611] NOT DETECTED** — **0.041×** an oracle that always takes the better
branch. The same rule at `SE_MULTIPLE = 0.5`, same R, **identical rollout cost**, takes **+0.01570
[+0.00639, +0.02575] DETECTED**, **0.251× the oracle**. The offline half of branch (a) clears: an
operating point with positive gain at resolve ≥ 20 % exists, and there are several.

**2 — 🚨 AND THE GATE IS BUYING ALMOST NOTHING OVER "JUST RE-RANK".** The no-gate rule — always take
the branch with the higher R-rollout mean — reads **+0.01570** at R = 4, matching the best gated
point *to five decimal places*, and **+0.02763** at R = 8, **above every gated cell there**. Only at
R = 16 does a gate edge it, by 0.0009, far inside the interval. **Across the grid the bar's entire
contribution is within noise of zero; what it reliably does is cost coverage.**

**3 — 🚨 AND IN GAMES IT STILL DOES NOT PAY — BUT NOW FOR A DIFFERENT REASON, AND THE POINT ESTIMATE
IS ON THE WRONG SIDE.** The live cell **acted**: 6,886 decisions, **11.88 % of them changed**
(against **0.4 %** for the shipped gate on 2026-09-19 — a **30× larger intervention**), realized
R = **3.99**. Its paired mirror win rate is **L2 = 0.4719 [0.4021, 0.5416]** over 80 pairs — **NOT
DETECTED, and below 0.50.** Against its own contemporaneous controls on shared indices:
`base − pfk05` = **+0.0594 [−0.0125, +0.1344]**, `defB − pfk05` = **+0.0469 [−0.0407, +0.1375]**,
both NOT DETECTED and both pointing the same way. **The 2026-09-19 diagnosis — "the leaf out-ranks,
the arm does not act" — is now half refuted: the arm acts, and the games do not move.**

**4 — 🚨 THE UNSEARCHED `base` NULL IS 0.5150, NOT 0.5000.** Run for the first time here at n = 200,
the unsearched policy against its own unsearched self reads **L2 = 0.5150 [0.4967, 0.5333]** — it
covers 0.50, so it is NOT DETECTED, but its interval only *just* does. **Every L2 bar within ±0.02
of 0.50 in this battery should be read against `base`, not against the constant 0.50**, and against
`base` the playoff cell is 5.9 points down rather than 2.8.

**5 — 🚨 AN ARM THAT ACTUALLY ACTS DESTROYS THE MIRROR'S OWN PRECISION, WHICH IS WHY THE ≥ 100-PAIR
BAR IS THE WRONG BAR.** A near-null cell's pair scores are dominated by 0.5 ties: `base` measured
sd **0.132** and `defB` **0.243**. The playoff cell measured **sd 0.318**, *because* it changes 12 %
of decisions. So its 95 % half-width at n = 80 is **±0.070**, not the ±0.040 an n = 100 null cell
would give — **the more an arm does, the more pairs it needs to be read at the same width.**

**6 — THE COST IS NAMED, AND IT IS A TEACHER'S BUDGET.** **36.9 s per adjudicated decision** and
**950 s per battle** — 56× the unsearched `base` (17 s) and 19× the critic-leaf `defB` (51 s), for a
cell that ends 3–6 points *below* both.

**7 — THE DOSE STILL MATTERS MORE THAN THE BAR.** At fixed `k = 0.5` the per-decision gain goes
**+0.0157 → +0.0259 → +0.0320** across R = 4 → 8 → 16 (0.251× → 0.415× → 0.511× the oracle).
Doubling the dice buys more than any bar setting at fixed dice — and costs proportionally.

**8 — THE CEILING IS LOW, AND SAYING SO IS THE POINT.** **47.2 %** of the banked pairs are **TIED**
under the K′ = 8 label and E|label gap| is **0.1316**, so an ORACLE always taking the better sibling
gains only **+0.0625 win-prob units per contested decision [+0.0528, +0.0729]**. Every EV here is
quoted as a fraction of it. **A rule at 0.25× the oracle on 45 % of contested decisions is the
honest size of the prize** — not the +0.078 pairwise-accuracy headline, which is an accuracy and
not a value.

---

## 2. The frame

| | |
|---|---|
| **the policy** | `models/ai_v13_02_flywheel_winprob/final_model.zip` — **arm W**, step **75,005,952**, `--critic winprob`. FROZEN; nothing was trained in this read |
| **the states (JOB A)** | the **665** banked forks of the 2026-09-19 K-curve — held-out contested CRN forks, all three branches at FULL K = 16. **0 unjoined, 0 salt mismatches, 0 short rows** |
| **the gate's dice** | fresh re-rolls **8 … 8+R−1**: the CRN paired differences `d_k = score(top1,k) − score(top2,k)` that `PlayoffRunner.adjudicate` forms |
| **the label** | the banked **K′ = 8** mean (re-rolls **0–7**) — **DISJOINT** from the gate's dice, so nothing selects on the quantity it is scored against |
| **the rule** | `playoff.paired_stats` / `is_conclusive` / `decide`, **IMPORTED from production, never re-implemented** |
| **the flags** | `--playoff-se-k` and `--playoff-min-pairs` **already existed** and already reach `PlayoffConfig` → `decide`. **No `src/` change was needed or made** (PREDICTION.md S4; the instruction's `--playoff-se-multiple` is the shipped `--playoff-se-k`) |
| **JOB B** | `--arm playoff --budget 120 --playoff-rollouts 4 --playoff-se-k 0.5 --playoff-min-pairs 4 --max-worlds 4 --max-dice 2 --max-opp 6 --max-depth 1`, `--opponents self --games-seed 7 --impl node --device cpu`, the full **719-team** pool, 8 shards over contiguous game windows; `defB` and `base` over the whole range in the SAME window (rule 23) |

**Why the EV column is the objective and the agreement column is not** (registered as S2 before any
number existed): the K′ = 8 label is a mean of 8 Bernoulli draws, so it is an **unbiased** estimator
of a branch's value and the *expectation* of `label(pick) − label(top1)` is the true expected gain
whatever the label's variance. A pairwise ACCURACY has no such property — label noise drags every
accuracy toward 0.5 (the 2026-09-19 measured ceiling on this column is 0.6709). **Steer by the EV
column; sanity-check with the agreement column.**

⚠️ **The "agrees with the 16-rollout estimate" column is BIASED UPWARD BY CONSTRUCTION** at every
R < 16 — the gate's R draws are a strict SUBSET of the 16, so the scorer contains the scored. It
reads exactly 1.0000 at R = 16 for that arithmetic reason. Published because the instruction asked
for it, labelled every time it appears.

---

## 3. JOB A — THE OPERATING CURVE

### the GATE'S OPERATING CURVE — the production rule, imported, swept on the 665 banked forks

*The label is the banked K′ = 8 mean, DISJOINT from the gate's fresh dice. E\|label gap\| = **0.1316**, 0.472 of pairs label-TIED, so the **ORACLE CEILING on the per-decision gain is +0.06250** [+0.05282, +0.07293] — every EV below is quoted as a fraction of it.*

| R | SE mult | resolve rate [95 % CI] | overrules / resolved | agrees w/ K′=8 label | agrees w/ 16-roll ⚠️ | EV gain / RESOLVED | **EV gain / DECISION** [95 % CI] | × oracle |
|---|---|---|---|---|---|---|---|---|
| **4** | **2** *(production)* | 0.0451 [0.0316, 0.0617] | 0.4333 | 0.8696 (n=23) | 1.0000 | +0.0563 | **+0.00254** [-0.00038, +0.00611] ND | 0.041× |
| **4** | **1.5** | 0.1654 [0.1368, 0.1940] | 0.4909 | 0.7604 (n=96) | 0.8962 | +0.0591 | **+0.00977** [+0.00329, +0.01664] **DET** | 0.156× |
| **4** | **1** | 0.4120 [0.3759, 0.4496] | 0.5073 | 0.6682 (n=211) | 0.8642 | +0.0308 | **+0.01269** [+0.00488, +0.02143] **DET** | 0.203× |
| **4** | **0.5** ⬅ **CHOSEN** | 0.4556 [0.4180, 0.4932] | 0.5215 | 0.6653 (n=239) | 0.8476 | +0.0344 | **+0.01570** [+0.00639, +0.02575] **DET** | 0.251× |
| **4** | **NO GATE** *(always re-rank)* | 0.4602 [0.4226, 0.4977] | 0.5196 | 0.6639 (n=241) | 0.8456 | +0.0341 | **+0.01570** [+0.00620, +0.02556] **DET** | 0.251× |
| **8** | **2** | 0.1068 [0.0842, 0.1323] | 0.5352 | 0.8333 (n=60) | 0.9855 | +0.0845 | **+0.00902** [+0.00367, +0.01457] **DET** | 0.144× |
| **8** | **1.5** | 0.1925 [0.1609, 0.2241] | 0.5000 | 0.7870 (n=108) | 0.9675 | +0.0859 | **+0.01654** [+0.00958, +0.02481] **DET** | 0.265× |
| **8** | **1** | 0.4211 [0.3850, 0.4586] | 0.5179 | 0.7353 (n=204) | 0.9606 | +0.0623 | **+0.02622** [+0.01738, +0.03609] **DET** | 0.420× |
| **8** | **0.5** | 0.5053 [0.4662, 0.5429] | 0.5119 | 0.6944 (n=252) | 0.9349 | +0.0513 | **+0.02594** [+0.01570, +0.03666] **DET** | 0.415× |
| **8** | **NO GATE** *(always re-rank)* | 0.5474 [0.5098, 0.5850] | 0.5165 | 0.6918 (n=279) | 0.9271 | +0.0505 | **+0.02763** [+0.01758, +0.03816] **DET** | 0.442× |
| **16** | **2** | 0.1248 [0.1008, 0.1504] | 0.5181 | 0.9221 (n=77) | 1.0000 | +0.1107 | **+0.01382** [+0.00808, +0.02030] **DET** | 0.221× |
| **16** | **1.5** | 0.1789 [0.1504, 0.2075] | 0.5210 | 0.8426 (n=108) | 1.0000 | +0.0956 | **+0.01711** [+0.01015, +0.02472] **DET** | 0.274× |
| **16** | **1** | 0.4316 [0.3925, 0.4692] | 0.5296 | 0.7366 (n=205) | 1.0000 | +0.0564 | **+0.02434** [+0.01579, +0.03346] **DET** | 0.389× |
| **16** | **0.5** | 0.5414 [0.5038, 0.5805] | 0.5528 | 0.7126 (n=261) | 1.0000 | +0.0590 | **+0.03195** [+0.02152, +0.04277] **DET** | 0.511× |
| **16** | **NO GATE** *(always re-rank)* | 0.6180 [0.5820, 0.6541] | 0.5523 | 0.6974 (n=304) | 1.0000 | +0.0502 | **+0.03102** [+0.02020, +0.04258] **DET** | 0.496× |

### the MIN_PAIRS axis — INERT at R ≥ 4 by construction, and what it does where it bites

| R | SE mult | resolve, MIN_PAIRS = 2 | resolve, MIN_PAIRS = 4 | identical? |
|---|---|---|---|---|
| 2 | 0.5 | 0.3519 | 0.0000 | no — MIN_PAIRS BITES |
| 2 | 1 | 0.3519 | 0.0000 | no — MIN_PAIRS BITES |
| 2 | 1.5 | 0.0571 | 0.0000 | no — MIN_PAIRS BITES |
| 2 | 2 | 0.0571 | 0.0000 | no — MIN_PAIRS BITES |
| 3 | 0.5 | 0.3865 | 0.0000 | no — MIN_PAIRS BITES |
| 3 | 1 | 0.1293 | 0.0000 | no — MIN_PAIRS BITES |
| 3 | 1.5 | 0.1293 | 0.0000 | no — MIN_PAIRS BITES |
| 3 | 2 | 0.0241 | 0.0000 | no — MIN_PAIRS BITES |
| 4 | 0.5 | 0.4556 | 0.4556 | **yes** |
| 4 | 1 | 0.4120 | 0.4120 | **yes** |
| 4 | 1.5 | 0.1654 | 0.1654 | **yes** |
| 4 | 2 | 0.0451 | 0.0451 | **yes** |
| 8 | 0.5 | 0.5053 | 0.5053 | **yes** |
| 8 | 1 | 0.4211 | 0.4211 | **yes** |
| 8 | 1.5 | 0.1925 | 0.1925 | **yes** |
| 8 | 2 | 0.1068 | 0.1068 | **yes** |
| 16 | 0.5 | 0.5414 | 0.5414 | **yes** |
| 16 | 1 | 0.4316 | 0.4316 | **yes** |
| 16 | 1.5 | 0.1789 | 0.1789 | **yes** |
| 16 | 2 | 0.1248 | 0.1248 | **yes** |

### the two points the instruction asked for, named

| question | answer |
|---|---|
| the point maximising **EV gain × resolve rate** (= EV gain per DECISION) at R = 4 — **the point the live battery ran** | **R = 4, SE mult = 0.5**: +0.01570 per decision [+0.00639, +0.02575], resolve 0.4556, 0.251× the oracle |
| the same, over the WHOLE grid (cost ignored) | **R = 16, SE mult = 0.5**: +0.03195 per decision, resolve 0.5414, 0.511× the oracle |
| the point(s) where **agreement with the label stays ≥ 0.80** | **R = 4, SE mult = 2** (0.8696 at resolve 0.0451, EV/decision +0.00254), **R = 8, SE mult = 2** (0.8333 at resolve 0.1068, EV/decision +0.00902), **R = 16, SE mult = 2** (0.9221 at resolve 0.1248, EV/decision +0.01382), **R = 16, SE mult = 1.5** (0.8426 at resolve 0.1789, EV/decision +0.01711) |


### Reading the curve

**The shape, in one sentence.** Resolve rate and per-resolved quality trade against each other
exactly as the bar was designed to make them, and the product — **EV gain per DECISION**, the only
column a game can feel — is **monotone decreasing in `k` at every R**. The bar sits on the wrong
side of its own optimum at every dose.

**Why the no-gate rule is so hard to beat here.** At R = 4 the `k = 0.5` gate resolves 0.4556 and
the no-gate rule 0.4602; the 0.0046 difference is forks whose paired mean is non-zero but smaller
than half its own SE — the smallest true gaps, contributing almost nothing either way. **The gate
and the open rule are the same rule at low `k`**, and the production `k = 2.0` is where they finally
differ, by discarding 90 % of the coverage.

**What "0.87 agreement" was actually worth.** The 2026-09-19 read published that number with its
coverage beside it, as a selection effect. This read prices the selection: **the 0.87-agreement
point takes +0.0025 per decision and the 0.67-agreement point takes +0.0157.** A more-often-right
rule that almost never fires is worth a sixth of a more-often-wrong rule that fires ten times as
often — because a declined decision contributes **exactly 0.0**, not a small positive.

**The 2026-09-19 self-check.** The `k = 2.0`, `MIN_PAIRS = 4` column reproduces that read's
published resolve rates **exactly** — 0.0451 / 0.1068 / 0.1248 at R = 4 / 8 / 16 — and its
agreements — 0.8696 / 0.8333 / 0.9221. Prediction **A1 HELD**; same forks, same imported rule.

---

## 4. JOB B — THE LIVE BATTERY AT THE CHOSEN POINT

*The registered selection rule (PREDICTION.md §3, written before the curve existed) pinned **R = 4
by COST** and then took the SE multiple maximising EV gain per decision: **`--playoff-se-k 0.5`,
`--playoff-min-pairs 4`**. The runner-up `k = 1.0` was at 81 % of the maximum, outside the 10 %
tie-break band, so no judgement entered the choice.*

### the BATTERY — the playoff cell at the chosen point and its two contemporaneous controls

| cell | battles | unfin | pairs | **L2 paired** [normal CI] | [bootstrap CI] | unpaired Wilson | changed / decision | s / battle |
|---|---|---|---|---|---|---|---|---|
| **base** | 400 | 0 | 200 | **0.5150** [0.4967, 0.5333] | [0.4975, 0.5350] | 0.5150 [0.4661, 0.5636] | 0.0000 | 17 |
| **defB** | 400 | 0 | 200 | **0.5025** [0.4688, 0.5362] | [0.4675, 0.5375] | 0.5025 [0.4537, 0.5512] | 0.0177 | 51 |
| **pfk05** | 165 | 0 | 80 | **0.4719** [0.4021, 0.5416] | [0.4000, 0.5437] | 0.4634 [0.3888, 0.5397] | 0.1188 | 950 |

### the MECHANISM row — the primary read

| cell | decisions | screen_decisive | playoffs RUN | RESOLVED (played) | inconclusive | no_budget | failed rollouts | **live resolve rate** | realized R | s / adjudicated decision |
|---|---|---|---|---|---|---|---|---|---|---|
| **base** | 18538 | 0 | 0 | 0 | 0 | 0 | 0 | **0.0000** | 0.00 | 0.0 |
| **defB** | 18183 | 0 | 0 | 0 | 0 | 0 | 0 | **0.0000** | 0.00 | 0.0 |
| **pfk05** | 6886 | 1077 | 4047 | 1127 | 2920 | 728 | 2913 | **0.2785** | 3.99 | 36.9 |

### paired deltas on SHARED game indices

* `base - defB` = +0.0125 [-0.0176, +0.0426] NOT DETECTED (n = 200 shared pairs)
* `base - pfk05` = +0.0594 [-0.0125, +0.1344] NOT DETECTED (n = 80 shared pairs)
* `defB - pfk05` = +0.0469 [-0.0407, +0.1375] NOT DETECTED (n = 80 shared pairs)

### the OFFLINE→LIVE comparison — the registered prediction, scored

| quantity | offline curve's prediction for this point | measured LIVE | |
|---|---|---|---|
| **resolve rate** (of playoffs that RAN) | **0.4556** [0.4180, 0.4932] | **0.2785** (1,127 of 4,047) | ratio **0.61** — inside the registered 0.2–0.8 band (**B1 HELD**) |
| **change rate** (of all decisions) | 0.4556 × 0.5215 = **0.2376** overrules | **0.1188** (818 of 6,886) | exactly **half** the offline rate — and **30×** the shipped gate's 0.4 % (**B2 HELD**) |
| **realized R** | 4 requested | **3.99** | `short_r_refusal` did not fire on the pooled cell (**B3 HELD**) |
| screen decisive | — | **15.6 %** (1,077 of 6,886) | reproduces 2026-09-19's 14 % |

**The direction of the resolve-rate miss is the one the 2026-09-19 proxy named**: offline both sides
are GREEDY against a banked sentinel, live both are the same network **STOCHASTIC at T = 1**, which
adds variance to every draw and widens the SE, so the offline rate is an UPPER bound. It was, by a
factor of 1.6.

### 🚨 THE ARITHMETIC OF BRANCH (b), as registered in PREDICTION.md B7

The offline curve says the gate at this point takes **+0.0157 win-prob units per contested fork
decision**, from **23.8 %** of decisions overruled at **+0.066** each. Live it overruled **11.9 %**
of decisions. If each live overrule were worth the offline +0.066, the per-decision gain would be
**+0.0079 win-prob units**, and over the cell's **41.7 decisions per battle** a naive *sum* would be
**+0.33** — an absurd number, which is the point: **per-decision advantages measured on selected
contested forks are not additive across a game, so the offline curve predicts the SIGN of a game
effect and nothing about its size.**

What the cell can and cannot see, with its own measured spread:

| | |
|---|---|
| realized paired sd for THIS arm | **0.318** (against 0.132 for `base`, 0.243 for `defB`) |
| 95 % half-width at the 80 pairs achieved | **±0.070** |
| smallest L2 whose lower bound clears 0.50, at n = 80 | **0.570** — a **+7.0 pp** effect |
| the same at n = 100 (the instruction's target) | **0.562** — **+6.2 pp** |
| the same at n = 400 (rule 25's bar) | **0.531** — **+3.1 pp** |
| **measured** | **0.4719** |

**So the honest statement is sharper than "under-powered".** The cell is under-powered for a small
positive effect — it could not have resolved +2 pp. But it is **not** under-powered for anything
close to the transfer the offline gain would imply, and **the point estimate is on the wrong side of
the null**: even at a hypothetical 400 pairs the observed 0.4719 would read [0.441, 0.503] and still
not clear 0.50. **Branch (b), with the sign against the arm.**

---

## 5. THE BRANCH

**Registered branch (b): an offline operating point with positive EV at resolve ≥ 20 % EXISTS (several do), and the live L2 lower bound does not clear 0.50.** The sharpening is the finding:

> **The 2026-09-19 read moved the binding constraint from the LEAF to the ACTING RULE. This read
> moves it again — and off the rule.** Opening the gate did everything it was supposed to do: the
> arm went from changing 0.4 % of decisions to changing **11.9 %**, the offline expected value went
> up **6.2×**, and the mirror win rate went **down** to 0.4719 [0.4021, 0.5416]. **Neither the leaf
> nor the gate is what is failing.** Two candidates remain and this read cannot separate them: the
> **CANDIDATE PAIR** (live the playoff adjudicates the SCREEN's top-2 — a biased critic sweep that
> was decisive in the policy's favour on only 15.6 % of decisions), and the
> **ESTIMAND** (a self-rollout against a stochastic mirror is not the banked tree's greedy-sentinel
> rollout, and the K′ = 8 label that prices the gain was measured against the latter).

Three consequences, in cost order:

1. **The cheapest next read is free and is about the CANDIDATE PAIR, not the gate.** The banked
   forks score the POLICY's top-2; the live arm plays the SCREEN's top-2. Re-running JOB A with the
   pair chosen by the screen's own V-ranking — which the banked tree's third `rand` branch partly
   supports — would say whether the offline +0.0157 survives the substitution. If it does not, the
   screen is the defect and the playoff is fine.
2. **A rollout leaf is still only affordable where the ply is amortized.** 36.9 s per adjudicated
   decision and 950 s per battle is a TEACHER's budget, not an inference budget — which is the
   `q_winprob_head` path (built, OFF): pay the rollouts once, offline, and distil the ordering.
   That conclusion is unchanged and is now supported by a cell that *did* act.
3. ⚠️ **Do NOT spend another arm on the gate's constants.** The curve is measured, its maximum is
   located, the maximum was played in games, and it lost. `SE_MULTIPLE` is a closed question.

---

## 6. Hazards and findings about the instruments

Every one of these is a finding.

1. 🚨 **`short_r_refusal` IS ROW-LEVEL AND IT MIS-FIRES ON A LOADED BOX.** Two of eight shards were
   killed by the guard that landed on 2026-09-19 — at realized mean R of **3.27** and **1.50 on a
   single game** — while their POOLED realized R was **3.96** and **3.98** and only **1.2 %** of all
   165 rows sat below the 3.6 floor. Dropping both refused shards moves the cell from **0.4719
   (n = 80)** to **0.4722 (n = 63)** — three ten-thousandths ([`shortr.json`](shortr.json),
   [`shortr_sensitivity.py`](shortr_sensitivity.py)). **The guard cost 17 pairs — 21 % of the
   cell — and bought nothing.** It is the right guard at the wrong granularity: a cell is defined by
   its POOLED realized R, and one slow game on a box at load 40 is not a mis-specified cell. Suggest
   a **trailing-window** floor (last k games) rather than a per-row one, and raising the seeded
   `rollout_cost_s` from its 1.0 s default toward the measured ~10 s, which is the 2026-09-19
   hazard's still-open half.
2. 🚨 **15.3 % OF LIVE ROLLOUT PAIRS FAILED under `--impl node` — 7× the 2026-09-19 rate** and far
   outside this read's registered ≤ 3 % (**B10 REFUTED**). 2,913 failed pairs against 16,142
   successful ones, against 16 failed pairs in ~760 on 2026-09-19 (2.1 %). A failed pair
   contributes nothing (correctly — never a half-pair), so it costs
   both budget and the gate's own `n`, which *lowers the live resolve rate* and is a candidate
   explanation for part of the 0.61 offline→live ratio. **"node works" is not "node is exact", and
   the gap has widened.**
3. 🚨 **`n_playoff_no_budget` = 728 of 4,775 playoff situations (15.2 %)** — at `--budget 120` on a
   box at load 40, one playoff in seven bought no pair at all. Sizing the budget as
   `screen + 2·R·(rollout cost)` needs the rollout cost measured **at the load the cell will run
   at**, not at the load the smoke ran at.
4. 🚨 **THE UNSEARCHED `base` NULL IS NOT 0.5000** (verdict 4). This is the first time this battery
   has run unsearched-vs-unsearched at n = 200, and it reads **0.5150 [0.4967, 0.5333]**. It covers
   0.50 so it is NOT DETECTED and nothing here is invalidated — but a structural null that must be
   exactly 0.50 and measures 1.5 pp above it means **every future bar in this battery is set against
   `base`, not against the constant.** Cheap, and it should have been run every time.
5. 🚨 **A CELL THAT ACTS IS READ AT A WIDER INTERVAL THAN A CELL THAT DOES NOT** (verdict 5). The
   mirror's ±0.04-at-100-pairs precision is a property of *null* cells, where ties dominate. Rule
   25's 400-pair bar is calibrated on sd ≈ 0.2; at sd = 0.318 the same width needs **2.5× the
   pairs**. **Quote the arm's own sd when stating what a battery can resolve.**
6. 🚨 **`MIN_PAIRS` IS AN INERT AXIS AT R ≥ 4, AND THAT WAS REGISTERED IN ADVANCE.**
   `is_conclusive` bars on `n < min_pairs` and offline `n = R`, so `MIN_PAIRS ∈ {2, 4}` is
   **bit-identical** at R ∈ {4, 8, 16} — asserted by the script, which REFUSES if it ever is not. A
   sweep reporting it as a tuned knob would have published two identical columns as a comparison.
7. 🚨 **`MIN_PAIRS = 4` IS AN OFF SWITCH AT R ∈ {2, 3}, NOT A BAR** — resolve exactly 0.0000 at every
   SE multiple. At `MIN_PAIRS = 2` those doses take **+0.0150 / +0.0148 per decision**, *more than
   the shipped R = 4 point at half the rollout cost*. **The cheapest useful operating point in the
   whole table is R = 2, `MIN_PAIRS = 2`, `k ≤ 1.0`, and the shipped constant forbids it.**
8. ⚠️ **The 16-rollout agreement column contains its own scorer** below R = 16 and is labelled
   wherever it appears — the same treatment the 2026-09-19 read gave the split-half `q`.
9. ⚠️ **The oracle ceiling is a LABEL oracle, not a truth oracle** — `E[max(lab2 − lab1, 0)]` on a
   K′ = 8 label, so it inherits the label's noise and the true ceiling is somewhat higher than
   +0.0625. It is a common yardstick for comparing rules on this data, not a physical bound.
10. ⚠️ **Every JOB A fork is a CONTESTED, held-out, seeded fork.** A per-decision number measured on
    them does **not** multiply by a game's decision count; §4's arithmetic is the whole of what the
    offline curve licenses.
11. **JOB A cost one CPU-minute.** The 2026-09-19 read's §5.1 said re-tuning the gate "costs
    nothing"; that was literally true, and it is the reusable lesson about banking dice at full K.
12. **Nothing under `src/` was changed by this read** — the two flags the instruction proposed
    adding were already shipped, verified by executing the parser and the rule together
    (`decide(...)` flips from `inconclusive` to `played` on the same statistic at k = 0.5 vs 2.0)
    rather than by reading the argparse block.

---

## 7. Files

| | |
|---|---|
| [`PREDICTION.md`](PREDICTION.md) | the registration, landed at `48743b41` before the first number |
| [`gate_curve.py`](gate_curve.py) · [`gate_curve.json`](gate_curve.json) | JOB A — the production rule imported and swept; the oracle ceiling; the MIN_PAIRS inertness guard |
| [`run_battery.sh`](run_battery.sh) · [`report_battery.py`](report_battery.py) · [`battery.json`](battery.json) | JOB B's cells and their read (the 2026-09-11 pairing instrument, reused; one added rate) |
| [`shortr_sensitivity.py`](shortr_sensitivity.py) · [`shortr.json`](shortr.json) | the guard-refusal sensitivity — hazard 1 |
| [`make_table.py`](make_table.py) | renders the JSONs into this file's tables — **no number above was retyped by hand** |
| not committed | the battery rows, under `/home/goodlad/.claude/jobs/9ab51de6/tmp/gatecurve/` |

---

## 8. The registered predictions, scored

**JOB A — eight of eleven held.**

| # | registered | outcome |
|---|---|---|
| **A1** | the `k = 2.0` column reproduces 2026-09-19 exactly | **HELD** — 0.0451 / 0.1068 / 0.1248 and 0.8696 / 0.8333 / 0.9221, to every published digit |
| **A2** | resolve monotone in `k` and in R | **HELD** on both axes at every cell |
| **A3** | resolve ≥ 0.40 at `k = 0.5`, R = 4 | **HELD** — 0.4556 |
| **A4** | at `k = 0.5`, agreement within ±0.04 of the unconditional `ROLLOUT_R` level | **NEAR MISS** — 0.6653 / 0.6944 / 0.7126 against 0.6125 / 0.6524 / 0.6709, i.e. +0.053 / +0.042 / +0.042. Declared cause: even a half-SE bar declines the ~47 % of forks whose paired mean is exactly 0, and those are the least informative ones, so the conditional agreement sits above the unconditional |
| **A5** | agreement ≥ 0.80 at `k ≥ 1.5` (R = 4) and `k ≥ 1.0` (R = 16) | **REFUTED, in the TIGHT direction** — at R = 4 only `k = 2.0` reaches 0.80 (k = 1.5 is 0.7604); at R = 16 it needs `k ≥ 1.5`. The gate must be tighter than registered to be that right |
| **A6** | EV per RESOLVED decision positive everywhere, falling as `k` falls | **HALF** — positive at all 12 cells (held); the monotone fall is violated at two steps (R = 4: k = 1.5's +0.0591 above k = 2.0's +0.0563; R = 16: k = 0.5's +0.0590 above k = 1.0's +0.0564), both far inside the noise |
| **A7** | EV per DECISION rises as `k` falls at every R, maximised at `k = 0.5` | **HELD at R = 4 and R = 16**; at R = 8 the maximum is `k = 1.0` (+0.02622) over `k = 0.5` (+0.02594), a 0.0003 difference inside the interval |
| **A8** | the no-gate rule is STRICTLY above every gated cell at each R | **HALF/REFUTED** — it ties to five decimals at R = 4, wins at R = 8, and **loses by 0.0009 at R = 16**. The defensible statement is the weaker one: **the gate at `k ≤ 1.0` is indistinguishable from no gate at all** |
| **A9** | no-gate EV/decision in [+0.010, +0.045] at R = 4 and [+0.020, +0.070] at R = 16 | **HELD on both** — +0.01570 and +0.03102 |
| **A10** | MIN_PAIRS bit-identical at R ∈ {4, 8, 16} | **HELD** — asserted in code, and it bites only at R ∈ {2, 3} as registered |
| **A11** | a point with EV > 0 at resolve ≥ 20 % exists at R = 4 | **HELD** — `k = 0.5` (0.4556) and `k = 1.0` (0.4120), both DETECTED |

**JOB B — six of ten held.**

| # | registered | outcome |
|---|---|---|
| **B1** | live resolve rate 0.2–0.8× the offline prediction | **HELD** — 0.2785 against 0.4556, ratio **0.61** |
| **B2** | live change rate ≥ 5× the shipped gate's 0.4 % | **HELD, at 30×** — 11.88 % |
| **B3** | `--budget 120` realizes R ≥ 3.6, no refusal | **HELD for the cell (3.99), REFUTED per-shard** — the row-level guard killed 2 of 8 shards anyway (hazard 1) |
| **B4** | `defB` reads 0.47–0.56 straddling 0.50 | **HELD** — 0.5025 [0.4688, 0.5362] over 200 pairs |
| **B5** | `base` is 0.50 by construction up to noise | **HELD but only just, and it is a finding** — 0.5150 [0.4967, 0.5333] (verdict 4) |
| **B6** | half-width ≈ 0.40/√n, i.e. ±0.064 at n = 40 | **REFUTED** — that formula is calibrated on a NULL cell's sd ≈ 0.2. This arm's sd is 0.318 and the realized half-width at n = 80 is **±0.070**, not the ±0.045 predicted (verdict 5) |
| **B7** | L2 in [0.50, 0.50 + 42·g/2], most likely the lower half | **REFUTED, low** — L2 = **0.4719**, below the registered interval's floor. The registration did not allow for a negative point estimate and should have |
| **B8** | 40–70 s per adjudicated decision, 1,100–2,500 s per battle | **HALF** — 36.9 s (just below) and 950 s (below). The cell was cheaper than registered because its games were shorter (41.7 decisions/battle) |
| **B9** | pairs achieved ∈ [30, 110] | **HELD** — **80**, stopped at the registered CPU deadline |
| **B10** | ≤ 3 % of live rollouts raise under node | **REFUTED** — **15.3 %** of pairs failed (hazard 2) |

---

## 9. Ready-to-append ledger paragraph

> ### 🎯 WHERE IS THE PLAYOFF GATE'S OPERATING POINT? — **the shipped 2·SE bar takes 4 % of the available value and its gain is NOT DETECTED; opening it to 0.5·SE takes 6.2× more offline and makes the arm act on 11.9 % of decisions instead of 0.4 % — and 🚨 THE MIRROR WIN RATE GOES DOWN, to 0.4719 [0.4021, 0.5416].** Branch (b), and the binding constraint is now neither the leaf nor the gate (2026-09-20)
>
> Record `designs/research_state/measurements/playoff_gate_operating_point_2026-09-20/`
> (`PREDICTION.md` landed on main at **`48743b41`** before the first number; JOB A at **`1bdd233e`**
> before JOB B was read). **JOB A, FREE: the production rule (`playoff.paired_stats` /
> `is_conclusive` / `decide`) IMPORTED and swept over R × SE-multiple × MIN_PAIRS on the 2026-09-19
> read's 665 banked forks** — K = 16 fresh CRN re-rolls per branch, scored against the **DISJOINT**
> banked K′ = 8 label, so no winner's curse; **0 unjoined, 0 salt mismatches, 0 short rows**, one
> CPU-minute, no new rollout. **JOB B: a 965-battle live mirror battery** at the point JOB A picks,
> `--impl node`, CPU only, ~28 CPU-h, nothing under `models/`.
>
> **THE OPERATING CURVE, in win-prob units per DECISION, beside its ORACLE CEILING.** 47.2 % of the
> banked sibling pairs are **TIED** under the K′ = 8 label and E|gap| = 0.1316, so an oracle always
> taking the better branch gains only **+0.0625/decision [+0.0528, +0.0729]** — every level is
> quoted as a fraction of that rather than as a bare number. At R = 4: **`k = 2.0` (SHIPPED)
> +0.00254 [−0.00038, +0.00611] NOT DETECTED = 0.041× oracle at a 0.0451 resolve rate** · `k = 1.5`
> +0.00977 DET (0.156×, resolve 0.1654) · `k = 1.0` +0.01269 DET (0.203×, 0.4120) · **`k = 0.5`
> +0.01570 [+0.00639, +0.02575] DET = 0.251× oracle at 0.4556** — **6.2× the shipped point's gain
> for the identical rollout budget.** At R = 16 the best point is **+0.03195 = 0.511× oracle**.
> 🚨 **THE BAR ITSELF BUYS ALMOST NOTHING: the NO-GATE rule (always take the higher R-rollout mean)
> reads +0.01570 at R = 4 — matching the best gated point to five decimals — and +0.02763 at R = 8,
> ABOVE every gated cell there**; only at R = 16 does a gate edge it, by 0.0009. The trade is real
> and it is bad: tightening lifts agreement with the label 0.665 → 0.870 but collapses coverage
> 0.456 → 0.045, and **a decision the gate declines contributes EXACTLY 0.0**, which is why the
> 0.87-agreement point is worth a sixth of the 0.67-agreement one. 🚨 **`MIN_PAIRS` is an INERT axis
> at R ≥ 4 by construction** (registered in advance as S1, asserted in code, refuses if violated)
> and at R ∈ {2, 3} it is **an OFF SWITCH, not a bar** — resolve exactly 0.0000 — which **forbids
> the cheapest useful point in the table** (R = 2, `MIN_PAIRS = 2`, +0.0150/decision at HALF the
> rollout cost of the shipped R = 4 point).
>
> 🚨 **AND IT STILL DOES NOT PAY IN GAMES — branch (b), with the sign against the arm and the
> mechanism no longer the gate.** The live cell **acted**: 6,886 decisions, **11.88 % changed**
> against the shipped gate's 0.4 % on 2026-09-19 (**a 30× larger intervention**), realized
> **R = 3.99**, live resolve **0.2785 of playoffs run** against the offline 0.4556 (ratio 0.61,
> inside the registered 0.2–0.8 band — the direction the 2026-09-19 proxy named, greedy-offline vs
> stochastic-live). **L2 = 0.4719 [0.4021, 0.5416] over 80 pairs, NOT DETECTED and BELOW 0.50**;
> `base − pfk05` = +0.0594 [−0.0125, +0.1344] and `defB − pfk05` = +0.0469 [−0.0407, +0.1375], both
> ND, both the same way. **Cost: 36.9 s per adjudicated decision, 950 s per battle — 56× `base`
> (17 s) and 19× `defB` (51 s).** The branch-(b) arithmetic, registered before the read: 11.9 %
> overruled × the offline +0.066 per overrule = +0.0079/decision, which over 41.7 decisions/battle
> would be +0.33 as a naive sum — **absurd, and that is the point: per-decision advantages on
> selected contested forks are NOT additive across a game, so the offline curve predicts a SIGN and
> nothing about size.** Power, from the arm's OWN spread: sd = 0.318, half-width ±0.070 at n = 80,
> so the cell could not have resolved +2 pp — **but the point estimate is on the wrong side, and
> even at 400 pairs 0.4719 would read [0.441, 0.503].** **Out-ranking was not the constraint
> (2026-09-19); the acting rule is not the constraint either. The two survivors this read cannot
> separate are the CANDIDATE PAIR (live the playoff adjudicates the SCREEN's top-2 — a biased
> critic sweep decisive in the policy's favour on only 15.6 % of decisions) and the ESTIMAND (a
> stochastic-mirror self-rollout is not the banked tree's greedy-sentinel rollout that priced the
> gain).**
>
> 🚨 **FIVE INSTRUMENT FINDINGS, and two of them change how this battery is read.** (1) **The
> unsearched `base` NULL is 0.5150 [0.4967, 0.5333], not 0.5000** — run for the first time at
> n = 200; it covers 0.50 so nothing is invalidated, but **every L2 bar within ±0.02 of 0.50 is
> henceforth set against `base`, not against the constant.** (2) **An arm that ACTS is read at a
> wider interval than one that does not**: `base` sd 0.132, `defB` 0.243, the playoff cell
> **0.318** — because it changes 12 % of decisions — so rule 25's 400-pair bar, calibrated on
> sd ≈ 0.2, needs **2.5× the pairs** at this sd; quote the arm's own sd when stating what a battery
> can resolve. (3) **`short_r_refusal` is ROW-level and MIS-FIRES on a loaded box**: it killed 2 of
> 8 shards at single-game R of 3.27 and 1.50 while their POOLED R was 3.96 and 3.98 and only 1.2 %
> of 165 rows sat below the floor — **it cost 17 pairs (21 % of the cell) and moved L2 by 0.0003
> (0.4719 → 0.4722)**; the right granularity is a trailing window, and `rollout_cost_s` still seeds
> at 1.0 s against a measured ~10 s. (4) **15.3 % of live rollout PAIRS failed under `--impl node`
> (2,913 of 19,055) against 16 in ~760 (2.1 %) on 2026-09-19 — 7× the rate, and far outside the
> registered ≤ 3 %**; a failed pair
> correctly contributes nothing, but it costs the gate's own `n` and is a candidate for part of the
> offline→live resolve gap. (5) **15.2 % of playoffs bought NO pair at all** (`no_budget` 728 of
> 4,775) at `--budget 120` on a box at load 40 — size the budget at the load the cell will run at.
> **EIGHT of ELEVEN JOB-A predictions and SIX of TEN JOB-B held**; the informative misses are **A5**
> (agreement ≥ 0.80 needs a TIGHTER gate than registered), **A8** (the no-gate rule is not strictly
> best — it is *indistinguishable*, which is the weaker and correct claim), **B7** (the registration
> did not allow for a NEGATIVE point estimate and should have) and **B10** (the node failure rate).
> **WHAT REMAINS:** (1) 🚨 **the gate's constants are a CLOSED question — do not spend another arm
> on them**; (2) the free next read is the **CANDIDATE PAIR**: re-run JOB A on the SCREEN's top-2
> rather than the policy's, which the banked `rand` branch partly supports and which costs nothing;
> (3) the **`q_winprob_head`** amortizer (built, OFF) — 36.9 s per adjudicated decision is a
> teacher's budget, not an inference budget, and that conclusion is now backed by a cell that
> actually acted. Tag: **MEASURED (MAJOR) · THE SHIPPED 2·SE GATE IS AT THE WRONG END OF ITS CURVE
> (0.041× oracle, NOT DETECTED) AND THE BAR BUYS NOTHING OVER PLAIN RE-RANKING · MIN_PAIRS IS INERT
> AT R ≥ 4 AND AN OFF SWITCH BELOW IT · 🚨 OPENING THE GATE MADE THE ARM ACT 30× MORE AND THE WIN
> RATE WENT DOWN (L2 0.4719 [0.402, 0.542]) — NEITHER THE LEAF NOR THE RULE IS THE CONSTRAINT · the
> unsearched base null is 0.515, not 0.500 · an arm that acts needs 2.5× the pairs**.
