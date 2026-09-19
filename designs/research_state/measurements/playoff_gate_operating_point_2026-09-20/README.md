# WHERE IS THE PLAYOFF GATE'S OPERATING POINT? — the acting rule swept on banked dice, and tested in games

*JOB A measured 2026-09-19 20:20 UTC, **free**: the production decision rule IMPORTED and swept
over R × SE-multiple × MIN_PAIRS on the **665 banked forks** of
[`rollout_leaf_kcurve_2026-09-19/`](../rollout_leaf_kcurve_2026-09-19/README.md) (K = 16 fresh CRN
re-rolls per branch, read against the DISJOINT banked K′ = 8 label) — **no new rollout was paid
for**. JOB B is the live mirror battery at the point JOB A picks. CPU only
(`CUDA_VISIBLE_DEVICES=""`), `nice 15`, `--impl node`, **nothing written under `models/`**; the GPU
holds `ai_v13_09_wcont` and was not touched, 8000 / 8001 untouched.*

**Pre-registered in [`PREDICTION.md`](PREDICTION.md), landed on main at `48743b41` before the first
number of this read existed.**

> 🚧 **JOB B (the live battery) IS RUNNING as this is written.** Everything below §5 is JOB A only;
> the battery tables, the branch and the ledger paragraph land in the follow-up commit.

---

## 1. VERDICT (JOB A) — **the gate is throwing away five sixths of the value it could be taking**

**1 — 🚨 THE PRODUCTION OPERATING POINT CAPTURES 4 % OF THE AVAILABLE VALUE, AND ITS GAIN IS NOT
EVEN DETECTED.** At the shipped `SE_MULTIPLE = 2.0`, `MIN_PAIRS = 4`, R = 4, the expected value gain
is **+0.00254 win-prob units per decision [−0.00038, +0.00611] NOT DETECTED** — **0.041×** an oracle
that always takes the better branch. Loosening the same rule to `SE_MULTIPLE = 0.5` at the same R
and the same cost takes **+0.01570 [+0.00639, +0.02575] DETECTED**, **0.251× the oracle** — a
**6.2× larger gain for the identical rollout budget.**

**2 — 🚨 AND THE GATE IS BUYING ALMOST NOTHING OVER "JUST RE-RANK".** The no-gate rule — always take
the branch with the higher R-rollout mean, no bar at all — reads **+0.01570 [+0.00620, +0.02556]**
at R = 4, **identical to the best gated point to five decimal places**, and **+0.02763** at R = 8,
**above every gated cell there.** Only at R = 16 does a gate (at `k = 0.5`) edge it, by 0.0009 —
far inside the interval. **Over the whole grid the gate's entire contribution is within noise of
zero; what it reliably does is cost coverage.**

**3 — THE TRADE THE GATE MAKES IS REAL AND IT IS A BAD TRADE.** Tightening does exactly what it was
designed to do: agreement with the K′ = 8 label climbs **0.665 → 0.668 → 0.760 → 0.870** as `k` goes
0.5 → 1.0 → 1.5 → 2.0 at R = 4, and the gain per RESOLVED decision climbs with it. But the resolve
rate collapses **0.456 → 0.412 → 0.165 → 0.045**, and the product — the only quantity a game sees —
falls monotonically. **A 10× loss of coverage is not paid for by a 1.3× gain in per-decision
quality.**

**4 — 🚨 `MIN_PAIRS` IS AN INERT AXIS OVER THE WHOLE REGISTERED GRID, AND THAT WAS REGISTERED IN
ADVANCE.** `is_conclusive` bars on `n < min_pairs` where `n` is the completed pair count, so at
R ∈ {4, 8, 16} every fork has `n = R ≥ 4` and `MIN_PAIRS ∈ {2, 4}` is **bit-identical** — verified,
and the script REFUSES if it ever is not. The axis bites only at **R ∈ {2, 3}**, which this read
swept as the registered remedy: there `MIN_PAIRS = 4` is not a bar but an **off switch** (resolve
exactly 0.0000 at every SE multiple), while `MIN_PAIRS = 2` resolves 0.35 / 0.39 and takes
**+0.0150 / +0.0148 per decision** — *more than the shipped R = 4 point, at half its rollout cost.*

**5 — THE CEILING IS LOW, AND SAYING SO IS THE POINT.** 47.2 % of the banked pairs are **TIED under
the K′ = 8 label** and E|label gap| is **0.1316**, so an ORACLE that always takes the better sibling
gains only **+0.0625 win-prob units per contested decision [+0.0528, +0.0729]**. Every EV number
here is quoted as a fraction of it. **A rule at 0.25× the oracle on 45 % of contested decisions is
the honest size of the prize** — not the +0.078 pairwise-accuracy headline, which is an accuracy,
not a value.

**6 — THE DOSE STILL MATTERS MORE THAN THE BAR.** The single largest move in the table is R, not
`k`: at a fixed `k = 0.5` the per-decision gain goes **+0.0157 → +0.0259 → +0.0320** across
R = 4 → 8 → 16 (0.251× → 0.415× → 0.511× the oracle). **Doubling the dice buys more than any bar
setting at fixed dice** — and costs proportionally.

---

## 2. The frame

| | |
|---|---|
| **the policy** | `models/ai_v13_02_flywheel_winprob/final_model.zip` — **arm W**, step **75,005,952**, `--critic winprob`. FROZEN; nothing was trained |
| **the states** | the **665** banked forks of the 2026-09-19 K-curve — held-out contested CRN forks, each with all three branches at FULL K = 16 fresh re-rolls. **0 unjoined, 0 salt mismatches, 0 short rows** |
| **the gate's dice** | fresh re-rolls **8 … 8+R−1**, the same common-random-number paired differences `d_k = score(top1,k) − score(top2,k)` that `PlayoffRunner.adjudicate` forms |
| **the label** | the banked **K′ = 8** mean (re-rolls **0–7**) — **DISJOINT** from the gate's dice, so nothing here selects on the quantity it is scored against |
| **the rule** | `playoff.paired_stats` / `is_conclusive` / `decide`, **IMPORTED from the production module, never re-implemented** — a second hand-written copy of a gate is how a gate's constant gets quietly forked |
| **the flags** | `--playoff-se-k` and `--playoff-min-pairs` **already existed** and already reach `PlayoffConfig` → `decide`. **No `src/` change was needed or made** (PREDICTION.md S4) |

**Why the EV column is the objective and the agreement column is not** (registered as S2 before any
number): the K′ = 8 label is a mean of 8 Bernoulli draws, so it is an **unbiased** estimator of a
branch's value and the *expectation* of `label(pick) − label(top1)` is the true expected gain
whatever the label's variance. A pairwise ACCURACY has no such property — label noise drags every
accuracy toward 0.5 (the 2026-09-19 measured ceiling on this column is 0.6709). **So the EV column
is steered by and the agreement column is sanity-checked with.**

⚠️ **The "agrees with the 16-rollout estimate" column is BIASED UPWARD BY CONSTRUCTION** at every
R < 16: the gate's R draws are a strict SUBSET of the 16, so the scorer contains the scored. It
reads 1.0000 at R = 16 for the arithmetic reason that the gate and the scorer are then the same
number. It is published because the instruction asked for it and labelled every time it appears.

---

## 3. THE OPERATING CURVE

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

---

## 4. Reading the curve

**The shape, in one sentence.** Resolve rate and per-resolved quality trade against each other
exactly as designed, and the product — **EV gain per DECISION**, the only column a game can feel —
is **monotone decreasing in `k` at every R**. The bar is on the wrong side of its own optimum at
every dose.

**Why the no-gate rule is so hard to beat here.** At R = 4 the gate at `k = 0.5` resolves 0.4556
and the no-gate rule 0.4602; the 0.0046 difference is forks whose paired mean is non-zero but
smaller than half its own SE, and those are the forks with the smallest true gaps, which contribute
almost nothing either way. **The gate and the open rule are the same rule at low `k`**, and the
production `k = 2.0` is where they finally differ — by discarding 90 % of the coverage.

**The 2026-09-19 self-check.** The `k = 2.0`, `MIN_PAIRS = 4` column reproduces that read's
published resolve rates **exactly** — 0.0451 / 0.1068 / 0.1248 at R = 4 / 8 / 16 — and its
agreements — 0.8696 / 0.8333 / 0.9221. Prediction **A1 HELD**; the instrument is the same
instrument.

**What "0.87 agreement" was actually worth.** The 2026-09-19 read published that number with its
coverage beside it, as a selection effect. This read prices the selection: **the 0.87-agreement
point takes +0.0025 per decision and the 0.67-agreement point takes +0.0157.** A more-often-right
rule that almost never fires is worth a sixth of a more-often-wrong rule that fires ten times as
often — because a decision the gate declines contributes **exactly 0.0**, not a small positive.

---

## 5. Hazards and findings about the instrument

1. 🚨 **`MIN_PAIRS` is an inert axis at R ≥ 4** (verdict 4). A sweep that reported it as a tuned
   knob would have published two identical columns as if they were a comparison. The script asserts
   it and REFUSES if the two ever differ at R ≥ 4 — an axis that *must* be degenerate is worth a
   guard precisely because a difference there means the harness is not reading the production rule.
2. 🚨 **`MIN_PAIRS = 4` is an OFF SWITCH at R ∈ {2, 3}, not a bar** — resolve is exactly 0.0000 at
   every SE multiple, because `n = R < 4` can never clear it. The 2026-09-19 read saw this live and
   correctly called it "below `MIN_PAIRS` it cannot conclude at all"; the sweep shows the same
   constant makes the two cheapest doses **unusable**, while at `MIN_PAIRS = 2` they take +0.0150
   per decision — more than the shipped R = 4 point at half the rollouts. **The cheapest useful
   operating point in this entire table is R = 2, `MIN_PAIRS = 2`, `k ≤ 1.0`**, and the shipped
   constant forbids it.
3. ⚠️ **The 16-rollout agreement column contains its own scorer** below R = 16 and is labelled
   everywhere it appears. It is reported, never barred against — the same treatment the 2026-09-19
   read gave the split-half `q`.
4. ⚠️ **The oracle ceiling is a LABEL oracle, not a truth oracle.** It is `E[max(lab2 − lab1, 0)]`
   on a K′ = 8 label, so it inherits the label's own noise: an oracle reading a noisier label would
   score lower, and the true ceiling is somewhat higher than +0.0625. It is a common yardstick for
   comparing rules on this data, not a physical bound.
5. ⚠️ **Every fork here is a CONTESTED, held-out, seeded fork.** A per-decision number measured on
   them does NOT multiply by a game's decision count — most live decisions are not contested forks.
   The conversion from this curve to a win rate is the battery's job and is registered in
   PREDICTION.md B7.
6. **The whole of JOB A cost one CPU-minute** — the dice were banked on 2026-09-19 and the rule is
   35 lines of arithmetic. The 2026-09-19 read's §5.1 said re-tuning the gate "costs nothing"; that
   was literally true, and it is the reusable lesson about banking dice at full K.

---

## 6. Files

| | |
|---|---|
| [`PREDICTION.md`](PREDICTION.md) | the registration, landed at `48743b41` before the first number |
| [`gate_curve.py`](gate_curve.py) · [`gate_curve.json`](gate_curve.json) | JOB A — the production rule imported and swept; the oracle ceiling; the MIN_PAIRS inertness guard |
| [`run_battery.sh`](run_battery.sh) · [`report_battery.py`](report_battery.py) | JOB B's cells and their read (the 2026-09-11 pairing instrument, reused) |
| [`make_table.py`](make_table.py) | renders the JSONs into this file's tables — **no number above was retyped by hand** |
| not committed | the battery rows, under `/home/goodlad/.claude/jobs/9ab51de6/tmp/gatecurve/` |
