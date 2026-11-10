# BOUNDARY-1's from-scratch run: DRAFT design (2026-11-09)

**Status: the design behind a REGISTERED experiment. The orchestrator adopted §7's 2 × 2 on 2026-11-09 and the
registration is [`PREREG.md`](PREREG.md), which wins any disagreement with this file** (breadth is decided on the
in-house cross there, not on Kakuna; the seeds, looks and tooling are as built). This file replaced
`EXPERIMENT_BACKLOG.md` §2.0's MF-5 card and stays as the argument for the options. Explicit-only: nothing
auto-updates it. Rows whose facts moved before registration were corrected on 2026-11-09 and say so.

Names used throughout. **P\*** is the plateau baseline (`rb_x26_s2601` at 280,000,170 steps, 2,609,183 parameters).
**Kakuna** is Metamon's best public model. **home72** is our 72 training teams. **U** is the three unfamiliar
panels at equal weight: **H** (`heldout72_hpfix`, ladder teams), **F** (`forum72`, forum teams) and **P72** (`p72`,
procedural teams). **BOUNDARY-1** is the next retrain boundary: main's code after the opponent-side fidelity fixes
(config v167, `566d878b2`; E13, "their max HP as a belief", `../../../endstate/obs_enrichment_backlog.md`, built since at `3b1733f50`, v168). A
**head-to-head cross** is every seed of one arm against every seed of the other, mirrored pairs, both sides greedy
(`python -m main.h2h play-many --cells`, statistic `main.h2h.cross.cross_stat`).

## 0. The design on one page

- **The question.** Does a network trained FROM SCRATCH with a changed model end stronger than the production
  recipe trained from scratch at the same code? The plateau-fork frame has read one lever at a time on P\* and
  cannot test a model that is different from step 0.
- **Both options are the same eight runs.** Four seeds an arm is the size at which a +4 pp gain clears the measured
  seed-to-seed spread (§3, §5). Four production-width runs and four wide runs, each to 100M steps, cost
  **≈ 207 GPU-h** (≈ 9 days of our card), whichever option is picked. The wide arm dominates the bill, so the
  options differ in what the new arm contains and in what a result can retire, not in price.

| | option B, the bundle | option S, one lever | the same runs crossed 2 × 2 (§7) |
|---|---|---|---|
| control arm | 4 × production (128 wide, the pool) | 4 × production | 2 × production, 2 × production + breadth |
| new arm | 4 × (256 wide + team breadth from step 0) | 4 × 256 wide | 2 × wide, 2 × wide + breadth |
| GPU-h to the 100M look | ≈ 207 (180–227) | ≈ 207 (180–227) | ≈ 207 (180–227) |
| P(WIN) at a true +4 / +5 / +6 pp (seed SD 1.76 pp) | 0.81 / 0.95 / 0.99 | 0.81 / 0.95 / 0.99 | 0.70 / 0.89 / 0.97 per factor |
| P(NO-USEFUL-GAIN) at a true 0 | 0.80 | 0.80 | 0.70 |
| a WIN says | the bundle is stronger; which part is unknown | width is stronger | which factor is stronger |
| a null retires | nothing (either part may have masked the other) | width from scratch at this size | each factor on its own |
| builds before launch | width landing, E13, the kits (§8) | the same | the same + the 2 × 2 statistic |

- **Recommendation (§10).** Between B and S: **S**. It costs the same, its WIN and its null both mean something,
  and the one built element B adds (team breadth) has a measured in-house cost that is expected to cancel its gain
  on the meter that has power. Better than either, at the same price: the 2 × 2, which keeps one-lever attribution
  AND contains the bundle cell.
- **What the review got right and wrong (§11).** Its cost per run checks out (X26: 19.6 GPU-h to 100M, 56.5 to
  280M). Its one-seed-an-arm pair does not: one seed an arm cannot clear the seed-to-seed spread it cites, and that
  spread ("25 Elo") is one pair of pre-era seeds. Properly powered, the pair costs ≈ 3× its ≈ 60–80 GPU-h.
- **Honest expectation.** The backlog's own forecast for width is +1.5 pp (−1, +5). At a true +2 pp this design
  reads WIN with probability 0.28; no affordable seed count changes that (8 seeds an arm: 0.62 for ≈ 414 GPU-h). The
  run is sized to detect the LARGE effect the review predicts and to retire the scale bet if it is not there.

## 1. The frame

- **Why from scratch.** Six breadth reads and three capacity reads at the plateau are null or undecided (§2).
  Growth at a fork is a different intervention from width at step 0 (N4's own §6.2), so the fork frame has no test
  of the model's size.
- **The base both arms share** is BOUNDARY-1: main at config v167, plus E13 (landed `3b1733f50`, v168), plus whatever
  X60-T's verdict routes into the team mix (`EXPERIMENT_BACKLOG.md` §2.0, "X60-T's outcome branches"). Both arms
  are `--arch production`, which supplies the architecture and the recipe; the new arm types only its lever.
- **The comparator is a fresh control at the same code, not P\*.** Three reasons.
  1. P\* was trained at pin `9c033489` (config v158). Main's v167 moves 96 % of P\*'s rows (54 top-action changes;
     `566d878b2`'s census), so P\* at main plays on an operator it was not trained on.
  2. P\* is one seed. The seed-to-seed SD is 1.76 pp (§3); a one-seed comparator carries it with no estimate.
  3. P\* is 280M deep and X26 carried five ride-along heads. X26's 100M snapshot fixes the depth but not 1 or 2.
- **P\* returns at the adoption step** (§5.4): a new lineage is adopted against the incumbent, each model reading
  Kakuna at its own pin, and there the code difference is part of what ships.
- **The control arm is not wasted**: it is BOUNDARY-1's own production lineage (MF-2's fidelity changes trained
  in), and its head-to-head against X26's matched-step snapshot is the REPORTED read of whether those changes moved
  strength (one seed on X26's side: the single-arm caveat applies).

## 2. The levers and their evidence

"Built" means a flag on main today. Costs of the unbuilt are in §8. Every interval is 95 % unless said.

| lever | built? | evidence for | evidence against / hazard | in B | in S |
|---|---|---|---|---|---|
| **Width**: `--d-model 256 --trunk-ffn-dim 512` (6,258,079 parameters, 2.40×) | **on main** since `0220f8408` (config v169; corrected 2026-11-09, it was on a branch at draft time) | At the plateaus P\* (2.61M) is 7.3–10.5 pp above the 1.42M toy against Kakuna (`UNDERSTANDING.md` §4.9; DESCRIPTIVE, the two differ by more than width). Kakuna is 142.8M parameters. | Every growth screen missed: N4 toy +0.46 pp [−0.28, +1.20]; X48 in-house −0.55 [−1.32, +0.21] (its Kakuna F point +2.06 [−0.06, +4.19]); all growth AT a plateau. No causal evidence for width from step 0 exists. | yes | **the lever** |
| **Team breadth from step 0**: `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` (12,000 procedural teams; X61P's arm) | yes (`main.checkargs` passes at `0ecd1b4b4`) | Fork reads on U against Kakuna all point the same way: X61P U +2.52 [−0.02, +5.05], D1 +2.10 [−0.92, +5.11], D3 +2.33 [−0.69, +5.35], X52 +3.2 [−0.2, +6.8]. Familiarity travels with the team (H-CTX SUPPORTED). | None of those intervals excludes 0. X61P's clean panel F read +0.55 [−1.26, +2.36] (FINAL NULL) and its in-house strength fell −3.28 [−4.14, −2.43]. **Hazard M4:** 62 % of H's full sets and 56.5 % of P72's are in the corpus (F: 19 %), so H and P72 are generator-adjacent for this arm. | yes | no |
| Depth: `--trunk-layers 4` | yes | none measured | X47 (toy, 2 → 4 rounds, growth) −0.88 [−3.20, +1.44], UNDECIDED with the point ≤ 0 | no | no |
| A deeper critic (corrected 2026-11-09: the head is NOT linear on 128, it already has a hidden layer, which the architecture diagnosis `a8265f36b` found unused) | **no** | The critic is miscalibrated on U at turn 10: Δbias +0.076 [+0.016, +0.139] (`f200258f6`). | Its ordering (AUC) is equal on both sets, and a level bias cancels in a one-ply argmax (backlog MF-4). Width already doubles the head's input. **UNVERIFIED** that a deeper head helps. | no | no |
| A non-saturating policy tower (today `[512, 512]` tanh) | **no** (removing the tower is built: `--policy-readout trunk`) | 84–92 % of layer-1 units sit at \|a\| > 0.99 (`UNDERSTANDING.md` §5, the plasticity probe). | The same probe read NO-LOSS of plasticity (`8bebdfe31`). **UNVERIFIED** that saturation costs strength. | no | no |
| Entropy annealed 0.05 → 0.01 | **no** schedule; a typed `--ent-coef` on a fork works | Entropy is flat at ≈ 1.09 nats for 71M steps; top-action mass 0.56; greedy play beats sampled play by +23.0 pp [+9.5, +35.4] against Kakuna. The literature's range is 0–0.01. | No strength A/B at any dose. 0.05 was CHOSEN (2026-09-12) to reproduce the earlier era's entropy regime, against controls, on an entropy criterion. | no (§9) | no |
| PPO epochs (10 fresh; 5 on forks) | yes (`--n-epochs`) | A 5-epoch update is 2.0× faster. | Fresh 5 epochs read −10.56 pp on untaught teams, one seed, with ≈ 60 % of the control's dose (confounded); the fork result that adopted 5 ran on the miscompiled learner. | no (§9) | no |
| Fidelity (v166, v167, E13) | all on main (E13 at `3b1733f50`, v168; corrected 2026-11-09) | It is the base of BOTH arms, never a lever here. | forecast +0.3 pp (backlog MF-2) | base | base |

**Why B is width + breadth and no more.** A bundle arm loses if ANY element hurts, and a bundle null cannot say
which. So an element joins only if it is built and gated before the launch date, has a mechanism plus at least
descriptive evidence that it binds, and has no measured harm it cannot answer. Depth, the critic's hidden layer
and the policy tower's activation fail the second test or the first; each also adds a version bump and a
silent-bug surface to the arm we most need to trust. The recipe knobs are §9's.

**Why S is width, not breadth.** By direct evidence breadth is ahead: it is the only lever with repeated positive
points on the goal's meter. It is still the worse from-scratch experiment:
- The fork frame CAN test breadth and is doing so now (X60-T look 2). It cannot test width at all.
- Breadth's gain shows against Kakuna on U, where a fresh pair resolves ≈ 3–4 pp at four seeds an arm (§4.3), and
  its fork reads are ≈ +2 pp. Its in-house cost (−3.28 pp) lands on the head-to-head, the meter with power.
- A fresh breadth pair is already designed (`../../../endstate/design_x53_n1_fresh_breadth.md`, never run); its own
  power table reads a true +3 pp as a GAIN with probability 0.2–0.5.
- Width's plausible effect is the largest in the book (the 7.3–10.5 pp plateau gap), and its null at four seeds an
  arm retires the owner's open scale bet.

## 3. The measured seed-to-seed spread

[`seed_spread.py`](seed_spread.py) → [`seed_spread.log`](seed_spread.log), from the banked crosses of FRESH 15M
production-family runs (each cell 1,000 mirrored pairs). "Seed SD" is the SD across one arm's seeds of a seed's
mean score against the other arm's seeds, with the cell noise removed: the unit `cross_stat`'s variance is built
from.

| cross (15M finals) | seed SD, rows / columns, pp |
|---|---|
| X5 A/B look 3 (three 8 × 8 crosses) | 2.43 / 0.60, 1.81 / 1.03, 1.55 / 0.66 |
| static screen look 3 (8 × 8) | 1.66 / 1.12 |
| end-state closing test (8 × 8) | 1.60 / 1.31 |
| bf16 A/B stage 1 (4 × 4), stage 2 (8 × 8) | 1.61 / 3.09, 2.89 / 2.09 |
| **pooled (df-weighted, 14 arm groups)** | **1.76** (range 0.60–3.09) |

- 1 pp is ≈ 7 Elo near 50 %, so 1.76 pp is ≈ 12 Elo a seed and ≈ 17 Elo for the difference of two seeds.
- **Not independent:** the arm groups share seeds across looks and stages, so the 90 nominal df overstate it.
- **Unmeasured at depth.** Every row is at 15M. Only X26 reached the plateau. The one deeper hint is the entity
  toy's three seeds against the production 15M finals: SD 0.67 pp at ≈ 47M and 1.82 pp at its 100M cap
  (`../toy_crosses_2026-10-16/RESULTS.md`; one seed carries a hazard). **UNVERIFIED** that 1.76 pp holds at 100M,
  so the rule floors each variance at 1.76 pp and estimates the rest from the run itself (§5.2), and the power
  table carries 2.5 and 3.1 pp columns.
- **Against Kakuna** the seed SD of a FRESH run is unmeasured (X53-N1's stage 0 never ran). Fork finals show none
  beyond binomial (1.46 pp against 1.44, X60-T PREREG §5.1), which does not transfer to fresh runs.

## 4. The meters

### 4.1 Decisive: the head-to-head cross, new arm against control

- Every new-arm seed against every control seed at the matched step: a 4 × 4 cross, 16 cells.
- Each cell: 1,000 mirrored pairs on home72 and 500 each on H, F and P72 (`--panel`), both sides greedy, one engine
  at the run's pin. 80,000 games a look.
- The cell's score is **N = ½ · home72 + ½ · U**, with U the equal-weight mean of the three panels. That is the
  owner's exchange rate r = 1 (`../../../STATUS.md` §6.4, decided for X62 / X63 on the Kakuna meter). Carrying it to
  this meter is a DRAFT choice the owner confirms.
- Why this meter decides: it is the only one that clears the seed spread at an affordable size (§5.3), and it
  compares the arms directly instead of through a third party.
- **The breadth guard.** For an arm that trains on `n1_proc_v1`, H and P72 are generator-adjacent (hazard M4), so
  its F component is also reported alone, and a WIN whose F component's point is ≤ 0 is classed HAZARDED.

### 4.2 Reported: curve dominance on the eval ladder

- Each run's own snapshot ladder (`<run>/snapshot_ladder/ladder.json`, ± 10 Elo), compared across runs only at
  matched snapshot COUNT (`python -m main.elo compare`), the newest node excluded (it reads inflated until the run
  ends).
- DOMINATES iff the new arm's seed-mean is above the control's at every matched count from 30M to 100M and at least
  three of its four seeds end above the control's median.
- It is pace evidence, never the verdict: ladders of different runs meet different opponent pools.
- Also reported from the same checkpoints, free of GPU: the new arm at the step where its cumulative GPU time equals
  the control's at 100M, against the control at 100M (the equal-compute read). The toy led at equal compute and
  lost at the plateau (`UNDERSTANDING.md` §4.9), so this is a cost statement, not a ranking.

### 4.3 Confirming: Kakuna on U, clustered by team

- Each 100M final plays Kakuna on H, F and P72, 720 games a panel (2,160 a final; 17,280 for eight finals).
- The statistic is the new arm's mean minus the control's on U, with a two-way bootstrap: TEAMS resampled within a
  panel on one index set shared by both arms, and SEEDS resampled within an arm (`UNDERSTANDING.md` §8 rule 10: the
  team is the unit; the estimator is the one the 2026-11-09 team-clustered re-analysis lands, coded as a
  `main.reads` rule before registration).
- Resolution ([`power.log`](power.log) part 2): SE 1.07–1.33 pp at a Kakuna seed SD of 1.0–1.5 pp, so it detects
  ≈ 2.7–3.3 pp with power 0.80. It confirms or contradicts a WIN; it cannot find a +2 pp gain on its own.

## 5. Option B: the bundle

### 5.1 Arms

| arm | argv beyond `--arch production --steps 100000000 --seed <s>` | seeds | parameters |
|---|---|---|---|
| **C**, the control | none | 4 | 2,609,183 |
| **W+B**, the bundle | `--d-model 256 --trunk-ffn-dim 512 --allow-nonproduction-arch --team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` | 4 | 6,258,079 |

- The same four seed numbers in both arms. Runs alternate C, W+B, C, W+B, … so a fault leaves the arms balanced.
- All eight run to 50M first, then all eight resume to 100M (§5.2's interim look).
- If X60-T reads ADOPT-CANDIDATE, human teams go into the base of BOTH arms; `--team-mix` and `--human-data
  teams:` cannot combine (one team lever a run), so the bundle then has no breadth element and **B becomes S**.

### 5.2 The rule (one decision, at 100M)

`cross_stat` on the 4 × 4 matrix of N: Δ̂ = mean − 50, V̂ = s²_rows / 4 + s²_columns / 4 with each s² floored at
(1.76 pp)², a 90 % interval on 6 df (t = 1.943).

| class | condition | what follows |
|---|---|---|
| **WIN** | the interval's lower end > 0 | §5.4 (the plateau and adoption), then §5.5 (the ablation) |
| **LOSS** | its upper end < 0 | the bundle is retired as specified; the control is BOUNDARY-1's lineage |
| **NO-USEFUL-GAIN** | not WIN, upper end < +4 pp | stop; a bundle null retires neither element (§10) |
| **UNDECIDED** | otherwise | to the owner, with the price of two more seeds an arm (≈ 104 GPU-h) |

- **+4 pp** is the smallest gain worth an arm that costs ≈ 1.65× a step: it is goal (b)'s gap (P\* is about 4 pp
  short of Kakuna on unfamiliar teams, `../../../STATUS.md` §1).
- **The 50M interim look stops only on harm:** the same statistic, STOP iff its upper end < −2 pp. No futility
  stop, because a gain may arrive late. The 50M and 75M values are otherwise reported.
- Kakuna on U (§4.3) is read at 100M on every outcome except an interim stop. A WIN whose Kakuna interval lies
  wholly below 0 is IN-HOUSE-ONLY and goes to the owner; a Kakuna gain without a WIN is reported, never adopted.

### 5.3 Power ([`power.py`](power.py) → [`power.log`](power.log), 50,000 draws a cell)

P(WIN) by the true gain in N, the number of seeds an arm and the true seed SD:

| seeds an arm | seed SD, pp | +2 | +3 | +4 | +5 | +6 | +8 | P(NO-USEFUL-GAIN) at 0 | false WIN at 0 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.76 | 0.03 | 0.07 | 0.16 | 0.31 | 0.48 | 0.77 | 0.16 | 0.00 |
| 3 | 1.76 | 0.16 | 0.35 | 0.60 | 0.80 | 0.92 | 0.99 | 0.59 | 0.01 |
| **4** | **1.76** | **0.28** | **0.56** | **0.81** | **0.95** | **0.99** | **1.00** | **0.80** | **0.02** |
| 4 | 2.5 | 0.22 | 0.40 | 0.60 | 0.77 | 0.90 | 0.99 | 0.57 | 0.03 |
| 4 | 3.1 | 0.19 | 0.31 | 0.47 | 0.62 | 0.77 | 0.93 | 0.43 | 0.04 |
| 5 | 1.76 | 0.38 | 0.71 | 0.92 | 0.99 | 1.00 | 1.00 | 0.90 | 0.02 |
| 8 | 1.76 | 0.62 | 0.92 | 0.99 | 1.00 | 1.00 | 1.00 | 0.94 | 0.03 |

- **One seed an arm has no row**: with no within-arm variance there is no interval. Two seeds an arm read a true
  +4 pp as a WIN 16 % of the time.
- **Four** is the smallest size that reads +4 pp at 0.8 under the measured spread and retires a true null at 0.8.
  If the spread at 100M is 2.5 pp, +4 pp reads at 0.60 and the result is likelier UNDECIDED than wrong.

### 5.4 After a WIN: the plateau and adoption

- The two new-arm seeds nearest their arm's median at 100M continue to the plateau (`python -m main.plateau tick`,
  the PR rule), ≈ 61 GPU-h each if the plateau is again near 280M (GUESS: +180M at the wide rate).
- **Adoption is against P\*:** the candidate's plateau checkpoint and P\* each read Kakuna on U at their own pins
  (§4.3's statistic, teams paired). P\*'s banked U is 0.4947 [0.4744, 0.5150]. The control arm stops at 100M; "the
  bundle plateaus higher than the control would" is NOT claimed.

### 5.5 The ablation, conditional on a WIN

- Leave one element out: **W alone** (wide, the pool) and **B alone** (production width + breadth), to 100M, each
  crossed against the banked bundle seeds and the banked control seeds by §5.2's statistic.
- At two seeds each: ≈ 104 GPU-h, and a 4 v 2 contrast reads a true +4 pp at well under 0.5. At four seeds each:
  ≈ 207 GPU-h, the price of the original run again.
- So a bundle WIN is followed by ≈ 100–200 GPU-h before anyone knows what to keep. §7 buys that attribution inside
  the first 207 GPU-h.

## 6. Option S: width alone

| arm | argv beyond `--arch production --steps 100000000 --seed <s>` | seeds | parameters |
|---|---|---|---|
| **C**, the control | none | 4 | 2,609,183 |
| **W**, wide | `--d-model 256 --trunk-ffn-dim 512 --allow-nonproduction-arch` | 4 | 6,258,079 |

- The frame, the meters, the rule, the interim look and the power table are §4–§5.3's, unchanged; the breadth guard
  does not apply.
- **WIN** → §5.4, with no ablation owed: the lever is the arm.
- **NO-USEFUL-GAIN** → width from step 0 at 2.4× the parameters is retired for a 100M budget; the owner's scale
  question (`../../../STATUS.md` §6.5) is answered "not at this size". It does not speak to 10× or to the plateau.
- X60-T's ADOPT-CANDIDATE puts human teams into both arms and changes nothing else.

## 7. The same eight runs, crossed 2 × 2

Two runs in each of four cells: production, production + breadth, wide, wide + breadth. Same 207 GPU-h.

- **Each factor is read at 4 runs against 4** (its ON runs against its OFF runs, 16 cells), with the pooled
  within-cell variance on 4 df: P(WIN) 0.70 / 0.89 / 0.97 at +4 / +5 / +6 pp, against 0.81 / 0.95 / 0.99 for the
  straight 4 v 4 (`power.log` part 1b). That is the price of the attribution.
- **It contains the bundle cell and the interaction.** The review's strongest argument for a bundle is that a
  narrow network pays for breadth with in-pool strength (X61P's −3.28 pp) and a wide one may not. Only the wide +
  breadth cell against the other three reads that; at two seeds a cell it is a point estimate, reported.
- **Width is read on the head-to-head (§4.1), breadth on Kakuna on U (§4.3)**, with the head-to-head as breadth's
  cost meter, exactly as the fork designs read it.
- It needs one more coded statistic (§8) and is void if X60-T routes human teams into every run (then it is S).

## 8. Cost, and what must be built first

**Measured** (the run directories' checkpoint timestamps, 2026-11-09):

| quantity | value | source |
|---|---|---|
| a production run to 25M / 50M / 100M | 4.40 / 9.48 / 19.55 GPU-h | `rb_x26_s2601`, alive time between restarts |
| to P\*'s plateau (280M) / to 380M | 56.5 / 77.0 GPU-h (58.2 / 80.3 h calendar) | the same |
| a fresh run's first 15M | 2.47–2.68 h (1,560–1,689 steps/s) | `rb_es_prod_s2001`–`s2008` |
| a P\* + 20M fork | 3.8–4.1 h (1,368–1,477 steps/s) | `rb_x35_c0_s3501`–`s3505` |
| Kakuna games a lane-hour | ≈ 144 | X60-T PREREG §23.9 |
| an 8 × 8 cross, 128,000 games | 52 min under the GPU lock | the end-state closing test |

- X26 carried five detached ride-along heads; a plain production run may be slightly cheaper. **UNVERIFIED.**
- **The wide arm's cost a step is a GUESS: 1.65× (1.3–1.9×).** On CPU the learner's forward + backward is 1.94× and
  the policy forward 1.27–1.48× (`536aed01b`); the learner is ≈ 60 % of a bf16 cycle (42 of ≈ 70 s). GPU step
  time, the bf16 compile and memory at 256 are unmeasured: **a 15-minute GPU pilot is a launch prerequisite**, and
  a measured ratio above 1.9× sends the budget back to the owner.

| per option (B, S and the 2 × 2 alike) | GPU-h | CPU | wall |
|---|---|---|---|
| phase 1, eight runs to 50M | ≈ 100 (4 × 9.5 + 4 × 15.6) | — | ≈ 4.4 days |
| phase 2, 50M → 100M | ≈ 107 (4 × 10.1 + 4 × 16.6) | — | ≈ 4.6 days |
| **to the decision** | **≈ 207 (180–227)** | | **≈ 9 days** |
| three head-to-head looks (50 / 75 / 100M), 80,000 games each | ≈ 0.6 each under the GPU lock, between runs | or ≈ 8 h a look on the three agent cores | |
| Kakuna on U at 100M, 17,280 games | — | ≈ 120 lane-h | ≈ 40 h on 3 lanes |
| after a WIN: two candidates to the plateau | ≈ 122 (GUESS) | ≈ 30 lane-h (adoption read) | ≈ 5 days |
| B only, after a WIN: the ablation | ≈ 104–207 | ≈ 60–120 lane-h | ≈ 4–9 days |

**Builds** (agent-days are estimates; none is started by this draft):

| build | needed by | agent-days | state |
|---|---|---|---|
| Land per-run width: rebase `mf5-width-lever`, renumber its config version (it claims v167, which main now uses), gate, ship | B, S, 2 × 2 | 0.5–1 | **DONE** `0220f8408` (config v169) |
| The GPU pilot of the wide arm (step time, bf16 compile, memory) and a check that the head-to-head engine pairs a 128 and a 256 model | B, S, 2 × 2 | 0.25 + a 15-minute GPU window | not started: `PREREG.md` §13 items 4 and 5 |
| E13, their max HP as a belief (an operator-input change, a version bump) | the base of every option | 2–3 | **DONE** `3b1733f50` (v168) |
| The fresh-run kit: arm describe / check (pin, seed, argv) in the shape of `src/main/x61/arms.py`, and the resume-at-50M check (§12 item 6) | all | 0.5 | **DONE** `src/main/b1/arms.py`, `resume_check.py` |
| The multi-panel cross: cells plan, N per cell, the floored `cross_stat` rule as code with its self-check | all | 0.5 | **DONE** `src/main/b1/cross.py`, `look.py` |
| The Kakuna-on-U rule with the team × seed bootstrap (`main.reads`) | all | 0.5 | **DONE** rule `b1_kakuna` (`src/main/b1/kakuna_rule.py`); the registered interval is analytic, the bootstrap reported |
| The 2 × 2 statistic (pooled within-cell variance) | 2 × 2 only | 0.5 | **DONE** `src/main/b1/cross.py` |
| NOT needed by any option as drafted: a critic hidden layer (1–2), a policy-tower activation flag (0.5), an in-loop entropy schedule (1–1.5), E9 (2–4) | — | — | — |

E13 was the long pole and is built. The kits, the cross statistic, the Kakuna rule and the 2 × 2 statistic are built
(`PREREG.md` §15); the GPU pilot is `PREREG.md` §13. E9 joins only if it lands before the launch date; it does not
hold it.

## 9. The recipe knobs: screen them, do not fold them in

`ent_coef` and the PPO epoch count are properties of the recipe, not of the model. Folding either into the new arm
would give a bundle one more way to lose for a reason that is not the model. **A knob that passes its screen moves
into the BASE of both arms; it never rides in the new arm alone.** Both screens fit in the launch gap (§9.3).

### 9.1 Entropy: a three-seed fork of P\*, ≈ 12 GPU-h

- **Arm:** P\* + 20M with `--ent-coef 0.0125` typed (a fork applies a typed value; no schedule build), three
  seeds, at the pin of the banked plateau controls (the ten C0 finals), read in the X35 frame, which resolves ± 2 pp
  with three seeds a side.
- **Why a fork and not a small-dose fresh screen.** The hypothesis is about the late regime (a policy kept too
  flat: 0.56 top-action mass, sampled play 23 pp under greedy). A 15M fresh curve reads early learning speed, and
  that screen was in effect run on 2026-09-12, when 0.05 was chosen on an entropy criterion.
- **This is X23's question.** The backlog superseded X23 on 2026-11-09 because no entropy COLLAPSE was found. The
  hypothesis here is the opposite failure, too much entropy, which the collapse battery does not address (§11).
- **Limit:** a fork cannot exclude that annealing from step 0 finds a different basin. If the fork reads a gain, an
  in-loop schedule (1–1.5 agent-days) is built for the plateau continuation (§5.4).

### 9.2 Epochs: the standard 8 × 8 fresh cross at 15M, ≈ 35–41 GPU-h

- **Arms:** `--arch production` against `--arch production --n-epochs 5` with the learning rate matched by DOSE
  (`python -m main.dose`; X27's design), eight seeds each to 15M, the owner's closing rule (PASS iff the 90 %
  interval's upper end > 0 and its lower end > −2.0 pp; `../endstate_closing_test_2026-10-09/closing_rule.py`).
- **What a PASS buys:** five epochs on both arms. The update is 2.0× faster and ≈ 60 % of the cycle, so ≈ 30 % of
  the main run's GPU-h (≈ 60 of 207). **UNVERIFIED** at bf16 and at depth: a 15M PASS does not cover 100M.
- **Default if it is not run or does not PASS: ten epochs on both arms.** It never holds the launch.

### 9.3 The gap they fit in

The GPU frees ≈ Tue 11-10 14:00 PST (§10's table); E13 holds the main launch until ≈ Fri 11-13. That is ≈ 60–70
GPU-h, enough for MF-1 and MF-3 (≈ 5 GPU-h), the entropy fork (≈ 12) and the epoch cross (≈ 35–41).

## 10. The queue, and the recommendation

| when (PST) | GPU | note |
|---|---|---|
| now → Tue 11-10 ≈ 11:20 | X60-T look 2's last five runs | ≈ 4 h 02 min a run, measured on s3504 / s3505 |
| Tue 11-10 ≈ 11:20 → 14:00 | X66 (ExIt-0H students, ≈ 2.25 GPU-h) | registered on a branch; its start is UNVERIFIED |
| Tue 11-10 → Fri 11-13 | MF-3 step b (≈ 1–2); §9's screens if the orchestrator wants them | the builds of §8 run in parallel on agents |
| **≈ Fri 11-13** | **launch, any option** | gated by E13 (2–3 agent-days from 11-10), the width landing and its pilot |
| ≈ Wed 11-18 | the 50M look (harm stop only) | |
| ≈ Sun 11-22 | the 100M head-to-head verdict | |
| ≈ Tue 11-24 | Kakuna on U | X60-T's reads hold the lanes from ≈ 11-12 for ≈ 4 days: no conflict |

- **Without E13** the launch could be ≈ Wed 11-11 (the width landing, its pilot, the kits), and the lineage would
  owe a second retrain boundary. Waiting two days for a nine-day run is the better trade.
- Options B, S and the 2 × 2 launch the same day: none needs a build the others do not, except the 2 × 2's
  statistic, which is needed only at the first look.
- A second card (`../../../ops/cloud_gpu_plan.md`) would halve the nine days; the price is the owner's open question.

**Recommendation: S over B; the 2 × 2 over both.**
1. **Same price.** The wide arm sets the bill in every option.
2. **A B null retires nothing.** Breadth's measured in-house cost (−3.28 pp at a fork) and its unfamiliar-team gain
   (≈ +2 pp) are expected to roughly cancel in N, so breadth adds variance and a failure mode to the bundle on the
   meter that has power, and its own gain shows only on the meter that does not.
3. **A B win is not actionable until ≈ 100–200 more GPU-h of ablation** (§5.5).
4. **The bundle's real argument is an interaction** (a wide network may absorb breadth without the in-pool cost).
   The 2 × 2 tests exactly that, at the same price, and keeps the owner's one-lever rule for each factor.
5. **The review's wider bundle** (critic layer, policy activation, entropy anneal, depth) is 3–5 agent-days of
   unbuilt structure with no measured evidence that any of it binds. Each can be a later single lever on the
   winning width. Read since on P\*, claim by claim (REPORTED):
   [`../arch_diag_2026-11-09/README.md`](../arch_diag_2026-11-09/README.md). It leaves all three out of the base and
   corrects §2's critic row: the head already has a hidden layer.

## 11. FINDINGs: the review against the record

1. **"25-Elo seed floor" is one pair of pre-era seeds** (`ai_v13_04_flywheel_winprob`, ledger 2026-09-18, "THE 75M
   RUN-LEVEL FLOOR IS MEASURED"): n = 2, another era. The in-era record is §3: seed SD 1.76 pp (≈ 12 Elo) at 15M,
   so a two-seed difference of 25 Elo is unremarkable (1.5 SD). The number is roughly right; its basis is not.
2. **One seed an arm cannot be read against any seed floor.** The review's arm pair has no within-arm variance; its
   "≥ 2× the floor" bar imports a 15M (or pre-era 75M) spread to 100M. Powered for +4 pp, the pair is four seeds an
   arm and ≈ 207 GPU-h, not ≈ 60–80.
3. **Its cost per run checks out.** X26: 19.55 GPU-h to 100M and 56.5 to 280M (review: ≈ 21 and ≈ 59, which match
   the calendar hours). The backlog's "X26's 380M ≈ 79 GPU-h" is 77.0 alive, 80.3 calendar.
4. **X61P's H interval exists and spans zero**: +4.14 pp [−1.08, +9.34]
   (`../../../endstate/design_beat_metamon_path.md` §6.5b). And H is not independent of the procedural corpus: 62 %
   of H's full sets are in `n1_proc_v1` (hazard M4), so "the H gain is not P72 familiarity" does not follow.
5. **`ent_coef` 0.05 was not untested**: it was chosen on 2026-09-12 against controls to reproduce the earlier
   era's entropy (ledger, "`ent05` PASSES"). What has never been run is a STRENGTH A/B at any dose.
6. **"≈ 4× trunk parameters" is not what is built.** The width flag at 256 gives 2.40× the parameters (extractor
   2.8×); no other width has been measured, and the actor tower, the heads' readouts and the depth stay fixed.
7. **X23's supersession does not cover the review's entropy hypothesis** (§9.1): the collapse battery tests too
   little entropy, the review argues too much.

## 12. FINDINGs: hazards and things not verified here

1. The seed SD at 100M, and any fresh run's seed SD against Kakuna, are unmeasured (§3). The rule's variance floor
   and the 2.5 / 3.1 pp power columns are the guard; the first four control runs measure both.
2. The wide arm's GPU cost, bf16 compile and memory are unmeasured (§8); the 207 GPU-h figure moves with them.
3. (Resolved 2026-11-09: width landed on main at `0220f8408` as config v169.) Whether the head-to-head engine pairs a 128 and a 256 model is unverified.
4. Carrying the owner's r = 1 from the Kakuna meter to the head-to-head N is this draft's choice (§4.1).
5. Hazard M4 makes H and P72 unclean for any arm trained on `n1_proc_v1`; set-level overlap of the other corpora
   with the panels is not stated in any doc found.
6. A run stopped at 50M and resumed must be the same experiment as one launched for 100M. No schedule was found
   that reads the total step count (X26's learning rate is adaptive), but this was not executed: the kit checks it.
7. The equal-weight mapping from head-to-head pp to Kakuna pp is unknown; +4 pp in N is not +4 pp against Kakuna.
8. `../../../STATUS.md` §3 and §6.5 now carry this draft's size beside the first pricing; §6.5's recommendation
   ("yes, staged") was written for the one-seed pair and is the orchestrator's to rewrite.
9. X26's entropy trajectory was not re-read for this draft (its last logged `train/entropy_loss` is −0.996 at 380M);
   the entropy figures are the ledger's. Its snapshot ladder has nodes only from 164M (+32 Elo by 280M, +50 at 376M,
   ± 10); the older rating file, a different fit, reads 2058 / 2084 Elo at 50M / 100M and ≈ 2115 at 165M. So the
   control is still rising at 100M: the decisive look is a pre-plateau read, as §4.2 says of the toy.
10. The alive GPU-hours of §8 include each segment's startup, compile and in-loop evals, and come from one source
   (checkpoint and event-file timestamps); the rollout / train / eval split of a cycle was not aggregated.
