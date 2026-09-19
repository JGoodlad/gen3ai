# PRE-REGISTRATION — HOW MANY ROLLOUTS DOES A LEAF NEED, AND DOES IT PAY IN GAMES?

*Registered 2026-09-19, landed on main BEFORE the first rollout of either job. Two jobs on one
frozen policy — **arm W**, `models/ai_v13_02_flywheel_winprob/final_model.zip` @ 75,005,952,
`--critic winprob`. JOB 1 extends the banked dice of
[`leaf_ceiling_controls_2026-09-19/`](../leaf_ceiling_controls_2026-09-19/README.md); JOB 2 is the
first battery cell whose LEAF is a rollout. CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, rust
sim bridge, ports 9700–9799, everything under `/home/goodlad/.claude/jobs/9ab51de6/tmp/kcurve/`,
**nothing written under `models/`**. A training arm holds the GPU and is not touched; 8000/8001
untouched; other agents' ports (9400–9499) untouched.*

---

## 1. The question, and why it is the one left

The 2026-09-19 controls closed the leaf question in cell **N × L**: no hand evaluator beats the
head, but the single-rollout LABEL mis-orders the two siblings **39.7 %** of the time, the
published `top1|top2` 0.545 was a measurement through a near-coin-flip, and **a FOUR-rollout leaf
beat the 75M-step critic by +0.0566 [+0.0282, +0.0831] pooled, DETECTED** — the first object in the
campaign to out-rank the head on this instrument. ⚠️ On `top1|top2` alone, the column a re-ranking
search leaf actually consumes, the same read was **+0.0275 [−0.0176, +0.0705] NOT DETECTED**.

So exactly two questions remain, and they are a DOSE and a TRANSFER:

* **JOB 1 — the DOSE.** How many rollouts does the leaf need before it out-ranks the head on the
  leaf column? The answer is a curve, not a bar, and it has a COST axis: a rollout is ~27 further
  sim turns (measured: mean 26.7 remaining turns per rollout over the 24,048 banked re-rolls).
* **JOB 2 — the TRANSFER.** Out-ranking successors on banked forks is not the same event as winning
  games. The battery has never run a cell whose leaf is a rollout; every previous cell scored its
  leaf with a critic, and all six heads sat on the structural null.

## 2. JOB 1 — THE K-CURVE

### 2.1 The dice

`reroll_extend.py` draws **16 fresh post-divergence re-rolls per branch** on the **exact 1,002
banked forks** of the 2026-09-19 control — `fresh_seeds(24, salt)` is a strict EXTENSION of the
banked `fresh_seeds(8, salt)`, so indices 8–23 are dice the banked label has never seen while
indices 0–7 are the banked ones unchanged. 1,002 forks × 3 branches × 16 = **48,096 new rollouts**.

| | |
|---|---|
| **THE LABEL** | re-rolls **0–7**, the BANKED K′ = 8 mean — **one label set for EVERY K**, so the curve's levels are comparable to each other AND to the head, and the head's number on it is already published (§5.1 of the control: **0.5801** pooled, **0.5610** `top1\|top2`) |
| **THE LEAF** | re-rolls **8–23**. `ROLLOUT_K` = the mean of re-rolls 8 … 8+K−1 for **K ∈ {1, 2, 4, 8, 16}** — INDEPENDENT dice, the same states, so whatever it reads is what a value function of K-rollout quality reads on this label. Two scorers, one label, one pair set: no modelling, no tie-rate assumption, no ceiling arithmetic |
| **NESTED, declared** | a smaller K is a strict SUB-SAMPLE of a larger one's dice. The levels are therefore POSITIVELY CORRELATED — the right design for a monotone dose curve (a K-to-K difference is paired and carries no between-K dice noise), and the wrong one for treating two levels as independent samples. Every CI is bootstrapped over **FORKS** (2,000 resamples, `refit.boot_ci` by path) |
| **CRN within a re-roll index** | inherited unchanged: for re-roll k the three branches share one seed, so the k-th `top1` and the k-th `top2` differ in exactly the action |

**The join is ASSERTED.** The fork list is read from the BANKED rows rather than re-derived from the
sampler, every row's recomputed `seed_salt` must equal the banked one, and `--verify 4` per shard
re-runs **re-roll index 0** under the banked run's own player tag and requires the score to
reproduce EXACTLY. That is a CRN reproduction check of the whole harness — model load, sentinel
resolution, replay, reseed, scoring.

### 2.2 The read

Pairwise accuracy `sign(V_a − V_b)` against the K′ = 8 label on non-tied pairs, in the three
pair-type columns (`top1|rand`, `top2|rand`, **`top1|top2`**) and POOLED; **Δ vs the head on
IDENTICAL pairs** (`refit.paired_delta`); the **LABEL CEILING beside every level** (rule 25 as
amended 2026-09-19); and the **COST** column.

**Three ceiling instruments, ranked by how much they assume, and the first is the headline:**

1. **MEASURED, model-free — the K = 16 level itself.** The best estimator of the truth whose dice
   are independent of the label. It is a LOWER BOUND on the ceiling, not the ceiling: 16 rollouts
   are still noisy.
2. **SPLIT-HALF of the LABEL** (re-rolls 0–3 vs 4–7) → the implied `q = (1+√(2A−1))/2`.
   ⚠️ **CROSS-CHECK ONLY** — hazard 4 of the control: the inversion is biased UP by Jensen over
   heterogeneous gaps and DOWN by tie-scoring at 0.5.
3. **PARAMETRIC** `ACC_oracle(gap_sd, K′=8)` from the moment-recovered gap distribution with
   `ddof=1`. ⚠️ **CROSS-CHECK ONLY** — it over-read the measured 8-rollout number by ~0.06 last time.

**COST, per leaf evaluation of ONE successor:** `K × (realized mean remaining turns per rollout)`,
with the realized mean recorded, plus the realized wall-clock seconds per rollout from the shard
metas. A top-2 playoff evaluates TWO successors, so its per-decision cost is `2K` rollouts.

### 2.3 The registered bars

| bar | statement |
|---|---|
| **BAR K1** | the smallest K at which `ROLLOUT_K`'s **`top1\|top2` CI LOWER BOUND exceeds the head's POINT estimate** on the same pairs |
| **BAR K2** | the smallest K at which that lower bound exceeds **0.60**, if any — reported BESIDE the measured ceiling, never as a bar in the abstract (rule 25 as amended: 0.60 was near-unreachable against a single-rollout label and is a different number against a K′ = 8 one) |
| **BAR K3 — the shape** | `acc(K)` rises with K and its increments SHRINK (the estimator sd falls as `1/√K`). Reported as the level table plus the marginal table |
| **BAR K4 — the knee** | the smallest K at which the MARGINAL gain per rollout falls below the AVERAGE gain per rollout spent so far: `m(K) = [acc(2K) − acc(K)] / K  <  [acc(K) − acc_head] / K`, i.e. `acc(2K) − acc(K) < acc(K) − acc_head`. Beyond it, doubling the budget buys less than the budget has already bought |
| **BAR K5 — the instrument** | `verify_bad` = 0, `salt_mismatch` = 0, `errors` = 0, and the head reproduces the control's §5.1 numbers to **±0.01** on the same label (the pair set differs slightly: this read does not require the hand scorers to be finite, so it carries a few more pairs — the count is published and every Δ is on identical pairs). **Below either of the first two, the read is INCONCLUSIVE** |

### 2.4 The predictions

| # | prediction |
|---|---|
| **K1** | the curve is MONOTONE in K on all four columns, and CONCAVE (increments shrink) |
| **K2** | the head reproduces **0.5801 ± 0.01** pooled and **0.5610 ± 0.01** on `top1\|top2` against the banked K′ = 8 label — an identity check, not a measurement |
| **K3** | **BAR K1 is met at K = 2 or K = 4**, and I register **K = 4**. At K = 4 against the NOISIER K′ = 4 half-label the leaf already read 0.5980 [0.5705, 0.6241] where the head read 0.5705; a cleaner label should lift both and widen the gap |
| **K4** | `ROLLOUT_16` on `top1\|top2` lands in **0.63–0.72**, and `ROLLOUT_1` lands in **0.53–0.58** — a single fresh rollout is a WORSE leaf than the head on this column, or at best its equal |
| **K5** | **BAR K2 (lower bound > 0.60 on `top1\|top2`) is met at K = 8**, and is NOT met at K ≤ 4 |
| **K6** | the POOLED curve is ABOVE the `top1\|top2` curve at every K — the `rand` columns are easier for every scorer measured so far |
| **K7** | **the knee (BAR K4) is at K = 4**: the step 1→2→4 buys more than the step 8→16 |
| **K8** | the label's own SPLIT-HALF agreement on `top1\|top2` (halves of K′ = 8, i.e. two K = 4 labels) reproduces the control's **0.6303 ± 0.02**, and the implied `q` **OVER-states** the measured `ROLLOUT_16` level — i.e. the parametric ceilings remain cross-checks, exactly as hazard 4 said |
| **K9** | the realized mean remaining turns per rollout is **25–29** (banked: 26.7), so the cost of a K = 4 leaf evaluation is ~107 sim turns and of a K = 4 top-2 PLAYOFF decision ~214 |
| **K10** | `verify_ok` = 24 / 24 and `salt_mismatch` = 0 |

## 3. JOB 2 — THE BATTERY WITH A ROLLOUT LEAF

### 3.1 The cell

`python -m main.search_dividend models/ai_v13_02_flywheel_winprob/final_model.zip --arm playoff`
— the TOP-2 PLAYOFF: the depth-1 critic sweep demoted to a **SCREEN** that nominates two actions,
and the nomination settled by **paired rollouts to a terminal** under shared dice and a shared
policy-sampling seed. `--opponents self` (the MIRROR: the null is **0.50 by construction**),
side-swap pairs, `--games-seed 7`, **400 pairs**, the FULL 719-team pool (no team pin exists for
arm W), `--playoff-rollouts R` where **R = the K that BAR K1 returns** (default **R = 4** if the
curve is not decisive), `--max-depth 1`, `--impl rust`, `--battle-timeout-s` / `--battle-idle-s`
raised (a nested rollout family silences the live stream for a whole decision, and a timed-out game
POISONS THE REST OF THE CELL).

**TWO CONTEMPORANEOUS CONTROLS, in the same window, on the same game indices** (rule 23 — a meter
whose value depends on the box's throughput needs a contemporaneous control, and a battery cell is
such a meter):

| cell | what it is |
|---|---|
| **`playoff` @ R** | the rollout leaf |
| **`defB` — the critic-leaf defensive cell at the REGISTERED operating point** | `--root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0 --defensive-contested-deadline-s 3.0 --arm honest --budget 1` — the cell six heads were read on |
| **`base`** | both sides unsearched. The EXACT-0.50 control: it is the harness's own null and it is cheap |

### 3.2 🚨 WHAT IS AND IS NOT WIDTH-MATCHED — registered in advance

**This is the first battery cell that is NOT a width meter, and the rule that governs it changes.**
Every previous cell bought its search width with a WALL-CLOCK budget, which is why L1 read 0.058 /
0.128 / 0.365 on one checkpoint across three contention regimes (rule 23). The playoff's rollouts
are **COUNT-budgeted**: R paired rollouts are R paired rollouts whether the box is idle or loaded,
and the realized R is recorded per decision. So:

* **the playoff cell's rollout budget is NOT width-matched to the control's clock budget, and
  cannot be** — they are different currencies. Matching them would require pricing a rollout in
  seconds, which is exactly the number the box's load moves;
* what IS reported instead is **the honest COST**: realized rollouts per decision and **wall-clock
  seconds per decision**, measured in the same window as the control's. That is the number a
  deployment decision needs and no previous cell has published;
* the control's own SCREEN width (realized K worlds, arms scored) is still contention-coupled, so
  the control is quoted as a CONTEMPORANEOUS anchor and never against a fixed L1 bar.

### 3.3 The registered bars

| bar | statement |
|---|---|
| **BAR B1 — SEARCH PAYS** | the playoff cell's **paired mirror win rate (L2) CI LOWER BOUND > 0.50** |
| **BAR B2 — the control** | the `defB` critic-leaf cell in the same window straddles 0.50 (its six-head prior is 0.485–0.507) and `base` is consistent with 0.50 |
| **BAR B3 — the mechanism** | the realized **`played` / `inconclusive` / `screen_decisive`** split, the overrule rate, and the separation-of-raced of the control cell. A high inconclusive rate is the instrument saying the rollout budget did not resolve the pair — a reportable finding, not a failure |
| **BAR B4 — hygiene** | > 25 % of attempted battles unfinished ⇒ the cell is **INCONCLUSIVE, never a win rate** (`INCONCLUSIVE_FRAC`, `report2.py`) |

⚠️ **Declared instrument deviation.** The instruction asks for a **Wilson** CI on L2. A side-swap
PAIR scores 1 / 0.5 / 0 and is not a Bernoulli trial, so Wilson does not apply to it; the registered
instrument for every previous cell is the **normal interval on the mean of the pair scores**, and
that is what is quoted, with a **bootstrap over pairs** printed beside it and a **Wilson interval on
the UNPAIRED decisive-game win rate** printed as the third column. All three are published; the
paired one is the headline because the team draw is most of the variance at these n.

### 3.4 The predictions

| # | prediction |
|---|---|
| **B1** | 🚨 **THE PLAYOFF IS MOSTLY INCONCLUSIVE AT R = 4, MECHANICALLY.** The gate is `\|mean(d)\| ≥ 2·SE(d)` over R paired differences in {−1, −0.5, 0, +0.5, +1}; at R = 4 that needs roughly three of four pairs agreeing decisively. With a true sibling gap of `E\|gap\| ≈ 0.18` the per-pair difference is dominated by dice, so I register **≥ 60 % of run playoffs INCONCLUSIVE at R = 4** (and ≥ 35 % at R = 8) |
| **B2** | **L2 lands in 0.49–0.53 with a CI straddling 0.50 — BAR B1 NOT MET, i.e. branch (b).** The prior is six heads on the structural null and a seventh object that out-ranks them offline; but the playoff only ever adjudicates the SCREEN's top two, and the screen is the same critic whose `top1\|top2` resolution this campaign measured at 0.545 |
| **B3** | the playoff cell nonetheless **beats the `grid` disaster by a mile** (`grid` is 0.19–0.32 across six heads): a rollout-settled override is not the same object as a critic-argmax override, and the 2·SE gate refuses most of them |
| **B4** | the control `defB` cell reads **0.48–0.53**, straddling 0.50, in this window |
| **B5** | **the honest cost is 1–3 ORDERS OF MAGNITUDE above the control's.** At R = 4, ~8 rollouts × ~27 remaining turns ≈ 216 sim turns per adjudicated decision against a 3 s critic sweep; in wall clock I register **5–30 s per adjudicated decision** and **≥ 2 minutes per game** |
| **B6** | `screen_decisive` fires on **40–75 %** of decisions (the screen's top-1 is the policy's action by a margin wider than 0.023), so the playoff runs on a MINORITY of decisions and the cell's total cost is far below R × decisions |
| **B7** | zero timeouts and zero unfinished games with the backstops raised; if any cell exceeds 25 % unfinished it is reported INCONCLUSIVE and not as a win rate |

## 4. The branches, registered

| branch | condition | reading |
|---|---|---|
| **(a) THE ROLLOUT LEAF PAYS** | BAR K1 met at some K ≤ 16 **AND** BAR B1 met | a rollout leaf pays in GAMES. The search-teacher path opens with rollouts as the leaf, at the named cost (`2K × 27` sim turns per decision) |
| **(b) OUT-RANKING DOES NOT TRANSFER** *(predicted)* | BAR K1 met, BAR B1 not | ranking successors better on banked forks does not become wins at this K and this budget. The named suspects, in order: the SCREEN (a critic that nominates the wrong two), the 2·SE gate (an override budget the dice cannot certify at R = 4), and the mirror's own ceiling |
| **(c) THE DOSE IS UNAFFORDABLE** | BAR K1 not met at any K ≤ 16 | the leaf-column gain is below the head's noise at any affordable K, and the rollout leaf is closed at this depth |

## 5. Declared deviations, and what could make this read wrong

* **The K-curve's forks are the 1,002 of the control, not all 5,040** — the other 4,038 have no
  rollout labels at all and buying them is 24 rollouts each. The 1,002 were drawn by a seeded
  shuffle whose selection read NO outcome, and the `top1|top2` column carries **543** non-tied pairs
  at the K′ = 8 label (banked, counted before this registration). Intervals there are ~1.7× the
  pooled ones and every "not detected" is reported WITH its width.
* **The curve is NESTED in K** (§2.1) — declared, and it is the reason the K-to-K comparison is
  strong and the between-K independence assumption is unavailable.
* **`ROLLOUT_K` is not a deployable leaf as measured.** It scores a successor by rolling OUR policy
  against the SENTINEL opponent of the banked tree, both greedy; the battery's playoff rolls out
  against the same network STOCHASTIC at temperature 1. The two are different estimands and the
  curve is a statement about the FORK instrument, not a prediction of the battery's L2 — which is
  exactly why JOB 2 exists.
* **One frozen policy, one seat regime.** Arm W, the mirror. Nothing here says a differently-trained
  trunk or a scripted-roster opponent behaves the same.
* **The battery's R is chosen by JOB 1's result**, which makes JOB 2 conditional on JOB 1 and not
  independent of it. Registered as such: R = BAR K1's K, default 4, and the second cell at R = 8
  runs only if the wall clock allows — a cell that does not run is reported as not run.
* **The box carries a training arm and two other agents' jobs throughout.** Wall-clock costs are
  reported as measured under that load, with the load stated; sim-turn costs are load-free and are
  the portable number.
