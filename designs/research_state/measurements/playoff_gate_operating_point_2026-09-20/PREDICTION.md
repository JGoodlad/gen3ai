# PRE-REGISTRATION — THE PLAYOFF GATE'S OPERATING POINT

*Written 2026-09-19, landed on main BEFORE the first number of this read exists. The banked dice of
[`rollout_leaf_kcurve_2026-09-19/`](../rollout_leaf_kcurve_2026-09-19/README.md) are REUSED; no new
rollout is paid for JOB A. JOB B is a live mirror battery.*

---

## 0. Why this read exists

The 2026-09-19 K-curve settled that a **rollout leaf out-ranks the 75M-step win-prob critic** on the
column a re-ranking search consumes (`top1|top2`: head **0.5741**, `ROLLOUT_8` **0.6524**, +0.0783
[+0.0273, +0.1279] DETECTED) — **and that the `playoff` arm buys nothing in games, because its own
acting rule (`|mean(d)| ≥ 2·SE`, `MIN_PAIRS = 4`) resolves 4.51 % of pairs at R = 4 and the live
cell changed 1 of 250 decisions.** The binding constraint moved from the LEAF to the DECISION RULE.

This read asks the next question and it is the cheap one: **where is the gate's operating point, and
does moving it convert the leaf's ranking edge into value?**

---

## 1. The two jobs

**JOB A — the operating curve, FREE, on the banked dice.** For R ∈ {4, 8, 16} × SE multiple
∈ {0.5, 1.0, 1.5, 2.0} × `MIN_PAIRS` ∈ {2, 4}, on the 665 banked forks (each with K = 16 fresh CRN
re-rolls per branch and an independent banked K′ = 8 label):

* **RESOLVE RATE** — the fraction of the 665 forks the gate would act on;
* **AGREEMENT** of the gate's pick with (a) the K′ = 8 label and (b) the 16-rollout estimate;
* **EXPECTED VALUE GAIN per decision** — mean over RESOLVED forks of
  `label_value(gate's pick) − label_value(policy top-1)`, in win-prob units;
* the same four for the **no-gate alternative**: always take the branch with the higher K-rollout
  mean.

**JOB B — a live mirror battery at the chosen point**, `--impl node`, side-swap pairs,
`--games-seed 7`, full pool, against the contemporaneous **critic-leaf defensive (`defB`)** cell and
the **unsearched `base`** arm on the same game indices.

---

## 2. Structural facts registered BEFORE the data (derived from the code, not from numbers)

**S1 — `MIN_PAIRS` is INERT over the registered R grid.** `is_conclusive` bars on `n < min_pairs`
where `n` is the number of completed pairs; offline every fork has all R pairs, so `n = R ≥ 4 ≥ 4`
for every cell of the grid. **`MIN_PAIRS` ∈ {2, 4} must therefore produce IDENTICAL numbers at
R ∈ {4, 8, 16}.** It is swept anyway because the instruction names it, and because an axis that
reads identically is evidence the harness is reading the production constant rather than a copy.
**Remedy, registered here:** the sweep is EXTENDED to R ∈ {2, 3} where `MIN_PAIRS` bites, so the
axis has content. If the two `MIN_PAIRS` columns differ at any R ≥ 4, the harness is wrong and the
read is refused.

**S2 — the EV-gain metric is NOT compressed by label noise, and AGREEMENT is.** The K′ = 8 label is
a mean of 8 Bernoulli draws, so it is an UNBIASED estimator of the branch's true value: the
*expectation* of `label(pick) − label(top1)` equals the true expected gain, whatever the label's
variance. Pairwise ACCURACY has no such property — label noise drags every accuracy toward 0.5
(measured ceiling 0.6709 on `top1|top2`). **So the EV column is the one to steer by and the
agreement column is the one to sanity-check with**, and this is stated before either is computed.

**S3 — the gate's dice and the label's dice are DISJOINT.** The gate reads fresh re-rolls 8–23; the
label is banked re-rolls 0–7. There is no selection-on-the-label, so the EV column carries no
winner's-curse term from the gate's own choice. (The gate DOES select on its own dice, which is why
the gate's *own* K-rollout mean is not a legitimate scorer of itself — the 16-rollout agreement
column is reported as the biased cross-check it is, never as truth.)

**S4 — flags already exist.** `--playoff-se-k` and `--playoff-min-pairs` are already on
`python -m main.search_dividend` and already reach `PlayoffConfig.se_multiple` / `.min_pairs` and
thence `decide(...)`. **No `src/` change is planned.** (The instruction named a
`--playoff-se-multiple`; the shipped spelling is `--playoff-se-k` and it is the same knob — using
the existing flag rather than adding an alias.)

**S5 — `--impl rust` breaks the playoff's nested rollouts** (2026-09-19 hazard 1, backlog P1). JOB B
runs `--impl node` and says so in every table.

---

## 3. The registered selection rule for the live operating point

Registered BEFORE the curve exists, so the point cannot be chosen to suit the battery:

1. **R is pinned to 4 by COST, not by the curve.** A playoff decision at R = 4 measured 43.3 s live;
   R = 8 doubles it and R = 16 quadruples it, and the CPU budget for this read is ~60 CPU-h. R = 4
   is the only rung a ≥ 100-pair cell could even approach.
2. Among the R = 4 cells, take the **SE multiple maximising `EV gain per resolved decision ×
   resolve rate`** (i.e. the expected value gain per DECISION, not per resolved decision).
3. Ties or near-ties (within 10 % of the maximum) break toward the **higher** SE multiple — the
   more conservative gate.
4. **Contingency:** if NO R = 4 cell has a positive `gain × resolve`, the live cell is run at the
   smallest R with a positive one and the pair count is reported as whatever the clock bought. If no
   cell at any R has a positive one, **JOB B is not run at all** and branch (c) is declared on JOB A
   alone — spending 40 CPU-h to play an arm the free read says is worthless would be the exact
   mistake the 2026-09-19 read named in its §5.3.

---

## 4. The predictions

### JOB A — the operating curve

| # | registered |
|---|---|
| **A1** | The `k = 2.0`, `MIN_PAIRS = 4` column **REPRODUCES the 2026-09-19 numbers EXACTLY** — resolve 0.0451 / 0.1068 / 0.1248 at R = 4 / 8 / 16 (to ±0.002; same forks, same imported rule). A miss here is an instrument failure, not a finding. |
| **A2** | Resolve rate is **monotone increasing as k falls** at every R, and monotone increasing in R at every k. |
| **A3** | At `k = 0.5`, R = 4, resolve ≥ **0.40**. (The statistic is a mean of 4 atomic-in-halves draws; half an SE is a very low bar and most non-degenerate forks clear it.) |
| **A4** | Agreement with the K′ = 8 label is **monotone increasing in k** at every R, and at `k = 0.5` it lands within **±0.04 of the unconditional `ROLLOUT_R` level on `top1\|top2`** (0.6125 / 0.6524 / 0.6709) — i.e. a nearly-open gate is just the leaf. |
| **A5** | **Agreement ≥ 0.80 requires k ≥ 1.5 at R = 4**, and is met at k ≥ 1.0 at R = 16. |
| **A6** | 🚨 **The EV gain per RESOLVED decision is POSITIVE at every cell of the grid** — the leaf out-ranks, so an override is right more often than not — and it **falls as k falls** (the gate concludes first where the true gap is largest). |
| **A7** | 🚨 **The EV gain per DECISION (`gain × resolve`) RISES as k falls at every R**, because coverage grows faster than per-resolved quality decays, and the maximiser over the registered grid is at **`k = 0.5`**. |
| **A8** | 🚨 **The NO-GATE rule ("always take the higher K-rollout mean") has the HIGHEST per-decision EV gain of anything measured at each R** — strictly above every gated cell at that R. If this is refuted, the gate is buying something and that is the read's surprise. |
| **A9** | The no-gate rule's per-decision EV gain at R = 4 lands in **[+0.010, +0.045]** win-prob units, and at R = 16 in **[+0.020, +0.070]**. |
| **A10** | `MIN_PAIRS` = 2 and 4 are **bit-identical at R ∈ {4, 8, 16}** (S1) and differ only at R ∈ {2, 3}. |
| **A11** | An operating point with **EV gain > 0 at resolve ≥ 20 %** EXISTS at R = 4 (this is branch (a)'s offline half, and I register that it clears). |

### JOB B — the live battery

| # | registered |
|---|---|
| **B1** | The realized live resolve rate (`n_playoff` / `n_playoff_ran`) is **BELOW the offline curve's prediction for the chosen point, by a factor of 0.2–0.8**. Direction from the 2026-09-19 proxy: offline rollouts are greedy against a banked sentinel, live ones are the same net STOCHASTIC at T = 1, which adds variance to every draw and widens the SE. |
| **B2** | The realized live CHANGE rate (`n_changed` / decisions) is **≥ 5× the 0.4 % the `k = 2.0` cell measured** — moving the gate is the whole point, and if it does not move the change rate the flag did not reach the rule. |
| **B3** | `--budget 120` at R = 4 realizes **R ≥ 3.6** and `short_r_refusal` does NOT fire (the 2026-09-19 `pf4b` cell realized exactly 4.00 at that budget). |
| **B4** | The `defB` control reads **L2 ∈ [0.47, 0.56] straddling 0.50** (it read 0.5243 [0.4864, 0.5623] in the 2026-09-19 window). |
| **B5** | The `base` arm is the unsearched policy against itself, so its L2 is **0.50 BY CONSTRUCTION up to sampling noise** — it is a harness null, and a `base` cell clear of 0.50 falsifies the whole battery rather than saying anything about search. |
| **B6** | 🚨 **The paired-mirror L2 interval is NARROWER than the instruction's ±0.10.** A side-swap pair scores in {0, 0.5, 1} and ties dominate; the 2026-09-19 `defB` cell measured **sd = 0.206** over 113 pairs, so the 95 % half-width at n pairs is ≈ `0.40/√n` — **±0.064 at n = 40, ±0.040 at n = 100**. The bar is still a lower bound above 0.50, but a 4-point effect is resolvable at ~100 pairs and a 6-point one at ~40. |
| **B7** | ⚠️ **The arithmetic that connects the two jobs, registered so branch (b) is quotable.** With `g` = offline EV gain per decision, `f_live` = realized live resolve rate and ~42 decisions per battle, the per-GAME win-prob shift is bounded below by the single largest realized gain and above by `42 · g` (gains at different decisions overlap, so the sum is a strict upper bound). I register the honest prediction: **the live L2 lands in [0.50, 0.50 + 42·g/2] and most likely in the lower half**, and that if the point estimate is below 0.52 the correct reading is branch (b) with the arithmetic quoted, NOT "search is refuted". |
| **B8** | Wall clock: **40–70 s per adjudicated decision** and **1,100–2,500 s per battle** at R = 4 on this box (measured 43.3 s / 1,414 s on 2026-09-19 at load 30–60; this read starts at load ~35). |
| **B9** | **Pairs achieved ∈ [30, 110]** within the hard stop. I register in advance that the cell may be under-powered and will be reported as a DESCRIPTOR with its width if so, exactly as the 2026-09-19 cell was. |
| **B10** | Zero unfinished battles above the 25 % INCONCLUSIVE bar; ≤ 3 % of live rollouts raise under `--impl node` (1.05 % measured 2026-09-19). |

---

## 5. The branches (from the instruction, restated so they are scored)

* **(a)** an operating point with EV gain > 0 at resolve ≥ 20 % exists offline **AND** the live L2
  lower bound clears 0.50 ⇒ **the rollout-leaf search PAYS in games**;
* **(b)** offline gain exists, live L2 NOT DETECTED ⇒ the gain per decision × resolve rate is below
  the mirror's floor at this n — **and the arithmetic of B7 must be quoted**;
* **(c)** no operating point has positive EV gain ⇒ the leaf's ranking edge does not convert to
  value at the top-2, i.e. **the E|gap| = 0.18 is not where the gate looks**.

I register **(b)** as the most likely outcome, at roughly 0.55 credence, with (a) at 0.25 and (c) at
0.20 — and I register that A7/A8 (the gate is costing more than it saves) is the finding I most
expect this read to produce.

---

## 6. Regime, hygiene, and what must NOT happen

* CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, ports **9700–9799**, outputs under
  `/home/goodlad/.claude/jobs/9ab51de6/tmp/gatecurve/`. **Nothing written under `models/`.**
* The GPU hosts `ai_v13_09_wcont`; it is not touched. **8000 and 8001 are not touched.** No process
  this read did not start is signalled, and kills are by explicit PID.
* JOB B runs from the MAIN checkout (that is where `models/` lives); the worktree holds the
  deliverable. No `git add`/`commit`/`push` from the main checkout.
* Budget: **≤ ~60 CPU-h** for JOB B, with a declared hard stop; pairs achieved are reported whatever
  they are.
* A hazard found is a FINDING and is written into the README's hazards section, not dropped.
