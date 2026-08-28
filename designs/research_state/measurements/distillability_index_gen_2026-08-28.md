# The DISTILLABILITY INDEX — does readiness to absorb external behaviour rise with training age?

**Date** 2026-08-28 · **Producer** `distillability_index_probe.py` (this directory) ·
**Data** `distillability_index_gen_2026-08-28.json` (every curve, every cell) ·
**Cost** 41 probe cells, 6.1 CPU-hours, ~2 h wall on 3 `nice 15` single-thread processes, no GPU,
`models/` read-only, zero failures.

The question: the fold of an exploiter teacher back into the generalist converts only **~13%** of the
teacher's on-slice edge. The working hypothesis was that transfer efficiency is a property of the
**student's consolidation state** — a converged network accepts new behaviour as an *annex* (fast,
cheap), a plastic one has to rewire (slow, or expensive to what it already knows). This is the first
direct measurement of that quantity.

---

## 1. The instrument

For a student checkpoint `C` and a fixed teacher `T`:

* **ON-SLICE** — 4 200 recorded decision states from `T`'s own eval traces, i.e. states reached while
  piloting `T`'s two pinned teams. Split **by battle file**, 75/25, into a 3 000-state training pool
  and a **1 200-state held-out pool from 64 battles never trained on**. Held-out is the primary
  absorption meter; the training pool's agreement is reported alongside and reaches 0.98–0.99, i.e.
  it measures memorisation, not transfer.
* **OFF-SLICE** — 1 500 states from the *parent* run's eval traces spanning **233 distinct teams and
  12 eval steps**, with every state on either of `T`'s pinned teams excluded (117 such rows dropped).
* **Probe** — the student's **full policy** (extractor + both heads, nothing frozen — that is what a
  real fold updates) is trained with Adam on masked cross-entropy to the teacher's **argmax**,
  batch 256, 400 steps, evaluated at 14 log-spaced points.
* **ABSORPTION** = held-out on-slice top-1 agreement with the teacher.
  **COLLATERAL** = off-slice divergence from the student's *own* pre-probe policy: masked
  `KL(now ‖ original-C)`, top-1 agreement with original-C, and mean `|ΔV|`.

States with fewer than two legal actions are excluded (a forced decision carries no policy signal).
Every model is loaded through the prober's read-only path (`sanitized_load_custom_objects` +
`MaskablePPO.load`); **`dropped_kwargs` was empty on all 41 loads**, so every rebuilt extractor is
the one that played. All students and both teachers carry `arch_signature =
gen3_critic_route_wave_v1`, obs dim 2501. The extractor's four `Dropout` modules all have `p = 0.0`,
so the forward is deterministic and train/eval mode is a no-op — the *only* stochasticity in a cell
is the batch order, which is seeded and **identical across cells at a given seed**.

### Cells

| family | students | teacher | lr | seeds |
|---|---|---|---|---|
| **age series** | `ai_v9_29_rev1_0823` snapshots 2/6/12/18/24M + `final_model` (25M) | A | 3e-4 | 1, 2 |
| **age series, robustness lr** | same six | A | 1e-4 | 1, 2 |
| **ancestry-free control** | `ai_v9_21_gen17_pfspoff_0820` snapshots 2/6/12/18/24M + final | A | 3e-4 | 1 (all six), 2 (2M/12M/25M) |
| **content control** | rev-1 2M/12M/25M, targets = **the student's own argmax** | — | 3e-4 | 1 |
| **sanity** | fresh-init ×2 seeds; teacher into itself | A | 3e-4 | 1, 2 |
| **bonus** | `ai_v9_59_R2ACTION_0827` final; rev-1 final vs teacher **B** | A / B | 3e-4 | 1 |

Teacher A = `ai_v9_53_R2F5a_0826/final_model.zip` (28.07M steps), an exploiter forked from
`ai_v9_29_rev1_0823/final_model.zip`, pinned to teams `eccfe630ec08de27` + `023a2d47648b85e6`.
Teacher B = `ai_v9_54_R2F5b_0826`.

---

## 2. Instrument admission

**ADMITTED for the trajectory-level indices. NOT admitted for the single-step shock as a scalar.**

| sanity cell | pre-registered expectation | measured | verdict |
|---|---|---|---|
| teacher distilled into **itself** | step-0 agreement ≈ 1.0, index degenerate | agreement **exactly 1.000**, off-slice KL **exactly 0.000**, `\|ΔV\|` **exactly 0.00**; gain-index MISS by construction | ✅ harness correct |
| **fresh-init** policy (2 seeds) | fast absorption AND very high collateral | **the largest gain in the battery** (0.070 → 0.677 / 0.672, i.e. **+0.607 / +0.603**) and the **lowest** off-slice self-agreement anywhere (**0.137 / 0.112** at step 400) | ✅ as predicted |
| **repeatability** (2 probe seeds) | the age ordering must be stable | see below | ✅ on every trajectory index |

Seed-to-seed reproduction, over 22 paired cells: `gain@400` reproduces to **≤ 0.018** absolute
everywhere (median 0.006); `KL@400` to **≤ 0.097** (median 0.041). Spearman-vs-age of the headline
quantities is stable in **sign and near-magnitude** across seeds in every arm (§4).

**The one metric that failed admission** is the *step-1 shock* — the off-slice KL after a single Adam
step. Its ordering vs age is robust (ρ = +0.77 … +1.00 in every arm) but its **value** is not: on
`age_25M_final` it read **1.116** on seed 1 and **0.420** on seed 2. It is a single-minibatch
quantity and is reported only as an ordering, never as an index.

One caveat that is *not* a defect but bounds interpretation: the fresh-init cell's off-slice **KL**
stays near zero early while its off-slice **argmax agreement** collapses to 0.05 after one step. A
near-uniform reference makes KL insensitive; for a fresh net the argmax meter is the honest one.
This is why both meters are reported for every cell.

---

## 3. The age curve

Full curves for all 41 cells are in the JSON. `a0` = step-0 held-out agreement with the teacher;
`a_max` = the best it reaches in 400 steps; `KL@Δ+0.05` = off-slice KL at the point absorption has
risen 5 pp above its own start; `KL@A*=0.78` = off-slice KL at the point held-out agreement first
reaches the absolute level 0.78.

**ANCESTOR lineage (rev-1 — the run the teacher was forked from) · lr 3e-4 · seed 1 / seed 2**

| age | a0 | a_max | KL@Δ+0.05 | KL@A\*=0.78 | KL@400 | off-agree@400 | \|ΔV\|@400 |
|---|---|---|---|---|---|---|---|
| 2M | 0.549 / 0.549 | 0.760 / 0.766 | 0.125 / 0.199 | **MISS** / **MISS** | 0.775 / 0.789 | 0.598 / 0.587 | 4.77 / 5.62 |
| 6M | 0.626 / 0.626 | 0.812 / 0.807 | 0.303 / 0.237 | 0.342 / 0.371 | 0.727 / 0.749 | 0.642 / 0.651 | 7.51 / 7.07 |
| 12M | 0.703 / 0.703 | 0.818 / 0.814 | 0.388 / 0.388 | 0.411 / 0.458 | 0.668 / 0.765 | 0.667 / 0.633 | 4.52 / 4.19 |
| 18M | 0.714 / 0.714 | 0.839 / 0.830 | 0.396 / 0.525 | 0.455 / 0.523 | 0.721 / 0.805 | 0.664 / 0.649 | 4.22 / 8.84 |
| 24M | 0.735 / 0.735 | 0.828 / 0.827 | 0.563 / 0.514 | 0.581 / 0.521 | 0.717 / 0.695 | 0.665 / 0.665 | 5.26 / 5.71 |
| **25M final** | 0.758 / 0.758 | 0.832 / 0.834 | 0.570 / 0.519 | 0.480 / 0.523 | 0.750 / 0.819 | 0.669 / 0.673 | 4.29 / 5.66 |

**ANCESTRY-FREE lineage (gen-17, independently fresh-init, shares no weights with either teacher) · lr 3e-4**

| age | a0 (s1 / s2) | a_max | KL@Δ+0.05 | KL@A\*=0.78 | KL@400 | off-agree@400 |
|---|---|---|---|---|---|---|
| 2M | 0.533 / 0.533 | 0.756 / 0.729 | 0.121 / 0.086 | **MISS** / **MISS** | 0.829 / 0.733 | 0.533 / 0.581 |
| 6M | 0.581 / — | 0.782 / — | 0.497 / — | 0.709 / — | 0.855 / — | 0.605 / — |
| 12M | 0.613 / 0.613 | 0.787 / 0.782 | 0.618 / 0.539 | 0.633 / 0.658 | 0.845 / 0.879 | 0.635 / 0.619 |
| 18M | 0.645 / — | 0.793 / — | 0.669 / — | 0.713 / — | 0.861 / — | 0.654 / — |
| 24M | 0.620 / — | 0.798 / — | 0.649 / — | 0.693 / — | 0.916 / — | 0.615 / — |
| **25M final** | 0.635 / 0.635 | 0.797 / 0.792 | 0.756 / 0.580 | 0.671 / 0.656 | 0.923 / 0.890 | 0.615 / 0.619 |

**ANCESTOR lineage · lr 1e-4 (robustness arm) · seed 1 / seed 2**

| age | a0 | a_max | KL@Δ+0.05 | KL@A\*=0.78 | KL@400 | off-agree@400 | \|ΔV\|@400 |
|---|---|---|---|---|---|---|---|
| 2M | 0.549 | 0.756 / 0.757 | 0.091 / 0.110 | **MISS** | 0.662 / 0.603 | 0.591 / 0.615 | 3.69 / 3.91 |
| 6M | 0.626 | 0.820 / 0.818 | 0.086 / 0.114 | 0.258 / 0.214 | 0.630 / 0.679 | 0.669 / 0.649 | 6.14 / 5.93 |
| 12M | 0.703 | 0.834 / 0.831 | 0.147 / 0.151 | 0.206 / 0.203 | 0.496 / 0.531 | 0.717 / 0.706 | 5.30 / 5.56 |
| 18M | 0.714 | 0.846 / 0.839 | 0.177 / 0.188 | 0.233 / 0.224 | 0.494 / 0.538 | 0.709 / 0.701 | 2.94 / 6.31 |
| 24M | 0.735 | 0.850 / 0.849 | 0.174 / 0.160 | 0.172 / 0.160 | 0.454 / 0.501 | 0.735 / 0.723 | 3.35 / 3.33 |
| **25M final** | 0.758 | **0.854 / 0.852** | 0.183 / 0.177 | **0.080 / 0.115** | **0.436 / 0.477** | **0.750 / 0.739** | 3.55 / 2.60 |

**CONTENT CONTROL — same optimizer, same states, targets = the student's OWN argmax (zero new content) · lr 3e-4**

| age | step-1 shock KL | KL@400 | off-agree@400 | \|ΔV\|@400 | for comparison: teacher-target KL@400 (s1) |
|---|---|---|---|---|---|
| 2M | 0.148 | 0.584 | 0.733 | 5.97 | 0.775 |
| 12M | 0.435 | 0.518 | 0.719 | 8.63 | 0.668 |
| 25M final | 0.522 | 0.595 | 0.695 | 4.82 | 0.750 |

**SANITY / BONUS**

| cell | a0 | a_max | max gain | KL@400 | off-agree@400 |
|---|---|---|---|---|---|
| fresh-init, seed 777 | 0.070 | 0.677 | **+0.607** | 0.891 | **0.137** |
| fresh-init, seed 778 | 0.070 | 0.672 | **+0.603** | 0.973 | **0.112** |
| teacher into itself | **1.000** | 1.000 | +0.000 | 0.613 | 0.701 |
| `R2ACTION` final (already folded once) | **0.788** | **0.846** | +0.057 | 0.704 | 0.695 |
| rev-1 final vs **teacher B** | 0.702 | 0.771 | +0.069 | 0.762 | 0.647 |

---

## 4. Monotonicity vs age (Spearman ρ, n = 6 unless noted)

| quantity | rev-1 s1 | rev-1 s2 | gen-17 s1 | gen-17 s2 (n=3) | lr 1e-4 s1 | lr 1e-4 s2 |
|---|---|---|---|---|---|---|
| `a0` (step-0 agreement) | +1.00 | +1.00 | +0.83 | +1.00 | +1.00 | +1.00 |
| **`a_max` (absorption ceiling)** | **+0.83** | **+0.94** | **+0.94** | **+1.00** | **+1.00** | **+1.00** |
| `gain_max` | −0.94 | −0.94 | −0.71 | −1.00 | −0.94 | −1.00 |
| `KL@Δ+0.05` | +1.00 | +0.83 | +0.94 | +1.00 | +0.89 | +0.83 |
| `KL@A*=0.78` (n=5) | +0.90 | +0.90 | −0.20 | — | **−0.90** | **−0.70** |
| `KL@400` | −0.26 | +0.26 | +0.94 | +1.00 | **−1.00** | **−0.89** |
| `off-agree@400` | +0.83 | +0.83 | +0.49 | +1.00 | +0.94 | +0.94 |
| step-1 shock KL *(ordering only)* | +0.77 | +0.94 | +1.00 | +0.50 | +1.00 | +0.94 |

---

## 5. What this says, scored against the three pre-registered readings

### 5.1 One thing is unambiguous and survives every control: **the absorption CEILING rises with age**

`a_max` climbs monotonically with training age in **all six arms**, ρ = +0.83 … +1.00 — including the
**ancestry-free** lineage, which shares no weights with the teacher, and both learning rates. After
the identical 400 steps of identical distillation on identical states, an older student ends up
**more like the teacher** than a younger one (rev-1: 0.760 → 0.832; gen-17: 0.756 → 0.797; at lr 1e-4:
0.756 → 0.854).

The mirror statistic, `gain_max`, falls with age (ρ ≈ −0.95) — but that is arithmetic, not a finding:
`a0` rises faster than `a_max` does. **"Absorption gets slower with age" is FALSE as stated.** The
endpoint improves; the required displacement shrinks.

### 5.2 The COLLATERAL half is **learning-rate-dependent, and the sign flips**

At the pre-registered **lr 3e-4** — the project's own training lr, hence the pre-registered choice —
collateral at matched absorption *rises* with age: `KL@Δ+0.05` ρ = +1.00 / +0.83 on the ancestor
lineage and +0.94 / +1.00 on the ancestry-free one, i.e. **the reverse reading**, reproduced across
two independent lineages and two seeds.

At **lr 1e-4** the same students, states and teacher give the opposite: `KL@A*=0.78` ρ = **−0.90 /
−0.70** and `KL@400` ρ = **−1.00 / −0.89**, with off-slice self-agreement *rising* (ρ +0.94) and
`|ΔV|` falling. Going 2M → 25M at lr 1e-4 is a **strict Pareto improvement on both axes at once**:
ceiling 0.756 → 0.854 *and* KL@400 0.662 → 0.436 *and* off-agree@400 0.591 → 0.750.

### 5.3 The content control explains the flip — and it is the most important number here

Distilling a checkpoint onto **its own argmax** — same optimizer, same states, same step count,
**zero new behavioural content** — still costs it, at lr 3e-4:

* 25M final: off-slice KL **0.595**, against **0.750** for the real teacher — the content-free
  control accounts for **~79% of the measured collateral**. On the top-1 meter the two are barely
  distinguishable: off-slice self-agreement ends at **0.695** with zero new content versus **0.669**
  with the teacher's. It also loses **17.6 pp** of agreement with *its own* on-slice argmax while
  training on exactly those labels.
* And the content-free collateral itself **rises with age** (step-1 shock 0.148 → 0.435 → 0.522, ρ = +1.00).

Adam's first step is `lr · sign(g)` elementwise, so it displaces every network by the same *weight-space*
norm regardless of gradient scale. A network whose loss landscape has sharpened with training is moved
further in *function* space by that same displacement. **The lr-3e-4 "reverse" reading is therefore
substantially a measurement of landscape sharpening under a fixed step size, not of a network's
willingness to accept external content.**

### 5.4 Verdict

**The consolidation hypothesis is SUPPORTED on the absorption axis unconditionally, and on the
collateral axis conditional on the step size being inside the network's local regime.** The reverse
reading is *also* real, and it is what the fold currently operates in: at the lr the project actually
trains at, a mature student is damaged more per unit of absorbed behaviour than a young one — but the
content control shows most of that damage is the optimizer overshooting a sharper landscape, not the
teacher's content being rejected.

### 5.5 The consequence that is actionable

At matched steps, states, teacher and seed, **lowering the distill step size from 3e-4 to 1e-4 moves
the 25M student to a strictly better place on both axes**: ceiling 0.832 → 0.854 while off-slice KL
drops 0.750 → 0.436 and off-slice self-agreement rises 0.669 → 0.750 and `|ΔV|` falls 4.29 → 3.55.
Nothing was traded. If the ~13% conversion figure is limited by collateral rather than by absorption
— and §5.1 says absorption is not the binding constraint at 25M — then **the fold is running above
the mature student's damage threshold, and the cheapest available lever is a smaller effective step
on the distill term** (a lower lr for the distill phase, or a trust-region / smaller `distill_coef`
at the same lr). This is a prediction, not a result: it has not been tested inside a real PPO fold.

### 5.6 Two secondary findings

* **A fold does not consume distillability.** `R2ACTION` — a checkpoint that has *already* absorbed
  one action-level fold — has the **highest** `a0` (0.788) and the **highest** `a_max` (0.846) of any
  lr-3e-4 cell. Whatever a fold spends, it is not the capacity to take the next one.
* **The critic is the main casualty, by a wide margin.** Action-level CE through the shared trunk
  drives off-slice `|ΔV|` to 4–9 on a value scale of roughly ±12 (`--value-dist-vmin/vmax = ∓12`),
  and value-vs-original correlation on the 25M cell falls to **0.42 within 7 steps** before
  recovering only to ~0.90 by step 400. At lr 1e-4 the mature cells stay at `|ΔV|` 1.5–3.5. A fold
  that adds no value-side term is paying for its policy transfer in critic accuracy.

---

## 6. Caveats — what a micro-probe can and cannot say

1. **This is not a fold simulation.** There is no PPO loss running beside the distill term, no
   `--distill-team-bias` sampling, no environment interaction, no advantage/entropy pressure pulling
   the policy back, and no learning-rate schedule. It measures **capacity to absorb**, i.e. the
   student-side term of transfer efficiency, in isolation. A real fold's outcome is that term times
   everything this probe removed.
2. **The absolute-absorption index is not usable across the whole age range, by construction.** The
   teacher was forked from the rev-1 *final* checkpoint, so `a0` rises with age partly by ancestry
   (25M sits at 0.758 against the ancestry-free 25M's 0.635 — the shared-weights bonus is ≈ +0.12).
   The gen-17 control exists precisely because of this and shows the `a_max` and `KL@Δ+0.05` trends
   survive it; the `KL@A*` trend does *not* survive it at lr 3e-4 (ρ +0.90 on the ancestor lineage,
   −0.20 on the ancestry-free one) and is reported as such.
3. **State provenance is eval-trace-biased.** Both sets come from `eval_traces/`, which the trainer
   populates under win/loss forensic quotas, so they are not an unbiased sample of on-policy states.
   The off-slice set additionally comes from *one* run's traces (233 teams, 12 eval steps) and is not
   any given student's own on-policy distribution — collateral is well-defined regardless (it is
   drift from that student's own reference on those states) but it is not "damage to what this
   student would actually have done next".
4. **400 steps at lr 3e-4 over-trains.** The self-distill cell ends 17 pp *less* like itself than it
   started while fitting its own argmax; every mature cell converges to held-out agreement ≈ 0.82
   regardless of where it began. The informative regime is the first ~64 steps, which is where the
   `KL@Δ` and `KL@A*` crossings sit (steps 2–75). `KL@400` is reported but should be read as an
   over-training endpoint, not as a fold outcome.
5. **Teacher-independence is untested.** One bonus cell (teacher B on the rev-1 final) exists; it
   reproduces the qualitative position but a single age point cannot test an ordering.
6. **`KL` and top-1 agreement are not interchangeable meters**, and the fresh-init cell is the proof
   (KL ≈ 0 while argmax agreement collapses to 0.05). Both are reported everywhere for this reason.

---

## 7. MISSING cells — never interpolated

| cell | why |
|---|---|
| `KL@A*=0.78` for **every 2M student** (rev-1 and gen-17, all seeds and both lrs) | the student's absorption **ceiling** (0.729–0.766) never reaches 0.78 in 400 steps. This is a result, not a gap. |
| `KL@A*=0.78/0.80` for the whole **gen-17 lr-3e-4** series at 0.80 | same reason (ceilings 0.729–0.798). |
| `KL@Δ+0.05` for the **self-distill and content-control** cells | degenerate by construction — step-0 agreement is 1.000, so there is no gain to reach. |
| gen-17 seed 2 at **6M / 18M / 24M** | not run (budget). Seed-2 coverage of the ancestry-free arm is 2M / 12M / 25M only. |
| content control at **6M / 18M / 24M**, and at lr 1e-4 | not run (budget). The control is 3 ages at one lr. |
| a second teacher across the **age series** | not run (budget). Teacher-independence of the ordering is therefore **unanswered**, not answered. |
| anything about a **full PPO-context fold** | out of scope by construction — see caveat 1. |

---

## 8. Reproducing

```bash
export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd designs/research_state/measurements
nice -n 15 python distillability_index_probe.py build-states
nice -n 15 python distillability_index_probe.py probe age_25M_final__s1 \
    /home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/final_model.zip A 1 400 3e-4
nice -n 15 python distillability_index_probe.py aggregate
```

`<teacher_set>` accepts `A*` for the content control (targets become the student's own argmax);
`<student>` accepts `FRESH:<seed>` and `TEACHER`. One cell is ~9 min of one CPU core. The producer
writes `states_{A,B}.npz`, `teacher_targets_{A,B}.npz`, `results/<cell>.json` and `aggregate.json`
into its own directory; it never writes to `models/`.
