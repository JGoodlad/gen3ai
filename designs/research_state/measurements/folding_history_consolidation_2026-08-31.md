# FOLDING HISTORY vs POST-FOLD CONSOLIDATION — does a folded network absorb better, and does a fold need time to settle?

**2026-08-31 · producer `folding_history_probe.py` (this directory) · data
`folding_history_consolidation_2026-08-31.json` (the lineage DAG, every cell, every curve, all four
predictions scored).** 19 probe cells · CPU only, `nice 15`, **2 single-thread workers** ·
`models/` read-only · zero cell failures, `dropped_kwargs` empty on every load. Measured beside the
live 20-arm GPU fleet at load 23–41, which stretches wall time (679 s → ~950 s per cell) and cannot
move a result: every cell is deterministic offline compute on cached states.

Two "history of the parent" hypotheses that no probe had tested:

* **(a) ACCUMULATED FOLDING** — a network that has absorbed teachers before may be organised to
  absorb them again. The 2026-08-28 distillability index gestures at it (`R2ACTION`, already folded
  once, posted the highest `a0` *and* the highest `a_max` of any lr-3e-4 cell) but never varied
  PRIOR-FOLD COUNT as the variable.
* **(b) POST-FOLD CONSOLIDATION** — distilled content may need training time after the fold to
  settle into a form that generalises, in which case "transfer quality" is partly "consolidation
  time".

> ## VERDICT
>
> **(a) SURVIVES, WEAKLY, AND IS NOW MECHANISTIC.** At byte-identical age (28,067,760 steps, same
> parent, same arm table) the once-folded student reaches a **higher absorption ceiling than BOTH
> 0-fold replicates, on both probe seeds — 4 of 4 comparisons, mean +0.0096** — and the ordering
> holds on a second teacher, on a second independent lineage, and up the 0→1→2 ladder. But it
> **FAILS the registered 0.018 bar** (margins +0.007 … +0.012) and clears the measured run-to-run
> noise by only ~0.002. It is a consistent direction of small magnitude, not an established effect.
> **The mechanism is sharper than the effect:** a fold carrying EXTERNAL teacher content raises
> absorbability; a **zero-content SELF-fold does not** — `fold2self` falls back to the 0-fold level
> (0.809) and **doubles** collateral (KL 1.103 vs 0.539). It is not fold COUNT, it is folded CONTENT.
>
> **(b) IS ELIMINATED AS A SEPARABLE QUESTION, NOT ANSWERED.** Post-fold steps order *perfectly*
> with the untaught outcome (ρ = +1.00, n = 3) — the opposite of the registered prediction — but
> **the distill term is active for the whole fold run**, so that column is simultaneously the DOSE,
> and it is additionally rank-identical to parent maturity and to architecture era. No re-analysis
> of these three runs can separate them. The ordering is reported; no coefficient is fitted.

---

## 1. The lineage DAG — and how it was read

**Ancestry comes from `metadata.json` → `original_command`, and from nothing else.**
`cli_args["model"]` is overwritten by every RESUMING process and ends up naming the run's *own*
`final_model_interrupted.zip` — self-referential, and useless for ancestry. A run is a **FOLD** when
it names `--distill-teacher` **and** carries a non-zero `--distill-coef`; the coefficient clause is
what keeps `ai_v9_58_R2CTRL_0827` (five teachers listed at coef 0) out of the fold count.

Three structures the walk has to get right, each of which would silently corrupt a fold count:

* **An `init_model.zip` fork inherits the parent's INITIALISATION, not its training.**
  `ai_v8_03_zarch_control_0718` forks `ai_v8_01`'s `init_model.zip`, so the DAG edge is real but the
  parent contributes zero trained steps. Differencing step counts across that edge reports −97M
  own-steps for a run that trained 267M. Flagged and handled, never differenced.
* **A SELF-FOLD is a fold.** `ai_v9_72_R3SELF_0828` distils its own parent. It counts for fold COUNT
  (the optimizer did the same work) and must NOT count as having been taught a teacher's behaviour —
  so slice exposure is computed twice, once for content and once including ecology. This distinction
  turns out to carry the mechanism (§4.4).
* ⚠️ **21 of 160 runs carry no `original_command` at all** (the ai_v5 / ai_v6 era predates it).
  Their parent is UNKNOWN, not absent. **The first version of the resolvability check was VACUOUS**
  and reported 0 unresolved: `parent_resolved or not parent` can never be False, because `parent` is
  *read from* `original_command`, so a run with no recorded command has no parent by construction and
  passes trivially. The honest predicate asks whether every run in the chain STATES its invocation.
  **None of the 21 is in the student roster or the consolidation table**, so every fold count below
  is exact rather than a lower bound — but that is now a measured fact, not an assumption.

### 1.1 The student roster

| student | run | arm | folds in ancestry | total steps | own steps | chain |
|---|---|---|---|---|---|---|
| `root` | `ai_v9_29_rev1_0823` | A | **0** | 24,988,992 | — (root) | 29_rev1_0823 |
| `plain0` | `ai_v9_62_R2PLAIN_0827` | A | **0** | 28,067,760 | 3,078,768 | 29_rev1_0823 → 62_R2PLAIN_0827 |
| `ecol0` | `ai_v9_58_R2CTRL_0827` | A | **0** | 28,067,760 | 3,078,768 | 29_rev1_0823 → 58_R2CTRL_0827 |
| `fold1` | `ai_v9_59_R2ACTION_0827` | A | **1** | 28,067,760 | 3,078,768 | 29_rev1_0823 → 59_R2ACTION_0827 |
| `fold2` | `ai_v9_76_R4ACTION_0830` | B | **2** | 32,595,648 | 4,527,888 | 29_rev1_0823 → 59_R2ACTION_0827 → 76_R4ACTION_0830 |
| `fold2self` | `ai_v9_72_R3SELF_0828` | B | **2** | 32,615,184 | 4,547,424 | 29_rev1_0823 → 59_R2ACTION_0827 → 72_R3SELF_0828 |
| `tick1` | `ai_v9_34_tick1_0824` | C | **1** | 35,068,512 | 10,079,520 | 29_rev1_0823 → 34_tick1_0824 |
| `tick1x2` | `ai_v9_37_tick1_dosext_0825` | C | **2** | 40,066,752 | 4,998,240 | 29_rev1_0823 → 34_tick1_0824 → 37_tick1_dosext_0825 |
| `gen17` | `ai_v9_21_gen17_pfspoff_0820` | D | **0** | 22,887,936 | — (root) | 21_gen17_pfspoff_0820 |

### 1.2 Two facts the DAG settles that the mission asked about

1. **The mission's v8 lineage claim is CONFIRMED from metadata.** `ai_v8_14_distill3_0725` — the fold
   whose +5.42pp untaught gift motivates this whole line — forks `ai_v8_04_distill_4teacher_0722`,
   and `ai_v8_04` is **itself a fold** (`--distill-coef 1.0`, teacher
   `ai_v7_15_tss_exploiter_vs14_0713` among four). v8's winning fold had **1 prior fold** in its
   ancestry.
2. **The same is true of rev-3 and NOT of rev-2**, which is the sharpest *observational* contrast the
   archive holds on (a): `ai_v9_70_R3ACTION_0828` folds a parent that had already been folded and
   posts **−0.75pp**; `ai_v9_59_R2ACTION_0827` folds a parent that had not and posts **−7.06pp**.
   Same architecture, same lr, near-identical coefficient, same base ancestor, same instrument.
   §5.4 lists why this pair still cannot carry a verdict.

---

## 2. The instrument and the cells

Mechanically the ADMITTED `distillability_index_probe.py` (2026-08-28 §2), unchanged: the student's
**full policy** (extractor + both heads, nothing frozen — what a real fold updates) trained with Adam
on masked cross-entropy to a fixed teacher's **argmax**, batch 256, 400 steps, 14 log-spaced eval
points; states with fewer than two legal actions excluded; the batch sequence seeded and identical
across cells at a given seed. **ABSORPTION** = held-out on-slice top-1 agreement, held out **by
battle file**. **COLLATERAL** = off-slice divergence from the student's *own* pre-probe policy
(masked `KL(now ‖ original)`, top-1 agreement, mean `|ΔV|`). The step-1 shock is recorded and
reported as an ordering only; it failed value-level admission in the source battery and nothing here
re-admits it. Primary lr is **1e-4**, the step size the 2026-08-31 lr-licensing probe licensed.

**What is new is the STUDENT axis.** Two roster facts do the work:

* **ARM A is the confound-breaking cell.** `plain0`, `ecol0` and `fold1` are all `ai_v9_29_rev1_0823`
  + exactly **3,078,768** steps and end at the byte-identical total **28,067,760**. Only `fold1`
  carries a distill term. **Maturity — the ledger's current prime suspect for the v8 gap — is held
  exactly fixed across this cell**, which is the whole reason it is worth running.
* ⚠️ **`ecol0` is NOT the ecology control it was designed to be, and that turns out to be an
  improvement.** `ai_v9_58_R2CTRL_0827` was launched to hold the team distribution constant against
  `fold1` while folding no loss. It did not: `apply_distill_team_bias`'s own docstring records that
  R2CTRL "got an effective bias of 0.0 (the pairs were parsed only above coef 0)" — the defect
  `gen3_distill_bias_at_coef0_v1` later fixed — and its metadata confirms it, `_distill_pairs`
  **empty** against 5 for `fold1`. So `ecol0` is a **second replicate of the plain +3M continuation**,
  giving something the pre-registration lacked: a **RUN-TO-RUN** noise floor for the 0-fold condition
  at matched steps. §4.5 shows that replicate overturning a result this battery would otherwise have
  reported.

### 2.1 Teacher choice, and the slice-exposure confound made explicit

Both teachers are `ai_v9_29_rev1_0823`-initialised exploiters targeting `ai_v9_59_R2ACTION_0827`, so
each sits at the **same DAG distance from every ARM-A student**. They differ in the one thing a
single teacher cannot separate from fold count — whether the student has already been *taught the
teacher's teams*, computed from metadata rather than asserted:

| student | folds | teams taught (content) | teams in fold ecology | teacher `f` slice taught | teacher `a` slice taught |
|---|---|---|---|---|---|
| `root` | 0 | 0 | 0 | no | no |
| `plain0` | 0 | 0 | 0 | no | no |
| `ecol0` | 0 | 0 | 0 | no | no |
| `fold1` | 1 | 9 | 9 | no | **YES** |
| `fold2` | 2 | 24 | 24 | **YES** | **YES** |
| `fold2self` | 2 | 9 | 12 | ecology only | **YES** |
| `tick1` | 1 | 7 | 7 | no | **YES** |
| `tick1x2` | 2 | 7 | 7 | no | **YES** |
| `gen17` | 0 | 0 | 0 | no | no |

`f` (`ai_v9_68_R3F6f_0828`) is **PRIMARY** precisely because its slice is content-untaught to every
arm of the decisive cell and to the whole tick lineage; only `fold2` has seen it, via `R4S3c`'s
superset, and that is flagged wherever `fold2` is read. `a` (`ai_v9_63_R3F6a_0828`) is the
teacher-independence check *and* the deliberate slice-EXPOSED contrast — `fold1` was taught its two
teams by `R2F5a`.

An ancestry-neutral teacher does not exist for a roster lying on one chain (source battery, caveat
2), so `a0` falls with DAG distance for structural reasons and **`a_max` is the primary index**.

---

## 3. All cells

| cell | student | folds | steps | T | lr | seed | a0 | a_max | gain | KL@400 | off-agree@400 | \|dV\|@400 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fold1_a1e4_s1` | `fold1` | 1 | 28.07M | a | 1e-04 | 1 | 0.724 | **0.830** | +0.106 | 0.425 | 0.751 | 2.55 |
| `plain0_a1e4_s1` | `plain0` | 0 | 28.07M | a | 1e-04 | 1 | 0.652 | **0.824** | +0.172 | 0.466 | 0.728 | 2.46 |
| `plain0_f3e4_s1` | `plain0` | 0 | 28.07M | f | 3e-04 | 1 | 0.675 | **0.780** | +0.105 | 0.757 | 0.634 | 7.59 |
| `ecol0_f1e4_s1` | `ecol0` | 0 | 28.07M | f | 1e-04 | 1 | 0.661 | **0.803** | +0.142 | 0.540 | 0.687 | 4.73 |
| `ecol0_f1e4_s2` | `ecol0` | 0 | 28.07M | f | 1e-04 | 2 | 0.661 | **0.811** | +0.150 | 0.520 | 0.684 | 4.76 |
| `fold1_f1e4_s1` | `fold1` | 1 | 28.07M | f | 1e-04 | 1 | 0.655 | **0.816** | +0.161 | 0.539 | 0.697 | 2.89 |
| `fold1_f1e4_s2` | `fold1` | 1 | 28.07M | f | 1e-04 | 2 | 0.655 | **0.818** | +0.162 | 0.554 | 0.700 | 3.60 |
| `plain0_f1e4_s1` | `plain0` | 0 | 28.07M | f | 1e-04 | 1 | 0.675 | **0.808** | +0.133 | 0.515 | 0.687 | 8.09 |
| `plain0_f1e4_s2` | `plain0` | 0 | 28.07M | f | 1e-04 | 2 | 0.675 | **0.806** | +0.131 | 0.511 | 0.693 | 9.44 |
| `root_f1e4_s1` | `root` | 0 | 24.99M | f | 1e-04 | 1 | 0.687 | **0.810** | +0.123 | 0.431 | 0.741 | 5.83 |
| `root_f1e4_s2` | `root` | 0 | 24.99M | f | 1e-04 | 2 | 0.687 | **0.819** | +0.133 | 0.421 | 0.735 | 7.99 |
| `fold2_f1e4_s1` | `fold2` | 2 | 32.60M | f | 1e-04 | 1 | 0.663 | **0.823** | +0.160 | 0.526 | 0.707 | 4.18 |
| `fold2_f1e4_s2` | `fold2` | 2 | 32.60M | f | 1e-04 | 2 | 0.663 | **0.818** | +0.155 | 0.542 | 0.716 | 2.98 |
| `fold2self_f1e4_s1` | `fold2self` | 2 | 32.62M | f | 1e-04 | 1 | 0.595 | **0.809** | +0.214 | 1.103 | 0.670 | 3.82 |
| `tick1_f1e4_s1` | `tick1` | 1 | 35.07M | f | 1e-04 | 1 | 0.642 | **0.814** | +0.172 | 0.583 | 0.686 | 5.58 |
| `tick1x2_f1e4_s1` | `tick1x2` | 2 | 40.07M | f | 1e-04 | 1 | 0.642 | **0.815** | +0.173 | 0.549 | 0.695 | 9.37 |
| `gen17_f1e4_s1` | `gen17` | 0 | 22.89M | f | 1e-04 | 1 | 0.603 | **0.792** | +0.189 | 0.610 | 0.663 | 3.95 |
| `ctrl_fold1_f1e4_s1` | `fold1` | 1 | 28.07M | f | 1e-04 | 1 | 1.000 | **1.000** | +0.000 | 0.282 | 0.794 | 3.33 |
| `ctrl_plain0_f1e4_s1` | `plain0` | 0 | 28.07M | f | 1e-04 | 1 | 1.000 | **1.000** | +0.000 | 0.336 | 0.774 | 7.22 |

---

## 4. Results

### 4.1 A1 — the matched-age cell. Direction consistent 4/4; magnitude below the registered bar.

| seed | a_max plain0 (0 folds) | a_max ecol0 (0 folds, 2nd replicate) | a_max fold1 (1 fold) | Δ vs plain0 | Δ vs ecol0 | run-to-run gap | margin over it | > registered 0.018 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.808 | 0.803 | **0.816** | +0.007 | +0.012 | 0.005 | +0.002 | NO |
| 2 | 0.806 | 0.811 | **0.818** | +0.012 | +0.007 | 0.005 | +0.002 | NO |

**The 1-fold student's ceiling exceeds both 0-fold replicates on both seeds — 4 of 4 comparisons,
+0.007 to +0.012, mean +0.0096.** Probe-seed noise within a student is tiny here (`plain0` 0.0025,
`fold1` 0.0017), and the two 0-fold *training runs* differ by 0.0050 on each seed. So the effect is
~2x the run-to-run gap and ~4-5x the probe-seed gap — but it **misses the registered 0.018 threshold
on every seed**, and a margin of +0.002 over the run-to-run floor is not a margin anyone should
bank. **A1 FAILS as registered.**

The gain column is where it is most visible: `fold1` starts **lowest** against this never-taught
teacher (a0 0.655 vs `plain0`'s 0.675 — the fold moved it off the rev-1 manifold that `plain0` and
the teacher still share) and finishes **highest**, for a gain of +0.161/+0.162 against +0.131/+0.150.

### 4.2 A2 — the 0 → 1 → 2 ladder

| lineage × seed | students | folds | steps (M) | a_max | ρ(folds) | ρ(steps) | monotone |
|---|---|---|---|---|---|---|---|
| `rev_lineage_s1` | plain0 / fold1 / fold2 | [0, 1, 2] | [28.07, 28.07, 32.6] | [0.808, 0.816, 0.823] | 1.00 | 1.00 | YES |
| `rev_lineage_s2` | plain0 / fold1 / fold2 | [0, 1, 2] | [28.07, 28.07, 32.6] | [0.806, 0.818, 0.818] | 1.00 | 1.00 | YES |
| `rev_lineage_self2_s1` | plain0 / fold1 / fold2self | [0, 1, 2] | [28.07, 28.07, 32.62] | [0.808, 0.816, 0.809] | 0.50 | 0.50 | NO |
| `tick_lineage_s1` | root / tick1 / tick1x2 | [0, 1, 2] | [24.99, 35.07, 40.07] | [0.81, 0.814, 0.815] | 1.00 | 1.00 | YES |

Monotone in fold count on the rev lineage at **both** seeds, and on the **tick lineage** — a second,
independent fold chain (`rev-1 → tick-1 → tick-1-dosext`, different teachers, different era) whose
slice teacher `f` never taught. ⚠️ **Only the 0→1 rung is age-matched.** The 1→2 rung adds 4.5M steps
*and* slice exposure (`fold2` was taught teacher `f`'s teams via `R4S3c`), so three variables move at
once there; the tick lineage's rungs are age-confounded throughout (25.0 / 35.1 / 40.1M).
**A2 PASSES on the rev lineage as registered, and the pass is worth less than it looks** for that
reason.

### 4.3 Mission item 4 — absorption gained per step, with a fold and without

| arm | folds | own steps | a_max(root) | a_max | Δa_max | Δa_max per 1M steps | Δa0 |
|---|---|---|---|---|---|---|---|
| `ecol0_s1` | 0 | 3,078,768 | 0.810 | 0.803 | -0.007 | -0.002 | -0.026 |
| `ecol0_s2` | 0 | 3,078,768 | 0.819 | 0.811 | -0.008 | -0.003 | -0.026 |
| `fold1_s1` | 1 | 3,078,768 | 0.810 | 0.816 | +0.006 | 0.002 | -0.032 |
| `fold1_s2` | 1 | 3,078,768 | 0.819 | 0.818 | -0.002 | -0.001 | -0.032 |
| `gen17_anchor_s1` *(LEVEL anchor, not a slope)* | 0 | 22,887,936 | 0.810 | 0.792 | -0.018 | — | -0.083 |
| `plain0_s1` | 0 | 3,078,768 | 0.810 | 0.808 | -0.002 | -0.001 | -0.012 |
| `plain0_s2` | 0 | 3,078,768 | 0.819 | 0.806 | -0.013 | -0.004 | -0.012 |

**3.08M steps of plain PPO does not raise absorbability — it slightly LOWERS it** (four 0-fold cells,
−0.002 … −0.013, mean **−0.0075**). The identical 3.08M *with* a distill term is flat to slightly
positive (**+0.0021** mean). The ~+0.010 difference is the same quantity A1 measures, seen from the
root instead of from the siblings.

This does not contradict the 2026-08-28 finding that absorption rises with age: that was measured
over a 2M → 25M span, and this says only that **3M is too short an interval for age to buy anything
at this maturity, while a fold inside the same 3M is not.** ⚠️ `root`'s own probe-seed spread is
**0.0092** (0.810 vs 0.819), the largest in the battery and nearly the size of the effect — which is
why this arm is corroboration for A1, not an independent result.

### 4.4 THE MECHANISM — it is folded CONTENT, not fold COUNT

`fold2self` (`ai_v9_72_R3SELF_0828`) is 2 folds deep, but its second fold distilled **its own
parent** — same optimizer work, zero external content. If absorbability tracked the number of
distillation episodes, it should sit with `fold2`. It does not:

| student | folds | 2nd fold content | a_max | KL@400 | off-agree@400 | a0 |
|---|---|---|---|---|---|---|
| `plain0` / `ecol0` | 0 | — | 0.803–0.811 | 0.511–0.540 | 0.684–0.693 | 0.661–0.675 |
| `fold1` | 1 | external | **0.816 / 0.818** | 0.539 / 0.554 | 0.697 / 0.700 | 0.655 |
| `fold2` | 2 | external | **0.823 / 0.818** | 0.526 / 0.542 | 0.707 / 0.716 | 0.663 |
| `fold2self` | 2 | **SELF (zero content)** | **0.809** | **1.103** | **0.670** | **0.595** |

The self-fold **erases `fold1`'s advantage** (0.809, back inside the 0-fold band) while **doubling
collateral KL** (1.103 against 0.526–0.554 for the content folds) and driving `a0` to 0.595, the
lowest in the battery. This is the micro-instrument's version of the rev-3 SELF-FOLD production
result (self-distillation actively destructive at production scale, ledger `ac40230`), and it is the
single cleanest statement this battery makes: **the thing that raises absorbability is external
teacher content passing through the network, not another round of distillation optimization.**

### 4.5 THE METHODOLOGICAL FINDING — the accidental second replicate overturns a headline

On the first three cells the `|ΔV|` column looked like the result: `fold1` at **2.89** against
`plain0`'s **8.09** — a 2.8x reduction in critic-side collateral *at higher absorption*. With one
0-fold arm that reads as a fold effect and this file would have reported it as one.

The second 0-fold replicate kills it. `ecol0` — same parent, same 3,078,768 steps, no distill term —
sits at **4.73 / 4.76**, and `root` at 5.83 / 7.99. The **0-fold condition itself spans 4.73–9.44**,
a run-to-run spread of ~4.7, while `fold1` sits ~1.4 below the nearer replicate. **The noise exceeds
the effect; `|ΔV|` cannot carry a fold claim** (and `tick1x2`, a 2-fold student, posts the
battery's second-highest `|ΔV|` at 9.37, which settles it).

`ecol0` was in the roster **by accident** — launched as an ecology control, the bias never applied,
and it survived as a replicate. The general rule: **a matched-age comparison needs two arms on the
reference side, not one**, because the estimand is a difference between *training runs* and the
run-to-run term is not observable from within a single run.

### 4.6 What the content controls say about collateral

Distilling each student onto **its own argmax** — same optimizer, same states, same steps, zero new
content — separates the student's landscape from the teacher's content:

| student | ctrl KL@400 | with-content KL@400 | overshoot share | NET content KL | ctrl \|ΔV\| | with-content \|ΔV\| |
|---|---|---|---|---|---|---|
| `plain0` (0 folds) | 0.336 | 0.515 | **65%** | 0.179 | 7.22 | 8.09 |
| `fold1` (1 fold) | 0.282 | 0.539 | **52%** | 0.257 | 3.33 | 2.89 |

Two readings, both modest and both n=1 seed:

* **The `|ΔV|` difference is a property of the STUDENT, not of the content.** With *zero* teacher
  content the two students still sit at 7.22 vs 3.33 — so whatever makes `fold1`'s critic less
  perturbable is in its loss landscape, not in what it is being taught. (§4.5's run-to-run caveat
  applies: no control was run on `ecol0`, so this could be the same replicate variance.)
* **`fold1` spends a smaller fraction of its displacement on content-free overshoot** (52% vs 65%)
  and carries **more net content divergence** (0.257 vs 0.179) at a higher ceiling. That is the
  direction (a) predicts, on the decomposition the lr-licensing battery introduced — and, per that
  battery's own warning, a KL difference of differences is a decomposition, not a per-arm score.

### 4.7 Teacher independence

| student | folds | a0 | a_max | gain | KL@400 | off-agree@400 | \|dV\|@400 |
|---|---|---|---|---|---|---|---|
| `fold1` | 1 | 0.724 | **0.830** | +0.106 | 0.425 | 0.751 | 2.55 |
| `plain0` | 0 | 0.652 | **0.824** | +0.172 | 0.466 | 0.728 | 2.46 |

On the second teacher the ceiling ordering **replicates**: `fold1` 0.830 vs `plain0` 0.824,
**+0.006**, the same direction and magnitude as teacher `f`. Note this arm is *slice-exposed* —
`fold1` was literally taught teacher `a`'s two teams — which shows up exactly where it should, in
`a0` (0.724 vs 0.652, a +0.072 head start), and **not** in an inflated ceiling: the advantage on the
exposed teacher (+0.006) is if anything **smaller** than on the never-taught one (+0.007/+0.012).
That is evidence the ceiling effect is not slice memory.

### 4.8 The ancestry-free anchor, and the one lr-3e-4 cell

`gen17` (`ai_v9_21_gen17_pfspoff_0820`, 0 folds, 22.89M, **shares no weights with the teacher or any
other student**) reaches `a_max` **0.792** from `a0` **0.603** — below every rev-lineage cell
(0.803–0.823). That reproduces the source battery's caveat-2 finding at the expected size: the
shared-weights bonus is worth roughly **+0.02 on the ceiling and +0.06 on `a0`**, i.e. *larger than
the fold effect itself*. It is a level anchor, not a slope — a slope needs two of its checkpoints and
only its final is in this battery — and it is why the fold comparison is confined to one lineage.

The single lr-3e-4 cell (`plain0`) reproduces the 2026-08-31 licensing result on these ingredients:
`a_max` 0.808 → **0.780** and KL@400 0.515 → **0.757** going from 1e-4 to 3e-4, i.e. 1e-4 dominates
on both axes here too. **It cannot say whether the fold ordering is lr-invariant** — that needs the
`fold1` cell at 3e-4, which the cut lost (§9).

---

## 5. Post-fold consolidation (hypothesis b)

### 5.1 The quantity, and why it is not what the hypothesis wants it to be

⚠️ **The distill term is active for the WHOLE fold run.** A fold is not a point event followed by a
consolidation window: the CE/KL term rides every PPO step from the fork to the final checkpoint. So
the only "post-fold steps" quantity the metadata can supply is **simultaneously the DOSE and the
consolidation window**. No run in the archive turns the distill term off partway and keeps training,
so no re-analysis can separate them. That is a property of the design, not a gap in the reading.

### 5.2 The table

| fold | run | parent | steps under the distill term | coef | lr | teachers | prior folds | arch signature | untaught outcome | CI | z |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v8** | `ai_v8_14_distill3_0725` | `ai_v8_04_distill_4teacher_0722` | **14,922,176** | 1.0 | 7e-05 | 3 | 1 | `gen3_opp_hp_typed_candidates_v1` | **+5.42pp** | [+3.44, +7.42] | +4.83 |
| **rev-3** | `ai_v9_70_R3ACTION_0828` | `ai_v9_59_R2ACTION_0827` | **4,553,328** | 0.1761 | 0.0003 | 6 | 1 | `gen3_critic_route_wave_v1` | **-0.75pp** | [-4.56, +3.00] | -0.39 |
| **rev-2** | `ai_v9_59_R2ACTION_0827` | `ai_v9_29_rev1_0823` | **3,078,768** | 0.181 | 0.0003 | 5 | 0 | `gen3_critic_route_wave_v1` | **-7.06pp** | [-10.56, -3.50] | -3.86 |

Spearman ρ(steps under the distill term, untaught outcome) = **1.00**, n=3.  Companion on (a)'s variable: ρ(prior folds, untaught outcome) = **0.50**.

### 5.3 The ordering, and the refusal

**Both candidate orderings are consistent, and they are mutually confounded.** Ranked by steps under
the distill term the three folds order v8 > rev-3 > rev-2, exactly their outcome order (ρ = **+1.00**).
Ranked by prior-fold count they order v8 = rev-3 > rev-2 (ρ = **+0.50**, the tie broken by 6.2pp of
outcome — most of the range — which is why the fold-count axis is the *weaker* of the two on this
table even though (a) is the hypothesis that survives §4).

**No coefficient is fitted and none should be.** With n = 3 a perfect ordering arises by chance with
probability 1/6 under the null, and the middle point is not securely separated from the bottom one:
the rev-3-minus-rev-2 difference is **+6.3pp at z = 1.88**, which the source probe itself reports as
*suggestive, not established*.

### 5.4 Four confounds, any one of which could carry the whole ordering

1. **v8 is a DIFFERENT ARCHITECTURE** (`gen3_opp_hp_typed_candidates_v1` against the roster's
   `gen3_critic_route_wave_v1`). It cannot be loaded by this instrument, cannot be a probe cell, and
   is not a controlled comparison with anything in the gen era.
2. **The fold hyper-parameters differ by ~5x** — v8 at coef 1.0 / lr 7e-5, the rev folds at
   coef ≈ 0.18 / lr 3e-4.
3. **Absolute maturity differs by an order of magnitude** (277M vs 25M / 28M parents) and is
   **rank-identical to steps-under-distill on these three rows**. The ledger's standing read is that
   maturity is the prime suspect by elimination; this table cannot separate "the fold consolidated"
   from "the parent was mature".
4. **rev-3's null has a live alternative that has nothing to do with consolidation.**
   `rev3_untaught_pulldown_2026-08-30.md` §3: R2-ACTION's mean win rate on the untaught set was
   already **0.4975** — the level at which there is no per-team edge left to redistribute. *"rev-3
   stopped robbing"* and *"rev-3 had nothing left to rob on this set"* predict the same −0.75, and
   that probe states it cannot separate them.

**⇒ (b) is eliminated as a separable question on the available evidence, not answered.** The
registered prediction that it would *not* order is FALSE; the ordering is real and perfect; and it is
uninterpretable at n = 3 with four confounds, one of which (maturity) is the field's own leading
hypothesis.

---

## 6. Registered predictions, scored

| id | statement (registered before any cell ran) | outcome |
|---|---|---|
| **M1** | prior-fold count is CONFOUNDED with age in EVERY cell the archive supports; the honest deliverable is the confounded table plus the cell that would break it | **FAILS — and this is the good news.** The archive holds a matched-total-step fold-count contrast at **28,067,760** across 12 non-exploiter runs (fold counts 0 and 1), and this battery MEASURED it. The confound is real at fold count ≥ 2 and absent at 0→1. |
| **M2** | post-fold consolidation does NOT order with the untaught outcome (a cheap elimination) | **FAILS as stated — it orders perfectly** (ρ = +1.00, n = 3). But the *conclusion* the prediction was reaching for survives by a different route: the axis is dose- and maturity-confounded and cannot be read as consolidation (§5.4). |
| **A1** | at 28,067,760 steps the 1-fold student's `a_max` exceeds BOTH 0-fold arms by > 0.018 on both seeds | **FAILS.** Direction correct 4/4 (+0.007 … +0.012), magnitude below the bar, margin over run-to-run noise +0.002. |
| **A2** | `a_max` monotone non-decreasing in fold count | **PASSES** on the rev lineage (both seeds) and on the tick lineage; **FAILS** when the zero-content self-fold is substituted at rung 2 — which is §4.4's mechanism, not a defect. |

Two of four registered predictions fail in the direction of *more* signal than expected, and A1 —
the one that would have licensed a claim — fails. That asymmetry is the honest summary: **the effect
is real in direction and small in size.**

---

## 7. Which of (a)/(b) survives, and what would decide it

**(a) survives, weakly, and with a sharper mechanism than the hypothesis proposed.** The hypothesis
as written was about *fold history* — a network that has been folded is organised to be folded
again. What the battery supports is narrower: **a fold that carries EXTERNAL content raises the
absorption ceiling by ~0.01 at matched age, and a content-free self-fold does the opposite.** The
direction survives two seeds, two 0-fold replicates, two teachers (one slice-exposed, one not), two
independent lineages and the 0→1→2 ladder. It does not survive its own registered effect-size bar.

**(b) does not survive as a separable question.** It is not refuted — it is unmeasurable on this
archive, because every candidate proxy for "consolidation time" is also dose, also maturity, and (for
v8) also architecture.

### The cell that would decide (a) — named, and priced

**A 0-fold or 1-fold GENERALIST at ~32.6M steps.** The archive has none: every rev-lineage checkpoint
past 30M is either team-pinned (an exploiter, so not a comparable student) or already carries 2
folds. Concretely the missing arm is a **plain +4.5M continuation of `ai_v9_59_R2ACTION_0827` with no
distill term** — the exact sibling of `ai_v9_76_R4ACTION_0830` the rev-4 table never launched. At the
fleet's rate (3M steps ≈ 1.5 GPU-h) it costs **≈ 2.3 GPU-hours**, and it converts the 0/1/2 ladder
from age-confounded to age-matched at every rung. Run it with **two replicates**, per §4.5.

Two cheaper additions that raise A1's power without new training: **more probe seeds on the existing
ARM-A cells** (the effect is ~4x probe-seed noise, so seeds 3–6 would tighten it substantially at
~11 min each), and **a third 0-fold replicate** — `ai_v9_40_fdC_ecology_0825` is also 0-fold at
28,067,760 and already on disk.

### The cell that would decide (b)

**A fold that turns its distill term OFF partway and keeps training** — `--distill-coef` annealed to
0 at the halfway point, against a dose-matched arm that keeps it on. That is the only design that
separates consolidation time from dose, and one +3M pair buys it. Nothing in the archive does this.

### What this says to the standing "maturity" hypothesis

The ledger's current read is that **maturity is the prime suspect by elimination** for v8's untaught
gift. This battery does not contest that and supplies one datum bearing on it: over the 3M interval
these arms span, **plain training bought no absorbability at all** (§4.3) while a fold in the same
interval bought ~0.01. If maturity is the mechanism, its per-step rate at 25–28M is below what this
instrument can see in 3M — consistent with the 2026-08-28 age curve, which needed a 23M span to
resolve it. **Folding and maturity are therefore not competing explanations at this scale; they act
on different timescales**, and the planned one-week run tests maturity while the 2.3 GPU-h sibling
arm above tests folding.

---

## 8. Caveats — what this probe can and cannot say

1. **This is not a fold simulation.** No PPO loss beside the distill term, no `--distill-team-bias`
   sampling, no environment interaction, no entropy/advantage pressure pulling the policy back, no lr
   schedule. It measures the student-side CAPACITY to absorb, in isolation; a real fold's outcome is
   that term times everything removed here. Source battery caveat 1, unchanged.
2. **The absorption effect is small relative to several noise estimates.** Run-to-run 0.0050,
   `root`'s probe-seed spread 0.0092, the effect ~0.0096. Its strength is consistency across
   independent axes, not any single margin.
3. **`a0` is not comparable across DAG distance.** The teachers are rev-1 children, so a student
   further down the chain starts further away for structural reasons unrelated to folding. `a_max` is
   primary for exactly this reason (source battery caveat 2).
4. **The registered 0.018 threshold is a PROBE-seed bound, not a run-to-run bound.** `plain0` vs
   `ecol0` supplies the latter for the first time; both are reported and the registered verdict is
   scored against what was registered.
5. **State provenance is eval-trace-biased**, and OFF is drawn from `ai_v9_29_rev1_0823` — the rev
   roster's own ancestor, but not `gen17`'s lineage at all. Collateral stays well-defined (drift from
   that student's own reference on those states) but is further from "damage to what this student
   would actually have done next" for the ancestry-free arm.
6. **400 steps over-trains**; `KL@400` is an over-training endpoint, not a fold outcome. The
   informative regime is the first ~64 steps (source battery caveat 4).
7. **Single-seed cells are marked as such** — the teacher-`a` arm, both content controls,
   `fold2self`, and the tick lineage are seed 1 only.

---

## 9. MISSING cells — never interpolated

| cell | why |
|---|---|
| a 0-fold or 1-fold GENERALIST at ~32.6M steps | **does not exist in the archive** — see §7. The single most valuable thing this probe could not buy. |
| `fold2self`, tick lineage, `gen17`, both content controls, the teacher-`a` arm at **seed 2** | not run — the battery was cut at 19 of 27 planned cells against a 5 h budget, **~50 min of which was lost to a `pgrep` self-match deadlock in the queue driver** (the waiter's pattern matched its own argv, so it waited on itself; killed by explicit PID and relaunched). Cells were priority-ordered before the cut, so the losses are the low-priority tail. |
| **`fold1` at lr 3e-4** (and `fold2`) | same cut. `plain0` at 3e-4 ran (§4.8) and reproduces the licensing verdict, but with only the 0-fold arm at that lr, **whether the ceiling ordering is lr-invariant is unanswered, not answered**. This is the cheapest missing cell in the battery — one 11-min run. |
| a content control on **`ecol0`** | not run. §4.6's student-landscape reading therefore rests on one 0-fold arm and inherits §4.5's replicate warning. |
| a 3-fold student | the archive's deepest fold chain is 2. |
| the v8 fold as a PROBE cell | `arch_signature = gen3_opp_hp_typed_candidates_v1` — it cannot be loaded by this instrument at all. The v8 row in §5.2 is METADATA ONLY; no absorption number exists for it. |
| tick-1 in the consolidation table | tick-1 was graded on three meters (ledger 2026-08-25) but never on the `untaught_pp` instrument the other three rows share. Adding it would mix meters. |
| a full PPO-context fold | out of scope by construction — caveat 1. |

---

## 10. Reproducing

```bash
export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd designs/research_state/measurements
nice -n 15 python folding_history_probe.py lineage          # the DAG, from metadata alone
nice -n 15 python folding_history_probe.py build-states
nice -n 15 python folding_history_probe.py probe fold1_f1e4_s1 fold1 f 1 400 1e-4
nice -n 15 python folding_history_probe.py probe ctrl_fold1_f1e4_s1 fold1 'f*' 1 400 1e-4  # QUOTE it
nice -n 15 python folding_history_probe.py aggregate
nice -n 15 python folding_history_probe.py report           # emits this file's tables
```

`<student>` is a key of `STUDENTS`; `<teacher_set>` is `f` / `a`, or `f*` / `a*` for the CONTENT
CONTROL on that set's states. **Quote the control token** — bare `f*` is glob-expanded by the shell
and the cell dies on argv parsing (it killed four cells on the lr battery). One cell is ~11-16 min of
one core beside the live fleet. The producer writes `fh_states_{a,f}.npz`,
`fh_teacher_targets_{a,f}.npz`, `fh_state_provenance.json`, `fh_lineage.json` and
`fh_results/<cell>.json` into its own directory and never writes to `models/`; the `.npz` caches and
per-cell results are regenerable and uncommitted, following the 2026-08-28 convention — the committed
`.json` carries every curve.
