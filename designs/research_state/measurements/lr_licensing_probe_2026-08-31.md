# LR LICENSING PROBE — is a smaller distill step licensed on the REAL rev-4 fold ingredients?

**2026-08-31 · producer `lr_licensing_probe.py` (this directory) · data
`lr_licensing_probe_2026-08-31.json` (every cell, every curve, both predictions scored).**
18 probe cells · **2.62 CPU-hours** (417–634 s per cell) on 2 `nice 15` single-thread processes ·
no GPU · `models/` read-only · **zero cell failures, `dropped_kwargs` empty on every load**.
Measured beside the live `ai_v9_77_G1LEAN_0830` run at load 20–37, which stretches wall time but
cannot move a result: every cell is deterministic offline compute on cached states.

The question, registered before the data (ledger `38fa4eb`, sharpened by `ac40230`): the
2026-08-28 distillability battery ended on a **prediction, not a result** — *"the fold runs above
the mature student's damage threshold, and lowering the distill-term step size buys ceiling and
collateral together"* — measured on the rev-1 lineage against a rev-2 teacher. The rev-3 SELF-FOLD
control then measured self-distillation as **actively destructive at production scale** (−9pp on
teams the base was good at), which is that battery's zero-content control reproduced in a real PPO
fold and raises the stakes: every fold's net is **teacher content minus overshoot damage**, so if a
smaller step halves the damage at equal absorption it beats any shape choice. This probe
manufactures the missing datum — the same instrument, run on the **actual ingredients of the fold
the revolution will use**.

**Verdict: LICENSED.** lr 1e-4 Pareto-dominates 3e-4 on **all six** teacher × seed arms — higher
absorption ceiling *and* lower collateral, with no arm trading. Both registered predictions PASS.

---

## 1. The ingredients — and a correction to the dispatch

The mission named `ai_v9_70_R3ACTION_0828` as the likely fold parent and instructed that the LIVE
rev-4 fold's recorded parent wins. It does not match:

| role | run | provenance |
|---|---|---|
| **STUDENT (fold parent)** | `ai_v9_59_R2ACTION_0827/final_model.zip` | `ai_v9_76_R4ACTION_0830/metadata.json` → `original_command` → `--model models/ai_v9_59_R2ACTION_0827/final_model.zip` |
| **TEACHER a** | `ai_v9_73_R4S3a_0829/final_model.zip` | the rev-4 fold's `--distill-teacher` list, entry 1 |
| **TEACHER b** | `ai_v9_74_R4S3b_0829/final_model.zip` | entry 2 |
| **TEACHER c** | `ai_v9_75_R4S3c_0829/final_model.zip` | entry 3 |

**The fold-arm table forks a COMMON base.** rev-2 (`ai_v9_59`), rev-3 (`ai_v9_70`) and the live
rev-4 (`ai_v9_76`) all record `--model models/ai_v9_59_R2ACTION_0827/final_model.zip` — that is what
keeps the frozen comparison arms matched. So `ai_v9_59` is the fold parent in the sense that
matters, and `ai_v9_70` is a *sibling arm's output*, not rev-4's parent. The live fold's production
lr is `3e-4` and its `--distill-coef` is `0.1761`, both read from the same recorded command.

Every checkpoint loads through the prober's read-only path with **`dropped_kwargs` empty**, so each
rebuilt extractor is the one that played; all four carry `arch_signature = gen3_critic_route_wave_v1`,
obs dim 2501.

## 2. The instrument

Mechanically the ADMITTED `distillability_index_probe.py` (2026-08-28 §2), unchanged: the student's
**full policy** (extractor + both heads, nothing frozen — what a real fold updates) trained with Adam
on masked cross-entropy to the teacher's **argmax**, batch 256, 400 steps, 14 log-spaced eval points;
states with fewer than two legal actions excluded; the batch sequence seeded and identical across
cells at a given seed.

* **ON-SLICE** — 4,200 states from *that teacher's own* eval traces, split **by battle file** 75/25
  into a 3,000-state training pool and a **1,200-state held-out pool from battles never trained on**.
  Each rev-4 teacher is pinned to 8 teams, matching the fold spec's per-teacher team list.
* **OFF-SLICE** — 1,500 states, every state on that teacher's pinned teams excluded.
* **ABSORPTION** = held-out on-slice top-1 agreement with the teacher.
  **COLLATERAL** = off-slice divergence from the student's *own* pre-probe policy (masked
  `KL(now ‖ original)`, top-1 agreement, mean `|ΔV|`).

**One deliberate deviation from the 2026-08-28 build, stated because it changes what "off-slice"
means.** That battery drew OFF from the student's parent run. Here the parent's own traces are far
too narrow to measure collateral breadth — measured: `ai_v9_59` has **474 trace files but only 9
distinct teams across 2 eval steps**, because a fold run's eval traces cluster on the fold teams.
Drawing OFF from it yielded 444–808 rows over 5–8 teams. OFF is therefore drawn from
**`ai_v9_29_rev1_0823`** — the student's own lineage ancestor and the original battery's off-source —
which has 2,456 files / 281 teams / 12 eval steps, giving the full **1,500 rows over 220–231 distinct
teams across 12 eval steps** per teacher set. Collateral is defined as drift from the student's own
pre-probe policy *on these states*, so breadth is what the meter needs; caveat 3 of the 2026-08-28
instrument (eval-trace-biased provenance, not the student's on-policy distribution) applies here
unchanged and is not repaired by this choice.

**Cells.** 3 teachers × {3e-4, 1e-4} × 2 seeds = 12 with-content, plus the **zero-content control**
(targets = the student's OWN argmax; same optimizer, same states, same step count) at both lrs × 2
seeds = 4. The step-1 shock is recorded but reported as an ordering only — it failed value-level
admission in the source battery and nothing here re-admits it.

## 3. Results

### 3.1 Per-arm cells

| arm | lr | a0 | a_max | gain | KL@400 | off-agree@400 | \|dV\|@400 | KL@gain+0.05 |
|---|---|---|---|---|---|---|---|---|
| `a_3e4_s1` | 3e-04 | 0.602 | 0.725 | +0.123 | 1.219 | 0.592 | 4.83 | 1.003 |
| `a_1e4_s1` | 1e-04 | 0.602 | **0.743** | +0.142 | **0.651** | 0.699 | 2.31 | 0.646 |
| `a_3e4_s2` | 3e-04 | 0.602 | 0.707 | +0.105 | 1.176 | 0.602 | 5.14 | 1.218 |
| `a_1e4_s2` | 1e-04 | 0.602 | **0.739** | +0.137 | **0.664** | 0.687 | 2.54 | 0.811 |
| `b_3e4_s1` | 3e-04 | 0.681 | 0.745 | +0.064 | 0.804 | 0.667 | 4.75 | 0.604 |
| `b_1e4_s1` | 1e-04 | 0.681 | **0.775** | +0.094 | **0.557** | 0.725 | 2.84 | 0.303 |
| `b_3e4_s2` | 3e-04 | 0.681 | 0.742 | +0.061 | 0.838 | 0.663 | 5.27 | 0.572 |
| `b_1e4_s2` | 1e-04 | 0.681 | **0.765** | +0.084 | **0.507** | 0.735 | 2.09 | 0.339 |
| `c_3e4_s1` | 3e-04 | 0.585 | 0.722 | +0.137 | 1.046 | 0.621 | 6.77 | 0.960 |
| `c_1e4_s1` | 1e-04 | 0.585 | **0.752** | +0.167 | **0.634** | 0.703 | 2.45 | 0.587 |
| `c_3e4_s2` | 3e-04 | 0.585 | 0.729 | +0.144 | 0.870 | 0.648 | 4.60 | 0.976 |
| `c_1e4_s2` | 1e-04 | 0.585 | **0.756** | +0.171 | **0.613** | 0.705 | 2.65 | 0.109 |
| `ctrl_a_3e4_s1` | 3e-04 | 1.000 | 1.000 | +0.000 | 0.615 | 0.717 | 5.42 | — |
| `ctrl_a_1e4_s1` | 1e-04 | 1.000 | 1.000 | +0.000 | **0.250** | 0.825 | 1.77 | — |
| `ctrl_a_3e4_s2` | 3e-04 | 1.000 | 1.000 | +0.000 | 0.600 | 0.715 | 4.50 | — |
| `ctrl_a_1e4_s2` | 1e-04 | 1.000 | 1.000 | +0.000 | **0.244** | 0.829 | 1.67 | — |

`a0` differs per teacher (0.585–0.681) because it is the parent's pre-probe agreement with *that*
teacher — a property of the teacher pair, not of the lr; both lr arms of a given teacher × seed share
it exactly, which is the check that the arms are matched.

The control's `a_max` is 1.000 and its gain +0.000 **by construction** (it starts at perfect agreement
with its own argmax), exactly as in the source battery; the `KL@gain+0.05` column is degenerate there
for the same reason and is printed as `—` rather than interpolated.

### 3.2 Scored prediction 1 — Pareto dominance on the real ingredients

> **P1 (registered):** lr 1e-4 Pareto-dominates lr 3e-4 on every teacher — absorption ceiling not
> lower (≥ ceiling@3e-4 − 0.018, the source battery's measured seed-to-seed `gain@400` bound) AND
> off-slice collateral KL@400 lower.

| arm | ceiling 3e-4 | ceiling 1e-4 | KL@400 3e-4 | KL@400 1e-4 | KL@matched gain 3e-4 | 1e-4 | ceiling not lower | collateral lower | **PARETO** |
|---|---|---|---|---|---|---|---|---|---|
| `a_s1` | 0.725 | **0.743** | 1.219 | **0.651** | 1.003 | 0.646 | YES | YES | **PASS** |
| `a_s2` | 0.707 | **0.739** | 1.176 | **0.664** | 1.218 | 0.811 | YES | YES | **PASS** |
| `b_s1` | 0.745 | **0.775** | 0.804 | **0.557** | 0.604 | 0.303 | YES | YES | **PASS** |
| `b_s2` | 0.742 | **0.765** | 0.838 | **0.507** | 0.572 | 0.339 | YES | YES | **PASS** |
| `c_s1` | 0.722 | **0.752** | 1.046 | **0.634** | 0.960 | 0.587 | YES | YES | **PASS** |
| `c_s2` | 0.729 | **0.756** | 0.870 | **0.613** | 0.976 | 0.109 | YES | YES | **PASS** |

**✅ P1 PASSES, 6 arms of 6.** The ceiling clause was written as a *non-inferiority* test with a
noise allowance; it did not need one — **1e-4 is strictly HIGHER on every arm** (+0.0183 to +0.0325,
mean **+0.0268**). Collateral falls by **30–47%** per arm (mean 0.992 → 0.604, −39.1%), off-slice
self-agreement rises on **every** arm (mean 0.632 → 0.709), and `|ΔV|` more than halves (mean
5.23 → 2.48).

⚠️ **The collateral half is decisive; the ceiling half is real but only just clears its own noise on
one arm.** Seed-to-seed |Δceiling| within a matched lr arm measures **0.0033–0.0183** here (largest:
teacher `a` at 3e-4), and the *smallest* lr effect is teacher `a`'s **+0.0183** — numerically equal
to that largest seed spread. So "the ceiling rises" is supported by its consistency (6/6, same sign,
two seeds) rather than by any single arm's margin, and a one-arm one-seed version of this probe would
not have established it. The collateral drop is 16–26× the seed noise and needs no such hedge.

The last two columns are the fairer comparison and the reason the verdict is not an artifact of
stopping at 400 steps: at **matched absorption gain** (+0.05, i.e. asking each arm what it paid to
buy the *same* behaviour change rather than the same number of steps), 1e-4 is cheaper on **all six
arms** too — this is the `collateral_lower_at_matched_gain` field, `true` everywhere.

### 3.3 Scored prediction 2 — the overshoot account

> **P2 (registered):** the zero-content control at 3e-4 shows collateral comparable to the
> with-content cells; at 1e-4 it shrinks by ≥40%.

"Comparable" was operationalized before the run as **the control carrying ≥60% of the with-content
collateral** (the 2026-08-28 lineage battery measured 79% on its 25M student). The control is scored
against the with-content cells at the **same seed and same lr**.

| seed | ctrl KL@400 3e-4 | mean with-content KL@400 3e-4 | **overshoot share** | ctrl KL@400 1e-4 | **shrink** | comparable (≥60%) | shrink ≥40% |
|---|---|---|---|---|---|---|---|
| 1 | 0.615 | 1.023 | **60.1%** | 0.250 | **59.4%** | YES | YES |
| 2 | 0.600 | 0.961 | **62.4%** | 0.244 | **59.4%** | YES | YES |

**✅ P2 PASSES on both clauses, on both seeds — but the first clause passes NARROWLY and the
direction of the miss is worth recording.** The overshoot share here is **60.1% / 62.4%**, against
79% in the lineage battery: it clears the pre-declared 60% bar by 0.1 and 2.4 points. Read honestly,
that is a *weaker* overshoot account on these ingredients than on the rev-1 lineage — a larger
fraction of this fold's collateral is genuine teacher content. Had the threshold been declared at
65% it would have failed. The second clause is not marginal: the control's collateral falls
**59.4%** on both seeds against a 40% bar, and its `|ΔV|` falls 5.42 → 1.77 and 4.50 → 1.67.

The control also loses **17.4pp / 19.5pp of agreement with its own on-slice argmax while training on
exactly those labels** (1.000 → 0.826 / 0.805 at 3e-4), reproducing the source battery's 17.6pp
signature on a different student, a different lineage and a different teacher set.

### 3.4 The content-minus-overshoot NET — the number the fold argv actually turns on

Subtracting the lr-matched and seed-matched control from each with-content cell separates *what the
teacher taught* from *what the optimizer broke*:

| arm | gain@400 | KL@400 | ctrl KL@400 | **NET KL (content)** | \|dV\| | ctrl \|dV\| |
|---|---|---|---|---|---|---|
| `a_3e4_s1` | +0.094 | 1.219 | 0.615 | 0.604 | 4.83 | 5.42 |
| `a_1e4_s1` | +0.132 | 0.651 | 0.250 | 0.402 | 2.31 | 1.77 |
| `a_3e4_s2` | +0.090 | 1.176 | 0.600 | 0.576 | 5.14 | 4.50 |
| `a_1e4_s2` | +0.135 | 0.664 | 0.244 | 0.420 | 2.54 | 1.67 |
| `b_3e4_s1` | +0.061 | 0.804 | 0.615 | 0.189 | 4.75 | 5.42 |
| `b_1e4_s1` | +0.073 | 0.557 | 0.250 | 0.307 | 2.84 | 1.77 |
| `b_3e4_s2` | +0.059 | 0.838 | 0.600 | 0.238 | 5.27 | 4.50 |
| `b_1e4_s2` | +0.068 | 0.507 | 0.244 | 0.263 | 2.09 | 1.67 |
| `c_3e4_s1` | +0.115 | 1.046 | 0.615 | 0.431 | 6.77 | 5.42 |
| `c_1e4_s1` | +0.153 | 0.634 | 0.250 | 0.384 | 2.45 | 1.77 |
| `c_3e4_s2` | +0.136 | 0.870 | 0.600 | 0.270 | 4.60 | 4.50 |
| `c_1e4_s2` | +0.166 | 0.613 | 0.244 | 0.369 | 2.65 | 1.67 |

Averaged over the twelve with-content cells:

| lr | mean gain@400 | mean TOTAL KL | mean NET (content) KL | **gain per unit TOTAL collateral** | gain per unit NET |
|---|---|---|---|---|---|
| 3e-4 | +0.0925 | 0.9920 | 0.3847 | **0.0932** | 0.2404 |
| **1e-4** | **+0.1210** | **0.6042** | 0.3574 | **0.2002** | **0.3384** |

**This is the sharpest result in the battery and it is what the licensing verdict rests on.** The
**NET content** term is essentially unchanged between the two step sizes (0.385 vs 0.357 — 1e-4 if
anything transfers *slightly less* raw content divergence) while absorbing **31% MORE** teacher
behaviour (+0.0925 → +0.1210). Nearly the entire collateral saving is the **content-free overshoot**
component (0.607 → 0.247). Per unit of total damage inflicted, the smaller step buys **2.15× the
absorbed behaviour**.

⚠️ **The decomposition assumes the two damage sources are additive in KL, and they are not
guaranteed to be.** The clearest evidence is `b`, where the NET reads *higher* at 1e-4 (0.307/0.263)
than at 3e-4 (0.189/0.238) — subtracting a large control from a large total leaves a small, noisy
residual, and `b` is the teacher with the smallest absorption gain to begin with. The **totals and
the ceilings are the measured quantities**; the NET column is a decomposition and should be read as
an account of *where the saving comes from*, not as a per-arm score. P1 and P2 are both scored on
measured quantities alone.

### 3.5 Robustness to the parent choice — the alternate-student arm

The revolution fold has not launched, so *which* checkpoint it forks is not yet a recorded fact.
The arm table's convention says the common base (`ai_v9_59`, the primary student here); the other
live possibility is the **rev-4 fold's own output**, `ai_v9_76_R4ACTION_0830/final_model.zip`
(written 18:03 on 2026-08-30, i.e. mid-battery). Two extra cells test the verdict against that
choice, scored separately and never pooled into P1/P2:

| arm | ceiling 3e-4 | ceiling 1e-4 | KL@400 3e-4 | KL@400 1e-4 | off-agree@400 | \|dV\|@400 | PARETO |
|---|---|---|---|---|---|---|---|
| `ai_v9_76` × teacher `a`, seed 1 | 0.744 | **0.757** | 0.911 | **0.575** | 0.641 → **0.714** | 5.08 → **2.24** | **PASS** |

Same direction, same magnitude (ceiling +0.013, collateral −37%, `|ΔV|` −56%). **The licensing
verdict does not depend on which of the two candidate parents the revolution forks** — one teacher,
one seed, so this is a robustness spot-check, not a second battery.

Note the `ai_v9_76` student's `a0` against teacher `a` is higher than `ai_v9_59`'s (it has already
absorbed these teachers once in the rev-4 fold), and its ceiling is the highest measured on teacher
`a` in this battery — consistent with the source battery's secondary finding that **a fold does not
consume distillability**.

## 4. Reading — what this does and does not license

**It licenses the argv change.** On the real parent, against all three real teachers, at both seeds,
lowering the distill-term step size from 3e-4 to 1e-4 moved the student to a strictly better place on
both axes with nothing traded, and the mechanism is the one the 2026-08-28 battery predicted: the
fold at 3e-4 is running **above this student's damage threshold**, and ~60% of what it costs is
Adam overshooting a sharpened landscape rather than the teacher's content being rejected.

**The one-line verdict the fold argv decision quotes:**

> **LICENSED — lower the distill-term step size to 1e-4.** On the actual rev-4 ingredients
> (`ai_v9_59` parent × the three `R4S3{a,b,c}` teachers, 2 seeds), lr 1e-4 Pareto-dominates 3e-4 on
> **6 of 6 arms**: off-slice collateral KL falls **39.1%** (0.992 → 0.604), off-slice self-agreement
> rises 0.632 → 0.709, `|ΔV|` more than halves 5.23 → 2.48, and the absorption ceiling is **strictly
> higher on every arm** (mean +0.027, smallest +0.018 — that one at the size of the seed noise) —
> and 1e-4 is cheaper at **matched absorption gain** too. ~60% of the 3e-4 collateral is content-free
> Adam overshoot (zero-content control 0.615 vs 1.023 with content), of which the smaller step
> removes **59%**, while the net teacher content transferred is unchanged.

**It does NOT license four things, and the fourth is the one that could mislead.**

1. **This is not a fold simulation.** No PPO loss beside the distill term, no `--distill-team-bias`
   sampling, no environment interaction, no entropy/advantage pressure pulling the policy back, no lr
   schedule. It measures the student-side capacity term in isolation; a real fold's outcome is that
   term times everything removed here. Caveat 1 of the source battery, unchanged.
2. **The step size is not the coefficient.** The live fold carries `--distill-coef 0.1761` at
   `--lr 3e-4`. This probe varied the **optimizer step on the distill term alone**, with no other
   loss present. Lowering the *coefficient* at a fixed lr is a related but **not identical**
   intervention (it rescales the distill gradient against the PPO gradient rather than shrinking the
   Adam step; Adam's first step is `lr · sign(g)` elementwise and is therefore substantially
   invariant to a uniform gradient rescale — which is precisely why the source battery's §5.3
   attributes the damage to step size rather than gradient magnitude). **A trust-region or
   coefficient change is NOT what was measured here.**
3. **The lr that trains the PPO half is not obviously the lr that should train the distill half.**
   Dropping the run's global `--lr` to 1e-4 would also slow the RL objective, which this probe says
   nothing about. What is licensed is a smaller effective step **on the distill term**.
4. **400 steps at 3e-4 over-trains, and `KL@400` is an over-training endpoint.** The source battery
   flagged this (its self-distill cell ends 17pp less like itself than it started) and it reproduces
   here. The verdict does not rest on it: the **matched-absorption-gain** columns cross at steps
   **1–49**, inside the informative regime the source battery identifies (the first ~64 steps), and
   agree with the endpoint on all six arms.

**One thing this probe strengthens beyond its brief.** The rev-3 SELF-FOLD result (`ac40230`) showed
self-distillation is actively destructive at production scale; this measures the same harm channel
in the micro-instrument on the fold's real parent and finds it accounts for ~60% of the fold's
collateral. Since exploitability has read **flat across three revolutions** and *fold-harm offset* is
one of the two live candidate accounts for that flatness (`741d9e3`), a lever that removes 59% of the
harm channel at no absorption cost is aimed directly at one of the two standing hypotheses. It does
not test that hypothesis — a flat-exploitability claim needs the fold run, not the probe.

## 5. MISSING cells — never interpolated

| cell | why |
|---|---|
| `KL@gain+0.05` for the **four control cells** | degenerate by construction — step-0 agreement is 1.000, so there is no gain to reach. Same treatment as 2026-08-28. |
| the content control on **teachers b and c** | not run (budget). The control is teacher-**a** states at both lrs × 2 seeds. P2's "comparable" ratio is therefore control-on-`a`-states against the mean of all three teachers' with-content cells; a per-teacher control would be the stronger form. |
| **seeds beyond 2** | not run (budget). Every claim is 2 seeds. |
| a **coefficient** arm (`--distill-coef` varied at fixed lr) | out of scope by construction — see caveat 2 above. This is the single most valuable follow-up, because the coefficient is the knob the fold argv can turn without touching the PPO half. |
| anything about a **full PPO-context fold** | out of scope by construction — caveat 1. |

## 6. Reproducing

```bash
export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd designs/research_state/measurements
nice -n 15 python lr_licensing_probe.py build-states
nice -n 15 python lr_licensing_probe.py probe a_1e4_s1  a   1 400 1e-4
nice -n 15 python lr_licensing_probe.py probe ctrl_a_3e4_s1 'a*' 1 400 3e-4   # QUOTE the control token
nice -n 15 python lr_licensing_probe.py aggregate
nice -n 15 python lr_licensing_probe.py report            # emits this file's tables
```

`<teacher_set>` is `a` / `b` / `c`; `a*` / `b*` / `c*` selects the CONTENT CONTROL on that set's
states. **Quote the control token** — bare `a*` is glob-expanded by the shell against the working
directory and the cell dies on argv parsing; that is not hypothetical, it killed all four control
cells on this battery's first run. The student is the fold parent and is not an argument;
`$GEN3_LR_PROBE_STUDENT` overrides it for an alternate candidate parent, and such cells are reported
separately under `alternate_student_arms` and never pooled into P1/P2. One cell is ~8–11 min of one
CPU core. The producer writes `lr_states_{a,b,c}.npz`, `lr_teacher_targets_{a,b,c}.npz`,
`lr_state_provenance.json`, `lr_results/<cell>.json` and the dated `.json` into its own directory;
it never writes to `models/`. The `.npz` caches and per-cell results are regenerable and
uncommitted, following the 2026-08-28 convention — the committed `.json` carries every curve.
