# GRAD-PROJECT OFFLINE — the projection on the REAL rev-4 ingredients

**Producer `grad_project_offline_probe.py` (this directory) · data
`grad_project_offline_2026-09-01.json` (every cell, every curve, all three predictions scored) ·
inputs `grad_project_offline_2026-09-01_inputs/` · per-cell curves `gp_results/`.**

**Verdict: NOT LICENSED as it ships.** At the default `m = 16` the projection removes **15%** of the
distillation gradient's energy — not the 80% its own smoke reported — and buys **nothing** with it:
the absorption ceiling is unchanged (|Δ| ≤ 0.0075 on 6/6 arms) *and* off-slice collateral does not
fall (`KL@400` **4–9% higher** on 6/6; at matched absorption **−15.4% to +5.2%**, mean −2.3%,
against a registered −30% bar), at **2.3–2.5× the compute**. **P1 PASS · P2 FAIL · P3 PASS.**
The ledger's open question — did the projection remove the leak or the teaching? — has a third
answer: **neither**.

_This file was written in two passes: §0 was committed to disk before the battery produced any
cell; everything after it was appended once the cells landed._

---

## 0. PRE-REGISTRATION — written before any battery number existed

**Registered `2026-09-02T00:04:55-07:00`** (worktree
`.claude/worktrees/agent-a622bdf36f9462dcc`, HEAD `cb6c98de`), immediately after the harness
smoke and immediately before the 25-cell battery launched at `00:04:45`.

### The question

`distill_grad_project.py` landed on 2026-09-01 as `2e99e6fe` with a ledger entry that stated its
own blind spot in as many words: its smoke read `proj_removed_frac` ≈ 0.75–0.89 on a `--debug` toy
config, **and that smoke could not tell "the projection removed the leak" from "the projection
removed the teaching"** — it carries no absorption meter at all. If ~80% of the distill gradient's
energy really lies in the span of the off-slice behaviour directions, then either the teaching and
the leak share their dominant directions (in which case a first-order projection cannot separate
them and the method has a hard ceiling), or the 0.80 is an artefact of a toy configuration and the
production number is much smaller.

Three registered questions:

* **Q1** — what fraction of the distillation gradient's energy lies in the span of the off-slice
  `∇log π(a*|s)` directions (`removed_frac`), as a function of `m` (8, 16, 32, 64) and of training
  step (early vs late in the 400)?
* **Q2** — with the projection applied every step, what is the absorption-vs-collateral Pareto
  curve against the unprojected run at the same lr, and **at matched absorption** is collateral
  lower?
* **Q3** — does the projection reduce absorption by LESS than it reduces collateral (**separable**
  — the teaching and the leak live in different directions) or by about the same (**not separable
  at first order** — the method's ceiling)?

### The predictions

| id | statement | scoring rule, fixed in advance |
|---|---|---|
| **P1** | `removed_frac` at `m = 16` on production gradients is **below 0.5** (i.e. the smoke's 0.80 was a toy-config artefact) | PASS iff every measured per-cell mean `removed_frac` at m=16 — monitor cells and projected arms alike — is < 0.5 |
| **P2** | at **matched absorption 0.70**, projected collateral is **at least 30% lower** than unprojected | PASS iff, on every m=16 arm, both arms reach held-out absorption 0.70 and `KL@0.70` falls ≥ 30%. An arm whose projected run never reaches 0.70 scores **FAIL(ceiling not reached)**, never "unscoreable" |
| **P3** | the absorption ceiling under projection is **within 0.05** of unprojected | PASS iff `abs(a_max_proj − a_max_unproj) ≤ 0.05` on every m=16 arm |

Scored against intervals over the **two seeds** the licensing probe used, on teachers **a and b at
minimum, all three if budget allows**. A prediction is scored on the m=16 arms only; the m sweep
answers Q1/Q3's trend and is not a second scoring of P1–P3.

🚨 **The scored set is gated on lr AND m in code, and the first draft of that gate was wrong.**
`aggregate()` originally selected scored arms by `proj_m == 16` alone, so when the lr-3e-4
robustness cell (§3.3) landed it was silently pooled into P1, P2 and P3 as a **seventh arm at the
wrong step size** — a pre-registered sample that grows itself when a side-arm arrives is not a
pre-registration. Fixed with `SCORED_LR = 1e-4` and a comment naming the defect; the m=8/32 sweep
cells were never eligible and are unaffected. **The verdicts are identical either way** (the extra
arm read −19.3% on P2 and −0.022 on P3, i.e. FAIL and PASS respectively) — but the per-arm rows and
the intervals were wrong for one aggregation, and this note is here because the next reader of this
directory should know the gate exists and why.

### ⚠️ One honesty note that belongs in the pre-registration, not below it

**P1's direction was already visible before the battery launched.** A 2-step harness smoke on
teacher `a` (run at `2026-09-01 23:5x`, to size the battery's cost) printed `removed_frac` at
m=16 of **0.1217 and 0.1282** — so P1's PASS was foreseeable from a sample of two steps on one
teacher. The prediction is kept exactly as the dispatch registered it and scored as written,
because deleting a prediction that the pilot already answered would misrepresent what was known
when; but a reader should treat P1 as **confirmed and quantified** by this battery rather than
**tested** by it. P2 and P3 had no pilot data of any kind — the smoke has no absorption meter,
which is the entire reason this probe exists.

### The instrument, and the one deviation

`lr_licensing_probe.py` (2026-08-31, the ADMITTED instrument the LR licensing verdict was written
on) is **imported, never edited**: its `load_policy` / `masked_logits` / `eval_probs` /
`eval_values` / `kl_rows` / `eval_points` / `crossing`, its student and teacher identities, and its
**committed** `lr_states_{a,b,c}.npz` + `lr_teacher_targets_{a,b,c}.npz` caches are used verbatim
(sha256 in `_inputs/inputs_manifest.json`). Full-policy Adam on masked cross-entropy to the
teacher's argmax, batch 256, 400 steps, 14 log-spaced eval points, seeded batch sequence shared
across arms at a given seed.

The projection is `agents.training.instrumented_ppo.distill_grad_project`'s own
`behaviour_constraints` / `orthonormalize` / `project_out` / `flatten_grads`, imported and called.
**No change to that module was needed** — those four are already pure module-level functions with
no dependence on the PPO object, so the "factor the pure parts out" contingency in the dispatch did
not apply and no commit was made against it.

**THE ONE DELIBERATE DEVIATION.** The licensing probe's 1,500-row OFF-SLICE pool is **SPLIT**, by a
fixed permutation (`seed 20260901`) shared by every arm, into

| pool | rows | role |
|---|---|---|
| CONSTRAINT | 500 | the projection samples its `m` constraint states from here, fresh each step |
| EVALUATION | 1,000 | collateral (masked `KL(now‖original)`, top-1 agreement, mean \|ΔV\|) is measured **only** here |

so **no state ever both constrains the update and scores it**. The split is at the **STATE** level,
not the team level, and that is deliberate: production draws its constraint rows from the same
off-slice distribution the collateral meter reads, so a team-level split would measure a different
operator than the one that shipped. Teams therefore overlap between the two pools, and that is
stated rather than hidden. Consequence: collateral here is read on 1,000 rows against the licensing
record's 1,500, so **absolute KL values in this file are not numerically interchangeable with that
record's** — every arm shares the identical pool, so the comparisons within this file are matched.

### The ingredients (unchanged from the licensing probe)

| role | run |
|---|---|
| **STUDENT (fold parent)** | `ai_v9_59_R2ACTION_0827/final_model.zip` |
| **TEACHER a / b / c** | `ai_v9_73_R4S3a_0829` / `ai_v9_74_R4S3b_0829` / `ai_v9_75_R4S3c_0829` |
| **OFF-slice state source** | `ai_v9_29_rev1_0823` (281 teams / 12 eval steps) |

3,147,887 trainable parameters. A uniformly random gradient's projection onto a random
16-dimensional subspace of that space would retain ≈ 5 × 10⁻⁶ of its energy, which is the number
every `removed_frac` below should be read against.

---

## 1. What was run

| | |
|---|---|
| **core battery** (§2) | 12 cells — {unprojected, projected m=16} × {teacher a, b, c} × {seed 1, 2}, all at lr 1e-4, 400 steps. **These twelve alone score P1–P3.** |
| **Q1 monitor** (§3.1) | 3 cells — teachers a/b/c at seed 1, trained UNPROJECTED, computing `removed_frac` at m ∈ {8,16,32,64} at each eval point and discarding it |
| **m sweep** (§3.2) | 4 cells — teacher `a`, both seeds, at m=8 and m=32. The trend; never a second scoring of P1–P3 |
| **production-lr check** (§3.3) | 2 cells — teacher `a`, seed 1, unprojected + m=16 at **lr 3e-4**, the rate the live fold runs |
| **cost** | **6.33 CPU-hours over all 21 cells**; **2.97** of them the core 12 (unprojected 519–530 s/cell, projected m=16 **1193–1324 s/cell** — the projection is **2.3–2.5×** the step; m=32 is **~2800 s/cell**) |
| **how** | 3 `nice 15` single-thread processes, `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`, no GPU, `models/` read-only |
| **failures** | zero; `dropped_kwargs` empty on every load |

Measured beside the live fleet at load 15–28 on 16 cores. **Wall-clock is reported for budgeting only
and is never read as a result** — every cell is deterministic offline compute on committed cached
states, and a projected arm shares its minibatch sequence with its unprojected twin exactly (the
projector draws its constraint rows from its own generator, never the batch stream).

**The instrument reproduces the record it was cloned from.** The six unprojected cells here are the
licensing probe's lr-1e-4 cells re-run on the 1,000-row evaluation half of the off pool, and they
land on top of it:

| arm | a0 | ceiling here | ceiling (2026-08-31) | KL@400 here | KL@400 (2026-08-31) |
|---|---|---|---|---|---|
| `a_s1` | 0.602 | 0.7433 | 0.743 | 0.656 | 0.651 |
| `b_s1` | 0.681 | 0.7750 | 0.775 | 0.558 | 0.557 |
| `c_s1` | 0.585 | 0.7517 | 0.752 | 0.639 | 0.634 |

so the pool split cost nothing in fidelity, and any difference below is the projection's.

---

## 2. THE RESULT — the projection changes neither side of the trade

**The projection removes 13–18% of the distillation gradient's energy and moves NOTHING: the
absorption ceiling is unchanged to within ±0.008 on every arm, and off-slice collateral does not
fall — at the 400-step endpoint it is 4–9% HIGHER on all six arms, and at matched absorption its
effect straddles zero.** It costs 2.3–2.5× the compute to do this.

| teacher | seed | ceil unproj | ceil proj | Δceil | KL@400 unproj | KL@400 proj | change | KL@abs 0.70 unproj | proj | change | removed_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | 1 | 0.743 | 0.751 | **+0.008** | 0.656 | 0.703 | **+7.1% worse** | 0.627 | 0.635 | +1.2% worse | 0.137 |
| a | 2 | 0.739 | 0.736 | −0.003 | 0.664 | 0.715 | +7.7% worse | 0.604 | 0.697 | +15.4% worse | 0.135 |
| b | 1 | 0.775 | 0.767 | −0.008 | 0.558 | 0.608 | +9.0% worse | 0.304 | 0.288 | **5.2% better** | 0.172 |
| b | 2 | 0.765 | 0.757 | −0.008 | 0.511 | 0.548 | +7.3% worse | 0.276 | 0.265 | **3.8% better** | 0.153 |
| c | 1 | 0.752 | 0.750 | −0.002 | 0.639 | 0.665 | +4.0% worse | 0.469 | 0.490 | +4.3% worse | 0.146 |
| c | 2 | 0.756 | 0.763 | +0.007 | 0.617 | 0.655 | +6.2% worse | 0.479 | 0.490 | +2.2% worse | 0.179 |
| **mean** | | | | **−0.001** | 0.607 | 0.649 | **+6.9% worse** | | | **+2.3% worse** | **0.154** |
| **range over arms** | | | | −0.008 … +0.008 | | | +4.0 … +9.0% | | | −5.2 … +15.4% | 0.135 … 0.179 |

**Read the two collateral columns differently, because they are not equally strong.**

* **KL@400 is consistent and it is in the WRONG direction.** All six arms, both seeds, all three
  teachers: the projected arm ends *further* from the parent than the unprojected one, by 4.0–9.0%.
  Six of six with the same sign is not noise. But `KL@400` is an **over-training endpoint** — the
  licensing record flagged that 400 steps over-trains this instrument — so this is best read as "the
  projection does not reduce accumulated displacement", not as "the projection is actively harmful
  by 6.9%".
* **KL at MATCHED ABSORPTION 0.70 — the fairer comparison, and the one P2 was written on —
  STRADDLES ZERO and its sign is a property of the TEACHER, not of noise.** Teacher `b` improves on
  both seeds (+5.2%, +3.8%); teachers `a` and `c` worsen on both (−1.2%/−15.4%, −4.3%/−2.2%). Mean
  −2.3%, range −15.4% to +5.2%. Against a registered bar of −30%, this is not a near miss; it is a
  different phenomenon.

### 2.1 The registered predictions

| id | statement | result | evidence |
|---|---|---|---|
| **P1** | `removed_frac` at m=16 on production gradients is **below 0.5** | ✅ **PASS**, 6/6 | max observed **0.1793**; range 0.1345–0.1793. The `--debug` smoke's 0.75–0.89 was a **toy-config artefact** |
| **P2** | at matched absorption 0.70, projected collateral is **≥30% lower** | ❌ **FAIL**, 0/6 | per-arm change −15.4% … +5.2%, mean **−2.3%**; both arms reach 0.70 on every cell, so this is a real reading and not a coverage gap |
| **P3** | the absorption ceiling under projection is **within 0.05** | ✅ **PASS**, 6/6 | \|Δceiling\| ≤ **0.0075** everywhere; mean −0.0008 |

**P1 and P3 pass and P2 fails, and that combination is the finding.** The method's registered
failure mode was that it would buy collateral by paying absorption. It does not pay absorption — and
it does not buy collateral either.

### 2.2 Q3 — the separability question, answered by a third option the question did not offer

The registered dichotomy was **separable** (absorption falls less than collateral) or **not
separable at first order** (both fall about the same — the ceiling). Neither happened:

| teacher | seed | absorption-gain change | collateral (KL@400) change |
|---|---|---|---|
| a | 1 | +5.3% MORE absorbed | 7.1% MORE collateral |
| a | 2 | 2.4% less | 7.7% more |
| b | 1 | 8.0% less | 9.0% more |
| b | 2 | 8.9% less | 7.3% more |
| c | 1 | 1.0% less | 4.0% more |
| c | 2 | +4.4% MORE absorbed | 6.2% more |
| **mean** | | **1.8% less (straddles zero)** | **6.9% more** |

🚨 **The "collateral removed per unit teaching removed" RATIO is NOT interpretable on this data and
must not be quoted.** Its denominator — the absorption-gain change — straddles zero across arms
(−5.3% to +8.9%), so the ratio divides by a quantity indistinguishable from nothing and flips sign
arm to arm (the computed values run 1.34×, −3.18×, −1.12×, −0.82×, −4.05×, 1.41× — six numbers that
say nothing). The `.json` carries `Q3_separability.ratio_interpretable: false` and a note saying
exactly this, so the artefact refuses the misreading rather than inviting it. **Read the two
columns separately.**

The honest answer to Q3 is therefore: **at m=16 the projection is a near-no-op on both axes.** It
does not separate the teaching from the leak, and it does not fail by removing both — it removes
neither, having removed 15% of the gradient's energy to do so.

---

## 3. Q1 — how much of the distillation gradient lies in the off-slice behaviour subspace

**At m=16 the answer is ~0.15, not ~0.80.** Every projected arm logged `removed_frac` at every one
of its 400 steps; `proj_rank` was **16.0 of 16 on every step of every arm**, so the sampled
constraint directions were fully independent and nothing was lost to Gram–Schmidt drops.

| arm | mean | first 32 steps | last 32 steps | min | max |
|---|---|---|---|---|---|
| `p16_a_s1` | 0.1370 | 0.1123 | 0.1305 | 0.0228 | 0.6023 |
| `p16_a_s2` | 0.1345 | 0.1567 | 0.1686 | 0.0201 | 0.4139 |
| `p16_b_s1` | 0.1722 | 0.1284 | 0.1931 | 0.0276 | 0.4757 |
| `p16_b_s2` | 0.1532 | 0.1351 | 0.1928 | 0.0198 | 0.4927 |
| `p16_c_s1` | 0.1459 | 0.1493 | 0.1332 | 0.0281 | 0.5979 |
| `p16_c_s2` | 0.1793 | 0.1342 | 0.3159 | 0.0271 | 0.7042 |

**Two things this table settles.**

1. **The 0.80 was configuration, not physics.** The shipped smoke ran `--debug --n-steps 512
   --batch-size 128`; here the same operator on the same 3,147,887-parameter extractor, at the
   instrument's batch 256 and against real teacher argmax targets, reads **0.135–0.179**. Anyone
   reasoning from the module's "at m=16 most of the teacher term's magnitude goes with the leak"
   line is reasoning from a toy.
2. **It is still four orders of magnitude above chance, and it drifts UP with training.** A
   uniformly random gradient would keep ≈ 5 × 10⁻⁶ of its energy in a random 16-dimensional
   subspace of a 3.1M-dimensional space; 0.15 is ~30,000× that. So the shared-direction *mechanism*
   the module describes is real and directly visible at the update — it is just far smaller than the
   smoke implied. Five of six arms read higher in their last 32 steps than their first 32
   (`c_s2` most sharply, 0.134 → 0.316), which is what "the taught content progressively occupies
   the directions that also move untaught boards" looks like. The per-step maxima (0.41–0.70) show
   the alignment is spiky, not steady.


### 3.1 removed_frac vs m — it grows SUBLINEARLY and never saturates

Three `monitor` cells (teachers a/b/c, seed 1) train **unprojected** and, at each of the 14 eval
points, compute `removed_frac` for every `m` on that step's real gradient and throw the result
away. So this is the m-table read along the trajectory the fold actually takes, with no
projection feeding back into it.

| m | mean | early (step ≤ 32) | late (step ≥ 135) | rank | per-cell (a / b / c) | ×  vs previous m |
|---|---|---|---|---|---|---|
| 8 | **0.0910** | 0.0950 | 0.0935 | **8.0 / 8** | 0.0895 / 0.0794 / 0.1041 | — |
| 16 | **0.1497** | 0.1442 | 0.1826 | **16.0 / 16** | 0.1375 / 0.1474 / 0.1642 | 1.65× |
| 32 | **0.2227** | 0.2242 | 0.2389 | **32.0 / 32** | 0.2139 / 0.2072 / 0.2469 | 1.49× |
| 64 | **0.3153** | 0.3225 | 0.3272 | **64.0 / 64** | 0.3004 / 0.3122 / 0.3332 | 1.42× |

**Three facts, and each one is load-bearing.**

1. **The rank is FULL at every m — 8/8, 16/16, 32/32, 64/64, on every cell.** Not one constraint
   direction was ever dropped as already-spanned by `orthonormalize`'s `GS_REL_TOL`. So the
   off-slice behaviour subspace is at least 64-dimensional at production shape, and there is no
   small set of shared directions in which "the leak" lives. This is the fact that makes the
   sublinear scaling below a structural statement rather than a sampling artefact.
2. **Doubling m multiplies `removed_frac` by ~1.5, not by 2** (1.65× / 1.49× / 1.42×, decaying).
   A power-law fit over the four points gives **`removed_frac ≈ 0.0275 · m^0.595`** (residuals
   ≤ 0.011 in absolute `removed_frac` — a good fit on four points, quoted as a *description of
   this range*, not as a law).
3. **The drift with training is mild and present only at m ≥ 16** — m=8 is flat (0.0950 → 0.0935)
   while m=16/32/64 rise (0.144 → 0.183, 0.224 → 0.239, 0.323 → 0.327). The larger the constraint
   battery, the more visible the "taught content progressively occupies directions that also move
   untaught boards" effect is.

**What the fit says about ever reaching the smoke's number.** Extrapolating (a real extrapolation,
1.5–4.5× beyond the measured range, and labelled as one): **`removed_frac = 0.5` needs m ≈ 131 and
`removed_frac = 0.8` needs m ≈ 288** — 8× and 18× the shipped default. The Gram–Schmidt is
**O(m²·|θ|)**, so m=288 is ~324× the projection cost of m=16, against a per-step projection that
already costs 1.5–1.9 s at m=16 (55–70% of a production `train()` by the module's own measurement).
**The 0.80 regime is not reachable at any price worth paying**, which is the second, independent
reason the `--debug` smoke's figure cannot be read as the production one.

⚠️ **This table does NOT say a larger m would work.** It says how much of the gradient a larger m
would REMOVE. Whether removing more helps is the m=8/32 trajectory question in §3.2 — and the
m=16 result is that removing 15% changed neither absorption nor collateral, so "remove more" is a
hypothesis this record does not endorse.

### 3.2 The m sweep — removing MORE does not start working

The natural objection to §2 is "m=16 removes only 15%; remove more." Teacher `a`, both seeds, at
m=8 (and m=32, below) tests it.

| m | seed | ceil unproj | ceil proj | Δceil | KL@400 unproj | KL@400 proj | change | removed_frac |
|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 0.7433 | 0.7492 | +0.0058 | 0.656 | 0.691 | +5.3% worse | 0.0914 |
| 8 | 2 | 0.7392 | 0.7450 | +0.0058 | 0.664 | 0.704 | +6.0% worse | 0.0829 |
| 16 | 1 | 0.7433 | 0.7508 | +0.0075 | 0.656 | 0.703 | +7.1% worse | 0.1370 |
| 16 | 2 | 0.7392 | 0.7358 | −0.0033 | 0.664 | 0.715 | +7.7% worse | 0.1345 |
| 32 | 1 | 0.7433 | 0.7408 | −0.0025 | 0.656 | 0.693 | +5.6% worse | 0.1920 |
| 32 | 2 | 0.7392 | 0.7458 | +0.0067 | 0.664 | 0.704 | +6.0% worse | 0.1928 |

**Across a 4× range in m — 8, 16, 32 — nothing changes.** `removed_frac` more than doubles
(0.091 → 0.137 → 0.192, `proj_rank` full at every width), and on all six arms the ceiling still
moves by **less than 0.008** and collateral is still **5–7% worse**:

| m | mean removed_frac | mean Δceiling | mean KL@400 change | projection cost / step |
|---|---|---|---|---|
| 8 | 0.087 | +0.0058 | +5.7% worse | ~0.55 s |
| 16 | 0.136 | +0.0021 | +7.4% worse | ~1.8 s |
| 32 | 0.192 | +0.0021 | +5.8% worse | **~5.9 s** |

Between m=8 and m=32 the operator removes **2.2× more** of the gradient, costs **11× more per
step**, and the measured consequence is **indistinguishable on both axes**. That is the shape of a
lever with no traction in this range — not of a lever that needs turning further. Combined with
§3.1's extrapolation (`removed_frac = 0.5` needs m ≈ 131 at O(m²) cost), "remove more" has no
affordable version that this data gives any reason to expect would work.

### 3.3 Production lr — the null holds at 3e-4 too, and there it is worse

The live rev-4 fold runs at **lr 3e-4**, not the 1e-4 the 2026-08-31 record licensed. One paired
cell (teacher `a`, seed 1) checks whether the null is an artefact of the smaller step:

| lr | ceil unproj | ceil proj | Δceil | KL@400 unproj | KL@400 proj | change | removed_frac |
|---|---|---|---|---|---|---|---|
| 1e-4 | 0.7433 | 0.7508 | +0.0075 | 0.656 | 0.703 | +7.1% worse | 0.137 |
| **3e-4** | 0.7250 | **0.7025** | **−0.0225** | 1.187 | 1.244 | +4.8% worse | 0.180 |

**It holds, and this is the ONE cell where the projection is unambiguously worse on both axes at
once** — the absorption ceiling falls 0.0225 *and* collateral rises 4.8%. `removed_frac` is also
higher at the larger step (0.180 vs 0.137), consistent with a sharper landscape putting more of the
distill gradient along the off-slice behaviour directions.

⚠️ **n = 1: one teacher, one seed.** It is a spot check that the verdict does not invert at
production lr, not a second battery. The −0.0225 ceiling drop is larger than any m=16 arm at 1e-4
(max |Δ| 0.0075) and would still PASS P3's ±0.05 bar, so it changes no scored prediction; it is
recorded because its DIRECTION is the one a reader would want to know before enabling the flag on a
3e-4 fold.

---

## 4. Reading — what this does and does not license

### 4.1 The one-line verdict

> **NOT LICENSED AS IT SHIPS.** On the actual rev-4 ingredients (`ai_v9_59` parent × the three
> `R4S3{a,b,c}` teachers, 2 seeds, lr 1e-4), `--distill-anchor-mode grad_project` at the default
> `m = 16` **removes 13–18% of the distillation gradient's energy and buys nothing with it**: the
> absorption ceiling is unchanged (|Δ| ≤ 0.0075, 6/6) and off-slice collateral does not fall —
> `KL@400` is **4–9% HIGHER on all six arms** and at matched absorption 0.70 the change is
> **−15.4% to +5.2%, mean −2.3%**, against a registered bar of −30%. It costs **2.3–2.5× the
> compute** to achieve this. The smoke's `proj_removed_frac ≈ 0.80` was a **toy-config artefact**;
> the production figure is **0.154**.

### 4.2 The isolation is what makes the leak claim CLEANLY testable, not what weakens it

The obvious objection is that this harness has **no PPO gradient**, and the module's whole design is
`g_ppo + P⊥ g_distill` — leave PPO's gift alone, remove the distill term's leak. So does removing
PPO invalidate the test?

**No, and the direction matters.** The module defines the LEAK as *the teachers' taught content
arriving on untaught boards through shared weights* — i.e. **the distill term's own off-slice
effect**. That is exactly and only what this harness contains. There is no PPO gradient to confound
the reading, so this is the leak claim measured in the cleanest possible isolation, and the answer
is that the projection does not reduce it.

**What is genuinely untested here is the GIFT half** — that PPO's orthogonal, off-slice habit change
survives the operation. It trivially survives an operation that never touches `g_ppo`, so there was
never much to test; and it is moot while the leak half does not work.

### 4.3 The most likely mechanism, and it is the one the module warned about

`distill_grad_project.py`'s docstring states its own first-order limitation in as many words: the
projection kills the distill term's **instantaneous** effect on the **sampled** off-slice
log-probabilities, and does **not** bound the **accumulated** displacement — the constraint set is
resampled every step and curvature carries the policy off the tangent plane.

This battery measures that gap and finds it total. Per step the operator does exactly what it
claims (it removes, by construction, the entire component of `g_distill` in the span of 16 exact
`∇log π(a*|s)` directions, at `proj_rank` 16.0/16 every step). Over 400 steps that buys **zero**
reduction in the accumulated `KL(now‖original)` on 1,000 *held-out* off-slice states. Two
compounding reasons, neither excluded by this data:

* **Resampling.** 16 constrained directions per step out of a 500-row constraint pool means any
  particular off-slice state is constrained on ~3% of steps and free on the other 97%. A
  first-order block on 3% of steps is not a trust region.
* **Generalization.** The 1,000 evaluation states are disjoint from the 500 constraint states, so
  even a perfectly enforced constraint on the constrained rows only helps the meter to the extent
  the constrained directions **span** the unconstrained ones — and `removed_frac` 0.154 says they
  span about a seventh of the gradient.

That the projected arm ends **further** from the parent (all 6 arms, 4–9%) rather than merely no
closer is consistent with the removal being a perturbation that changes the trajectory without
constraining its destination — but it is a small consistent effect on an over-training endpoint
and this record does not claim a mechanism for it.

### 4.4 What this does NOT license — five things

1. **It is not a fold simulation.** No PPO loss, no `--distill-team-bias` sampling, no environment
   interaction, no entropy/advantage pressure, no lr schedule, 400 steps rather than millions.
   Caveat 1 of the 2026-08-28 instrument, unchanged.
2. **It does not condemn `m > 16`.** The default is m=16 and that is what is measured and named.
   The m sweep is §3.1; a verdict at m=16 is a verdict on the shipped default, not on the operator
   at every width.
3. **The constraint rows here are EVAL-TRACE states, not live rollout states.** Production draws
   its `m` rows from the off-slice rows of the live minibatch. The distributional difference is
   real and unquantified.
4. **The collateral meter is drift-from-parent on eval-trace states, not the ledger's untaught-team
   WIN RATE.** The −5.66pp [−12.1, −0.2] the leak is worth is a game-outcome quantity; `KL@400`
   here is a policy-displacement quantity, and nothing in this record maps one onto the other.
5. **It does not license removing the module.** A negative result at one width, in one
   micro-instrument, without the PPO half, is a reason not to ENABLE the flag — it is not a reason
   to delete a correctly-implemented operator whose per-step behaviour is exactly as specified.

### 4.5 What it DOES settle

* **The `0.80` figure is retired.** Anyone reasoning from "most of the teacher term's magnitude
  goes with the leak" is reasoning from a `--debug` configuration. The production number is
  **0.154**, and the module's docstring should say so.
* **The ledger's open question — "removed the leak" vs "removed the teaching" — has a third
  answer: NEITHER.** The smoke could not distinguish those two because it had no absorption meter;
  with one, the answer is that at m=16 the operator is a near-no-op on both axes.
* **The `distill/kl stayed HIGHER in the projected arm ⇒ less absorbed` reading in the module's
  docstring is NOT reproduced.** Held-out absorption is statistically identical between arms
  (mean Δceiling −0.0008, |Δ| ≤ 0.0075 on 6/6). Whatever that smoke's `distill/kl` was showing, it
  was not a loss of teacher content.

---

## 5. MISSING cells — never interpolated

| cell | status | why |
|---|---|---|
| projected trajectory at **m = 64** | **NOT RUN** | ~2.1 CPU-hours for ONE cell (the Gram–Schmidt is O(m²·\|θ\|); the constraint battery alone measured 18 s/step at m=64 against 1.5 s at m=16). `removed_frac` at m=64 is measured by the monitor cells without paying for a trajectory, and the m=16/32 result decides the question |
| monitor cells at **seed 2** | **NOT RUN** | budget. Q1's m-table is 3 cells at seed 1 |
| the **zero-content control** | **out of scope by construction** | this probe compares PROJECTED against UNPROJECTED at fixed content; the Adam-overshoot decomposition is the 2026-08-31 record's question, not this one |
| **teachers b and c** in the m sweep | **NOT RUN** | budget; the sweep is teacher `a`, both seeds, at m=8 and m=32 |
| **seed 2** of the lr-3e-4 check | **NOT RUN** | budget; §3.3 is n=1 and labels itself a spot check |
| anything about a **full PPO-context fold** | **out of scope by construction** | §4.4 caveat 1 |

---

## 6. Reproducing

```bash
export PYTHONPATH=$PYTHONPATH:src
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd designs/research_state/measurements
# the state caches are COMMITTED (lr_states_{a,b,c}.npz + lr_teacher_targets_{a,b,c}.npz) —
# no build-states step, and the sha256 of each is in _inputs/inputs_manifest.json
python grad_project_offline_probe.py probe none_a_s1 a 1 400 1e-4 none
python grad_project_offline_probe.py probe p16_a_s1  a 1 400 1e-4 proj:16
python grad_project_offline_probe.py probe mon_a_s1  a 1 400 1e-4 monitor
python grad_project_offline_probe.py aggregate     # writes the dated .json, scores P1-P3
python grad_project_offline_probe.py report        # emits this file's tables
```

`<mode>` is `none` | `proj:M` | `monitor`. A cell is ~9 min (unprojected), ~20–22 min (`proj:16`),
~13 min (`monitor`) of one CPU core. The producer writes `gp_results/<cell>.json` and the dated
`.json` into its own directory and **never writes to `models/`**. Per-cell curves are committed
alongside the record, following the 2026-08-31 convention.

