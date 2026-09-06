# Teacher distance — is the fold's untaught delta a dose-response in how far its teachers moved?

**2026-09-05. Offline. No training, no launcher, no server.** ~25 CPU-minutes of battles,
CPU-only, `nice -n 10`, BLAS pinned to 1 thread, beside a live training run (load 17–25 on 16
cores).

[`PREREGISTRATION.md`](PREREGISTRATION.md) was frozen before a single state was generated, a
single teacher was loaded, or a single fold delta was recomputed. Nothing in it is edited after
data.

---

## Verdict

| prediction | result |
|---|---|
| **(i) PRIMARY** — Spearman(`D_off`, untaught delta) negative, CI excludes zero | **SIGNIFICANT at the TEACHER-SET level.** Fold unit `ρ = −0.756`, CI excludes zero under all three bootstraps; point unit `ρ = −0.900`; the clean within-parent contrast `ρ = −0.764` with both CIs excluding zero. But the point-level **SLOPE** under the nested bootstrap is `[−35.5, +0.4] pp per unit KL` — **it spans zero.** The ORDERING is solid; the magnitude is not. |
| **(ii)** `D_off` separates ROBBED from NEUTRAL/GIFTED folds | **FAILS.** The ranges overlap, and they overlap inside one teacher set: N1 (`+1.50`) and rev-4 (`−6.50`) sit at the **identical** `D_off = 0.7715`. **No fold in this table GIFTED at all**, so the "gifted" half of the prediction has nothing to test. |
| **(iii)** `D_on` does not predict once `D_off` is accounted for | **NOT SCORABLE.** `corr(D_off, D_on) = +0.965` over folds, `+0.949` over points. No teacher set is far off-slice and near on-slice, or the reverse, so there is no partial to take. Reported as unscorable, not as a null. |

**The one-sentence reading.** Across **five** distinct teacher sets the fold's off-slice damage
orders almost perfectly with how far those teachers sit from the fold parent — but the axis is
**rank-indistinguishable from plain teacher training budget** (`ρ(budget, delta) = −0.949`, at
least as good as `ρ(D_off, delta) = −0.900`), the effect is a **set-mean** effect that does not
survive down to individual folds (two byte-identical folds at one `D_off` differ by **5.94 pp**),
and 4 of the 5 points share one parent, one fork ancestor and one recipe family. **H5 is
SUPPORTED as an ordering over teacher sets and NOT ESTABLISHED as a causal dose-response.**

### One block, for the ledger

> **H5 (teacher distance) — SUPPORTED as a teacher-set ORDERING, NOT ESTABLISHED as a dose-response,
> CONFOUNDED with teacher budget.** 17 folds, **5** distinct (parent, teacher-set) points, 4 of them
> on one parent. Spearman(`D_off`, untaught Δ): fold unit **−0.756** [BOOT-TEAM −0.821, −0.582;
> BOOT-FOLD −0.932, −0.387; BOOT-BOTH −0.936, −0.257], point unit **−0.900**, within-parent
> **−0.764** (both CIs exclude zero). **The point-level SLOPE spans zero under the nested bootstrap
> ([−35.5, +0.4] pp per unit KL).** Set means: UNF 0.554 → **+0.87** · R3set 0.617 → **−2.50** ·
> FUND 0.696 → **−2.41** · R4set 0.772 → **−4.47** · R2set 0.218 → **+0.88** (cross-parent).
> **(ii) FAILS** — N1 (+1.50) and rev-4 (−6.50) sit at the *identical* `D_off`, and the N1/N2
> replicate draw is **5.94 pp**, the size of the whole between-set effect. **(iii) unscorable** —
> `corr(D_off, D_on) = +0.965`. **Confound not broken by any fold**: `ρ(budget, Δ) = −0.949` ≥
> `ρ(D_off, Δ) = −0.900`, and the two sets sharing a budget (R3set/FUND, 5.07M) share a Δ.
> **New: the INHERITED GAP** `KL(rev-1 final ‖ R2ACTION) = 0.3920` — a gen-era teacher is that far
> from the fold parent before it trains a step, so raw `D_off` is not comparable across a fork
> boundary. **In floor units: v8's teachers 4.5× (its fold GIFTED), our worst 15.0× (robs hardest)
> — a 3.3× gap, the strongest line here.** **Two hazards:** `content_locality` scored the wrong
> checkpoint in BOTH eras (its conclusions strengthen, its levels are wrong — v8 5.2× → 4.5×,
> `semistall3` −0.115); and the 2×2/K=6 per-team artifacts lived ONLY in a session job dir and are
> now banked in the tree — rescued into this probe's `inputs/`, then moved on 2026-09-06 to
> `teacher_content_2x2_2026-09-04/`, beside the readout that reads them (hazard 1).

---

## The fold table

Every delta is **RECOMPUTED** here from the per-team `wins/games` rows of the untaught-8 probe
artifacts and cluster-bootstrapped over the 8 teams (20 000 draws, seed 20260905) — nothing is
copied out of the ledger, so a transcription error cannot enter. Every provenance field is read
from the run's own `metadata.json` (`original_command` / `cli_args`), never typed. Every fold is
scored against **its own parent** on the standing stamp (stochastic · opponent
`ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · team set M · n = 200/team), and the
stamp is re-verified per file with a fatal assert.

`D_off` / `D_on` = mean forward `KL(teacher ‖ parent)` over legal actions, averaged over the
fold's teachers, on untaught / that teacher's own taught teams.

| fold | run | set | parent | nT | taught | `D_off` | `D_on` | untaught Δ pp | 95% CI | coef | gas | fork_lr |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| rev-2 / R2ACTION | `ai_v9_59` | R2set | rev-1 final | 5 | 9 | 0.2175 | 0.2247 | **+0.88** | [−1.33, +3.21] | 0.181 | 2 | — |
| TC_UNF_A | `ai_v9_162` | UNF | R2ACTION | 8 | 16 | 0.5536 | 0.5804 | **+2.00** | [−0.28, +4.56] | 0.1761 | 3 | 2.8e−5 frozen |
| TC_UNF_B | `ai_v9_163` | UNF | R2ACTION | 8 | 16 | 0.5536 | 0.5804 | **+1.94** | [−0.01, +3.67] | 0.1761 | 3 | 2.8e−5 frozen |
| TC_UNF_K6_A | `ai_v9_170` | UNF | R2ACTION | 8 | 16 | 0.5536 | 0.5804 | **+0.37** | [−1.74, +2.53] | 0.1761 | 6 | 2.8e−5 frozen |
| TC_UNF_K6_B | `ai_v9_171` | UNF | R2ACTION | 8 | 16 | 0.5536 | 0.5804 | **−0.81** | [−2.61, +1.22] | 0.1761 | 6 | 2.8e−5 frozen |
| rev-3 / R3ACTION | `ai_v9_70` | R3set | R2ACTION | 6 | 12 | 0.6172 | 0.8663 | **−2.50** | [−4.39, −0.11] | 0.1761 | 2 | — |
| TC_FUND_A | `ai_v9_160` | FUND | R2ACTION | 8 | 16 | 0.6957 | 0.7456 | **−2.50** | [−5.17, −0.22] | 0.1761 | 3 | 2.8e−5 frozen |
| TC_FUND_B | `ai_v9_161` | FUND | R2ACTION | 8 | 16 | 0.6957 | 0.7456 | **−2.31** | [−3.83, −0.26] | 0.1761 | 3 | 2.8e−5 frozen |
| B2 | `ai_v9_140` | R4set | R2ACTION | 3 | 24 | 0.7715 | 0.9022 | **−2.75** | [−5.33, −0.31] | 0.1761 | 2 | — |
| COMPFOLD | `ai_v9_91` | R4set | R2ACTION | 3 | 12 | 0.7715 | 0.9022 | **−3.88** | [−6.60, −1.61] | 0.1761 | 2 | — |
| N1 | `ai_v9_142` | R4set | R2ACTION | 3 | 6 | 0.7715 | 0.9022 | **+1.50** | [−1.00, +4.06] | 0.1761 | 2 | — |
| N2 | `ai_v9_143` | R4set | R2ACTION | 3 | 6 | 0.7715 | 0.9022 | **−4.44** | [−7.00, −2.28] | 0.1761 | 2 | — |
| R4DOSE12 | `ai_v9_150` | R4set | R2ACTION | 3 | 24 | 0.7715 | 0.9022 | **−5.81** | [−7.24, −3.94] | 0.1761 | 12 | 2.8e−5 frozen |
| R4DOSE6 | `ai_v9_151` | R4set | R2ACTION | 3 | 24 | 0.7715 | 0.9022 | **−6.31** | [−9.22, −3.42] | 0.1761 | 6 | 2.8e−5 frozen |
| R4DOSE3 | `ai_v9_152` | R4set | R2ACTION | 3 | 24 | 0.7715 | 0.9022 | **−7.56** | [−11.79, −3.45] | 0.1761 | 3 | 2.8e−5 frozen |
| rev-4 / R4ACTION | `ai_v9_76` | R4set | R2ACTION | 3 | 24 | 0.7715 | 0.9022 | **−6.50** | [−9.39, −3.44] | 0.1761 | 2 | — |
| *C1 — CONTROL* | `ai_v9_141` | R4set | R2ACTION | 3 | 24 | 0.7715 | 0.9022 | *+2.50* | *[−0.00, +5.06]* | **0.0** | 2 | — |
| *v8_14 — CROSS-ERA* | `ai_v8_14` | v8set | `ai_v8_04` | 3 | 22 | **0.2329** | 0.3385 | *+4.64 @ +1.09M* | *era meter* | **1.0** | — | — |

**C1 is excluded from every fit** and carried as a labelled control: its `--distill-coef` is
**0**, so no teacher content is transported at all and `D_off` cannot act. It sits at the table's
**highest** `D_off` (0.7715 — the same three teachers rev-4 distilled) with the table's **most
positive** delta (+2.50). That is not a counterexample to H5; it is the demonstration that
`D_off` matters only through the loss, which is exactly the channel the ledger convicted.

**v8 was to be REUSED, and the reuse turned out to be on the wrong checkpoint** — see the finding
below. `D_off = 0.2329` is a **new measurement** ([`v8_checkpoint_fix.py`](v8_checkpoint_fix.py),
run from the era checkout at `b13b30b2`) on the checkpoint the v8 fold actually loaded; content
locality's reused value was 0.2740. `D_on` and the taught column are still content_locality's
numbers and carry the same defect. v8's delta is `+4.64 pp` at +1.09M and **≈ +8.5 pp at our fold
length (interpolated from one curve — marked, and never fitted)**. It sits on its own parent, its
own untaught 8, a greedy meter and the node bridge, so it is **not on this table's y-scale** and
enters no fit.

⚠️ **v8's distillation COEFFICIENT was `1.0`; every gen-era fold in this table runs at `0.1761`** —
5.7× smaller. That is a covariate no probe in this program has manipulated, and it runs *against*
the naive reading: the era that gifted used the far larger coefficient. It is recorded here, not
explained.

### `D_off` in floor units — the era comparison that *is* legitimate

The matched-noise floor (two arbitrary nearby checkpoints of each era's own parent, same
statistic, same states) is almost identical across eras — gen **0.0374 / 0.0654**, v8 **0.0383 /
0.0664** — so a ratio to it is comparable where the raw level is not:

| set | `D_off` | × its own era's floor |
|---|---:|---:|
| R2set | 0.2175 | **4.2×** |
| **v8set** | **0.2329** | **4.5×** |
| UNF | 0.5536 | 10.8× |
| R3set | 0.6172 | 12.0× |
| FUND | 0.6957 | 13.5× |
| R4set | 0.7715 | **15.0×** |

*(gen floor mean 0.0514 from the two arbitrary R2ACTION checkpoints; v8 floor mean 0.0523 from the
two arbitrary `ai_v8_04` checkpoints — both re-measured here, and both reproduce content_locality
to four decimals.)*

v8's teachers — the only ones whose fold GIFTED — sit at **4.5×** the noise floor. Our worst
robber's teachers sit at **15.0×**, a **3.3× gap**. That is the H5 picture stated in the one
currency the two eras share, and it is the strongest single line in this probe. **The checkpoint
correction moved v8 DOWN** (5.2× → 4.5×), so the era gap is wider than content_locality's numbers
implied, not narrower.

### Folds REJECTED, and why

| rejected | reason |
|---|---|
| **R3SELF** | no untaught cell was ever run (ledger, M9 FINAL: *"R3SELF's untaught cell also unrun"*); only its taught `−8.96 pp` exists |
| **R4PLAIN** | does not exist — the named-and-priced matched plain control for rev-4 was never launched |
| **EXT_A** (K=3 unfunded extension) | still training; no endpoint artifact |
| **R2PLAIN / R2CTRL** | not folds — distillation-free arms. They supply the replicate FLOOR and have no teacher set, so no `D_off` exists for them |
| the **fdA / fdB / fdC / fdE / fdF and G1 / G2 family** (`ai_v9_38/39/40/42/45/48/49`) | **no untaught-8 artifact exists for any of them** — searched the tree and the job directories. They were scored on the TAUGHT side only, and they fork off **rev-1**, so even a future untaught pass would add a sixth parent rather than a within-parent point |
| every other rev-2/rev-3-era "fleet fold" | no untaught-8 artifact **on the standing stamp**. Probe Q's readings are a different team set *and* a greedy meter and may not be quoted beside these (ledger, 2026-09-01 meter-vs-composition entry) |
| **v8_14 in the FIT** | admitted to the table, excluded from every fit — different parent, era, team set, meter and bridge |

---

## `D_off` is a property of a TEACHER SET, not of a fold — and this is the load-bearing caveat

Seventeen folds. **Five** distinct `(parent, teacher-set)` points. **Four** of them share one
parent. Eight folds (rev-4, COMPFOLD, B2, C1, N1, N2, and the three dose arms) distil the
*identical three networks* and therefore carry the *identical* `D_off` — ties on x add no
information about a slope, however many folds carry them. This was registered in advance, and it
is why every statistic below is reported at both units.

```
  +3.5 |
  +2.9 |                                                                                *
  +2.4 |                                                 #                                
  +1.8 |                                                                                m
  +1.2 |  a
  +0.6 |                                                 d
  +0.1 |------------------------------------------------------------------------------------
  -0.5 |                                                 e
  -1.1 |
  -1.7 |
  -2.2 |                                                          f          #          k
  -2.8 |
  -3.4 |                                                                                j
  -4.0 |                                                                                n
  -4.5 |
  -5.1 |
  -5.7 |                                                                                o
  -6.3 |                                                                                #
  -6.8 |
  -7.4 |                                                                                q
  -8.0 |
       +------------------------------------------------------------------------------------
        0.198                                                                        0.791
```
`x = D_off`, `y = untaught delta (pp)`, `#` = two folds in one cell.
`a`=rev-2(R2set) · `b`,`c`=TC_UNF_A/B · `d`,`e`=TC_UNF_K6_A/B (UNF) · `f`=rev-3(R3set) ·
`g`,`h`=TC_FUND_A/B (FUND) · `i`=rev-4 · `j`=COMPFOLD · `k`=B2 · `m`=N1 · `n`=N2 ·
`o`,`p`,`q`=R4DOSE12/6/3 (R4set) · `*`=C1 control.

**Read the right-hand column, not the trend line.** Eight folds stack vertically at
`D_off = 0.7715`, spanning `+2.50` to `−7.56`. The x-axis explains the difference between
*columns*; within the widest column it explains nothing.

---

## The statistics

Three bootstraps, so the reader can see which noise source dominates. **BOOT-TEAM** resamples the
8 untaught teams with replacement and recomputes *every* fold's delta from the same resampled set
(legitimate because every fold is scored on those same 8 teams); **BOOT-FOLD** resamples folds (or
points); **BOOT-BOTH** is nested.

```
  FOLD  unit   Spearman rho = -0.7563   Theil-Sen slope = -21.80 pp per unit KL
    BOOT-FOLD  rho CI [-0.9316,-0.3873] excludes 0 | slope CI [-32.8,-8.4]   width_rho 0.544
    BOOT-TEAM  rho CI [-0.8213,-0.5819] excludes 0 | slope CI [-31.3,-13.1]  width_rho 0.239
    BOOT-BOTH  rho CI [-0.9361,-0.2572] excludes 0 | slope CI [-37.2,-6.5]   width_rho 0.679

  POINT unit   Spearman rho = -0.9000   Theil-Sen slope = -11.21 pp per unit KL
    BOOT-FOLD  rho CI [-1.0000,-0.1111] excludes 0 | slope CI [-38.1,-2.8]   width_rho 0.889
    BOOT-TEAM  rho CI [-1.0000,-0.6000] excludes 0 | slope CI [-22.5,-4.9]   width_rho 0.400
    BOOT-BOTH  rho CI [-1.0000,-0.1111] excludes 0 | slope CI [-35.5,+0.4]   width_rho 0.889
```

**Which noise dominates: the FOLD, decisively.** At both units the BOOT-FOLD interval is roughly
**twice** the BOOT-TEAM interval (fold 0.544 vs 0.239; point 0.889 vs 0.400), and BOOT-BOTH is
barely wider than BOOT-FOLD alone. Team-level sampling noise is not what limits this measurement —
**the small number of folds, and the spread between folds at one `D_off`, is.** Buying more games
per team would buy nothing; buying more distinct teacher sets would buy everything.

**The one interval that spans zero is the point-level SLOPE under the nested bootstrap**
(`[−35.5, +0.4]`). So: the *direction* survives every way of resampling; the *magnitude* does not
survive resampling folds and teams at once at the honest unit. Stated as registered — a CI that
includes zero is NOT DETECTED, and that verdict attaches to the **slope**, not to the ordering.

### Within-parent only — the clean contrast

Fifteen folds, four teacher sets, **one parent (R2ACTION), one fork ancestor (rev-1 final), one
recipe family**. No cross-parent, cross-era or cross-meter term.

```
  Spearman -0.7643   BOOT-TEAM CI [-0.8074,-0.5389] excludes 0
                     BOOT-FOLD CI [-0.9334,-0.3765] excludes 0

    UNF    D_off 0.5536  mean delta  +0.87   (4 folds: TC_UNF_A/B, TC_UNF_K6_A/B)
    R3set  D_off 0.6172  mean delta  -2.50   (1 fold:  rev-3)
    FUND   D_off 0.6957  mean delta  -2.41   (2 folds: TC_FUND_A/B)
    R4set  D_off 0.7715  mean delta  -4.47   (8 folds: rev-4, COMPFOLD, B2, N1, N2, DOSE12/6/3)
```

**It is not carried by one pair.** Dropping any one of the four sets leaves an interval that still
excludes zero (`drop FUND −0.743` · `drop UNF −0.557` · `drop R4set −0.777` · `drop R3set
−0.761`), and the same holds at the 5-point level for four of five leave-one-outs (only *drop
R4set* goes to `[−1.000, +0.000]`). The funded-vs-unfunded pair that motivated the probe is
therefore **not** the whole line — but with three points remaining after any drop, "survives LOO"
is a weak statement and is offered as one.

### The y-axis floor at IDENTICAL x

Same teacher set, same parent, same `D_off`, same recipe, one argv token apart:

| pair | Δ | 95% CI |
|---|---:|---|
| **N1 − N2** | **+5.94** | [+1.94, +10.12] |
| TC_FUND_A − TC_FUND_B | −0.19 | [−3.56, +2.94] |
| TC_UNF_A − TC_UNF_B | +0.06 | [−2.50, +2.62] |
| TC_UNF_K6_A − TC_UNF_K6_B | +1.19 | [−1.06, +3.69] |

The **5.94 pp** N1/N2 draw is the size of the *entire* effect the x-axis explains between the
lowest and highest teacher set (+0.88 → −4.47, a range of 5.35 pp). **A single fold's position on
this line carries no information.** That is prediction (ii)'s failure stated as a number, and it
is the reason the verdict is "set-level" rather than "dose-response".

*(The three frozen pairs are ~0 and the one controller-live pair is 5.94 — consistent with the
2×2 batch's finding that most of "the fold floor" was the KL controller's own wander. N1/N2 ran
controller-live; the TC arms are `--fork-lr-freeze`.)*

---

## 🚩 The confound the brief asked about: NOT BROKEN by any fold in this table

Every gen-era teacher is an exploiter fork off `ai_v9_29_rev1_0823/final_model.zip` (@25.0M).
Their exploiter spans:

| set | budget (M steps) | teams / teacher | `D_off` | mean Δ |
|---|---:|---:|---:|---:|
| R2set | 3.07 | 2 | 0.2175 | +0.88 |
| UNF | 3.07 | 2 | 0.5536 | +0.87 |
| R3set | 5.07 | 2 | 0.6172 | −2.50 |
| FUND | 5.07 | 2 | 0.6957 | −2.41 |
| R4set | 10.07 | 8 | 0.7715 | −4.47 |

```
    Spearman(budget, D_off) = +0.9487
    Spearman(budget, delta) = -0.9487      <-- at least as good as distance
    Spearman(D_off,  delta) = -0.9000
```

**Teacher training budget predicts the untaught delta at least as well as off-slice distance
does.** No fold here separates them: the two variables are rank-indistinguishable over all five
points, and the two sets that share a budget (R3set / FUND, both 5.07M) also share a delta
(−2.50 / −2.41). **"A teacher's off-slice drift causes the leak" and "a teacher trained longer
causes the leak" are the same claim on this evidence.** Separating them needs an arm this table
does not contain: a teacher trained to a *long* budget but held *near* the parent (a KL-anchored
exploiter), or a short-budget teacher pushed far.

### And a second confound the probe found on its own: **the INHERITED GAP**

`D_off` is a distance to the *fold parent*, and for four of the five sets the fold parent is
R2ACTION while the teachers forked from **rev-1**. Measured on the same untaught-8 states:

```
  KL(REV1FIN || R2ACTION) = 0.3920          <-- the gen-era teachers' starting distance
  KL(FLOOR_ckpt_28067760 || R2ACTION) = 0.0374
  KL(FLOOR_ckpt_27917760 || R2ACTION) = 0.0654
```

**A gen-era teacher is already 0.3920 from R2ACTION before it trains one step** — more than half
of UNF's entire `D_off`. R2set is the exception: its teachers' fork parent *is* its fold parent,
so its inherited gap is 0 and its `D_off` is pure earned displacement.

| set | `D_off` | inherited | earned (excess) | mean Δ |
|---|---:|---:|---:|---:|
| UNF | 0.5536 | 0.3920 | ~0.162 | +0.87 |
| R2set | 0.2175 | 0 | ~0.218 | +0.88 |
| R3set | 0.6172 | 0.3920 | ~0.225 | −2.50 |
| FUND | 0.6957 | 0.3920 | ~0.304 | −2.41 |
| R4set | 0.7715 | 0.3920 | ~0.379 | −4.47 |

*(KL does not subtract; the "excess" column is an ordering aid, never a level.)*

**The decisive row: R2set and UNF share a budget (3.07M) and a delta (+0.88 / +0.87) while their
RAW `D_off` differs 2.5× (0.2175 vs 0.5536).** The whole of that difference is the inherited gap,
and it moves the delta by nothing. So the pre-registered raw-`D_off` axis is the *wrong* axis at
the one cross-parent point, and gets the right answer there only because the error happens to push
it the right way. On the earned-displacement axis `ρ = −0.800`; on raw `D_off` `ρ = −0.900`.
Neither is preferred here — the point is that **the statistic H5 named is not clean across
parents, and every within-parent comparison in this probe is unaffected** (all four R2ACTION sets
inherit the same 0.3920).

---

## 🚩 FINDING against a probe that landed today: `content_locality` scored the WRONG CHECKPOINT

[`content_locality/gen_era_locality.py`](../content_locality/gen_era_locality.py) loads each
teacher from `{run}/final_model.zip`. The training path does **not**: `main/train/model_build.py`
resolves a `--distill-teacher` run-dir through
`agents.training.fixed_opponent_pool._resolve_zip_and_config`, whose first rung is
**`best_model/best_model.zip`**, with `final_model.zip` only as the fallback. **All 19 teacher runs
checked have a `best_model/best_model.zip` whose sha256 differs from their `final_model.zip`** — so
content_locality measured a network the fold never distilled from, for every teacher.

This probe uses the training path's own resolution. The two are reconciled on **byte-identically
reproduced states** (per-team untaught counts `[280, 399, 333, 458, 714, 592, 391, 301]` in both
runs, and the matched-noise floor reproduces to four decimals at 0.0374 / 0.0654):

| half / teacher | scored by content_locality | `best_model/` = what the fold loads | Δ |
|---|---:|---:|---:|
| UNF (8 teachers) | 0.5990 | **0.5536** | −0.0454 |
| FUND (8 teachers) | 0.6969 | **0.6957** | −0.0012 |
| **v8 `pool10`** | 0.3223 | **0.3176** | −0.0047 |
| **v8 `semistall3`** | 0.2190 | **0.1036** | **−0.1154** |
| **v8 `defensive10`** | 0.2807 | **0.2775** | −0.0032 |
| **v8 set mean** | 0.2740 | **0.2329** | −0.0411 |

| statistic | on `final_model` | on `best_model` |
|---|---|---|
| FUNDED − UNFUNDED untaught KL, paired on the 8 teacher pairs | +0.0979 [+0.0544, +0.1490] SIGNIFICANT | **+0.1421 [+0.0889, +0.2015] SIGNIFICANT** |

**The v8 arm has the same defect and worse**: it scored `final_model_interrupted.zip`, which is
**not a rung of the resolver at all** (`best_model/best_model.zip` → `final_model.zip` →
`best_model.zip`). All three v8 teachers have a differing `best_model/`, and one of them —
`semistall3` — moves by **−0.1154, more than half its value**. Re-measured here from the era
checkout at `b13b30b2` ([`v8_checkpoint_fix.py`](v8_checkpoint_fix.py)) on a state batch that
reproduces content_locality's **team-by-team** (`[266, 255, 260, 312, 270, 265, 303, 259]`) with
its floor reproducing to four decimals (0.0383 / 0.0664).

**No content_locality conclusion is retracted, and both of its headlines get STRONGER.** Its
within-era headline (funded teachers sit farther from the parent off-slice) is **45 % larger** on
the checkpoints the folds actually used; its cross-era headline (v8's teachers are the local ones)
also strengthens, since v8's `D_off` falls from 5.2× to 4.5× its floor. What must be corrected is
its *levels*: the unfunded half's untaught KL is 0.5536 not 0.5990, v8's set mean is 0.2329 not
0.2740, and every `L` and `R` ratio it publishes is computed on networks the folds never used. A
follow-up should re-run it through the training path's resolver.

**The durable lesson, which is the c-family one again:** an offline probe that names a teacher by
its run directory must resolve that directory **the way the training code does**, by importing the
resolver rather than picking a filename. This probe imports `masked_kl_rows`; it should have
imported `_resolve_zip_and_config` too, and now does so only indirectly — see *Hazards* below.

---

## Recipe and provenance

| | gen arm | rev-2 arm | parent-gap arm | v8 fix arm |
|---|---|---|---|---|
| code | this worktree | this worktree | this worktree | `/tmp/v8rep_era` @ `b13b30b2`, READ-ONLY, `PYTHONDONTWRITEBYTECODE=1` |
| parent (pilot) | `ai_v9_59_R2ACTION_0827/final_model.zip` | `ai_v9_29_rev1_0823/final_model.zip` | R2ACTION | `ai_v8_04/final_model_interrupted.zip` |
| opponent | `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` | same | same | `ai_v8_03_zarch_control` final |
| play | stochastic, seeded | stochastic, seeded | stochastic, seeded | **greedy** both sides (the era has no `GEN3AI_*_SEED`) |
| bridge | rust | rust | rust | **node** |
| teacher sets | R4set (3) · R3set (6) · FUND (8) · UNF (8) | R2set (5) | — (REV1FIN + 2 floor ckpts) | v8set (3), **both checkpoint variants** |
| teams piloted | 8 untaught @ 9 battles + 40 taught @ 3 | 8 untaught @ 9 + 9 taught @ 3 | 8 untaught @ 9 | 8 untaught @ 9 |
| states | 9 618 (3 468 untaught) | 4 144 (3 247 untaught) | 3 468 | 2 190 |
| wall | **814 s** | **281 s** | **215 s** | **~180 s** |

**Seeds, verbatim from `reuse_batch_2026-09-03/offline_collateral_kl/`** (via
`content_locality/gen_era_locality.py`): sim `[team_index+1, 2, 3, 4]`; pool sequence
`random.Random(61000 + team_index)`; pilot policy `71000 + team_index`; opponent policy
`72000 + team_index`; `stochastic=True` both sides; `concurrency=1`; rust bridge. **The v8 fix arm
uses the ERA recipe instead** — greedy both sides, node bridge, no policy seeds (they do not exist
at `b13b30b2`) — copied verbatim from `content_locality/v8_era_locality.py`, which is why its
states reproduce that artifact's batch.

**Reproduction is exact, and asserted rather than hoped for.** The untaught 8 hold indices 0..7 at
9 battles, so their states are content_locality's n = 9 batch in both eras. Per-team counts match
**team-by-team** in all four arms — gen `[280, 399, 333, 458, 714, 592, 391, 301]`, v8
`[266, 255, 260, 312, 270, 265, 303, 259]` — the parent-gap and v8-fix arms **hard-assert** it and
refuse to report on a mismatch, and both matched-noise floors reproduce to four decimals (gen
0.0374 / 0.0654, v8 0.0383 / 0.0664).

**GIGO checks that ran.** Team-set resolution ([`resolve_sets.py`](resolve_sets.py)) asserts, per
set, that the taught union does **not** intersect the untaught 8, that every team file exists, and
that every teacher checkpoint resolves — all five sets clean. The distance script asserts **ACID**
(no two teachers may produce an identical per-team KL vector; a mis-resolved path would otherwise
read as a perfect null) and it passed. The fold-table script asserts the probe **stamp** per
artifact (n = 200/team, the rev-1 24M opponent, pool 719) and that every artifact's team **order**
matches, so a swapped column is unrepresentable.

`masked_kl_rows` is **imported** from `agents.training.instrumented_ppo.distill_anchor` — the same
formula the live `--distill-anchor-monitor` logs, never reimplemented.

### Reproduce

```bash
export PYTHONPATH=$PYTHONPATH:src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
cd designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
$P resolve_sets.py
$P fold_table.py
GEN3AI_TIMEOUT_SCALE=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 nice -n 10 $P teacher_distance.py gen  dist_gen.json
# ...same env for: teacher_distance.py rev2 dist_rev2.json  and  parent_gap.py parent_gap.json

# the v8 checkpoint fix runs from the ERA checkout, not this tree:
cd /tmp/v8rep_era && PYTHONPATH=/tmp/v8rep_era/src PYTHONDONTWRITEBYTECODE=1 \
  ERA_ROOT=/tmp/v8rep_era GEN3AI_TIMEOUT_SCALE=8 nice -n 10 $P \
  <abs path>/v8_checkpoint_fix.py <abs path>/v8_checkpoint_fix.json

$P analyze.py
```

| file | what |
|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | frozen before any measurement |
| [`resolve_sets.py`](resolve_sets.py) → `teacher_sets.json` | teacher sets + taught teams from run metadata; the disjointness/existence GIGO gate |
| [`fold_table.py`](fold_table.py) → `fold_table.json` | every fold's delta RECOMPUTED from per-team rows + provenance + the rejection list |
| [`teacher_distance.py`](teacher_distance.py) → `dist_gen.json` · `dist_rev2.json` (+ `.log`) | `D_off` / `D_on` per teacher and per set |
| [`parent_gap.py`](parent_gap.py) → `parent_gap.json` (+ `.log`) | the inherited gap + the floor reproduction |
| [`v8_checkpoint_fix.py`](v8_checkpoint_fix.py) → `v8_checkpoint_fix.json` (+ `.log`) | v8's `D_off` on the checkpoint the fold actually loaded, both variants, era tree |
| [`analyze.py`](analyze.py) → `analysis.json` · `analysis.log` | the whole readout above, verbatim |
| `inputs/` | a POINTER only. The 2×2 and K=6 per-team artifacts this probe rescued now live at `teacher_content_2x2_2026-09-04/`; `fold_table.py` reads them there (hazard 1) |

---

## Hazards and findings

1. **The 2×2 / K=6 per-team artifacts existed ONLY in a session-scoped job directory**
   (`~/.claude/jobs/1046b1d6/tmp/probes/`), exactly like the admission artifacts before 2026-08-31.
   Every banked funded/unfunded/K=6 untaught number in the ledger rests on files one cleanup would
   have destroyed, and `teacher_content_2x2_2026-09-04/tc_readout.py` reads them from its **own**
   directory, where they are not. The 25 `untaught_*`/`taught_*` JSONs plus `untaught_teams.json`
   and `untaught_probe.py` / `taught_probe.py` were copied into `inputs/` by this probe — which
   rescued the FILES but left `tc_readout.py` unable to run as committed, the same defect
   content_locality found in `offline_collateral_kl.py`, in a second script.
   **CLOSED 2026-09-06.** The artifacts were moved out of `inputs/` to
   `teacher_content_2x2_2026-09-04/` — the batch that PRODUCED them, and where `tc_readout.py`
   already looked — so there is ONE copy in the tree and both readouts resolve it: this probe's
   `fold_table.py` now points at that directory (re-run; `fold_table.json` and `analysis.json`
   reproduce byte-identical apart from the six recorded `artifact` provenance paths, which used to
   name a deleted worktree). `inputs/` keeps a one-line README saying where they went, and
   `src/measurements_readout_gate_test.py` fails if either readout's inputs go missing again.
2. **`content_locality` scored the wrong checkpoint in BOTH eras** — `final_model.zip` (gen) and
   `final_model_interrupted.zip` (v8, not even a rung) where the fold loads
   `best_model/best_model.zip`. Full detail above. Levels are wrong by up to **−0.115 on a single
   teacher** (`semistall3`, more than half its value); both of its conclusions strengthen. **The
   probe was landed today** — this is a same-day correction, not an old defect.
3. **`D_off` is not comparable across parents** without accounting for the inherited gap (0.3920
   here). Any future use of this statistic across a fork boundary must measure that gap; this probe
   measures it because the R2set point forced the question.
4. **No fold breaks the distance-vs-budget confound**, and the probe says so rather than reporting
   a direction as if it were causal.
5. **No fold in the gen era GIFTED**, so prediction (ii) is half-untestable: the table separates
   "robbed" from "not detected", never "robbed" from "gifted". The only gifting fold on record is
   v8, which is off this scale.
6. Not a defect, but recorded: the worktree needed `git submodule update --init` plus the two
   build-artifact symlinks before a bridge battle would run, and `sim_bridge` was taken from the
   main checkout via `$POKESIM_SIM_BRIDGE_BIN` rather than rebuilt — so no `cargo build` saturated
   a box already carrying a training run at load 17–25.
7. Measured beside a live training run (load 17–25 on 16 cores). Bounds were scaled
   (`GEN3AI_TIMEOUT_SCALE=8`); **zero timeouts occurred** across all four arms. Wall times in the
   recipe table are therefore **not** clean benchmarks, and nothing in this probe is a wall-clock
   measurement.
8. **v8's distillation coefficient (1.0) is 5.7× every gen-era fold's (0.1761)** and no probe has
   manipulated it. It is a live covariate on every v8-vs-gen contrast this program makes, pointing
   the *opposite* way to the naive reading, and it is recorded rather than explained.

---

## Limits, stated plainly

* **Five points, four of them one parent.** Everything rests on five distinct teacher sets. The
  fold-level n of 16 is an accounting artifact of ties on x, not statistical power.
* **State distribution is parent-piloted** — every teacher is scored on states the *parent*
  reaches, not on states it would itself reach. Correct for a divergence-from-parent statistic and
  necessary to make teachers comparable, but it is not the distribution the fold's rollouts see,
  and these levels must never be merged with the live `distill/collateral_kl_vs_parent` column.
* **Dose and coefficient vary across folds** and are covariates, not controls (table above).
  Within the R4set column, `gas` spans 2–12 and the deltas span `+1.50` to `−7.56` with no
  ordering — consistent with the dose cell's own null.
* **Fold LENGTH varies** and is not modelled: rev-2/rev-3/rev-4 are full folds, the TC and reuse
  arms are matched 4.45M spans. Length is a known modifier of the untaught delta (the
  hole-then-recovery shape), so part of the between-set spread may be depth, not distance. The
  four within-parent sets are **not** length-matched to each other.
* **`D_on` at 3 battles/team** where `D_off` is at 9 — a declared asymmetry, because (iii) was the
  secondary prediction. It cannot be tightened into a partial anyway (collinearity +0.965).
* **v8's `≈ +8.5 pp` at our fold length is an interpolation from one curve** and is never fitted;
  its `+4.64` at +1.09M is the banked number.
* **Not causal.** `D_off` was read off teachers that already exist; nothing was manipulated.

---

## The lever this implies — a PROPOSAL, not a build

If the ordering is distance rather than budget, the actionable form is that **a teacher's
off-slice drift is a controllable quantity**: an exploiter could be trained with a **KL-to-parent
anchor on untaught states**, spending its budget on the taught slice while being held near the
fold parent everywhere else. The machinery already exists on the *fold* side —
`--distill-anchor-mode` / `--distill-anchor-target-kl` / `distill_grad_project.py` are exactly an
off-slice trust region — and the proposal is to move that instrument **one stage upstream, onto
the exploiter**, so the fold is handed a teacher that is local by construction rather than
anchored after the fact.

**Three reasons it is a proposal and not a request for GPU.**

1. **It is not distinguishable from "train exploiters less"** on this evidence, and that is free.
   The honest first experiment is the cheap one: a short-budget teacher set at a long-budget team
   count, or the reverse.
2. **The anchored-exploiter arm is precisely the arm that breaks the confound** — a *long* budget
   held *near* the parent. That is its real value: it is a discriminating experiment before it is
   a lever.
3. **A teacher held near the parent may have nothing left to teach.** The whole point of an
   exploiter is that it moved; the on-slice gain is `~+5 pp` in every gen-era fold and
   `D_on`/`D_off` are collinear at +0.965 here, so an anchor that removes off-slice drift may
   remove the taught-side content with it. Any such arm must score **both** meters, and the
   taught-side pass must be pre-registered beside the untaught one.

Cross-reference: v8's teachers sit at **4.5× their era's noise floor** (corrected checkpoint) and
its fold gifted; our worst sit at **15.0×** and its fold robs hardest. Whether that gap is drift, budget, or the two
eras differing in ten other ways is exactly what this probe cannot say.
