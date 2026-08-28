# Drift-anchor decomposition of the rev-2 fold arms (PROBE B, 2026-08-28)

**Question.** Rev-2's fold arm landed +7.4pp above its no-distill control but only +1.6pp (n.s.)
above the shared parent, because the control *declined* 5.8pp. The **drift-anchor** hypothesis says
the fold's apparent benefit was not teacher content but ANCHORING: distilling toward five
rev-1-descended teachers acts as a stay-near-parent regularizer that prevents the drift-decline.

**Answer: MIXED, the pre-registered likely outcome — and both halves are now quantified.**
Anchoring is real but only at the ARGMAX, is not dose-ordered, and is *reversed* on the
distribution for the winning arm. Content transfer is also real, teacher-SPECIFIC, and small:
**+4.0pp of argmax agreement** in a within-team natural experiment that holds team practice fixed.
A third result falls out and is arguably the sharpest: **the arm that absorbed the most teacher
content performed second-worst** — alignment is not benefit.

> ⚠️ **This is OBSERVATIONAL.** R3-SELF (distil the parent toward *itself*) remains the causal test
> of the anchoring account. Everything here is its **prior**, not its replacement. No number below
> establishes that anchoring *caused* the fold's benefit; they establish that anchoring is present,
> that content is also present, and how large each is at the policy-function level.

---

## Verdict table

| | Registered reading | Verdict | Deciding numbers |
|---|---|---|---|
| **ANCHOR-ONLY** | arms closer to parent by dose + content row flat | **REJECTED** | Content row is not flat: on-slice teacher-alignment gain above the CTRL null **+0.068** vs off-slice **+0.016–0.020**, and the practice-controlled ZapDug DiD is **+0.0400 [+0.011,+0.070] SIG** for R2-ACTION. Ordering is also anti-dose (TOPK, the *lowest* dose, is the most anchored). |
| **CONTENT** | on-slice gain clearly above off-slice and above CTRL noise | **HOLDS, small** | On-slice +0.0676 vs off-slice +0.0158/+0.0195; survives the team-practice control at +0.0400. |
| **MIXED** | anchoring holds AND a small real content signal | ✅ **SELECTED** | Argmax anchoring +2.3 to +3.5pp vs CTRL (all three arms, paired, SIG) **and** a teacher-specific content signal ≥+4.0pp. |

---

## Provenance

| | |
|---|---|
| Parent | `ai_v9_29_rev1_0823/final_model.zip` (25,067,760 steps), resolved from every arm's `--model` |
| Fold arms | `ai_v9_59_R2ACTION` (`--distill-target action --distill-topk 1`, coef 0.1810) · `ai_v9_60_R2TOPK` (topk 3, coef 0.1810) · `ai_v9_61_R2KL` (target `kl`, coef 1.0) |
| Control | `ai_v9_58_R2CTRL` (coef 0.0) |
| Teachers | `ai_v9_53..57_R2F5a..e`, each `--exploiter` off the same rev-1 snapshot, each `--trainee-teams` two pinned teams |
| Fork length | all arms `--steps 28067760` ⇒ **Δ3.0M steps** from the parent, identical across arms |
| Code | every run recorded git `77f922e`; **`git diff 77f922e..HEAD -- src/` is EMPTY** (the 8 intervening commits are docs), so the probe ran current code that is byte-identical to the era code — no era worktree needed. `arch_signature` `gen3_critic_route_wave_v1` on all ten checkpoints. |
| Compute | CPU only, `CUDA_VISIBLE_DEVICES=""`, BLAS pinned to 1, torch 2 threads, `nice -n 15` |

**Dose.** R2-ACTION's realized `distill/distill_share` is visible in its own `launcher_child.log` at
**0.227–0.249**, confirming the ≈0.24 of record. R2-CTRL's log carries **no `distill/` block at
all**. The 0.12 / 0.31 for TOPK / KL are the recorded doses from the capstone ledger, not
re-measured here.

### The EFFECTIVE fold map (verified from the code, not assumed)

`eccfe630ec08de27` ("ZapDug") is pinned by **both** F5a and F5e. `gen3_env._distill_mask()` assigns
the teacher-id of the **first** matching teacher and `break`s, and F5a is teacher-id 1 — so in the
fold, ZapDug was taught by **F5a only**. The brief's claim is confirmed *mechanically*:

| teacher | teams as pinned | teams it EFFECTIVELY taught |
|---|---|---|
| F5a | eccfe630 (ZapDug), 023a2d47 | both |
| F5b | 8e768980, 710d8d52 | both |
| F5c | 63eda9d8, f5a4f4f0 | both |
| F5d | e541f7be, 9eb3abdc | both |
| **F5e** | e0d97b0e, **eccfe630** | **e0d97b0e only** — lost ZapDug to F5a |

All five teachers were folded with the `:*` wildcard, which `parse_distill_teacher_spec` resolves
from each teacher's **own** recorded `--trainee-teams`. So the fold *was* slice-gated: teacher k's
term fired only on teacher k's own teams.

### State sets

| set | source | n | note |
|---|---|---|---|
| **general** | parent `ai_v9_29_rev1_0823/eval_traces/step_24000000`, pooled | 3000 | reused verbatim from the plasticity-forensics run; ~99% off-slice by construction |
| **on-slice** ×5 | each teacher's own `eval_traces/step_28000032`, **filtered to the exact effective pinned-team species keys** | 1000 each | F5a–F5d traces are 100% on their pinned teams; F5e's is 64% (the remaining 36% is exactly the ZapDug states F5a owns, correctly excluded) |
| **ZapDug** | `eccfe630` states pooled from **both** F5a's and F5e's traces | 1200 (629 F5a-trace / 571 F5e-trace) | the natural experiment; pooling both sources so neither teacher's own distribution is favoured |

The forensics on-slice sets were *not* team-filtered; these are, which is why "on-slice" here means
exactly "states where that teacher's distill term fired".

### Acid test — the input reconstruction is exact on the NEW pipeline

Traces record only `obs`; the Dict observation has ~15 further auxiliary channels, filled with
"unknown" defaults. Validated by forwarding the snapshot each trace itself shipped and comparing to
the recorded logits, **on the new on-slice sets specifically**:

| slice | max abs Δ logits | corr | top-1 agreement |
|---|---|---|---|
| F5a | 2.00e-05 | 0.99999999999954 | **1.000** |
| F5b | 3.24e-05 | 0.99999999999937 | **1.000** |
| F5c | 4.29e-05 | 0.99999999999902 | **1.000** |
| F5d | 4.24e-05 | 0.99999999999920 | **1.000** |
| F5e | 2.72e-05 | 0.99999999999895 | **1.000** |

Float32 noise. Every logit matrix below is faithful.

**Policy math.** Illegal actions are masked before every softmax; logits are mean-centred over the
legal entries before any geometric comparison, because a softmax policy is invariant to a per-row
additive constant. CIs are 2000-sample bootstraps over states; arm-vs-arm comparisons are **paired**
(same states).

---

## Row 1 — ANCHORING

Distance from the rev-1 parent, all four arms, the **same** general state set (n=3000).

| arm | dose | KL(parent‖arm) | top-1 agreement w/ parent | ‖Δlogit‖_F |
|---|---|---|---|---|
| **ACTION** | 0.24 | **0.4145** [0.3942, 0.4347] | 0.7563 [0.7407, 0.7720] | 232.4 |
| TOPK | 0.12 | **0.2201** [0.2093, 0.2317] | **0.7653** [0.7500, 0.7803] | 171.1 |
| KL | 0.31 | 0.2668 [0.2533, 0.2811] | 0.7533 [0.7383, 0.7680] | 184.9 |
| **CTRL** | 0.00 | 0.3245 [0.3070, 0.3416] | **0.7300** [0.7130, 0.7453] | 202.9 |

Paired against the control on the same states (positive Δagreement = **more** anchored):

| arm | Δ top-1 agreement vs CTRL | Δ KL vs CTRL |
|---|---|---|
| ACTION | **+0.0263** [+0.0077, +0.0447] SIG | **+0.0901** [+0.0684, +0.1130] SIG (**farther**) |
| TOPK | **+0.0353** [+0.0170, +0.0547] SIG | −0.1043 [−0.1215, −0.0872] SIG (closer) |
| KL | **+0.0233** [+0.0043, +0.0427] SIG | −0.0576 [−0.0767, −0.0376] SIG (closer) |

**Anchoring is real, but only where the loss looks.** All three fold arms keep the parent's *chosen
action* significantly more often than a plain continuation does (+2.3 to +3.5pp). But:

- **It is not dose-ordered.** The anchor account predicted the ordering KL (0.31) → ACTION (0.24) →
  TOPK (0.12) → CTRL. Measured, the most anchored arm is **TOPK, the lowest dose**, and the three
  fold arms sit within 1.2pp of each other while differing 2.6× in dose. Dose does not organise this
  row.
- **The winning arm moves the DISTRIBUTION further than the control does.** R2-ACTION is the most
  distant arm by KL (0.4145 vs the control's 0.3245, paired Δ +0.090 SIG) while being *more*
  argmax-anchored. That is not a contradiction, it is the action-form's signature: with
  `--distill-topk 1` the loss is an argmax cross-entropy, so it pins exactly the functional it
  optimises and leaves the rest of the distribution free. The KL-form arm shows the mirror image —
  closer in distribution, less anchored at the argmax than TOPK.

So "the fold acts as a stay-near-parent regularizer" is true in a **narrow** sense (argmax
retention) and false in the general one (it is not distributional proximity, and it is not
dose-graded).

---

## Row 2 — CONTENT (teacher-alignment above inheritance)

For each slice: agreement with that slice's teacher, minus the *parent's* agreement with the same
teacher, on the same states. The parent already agrees with each teacher ~0.68–0.75 by inheritance,
which is why the gain-above-inheritance form is the one that carries information.

**Slice means, argmax agreement:**

| state set | ACTION | TOPK | KL | CTRL |
|---|---|---|---|---|
| **on-slice** | **+0.0200** | +0.0070 | +0.0040 | **−0.0476** |
| off-slice (other slices) | −0.0462 | −0.0514 | −0.0580 | −0.0621 |
| off-slice (general) | −0.0423 | −0.0321 | −0.0405 | −0.0618 |

Everything drifts *away* from the teachers off-slice; the control drifts away on-slice too. Read
against the CTRL drift null (arm minus control, same states — the quantity the brief asks for):

| state set | ACTION | TOPK | KL |
|---|---|---|---|
| **on-slice** | **+0.0676** | +0.0546 | +0.0516 |
| off-slice (other slices) | +0.0158 | +0.0106 | +0.0041 |
| off-slice (general) | +0.0195 | +0.0297 | +0.0213 |

**on-slice gain > off-slice gain > ≈0** — the shape the CONTENT reading predicted. Per-slice, the
on-slice signal is significant in 5/5 slices for ACTION, 4/5 for TOPK, 4/5 for KL (F5e, the slice
reduced to a single team by the ZapDug reassignment, is the one that fails).

### ⚠️ But this contrast is confounded by TEAM PRACTICE — and the confound is a real defect

The capstone record describes R2-CTRL as carrying "all five fleet runs as `--distill-teacher` at
coef 0.0 (**team-bias constancy** over the same slices)", i.e. differing from the fold arms in
exactly the distillation loss. **It does not.** Both the teacher attach *and* the trainee team bias
are gated on the same `args._distill_pairs`, which `config.py:536-537` leaves empty unless
`distill_coef > 0`:

```
config.py:536        args._distill_pairs = []
config.py:537        if args.distill_coef and args.distill_coef > 0:      # 0.0 for R2-CTRL
matchup_setup.py:109     if getattr(args, "_distill_pairs", None):
matchup_setup.py:127         bias_prob=args.distill_team_bias            # the ONLY use of the flag
```

Replaying each arm's recorded argv confirms it: the spec is identical and well-formed on all four
arms (5 teachers / 10 teams), but the **effective** `bias_prob` is 0.4 for the three fold arms and
**0.0 for the control**.

| arm | `--distill-coef` | `--distill-team-bias` passed | **effective bias** |
|---|---|---|---|
| ACTION / TOPK / KL | 0.1810 / 0.1810 / 1.0 | 0.4 | **0.4** |
| **CTRL** | 0.0 | 0.4 | **0.0** |

So the fold arms drew the ten teacher teams ~40% of the time and the control drew them at the pool
rate. The control differs from the fold arms in **two** things — the distillation loss *and* team
exposure. The on-slice-vs-off-slice contrast above therefore cannot distinguish "absorbed the
teacher's content" from "practised the teacher's team 30× more". This is the same genre as the
matched-extraction-row lesson: the control's job is to make one subtraction mean one thing.

*(This does not disturb the capstone's headline bar, which is a win-rate comparison on matched
slices, but it does mean any mechanistic claim resting on R2-ACTION − R2-CTRL inherits the extra
variable.)*

### The ZapDug natural experiment — content, with practice held fixed

The confound is removable **within** one team. `eccfe630` (ZapDug) was pinned and trained by
**both** F5a and F5e, so both are equally expert on it and equally in-distribution; but only **F5a**
taught it in the fold. Comparing an arm's alignment gain toward F5a against toward F5e on the *same
ZapDug states* holds the team, the state distribution, and every gram of team practice fixed, and
varies only **which teacher's content was distilled**.

Gain above inheritance on ZapDug states (n=1200):

| target | taught in fold? | parent~teacher | ACTION | TOPK | KL | CTRL |
|---|---|---|---|---|---|---|
| **F5a** | **yes** | 0.7575 | **+0.0600** [+.036,+.087] | +0.0567 [+.032,+.082] | +0.0400 [+.013,+.067] | −0.0242 [−.051,+.003] |
| **F5e** | no | 0.7625 | +0.0200 [−.006,+.047] | +0.0433 [+.017,+.068] | −0.0008 [−.028,+.026] | −0.0192 [−.043,+.007] |

**Difference-in-differences** — (agree F5a − agree F5e), arm minus parent. CONTENT ⇒ positive for
the fold arms and ≈0 for the control; PRACTICE ⇒ ≈0 for all:

| arm | DiD | | teacher-shift absorbed, F5a vs F5e |
|---|---|---|---|
| **ACTION** | **+0.0400** [+0.0108, +0.0700] | **SIG** | +0.591 vs +0.538 (diff **+0.054**) |
| TOPK | +0.0133 [−0.0150, +0.0442] | n.s. | +0.615 vs +0.598 (diff +0.017) |
| **KL** | **+0.0408** [+0.0092, +0.0700] | **SIG** | +0.721 vs +0.617 (diff **+0.104**) |
| **CTRL** | **−0.0050** [−0.0325, +0.0217] | **n.s.** | +0.370 vs +0.369 (diff +0.001) |

**The control behaves exactly as a null must** (−0.005, flat, and it absorbs the two teachers'
shifts equally to three decimals) — which is what licenses reading the fold arms' DiD as signal.

**Content transfer is real and teacher-specific: ≈+4.0pp of argmax agreement.** On the ZapDug slice
the total alignment gain over the drift null is +0.0842 for ACTION, of which **+0.0400 (≈48%) is
teacher-IDENTITY-specific**. And that 48% is a **lower bound**: any content F5a and F5e *share* —
and two exploiters trained on the same team should share a great deal — cancels in a
difference-in-differences by construction. The honest statement is *at least* half of the on-slice
alignment gain is content rather than practice.

### Alignment is not benefit (the informational question, answered)

The brief asked whether the harmful full-KL arm aligned with the teachers *more* while performing
worse. **It did, on every measure:**

| arm | on-slice teacher-shift absorbed | ZapDug DiD | meter vs rev-1 |
|---|---|---|---|
| ACTION | **0.188** | +0.0400 SIG | **+0.0161** |
| TOPK | 0.418 | +0.0133 n.s. | −0.0239 |
| **KL** | **0.492** | **+0.0408 SIG** | **−0.0320** |
| CTRL | 0.347 | −0.0050 n.s. | −0.0580 |

R2-KL absorbed the most of the teachers' logit shift (0.492) and carried the largest
teacher-specific differential (absorbed diff +0.104, 1.9× ACTION's), and it finished 4.8pp *below*
R2-ACTION. Meanwhile R2-ACTION absorbed **less of the teacher direction than the control did**
(0.188 vs 0.347) while matching the teachers' *chosen action* best — it copies the decision, not the
direction. Copying more of a teacher is not the same as getting better, and the two channels the
arms differ in (which functional the loss targets) matters more than how much is copied. This is the
same shape as the project's existing target-form finding and its "rank decoupled from performance"
record.

---

## Row 3 — DRIFT vs DECLINE (directional only)

| arm | KL to parent | top-1 agreement | meter Δ vs rev-1 |
|---|---|---|---|
| ACTION | 0.4145 | 0.7563 | +0.0161 |
| TOPK | 0.2201 | 0.7653 | −0.0239 |
| KL | 0.2668 | 0.7533 | −0.0320 |
| CTRL | 0.3245 | 0.7300 | −0.0580 |

| correlation | n | r |
|---|---|---|
| top-1 agreement vs meter | 4 | **+0.657** |
| top-1 agreement vs meter (+ parent at 1.0 / 0.0) | 5 | +0.449 |
| KL-to-parent vs meter | 4 | **+0.523** |
| KL-to-parent vs meter (+ parent at 0 / 0) | 5 | −0.110 |

**The two distance metrics disagree, so this row supports nothing.** The drift account predicts
decline tracks distance: *negative* for KL-vs-meter, *positive* for agreement-vs-meter. Agreement
gives +0.657 (supports); KL gives +0.523, the **wrong sign** for the account (more distributional
drift, *better* meter). The disagreement is driven entirely by R2-ACTION, which is simultaneously
the farthest arm by KL and the best performer — the single point the drift account most needs to be
close, and it is not.

With **n=4** (5 with the parent) and one arm driving the sign, nothing here is claimable beyond the
direction, and the direction is metric-dependent. Reported as registered; no inference drawn.

---

## What this implies for R3 (a prior, not a result)

- **R3-SELF is still the causal test** and should still run. This probe cannot establish causation.
- The registered MIXED branch asks for the content signal to be quantified so R3's arms have a
  prior. It is: **≥+4.0pp of teacher-specific argmax agreement**, against an anchoring component of
  **+2.3 to +3.5pp of argmax retention** that R3-SELF would also receive.
- Therefore **R3-ACTION > R3-SELF is expected, but by a modest margin** — most of the fold-vs-control
  gap is anchoring plus team practice, not content. A "R3-ACTION ≈ R3-SELF" outcome would be
  *consistent* with these numbers only if content's win-rate value is near zero despite being
  functionally present; a "R3-ACTION ≫ R3-SELF" outcome would be *surprising* given how small the
  practice-controlled content signal is.
- **Do not convert +4.0pp of argmax agreement into win-rate points.** Nothing here licenses that
  mapping, and R2-KL is the standing counter-example: the most teacher-aligned arm was the
  second-worst performer.
- **R3's control should fix the team-bias gate.** If the intent is "differs only in the distillation
  loss", the control needs the 0.4 team bias actually applied — today `--distill-team-bias` is
  silently inert at `--distill-coef 0`. Either give the control a bias path that does not run
  through `_distill_pairs`, or state the exposure difference as a named variable.

---

## MISSING cells

| Cell | Reason |
|---|---|
| **Causal test of anchoring (R3-SELF)** | By design out of scope — it is a GPU training run, not an observational read. This probe is its prior. |
| **A team-bias-matched control** | Does not exist in the archive. R2-CTRL's 0.4 bias was inert (proved above), so no run isolates the distillation loss from team exposure at the whole-fleet level. The ZapDug DiD is the within-team substitute and covers exactly one of the nine effective slices. |
| **DiD on the other eight slices** | Structurally impossible: ZapDug is the **only** team pinned by two teachers, so it is the only place where "taught by X, equally expert Y available" exists. |
| **Value-function (critic) decomposition** | Not run. Every row here is policy-side (logits/argmax). The fold also carried `--distill-value-feat-coef 0.5`, whose transfer is unmeasured. |
| **Dose re-measurement for TOPK / KL** | Only R2-ACTION's `distill_share` (0.227–0.249) was read off its own log; 0.12 / 0.31 are quoted from the ledger, not re-derived. |
| **Off-slice ZapDug companion** | The DiD is on-slice only; the off-slice mirror would need a team two teachers pin but neither taught, which does not exist. |

## Caveats

- **On-slice states come from the teachers' own trace distributions**, off-slice from the parent's
  or from other teachers'. The on/off contrast is therefore confounded with a state-distribution
  shift as well as with team practice. Both confounds are *shared* across arms, so they largely
  cancel in the arm-minus-parent and arm-minus-control differences that carry the findings — and the
  ZapDug DiD removes both by construction, which is why it is the load-bearing measurement and not
  the on/off table.
- **The DiD is a lower bound on content**, not an estimate of it: content shared between F5a and F5e
  cancels. It should not be read as "only 48% of the gain is content".
- **n=5 slices, n=4 arms.** Row 3's correlations are decorative. The per-slice content signals are
  n=1000 states each and carry real CIs, but the *slice* is the unit for any claim about the fleet,
  and there are five of them (four, once F5e is reduced to a single team).
- **Argmax agreement is a coarse functional.** Two policies can agree on every argmax and differ
  materially in play; R2-ACTION's simultaneous high KL and high agreement is exactly that regime.
  The `teacher_shift_absorbed` column is the continuous companion and it tells a partly different
  story, which is reported rather than reconciled.
- **F5e's slice is degraded.** Losing ZapDug to F5a leaves it one effective team, and it is the
  slice where the on-slice content signal fails significance for TOPK and KL. Treat the F5e row as
  the weakest of the five.
- **Meter values are quoted, not re-measured.** ACTION +0.0161 / TOPK −0.0239 / KL −0.0320 /
  CTRL −0.0580 come from the capstone ledger.

## Reproduction

Scripts under `tmp/` in this worktree (untracked — `tmp/` is gitignored):
`dab_build_states.py` (team-filtered on-slice sets + the effective fold map),
`dab_forward.py` (`acid` / `onslice` / `general` — CPU forwards + the acid test),
`dab_analyze.py` (rows 1–3), `dab_zapdug.py` (the natural experiment),
`dab_bias_check.py` (the team-bias gate). Every raw number is in the sibling `.json`.
