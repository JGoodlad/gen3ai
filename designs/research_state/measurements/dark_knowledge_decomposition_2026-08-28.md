# Dark-knowledge decomposition — why full-distribution KL distillation underperforms (2026-08-28)

**Question.** Hinton-style dark knowledge says a teacher's *whole* distribution carries more
signal than its argmax. The R2 fold measured the opposite ordering: `action +1.6 > top-3 −2.4 >
full-KL −3.2 > nothing −5.8` (pp vs rev-1 final, ledger capstone 2026-08-27). Why?

**Answer: the harm ordering tracks how much of the TEACHER'S TAIL SHAPE each target form copies,
and the tail is the half of the teachers' divergence that carries no fork-specific content.**
Mode transfer helps; tail transfer is a drag that partly cancels it. But the registered
hypothesis is only **half** right — the teachers' divergence is **not** tail-dominated. It is
mode-dominated (61% of the KL is banked on the 27% of states where the argmax flips). The tail
is the *harmful* half, not the *big* half.

**One framing correction that matters before any number below is read.** Against the R2-CTRL null
(`−5.8`), **every** distill arm beat doing nothing: full-KL is `+2.6pp` over the control. Full-KL
does not *hurt*; it *helps least*. Every "harm" in this document means "benefit forgone relative
to the action form".

All numbers are read-only over `models/`, CPU-only, `nice -n 15`, ≤2 threads.

---

## Verdict on the registered hypothesis

| Clause of the registered hypothesis | Verdict | Deciding numbers |
|---|---|---|
| Teacher-vs-parent divergence is **TAIL-dominated** | **REFUTED** | State partition, off-slice pooled: **mode share 0.608 / tail share 0.392**. Argmax agreement 0.730, but the 27% flip states carry 61% of the KL (mean KL 0.812 on flips vs 0.194 on agreements, **4.2×**). |
| …**especially off-slice** | **directionally SUPPORTED, negligible** | tail share off-slice 0.392 vs on-slice 0.376 (+1.6pp). Both mode-dominated. |
| Harm ordering **tracks copied tail** | **SUPPORTED** | On-slice tail absorption above the CTRL null: ACTION **+0.040** → TOPK **+0.102** → KL **+0.116**; benefit over nothing **+7.4 → +3.4 → +2.6pp**. Monotone, and dose-clean at the TOPK/ACTION contrast (identical coef, and ACTION runs at **1.81× TOPK's loss magnitude** while transmitting *less* tail and delivering *more* benefit). |
| The copied tail is **noise** (drift, not exploit content) | **SUPPORTED** | A fork's tail-restricted policy shift aligns with a **no-fork control's** almost exactly as well as with a **sibling fork's**: inter-fork cos **0.327** vs fork-vs-CTRL **0.306**, excess **+0.021**. The mode half's excess is **+0.043** — twice as specific, and itself barely specific. |
| Corollary: v8's tails carry **more real structure** | **MIXED** | v8's tail share of divergence is **higher** (0.457 vs the gen era's era-symmetric 0.370) — but v8's *total* divergence is 2.5× smaller (KL 0.179 vs 0.445), so v8's absolute tail load is ~40% of the gen era's. The within-state coordinate split goes the other way (v8 0.390 tail vs gen 0.605). **No v8 no-fork control exists**, so v8's tail cannot be tested for noise the way the gen era's was. |

**Net: MIXED, with the mechanism SUPPORTED and the premise REFUTED.** "Full-KL copies dark noise"
is right about *what* it copies and *why that costs*; it is wrong that noise is most of what is
there. The corrected statement: **the mode half of the teachers' divergence is the transferable
half, the tail half is drift, and a target form's value is set by how selectively it copies the
first without the second.** That is a *selectivity* account, not a *magnitude* account — and it
predicts the ordering without any appeal to dose.

---

## Provenance

| | |
|---|---|
| Parent | `ai_v9_29_rev1_0823/final_model.zip` (25.07M steps) |
| Teachers | `ai_v9_53..57_R2F5a..e` finals (Δ3.05M each, 2 pinned teams each) |
| Fold arms | `ai_v9_58_R2CTRL` (coef 0.0) · `ai_v9_59_R2ACTION` (`--distill-target action --distill-topk 1`, coef 0.1810) · `ai_v9_60_R2TOPK` (`--distill-topk 3`, coef 0.1810) · `ai_v9_61_R2KL` (default `kl` target, coef 1.0) |
| v8 comparison | `ai_v8_04` parent + `ai_v8_06/09/13` teachers, read at the length-matched checkpoints |
| Off-slice states | `ai_v9_29/eval_traces/step_24000000`, **n = 3000**, mean **6.72 / 11** legal actions |
| On-slice states | each fork's own last trace step, **n = 1500** per fork |

**Logit source.** Parent / F5a–e / CTRL / ACTION and the v8 triple were reused from the plasticity
forensics' dumps (`/tmp/plast/fwd`); TOPK (`ai_v9_60`) and KL (`ai_v9_61`) were forwarded here.
`ai_v9_62_R2PLAIN` has no `final_model.zip` yet — **MISSING**.

**Dump caveat, load-bearing.** `plast_forward.py` calls `get_distribution(obs)` **without** the
`action_masks` argument, so the stored tensors are torch-Categorical log-probs over **all 11**
slots, not mask-normalised (verified: `exp().sum(1) == 1` over 11; legal-only sum 0.87–0.99). The
legal mask is re-applied here, which is the training-faithful choice — `_distill_loss` masks both
sides before the softmax.

**Acid test — pipeline reproduces recorded trace logits.** Forwarding the snapshot the trace
itself shipped, through *this* agent's load/forward stack: max |Δ| **1.91e-05**, corr
**0.9999999999996**, top-1 agreement **1.000** (n=256). Float32 noise, identical to the forensics
standard.

---

## M1 — the decomposition: mode vs tail

`KL(teacher ‖ parent)` in nats, legal-masked. Two complementary splits:

* **State partition** (the brief's definition) — KL banked on states where the argmax *agrees* is
  pure tail reshaping: the decision did not move, only the distribution around it.
* **Coordinate partition** — within every state, the KL sum restricted to the two
  decision-relevant slots (each side's argmax) vs all other legal slots.

| fork | slice | argmax agree | mean KL | KL on agree | KL on flip | **tail share (states)** | **mode share (states)** | tail share (coords) | mode share (coords) |
|---|---|---|---|---|---|---|---|---|---|
| F5a | off | 0.741 | 0.3379 | 0.1723 | 0.8109 | 0.378 | 0.622 | 0.490 | 0.510 |
| F5a | on | 0.768 | 0.3631 | 0.1885 | 0.9412 | 0.399 | 0.601 | 0.451 | 0.549 |
| F5b | off | 0.741 | 0.3358 | 0.1894 | 0.7545 | 0.418 | 0.582 | 0.595 | 0.405 |
| F5b | on | 0.692 | 0.5303 | 0.2387 | 1.1855 | 0.311 | 0.689 | 0.454 | 0.546 |
| F5c | off | 0.742 | 0.3248 | 0.1858 | 0.7235 | 0.424 | 0.576 | 0.593 | 0.407 |
| F5c | on | 0.751 | 0.4035 | 0.2226 | 0.9481 | 0.414 | 0.586 | 0.500 | 0.500 |
| F5d | off | 0.709 | 0.4363 | 0.2261 | 0.9484 | 0.367 | 0.633 | 0.565 | 0.435 |
| F5d | on | 0.725 | 0.4563 | 0.2369 | 1.0337 | 0.376 | 0.624 | 0.521 | 0.479 |
| F5e | off | 0.717 | 0.3728 | 0.1950 | 0.8230 | 0.375 | 0.625 | 0.570 | 0.430 |
| F5e | on | 0.736 | 0.3795 | 0.1948 | 0.8945 | 0.378 | 0.622 | 0.508 | 0.492 |
| **POOLED** | **off** | **0.730** | **0.3615** | **0.1937** | **0.8121** | **0.392** | **0.608** | **0.563** | **0.437** |
| **POOLED** | **on** | **0.734** | **0.4265** | **0.2163** | **1.0006** | **0.376** | **0.624** | **0.487** | **0.513** |

**Reading.** The forensics' P4 anomaly — high KL *with* high argmax agreement — is real but does
**not** imply a tail-dominated divergence. High agreement (0.730) coexists with mode dominance
because the minority of flip states are individually **4.2× more divergent**. A distribution that
moves without moving its decision moves *cheaply*; a distribution that flips its decision moves
*expensively*. The KL integral is dominated by the expensive minority.

The two splits disagree by construction and both are reported because they answer different
questions: **61%** of the KL is banked on states whose decision moved (state partition), while
**56%** of the KL sits on non-decision coordinates (coordinate partition). The first says
*most divergence is about decisions*; the second says *most of it is spread over slots nobody
picks*. Both are true; neither alone settles the hypothesis, which is why M2 measures transmission
directly.

### M1b — is the tail fork-specific, or is it drift?

Cosine between the tail-restricted (and mode-restricted) policy-shift vectors, off-slice. R2-CTRL
is the null: a run forked from the same parent, same steps, **`--distill-coef 0.0`**, no pins.

| | inter-fork (sibling ↔ sibling) | fork ↔ CTRL | **excess** |
|---|---|---|---|
| **MODE** coordinates | **0.382** (range 0.289–0.479) | **0.339** (0.307–0.366) | **+0.043** |
| **TAIL** coordinates | **0.327** (range 0.272–0.379) | **0.306** (0.278–0.342) | **+0.021** |

Five exploiter forks, each pinned to its own two teams, agree with each other about their **tail**
reshaping to within 2 cosine points of how well they agree with a run that never had a teacher or
a pin. **The tail carries essentially no fork-specific content.** The mode half is twice as
specific — and still barely so, which is the forensics' supply verdict restated at the coordinate
level.

*Caveat:* the forks ran `--grad-accum-steps 8`, R2-CTRL `2` — 4× the optimizer steps for the same
env steps. That inflates CTRL's total movement and therefore, if anything, *understates* the
forks' specificity. The M2b table below is free of this confound (all four arms ran at accum 2).

v8 counterpart, for whatever it is worth: inter-fork cos MODE **0.303**, TAIL **0.349** — v8's
teachers agree with each other *more* on the tail than on the mode, the reverse of the gen era's
ordering. **No v8 no-fork control exists** (the forensics' biggest MISSING cell), so this cannot
be turned into a specificity excess.

---

## M2 — what each target form transmits, vs the harm ordering

Every arm forked rev-1 final, so at step 0 **the student *is* the parent**: the distill loss the
optimiser sees is exactly `KL(q_target ‖ p_parent)` and its gradient wrt the student logits is
`(p_parent − q_target)`. That gradient is decomposed below into decision coordinates and the rest,
which makes the three forms exactly comparable.

### Static transmission at initialisation (off-slice pooled, n=3000×5)

| form | coef | loss at init | coef × loss | tail share of loss | **tail-signal L1** | **cos with full-KL's tail signal** | target off-mode mass |
|---|---|---|---|---|---|---|---|
| **full** (KL, arm 61) | 1.000 | 0.3615 | 0.3615 | 0.392 | 0.1721 | **1.000** | 0.3032 |
| **top-3** (arm 60) | 0.181 | 0.4149 | 0.0751 | 0.390 | 0.1967 | **0.916** | 0.2694 |
| **top-1** (action, arm 59) | 0.181 | 0.7520 | 0.1361 | 0.280 | 0.2149 | **0.308** | 0.0000 |

On-slice the same table reads full 1.000 / top-3 **0.935** / top-1 **0.341** for the tail cosine.

**The decisive column is `cos with full-KL's tail signal`** — how much of the *direction* of the
teacher's tail reshaping the form reproduces. It falls **1.000 → 0.916 → 0.308** exactly as harm
falls. Note what it separates: `tail-signal L1` (how much tail mass the form *moves*) is
**largest** for the one-hot form (0.215), because a one-hot target asks the student to push *all*
tail mass to zero. That is a large tail movement carrying **no teacher tail information** — which
is why "copied tail mass" is the wrong meter and "copied tail *shape*" is the right one. The
registered hypothesis named the former; the data select the latter.

### Realised transmission — on-slice, where `distill_mask` actually fires

Off-slice the distill term is masked out by construction, so this is the only place a form's
content can show up. Projection/cosine of each arm's own policy shift onto **its slice's teacher's**
shift, pooled over the five slices:

| arm | cos MODE | cos TAIL | proj MODE | proj TAIL | **mode excess over CTRL** | **tail excess over CTRL** | mode:tail selectivity | **benefit over nothing** |
|---|---|---|---|---|---|---|---|---|
| **CTRL** (coef 0) | 0.347 | 0.296 | 0.304 | 0.286 | — | — | — | 0.0 pp |
| **ACTION** (top-1) | 0.408 | 0.336 | 0.445 | 0.356 | **+0.061** | **+0.040** | **1.53** | **+7.4 pp** |
| **TOPK** (top-3) | 0.468 | 0.398 | 0.404 | 0.373 | +0.121 | +0.102 | 1.19 | +3.4 pp |
| **KL** (full) | 0.480 | 0.412 | 0.435 | 0.398 | +0.133 | +0.116 | 1.15 | +2.6 pp |

(`benefit over nothing` = the ledger's arm result minus R2-CTRL's `−5.8`.)

**Three things this settles.**

1. **Distillation transmits.** Every distilled arm sits above the no-distill null on both halves,
   and tail absorption rises monotonically with the form's tail fidelity: `0.296 → 0.336 → 0.398 →
   0.412`. The channel works; the question was only ever *what* to put in it.
2. **Benefit falls as tail transmission rises** — `+0.040 → +0.102 → +0.116` against `+7.4 → +3.4
   → +2.6 pp`. Monotone across all three forms.
3. **It is not dose.** ACTION and TOPK ran at the **identical** coefficient (0.1810), the identical
   gate (`none`), the identical AWR β. ACTION's loss magnitude is **1.81×** TOPK's (0.752 vs
   0.415), so a pure-dose account predicts ACTION is the *more* disruptive arm. It is the *less*
   disruptive one, transmits **less** tail (+0.040 vs +0.102) and **more** benefit (+7.4 vs +3.4).
   The sign is reversed against dose. (The ledger reaches the same conclusion by a different route:
   "R2-TOPK lost to R2-ACTION at HALF the dose, z=−2.96 — direction safe against the dose
   confound".)

The arm that works is the one with the highest **mode:tail selectivity** (1.53 vs 1.19 / 1.15): it
buys the most decision transfer per unit of tail drift imported.

⚠ **One asymmetry in the null.** R2-CTRL ran `--distill-value-feat-coef 0.0`; all three distilled
arms ran `0.5` (the FitNets cosine hint on `value_pooled`). So each *absolute* "excess over CTRL"
includes that hint as well as the policy term. It is **common to all three arms**, so the
**ordering** — the only thing the argument rests on — is untouched; only the absolute excesses are
inflated by an unknown shared amount.

**The dose confound that *does* exist, stated plainly.** R2-KL ran at coef 1.0 against the action
arms' 0.1810 — `coef × loss at init` of **0.362 vs 0.136 / 0.075**, and the ledger's measured dose
ratio is 1.32×. So part of KL's larger tail absorption is simply that it trained hotter. The
TOPK-vs-ACTION contrast is the dose-clean discriminator and it points the same way, which is why
the conclusion does not rest on the KL arm.

---

## M3 — entropy / decisiveness

Off-slice, n=3000, legal-masked. `H/logN` normalises by the per-state legal-action count.

| model | H (nats) | H/logN | ΔH vs parent | mean p(top-1) | frac states p>0.9 |
|---|---|---|---|---|---|
| **parent (rev-1)** | 0.7079 | 0.3889 | 0.0000 | 0.7307 | 0.301 |
| F5a | 0.7345 | 0.4012 | +0.0267 | 0.7193 | 0.310 |
| F5b | 0.7919 | 0.4295 | +0.0841 | 0.6979 | 0.261 |
| F5c | 0.8056 | 0.4375 | +0.0977 | 0.6934 | 0.251 |
| F5d | 0.8265 | 0.4491 | +0.1187 | 0.6855 | 0.237 |
| F5e | 0.8190 | 0.4446 | +0.1111 | 0.6883 | 0.236 |
| **CTRL (58)** | 0.7836 | 0.4289 | **+0.0757** | 0.7052 | 0.255 |
| **ACTION (59)** | **0.5687** | **0.3102** | **−0.1392** | **0.7851** | **0.407** |
| **TOPK (60)** | 0.7790 | 0.4260 | +0.0711 | 0.7026 | 0.243 |
| **KL (61)** | 0.7932 | 0.4334 | +0.0853 | 0.7001 | 0.245 |

On-slice, pooled over the five pinned slices (parent p(top-1) = 0.754):

| model | ΔH vs parent | p(top-1) |
|---|---|---|
| its own teacher | **+0.0671** | 0.727 |
| CTRL | **+0.0969** | 0.716 |
| ACTION | **−0.2970** | **0.866** |
| TOPK | +0.0076 | 0.750 |
| KL | +0.0540 | 0.737 |

**Did the teachers get more decisive? No — less.** Every exploiter teacher is *more* entropic than
its parent (+0.027 to +0.119 nats off-slice). But **so is the no-fork control** (+0.076), sitting
squarely inside the teachers' range. Entropy inflation is what 3M steps of continued training does
here; it is not a fork effect, and it is not "specialisation sharpening its choices".

**Did the KL student inherit teacher entropy? Yes, essentially exactly.** On-slice the KL arm's
ΔH (+0.054) tracks the teachers' (+0.067); TOPK inherits a fraction (+0.008); **ACTION reverses it
(−0.297)**. Off-slice the same: KL +0.085 ≈ teachers' +0.027…+0.119, ACTION −0.139.

**The decisiveness row and the transmission row are the same finding seen twice.** The one arm that
improved is the one arm that got *sharper* — p(top-1) 0.731 → 0.785 off-slice, 0.754 → **0.866**
on-slice, and the fraction of near-deterministic states 0.301 → **0.407**. Forms that copy the
teacher's tail also copy its diffuseness, and end up where the control ended up. A one-hot target
is an entropy *sink*: the only distill form here whose fixed point is more decisive than the
parent. (Feeding the owner's separate control-theory question: the fold's success case is a
**decisiveness** intervention, not an information-transfer one.)

---

## M4 — the code fact

`src/agents/training/instrumented_ppo/distill_terms.py`, reached from `ppo.py::train` (v103).

| | |
|---|---|
| **KL direction** | **FORWARD KL — `KL(teacher ‖ student)`**: `(p_t * (log p_t − log_softmax(student))).sum(-1)`. **Mass-covering.** The student is penalised wherever the *teacher* has mass and it does not, so **every teacher tail bump is a positive-mass constraint on the student**. The mode-seeking direction `KL(student ‖ teacher)` — which would let the student ignore teacher mass it does not want — is not implemented anywhere on this path. |
| **Temperature** | **NONE.** No temperature parameter exists on the exploiter-distill path; both sides are at T = 1. The only sharpening knob is `--distill-topk`. |
| **Mask ↔ tail** | `neg = (action_mask − 1) * 1e9` is added to **both** student and teacher logits before `softmax` / `log_softmax`, so both normalise over the **legal set only** and illegal-action mass is exactly 0 on both sides. **The tail under study is the legal tail** (mean 6.72 legal of 11) — there is no illegal-action leakage to blame. |
| **top-K** | `p_t.topk(k)` then renormalise over the kept set. `k = 1` ⇒ one-hot, whose KL form reduces to `−log π_student(a_teacher)` (plain CE). `k ≥ n_actions` reproduces `_distill_loss` exactly — the action path is a strict superset. |
| **⚠ AWR asymmetry between the arms** | The **action path carries a per-row weight the KL path does not**: `w = clamp(exp(|Â|/β), max=20)`, Â the minibatch-normalised advantage. `_distill_loss` uses a plain masked mean (`w ≡ 1`). So R2-ACTION and R2-TOPK trained at an *effective* dose **above** their nominal 0.1810, while R2-KL trained at exactly 1.0. This is **conservative** against the finding — it inflates the dose of the arm that won. |

**Why the direction matters here.** Forward KL is precisely the choice that cannot ignore a
teacher tail. Under `KL(teacher ‖ student)` a teacher slot at p = 0.03 that the student would set
to 0.001 costs `0.03 × log 30 ≈ 0.10` nats and pulls hard; under the reverse direction it would
cost the student nothing to drop. Given M1b — the tails are as shared with a *no-fork control* as
with a *sibling fork* — the implemented direction is the one that makes drift maximally
contagious. `--distill-topk` is currently the only lever in the tree that truncates that
contagion, and the R2 ordering is the measurement of what truncating it is worth.

---

## M6 — per-phase drift, and the opponent-prediction question

Weight-level, per-key relative Frobenius `‖ΔW‖/‖W‖` at matched Δ3.0M, read from the forensics'
`per_key_rel` dump (no new forwards). The forensics measured encoders ≈ 0 and the trunk drifting;
this localises the belief/intent families inside its G4 bucket.

| phase family | F5a | F5b | F5c | F5d | F5e | CTRL(58) | ACTION(59) |
|---|---|---|---|---|---|---|---|
| *(ref)* encoders + embeddings | 0.0421 | 0.0404 | 0.0408 | 0.0422 | 0.0422 | 0.0497 | 0.0565 |
| *(ref)* trunk (projection + team_transformer) | 0.0724 | 0.0734 | 0.0773 | 0.0751 | 0.0771 | 0.1075 | 0.1128 |
| **opp_intent (opponent-action prediction)** | **0.0787** | **0.0950** | **0.0912** | **0.0821** | **0.1049** | **0.1448** | **0.1585** |
| pair_outcome / switch_branch | 0.0855 | 0.0580 | 0.0603 | 0.0720 | 0.1203 | 0.1242 | 0.1333 |
| spread_belief | 0.0696 | 0.0705 | 0.0710 | 0.0723 | 0.0692 | 0.0969 | 0.0904 |
| move_belief / move_latent | 0.0672 | 0.0724 | 0.0697 | 0.0704 | 0.0706 | 0.0855 | 0.0808 |
| opp_belief (hidden-team) | 0.0680 | 0.0627 | 0.0654 | 0.0649 | 0.0687 | 0.0961 | 0.0909 |
| hp_type_belief | 0.0605 | 0.0712 | 0.0632 | 0.0641 | 0.0646 | 0.0800 | 0.0862 |
| item_belief | 0.0459 | 0.0499 | 0.0496 | 0.0414 | 0.0372 | 0.0668 | 0.0571 |
| value_dist | 0.0499 | 0.0447 | 0.0605 | 0.0478 | 0.0470 | 0.1448 | 0.1634 |
| cf_/twin/shadow/evidential | 0.0277 | 0.0263 | 0.0279 | 0.0288 | 0.0246 | 0.0516 | 0.0548 |

**`opp_intent` is the highest-drift phase in every single column** — above the trunk reference and
roughly **2× the encoders**, in forks and in fold arms alike. So yes: the opponent-prediction
pathway is the most mobile part of this network. **But it is not fork-driven** — the no-fork
control drifts *more* there (0.1448) than any exploiter fork (0.079–0.105), which is the same
shape the forensics found everywhere else: this is what continued training does, not what forking
does.

⚠ **Two confounds, both real.** (1) The forks ran `--grad-accum-steps 8`, the fold arms `2` — 4×
the optimizer steps per env step, which inflates the whole CTRL/ACTION column. Compare *within* a
column, not across. (2) CTRL/ACTION additionally enabled `--cf-records --cf-twin-heads
--cf-shadow-critic --cf-evidential --capacity-telemetry`, which is why `value_dist` jumps 3× there.
The **fork columns are the clean ones**, and within them `opp_intent` (0.079–0.105) and
`pair_outcome/switch_branch` (0.058–0.120) are the two families above trunk.

v8 has no `opp_intent`, `opp_belief`, `pair_outcome` or `cf_` modules at all (they postdate it), so
the cross-era version of this row is structurally **MISSING**. The families it does share sit at
`hp_type 0.018–0.030`, `move_belief 0.032–0.050`, `spread 0.031–0.045`, `value_dist 0.044–0.084`
against an encoder reference of 0.020–0.032 — the same ordering, at roughly half the magnitude.

---

## M5 — the v8 comparison point

Computed from the forensics' surviving v8 logit dumps (`/tmp/plast/fwd/v8`). **No new era worktree
was pinned**, per brief.

| era / fork | argmax agree | mean KL | tail share (states) | mode share (states) | tail share (coords) | mode share (coords) |
|---|---|---|---|---|---|---|
| v8 semistall3 | 0.660 | 0.1801 | 0.459 | 0.541 | 0.443 | 0.557 |
| v8 pool10 | 0.679 | 0.1856 | 0.437 | 0.563 | 0.214 | 0.786 |
| v8 defensive10 | 0.633 | 0.1720 | 0.474 | 0.526 | 0.513 | 0.487 |
| **v8 POOLED off-slice** | **0.657** | **0.1792** | **0.457** | **0.543** | 0.390 | 0.610 |
| **v8 POOLED on-slice** | **0.478** | **0.3737** | **0.306** | **0.694** | 0.411 | 0.589 |
| **gen POOLED off-slice, all-legal** | **0.656** | **0.4446** | **0.370** | **0.630** | 0.605 | 0.395 |

The last row is the **era-symmetric arm**: the gen era recomputed with the same all-legal mask v8
is forced to use, so the two columns compare like with like.

**Reading.** The corollary is **MIXED**.

* *Supports it:* v8's tail share of divergence is higher (**0.457 vs 0.370**) and, decisively, v8's
  on-slice behaviour is the specialisation signature the gen fleet lacks — agreement collapses to
  **0.478** on its own pinned teams (vs 0.657 off), mean KL doubles to 0.374, and the tail share
  drops to 0.306. **When a v8 teacher is on its own slice, it moves its decisions.** The gen fleet's
  on-slice agreement (0.734) is indistinguishable from its off-slice (0.730).
* *Cuts against it:* v8's *total* divergence is **2.5× smaller** (0.179 vs 0.445), so its absolute
  tail load is ~40% of the gen era's — v8 teachers were not richer in the tail, they were quieter
  everywhere. And the within-state coordinate split runs the other way (v8 0.390 tail vs gen 0.605),
  with `pool10` a large outlier (0.214).

⚠ **The v8 half of this table is not scale-comparable and must not be quoted alone.** v8 traces
carry **no `action_mask`**; the forensics reconstructed it as all-legal, so a v8 "tail" spans 11
slots against the gen era's 6.72. Everything above is the gen era dragged down to that footing, and
even so the residual era differences (obs 2992 vs 2501, module set, 277M vs 25M parent steps) are
irreducible. **No v8 no-fork control exists**, so v8's tail cannot be tested for drift-noise the way
M1b tests the gen era's — which is the single measurement that would convert this MIXED into a
verdict.

---

## What this implies

**Dark knowledge is not refuted; the regime is wrong, and the regime has a measurable signature.**
A teacher's tail is worth copying when it encodes structure the teacher learned; it is worth
truncating when it encodes where the teacher's weights happened to wander. This probe supplies the
discriminator: **compare a candidate teacher's tail-restricted policy shift against a no-fork
control's.** Here the excess was `+0.021` cosine — nothing — and full-distribution KL duly
underperformed a one-bit target. On a consolidated parent whose forks genuinely differentiate,
that excess should be large, and the ordering should invert.

Three consequences, none of them requiring a new run to state:

1. **The teacher-admission gate has a cheap new column.** `tail specificity excess` = inter-fork
   tail cosine − fork-vs-control tail cosine, computed from logits on ~3000 recorded states in
   seconds. It is the number that says whether a teacher's distribution is worth more than its
   argmax, and it is measurable **before** a fold arm is launched. Its precondition is a no-fork
   control — the forensics' biggest MISSING cell, and now a second program has needed it.
2. **`--distill-topk` is the dial that matters, and its ordering is now explained** rather than
   just observed. The recorded late-generation full-KL re-entry path stays live, and it now has an
   entry condition instead of a hope.
3. **The mode-seeking direction is unbuilt and is the missing arm.** Every form here truncates the
   teacher's tail *before* the KL; nothing in the tree lets the student *decline* teacher mass it
   does not want. `KL(student ‖ teacher)` would transmit the mode while treating the tail as
   permission rather than obligation — the one target form whose behaviour this probe cannot
   predict from its own data.

---

## MISSING cells

| Cell | Reason |
|---|---|
| **`ai_v9_62_R2PLAIN`** | No `final_model.zip` at read time (run in flight). It is the ledger's pending disambiguation of *why* R2-CTRL declined −5.8, and its absence is why "benefit over nothing" rests on a control that is itself un-diagnosed. |
| **v8 no-fork control** | Never run. Without it, v8's tail cannot be scored for drift-noise, so the corollary stays MIXED rather than resolved. Inherited from the plasticity forensics' MISSING table; a second program has now needed it. |
| **v8 true action masks** | v8 traces record no mask. All v8 numbers are all-legal over 11 slots; the gen era's era-symmetric arm is the only honest comparison, and residual era asymmetry survives it. |
| **belief/intent drift at the FEATURE level** | The forensics hooked only `pokemon_encoder` / `team_transformer` / `projection` / `cls_pool` / `mlp.{policy,value}_net`. The M6 row is therefore **weight-level only**; a CKA/representation drift row for the belief and intent phases would need a fresh 7-model forward with new hooks, which the brief excluded. |
| **v8 opp_intent / opp_belief / pair_outcome / cf_ drift** | Those modules do not exist in v8. Structurally unanswerable, not a gap in effort. |
| **A dose-matched full-KL arm** | R2-KL ran at coef 1.0 (1.32× measured dose). The TOPK-vs-ACTION contrast is dose-clean and carries the conclusion, but a coef-0.1810 full-KL arm would remove the last confound from the KL cell. |
| **Advantage-weight ($w$) magnitudes** | `w = clamp(exp(\|Â\|/β), 20)` needs live rollout advantages and cannot be reconstructed offline, so the action arms' *effective* dose is bounded below (≥ nominal) rather than measured. Conservative in direction. |

## Reproduction

`tmp/dk_forward.py` (the two arms the forensics did not dump, + the acid test) and
`tmp/dk_analyze.py` (all five measurements) in this worktree; the two addenda
(`dk_addendum.py` = M1b fork specificity, `dk_onslice_abs.py` = M2b on-slice absorption) under
`/tmp/dkd/`. Every raw number is in the sibling `.json`. Reused read-only: `/tmp/plast/`
(the plasticity forensics' state sets and logit dumps).
