# PRE-REGISTRATION — THE K LADDER (`ai_v13_17/18/19_fold_k{1,3,11}`)

**Committed BEFORE any K-ladder number exists.** `ai_v13_17_fold_k1` launched 13:30 PT 2026-09-21
and is the only arm running; K=3 and K=11 have not launched. The GO is the orchestrator's stage-B
specification of 2026-09-21 and the ledger entry `cc02cff5`.

---

## 0. What this ladder is, and the ONE sentence that must travel with every number

Three folds from **THE PLATEAU PARENT** (`models/ai_v13_12_plateau/final_model.zip` @95,158,272,
the checkpoint whose untaught-8 gain is exhausted — `3459ecce`), +12,000,000 steps each to
**107,158,272**, seed 1001, pin `6eb9c776`, `--fork-lr 2.8e-5 --fork-lr-freeze` (0.20×, fold-1's own
dose; all three match EACH OTHER, which is what the contrast needs). Teacher in all three:
**`ai_v13_13_exploit5_offense`**. Fold-1's FULL ecology, carried verbatim from its own argv:
`--distill-team-bias 0.4`, the teacher in `--stable-opponents`, **`--team-block-episodes 64`**,
`--distill-target action`, `--distill-gate none`, `--distill-coef 0.1761`, `--distill-beta 1.0`
(NAMED, not inherited — the standing rule of `bbb1f228`).

**THE ONLY DIFFERENCE BETWEEN THE THREE ARMS IS `--distill-topk` = 1 / 3 / 11.**

🚨 **THE TEACHER HAS NOTHING TRANSFERABLE BY THE GATE.** `ai_v13_13_exploit5_offense` reads
**−1.00 pp [−4.80, +2.13]** against this same parent on its own five teams (`be03bd74`), and the
redesigned distribution-trained teacher reads −1.37 (`cc02cff5`) — all four teachers this campaign
built were refused. **So this ladder is NOT a test of whether distillation gains anything.** That
half is already answered: there is nothing to gain. It measures **the COST of each target form on
the untaught 8, and whether widening the target removes it.**
**A flat untaught row on these arms means "the target form was cheap", NEVER "distillation was
neutral."** Any later reading that makes the second claim is misreading the design, and this
sentence is why it cannot be done innocently.

## 1. The rows, the refs and the depths

**ROW 1 — the untaught 8** (off-slice competence; the campaign's primary endpoint).
**ROW 2 — the OFFENSE slice**: the teacher's own five teams, the same manifest
(`…/tmp/admission/teams_offense.json`) and order used by every admission cell.

Depths **+3M / +6M / +12M** past the fork (98,158,272 / 101,158,272 / 107,158,272 nominal; the
realized checkpoints are whatever `checkpoints/` holds nearest, named in the README when read).
Baseline for both rows: **the plateau parent**, measured IN THE SAME INVOCATION as the arms it is
differenced against (CRN), never imported.

Harness: reused VERBATIM from `split_lossoff_read_2026-09-20/scripts/harness/` — registry opponent
`untaught_meter_opponent` (= `ai_v9_29_rev1_0823@24,000,000`), registry config, `--seed 0`,
concurrency 1, nice 15, CPU-only; row 1 at its default games/team, row 2 at `--games-per-team 800`.
Paired bootstrap over TEAMS (rule 10), 20,000 draws, one shared index set, seed 20260915.

**THE ARM-W REPRODUCTION CHECK RIDES IN ROW 1**: arm W must return **46.19 pp (739/1600)** with all
eight per-team rows identical. A failure makes the read INVALID, not patched — report and stop.

## 2. Floors, fixed now

| row | floor | provenance |
|---|---:|---|
| untaught 8 | **3.69 pp** | \|arm W − W_b\|, the 75M fresh-arm seed pair (IMPORTED) |
| offense slice | **4.75 / 8.50 pp** | the 2026-09-20 read's two slice cells; the **8.50** bar is the headline, since neither floor was measured on THIS five-team slice and the stricter one is the only honest choice |

**OUTSIDE** iff |Δ| > floor **AND** the 95 % CI excludes the floor point. Anything else is
**NOT DETECTED at the registered standard** — never "refuted", never "no effect".

## 3. The three pairwise contrasts, registered before the data

All at the **+12M** depth (the branch-carrying point), with +3M/+6M reported as trajectory:

- **C1 — K=11 vs K=1** (the headline): does the FULL renormalized target cost less off-slice than
  the argmax one? Fold-1's K=1 cost ≈10 pp on the untaught 8.
- **C2 — K=3 vs K=1**: is the defensible middle already enough?
- **C3 — K=11 vs K=3**: is the remaining tail worth anything, or does K=3 capture it?

**Registered expectation, so hindsight cannot improve it:** the *direction* we expect is
C1 ≥ C2 ≥ 0 — wider targets cost less — because a top-1 CE discards the teacher's tail shape and
concentrates the student's policy (the split arm measured exactly that concentration: entropy
0.7373 under the loss vs 0.9407 without it, `f389fce4`). **We do NOT predict any of the three is
OUTSIDE its floor.** A ladder in which all three contrasts read NOT DETECTED is a real result and
will be banked as one.

## 4. The branches, fixed now

1. **C1 OUTSIDE and positive** (K=11 costs less than K=1) ⇒ the target form is a live lever;
   the plain `--distill-target kl` arm becomes worth running as the FOURTH point, to separate the
   target form from the AWR weighting that K=11 still carries (see §5).
2. **C1 NOT DETECTED** ⇒ 🚫 **no `kl` arm.** Widening the target does not pay at this dose, and the
   loss-form question is answered in the negative for this parent and teacher.
3. **Any arm's untaught row WORSE than the parent OUTSIDE the floor** ⇒ that target form carries a
   real off-slice cost; report the cost with its CI and do not fold that form again here.
4. **The arm-W reproduction check fails** ⇒ the whole read is INVALID. Report and stop.

## 5. The caveat that must be quoted wherever K=11 is called "full distribution"

🚨 **K=11 is FULL-DISTRIBUTION TRANSFER, AWR-WEIGHTED.** `ACTION_SPACE_SIZE = 11`
(`src/agents/action/constants.py:3`), so at K=11 the top-K selection leaves the softmax untouched and
the per-row TARGET is identical to the plain `kl` term's. **The aggregation is not**:
`_gated_action_distill_loss` keeps the AWR row weight `w = clamp(exp(|Â|/beta), 20)`, and its own
docstring's identity holds only "when every Â is 0". So K=11 is the full distribution **in the
AWR-weighted family**, and is NOT today's `kl` term. That is exactly why the `kl` arm is held as a
conditional fourth point rather than folded into the ladder: it would move two things at once.

## 6. What this ladder cannot say

- Nothing about **gain**. The teacher has nothing transferable (§0).
- Nothing about **teamset size or archetype** — one teacher, one archetype, one set.
- Nothing about **dose**: all three run at 0.20×, and the campaign's standing caveat is that
  era-1 ran teachers at 1.78×. A cost measured at 0.20× is a cost at 0.20×.
- Nothing about **strength**: the untaught meter RANKS arms; its pp are not external pp, and a
  strength claim carries the anchor's number at ≥1,200 games or is not made.

---

## 7. AMENDMENT 1 — a free secondary, registered before K=3 and K=11 exist

**Committed 2026-09-21 ~19:05 PT, while `ai_v13_17_fold_k1` is still running and neither other arm
has launched.** K=1's distill-stop detector **FIRED** and the firing is itself a measurement nobody
budgeted for.

**What happened.** `🛑 [DISTILL-STOP] FIRED at 103,415,808 steps: distill/teacher_agreement_on_slice
EMA 0.8754 has PLATEAUED (improvement over 8 rollouts < 0.005) while
distill/collateral_kl_vs_parent EMA 0.32418 is still RISING, for 3 consecutive rollouts.`
**`mode=warn`, so training continues unchanged** — the arm still runs the full +12M with the loss
on, exactly as fold-1 did, and no number in §1–§6 is affected.

**Two facts fall out, and both are free:**

1. **The fold's own optimal length is +8,257,536 past the fork** (103,415,808 − 95,158,272), so the
   registered **+12M endpoint sits ~3.7M steps PAST it** while **+6M sits before it**. The
   pre-registered trajectory depths therefore BRACKET the stop point by construction — which was not
   designed, but means **+6M vs +12M on the untaught row is a direct read of what running past the
   stop costs.** The ledger's standing note is that v8 lost ~5 pp of untaught win rate doing exactly
   that (2026-09-01). Fold-1's detector fired at +7.67M, so K=1's optimal length is ~0.6M longer.
2. **K=1's agreement saturates at 0.8754**, against the era-1 fold's ≈0.82–0.83 — higher, on one
   teacher expanding to five teams where era-1 had two teachers expanding to two.

**REGISTERED SECONDARY (S1), fixed now for all three arms:** record each arm's **stop-rule firing
step and its `teacher_agreement_on_slice` EMA at firing**. The question: **does widening the target
delay the plateau or raise it?** A K=11 arm that plateaus LATER or HIGHER than K=1 is evidence the
wider target still had something to teach when the narrow one had run out.

🚨 **S1 IS A DESCRIPTOR, NOT A CONTRAST.** It has no floor, no CI and n = 1 per arm; the detector is
an EMA over a noisy ratio and its firing step is not a measurement with a replicate level. **It can
motivate a contrast; it can never settle one.** The verdicts in §3–§4 are unchanged and S1 does not
enter them. Registered here only so the three numbers are collected under one rule instead of being
noticed after the fact on whichever arm happens to look interesting.

---

## 8. AMENDMENT 2 — 🚨 THE LADDER VARIES TWO THINGS, NOT ONE. Registered BEFORE any untaught number exists.

**Committed 2026-09-22 ~06:05 PT: K=1 and K=3 have finished TRAINING, K=11 is running, and NO row of
§1 has been measured on any of them.** What follows is a training-side confound found in the
descriptors, not a result.

**THE FACT.** At the identical `--distill-coef 0.1761`, the distill term's share of the gradient
norm — `grad/distill_share`, the project's own §6.2 dose meter — differs almost 2× between the rungs:

| arm | mean | median | first 10 rollouts | last 10 |
|---|---:|---:|---:|---:|
| K=1 | **0.2498** | 0.2413 | 0.2676 | 0.2474 |
| K=3 | **0.1291** | 0.1209 | 0.1724 | 0.1116 |

n = 119 rollouts each, so this is a stable property of the arms and not sampling noise.

**WHY IT HAPPENS, and why it was foreseeable.** A top-1 target is a one-hot cross-entropy; a top-3
renormalized target spreads the same probability mass over three actions, so the per-row gradient is
smaller in norm for the same coefficient. **The coefficient is not the dose.** The campaign already
knows this in writing — the `--distill-target` help text says to watch `grad/distill_share` and the
design's own rule is that *a distill coefficient is set by `grad/distill_share`, never by eye* — and
the ladder was nevertheless built at fixed coef across K. That is on me: §1–§6 were written as
"one lever, K" and the sentence was wrong.

**WHAT THE LADDER THEREFORE MEASURES.** Not "the cost of a target form at matched pull", but **"the
cost of a target form at a fixed coefficient, pull included"**. Both are legitimate questions and
the second is the deployable one — nobody retunes coef per K in practice — but they are different,
and only the second is registered here.

🚨 **THE DIRECTION OF THE CONFOUND IS KNOWN, AND IT RUNS TOWARD THE EXPECTED HEADLINE.** §3 predicted
C1 ≥ C2 ≥ 0 — wider targets cost less. Wider targets ALSO pull less hard. **So a result of the
registered shape is confounded and may be explained entirely by the weaker pull; it does NOT isolate
the target form.** The asymmetry matters: a "wider costs less" finding is weak evidence about target
form, whereas **a "wider costs MORE despite pulling ~2× less" finding would be strong**, because the
confound runs against it.

**WHAT CHANGES, AND WHAT DOES NOT.**
- **No verdict in §3–§4 is altered** — C1/C2/C3 and the floors stand exactly as registered.
- **ADDED to the reported set, per arm:** `grad/distill_share` (mean, median, first-10, last-10).
  **No cost number from this ladder may be quoted without it.**
- **ADDED to §4 branch 1:** if C1 reads OUTSIDE and positive, that result is **NOT** attributable to
  the target form alone, and the follow-up that separates them is a K=11 arm at a coefficient raised
  to match K=1's share (~0.25), not the plain `kl` arm. The `kl` arm answers the aggregation
  question and is still conditional on C1; the share-matched arm answers the pull question. They are
  different fourth points and must not be conflated.
- **K=11 is NOT being stopped or rebuilt for this.** It is mid-flight, the three rungs must stay
  mutually matched in everything except K, and re-running two rungs at adjusted coefficients costs
  ~16 GPU-h to answer a question the registered read does not ask. That is an orchestrator call, not
  mine, and it is flagged to them with this amendment.
