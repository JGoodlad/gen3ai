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
