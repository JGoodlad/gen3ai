# design — FLYWHEEL REVOLUTION TWO: the capstone proof-of-concept

**Status: AUTHORIZED (owner, 2026-08-26 evening) — two days of box time. Pre-registered before any
fleet training begins.**

## 0. The claim under test, in one paragraph

One full flywheel revolution — a fleet of narrow exploiters (tock) folded into the live generalist
with the validated action-form channel (tick) — produces a generalist measurably better than the
same generalist given the identical training budget with no fold. This is the capstone of the
ai_v10 era: if it holds, the era closes and the next era is flywheel OPTIMIZATION (team
synergies/anti-synergies, exploiter team selection, per-team PFSP biasing toward hard teams,
exploiter-headroom ranking — all meaningless until a revolution demonstrably pays). If it fails,
the failure names the missing ingredient: every channel question was settled this week, so a
capstone failure indicts CONTENT AGGREGATION — the union-coverage assumption — and that too is
decision-grade.

**The one assumption imported from v8, tested here for the first time: folding several
VERIFIED-TEAM-LOCAL teachers produces union-wide gains.** Precision matters here (owner question,
2026-08-26 evening): v8's fold WAS a union — 3 teachers / 23 teams — so union folding per se has a
precedent. What v8 did NOT establish is this cell: its teachers were never transfer-tested, and
the plasticity account predicts they were NOT team-local (a fork off a rigid 276M parent
specializes without renovating, so it plausibly keeps its general competence — transfer-positive).
Ours are measured local (−8 to −10pp off-slice, same opponent), the student is plastic, and the
channel is action-form. v8 is encouraging precedent for a DIFFERENT cell, not evidence for this
one.

**REGISTERED PREDICTION — the per-team budget law (owner question, same exchange):** across the
four admitted teachers, net extraction tracks STEPS-PER-TEAM monotonically (0.75M → +0.0825 ·
1.0M → +0.0875 · 1.0M → +0.0875 [tock-2.0, 3× budget AND breadth, identical row] · 1.5M →
+0.1162), suggesting breadth never diluted anything — a fixed budget divided more ways did. The
fleet sits uniformly at 1.5M/team, so the law predicts **every F5x admission row lands near
+0.11**. Scattered rows (±0.05) kill the law and mean slice CONTENT matters; clustered rows
confirm it and make teams-per-exploiter a pure cost knob (~1.5M steps per team) for the next era.

## 1. Decisions of record folded into this spec

- **#2b RESOLVED (owner direction, 2026-08-26): union-coverage gate.** Teacher admission = on-slice
  extraction (the standing 800/arm matched harness) + a SET-level requirement that the fleet's
  slices union-cover the meter teams. The off-slice transfer row is measured and recorded per
  teacher as an INFORMATIONAL row (it is the never-query-off-pin constraint's evidence), not a
  per-teacher veto.
- **PCGrad is NOT an arm.** F1c falsified its premise (KL harms with no opposing gradient;
  measured interference Δcos −0.030 is near-orthogonal — projection would pass ~97% of the
  gradient through). REGISTERED PREDICTION, so the shelf decision is auditable: PCGrad+full-KL
  would land within noise of fdB's −7.5pp. It runs only if someone wants to falsify that sentence.
- **The third arm is the CONTENT dose-response (top-K), serving the recorded long-term aspiration**
  (owner #1 amendment: return to full distribution eventually): K interpolates the convicted
  dimension, and the largest safe K is the far end of the bridge.

## 2. The fleet (tock-2.x)

**Five fresh exploiters, tock-1c's shape** (the best measured extractor, +0.1162): each forks
rev-1 final, +3M steps, `--exploiter` vs the frozen rev-1 24M snapshot (same target as every
admitted teacher — holds the opponent axis fixed for comparability), **2 pinned teams each**.

**Slice assignment (10 slots = 9 meter teams + 1 deliberate overlap):**

| exploiter | teams |
|---|---|
| F5a | ZapDug · JynxSO |
| F5b | RaikouCelebi · MixZap |
| F5c | BlueOffense · MedichamCune |
| F5d | CBMetaCroCune · Q6a |
| F5e | Q6b · **ZapDug (overlap)** |

The ZapDug overlap is deliberate: two independently trained specialists sharing one slice gives the
first WITHIN-FLEET CONSISTENCY measurement (do two exploiters agree on how to pilot the same team —
extraction row F5a-on-ZapDug vs F5e-on-ZapDug, and their action agreement if cheap). No arm has
ever measured whether exploit content is convergent or idiosyncratic.

**Admission per teacher:** the standing 800/arm matched row (target + rev1final references, shared)
on its 2 teams + the off-slice informational row (2 held-out pool teams, the transfer-gate
instrument, its OFF_PIN_SEED convention and resolved-count asserts). Gate: on-slice net > 0 at
z ≥ 2 pooled over its slice. A teacher that fails is dropped and its slice noted UNCOVERED — the
fold proceeds with honest coverage accounting, never silent.

## 3. The fold battery (four arms, one base)

All arms fork rev-1 final, +3M, the admitted fleet as teachers, on-pin team bias as fdB
(`--distill-team-bias 0.4`), dose-calibrated to `grad/distill_share` ≈ 0.24 (the fdB-calibrated
target; null-dose band [0.119, 0.476] checked at 6 points), `--rank-tripwire warn` (activity
detector only — never gates a verdict, per the 2026-08-26 §6.4 amendment).

| arm | loss | purpose |
|---|---|---|
| **R2-ACTION** | `--distill-target action` (gate per G1′: ungated if G1′ ≈ G2, else gated) | **THE CAPSTONE ARM** |
| R2-TOPK | `--distill-target action --distill-topk 3` | content dose-response — where does distribution-matching turn toxic |
| R2-KL | full-distribution KL (today's default) | reproducibility control — the harm must replicate on a fresh fleet or the week's story is teacher-specific |
| R2-CTRL | no distill, same +3M | **the opportunity-cost baseline** — the counterfactual "just keep training" |

**Registered predictions (before any arm runs):** R2-KL −7±2pp pooled (replicates fdB/F1c);
R2-CTRL ≈ 0 (fdC's −1.2 n.s. band); R2-ACTION > R2-CTRL (the capstone bar, §4); R2-TOPK between
R2-ACTION and R2-KL — where it lands is the finding, not a success/failure.

## 4. Success bars (pre-registered)

The capstone claim **HOLDS** iff, on the standing 9-team piloting meter (n=300/team, paired draws)
plus the 2 held-out teams:

1. **R2-ACTION − R2-CTRL > 0 at z ≥ 2 pooled over the nine union teams** (the fold beats the same
   budget spent training normally — the flywheel's entire value proposition in one number);
2. **No off-slice regression:** R2-ACTION on the 2 held-out teams ≥ R2-CTRL − 2pp (the fold must
   not do to the generalist what specialization did to the teachers);
3. **Union-wide, not slice-local:** ≥ 6 of 9 team rows non-negative vs R2-CTRL (a fold carried by
   two teams while damaging four is tick-1's failure with better marketing).

**PARTIAL** (recipe works, aggregation doesn't): R2-ACTION > R2-CTRL on covered slices but bar 3
fails ⇒ the union assumption is wrong as stated; the next design iterates on content mixing (e.g.
per-slice coefficient, fold sequencing), not on the channel. **FAIL** (R2-ACTION ≤ R2-CTRL): the
G1 result does not survive scale-up to 5 teachers — the teacher-count axis re-opens at the fleet
scale (fdE tested 1-vs-2, never 5).

**The wheel-turns-twice commitment:** if the capstone HOLDS, revolution THREE (a fresh fleet
targeting R2-ACTION's product, folded the same way) is the confirmation run — pre-committed here
so one good revolution is never reported as "the flywheel works". The flywheel claim needs
compounding, or at least repetition.

## 5. Schedule (owner-authorized two days)

| when | what |
|---|---|
| tonight | G1′ (gate cell) + F2c (reversibility) land — G1′ sets R2-ACTION's gate flag |
| overnight | fleet trains (5 × 3M ≈ 8h serial GPU) |
| day 1 AM | admission + off-slice rows (CPU eval workers, niced); fold argvs frozen |
| day 1 PM → night | four fold arms (≈ 2h GPU each) + their meters |
| day 2 | full meter battery, verdict against §4, ledger + owner report |

## 6. Bookkeeping

- The era after this one (flywheel optimization: synergies, exploiter team selection, per-team
  PFSP hard-team biasing, headroom ranking) is NOT `ai_v11` — that number is the human-replay
  chapter. It gets the next free number when it opens.
- All run dirs `ai_v9_5x_R2*`; probes to the training session's `probes/` convention; nothing
  under `designs/` is written by the training session — landings flow through the ideation
  session per the standing division.
