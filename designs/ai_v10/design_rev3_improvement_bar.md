# design — REVOLUTION THREE: the IMPROVEMENT bar

**Status: AUTHORIZED (owner, 2026-08-27 evening) — "I want to see an improvement before we
deprecate anything that is now tech debt. Can you get an experiment where the bar is improvement.
We may need to both train exploiters more and have greater coverage." Pre-registered before any
rev-3 training begins.**

## 0. What changed from the capstone, and why this is simpler

Revolution two's bars divided by a control (R2-CTRL) that fell 5.8pp against a registered
prediction of flat — so every bar it passed measured "prevented a loss", not "added a gain". The
owner's direction replaces the opportunity-cost framing with an ABSOLUTE one: **the bar is
improvement over rev-1 final, measured directly.** No control arm, no denominator anomaly.
R2-PLAIN still runs/reads for the R2-CTRL diagnosis, but **it no longer gates revolution three**
(supersedes the capstone spec §4.2 gate — noted there).

**The DEPRECATION FREEZE (standing constraint, owner):** nothing the fold program has made
arguable tech debt — the full-distribution KL path, superseded distill flags, the old teacher
machinery — is deprecated or deleted until the improvement bar clears. Stabilization does not
license cleanup; improvement does.

## 1. Where the gain went in rev-2 — the accounting that sizes this design

Supply: five teachers at ~+12pp on-slice each, union-covering all 9 meter teams. Yield: +1.6pp
pooled vs the parent (n.s.). **Transfer efficiency ≈ 13%.** So the owner's two levers (supply)
are necessary but arithmetically not sufficient: at 13% efficiency, +14pp teachers over 12 slices
still lands ~+2–3pp — marginal against z≥2. The design therefore moves BOTH sides:

- **Supply** (owner's levers): per-team budget 1.5M → **2.5M** (the budget law's next prospective
  point; v8's forks ran 1.2–2.5M/team) and coverage 9 → **12 slices**.
- **Transfer** (the indicted step): fold length scaled to coverage (+3M → **+4.5M**), plus a
  HIGHER-DOSE hedge arm — the per-slice-dose hypothesis (banked with the R2-KL anomaly) says the
  same total dose spread over more slices teaches each slice LESS, so more coverage at fixed dose
  may self-dilute.

**This is a DEMONSTRATION run, not an ablation.** Several levers move at once, deliberately: if
the bar clears, attribution among them is confounded and we do not care; if it fails, the
accounting (admission rows = supply, pooled yield = transfer) localizes which side fell short.

## 2. The fleet (tock-3.x)

**Six exploiters, tock-1c's shape otherwise:** each forks rev-1 final, **+5M steps** (2.5M/team),
`--exploiter` vs **R2-ACTION's frozen final snapshot** (the wheel's product is the new target —
this is what makes it revolution three rather than a re-run), 2 pinned teams each.

**12 slots = the 9 meter teams + 3 NEW coverage teams.** No deliberate overlap slot this time —
the consistency question was rev-2's measurement. New-team selection rule (training session
executes): from the pool, the 3 teams with the lowest rev-1 piloting win rate among teams NOT in
the meter and NOT the 2 held-out, at most one per archetype class (`gen3_team_archetypes.json`).
The 2 held-out teams remain untouched by any pin — they are the narrowness instrument.

**Admission:** the standing 800/arm matched row per teacher ON THE NEW TARGET (R2-ACTION final +
rev-1 references) + the off-slice informational row. Gate unchanged: on-slice net > 0 at z ≥ 2
pooled per slice; failures logged SLICE UNCOVERED, never silent.

**FLEET RIDER — the C1 CURRICULUM DISCRIMINATOR (added 2026-08-28, probe C, pre-launch):** probe
C found the biggest un-tested era difference is the EXPLOITER'S OPPONENT REGIME — v8 forks ran
`--exploiter-keep-bots` (50% scripted bots) plus a WR-ratcheted difficulty curriculum
(`exploiter_temp_start 5.0`, ratchet mode; run-dir artifacts prove it completed), while every gen
fork faces one full-strength near-mirror from step 0 — sitting at WR≈0.5 vs its own parent,
where the advantage signal is weakest and the update is dominated by entropy/value/drift
(matching the forensics: fork shift ≈ no-fork-control shift). **One extra fleet arm, F6-CURR:**
same 2 pinned teams as one standard arm, +v8's curriculum flags, same +5M — **MEASUREMENT-ONLY,
never a fold teacher** (a duplicate-pin teacher would re-create the two-masters problem AND the
silent duplicate-pin bias defect probe C found). Registered prediction: F6-CURR shows the on/off
differentiation split the standard forks lack, and its admission row ≥ the standard arm's. ~2.7
GPU-h; the cheapest possible causal read on the top-ranked binding-constraint candidate.

**FOLD RECIPE RESTORATION (C2): all R3 fold arms take `--team-block-episodes 64`** — v8's fold
ran 64, every gen fold ran 1 (the worst case for learning a team-CONDITIONAL mapping; same shape
as the FiLM sample-starvation finding). Applied to ALL arms so within-rev-3 comparisons stay
valid; noted as a deliberate rev-2→rev-3 recipe change (demonstration, not ablation). Also at
argv freeze: read the existing `distill/*_value_feat_cos` TB scalars — no gen fold has ever run
ai_v8_14's literal recipe (it had `--distill-value-feat-coef 0.0`; every gen fold adds 0.5) —
and decide 0.0-vs-0.5 from the scalars, not from habit.

**Admission adds a TAIL-SPECIFICITY column (probe D):** `inter-fork tail cosine − fork-vs-control
tail cosine` — a seconds-cheap read of whether a teacher's DISTRIBUTION carries content beyond
its argmax (rev-2 fleet: +0.021, i.e. the tails were drift). Its no-fork control is R2-PLAIN
(ai_v9_62) once final. Informational; also the instrument that would someday re-license full-KL.

**Admission adds a DIFFERENTIATION row (2026-08-28, forensics-forced):** per teacher, argmax
agreement with its parent on-slice vs off-slice (the forensics instrument — CPU, recorded
states). The forensics found v8's teachers DIFFERENTIATED (changed more on their own slice:
0.417–0.519 agreement on-slice vs 0.543–0.577 off) while the R2 fleet did NOT (flat 0.69–0.77
everywhere) — an undifferentiated global shift may be why there was so little slice-conditional
content to fold. Registered reading: differentiation predicts foldable content; its absence
predicts the fold reduces to anchoring (see R3-SELF). Informational this revolution — it becomes
a gate only if rev-3 confirms the correlation.

**Registered predictions (fleet):**
- **Budget law, next point:** the measured curve ends at 1.5M/team → +0.1162. If monotone
  continues, 2.5M/team rows land **+0.13–0.16**; a plateau at ~+0.12 is equally informative (the
  knob's ceiling found). Scatter (±0.05) still kills the law.
- **Headroom row (new, free):** teachers vs the NEW target measure which gaps R2-ACTION closed —
  the first per-team headroom reading of the optimization era, a year early. No prediction; the
  row itself is the finding.

## 3. The fold (two arms, no control)

Both fork **R2-ACTION final** (compounding — if R2-PLAIN says continuation-is-costly, this is also
simply the strongest available base), +4.5M, the admitted fleet as teachers, action-form target in
**the same gate configuration R2-ACTION shipped with**, `--distill-team-bias 0.4` over the 12
slices, `--rank-tripwire warn`.

| arm | dose (`grad/distill_share`) | role |
|---|---|---|
| **R3-ACTION** | ≈ 0.24 (the calibrated standard) | **PRIMARY** |
| R3-ACTION-HI | ≈ 0.35 | the transfer hedge — tests the per-slice-dilution hypothesis from the supply side |
| **R3-SELF** | ≈ 0.24 | **the ANCHOR CONTROL (added 2026-08-28, forensics-forced):** distill target = the FROZEN parent (R2-ACTION final) itself — pure self-anchoring, no exploiter content (Learning-without-Forgetting shape) |

**Why R3-SELF exists (the drift-anchor hypothesis, from the plasticity forensics):** R2CTRL's
function drift sits INSIDE the fork range (KL 0.3245 vs forks' 0.269–0.349) — at this
plasticity, 3M of ANY training moves the policy that far, which both explains the R2-CTRL −5.8pp
anomaly (undirected drift is costly) and raises the possibility that the rev-2 fold's entire
+7.4-vs-control was ANCHORING (five rev-1-descended teachers averaging to "stay near rev-1"),
not content transfer. R3-SELF separates them: **content = R3-ACTION − R3-SELF.** If
R3-ACTION ≈ R3-SELF, the flywheel reduces to a self-anchor — which would itself be a major,
much cheaper discovery (no exploiters needed to prevent drift-decline), but is not the flywheel.

Multiplicity, pre-registered: the primary claim is R3-ACTION's. If only R3-ACTION-HI clears the
bar, the improvement claim stands WITH the two-arm multiplicity caveat quoted, and 0.35 becomes
the recipe of record for the confirmation turn. R3-SELF is a control, never a claimant.

**Additional registered predictions (2026-08-28, post-forensics, pre-launch):** R2-PLAIN lands
≈ R2-CTRL (≈ −6pp) — under the drift account plain continuation drifts and declines the same;
R3-SELF ≥ 0 vs rev-1 baseline-region (anchoring prevents decline); the capstone-relevant number
is R3-ACTION − R3-SELF > 0 at z ≥ 2 — that difference, not bar 1 alone, is what licenses the
word "flywheel". **Probe B prior (observational):** content is real (ZapDug DiD ≥+4.0pp
teacher-specific agreement, confound-free) but anchoring + practice carry most of the rev-2
fold-vs-control gap — so R3-ACTION > R3-SELF is expected by a MODEST margin; a null there
contradicts probe B and would itself be a finding.

**BIAS-PARITY ORDER (2026-08-28, probe B's confound discovery):** `--distill-team-bias` is inert
when `_distill_pairs` is empty (coef 0), which silently un-matched rev-2's control. For rev-3:
**R3-SELF must carry real distill pairs** — the frozen parent as teacher bound to ALL 12 slice
teams (`PARENT:t1,…,t12`) at coef 0.24 — so its team bias is effective and identical to
R3-ACTION/HI's. Every arm's EFFECTIVE bias is verified from telemetry (team-draw counts), never
from the argv. A config fix making bias-at-coef-0 work as recorded (pairs built whenever
`--distill-teacher` is given; teacher LOADING still skipped at coef 0) is being landed with a
loud guard + regression tests — R3 argvs re-checked with `checkargs` after it lands.

## 4. Bars (pre-registered)

Verdict meter: the standing 9-team piloting meter at **n = 500/team** (tightened from 300 — CI
~±1.9pp; the expected effect is small and the CPU is cheap) + the 2 held-out teams + the 3 new
coverage teams (informational rows), paired draws throughout.

1. **THE IMPROVEMENT BAR: R3-ACTION − rev-1 final > 0 at z ≥ 2, pooled over the 9 meter teams.**
   Absolute, control-free — the owner's bar.
2. **Compounding: R3-ACTION ≥ R2-ACTION** (pooled, point estimate; a significant regression vs
   R2-ACTION fails the run even if bar 1 squeaks).
3. **No narrowness: held-out 2 teams ≥ rev-1 final − 2pp.**

**RIDER — the transfer-efficiency decomposition (added 2026-08-27 late, owner exchange).**
Rev-2 converted ~+12pp/slice of teacher supply into +1.6pp pooled (~13%), and no measurement says
where the rest went. The verdict battery therefore adds **post-fold student–teacher ACTION
AGREEMENT per slice**, on recorded slice states, anchored by each-vs-rev-1 (the
shared-inheritance floor — the ZapDug protocol's trick, reused): HIGH agreement + small gain ⇒
the copied decisions were not where the value was (opponent-specific content, or ceiling) — more
dose will not help; LOW agreement ⇒ the channel is undertrained or the teacher's lines reach
states the student never visits (the imitation distribution-shift account) — dose/duration is the
right lever. This converts §4's failure branch from inference to measurement. CPU-only.

**CONDITIONAL FOLLOW-UP — the BLOCK-LENGTH ablation (owner order, 2026-08-28 late):** IF rev-3
shows a real content signal (gate: R3-ACTION − R3-SELF ≥ +4pp pooled — below that a modifier is
statistically invisible at n=500/team, where a two-arm difference resolves ~±2.4pp), THEN run
**R3-BLOCK1**: identical to R3-ACTION except `--team-block-episodes 1`, same fleet, same dose
(~2.5 GPU-h). The owner's registered reasoning: the block knob MODIFIES the conditional-learning
component, so its effect is only resolvable in proportion to the main improvement signal —
measure the modifier where the signal is strong, never where it is absent. Registered
predictions: R3-ACTION − R3-BLOCK1 > 0, and the student's PER-TEAM behavioral spread (the
differentiation instrument) higher under block-64 — the second prediction is the mechanism
check (batch-composition SNR), the first is the payoff check. A block LADDER (16/64/256) only
if the single ablation reads positive.

**HOLDS** = all three ⇒ the flywheel ADDS gain; the deprecation freeze LIFTS; the confirmation
turn (revolution four, same recipe off R3's product) is pre-committed before any "flywheel works"
claim — the wheel-turns-twice commitment transfers here unchanged. **FAIL** ⇒ read the
accounting: admission rows strong (+0.13-ish) but yield flat indicts TRANSFER — the next
experiment is fold-side only (dose ladder above 0.35, per-slice scheduling, fold sequencing) on
this same fleet, no new exploiter training; admission rows themselves flat at 2.5M/team indicts
SUPPLY (the budget law's ceiling) and the lever moves to team selection/headroom. Both branches
keep the freeze in place.

## 5. Schedule (~one box-day)

| when | what |
|---|---|
| now | R2-PLAIN reads out (diagnostic only); fleet argvs built + checkargs |
| overnight | fleet: 6 × 5M ≈ 30M steps serial GPU (~16h) |
| day AM | admission on the new target (CPU); fold argvs frozen; R3 arms (~2.5h each) |
| day PM | n=500 meter battery; verdict against §4; ledger + owner report |

## 6. Bookkeeping

Run dirs `ai_v9_6x_R3*`; probes to the training session's `probes/` convention; landings flow
through the ideation session. The capstone spec's §4.2 gate is marked superseded by this design
in the same pass.
