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

Multiplicity, pre-registered: the primary claim is R3-ACTION's. If only R3-ACTION-HI clears the
bar, the improvement claim stands WITH the two-arm multiplicity caveat quoted, and 0.35 becomes
the recipe of record for the confirmation turn.

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
