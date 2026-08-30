# ai_v12 CAPSTONE PROBE — risk modulation: does the win-probability value function buy correct gambling?

**Status: PRE-REGISTERED 2026-08-30 (owner-ordered: "toss this as a capstone probe of ai_v12
so we can see how effective the win probability value function is"). Predictions frozen here,
before any arm has trained. Edit only with evidence that predates the arms' data.**

## 0. The claim under test

Under a threshold objective, risk appetite is the curvature of P(win): convex below the
midpoint (gamble when behind), concave above (consolidate when ahead). An agent maximizing a
calibrated P(win) inherits correct risk modulation *for free* — no explicit risk system.
Material-flavored shaping dilutes the objective toward expected material, which is
risk-neutral exactly where correctness demands curvature. Therefore the clean world is a
risk-correctness experiment, not only a credit-assignment one, and risk modulation is a
*mechanism-level* readout of how much the win-prob value function is actually steering
behavior — complementary to the primary win-rate endpoints, and harder to fake.

## 1. The three instruments (all offline, existing tooling, no new battles required)

1. **The accuracy-tradeoff curve (primary).** From eval traces: decisions where a same-type
   power/accuracy pair is simultaneously legal (canonical gen3 pairs: Surf/Hydro Pump,
   Ice Beam/Blizzard, Thunderbolt/Thunder; membership resolved from `gen3_moves.json`, not
   hardcoded). Plot P(chose the inaccurate/high-power option) against the recorded win-prob
   at the decision. Read the SLOPE: falling = risk-correct; flat = risk-blind. Cluster
   bootstrap over battles (the Simpson trap rule); report per-pair curves beside the pooled
   one because base rates differ per pair.
2. **Explosion timing.** Boom frequency (Explosion/Selfdestruct chosen | legal) as a function
   of recorded win-prob, from the event log. Correct play booms from behind, almost never
   comfortably ahead.
3. **The general spread curve.** Per-decision, per-legal-action outcome spread via CRN
   rerolls (the falsifier's aleatoric instrument repointed: `reroll_many` at the decision,
   spread of V(s′) over dice per action). Correlate the CHOSEN action's spread with the
   recorded win-prob. This is the action-space-wide version of (1); run it on a subsample
   (it costs sim time; instruments 1–2 are trace-only).

**Guess-point companion (secondary, from the Nash note):** policy entropy at states the α/β
posterior flags as genuine guess points vs dominated states — state-*selective* entropy is
the healthy signature; uniform sharpness or uniform flatness are both findings.

## 2. Frozen predictions (per arm of the clean-world experiment)

| arm | prediction |
|---|---|
| **cw1_sparse** (±1 terminal only) | **Steepest accuracy-tradeoff slope** of the three arms — risk-correct by construction in the limit. Directionally negative slope REQUIRED for the claim to survive. |
| cw2_self_phi / cw3_frozen_phi | Slope between sparse and the shaped baseline. PBRS is policy-invariant in theory, so a *large* slope deficit vs sparse would itself be evidence the shaping is leaking policy pressure (a finding against the invariance implementation, not against the theory). |
| the shaped-world control (pt_shaped / gen-15 lineage) | Shallowest slope (most risk-neutral) — its objective is the most material-diluted. |
| all arms | Explosion curve agrees in sign with the accuracy curve (two instruments, one construct). |

**Falsifier honestly stated:** if the SPARSE arm's slope is flat (≈0 within its CI), the
"P(win) buys risk modulation for free" claim is WRONG at our scale/architecture — the
curvature exists in the objective but the policy gradient did not deliver it into behavior.
That would be a major, publishable-grade negative and must be reported as loudly as a pass.

**Confound named in advance:** move choice correlates with board state through channels other
than risk (PP, opposing typing, Water Absorb-class abilities). The accuracy-pair design
controls type by construction; residual confounds are why the slope (within-pair, across
win-prob) is the endpoint rather than the level, and why the spread-curve instrument (3)
exists as the assumption-free cross-check.

## 3. When it runs

After the ai_v12 arms complete (capstone = end-of-era battery, beside the primary endpoints
in `design_winprob_behavior_coupling.md`). Instruments 1–2 can be BASELINED NOW on gen-15
traces (the shaped-world row costs nothing and pins the pre-clean-world slope); doing so
before the arms finish keeps the comparison honest against drift in the trace format.

## 4. Cross-references

Theory: `designs/learning/temperature_mixing_and_risk.md` §3. Registration: ledger
2026-08-30 (capstone entry). Primary endpoints and arms: `design_winprob_behavior_coupling.md`,
`launch_runbook.md`. The knowing≠using precedent this probe exists to test against: the bait
verdict (head knew 96.4% of whiffs; credit convicted).
