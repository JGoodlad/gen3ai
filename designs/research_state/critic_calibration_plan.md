# THE CRITIC-CALIBRATION PLAN — the era's binding constraint, attacked in layers
*Written 2026-08-24, the day four independent instruments converged on the same object. This is
the standing plan; amendments require new evidence, stated beside the edit.*

## 0. The case file (what "the critic is biased" precisely means — all measured)
- **RESOLUTION, not level**: population-mean gaps are small and sign-flip by ecology (−0.065 vs
  bots, +0.058 vs pool); the disease is BLUR — within-decile true spread 0.11–0.36. Meter:
  `sd_true_excess`, floor-subtracted. (G0.)
- **~39% of conviction-region blur is IRREDUCIBLE** hidden-info floor, concentrated in ~10–20%
  of states. No value head can remove it; only belief/revelation can. Quote effects on the EXCESS.
- **Credit mispricing at rare states**: the advantage signal actively defends bait-class
  mistakes (falls-then-reverts, code-matched). On-policy exploration cannot fix it.
- **Bias, not variance**: the R-ladder (32× averaging, flat) proves search amplifies a
  SYSTEMATIC leaf error; the α opponent-model carries its own bias term (narrow-width cells
  are worse; α ratio 0.97 flat).
- **The first fix attempt taught its own lessons** (R1 read, replicated): tight-MC precision
  WORKS (C−B −0.036); surprise-prioritized sampling HURTS (B−A +0.065 — selection bias,
  almost certainly the classic prioritized-replay defect: priority without importance
  correction); the self_current estimand is +0.13 optimistic; label heads re-centred without
  sharpening (the wrong-meter trap, caught by the meter built for it).
- **Room exists**: critic rank re-expands under richer targets (plasticity null); residuals
  are sub-Gaussian (no tail trick available — the distributional lever is dead, the
  parameterization already adequate).

## 1. THE METERS (all built; every arm reads against all four)
`sd_true_excess` (floor-subtracted, per-population) · the MIRROR TABLE (behavioral resolution;
a cell crossing 0.50 is the wake-search signal) · capacity `value_pooled` PR (richer targets
should RAISE it off the 2.5 steady state) · the §2 paired-head read (per-component attribution).

## 2. LAYER 1 — THE v2 LABEL FACTORY (rev-2; the R1 read's direct descendants)
1. **Estimand fix**: thread OPPONENT IDENTITY through the training tap so rollouts play the
   actual opponent class, killing the +0.13 self_current optimism. (The runbook named this
   day one; it is now measured, not hypothetical.)
2. **Sampler fix**: replace surprise-priority with uniform-over-states OR priority WITH
   importance-sampling correction (the PER lesson). The B−A contrast is the arm-vs-arm gate.
3. **Variance allocation**: spend R on OPPONENT-ACTION draws, not dice (behavior-weighted
   opponent share of target variance = 59.7% vs dice 26.5% — the three-axis ordering).
4. **Dose**: the warm path sustains 600–900/h ⇒ 15–20k labels/run. Tick-1 is replicating the
   v1 factory at dose FIRST (pre-registered; do not conflate the dose answer with the v2 fixes).

## 3. LAYER 2 — THE OBJECTIVE (staged promotion, each stage gated by the meters)
heads-only twins (running) → **win-prob TRUNK-OPEN** (`--cf-winprob-coef`, the §1 grad-share
ladder) only after a POSITIVE §2 read → **shadow critic** promotion (mc_return value twin) →
the real critic (R2 counterfactual-successor labels; R3 k-step grounded targets). The win-prob
head is the de-risked front door; V is the destination. Never skip a stage — each one's meters
license the next.

## 4. LAYER 3 — THE OPPONENT MODEL (elevated by the R-ladder's secondary finding)
α's flatness is a bias term in every downstream consumer (search leaves, threat weighting,
eventually R2 rollouts). The "what trains α" batch (#17: grad-mode ladder + coef probe +
bot-weight) runs in the rev-2 era, now with a VALUE-denominated motivation, plus the twin-
delivery arm (a PPO-shaped read beside the calibrated head). Read `_pool`-stratified, always.

## 5. LAYER 4 — THE FLOOR (accepted, not attacked)
The 39% irreducible share belongs to the hidden-team BELIEF line (the floor probe's
value-denominated case). Parked behind Layers 1–3: oracle≈honest at 1-ply says sim-level truth
is not the near-term lever; the floor states are the eventual customer of better beliefs.

## 6. SEQUENCE + KILL CONDITIONS
- **Tick-1 (now)**: v1 factory at dose. SIGNAL = C−A negative WITH C's blur < A's.
  Dosage-null = C−B stuck ≈−0.036, B−A ≈+0.06 ⇒ Layer-1 fixes are the move (already queued).
- **Rev-2**: v2 factory + (if §2 positive) trunk-open at the grad-share band; capacity +
  mirror + sd_true_excess before/after.
- **KILL**: if the v2 factory at full dose still nets NEGATIVE on the paired read, the
  label-grounding line dies; the remaining critic levers are ecology (opponent diversity via
  the flywheel — the sign-flip says population is a first-order term) and acceptance of the
  floor. Write the kill honestly and move the era's weight to the flywheel proper.
