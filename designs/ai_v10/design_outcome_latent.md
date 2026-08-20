# design — THE PER-ACTION OUTCOME LATENT: learned consequences at the pointer terminal

> **[STATE 2026-08-19] FORWARD DESIGN — nothing built, nothing scheduled.** Owner-initiated from
> the c-family conditioned read + the α/β injection probe (ledger, 2026-08-19): "good to think
> about for the future." Earliest sensible slot is post-gen-16, after the substrate verdicts land
> — this is the NEXT mechanism class in the same lineage, not a competitor to what is training.
> It refines ai_v6's "Meaning B" latent-predictive design with three years of measured evidence
> about which delivery channels and which representation pressures actually work here.

## 0. The object

`g(trunk, a) → z_a` — a LEARNED predicted-outcome latent per legal action, delivered where the
injection probe proved per-action content moves the policy: as token content at the pointer
terminal (and, via `V(z_a)`, readable by the critic). It is route 3 of the delivery taxonomy —
per-action absolute content at T2 — with learned content instead of curated content.

That one substitution attacks the two structural downsides of the pointer-cell programme:

- **The curation ceiling.** A cell carries exactly what we thought to compute, at the correctness
  of our physics (the de-timid class; `is_boost` delivering 4–6× its own consequence columns;
  unrevealed-bench default cells out-delivering the real active cell — all measured 2026-08-19).
  A learned predictor is supervised by what actually happened, so it can carry consequences
  nobody named.
- **Policy-only delivery.** Pointer cells never reach the critic; `z_a` does, through `V(z_a)` —
  one object serving both heads instead of every fact needing two routes.

## 1. Why this is the next mechanism class (the evidence lineage)

1. **Channel**: the injection probe measured the same information at ~0.01 nats through a softmax
   ratio and **41 percentage points** through a per-action cell. Per-action absolutes are the
   terminal that works.
2. **Shape**: `IntentThresholdMoveCell` — right channel, wrong shape (no arrival axis) — was left
   at |W|₁ 0.48 by the policy while its value twin trained to 15.4. Direct delivery is necessary,
   not sufficient; the content must match the decision's shape. A predicted successor state IS
   the decision's shape.
3. **Content**: the c-family read proved the hypothetical-world CONTENT is learnable and used
   (c1 carries 20.5% of the boost decision on its home turf) — the concept survived; only its
   ratio delivery died.
4. **Need**: the sentinel sweep's loss driver is CRITIC SURPRISE (median P(win) 0.827 on the
   top-50 losing decisions), and ledger C6 closed the critic *delivery* line with "reopen only
   with a NEW pre-registered mechanism." A self-predictive representation is a new mechanism
   class — representation-level, not delivery-level.

## 2. The causal claim, and why STATE beats VALUE as the target

Predicting a per-action VALUE (Q-like) compresses to one number and can be satisfied by
correlation ("boosting tends to be good here"). Predicting the per-action STATE forces the
representation to carry the MECHANISM: after Spikes, the hazard bit is set; after Earthquake into
the pivot, their Salamence is at 100%. The policy today only needs to rank actions; this
auxiliary makes "what happens if" a represented object — the causal structure the injection probe
showed the policy side lacks.

Simultaneous moves make the honest one-ply target an expectation over opponent response:
`z_a = E_{k~α}[ s'(a, k) ]` — the same Contract-W contraction the substrate uses, so this lives
at **T2** and inherits the intent machinery (and its `label_only` publication discipline) whole.

## 3. The Spikes factorization — the honest route for facts we cannot model

Spikes' worth is a tax on the opponent's future switch RATE — a long-horizon quantity the one-ply
cells refuse on purpose ("modelling it here would be confidently wrong"). The factorization:

- the PREDICTOR supplies the mechanism: *this click puts Spikes on the field* (a hard, named,
  next-obs target no degenerate latent can fake);
- the CRITIC supplies the horizon: *boards with Spikes up win more* — V is trained by
  bootstrapping across whole games, the one component that legitimately compresses horizons.

Mechanism learned, valuation learned, nothing hand-asserted. This is the template for every
mechanic the rules engine cannot price honestly at one ply.

## 4. The build ladder (each rung graduates on a BEHAVIORAL or VALUE delta — decodable ≠ helps)

- **G0 — probe before building (free).** Does the critic already price the mechanism? The
  prober's intervention sweep can flip the hazard bits and read ΔV today. The opp-action head was
  falsified by exactly this kind of probe; this design earns a rung only where G0 finds a gap.
- **R1 — taken-action prediction (cheap).** Predict the next state-latent for the action actually
  taken; labels are free from the rollout (SPR-style). Shapes the trunk causally; not per-action.
- **R2 — counterfactual labels (the unique asset).** The battle-reconstruction layer materializes
  the TRUE s′ for every legal action under common random numbers — supervised labels for actions
  never played, no off-policy correction. Nobody else has these labels; selective generation (the
  search-teacher cost profile), never exhaustive.
- **R3 — outcome-token injection.** `z_a` joins the pointer inputs per action; `V(z_a)` joins the
  critic read. The ai_v6 Meaning B end state, reached only if R1/R2 pass their gates.

## 5. What forces the latent to be RICH — the pressure menu, ranked by our own body count

A latent is exactly as rich as the set of questions asked of it that cannot be shortcut. Richness
comes from the DECODER side; the encoder-side regularizers are hygiene at best. Empirical ranking
on this codebase:

1. **USE with gradient** — the strongest measured pressure. The DamageOperator trains the move
   belief through consumption (the differentiable-expert effect); the SimSiam latent, which
   nothing read, learned geometry and helped nothing (deleted v75). Feed `z_a` to the pointer;
   evaluate under a stop-grad ladder so use-pressure and measurement stay separable.
2. **Named-block decoding** — decode the predicted latent into the STRUCTURED next-obs blocks
   (HP, statuses, hazards, boosts, faints), weighted by decision relevance. Collapse-proof: the
   targets are external facts. Our typed obs is what makes this affordable.
3. **Reward + value consistency** (the MuZero/value-equivalent pressure) — `V(z_a)` must match
   the bootstrapped value at t+1, plus reward prediction. Rich where it matters, not everywhere.
4. **Action-conditioned contrastive with CRN negatives** — discriminate which of the k legal
   actions' successor states YOUR action caused, all under the same dice. The negatives differ
   ONLY in the causal effect of the action; InfoNCE is collapse-immune by construction. The
   sharpest causality pressure on the menu, and uniquely ours (the reroll layer manufactures the
   negatives on demand).
5. **Multi-head breadth** (the BeliefBank pattern) — many small maskable decoders (what they
   clicked, damage dealt, faint causes) pulling on one latent.
6. **Multi-step consistency** (SPR k-step) — shortcut features that survive one step die at three.
7. **Variance/whitening regularizers — hygiene only, never the mechanism.** The scar tissue:
   seed-VICReg satisfied every term at effective rank 1.05, and the shared-readout quantile loss
   constrained each seed only along its own weight vector — every orthogonal direction stays
   free. Spread is not information.

**Pre-registered composition for R2/R3**: named-block decode (mechanism) + reward/value
consistency (worth) + CRN contrastive (causality), `z_a` fed forward under the stop-grad ladder,
variance terms only if collapse monitors fire.

## 6. Costs and risks, stated now

- **Collapse and its cousin, useless competence**: every rung's gate is behavioral/value, never
  its own loss (the SimSiam rule, written in blood).
- **Compute**: a `g` forward per legal action (~9× per decision at the head, though `g` is small);
  R2 label generation is real CPU and must be selective.
- **One-ply honesty**: `z_a` is E over α — a switching opponent makes s′ multimodal, and an
  expectation over modes can be a state that never occurs. The contrastive pressure (which uses
  REAL per-(a,k) successors) partially compensates; a mixture/per-k formulation is the fallback
  if the expectation proves dishonest in practice.
- **Measurement**: use-pressure confounds attribution; the `label_only`-pattern ladder is
  mandatory, and every enabling follows the C4-style offline-gate discipline.

## 7. What this doc is NOT

Not search at inference (the no-search-on-the-model constraint stands; the sim is a supervision
oracle only). Not the fingerprint aux. Not a gen-16 rider and not scheduled — it queues behind
the substrate verdicts and competes on evidence like everything else. Its relation to
`designs/ai_v6/design_latent_predictive_representation.md` is refinement, not replacement: same
end state, re-derived from the 2026-08 evidence, with the CRN-negative contrastive as the new
ingredient that infrastructure built since then makes possible.
