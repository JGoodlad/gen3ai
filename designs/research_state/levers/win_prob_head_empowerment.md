# Win-prob head empowerment — the best judge in the network, and nothing listens to it

**Status:** 🟡 **ACTIVE — the binding constraint in three programmes at once** · **Ledger:**
`9cb825c` (programme registered) · `596608e` + `8b83cff` (value foundations) · `b070d6e`
(barometer/coach) · `bda8382` (probe L) · `5edbd05` (E5) · `195ce9f` (harvest pilot) ·
`85aadd4` (the α/β + CfEvidentialHead corrections)

One-line claim: *the win-prob head already knows what the policy gets wrong — 96.4% of immune whiffs
at decision time — and the knowledge cannot reach behavior through any coefficient, because the
quantity that carries it is the head COMPOSED WITH A SIMULATOR, a composition PPO never performs.*

## Known (cleared the honesty gates)

- **Two jobs, two instruments (`596608e`).** The shaped-return critic is *definitionally* correct
  for its job: GAE advantages must be estimated in the units of the reward stream being optimized,
  so given shaped rewards no other critic is legal. The win-prob head is the correct GAME value —
  outcome units, no γ distortion, no PopArt drift. The two-head structure is the automatic
  CONSEQUENCE of choosing shaped rewards, not a design accident. *The only error was the search
  battery using instrument A for job B — found by probe G, fixed.*
- **The critic is PLUMBING (`8b83cff`).** It holds no knowledge; its entire job is policy-gradient
  variance reduction. **Epistemically second-tier BY DESIGN, operationally first-tier BY NECESSITY.**
  Reward units are an accounting identity, not a preference (AlphaGo's V ≡ P(win) because it didn't
  shape — a different REWARD choice, the same law).
- **BAROMETER, not COACH (`b070d6e`).** The live `win_prob_mode="shaping"` @0.05 is REPRESENTATION
  shaping: BCE-on-terminal-outcome pushes outcome-predictive features into the shared trunk and
  exerts **zero force on behavior** — there is no gradient path from predict-wins to
  choose-winning-actions. And the labels are **self-referential**: habitual whiffs that still win 55%
  teach the head "55%", never "the whiff was the mistake". Action-level badness needs a
  counterfactual contrast the state label lacks.
- **🏆 Probe L: the head KNOWS (`bda8382`).** 617 immune-whiff decisions over 834 battles, each
  scored by its OWN snapshot: the one-ply win-prob ranking prefers a non-whiff action at decision
  time on **0.964 [0.948, 0.978]** (bar was ≥60%), median margin **0.049 win-prob units** against a
  within-decision sd of **0.00062** — clearing the floor by two orders of magnitude, and surviving
  all six dice streams on 86.7%. It is **whiff-SPECIFIC**, not a generic edge: +0.213 vs
  hit-pivot, +0.342 vs no-pivot.
- **The repeat-offender hypothesis is refuted BY A CEILING:** the head reads **1.000 on the FIRST
  click of a loop.** It knew immediately, forever, and was ignored every time — because **the policy
  samples the head's preferred action at median p = 0.002** (77% below 5%).
- **α and the head divide labor correctly** (`bda8382`, `85aadd4`): the opponent-intent **α flags
  THE PIVOT** (+0.209 vs no-pivot) and is null on the whiff; the whiff knowledge lives in the
  win-prob head. Note there are TWO α/β pairs — the intent `alpha_head`/`beta_head` (α feeds every
  Σα·f op reduction and search α-pruning) and the **`CfEvidentialHead` Beta(α, β) confession head**
  on win probability, ALWAYS-DETACHED by design because a confession must not influence the
  confessor.
- **The "shaping" lever is STRUCTURALLY refuted (`bda8382`).** Trunk share **1.02%** (an L1 upper
  bound) at cosine **−0.133 AGAINST** the policy gradient; the reward registry has no win-prob
  member; even a hypothetical PBRS @0.05 is homeopathic (1.6e-3/step, 5.4e-5 of terminal).
  **"Raise the dose" names no real mechanism.** The distillation branch fires instead, with the
  argument the registration lacked: *the head's ranking is not a quantity the network computes.*
- **The uncertainty machinery works (`bda8382`, measurement 4).** The CfEvidentialHead is LIVE
  (mean tracks the head r=0.82, 0% at Beta(1,1)) and **CONFIDENT where it disagrees** (evidence
  10.07 at whiffs vs 9.24 ordinary) — its designed role is the label factory's priority sampler.
- **🔴 The harvest pilot's FAILURE is the finding (`195ce9f`).** Supply census: 54,487 labelable
  decisions / 1,404 battles, **100% reconstruction coverage** — no fresh self-play needed. And naive
  head fine-tuning **REGRESSED**, caught only by the untouched LONG-WIN control: the labels were
  excellent **and for the wrong states** (fit set turns 60–152; 29.3% of eval turns beyond its max).
  ***A label factory that never samples the region its meter scores is extrapolating.*** Damage
  scaled with the fit-set mean offset across two independent runs ⇒ selection bias convicted, not a
  hyperparameter — hence `--anchor-coef` defaults ON, because 0.0 was **measured** destructive.
- **The head has NO credit-assignment-through-time problem BY CONSTRUCTION (`40f3da6`).** MC labels
  stamp the terminal outcome onto every step, so turn-100 of a capped game is labeled 0 directly.
  The failure is a CENSUS problem (discrimination mass at time slices), not signal travel.

## Not-known

- **Does a properly-dieted head fix the 0.999 tails?** The reducibility probe is queued and
  currently BLOCKED on data (see `stall_tail_overconfidence.md`).
- **Does route 2 actually move behavior?** Nothing has run. The knowledge is proven present and
  proven unreachable by coefficient; that the explicit-teacher route delivers it is a hypothesis.
- **EPISTEMIC uncertainty for the racer.** The winner's-curse hole is that paired CIs see sampling
  noise, not leaf error. First version = checkpoint-DISAGREEMENT spread (no retraining, 2–3× a cheap
  one-ply cost), validated against probe K's labels: *does disagreement predict leaf error?* ⚠️ A
  learned variance head captures ALEATORIC (the ~39% irreducible hidden-info floor), **not** the
  epistemic bias that burned iteration 2 — the flavor matters.
- Whether v29's existing calibrated distributional value head can serve as a SEARCH-uncertainty
  source; its "not a training lever" verdict never tested this job. *(Note: for a binary outcome the
  mean IS the full aleatoric distribution, so a distributional WIN-PROB head has nothing to add.)*
- Label freshness: ground truth is Q under the CURRENT policy's continuation.

## Pros

- The knowledge is already there, measured, and enormous (96.4% at a 0.049 margin) — this is a
  DELIVERY problem, historically the cheapest class in this project to fix.
- One label serves two consumers (E5): the same counterfactually-grounded re-rolls train the Q head
  *and* act as route-2 distillation targets.
- Grounded labels are genuinely NEW content, unlike the barometer's self-referential ones — which is
  what makes a light aux loss on them defensible where "raise the shaping dose" is not.
- The programme carries its own retirement criterion: the **amortization residual** (Q head vs true
  re-roll, per state class) is the value of one-ply search as a number — shrinking ⇒ the AlphaZero
  ratchet; stubbornly large classes ⇒ the states that genuinely need live search.

## Cons

- **The naive version is destructive** — proven, twice, by the pilot. Every future consumer inherits
  the anchor + tail-stratum requirement.
- **The starvation trap for the Q head (`229e9f1`):** on-policy data labels only the taken action,
  and the preferred alternative is sampled at p = 0.002 — so a naively-trained Q head is untrained
  *exactly where it matters* and confidently wrong on the never-tried moves. Counterfactual labels
  are not an optimization here, they are the precondition.
- Route 2 imports the head's differential bias into the weights with no invariance shield.
- **`q_labels` has no producer yet** — the whole E5 loop is supply-blocked on one build.

## Next test

Ordered by leverage (`9cb825c`, amended by probe L):

1. **The `q_labels` producer** — extend the counterfactual producer to emit per-action labels at
   schema v1. Everything downstream is blocked on it.
2. **Route 2's first arm** — one-ply win-prob-ranking targets through the landed search-teacher mode,
   gated by confirmed overrules. Route 1 is suppress-only and, per probe L, must start its ladder
   *far* above 0.05 to be non-homeopathic ({0, 3, 9} as fractions of `VICTORY_VALUE`).
3. **Epistemic uncertainty via checkpoint disagreement**, validated against probe K's labels, then
   wired into racing separation thresholds if it predicts leaf error.
4. **Audit the Beta head's evidence output as the racing threshold input** before building
   checkpoint ensembles — the cheap version of (3), and it is already live.
