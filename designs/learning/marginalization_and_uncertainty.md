# Marginalization, Jensen, and How a Model Carries Uncertainty

> **What this is.** A durable explainer for one concept cluster in our models: why we
> **marginalize** (sum outcomes over a belief) instead of **mean-field** (plug the average
> state into the calc), why that matters for thresholds like P(KO) / P(outspeed), and the
> machinery a neural net actually uses to *represent* and *reason over* uncertainty. Written
> to teach — intuitive first, then technical, no code. Grounded in the `DamageOperator`, the
> belief heads, and the distributional value head.

---

## TL;DR

- A **belief is a distribution**, not a guess, because Gen 3 hides spreads, items, and most
  moves (no team preview).
- **Mean-field** = `dmg(E[state])`: collapse the belief to its average, run the calc once.
  **Marginalize** = `E[dmg(state)] = Σ P(state)·dmg(state)`: run the calc per possible world,
  average the *results*.
- They differ by the **Jensen gap** (`E[f(X)] ≠ f(E[X])` for nonlinear `f`). Damage is
  nonlinear and the **KO / outspeed questions are step functions**, which is the worst case
  for mean-field — averaging smears the dangerous tail into the middle and erases it.
- A plain feedforward net **propagates a point, not a distribution**, and (trained with MSE)
  is *rewarded* for predicting the mean. So carrying a distribution is something you must
  deliberately engineer.
- Our highest-leverage move: **factor the known nonlinear marginalization out of the learned
  net and into a differentiable operator** (the `DamageOperator`). The net learns only the
  *epistemic belief*; the operator does the exact marginalization, gradients still flow.

---

## Part 1 — Marginals and Jensen (intuitive)

### A belief is a distribution

In Gen 3 there's no team preview, and even revealed mons hide their EV spread, item, and 2–4
moves. When the model asks *"how hard does this opponent hit me?"* it doesn't know the one true
answer — it holds a **belief**: a probability distribution over the possibilities (species,
spread, the hidden move slot).

### Two ways to turn a belief into a number

1. **Mean-field / plug in the average** — collapse the belief to its mean state, run the
   damage calc *once*: `dmg(E[state])` (damage *of the mean*).
2. **Marginalize** — run the calc on *every* possible state, average the **results** weighted
   by likelihood: `Σ P(state)·dmg(state) = E[dmg(state)]` (the *mean of the damages*).

These sound the same. They are not, and the gap between them is the whole point.

### The toy example

Opponent is, 50/50, either holding a **Choice Band** (hits for 120% of our HP → KO) or **not**
(60% → no KO).

- **Mean-field:** average the two attack stats → one middling damage ≈ 90% of HP → reads
  "survives" → **P(KO) ≈ 0**.
- **Marginalize:** half the time it's a KO → **P(KO) = 0.5**.

The mean-field answer is catastrophically wrong *for the decision*: it says you're safe when
you're one-shot half the time. **Averaging the inputs smeared the dangerous tail into the
middle and made it disappear.** P(KO) = 0.5 is a switch; "you'll be at 10%" is a stay.

### Why thresholds make it acute

Nearly every decision hinges on a **threshold**, not a smooth average:

- *Will I be OHKO'd?* → is damage **≥** my remaining HP? (a step: yes/no)
- *Will I outspeed?* → is my speed **>** theirs? (a step)

Threshold questions are the worst case for plug-in-the-mean, because near the boundary the
answer flips discontinuously. What you care about is **how much probability mass crosses the
line** — and only marginalizing (evaluate the question per world, count) gives you that. The
average state sits on one side of the line and reports a confident 0 or 1, throwing the real
uncertainty away.

**Slogan: marginalize, don't mean-field.**

---

## Part 2 — Marginals and Jensen (technical)

### Jensen's inequality is the formal reason

For a convex `f` and random variable `X`: `E[f(X)] ≥ f(E[X])` (reversed for concave; equal
only if `f` is linear or `X` is deterministic). The difference `E[f(X)] − f(E[X])` is the
**Jensen gap**.

Map onto us: `X` = the hidden state drawn from our belief, `f` = the damage calc (or the KO
indicator). Because `f` is nonlinear, the plug-in estimate `f(E[X]) = dmg(E[state])` is a
**biased** estimator of the true expected outcome `E[f(X)] = E[dmg(state)]`. The bias is
exactly the Jensen gap; it's zero only when damage is linear (it isn't) or the belief is a
point mass (everything revealed — no uncertainty).

### Why damage / KO is nonlinear

Two sources stack:

1. **The damage formula** has products and conditional multipliers — STAB, type effectiveness
   (×0, ×0.5, ×2, ×4), Choice Band ×1.5, boosts, burn, screens. Multiplying *averaged* inputs
   ≠ averaging the *products*. A ×4 / ×0 type split is brutal: the mean multiplier (×2)
   corresponds to nothing real.
2. **The KO question is an indicator** — `1[dmg(state) ≥ HP]` — the most nonlinear "function"
   there is (a step). `P(KO) = Σ P(state)·1[dmg(state) ≥ HP]` is genuinely a probability in
   (0,1); the plug-in version evaluates the step *once* at the mean damage and can only return
   0 or 1.

### Marginalization = expectation under the belief

Summing a hidden variable out against its probability:

- `expected_damage = Σ_s P(species=s)·dmg(s)`
- `P(KO) = Σ_s P(species=s)·P(KO | species=s)`

…and likewise over spreads, over the 16 typed Hidden-Power candidates (each weighted by
`w·hp_prob`), over the move belief. Each uncertain axis is summed out against the head's
predicted distribution — which is why the operator consumes the belief's *full* predicted
move distribution and the *species posterior*, not an argmax, and why the gradient can sharpen
those distributions toward real KO threats (the loss flows through every weighted term).

### The three-way principle

Organize by the *type* of uncertainty:

- **Deterministic facts → compute.** Type chart, base power, our own item. Apply the physics
  exactly; no distribution needed.
- **Epistemic uncertainty → learn a belief, then marginalize.** Hidden species/spread/moves —
  unknown but they *have* an answer. The head predicts `P(·)`; sum over it. *Reducible.*
- **Aleatoric uncertainty → carry the distribution.** The damage roll (85–100%, 16 rolls) and
  the crit coin-flip are irreducibly random even with perfect knowledge. Don't collapse to a
  mean; keep the distribution and integrate it into P(KO).

So "expected damage to our mon" is a **double** expectation — over the epistemic belief *and*
the aleatoric rolls — both taken by summing outcomes, never by averaging inputs.

### Where this lives in our architecture

- **`gen3_incoming_crit_split_v1`** — splits P(KO) into a **modal no-crit** term + a separate
  **crit-risk delta** (∈ [0, crit_p]). A marginalization choice: don't blend the 1/16 crit
  world into one mushy mean; expose the two regimes so the policy prices the modal line
  correctly *and* sees the tax. Same spirit as the v28 Choice-Band-conditional OHKO tail,
  decorrelated from `p_cb` because OHKO is a nonlinear threshold a blend would blur.
- **v36 `--threat-prob-outspeed`** — divide the speed gap by the **believed speed std**, push
  through a sigmoid (≈ a normal CDF). That marginalizes the threshold over the spread belief:
  `P(speed_us > speed_them)` instead of `1[speed_us > E[speed_them]]`. A point estimate is the
  degenerate case where you pretend the std is zero.
- **v36 expected-latent defender** — the deliberate *exception*: for an unrevealed switch-in
  the op marginalizes `P(species)` through the type chart + spread prior for expected damage,
  but **nulls P(KO)** — a full-HP switch-in is ~never OHKO'd, so the KO-threshold term carries
  almost no signal while adding all the Jensen-gap subtlety. Dropping the nonlinear threshold
  where it doesn't pay is the exception that proves the rule.

---

## Part 3 — How a model represents and reasons over uncertainty

### The crux: a feedforward net propagates a *point*

Feed a vector in, each layer applies a nonlinearity, a vector comes out. If you feed `E[x]`
you get `f(E[x])` — **mean-field by construction**. Uncertainty does *not* automatically flow
through the layers; the net just transforms the point it was handed. Carrying a distribution
must be engineered. There are four ways.

### 1. Output the *parameters* of a distribution

- **Categorical → softmax.** A logit vector *is* a representation of "50/50 between A and B."
  Our move-belief head outputs `P(move)` — a carried distribution.
- **Gaussian → mean + log-variance** (heteroscedastic regression): "80% damage, ± a lot" vs
  "± a little."
- **Mixture (MDN — mixture density network) → weights + means + vars.** The only one of these
  that represents genuine **bimodality** (CB-KO *or* no-KO — two humps, not one smeared mean).
  Single-Gaussian and point heads physically cannot.

### 2. Discretize the outcome, predict probability-per-bin (distributional RL)

Fix a support of atoms (returns from −1..+1 in N bins), predict a softmax weight per atom. The
output is an arbitrary-shaped histogram: sharp = confident, wide = uncertain, bimodal =
coinflip. This is the canonical "carry the distribution" (C51 / QR-DQN / IQN). **It is exactly
our `ValueDistHead` (v29).**

### 3. Carry sufficient statistics + propagate analytically

Sometimes a couple of **moments** + a closed-form push-through suffices. Carry mean *and* std,
propagate through the nonlinearity analytically. **v36 `--threat-prob-outspeed`** is precisely
this: one extra scalar (the speed std) → a hard 0/1 threshold becomes a calibrated
`P(outspeed)` via a sigmoid (≈ normal CDF). Cheap, no sampling, a genuine marginalization.

### 4. Implicit / latent representation (free-form)

A transformer token after attention can encode a superposition ("probably Suicune, maybe
Starmie") that downstream layers decode threshold-style. Most flexible, least guaranteed —
the model uses the latent this way only if training rewards it (probe to find out). Also in
this family: **ensembles / MC-dropout / sampling** — represent uncertainty as the *spread
across forward passes* (textbook epistemic uncertainty; we don't lean on it).

### Is some *structure* better?

The backbone (CNN/transformer/MLP) matters less than **(a)** the output parameterization,
**(b)** whether hypotheses are separable units, **(c)** the loss. Within that, two structures
have the right bias:

- **Attention is a soft marginalization primitive.** `output = Σ_i softmax(scores)_i·value_i`
  is literally `Σ P(i)·v_i` — a weighted sum over hypotheses, the shape of a marginalization.
  Arrange each hypothesis as its own **token** (a candidate species, a candidate move) and
  attention can *learn* to weight-and-sum them. This is why we use **belief slots** (distinct
  unknown-mon tokens) and the **top-K discrete incoming block (v30)** (each candidate move is
  its own addressable row) *instead of* the collapsed `_chan_max` worst-case. The move from a
  single pooled vector to a *set of tokens* is the whole game: **a pooled vector forces
  mean-field (nothing left to weight-and-sum); separable units enable marginalize.**
- **Distributional / quantile heads** are the proven tool for "predict a distribution over a
  scalar I care about," beating a Gaussian head on multimodal truth and a point head whenever
  you care about the tail.

### The loss is half the battle: MSE *bakes in* mean-field

**The minimizer of MSE is the conditional mean.** Train a regression head with MSE and even
if the truth is "50% chance of 0, 50% chance of 200," the loss-optimal output is **100** — the
mean — every time. MSE literally instructs the net to mean-field; no architecture escapes it.
To carry spread you need a **proper scoring rule** whose minimizer is the *distribution*:

- **Cross-entropy / NLL** (categorical or Gaussian-param heads),
- **Quantile / pinball loss** (minimizer is the quantile, not the mean),
- **CRPS** (full distributional outputs).

So "help it carry the distribution" is as much about the *loss* as the *head*. Our belief
heads train with cross-entropy against the true hidden species/moves (a proper scoring rule);
the distributional value head is set up for a distributional aux loss, not MSE-to-the-mean.

### How to help the model marginalize, not mean (ranked by leverage)

1. **Factor the nonlinear marginalization OUT of the learned net and into a differentiable
   operator.** Our `DamageOperator` philosophy and the biggest lever. Don't make ReLUs learn
   `Σ P(s)·dmg(s)` from scratch — *compute it exactly*: belief distribution in, multiplicative
   physics + threshold integral done in hand-written differentiable code, marginalized P(KO)
   out. The net learns only the *epistemic belief*; the math is exact every time and gradients
   still sharpen the belief. ("The operator does the multiplicative physics so the ReLU head
   stays additive.") This is the provide-vs-learn principle applied to the marginalization
   machinery — don't burn capacity badly relearning physics we know.
2. **Represent hypotheses as separable tokens/rows, never a pre-blended vector** — so attention
   or the head can weight-and-sum (belief slots, top-K moves). Any pre-collapse forces
   mean-field upstream.
3. **Output distribution parameters + train with a proper scoring rule** (categorical+CE,
   quantile+pinball). Never MSE if you care about spread.
4. **Carry moments + propagate analytically** when the question is a threshold (mean+std → CDF;
   v36 outspeed).
5. **Enumerate-and-sum when the support is small** — the aleatoric rolls aren't sampled; we
   carry the explicit 16-roll support + 2-point crit distribution and integrate
   `P(KO|hit) = fraction of rolls ≥ HP`. Exact Monte-Carlo over a tiny support (same for the
   16 typed Hidden-Power candidates).
6. **Decorrelate regimes** so an additive head can't re-collapse what you marginalized
   (crit-split, CB-conditional tail).

---

## The synthesis

A deterministic net mean-fields by default because it propagates a point and (under MSE) is
*rewarded* for predicting the mean. You make it carry and marginalize distributions by some
mix of: **(a)** parameterizing the output as a distribution (softmax / categorical / mean+var
/ mixture), **(b)** training with a proper scoring rule not MSE, **(c)** representing
alternatives as separable units so attention does the weighted sum, and — highest leverage for
us — **(d)** pulling the known nonlinear marginalization out of the learned function and
computing it exactly in a differentiable operator, leaving the net to learn only the belief
that feeds it. Attention is the one backbone primitive that's natively a soft marginalization;
distributional heads are the proven structure for outcome uncertainty; neither rescues you
from an MSE loss or a pre-collapsed input.

---

## See also

- Root `CLAUDE.md` → Feature Extractor Architecture (the `DamageOperator`, belief heads,
  `ValueDistHead`), and Model Versioning (v19/v21 damage op, v29 distributional value head,
  v30 top-K incoming, v36 bidirectional threat / prob-outspeed).
- `src/agents/model/CLAUDE.md` — the 7-phase extractor contract.
- `designs/ai_v6/design_differentiable_damage_op.md`, `design_distributional_value_critic.md`,
  `design_bidirectional_threat_trunk.md`.
- Memory: `project_gpu_damage_op.md` (COMPUTE deterministic / learn epistemic / distribution
  aleatoric; Jensen — MARGINALIZE not mean-field).
