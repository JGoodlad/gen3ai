# Richer objectives force richer representations (and why that makes distillation more efficient)

**TL;DR.** A trained network's representation is approximately the *minimal sufficient statistic* for
its objective — it keeps exactly the directions it is *forced* to keep, and gradient descent actively
collapses the rest ("it doesn't add richness for free"). So a **scalar-mean** value target
under-determines the trunk and lets it collapse (we measured this: scalar value-distill dropped
`rank/value_cls_pr` 4.15→3.62 on ai_v7_20 — a textbook case of Kumar et al.'s *implicit
under-parameterization*). A **richer** output objective — a distribution, a feature hint, a set of
predictions — pins more directions and keeps the representation full-rank. The same fact powers
distillation: a richer teacher target carries **more bits per decision**, so the student learns more per
turn (Hinton's "dark knowledge," extended to the value channel).

---

## The intuitive version

A neural net under SGD is lazy in a specific, provable way: among all the representations that minimize
its loss, the optimization is biased toward the *simplest / lowest-rank* one (the "simplicity bias" /
implicit regularization of gradient descent). It represents what it is *required* to represent and no
more — richness is never added for free.

Now ask: *what does each objective require?*
- **Scalar value (MSE).** To predict one number `V(s) = E[return]`, the trunk only needs the directions
  that move that scalar. Everything orthogonal to "who's winning, on average" is free to be discarded —
  and it *is* discarded. The representation collapses toward a 1-D-ish "value axis."
- **A distribution over returns.** To predict `P(return = z)` for 51 atoms, the trunk must keep every
  direction that changes the *shape* — spread (uncertainty), skew, bimodality (a coinflip). That's a
  strictly larger sufficient statistic → strictly more directions retained.
- **A feature hint (FitNets).** To reproduce a 128-D teacher `value_pooled`, the trunk must keep the
  directions that span the teacher's value geometry — not just the projection onto the mean.

So "a richer function forces the trunk to develop richer features" is the *contrapositive* of the
simplicity bias: the bias removes everything unforced, so the only way to keep a direction alive is to
make the loss *need* it.

## The technical backbone — four names for the same idea

1. **Minimal sufficient statistic / Information Bottleneck (Tishby).** The optimal representation of `X`
   for predicting `Y` is the minimal statistic that is sufficient for `Y`. Change `Y` from "the mean
   return" to "the return distribution" (or "a set of value functions") and the minimal *sufficient*
   statistic grows. The representation tracks the objective's information demand, not a fixed "richness."

2. **Implicit under-parameterization (Kumar, Agarwal, Ghosh, Levine 2020).** In *value-based RL
   specifically*, TD bootstrapping + MSE + gradient descent drives a **rank collapse** of the feature
   matrix: after an initial phase the effective rank *falls*, leaving only ~20–100 active singular
   directions out of 512 — "significant underutilization of network capacity." This is the pathology,
   not just a tendency, and **DR3** (Kumar 2021) is an explicit regularizer against it. *Our
   crystallization (`value_cls_pr` 4.15→3.62 under scalar distill) is this phenomenon in miniature.*

3. **The optimal representation must predict a *set* of value functions (Bellemare et al. 2019, "A
   Geometric Perspective on Optimal Representations for RL").** A single value function under-determines
   the representation (a whole polytope of representations predicts it equally). Forcing the trunk to
   linearly predict *many* value functions (adversarial value functions / auxiliary predictions) pins
   down a rich, transferable representation. Distributional RL and auxiliary tasks are instances:
   Lyle et al. ("On the Effect of Auxiliary Tasks on Representation Dynamics", 2021) and Such et al.
   (2019) show distributional RL's empirical gains are **largely a representation-learning effect**, not
   the risk/return modeling per se.

4. **Neural collapse (Papyan, Han, Donoho 2020).** Trained past zero error, a classifier's penultimate
   features collapse to their class means — variability within a class is destroyed (minimal structure).
   The flip side is the lever: when the number of targets (classes) is ≥ the feature dimension, the
   features are forced to stay *full-rank* (generalized neural collapse). More distinct things to
   predict ⇒ more structure the representation must keep.

## The RL-specific reason richness actually pays (not just "richness is nice")

Richness is **not** monotonically good — the whole point of a *minimal* sufficient statistic is "as rich
as necessary, no richer." The reason the extra directions matter *in RL* is **non-stationarity**: the
value target is the value of the *current* policy, and the policy keeps changing. Farebrother et al.
(2024) put it exactly: MSE representations become "incapable of fitting target values observed during
subsequent training" — the collapse is fine for *this* policy's value and catastrophic for the *next*
one. Bellemare's "set of value functions" and Dabney's "value-improvement path" are the same point: the
representation must stay sufficient for a *moving* target, so the directions a scalar-MSE objective
discards are precisely the ones you need a few thousand steps later.

## The honest nuance (Farebrother's ablation)

"Richer output" and "better-conditioned loss" are entangled, and the paper is careful about which does
the work: **HL-Gauss cross-entropy beats MSE, but `MSE + softmax` ≈ `MSE`** — i.e., the win is the
**classification loss**, not the categorical parameterization; and **HL-Gauss beats C51 despite not
modeling the return distribution at all.** So the mechanism is dual: (a) a richer target *constrains*
more directions (the sufficiency argument above), and (b) cross-entropy over a bounded ordinal support
is simply *better-conditioned* against noisy, non-stationary targets than MSE (robust: "HL-Gauss
degrades slower than MSE as noise increases"). Both push effective rank up; don't over-attribute to (a).
Recommended HL-Gauss: σ ≈ 0.75·(bin width) (mass over ~6 atoms) — which is what our `_value_dist_loss`
already uses.

## Why this makes distillation more effective — the bits-per-decision ladder

Your second intuition — "a richer target lets us distill more information per turn" — is Hinton's
**dark knowledge** (2015): a soft target carries the teacher's *relational* structure (inter-class
similarity), far more bits than a hard label, so the student learns more per example. Extended to our
value/policy channels, the signal-per-decision ladder is:

| Distill target | ~info per decision | our flag |
|---|---|---|
| Game outcome (win/loss) | ~1 bit / **game** | the RL reward itself |
| Policy KL (full action dist) | ~log₂\|A\| bits / decision | `--distill-coef` (≈7–10× more step-efficient than outcome — see [[on_policy_self_distillation]]) |
| Scalar value V | 1 real / decision (and it *crystallizes*) | `--distill-value-coef` |
| **FitNets feature hint** | ~128-D direction / decision | `--distill-value-feat-coef` (SHIPPED — [[project_value_distill_fitnet]]) |
| **Return distribution (KL)** | ~50-D distribution / decision | future `--distill-value-dist-coef` (side `ValueDistHead`) |

More bits/turn = more sample-efficient transfer — **provided two things hold**: the student has a head
capable of *representing* those bits (a scalar critic can't receive a distribution), and the teacher's
extra bits are real signal, not noise. Our geometry check (the 4 teachers' value subspaces are low-rank,
complementary, non-competing — `tmp/fitnet_analysis.py`) is exactly the "are the extra bits real"
verification before spending capacity on them.

## Case study: why AlphaZero's *scalar* value looks robust (and isn't a counterexample)

The apparent paradox: AlphaGo/AlphaZero used a **scalar** value head, widely called "extremely robust
and trustworthy" — yet everything above says a scalar target crystallizes. Resolution: **"robust" and
"crystallized" are two faces of one coin** — rank collapse *is* specialization to the current
distribution, which makes you *more* accurate in-distribution (reads as trustworthy) and *less* able to
extrapolate. A value can be a low-error, well-calibrated *predictor* (an output claim) while having a
*low-rank representation* (an internal-plasticity claim) — these measure different things and don't
conflict. AlphaZero could afford the crystallization for four reasons, three of which we lack:

1. **A rich policy head keeps the shared trunk full-rank.** AlphaZero's scalar value rides a trunk that
   a policy cross-entropy — predict a softmax over *hundreds* of moves toward the MCTS visit
   distribution — keeps rich. The scalar value benefits from a trunk enriched by the *policy* objective.
   **We measured exactly this on ai_v7_20:** the shared trunk stayed ~49 effrank while only the
   value-dedicated `value_cls` readout thinned (4.15→3.62). So "scalar value is fine" really means
   "scalar value *head* on a policy-enriched *trunk*."
2. **Monte-Carlo outcome target, not bootstrapped TD.** AlphaZero regresses the value toward the actual
   game result `z ∈ {−1,+1}` with `(z−v)²` — no bootstrapping — so it largely dodges Kumar's
   bootstrapping-induced rank collapse. (We're a mixed case: GAE carries some bootstrapping, and our
   crystallization came from distilling a *scalar teacher V*.)
3. **Search launders value errors.** AlphaZero's *trustworthy evaluation* is value **+ MCTS** — the tree
   averages many leaf values and does policy improvement, washing out value mistakes. **We have no
   search** (owner constraint), so the critic *is* the evaluation → a rich critic matters *more* for us.
4. **Continuous refit on fresh self-play + a clean, quasi-stationary target.** Go is deterministic,
   fully observed, clean ±1 (low-noise → MSE collapse is mildest), and the value is constantly refit as
   the policy improves *slowly along one self-play path*, so "current distribution" tracks the policy.

**The honest kicker — AlphaZero's value bias is real and was demonstrated exploitable.** Wang et al.
2023 ("Adversarial Policies Beat Superhuman Go AIs") beat KataGo (an AlphaZero-style superhuman system)
**>97%** with an adversarial policy trained on <14% of the compute — by steering it into
**out-of-distribution** blind spots its value net misjudged; the exploit transferred zero-shot to other
superhuman Go AIs and **persisted even after adversarial training**. So the folklore "extremely robust"
is true *in-distribution* and false *OOD* — a vivid, superhuman-scale confirmation of the very
policy-distribution bias the rank story predicts, not a counterexample to it.

**The one-sentence reconciliation:** crystallization is over-fitting to the current distribution; it
*looks* like robustness right up until the distribution moves — AlphaZero kept the distribution moving
slowly and covered the residue with search, so its value's specialization read as trustworthiness; we
move the distribution *deliberately* (grafting per-team specialist behaviors in via distillation) and
have *no* search, so the same specialization reads as a limit — which is why keeping the critic
representation rich matters more for us than it did for them.

## Synthesis

The trunk keeps what the objective forces it to keep and collapses the rest; SGD makes that collapse
active, and TD-bootstrapping makes it pathological (rank collapse). A scalar-mean target is the thinnest
possible objective, so it crystallizes the critic — which is exactly what we watched happen on ai_v7_20.
The fix, from both the representation literature (Kumar/DR3, Bellemare, Lyle) and the distillation
literature (Hinton), is the same move: **give the objective more to be sufficient for** — a feature
hint (FitNets, done), a distribution (Farebrother's classification loss / a distributional critic or KL
distill, next). And because a richer target is also a higher-bandwidth distillation channel, the two
motivations compound: the same enrichment that keeps the representation full-rank also transfers more of
the teacher per turn.

## See also
- `designs/learning/shortcut_learning_and_feature_delivery.md` — **the input-side dual of this note**:
  richer *targets* force richer reps; richer/easier *inputs* permit lazier ones (gradient starvation,
  amortization-vs-bottleneck, the axis rule, the P1 ablation numbers)
- `src/agents/model/CLAUDE.md` → distributional value head (v29, `ValueDistHead`, HL-Gauss `_value_dist_loss`)
- `src/agents/training/CLAUDE.md` → Exploiter distillation (`--distill-value-coef` scalar vs `--distill-value-feat-coef` FitNets); Tail-weighted value loss; grad-balance rank notes
- `designs/ai_v6/design_distributional_value_critic.md` (Phase A done / Phase B — the distributional critic)
- `designs/learning/marginalization_and_uncertainty.md` (how a net carries uncertainty — the distributional head is where that lives)
- `designs/learning/on_policy_self_distillation.md` (the bits-per-decision / dense-signal argument for policy distillation)
- Memory: [[project_value_distill_fitnet]] (scalar crystallizes → FitNets hint), [[project_representation_probe]] (our rank/effective-dim probe harness)
