# Shortcut learning and feature delivery — when feeding the head directly is a plus

> **What this is.** A durable explainer for the question *"if I hand the model a strong computed
> feature straight into the head, does it just get lazy — and is that good or bad?"* Covers the
> real mechanism (simplicity bias → **gradient starvation**), the distinction that resolves it
> (**amortization vs. bottleneck**), the axis rule for collapsed summaries, the four tests that
> actually discriminate laziness from genuine use, and what we **measured** on our own
> `DamageOperator`. Intuitive first, then technical, no code.
>
> **This is the input-side dual of [[objective_richness_and_representation]]**: that note says
> *richer output targets force richer representations*; this one says *richer/easier inputs
> permit lazier representations*. Same simplicity bias, opposite side of the network.

---

## TL;DR

- The folk statement "models learn the simplest function that minimizes the loss" is imprecise.
  What's true: SGD learns **simple things first** (spectral bias), and once the easy feature
  drives the loss down, the hard route **stops receiving gradient** — *gradient starvation*
  (Pezeshki 2021). Features don't compete on merit; the first adequate explanation suppresses
  the rest.
- **RL amplifies this brutally.** PPO on win/loss carries **~1 bit per game**. There is often
  never enough signal to fund the hard route once an adequate shortcut exists.
- **The question is not "did I feed it directly."** It is: **is what I fed a sufficient statistic
  for the decision I am asking it to make?**
  - Correct + sufficient + cheap ⇒ feeding it directly is **amortization**, a pure win. Making
    the trunk re-derive exact Gen 3 damage arithmetic is waste.
  - Lossy summary ⇒ feeding it directly is an **information bottleneck you built yourself**.
- **The axis rule:** *collapse along axes you are not choosing over; never collapse along an axis
  you must choose along.* A worst-case threat summary is nearly sufficient for **V(s)** and
  catastrophically insufficient for **π(s)**, because pivot choice is a choice along the very
  axis the max collapsed.
- **What we measured, and it surprised us:** the model did **not** lazily prefer the collapsed
  summary — it **ignored** it. Un-collapsed per-action blocks dominate dependence (OUTGOING =
  65.7% of the ablation ceiling); collapsed aggregate blocks are near-inert (incoming secondary
  **0.1%**). A lossy summary usually doesn't become an attractive shortcut; it becomes dead
  weight you paid compute for. **But** that only holds *because the un-collapsed blocks sit next
  to it* — a shortcut is attractive only when it's the best thing available.
- **The reframe:** you cannot stop gradient descent being lazy, so **make the lazy path be the
  correct path.** That is exactly what the pointer head does.
- **The op head-concat did NOT starve the edges — and the edges did not absorb the concat.**
  Measured three times (gen-1 @40M, gen-2 @40M, gen-2.5 @25M): the concat arm flips *more*
  actions than turning the **entire** edge system off, while edge dependence still **grew ~3×**
  with training. Paths compete only when they are **substitutes**; these two do different jobs.
- **The structural reason (Part 6):** softmax attention weights are **normalized** — an edge
  bias can move *who attends to whom*, but the number it writes into the residual stream is a
  **ratio within its row**, not an absolute ("53% of max HP"). The two entity-native channels
  that *can* carry an absolute are **token content** (the `prefuse_proj` injection) and
  **per-action cells at the logits** (`pointer_cells`). Not an impossibility proof — a
  **capacity + conditioning** argument.
- **Pre-registered decision rule** (recorded 2026-08-07, before gen-3's audit): absorption ⇒
  mask → A/B → delete; a fourth replication ⇒ **widen token-content delivery** and localize the
  residual per sub-block, then re-audit. Explicitly banned: *"wait longer"* and *"delete
  anyway."*
- **Corollary — top-K is two things.** As *representation* (v30/v35/E4: surface the K
  most-believed moves individually) it un-collapses an axis and is free. As *truncation* (v49:
  drop candidates below rank K) it removes mass, and its error is a **cliff, not a slope**
  (top-16 owns 94.2% of channels but **misses are BIMODAL**). Truncation converts belief quality
  from a **calibration** problem into a **ranking** problem — so it *creates* the requirement for
  a better belief. Fix order: **raise K** (the spike says it's ~free) → **add the tail bound**
  (a sound bound beats a better estimate when the failure is a cliff) → only then touch the
  belief, and only after the recall@K gate.

---

## Part 1 — The mechanism, stated correctly

### Intuitive

Three separate things get conflated under "nets are lazy":

1. **Spectral / simplicity bias.** SGD fits low-complexity, low-frequency structure first
   (Rahaman et al. 2019; Kalimeris et al. 2019, *SGD on Neural Networks Learns Functions of
   Increasing Complexity*). This is an **ordering**, not a guarantee that the final function is
   simplest.
2. **Gradient starvation** (Pezeshki et al. 2021). The real killer. Once an easy feature explains
   the label, the loss is low, so the gradient reaching a competing hard feature collapses toward
   zero — and it never develops. The features are not evaluated on merit; **the first adequate
   one suppresses the alternatives.**
3. **Shortcut learning** (Geirhos et al. 2020). The behavioural consequence: the model solves the
   benchmark by a correlate that fails off-distribution.

### The RL amplifier

Supervised learning with abundant data eventually funds the hard feature — there is always more
gradient. PPO on terminal win/loss delivers **~1 bit per game**. Once an adequate shortcut
exists, the hard route may *never* be funded. (This is the same accounting that makes on-policy
distillation ~7–10× more step-efficient — see [[on_policy_self_distillation]].)

A fourth, RL-specific mechanism compounds it: **implicit under-parameterization** (Kumar et al.
2020) — TD bootstrapping + MSE actively *collapses* the feature matrix's effective rank. See
[[objective_richness_and_representation]] for that side.

---

## Part 2 — Amortization vs. bottleneck: the distinction that decides it

> **Laziness about a KNOWN, SUFFICIENT function is amortization — free money.**
> **Laziness about a LOSSY SUMMARY is an information bottleneck you built with your own hands.**

The variable is not the delivery route. It is **sufficiency for the decision**.

**Case A — the damage formula (a plus, unambiguously).** The `DamageOperator` is the exact Gen 3
formula in PyTorch, fuzz-validated against constructed Showdown probes
(`damage_op_probe_fuzz_test.py`). There is no version of "the trunk learns the arithmetic from
scratch" that beats the arithmetic. And the shortcut is *simultaneously the training signal for
the hard part*: the op consumes the believed opp moveset from `MoveBelief`, so policy loss →
damage → belief. The lazy route trains the non-lazy component. See
[[entity_tokens_biases_pointers]] → the differentiable expert.

**Case B — a worst-case threat summary (a minus).** `_chan_max` collapses over the move axis:
max phys / max spec damage per defender. That is:

- **nearly sufficient for `V(s)`** — "how bad is my position" is essentially a survival-probability
  question, and expected/worst-case damage taken determines it;
- **catastrophically insufficient for `π(s)`** — the whole point of choosing a pivot is that
  damage differs per (pivot, move) pair. Thunder Wave → a Ground type is zero; a collapsed max
  does not know the move was Thunder Wave.

### The axis rule

> **Collapse along axes you are not choosing over. Never collapse along an axis you must choose
> along.**

This rule *is* the v30→v39 progression, each version un-collapsing an axis the policy chooses along:

| Version | Axis un-collapsed |
|---|---|
| v30 `--damage-topk` | move identity — the opp active's K most-believed moves, individually |
| v34 `--damage-matrices outgoing` | defender axis — our moves × their 6 mons (KO a switch-in) |
| v35 `--damage-matrices incoming` | (our mon × their move) jointly, with per-cell `status_lands` |
| v39 `--damage-matrices-outgoing-all` | attacker axis — our 6 mons × their active (forced-switch offense) |

---

## Part 3 — What we actually measured (and the surprise)

> ⚠️ **PROVENANCE — the table below is a HISTORICAL result, not the current model.** Measured
> **2026-07-25** on a pre-pointer-head snapshot, in a config that included the `outgoing_matrix` and
> `outgoing_attacker_matrix` blocks. **The current production config has neither**, and the head
> route it measured (a flat positional `action_net`) no longer exists. The current-config
> replacement is
> [`designs/research_state/measurements/gen3_op_block_dependence_6k.json`](../research_state/measurements/gen3_op_block_dependence_6k.json)
> (gen-3 @9.6M, 6000 states, 2026-08-07) — reported later in this file under *MEASURED 2026-08-07* —
> and **it reverses this table's headline**: the incoming matrix carries essentially the whole
> concat ceiling while the outgoing single-active block sits at its own shuffle control. Keep this
> table for the *reasoning* it supports (collapsed aggregates were dead weight, which is why they
> were deleted); do not quote its percentages as current. Its raw output was never committed.

**P1 ablation probe** (`tmp/op_block_ablation_probe.py`, 2026-07-25; 4000 real eval states, exact
producing snapshot, per-block zero → masked KL against the policy's own distribution; a SHUFFLE
control exceeded zeroing everywhere, so the head reads state-specific content):

| Block | % of the zero-whole-op ceiling | In the current production config? |
|---|---|---|
| OUTGOING (per-action, un-collapsed) | **65.7%** (75% of the *moves* ceiling) | yes |
| outgoing-attacker matrix (our 6 mons → their active) | 21.4% | **no** |
| incoming matrix (per mon × move) | 15.4% | yes |
| incoming per-mon | 12.7% | yes |
| status-landing | 8.8% | yes |
| outgoing matrix (our moves → their 6 mons) | 6.3% | **no** |
| Choice-Band | 2.9% | yes |
| incoming **effect** (collapsed) | 1.2% | deleted from the code |
| incoming **secondary** (collapsed) | **0.1% — INERT** | deleted from the code |

**The feared failure mode did not occur.** The model did not lazily prefer the collapsed
summary; it **ignored** it. Un-collapsed per-action blocks dominate; the collapsed aggregates are
dead weight. **The audit's delete list was EXECUTED — `gen3_op_block_trim_v1`:** incoming
per-status secondary (10 dims, 0.1%), incoming believed-effect (6 dims, 1.2%), the OUTGOING
slp/psn/tox columns (12 dims — *structural zeros*, measured: gen3 has no damaging move that
inflicts sleep, and the psn/tox carriers appear on 1 / 0 of the 773 pool teams), and the v30 lean
`_topk_block` (a strict subset of the incoming matrix, measured at **0 calls per forward**).
Removing the two collapses also took the unmasked belief read out of forward entirely, leaving
`w_all` as the op's single belief read. `damage_topk_k` now means "the incoming matrix's K", and
K>0 without the matrix RAISES in both the extractor and the op — never a silent empty block.
A third outcome worth naming: *a lossy summary usually doesn't become an attractive shortcut — it
becomes useless, and you pay compute for nothing.*

**The load-bearing caveat: a shortcut is attractive only when it is the best thing available.**
The collapsed blocks read as inert *because the un-collapsed ones sit right beside them* (the
audit flags this itself: marginal, not Shapley). Ship only `_chan_max` and the model would use
it — and you would get precisely the move-blind policy the worry predicts. **So the defense
against "it will just reason over the tail risk" is NOT to withhold the tail risk; it is to also
provide the per-action axis and let the collapsed version die on its own.**

**Compute is allocated inversely to dependence.** The INCOMING family is ~⅓ of dependence but
~64% of batch compute (`_damage_rolls` = 49% of the op at B=256) — which is what motivates
top-K candidate capping (v49 `--damage-candidate-k`).

### Corollary: top-K is TWO things, and only one of them is free

The axis rule cuts a distinction the flag names blur — "top-K" appears in our stack doing two
opposite jobs:

| | **Top-K as REPRESENTATION** | **Top-K as TRUNCATION** |
|---|---|---|
| Where | v30 `--damage-topk`, incoming matrix, ai_v9 **E4** | v49 `--damage-candidate-k` |
| Does | surfaces the K most-believed moves *individually* instead of a max | **drops** candidates below rank K from the physics sweep |
| Information | **un-collapses** the move axis — strictly more | **removes** mass — strictly less |
| Verdict | unambiguously good (this *is* the axis rule) | a real trade; needs a floor |

**Truncation's specific hazard is that its error is a CLIFF, not a slope.** The v49 probe
measured top-16 owning **94.2% of channels** — but the **misses are BIMODAL**: truncation loses
a candidate *entirely* rather than shaving its estimate. And the loss lands on our already-weakest
read — the OHKO belief is well calibrated overall (AUC 0.79) but **under-reads surprise OHKOs**
([[project_incoming_damage_outcome]]). v49 shipped with **no tail bound**, which is defensible as
a *learner* compute knob (+63.5% op at B=256, +0.3% at B=1) and wrong as the representation the
policy reasons over.

**The second-order effect is the one worth carrying:**

> **A full sweep makes belief quality a CALIBRATION problem. Truncation makes it a RANKING
> problem.** Full sweep: a mediocre belief under-weights a dangerous move — still priced, just
> discounted; degradation is smooth. Truncated: a mis-*ranked* belief means the move is not
> priced at all.

So truncation does not merely *benefit* from a better belief — it **creates the requirement** for
one. That obligation should be priced into the decision, and the ordering of fixes follows from
it, cheapest first:

1. **Raise K.** The ai_v9 feasibility spike settles this: the biased trunk at B=1 CPU is 0.183 ms
   at n=14 → 0.374 ms at n=50, sub-quadratic (dispatch-bound), so the roadmap's own conclusion is
   *"bench-K sizing is NOT compute-constrained in this range — choose K on belief-quality
   grounds, not budget."*
2. **Add the tail bound (E5) — STILL UNSHIPPED as of v56, and now the standing gap.** Stage 1
   landed E3/E4 (`gen3_entity_move_seats_v1`, `--entity-topk-seats K`) and Stage 2 landed the
   D3/S3 edge families priced at **the same detached top-K candidate selection** — so truncation
   is now load-bearing in *three* places (the op's candidate axis, the E4 seats, and the D3/S3
   edges) with no floor under any of them. The insurance is
   `[P(tail mass), tail worst phys, tail worst spec]` from a
   precomputed per-(type, category) max-BP table (damage is monotone in BP × effectiveness, so
   the tail *maximum* needs no sweep; expectation error ≤ tail mass). **General principle: when
   the failure mode is a cliff, prefer a sound BOUND over a better ESTIMATE** — a bound is
   correct regardless of belief quality.
3. **Only then, improve the belief** — and against a high build bar (below).

**The gate before any belief work** (offline, eval traces, no training): **recall@K** (of the
opp's moves actually used later, what fraction were in top-K at decision time) and, the one that
matters, **damage-weighted miss rate** (what fraction of *realized* damage came from a move
outside top-K). Note this also exposes an objective mismatch: the belief is trained with per-move
**BCE**, which optimizes average calibration — **not recall@K**. If the gate fails, that
mismatch is the first suspect, not capacity.

**And the record on "make the belief richer" is discouraging enough to demand the gate first:**
under `--belief-grad-mode detached` the belief heads **collapsed to chance** (latent cosine
0.004–0.013, species-acc → baseline) across a run that hit **best-ever 0.92 WR / ~1998 ELO**; the
belief-latent probe found species geometry strong but the **move-id table NONE**; and the standing
verdict on the family is *"LEARN but unmeasured if they HELP."* If the gate does fail, the
mechanism-bearing upgrade is the **factorization, not capacity**: the belief predicts *independent
per-move marginals*, but real movesets are archetypal (Spikes ⇒ probably Roar; Choice Band ⇒ not
Recover), so independent Bernoullis systematically produce incoherent top-K *sets* even when each
marginal is well calibrated. A set-level belief (low-rank mixture over moveset archetypes, or an
autoregressive factorization) targets exactly the quantity truncation depends on, and the
ingredients exist (Smogon usage, `gen3_learnset.json`, `data/teams/gen3_team_archetypes.json`).

---

## Part 4 — Four tests that discriminate laziness from genuine use

"The model likes it" is ambiguous evidence. In increasing strength:

**(a) Ablation-KL.** Says a block is *used*. Does **not** distinguish "used well" from "used as a
crutch." Partly OOD (zeroing produces states never seen), marginal rather than Shapley. The
standing caveat is the audit's own: **KL = USE, not VALUE — "learns ≠ helps."**

**(b) Linear probe on the trunk** (`python -m main.prober.query probe <run> <target>`,
[[project_representation_probe]]). Does the model *internally represent* the quantity, or only
read the answer? This gives the cleanest evidence for the laziness thesis:

- `damage_taken`: trunk r² only **+0.06** vs the provided feature's +0.02 — *"the rep barely
  encodes damage magnitude beyond the mean; the SPREAD is NOT there."*
- `is_faster`: trunk AUC **0.94** on contested states vs the provided feature's 0.75 — speed is
  genuinely computed internally, so a speed feature would be redundant.

**Where we hand it the answer, it doesn't build the concept; where we don't, it does.** That
contrast is the phenomenon, measured on our own model.

**(c) Behavioural counterfactual — the sharp test, NOT YET BUILT.** Construct state pairs with
the *same* collapsed summary but *different* move identity, where the correct pivot differs. If
the policy's pivot doesn't change, it is reasoning over the summary. Directly buildable on the
existing `reroll_many` / `better-line` / `replay-counterfactual` infrastructure; currently the
highest-value cheap diagnostic on this question.

**(d) Held-out generalization.** The real discriminator, because a shortcut and a genuine feature
are **identical in-distribution by construction**. They diverge only where the feature's coverage
breaks (unrevealed movesets, novel teams, off-meta sets). Our instance is the amortization-gap
line — [[generalist_specialist_amortization_gap]], [[amortization_gap_and_conditioning]].

---

## Part 5 — The reframe, and the levers

You cannot stop gradient descent being lazy. So:

> **Make the lazy path be the correct path.**

That is what the **pointer-native head** does, and it is worth stating in these terms. Under
a flat `Linear(latent, 11)`, the correct route (get move *k*'s physics to logit 6+*k*) requires
the projection to *learn an alignment* — the hard, starvable route — while the easy route is
"infer something vague from the pooled context." The pointer head makes the aligned route a
**direct linear read** of the cell attached to that action (`DamageOperator.pointer_cells`), and
the vague route comparatively harder. No capacity was added; the simplicity bias was re-ordered
in our favour.

Levers, in order of how much they're trusted here:

1. **Don't create the lossy summary.** Un-collapse the choice axis. (Done: v30/v34/v35/v39.)
2. **Structure delivery so the correct route is cheapest.** (Done: v51.)
3. **Feature dropout on the op block** — randomly zero it in training so the trunk must sometimes
   carry the load. The only one of the three that applies **pressure** rather than merely
   offering **availability**. *Hypothesis, not recommendation:* the closely-related "give the
   trunk the physics" interventions measured NULL three times (ledger K9/K10), and any new
   proposal of that family carries a standing HIGH build bar (say why it is a difference of
   content, not form, and bring a cheap pre-build probe). Dropout is arguably a different
   intervention class, which is the only reason it is listed at all.
4. **Enrich the objective instead of the input** — [[objective_richness_and_representation]].
5. **Stop-grad the easy path.** `--belief-grad-mode detached` is this pattern applied to the
   belief heads. **Caution from the record:** under `detached`, the belief heads collapsed to
   chance (latent cosine 0.004–0.013) across an entire run that nonetheless hit best-ever
   0.92 WR / ~1998 ELO — cutting a gradient can make the machinery inert without hurting, which
   is its own kind of finding.

### A standing confound: M1

SB3's `ActorCriticPolicy._build()` orthogonally re-initialized **13** Linears documented as
zero-init — including `refine_proj`, the trunk-injection path — in **every real run until
2026-08-01** ([[project_sb3_ortho_init_clobber]]). So "the trunk route is starved" is entangled
with "the trunk route was partly broken." This caveat stands over the whole K9/K10 trunk-delivery
result family and should be restated whenever those nulls are cited as evidence about laziness.

---

## Part 6 — "Magnitude needs an entity home": the concat end-state

This part is the live case, and it is the most instructive one in the note because **the feared
mechanism did not fire** and the interesting question turned out to be structural rather than
about laziness at all.

### The setup

Computed physics reaches the policy by **three** routes today:

| Route | What it is | Bandwidth per forward |
|---|---|---|
| **Flat head-concat** | the op's ~807-dim block concatenated into *both* projection heads (`ProjectionAssembler`) | ~807 float dims, wired straight in, no bottleneck |
| **Edge biases** (v56) | per-pair cells → zero-init `Linear(cell → 2·n_heads)` → additive attention-logit bias | **n_heads scalars per pair per direction**, then softmax-normalized |
| **Pointer cells** (v51) | `DamageOperator.pointer_cells` concatenated onto the entity token that a logit selects | full cell, affine, per-logit — but read through one shared zero-init scorer |

### What we measured (three times)

The audit arm zeroes one route on ~4000 real eval states and reports masked KL / argmax-flip %
/ |ΔV| against the model's own distribution.

| Checkpoint | concat arm | ALL edges off |
|---|---|---|
| gen-1 @40M (6 families) | kl **0.482** / **35.5%** flips / **\|dV\| 7.45** | 0.330 / 26.9% / 2.51 |
| gen-2 @40M (11 families) | kl **0.537** / **33.1%** / **\|dV\| 7.44** | 0.491 / 31.5% |
| gen-2.5 @25M (15 families) | **31.4%** | 14.3% (mid-curve — 25M, not 40M) |

Two facts have to be held **at the same time**:

1. The edges are **real and growing** — gen-1's all-off dependence grew ~3× from 9.6M → 40M,
   and gen-2's all-off (31.5%) is larger than gen-1's (26.9%) with more families.
2. They **added on top of** the concat rather than replacing it. `concat_cells` (the op fully
   out of the heads) reads 0.650 / 40.4% — the head route is still the single largest thing the
   policy depends on.

So: **no starvation.** The easy path did not prevent the hard path from developing.

### Why not? (the general rule)

Gradient starvation (Pezeshki 2021) is a statement about **substitutes**. Its mechanism needs
two conditions:

- the two features **explain the same residual** — they are correlated/redundant in what they
  predict, so once one is fitted the other's gradient shrinks toward zero; and
- the loss actually **approaches its floor**, so there is little residual gradient to compete
  over.

Our stack violates both.

- **Different jobs.** The concat delivers *how much*; the edges deliver *who should look at
  whom*. A routing improvement reduces error that the concat's magnitudes cannot reduce, so the
  edge gradient never goes to zero.
- **PPO's gradient does not decay like a supervised loss.** Advantages are recentered (and
  normalized) per batch, and self-play is **non-stationary** — the opponent pool keeps moving,
  so fresh error keeps arriving. Starvation's "the loss is already low" premise is much weaker
  in on-policy RL than in supervised learning. (This is the *opposite* sign to the ~1-bit-per-game
  argument in Part 1: RL is sample-starved but not *gradient*-saturated.)
- **Zero-init ≠ starved-at-init.** A zero-init map has zero *output* but a **non-zero weight
  gradient** (∂out/∂W = input × upstream grad). Identity-at-init buys a clean A/B without
  costing the path its funding. (Standing caveat: until 2026-08-01 SB3 clobbered those inits —
  [[project_sb3_ortho_init_clobber]], ledger **M1**.)

> **Rule to carry:** two delivery paths compete when they are *substitutes* and complement when
> they are *complements*. Before worrying about starvation, ask what function each path is
> uniquely able to compute — if you cannot name a function only the hard path can express, you
> have built a substitute and starvation is the right worry.

### The structural argument — an edge bias cannot carry an absolute

Softmax attention computes, for query *i*:

&nbsp;&nbsp;&nbsp;&nbsp;`out_i = Σ_j α_ij · V_j`, with `α_ij = softmax_j( q_i·k_j/√d + b_ij )`

The edge bias `b_ij` enters **only** through the logits, and the logits pass through a softmax.
That has three consequences, in increasing importance:

1. **The output is a convex combination of the values.** `α ≥ 0`, `Σ_j α_ij = 1`. So `out_i`
   lives in the **convex hull of `{V_j}`**. No bias, however large, moves the output outside
   that hull. If all the value vectors are equal, the bias is *exactly* a no-op.
2. **The code is relative, not absolute.** `α_ij` depends on `b_ij` **and on every other `b_ik`
   in the row** (the partition function). The same "53% of max HP" edge produces a different
   `α` depending on what else is on the board. What survives softmax is a **ranking / ratio**;
   what is destroyed is the **scale**.
3. **It saturates, and it shares a channel.** Softmax is monotone but squashing: past a few
   nats the bias stops changing `α`. And the same scalar channel is what content-based routing
   (`q·k`) uses — magnitude and relevance are forced to share one number per head. Downstream,
   the residual stream is LayerNormed, which further removes scale.

Contrast the two channels that **can** carry an absolute:

- **Token content (values).** Write the number into `V_j` — e.g. `prefuse_proj` injecting the
  op's per-our-mon incoming rows onto our role tokens (v50). Now any query that attends to that
  token receives the number **linearly**, and an MLP can compute a genuine **threshold**
  ("is this ≥ my remaining HP?"). Thresholds are the operation that matters here — P(KO) is a
  comparison of two absolutes — and a comparison is exactly what a normalized weight cannot do.
- **Per-action cells at the logits.** `pointer_cells` concatenates the cell onto the selected
  entity's token and scores it affinely. Lossless per-logit, zero interference with routing.

**Is the argument airtight? No — and the honest version is more useful.** Attention *can*
smuggle magnitude, and it is worth knowing how, because those routes are the ones a model would
have to discover:

- `α_ij` is itself a continuous scalar in (0,1), so if `V_j` contains a dedicated "one unit of
  damage" direction, the residual picks up `α_ij` along it. A model **can** allocate a head
  whose keys are near-constant so the bias row alone determines `α` — a "magnitude head." The
  cost is a whole head, and the readout is still the **softmax-normalized** row, so the decoded
  quantity is `damage_ij / Σ_k f(damage_ik)` — recoverable only up to that row-dependent
  denominator, and only in the softmax's non-saturated regime.
- Multiple heads/layers could in principle triangulate the denominator away. That is a
  *learnable* but ill-conditioned function, competing for the same parameters as routing.

So the correct claim is **not** "impossible" — it is a **capacity-and-conditioning** claim:
*the edge channel carries ~n_heads normalized scalars per pair, in a relative code, on a
saturating nonlinearity, in competition with routing; the concat carries ~807 unnormalized
floats with no bottleneck.* Expecting the first to absorb the second is expecting a very
inefficient encoding to win a race against a free one. **That framing is falsifiable**: if the
residual concat dependence localizes to blocks the pointer cells already carry losslessly, the
magnitude story is *wrong* and something else (see the confounds) is the cause.

### The pre-registered decision rule

Recorded in `designs/ai_v9/design_generation_roadmap.md` on 2026-08-07, **before** gen-3's
end-of-run audit:

- **Branch A — concat flips < the all-edges-off arm** ⇒ the entity paths absorbed the role ⇒
  **mask → A/B → delete**, then the Stage-3 CPU refund.
- **Branch B — concat holds ≥ all-edges-off (a fourth replication)** ⇒ read it as *magnitude
  still has no entity home of equal fidelity* ⇒ **widen token-content delivery** (generalize
  `prefuse_proj` to inject the full per-mon op rows onto **both** sides' tokens) **and** run
  **per-sub-block concat ablations** to localize the residual; then re-audit. Delete only once
  the re-homed form matches the concat's measured contribution.
- **Explicitly not allowed:** *"wait longer"* or *"delete anyway."*

**Why branch A is the right inference — and why it is still not enough to delete.** Ablation-KL
is a **marginal** measure at **fixed weights**: "if I remove this input from the trained
network, does behaviour change?" A collapse to below the all-edges arm demonstrates
**redundancy** — the information is available elsewhere *to a network that was trained with the
concat present*. It does **not** demonstrate that a network trained **without** it reaches the
same policy. Hence the ordering `mask → A/B → delete`: the mask arm is cheap and reversible, the
A/B is the actual test, deletion is the irreversible bookkeeping afterwards. (Test **(a)** in
Part 4: KL = **use**, not value.)

**Why branch B's response is right even though its *reading* might be wrong.** The measurement
says only: *"we failed to demonstrate absorption."* The **action** — widen the channel that
provably can carry absolutes, and localize the residual — is correct under several competing
explanations, which is what makes it a good hedged response. The *causal story* is what needs
testing, and the sub-block localization **is that test**, not merely targeting:

- residual localizes to **per-mon incoming rows** ⇒ consistent with the magnitude story ⇒
  widening `prefuse` is aimed correctly;
- residual localizes to the **OUTGOING per-move blocks** ⇒ the magnitude story is **damaged**,
  because `pointer_cells` already delivers exactly those numbers per-logit. Then the cause is
  something else — plausibly that the pointer scorer is a *narrow shared affine read* while the
  concat feeds the full projection MLP (a **capacity of the readout** problem, not a *delivery*
  problem), which implies a different fix (widen the scorer / give it a small MLP), not a
  trunk-injection one.

### Four ways branch B's read could be wrong

1. **The value head — and we already have evidence for this one.** The concat feeds **both**
   projection heads; the pointer head serves only the policy, and the critic has *no* pointer
   path at all. The audit shows the concat arm's **|dV| = 7.45 vs the whole edge system's 2.51**
   — the critic is hit ~3× harder than by every edge combined. So a large part of what keeps the
   concat alive may be the **value** objective, measured through a **policy** metric on a shared
   trunk. *Distinguishing experiment (cheap, offline, decisive):* run the concat ablation
   **per head** — zero it in the `pi` projection only, then the `vf` projection only — and
   report flips and value error separately. If the policy-only arm is small, "magnitude has no
   entity home" is really "the **critic** has no entity-native readout," and the fix is a
   value-side pooled physics read, not a wider prefuse.
2. **Optimization path-dependence (first-mover).** The concat is at full width from step 0;
   every edge map is zero-init and has to grow. Whoever is available first can own the function
   permanently without being structurally better. *Distinguishing experiment:* a run with the
   concat **masked from step 0** (or annealed out on a schedule). If edges+pointer alone reach
   comparable anchored ELO, the structural claim is refuted and it was a race, not a capacity
   limit. This is the single most decisive experiment available, and it is a retrain.
3. **Mid-training vs converged.** Dependence **grows with training** (gen-1: ~3× from 9.6M →
   40M; gen-2.5's 14.3% all-off at 25M is explicitly mid-curve). A fixed-step comparison of two
   arms that are both still moving is a **snapshot of a race**, not its limit. This is precisely
   why *"wait longer"* is banned as a *response* — but it is a legitimate *caveat* on the read,
   and the answer is to compare arms at matched tranches, not to defer the decision.
4. **The two arms are not perturbation-matched.** Zeroing ~807 input dims to a Linear is a much
   larger, much more off-manifold intervention than zeroing a set of small additive logit
   biases — attention still runs in the second case. Comparing them is comparing two
   differently-sized hammers. *Fixes, cheap:* (i) **mean-substitution** instead of zeroing
   (replace with the dataset mean so the input stays on-manifold — standard interpretability
   practice); (ii) the **shuffle control** the P1 probe already used (shuffle the block across
   states: preserves marginals, destroys state-specificity); (iii) report a **both-off** arm so
   the interaction term (redundancy vs necessity) is identified rather than assumed.

A fifth, smaller one: with partially-redundant paths, *marginal* ablation systematically
**understates** each path (each looks droppable alone). That biases toward deletion, not against
it — so it does not threaten branch B, but it does mean branch A's threshold should not be read
as a Shapley value.

### One gap in the rule as written

Branch A pre-registers the **sequence** (mask → A/B → delete) but not the **acceptance
criterion** for the A/B. That leaves a forking path exactly where it matters. It should be a
**non-inferiority margin fixed in advance** — e.g. *"delete only if anchored ELO at 40M is
within −15 of the concat-on arm with a CI that excludes a −40 regression."* Without a number,
"the A/B looked fine" is a post-hoc judgement of the same kind pre-registration exists to
prevent.

### Why pre-registration matters here specifically

Generic reason: **researcher degrees of freedom**. An ablation study has many — which
checkpoint, which state distribution, KL vs flips vs |ΔV|, which baseline arm you compare
against, what counts as "large." Post-hoc, *any* result can be narrated into the conclusion you
already preferred; that is the "garden of forking paths" (Gelman & Loken) and HARKing
(hypothesizing after results are known). Writing the if/else down converts the analysis from
**exploratory** to **confirmatory** — pre-registration does not forbid exploring, it forbids
*relabelling* exploration as confirmation.

Three ML-specific sharpenings:

- **The two universal escape hatches.** Almost every ablation dispute ends in one of *"the run
  wasn't long enough"* or *"the metric doesn't capture it, ship it anyway."* Both are
  unfalsifiable and both are available *whatever the number is*. Naming and banning them in
  advance is most of the value of this particular pre-registration.
- **Pre-register the RESPONSE, not just the threshold.** A threshold alone still permits
  "measurement says B, but let's do A because it's cleaner." Binding each branch to an action is
  strictly stronger.
- **Goodhart.** The moment an ablation number becomes the deletion criterion, it becomes
  optimizable — e.g. training with concat dropout would *lower* the concat arm without the
  entity paths becoming any better. Pre-registering also means pre-registering **that the
  training recipe is held fixed** between the arms being compared.

### The endgame — and why it *dissolves* rather than answers

Stage 3 removes the flat observation vector entirely, so every fact's only delivery is
entity-attached: the Part-5 reframe applied to the input ("make the lazy path **be** the entity
path"). Note what that does epistemically:

> Removing the shortcut does not tell you whether the entity path was as good. It removes the
> **competition**, so the question stops being askable.

That is genuinely valuable — starvation has nothing left to feed on, one delivery convention
holds for every future fact, and the compute comes back. But the risks are real and worth
stating plainly:

- **You may lose capability and not be able to see it.** With no concat arm, there is no A/B —
  a regression will be attributed to whatever shipped next. *Mitigation: keep a concat-on
  control arm inside the same generation.*
- **Removing a crutch does not teach walking.** "No shortcut" is not a mechanism for making the
  entity path good; it can equally produce a strictly worse model that merely has a tidier
  architecture.
- **It is retrain-class and irreversible in practice.** Which makes the A/B expensive, which
  makes skipping it tempting — the exact pressure pre-registration exists to resist.
- **Path dependence cuts both ways.** Delete it and every subsequent result is measured on the
  new stack; the counterfactual gets more expensive with every generation.

The middle path worth taking seriously is **scheduled removal** rather than a cliff: anneal
concat dropout upward across training. That applies *pressure* (Part 5, lever 3) instead of
merely offering *availability*, keeps a measurable capability floor while the entity paths grow,
and yields a **dose-response curve** — which is far more informative than a single on/off A/B.

---

## Part 7 — A case where delivery is NOT the cause: incoming vs outgoing

Worked example of how to **rule out** the delivery hypothesis, because the instinct "it reads low
because we hand it to the head" is the first thing this note makes you reach for — and here it is
wrong.

**The observation.** In human gen3ou, incoming damage drives the core pattern *defensive pivot →
offensive pivot*. Our model prices incoming far below outgoing:

- P1 concat blocks: OUTGOING per-action **65.7%** of the ceiling, v39 attacker matrix 21.4%,
  v34 defender matrix 6.3% — against incoming matrix 15.4%, incoming per-mon 12.7%, CB 2.9%,
  incoming effect 1.2%, incoming secondary 0.1%. Roughly **⅓ of dependence, ~64% of the op's
  batch compute.**
- gen-2 trained edge audit: d2 **23.8%** flips and d1 10.1% (both OUTGOING) against d4 2.1% and
  d3 ≈ decorative (both INCOMING).

**The decisive test — the asymmetry is INVARIANT to delivery form.** Incoming and outgoing sit in
the *same* concat, and the concat's own blocks differ 3–5×. A property of the channel cannot
explain a difference *within* the channel. Then the edges reproduce the same ordering in a
completely different channel. Two independent delivery forms, same ranking ⇒ **the cause is the
content, not the route.**

> **General rule: to test "the delivery form caused it," find a second delivery form. If the
> effect replicates across forms, the form isn't the cause.**

**Four candidates that do survive, ranked by evidence:**

1. **Information asymmetry (strongest).** Outgoing physics is EXACT — our moves, our stats.
   Incoming is BELIEVED. And the measured failure lands exactly there: across healthy-mon
   single-turn OHKO deaths the belief fired (pko ≥ 0.7) only ~10–20% while **under-reading
   (pko < 0.3) ~53–61%**, because the killer had just switched in with an unrevealed moveset.
   Down-weighting a channel that goes silent precisely when it matters is **correct Bayesian
   behaviour**, not laziness.
2. **Counterfactual credit assignment.** Offense produces a realized event (they faint; material
   PBRS fires). Defense produces a **non-event** — the damage you didn't take. A return-based
   estimator sees the first densely and the second only diffused into later returns. This is the
   documented defensive/positional value blindness that motivated v43 `--pubval-mode`.
3. **It is a TWO-PLY PLAN, i.e. the amortization gap.** *Defensive pivot → offensive pivot* pays
   at ply 2; ply 1 alone looks bad (tempo surrendered, chip taken). A one-step amortized policy
   scores each decision on its own expected value. More *information* about incoming damage
   cannot manufacture a plan — see [[pbs_value_functions_and_search]] §0.5, and note that every
   null in this family varied information while none varied computation per decision.
4. **Self-confirming measurement.** Ablation dependence is measured on the policy's OWN state
   distribution. An offense-leaning policy visits states where offense decides, so a feature that
   would matter for a behaviour it doesn't exhibit reads inert. Dependence answers *"does this
   change what I do here,"* not *"would this matter if I played differently."*

**The premise may also be partly an aggregation artifact.** Conditioned on the threat being real
(slower), switch-rate climbs monotonically **22% → 38%** with pko, while conditioned on faster it
is correctly flat ~20%. So the policy *does* respond to incoming where it fires; the aggregate
dilutes it. And **switch RATE is a first-moment match that says nothing about pivot quality** —
the model now switches 31.4% vs strong humans' 28.0% (under-switching is RESOLVED), which leaves
*which* pivot and *in what sequence* entirely unmeasured.

### MEASURED 2026-08-07 — gen-3 @9.6M

Sources: [`gen3_op_block_dependence_6k.json`](../research_state/measurements/gen3_op_block_dependence_6k.json)
and [`gen3_oracle_belief_voi.json`](../research_state/measurements/gen3_oracle_belief_voi.json)
(both `run_20260807_135637_gen3` @ `checkpoint_9600000_steps.zip`; the probe scripts were
one-offs in a gitignored `tmp/` and were never committed — see the measurements README).

**FIRST — the premise itself is stale for this generation.** gen-2/gen-3 run the incoming matrix
but **NOT** the v34/v39 outgoing matrices (op `out_dim` 660 = incoming 85 + outgoing/status 53 +
`in_matrix` 522), so P1's "outgoing dominates" was measured on a block set this generation does
not have. Width-fair (**shuffle** control: permute the slice across states — same width, same
marginals, state-specificity destroyed), 6000 states:

- `in_matrix` (incoming per-move, 522d) — **16.27%** flips
- `out_active` (outgoing per-move, 45d) — 6.25%
- `in_permon` (incoming per-mon, 72d) — 4.52% · `in_cb` 1.28% · `out_status` 1.00%
- whole concat — 18.58%

So **in the HEAD CONCAT, incoming DOMINATES** (~2.6× the outgoing block), while in the EDGES
outgoing dominates (d2 23.8% / d1 10.1% vs d4 2.1% / d3 decorative). That division of labour is
exactly what Part 6's channel argument predicts: **absolutes** ("does this kill me") ride the
channel that carries absolutes; **relational ranking** ("which target, which move") rides the
channel that carries ratios. Note also how much the zero-vs-shuffle gap matters — `in_matrix`
reads 24.15% zeroed but 16.27% shuffled, i.e. **a third of the naive number was the mean shift
from blanking 522 dims**, which is the perturbation-mismatch confound Part 6 names.

**THE RELATIVE WEIGHT ON gen-3 @9.6M** (`edge_ablation_audit`, the SAME 6000 states). Against the
`concat_cells` ceiling (the op fully out of the heads: kl 0.637 / 37.8% flips):

- **op head-concat** kl 0.244 / **23.7%** flips / |dV| **5.67** → **38% of the KL ceiling**
- **all 15 edge families** kl 0.101 / **13.9%** flips / |dV| 1.86 → **16%**
- families, by flips: **d2 7.63% · d1 6.05%** · v 2.90% · d3 1.85% · d4 1.10% · s3 0.85% ·
  s1 0.83% · t 0.65% · c1 0.38% · c2 0.35% · x 0.33% · c3 0.25% · c5 0.23% · g 0.12% · c4 0.05%
  (d1+d2 alone are 76% of the summed edge KL; the individual rows sum well above the 13.9%
  all-off arm, i.e. the families are **redundant**, and marginal ablation understates each)

Split by DIRECTION, and the two channels disagree by construction:

- inside the **concat** (shuffle-controlled flips): incoming 22.1 vs outgoing 7.3 ⇒ **~75%
  incoming**
- inside the **edges** (summed KL): outgoing ≈ 0.080 vs incoming ≈ 0.003 ⇒ **~92% outgoing**
- composed (illustrative — marginal arms do NOT partition, and concat+edges overlap):
  **≈50% outgoing / ≈48% incoming / ≈2% speed-board**, with each direction living in the channel
  that suits it.

The critic split replicates gen-1/gen-2: |dV| **5.67 concat vs 1.86 all-edges** (base V σ 14.9) —
the value head leans on the concat ~3× harder than on every edge combined.

**EXPERIMENT 1 — the dilution hypothesis is REFUTED.** Restricting to THREAT states (slower ∧
active pko ≥ 0.5 ∧ legal switch ∧ a safe pivot ≤ 0.35 — 8.1% of states, mean pko 0.87) leaves
shuffle-controlled dependence **unchanged**: threat/all flip ratios `in_permon` **1.02×**,
`INCOMING_all` **1.06×**, `out_active` 1.02×, `in_matrix` 0.76×. There is no concentrated
signal hiding under the average. The policy *is* behaviourally responsive in those states —
switch mass **0.529 → 0.715**, entropy 1.271 → 1.084 — so the old inversion really is gone.

**EXPERIMENT 2 — information is NOT the binding constraint.** A **look-ahead oracle** (per
battle, the union of every move each opponent species is ever seen using, fed through
`MoveBelief.move_logits`' existing reveal-pinning path) on the 44% of states where the OPP
ACTIVE gains ≥1 move: policy KL **0.128**, **19.3% argmax flips**, |ΔV| 1.62 (base V σ 14.9),
active P(KO) 0.2575 → **0.2917**. So the physics updates and the policy genuinely moves. But:

- **switch mass 0.4817 → 0.5003 (+0.019)** — better incoming information flips a fifth of the
  actions and shifts *pivoting* by under two points;
- **head dependence on the incoming blocks does NOT rise** in the oracle world (`in_permon`
  7.00% → 7.45%, `INCOMING_all` 7.15% → 7.45%, `in_matrix` 30.1% → **29.6%**).

⇒ cause **1 (belief noise) is not supported as the reason incoming looks under-used**: making the
belief more reliable does not make the head lean on it more, and does not make the policy more
defensive. The weight shifts to **2 (counterfactual credit assignment)** and **3 (the two-ply
plan)**.

**Caveats, held honestly.** gen-3 @9.6M is EARLY (gen-1's dependence grew ~3× from 9.6M → 40M),
so levels are mid-curve — but the conditional/oracle comparisons are *within* one checkpoint,
which is what those two experiments need. The oracle is **partial** (moves never used all game
are absent, +0.20 moves per revealed mon overall), so every VoI number is a **lower bound**. And
ablation KL is still **use, not value**.

**Still open — the third experiment.** `better-line --depth 2+`: does the search find
pivot-then-threat lines the policy misses, and do they win? That measures the *plan* gap rather
than the *information* gap — cause 3, whose lever is a teacher, not an obs block. The
**win-rate** half of experiment 2 is also unrun: it needs the oracle threaded through live
battles, a substantially bigger build than the offline probe.

---

## Synthesis

Feeding a computed feature straight to the head is a **plus** exactly when the feature is
correct, sufficient for the decision at hand, and cheap — then the model's laziness is
*amortization*, and the capacity it saves goes somewhere useful. It is a **minus** exactly when
the feature is a lossy summary over an axis the policy must choose along, because then laziness
is not the model's failing but the encoding's.

The practical discipline is three rules:

1. **Sufficiency, per head.** Ask separately whether the feature is sufficient for `V(s)` and for
   `π(s)`. They have different answers surprisingly often; the worst-case threat summary is the
   canonical case.
2. **Never collapse a choice axis.** And when you do provide a collapsed convenience channel,
   expect the ablation to show it inert — then delete it.
3. **Bias the simplicity bias.** Since the easiest route wins, spend the design effort making the
   structurally-correct route the easiest one, rather than trying to force the model up the hard
   one.
4. **Ask what a channel can physically CARRY, not whether it is expressive in principle.** A
   normalized channel (softmax weights) transmits ratios; absolutes need token content or a
   logit-adjacent read. Two paths compete only if they are substitutes — name the function only
   the hard path can compute, or accept that you built a substitute.
5. **Pre-register the branch AND the response, with the escape hatches named.** For an ablation
   that decides a deletion: fix the metric, the arms (perturbation-matched, with a
   mean-substitution or shuffle control and a both-off interaction arm), the threshold, the
   non-inferiority margin for the follow-up A/B, and an explicit ban on *"wait longer"* /
   *"do it anyway."*

## See also
- [[pbs_value_functions_and_search]] §0.5 — **the one-level-up version of this note**: why the
  model reasons in expectations rather than discrete lines (every gradient we give it is smooth),
  amortization vs deliberation, "it declines to spend capacity" (not "spends it on heuristics"),
  the policy shaping its own state distribution to protect its heuristic, and the observation that
  every null we hold varied *information* while none varied *computation per decision*
- [[objective_richness_and_representation]] — the **output-side dual**: richer targets force
  richer representations (rank collapse, implicit under-parameterization, distributional targets)
- [[entity_tokens_biases_pointers]] — the sorting rule (token / edge / distribution summary /
  attention), the differentiable expert, the pointer head
- [[marginalization_and_uncertainty]] — why the operator marginalizes rather than mean-fields
- [[on_policy_self_distillation]] — the ~1-bit-per-game accounting and the dense-signal alternative
- `designs/ai_v9/design_generation_roadmap.md` §3 — the **concat end-state decision rule** as
  recorded pre-gen-3, plus the gen-1/gen-2/gen-2.5 edge-and-concat audit numbers Part 6 cites
- `designs/research_state/ledger.md` → K9 / K10 / K10a / P1 / M1 — the primary result records
- `src/main/prober/CLAUDE.md` — the `probe` representation-probe harness (test **b** above)
