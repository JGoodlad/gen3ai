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
  correct path.** That is exactly what the v51 pointer head does.
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

**P1 ablation probe** (`tmp/op_block_ablation_probe.py`, 2026-07-25; 4000 real eval states, exact
producing snapshot, per-block zero → masked KL against the policy's own distribution; a SHUFFLE
control exceeded zeroing everywhere, so the head reads state-specific content):

| Block | % of the zero-whole-op ceiling |
|---|---|
| OUTGOING (per-action, un-collapsed) | **65.7%** (75% of the *moves* ceiling) |
| v39 outgoing-attacker matrix | 21.4% |
| v35 incoming matrix (per mon × move) | 15.4% |
| incoming per-mon | 12.7% |
| status-landing | 8.8% |
| v34 outgoing matrix | 6.3% |
| Choice-Band | 2.9% |
| incoming **effect** (collapsed) | 1.2% |
| incoming **secondary** (collapsed) | **0.1% — INERT** |

**The feared failure mode did not occur.** The model did not lazily prefer the collapsed
summary; it **ignored** it. Un-collapsed per-action blocks dominate; the collapsed aggregates are
dead weight. **The audit's delete list was EXECUTED — v55 `gen3_op_block_trim_v1`:** incoming
per-status secondary (10 dims, 0.1%), incoming believed-effect (6 dims, 1.2%), the OUTGOING
slp/psn/tox columns (12 dims — *structural zeros*, measured: gen3 has no damaging move that
inflicts sleep, and the psn/tox carriers appear on 1 / 0 of the 773 pool teams), and the v30 lean
`_topk_block` (a strict subset of v35's incoming matrix, measured at **0 calls per forward**).
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
| Where | v30 `--damage-topk`, v35 incoming matrix, ai_v9 **E4** | v49 `--damage-candidate-k` |
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
   landed E3/E4 (v54 `gen3_entity_move_seats_v1`, `--entity-topk-seats K`) and Stage 2 landed the
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

That is what the **v51 pointer-native head** does, and it is worth stating in these terms. Under
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

## See also
- [[pbs_value_functions_and_search]] §0.5 — **the one-level-up version of this note**: why the
  model reasons in expectations rather than discrete lines (every gradient we give it is smooth),
  amortization vs deliberation, "it declines to spend capacity" (not "spends it on heuristics"),
  the policy shaping its own state distribution to protect its heuristic, and the observation that
  every null we hold varied *information* while none varied *computation per decision*
- [[objective_richness_and_representation]] — the **output-side dual**: richer targets force
  richer representations (rank collapse, implicit under-parameterization, distributional targets)
- [[entity_tokens_biases_pointers]] — the sorting rule (token / edge / distribution summary /
  attention), the differentiable expert, the v51 pointer head
- [[marginalization_and_uncertainty]] — why the operator marginalizes rather than mean-fields
- [[on_policy_self_distillation]] — the ~1-bit-per-game accounting and the dense-signal alternative
- `designs/research_state/ledger.md` → K9 / K10 / K10a / P1 / M1 — the primary result records
- `src/main/prober/CLAUDE.md` — the `probe` representation-probe harness (test **b** above)
