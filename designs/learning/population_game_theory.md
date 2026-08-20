# Population game theory — PSRO, Nash averaging, and spinning tops

*The flywheel era's native language: what "stronger" means when strength is a matrix, not a
number. Two levels: intuition first, then the formal objects, grounded in this project's own
machinery and measurements.*

## 1. Intuitive: strength is a matrix

A single number ("ELO 2056") pretends the game is **transitive**: if A beats B and B beats C,
A beats C. Real strategy games are only *mostly* transitive. Team matchups in ADV OU carry
genuine rock-paper-scissors structure — a stall team farms one archetype and folds to another —
so the true object is the **payoff matrix** between policies, and any single number is a lossy
projection of it.

Three of this project's standing facts are population-game facts in disguise:

- **`win_rate_vs_pool` pins near 50%** — not because progress stopped, but because the promotion
  gate *constructs* the pool as a sliding window of recent selves. A self-play pool is a moving
  mirror; the number measures the gate, not the skill.
- **The anchored Bradley-Terry ladder** (`eval/elo`, `snapshot_ladder/ladder.json`) is a
  transitive FIT over the matrix. It recovers the transitive component well (which is why it is
  the generation gate) and is structurally blind to cycles — two snapshots with equal ELO can
  have a lopsided head-to-head. **The width instrument now exists**: `agents/training/hodge.py`
  splits the same matrix into spine + cycle and tests the cycle against its own binomial noise
  floor — read it offline with `python -m main.elo <run>` (the live `eval/hodge_width_elo` /
  `eval/hodge_cyclic_fraction` scalars are the weak per-cycle counterpart); details in
  `src/agents/training/CLAUDE.md` → *Hodge decomposition*. First reading: gen-15's ladder is ~96%
  spine but carries **46 ELO of excess width (p = 0.005)** and three significant snapshot
  3-cycles, so the cycles discussed below are a measured quantity here, not an analogy.
- **The exploiter random walk** (ai_v10 §9: "past N teams the restoring force is too small and we
  randomly walk") is what walking the *non-transitive* dimension feels like from the inside —
  effort spent circling a cycle instead of climbing the spine.

## 2. The formal objects

**Empirical game / meta-game.** Freeze a set of policies, play them pairwise, record the win
matrix. That matrix IS a normal-form game whose "actions" are policies. Every pool decision —
who to train against, how to weight them, when to promote — is a move in this meta-game, whether
or not it is analyzed as one.

**Nash averaging** (Balduzzi et al. 2018, *Re-evaluating Evaluation*). Uniform-average win rate
against a pool is corrupted by REDUNDANCY: add ten copies of a weak bot and every policy's
average rises, no skill required. Evaluating against the **Nash equilibrium mixture** of the pool
instead is invariant to duplicates and to adding dominated members. This is the formal version of
this project's ELO-reading discipline (matched snapshot counts, anchor bots, "never quote a
mid-run delta") — and the upgrade path if pool composition ever becomes contested: weight
opponents by the meta-game Nash, not by count.

**Spinning tops** (Czarnecki et al. 2020). Real games have the geometry of a top: a long
**transitive spine** (skill) wrapped by **non-transitive width** (strategy cycles), and the width
is LARGEST at middling skill, narrowing toward both the floor and the optimal-play tip. Three
consequences that map straight onto this project:

1. Mid-ladder, population DIVERSITY is load-bearing — a learner training against too narrow a
   pool climbs into a cycle wall it cannot see. (PFSP and pool spread are the levers.)
2. Near the tip, diversity matters less and pure spine-climbing more — expect the value of
   exploiter breadth to *fall* as the generalist strengthens, which is a testable trend, not a
   disappointment.
3. A flat ELO with rising exploitability means motion along the width, not the spine — the
   flywheel's kill condition ("two revolutions flat ELO AND flat piloting") is a width-vs-spine
   diagnostic stated operationally.

**PSRO** (Policy-Space Response Oracles, Lanctot et al. 2017). The general loop: keep a
population; a **meta-solver** computes a mixture over it (uniform, Nash, or weighted); an
**oracle** trains a best response against that mixture; add the response to the population;
repeat. Now read the tick-tock in those terms: the **tock's exploiters are the oracle**
(best-responders against a slice-restricted distribution), the **pool + PFSP weighting is the
meta-solver** (PFSP ≈ the "rectified"/weighted-response family AlphaStar used), and the **tick's
distillation is a projection step PSRO does not have** — PSRO grows the population forever; the
flywheel compresses it back into one deployable policy. That compression is the novel part and
the part that owes evidence (headroom capture is its per-fold instrument).

**Exploitability.** The distance-to-equilibrium metric: how much can ANY best response gain
against you? Unmeasurable exactly (needs the true best response), but every exploiter run
produces a LOWER BOUND: `exploiter_wr − baseline` against the frozen generalist is measured
exploitability on that slice. A generalist whose ELO rises while its exploiter lower bounds also
rise is climbing the spine while leaking width — both numbers, always.

## 3. What this buys operationally

- Read `win_rate_vs_pool` as a gate diagnostic, never progress; read ELO as the spine only.
- When pool composition decisions get contested (revolution 3+), the principled tools are
  meta-game Nash weights and per-slice exploitability bounds, not uniform averages.
- The archetype-novelty regression (T1.5) and headroom capture are QD/width instruments; the
  dense ladder is the spine instrument. Keeping them separate is keeping the top's two axes
  separate.

**The question you can answer after this note:** *when the flywheel's ELO goes flat, how do you
tell "converged" from "circling a cycle"?* — check whether fresh exploiters still find headroom
(exploitability lower bounds still fat ⇒ width remains; thin everywhere ⇒ the spine is genuinely
exhausted at this capacity).

Related notes: [`entity_tokens_biases_pointers.md`](entity_tokens_biases_pointers.md) §what
search would look like (the equilibrium framing); the flywheel design
(`../ai_v10/design_flywheel_tick_tock.md`) is the operational instance of this whole note.
