# Imperfect information and equilibria — what the belief stack approximates

*CFR, public belief states, and why optimal play in a simultaneous-move hidden-information game
is an equilibrium rather than a best line. The theory our belief stack and α/β heads are an
empirical approximation of.*

## 1. Intuitive: the opponent's uncertainty is part of the board

In chess, the state is the board. In a hidden-information game, the state includes **what each
side believes** — and because actions leak information (revealing a move updates the opponent's
posterior), every action has two payoffs: its board effect and its information effect. Bluffing
exists because of the second payoff. And because both sides move simultaneously, "the best move"
is not even well-defined against a fixed opponent: if my play is deterministic, the opponent's
best response punishes it, so optimal play **randomizes** — the object of correctness is an
equilibrium (a pair of strategies stable against each other), not a best path.

Pokémon at one ply is exactly this: the turn is a matrix game (my 9 actions × their ~5), the
hidden team is the private information, and the pivot/attack mind-game (the bait-loop pathology's
home) is its purest expression.

## 2. The formal objects

**Information sets and CFR.** Poker-style solving groups states the player cannot distinguish
into INFORMATION SETS and minimizes regret per set (Counterfactual Regret Minimization) —
converging to Nash in two-player zero-sum. CFR itself is not our tool (the game tree is too rich
and we forbid inference-time search), but its vocabulary is load-bearing: our obs IS an
information set encoding, and "the model can't distinguish these states" is a statement about
which facts the obs carries — the coverage audits are information-set audits.

**The public belief state (PBS)** — the DeepStack/ReBeL move. Re-root the game on what is COMMON
knowledge: the distribution over private states given the public history. A policy over PBSs can
be solved with perfect-information tools because the uncertainty has been made part of the state.
**Our belief stack is an empirical PBS approximation**: the species posterior over hidden slots,
the move/spread/item beliefs, all computed from public history, all fed to the physics. What we
do NOT do is solve on it — we train a policy that reads it. The gap between those two is exactly
what ReBeL closes with search, and what our no-search constraint deliberately leaves on the
table for the training loop to approximate.

**α as a trained fixed point — and why it must not read our logits.** The opponent's action
distribution depends on their read of OUR policy, which depends on our read of theirs —
an infinite regress (level-k reasoning). The resolution on record (owner reconciliation,
2026-08-11): α is a function of the BOARD, and since our policy is also a function of the board,
self-play training drives (π, α) toward a mutually-consistent pair — **the fixed point is found
by training, never computed in the forward pass**. Reading our own logits into α would create a
forward-pass cycle (level-3 reasoning at the cost of well-definedness), which is why it is
structurally refused. This is equilibrium-finding by gradient flow instead of by solver.

**Determinization vs expectation.** Two ways to act under a belief: SAMPLE a concrete world and
plan in it (PIMC — the search-time approach, and the team-completion model's someday-job), or
MARGINALIZE — act on expectations under the posterior (our entire forward pass; see
[`marginalization_and_uncertainty.md`](marginalization_and_uncertainty.md) for when each is
exact). The known costs of our choice, measured: per-slot independent marginals cannot express
the species clause (the 14.2% duplicate-top-1 display finding), and expectation over multimodal
successors can name a state that never occurs (the outcome-latent design's §6 honesty item).

**Exploitability vs exploitation — the two corners of correct play.** A Nash strategy is SAFE
(bounded loss vs anything) but leaves value against weak opponents on the table; a best response
MAXIMALLY exploits one opponent and may be wide open to others. Our system deliberately occupies
both corners at different layers: the GENERALIST trains toward the safe center (self-play against
a pool ≈ an approximate equilibrium of the meta-game), while the EXPLOITERS are pure best
responses to frozen opponents — maximum exploitation, zero robustness, safe only because they are
distilled back rather than deployed. Naming the corners keeps the roles from blurring: an
exploiter's win rate is not a strength claim, it is an exploitability measurement of its target.

**The honest gap: we never reason about what our OWN actions reveal.** The belief stack models
their hidden state; nothing models their belief about ours, so information-hiding play (holding a
reveal, mixing to stay unreadable) is unrepresentable and unrewarded except as an emergent
accident of self-play mixing. That is a known scope cut at one ply, not an oversight — recorded
here so it reads as a boundary, not a bug.

## 3. What this buys operationally

- Belief-quality metrics (species acc, α/β info gain) are measurements of a PBS approximation —
  their ceiling arguments (β capped by belief coverage, mask rates as the belief's failure) are
  the standard PBS decomposition, not ad-hoc bookkeeping.
- Any future search layer must search over BELIEFS, not sampled worlds alone (or must sample
  many) — the CRN-anchored beam already respects this by branching from the recorded belief
  state.
- "Is α correct?" has a precise meaning: calibrated against the CURRENT opponent population's
  play at this fixed point — which is why `_pool`-stratified α metrics are the real ones and the
  bot-stratified ones are decision-tree modeling.

**The question you can answer after this note:** *why can't we just train α to predict the
theoretically-optimal opponent?* — because the optimal opponent is an equilibrium object defined
jointly with our own policy; there is no fixed label. The only labels that exist are what THIS
population actually did, and the fixed point improves as the population does.
