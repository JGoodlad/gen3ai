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

## 4. The population game — the literature map for the flywheel era (added 2026-08-23)

*One ply of Pokémon is a matrix game (§1); a TRAINING RUN is a game too — over strategies. This
section maps the population-level literature onto our components.*

**The algorithm family tree, in ascending order of what they keep:**
- **Fictitious play** (Brown 1951): best-respond to the AVERAGE of the opponent's history. In
  two-player zero-sum the average converges to Nash — but only the average; the last iterate can
  cycle forever. Self-play against a snapshot pool IS approximate fictitious play; the pool is
  the "history average" made concrete.
- **Double oracle** (McMahan 2003): keep a restricted strategy set, solve its meta-game exactly,
  add each side's best response to the meta-Nash, repeat. Converges because each best response
  either beats the current equilibrium (progress) or certifies it (termination).
- **PSRO** (Policy-Space Response Oracles, Lanctot 2017): double oracle with RL as the
  approximate-best-response oracle and an empirical payoff matrix as the meta-game. Our
  EXPLOITERS are PSRO's oracle step verbatim.
- **AlphaStar league** (Vinyals 2019): PSRO industrialized — main agents (train vs everything),
  main exploiters (best-respond to the current main), league exploiters (best-respond to the
  whole league), PFSP opponent sampling weighted toward beatable-but-not-beaten. Our
  main-run + exploiter + fold-back recipe is closest to this, with ONE structural difference:
  we DISTILL the exploiters back into a single policy rather than keeping a population + a
  meta-mixture at play time.
- **NeuPL** (Liu 2022): the population held in ONE conditioned network rather than N frozen
  copies — the population version of our conditioning/FiLM line.

**The convergence subtleties that bite in practice:**
- **Distillation is our convergence operator, not just compression.** Fictitious play converges
  in the AVERAGE; a last-iterate self-play policy can orbit a cycle indefinitely. Folding
  exploiters + pool behavior into one network is an averaging step — the thing the theory says
  actually converges. (The retention ablation — distilled skill sticks at ~76% with teachers
  retired — is what makes this real rather than aspirational.)
- **Entropy-regularized PPO converges to a QUANTAL RESPONSE equilibrium, not Nash** (QRE:
  logit-smoothed best responses; equivalently the Nash of an entropy-bonused game). Our
  `--ent-coef` sets the smoothing temperature — the fixed point is deliberately soft. Mindful:
  the smoothing is not uniform across states; the bait saturation (0.97 confidence) shows local
  corners survive a global entropy bonus.
- **Cycles are load-bearing, not noise.** Bait propagates through self-play pools (measured);
  RPS-like dynamics mean a strategy can be "beaten" by re-discovering its counter's counter.
  The pool's retention + the anchor bots are what prevent re-cycling; deleting old snapshots is
  how populations forget and re-enter orbits.

**The geometry of real games — why exploiter COVERAGE is the right frame:**
- **Hodge/potential-harmonic decomposition** (Candogan 2011): every finite game splits into a
  TRANSITIVE (potential) component and a CYCLIC (harmonic) component. Our HodgeRank spine/width
  split is this, applied to the empirical payoff matrix.
- **Spinning tops** (Czarnecki 2020): real-world games have a long transitive axis and a cyclic
  width that is WIDEST at mid-skill, pinching toward both the floor and the Nash tip. Two
  operational corollaries: (1) expect `hodge_width` to shrink as absolute strength rises — a
  rising width instead says the population is entering the fat mid-band, not regressing; (2) at
  the tip, progress requires NEW strategic dimensions (for us: team/archetype coverage), which
  is the quality-diversity note's territory.
- **Diversity-aware PSRO** (Perez-Nieves 2021, Liu 2021): plain best responses collapse to
  similar policies; adding a behavioral-diversity term to the oracle finds the strategies the
  meta-game is missing. Our exploiter coverage board (per-archetype/team coverage) is the
  hand-rolled version — the literature's warning is that UNDIVERSE exploiters measure the same
  hole repeatedly while the meta-game's real gaps go unprobed.

**Evaluation under populations:**
- **Elo/Bradley-Terry is structurally transitive** — it projects the cyclic component to zero.
  Nash averaging (Balduzzi 2018, "Re-evaluating evaluation") reweights opponents by the
  meta-Nash so redundant weak opponents can't inflate a rating. Our mitigations: anchored BT +
  the Hodge width companion + matched-snapshot-count comparisons + the exploiter as a targeted
  exploitability probe.
- **Exploitability is the meter that survives the promotion gate.** `win_rate_vs_pool` is pinned
  ~50% by construction; the honest strength meter for a population-trained agent is "how much
  does a fixed-compute best response extract" — i.e. TRAIN AN EXPLOITER AS A MEASUREMENT,
  tracked across generations at matched exploiter compute. (Fixed compute matters: true
  exploitability needs the true best response, which no one has; the fixed-budget approximation
  is only comparable at matched budget.)

## 5. Cautions checklist as we lean into Nash approximation / exploiters / opponent ecology

1. **Name which game.** The POLICY game (fixed teams, choose actions), the TEAM meta-game
   (choose teams), and the POPULATION meta-game (choose training opponents) are three different
   games with three different equilibria. A claim like "approach Nash" must say which; our
   ladder constraint (play humans) means the TEAM meta-game's equilibrium is ultimately set by
   the human ladder distribution, not by our pool.
2. **An exploiter is an instrument until distilled** (BaitBot lesson): finding a hole ≠ teaching
   the fix; the credit conviction says the fix arrives only through the distillation target.
3. **Do not chase last-iterate Nash with on-policy RL** — smooth (QRE) fixed points + averaging
   via distillation is the convergent recipe we already run; adding explicit Nash solvers at
   play time is ruled out by the no-search constraint and unnecessary at current evidence.
4. **Exploitability curves need matched budget AND matched fork policy** (warm-fork vs scratch
   changed the measured number by ~0.2 at 10x compute).
5. **Population forgetting re-opens closed holes** — retention policy on the pool is a
   correctness parameter, not a disk-space one.
6. **Diversity of exploiters > count of exploiters** (coverage board over Q-queue length).
7. **Every population metric is ecology-relative** (the G0 sign-flip lesson): name the opponent
   population on every number; a rating, a bias, a bait rate all flip meaning across ecologies.
