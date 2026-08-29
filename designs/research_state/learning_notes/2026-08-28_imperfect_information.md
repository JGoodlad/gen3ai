# Learning note — imperfect information: the theory under the beliefs, the search arms, and the ladder (2026-08-28)

Owner-requested deep note (side-project). Sequel to the Nash/PSRO note — expands its "honest
caveat" into the full picture.

## 1. Hidden info changes the OBJECT, not just the difficulty
Perfect info: every state has a value; backward induction works. Hidden info: you act on
INFORMATION SETS (all states consistent with observations); an info set's value depends on the
distribution over its states, which depends on the OPPONENT'S strategy → global circularity, no
local solving. Consequence: RANDOMIZATION is part of optimal play — bluffing is the
mathematically correct frequency of lying, not psychology. Policy entropy at a double-switch
decision can be CORRECT, not indecisive.

## 2. Determinization's two named diseases
PIMC (sample a world, solve perfect-info, average): **strategy fusion** — the per-world solver
acts differently in states you cannot distinguish, booking profits from knowledge you don't
have; **non-locality** — a node's value depends on what the opponent infers from your reaching
it, i.e. on distant parts of the tree; independent subgame solving is unsound. **Our
search_dividend arms are this literature's design**: honest = belief-determinized (fusion-
exposed), oracle = true hidden state (cheating upper bound); oracle−honest = the price of
information. Empirical rhyme: the three-axis variance split (OPP 36–60% >> DICE 14–27% ⇒
"marginalize the opponent, never the dice").

## 3. CFR — what actually solves these games
Counterfactual Regret Minimization (Zinkevich 2007): per-info-set regret matching in self-play;
the TIME-AVERAGED strategy converges to Nash (the current one cycles — averaging tames the
churn). "Counterfactual" = values weighted by the opponent's reach probability. Libratus/
Pluribus = CFR + abstraction + subgame refinement. We don't use it because Pokémon's info sets
resist bucketing at scale; our route is IMPLICIT mixing via PPO on rich obs — no guarantee, no
way to read off whether mixing frequencies are right (which is why exploiters keep finding
+12pp holes).

## 4. ReBeL — and how close our architecture already is
Make the belief state part of the state (public belief state) ⇒ the game becomes perfect-info
over beliefs; search works; fusion impossible by construction. Price: value functions over
beliefs + strategy-dependent belief transitions. **We are halfway there empirically: the belief
heads' posteriors feed the trunk and critic, so V is already ≈ a function of the belief
state.** Missing: search over belief TRANSITIONS — the upgrade path if the search dividend ever
becomes priority (not better determinization).

## 5. Safe exploitation — the ladder's theory
Nash leaves profit vs weak opponents; max-exploitation is maximally exploitable. **Restricted
Nash Response** (Johanson 2007): sell ε of your own exploitability for exploitation of a model;
practical form = play best-response-to-model with prob p, equilibrium with 1−p; the (p, ε)
curve is computable. Our exploiters = p=1 vs a frozen model (by design); the generalist aspires
to p=0; a ladder agent should live at a CHOSEN point — the α/β intent heads are the opponent
model, and humans at 1400–1600 adapt slowly within a game, so some exploitation is nearly free
and "how much" is a measurable curve.

## 6. Honesty
We lack explicit info sets, mixing-convergence guarantees, and belief-space search. Measurable
but unmeasured: whether our mixing frequencies at canonical mind-game decisions approximate
equilibrium frequencies (α/β + traces could estimate; "bluffs at roughly the right rate" would
be a novel reading). Caution from the poker lineage: naive subgame re-solving reintroduces
non-locality — Libratus's "safe subgame solving" exists because the obvious version is wrong;
required reading before any mid-game search on a ladder agent.

Through-line: the opponent-axis variance dominance, belief-fed critic, three search arms,
exploiter locality, entropy at prediction points — all built without this theory, all landing
on shapes it names.
