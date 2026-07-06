# Learning Exercises & Reading Plan

A guided, LLM-assisted self-study plan for the imperfect-information / value-function / metric-learning
cluster relevant to gen3ai. Sibling to `rl_concepts_studyguide.md` (glossary) and
`pbs_value_functions_and_search.md` (theory narrative).

**Method (how to use this):**
- **Implement to understand.** For each exercise, code the *smallest* version (a toy game or a tiny head) and
  watch it work. A 100-line toy that converges teaches more than re-reading the paper.
- **LLM-assisted, paper-verified.** Use an LLM to unpack notation, answer "why does this step hold," and work
  small examples — but **cross-check the paper's pseudocode for the exact update equations** (LLMs hallucinate
  equations). Trust the paper's algorithm box over any recollection of it.
- **Ground each read.** For every deep read, write down "what would I change in my PPO/obs/trunk to add this?"
  If a paper produces no concrete code-change or toy impl, it's *idea-level* — read it lighter.

---

## Exercise 1 — Regret matching + CFR on toy games (the engine)

- **Goal:** implement regret matching on Rock-Paper-Scissors, then CFR on Kuhn poker; watch the *average*
  strategy converge to Nash while the *current* strategy oscillates.
- **Context:** CFR is the engine inside every imperfect-info search (ReBeL, DeepStack). This is the foundation.
  See `pbs_value_functions_and_search.md` §5 (CFR) and §6 (the hand-worked 2×2 node solve).
- **Read (skim):** Zinkevich et al. 2007, *Regret Minimization in Games with Incomplete Information* (CFR);
  Tammelin 2014, *Solving Large Imperfect Information Games Using CFR+*.
- **Do:** (a) regret matching on RPS → converges to uniform; (b) CFR on Kuhn poker → converges to the known
  analytic Nash (a 1-parameter family — look it up and check).
- **Done when:** your average strategy matches Kuhn's Nash; you can *show* the current-iterate oscillation vs
  average-iterate convergence.
- **Questions to answer for yourself:** Why does the *average* converge but not the current strategy? What does
  the *counterfactual reach weighting* buy you (why weight by the opponents' reach, not your own)? Why does
  zero-sum + regret-min → Nash (and only coarse-correlated-eq in general-sum)?

## Exercise 2 — R-NaD / NeuRD on a matrix game (the actionable one)

- **Goal:** reproduce, on a 2×2 zero-sum game, the sequence: naive softmax policy-gradient self-play **cycles**
  → NeuRD still cycles → **+ regularization toward a reference** converges to a regularized Nash → **+ periodic
  re-anchoring** converges to the *true* Nash.
- **Context:** R-NaD (DeepNash) is the *model-free* route to Nash — a **modification of the PPO you already
  run**, no search or belief states at inference. It targets your exact failure mode (self-play converging to
  an exploitable fixed point). This exercise is the prerequisite to deciding whether to add it.
- **Read (DEEP):** Perolat et al. 2022, *Mastering the Game of Stratego with Model-Free Multiagent RL*
  (Science) — DeepNash/R-NaD; Hennes et al. 2020, *Neural Replicator Dynamics* (NeuRD, the logit update);
  Perolat et al. 2021, *From Poincaré Recurrence to Convergence in Imperfect Information Games* (why
  regularization tames cycling). Optionally Heinrich & Silver 2016, *NFSP* (the simpler ancestor).
- **Do:** implement (a) softmax PG self-play on RPS (watch it orbit); (b) NeuRD = gradient ascent on **logits**
  by the advantage (still orbits); (c) add `−η·KL(π ‖ π_reg)` regularization → spirals into the regularized
  eq; (d) periodically set `π_reg ← π` → converges to the true Nash. Plot the exploitability over time.
- **Done when:** you have the cycling-vs-converging plot and can state what each of {NeuRD, regularization,
  re-anchor} contributes independently.
- **Questions:** Why does softmax PG *cycle* in zero-sum (and why does the softmax Jacobian scaling matter)?
  Why does regularization give a *unique* equilibrium with *last-iterate* convergence? Why does re-anchoring
  reach the *true* Nash (the self-consistent fixed point where `π_reg = π`)?
- **Code-change output:** "To add R-NaD-style regularization to my PPO: KL-to-a-frozen-self penalty + a
  schedule that advances the anchor. My snapshot pool is already ~the reference-policy machinery — what's the
  minimal diff?"

## Exercise 3 — Basic search (node solve → PIMC)

- **Goal:** at one decision, build the K×M payoff matrix by 1-ply lookahead, solve it for a mixed Nash (Ex. 1's
  regret matching), and sample. Then extend to **PIMC**: sample opponent determinizations, solve each
  perfect-info, average.
- **Context:** the "game-theory system" from the design discussions. Your prober's `lookahead` already reads
  successor values — this turns it into a *searched* decision. See §6 (node solve), §7 (why a fixed leaf value
  is unsound — the gadget), §9 (rollout vs bootstrap) of the theory doc.
- **Read:** Silver et al. 2018, *AlphaZero* (MCTS/PUCT context); the Wang thesis (Pokémon MCTS + PIMC precedent,
  skim); optionally Moravčík et al. 2017 *DeepStack* + Brown et al. 2020 *ReBeL* for the PBS + value-net + CFR
  loop **at idea level** (skim proofs).
- **Do:** (a) matrix-game node solve on the toy Pokémon turn (reuse the §6 2×2) — reproduce its analytic Nash;
  (b) PIMC: sample K opponent teams from a belief, solve each as perfect-info, average → a mixed strategy.
- **Done when:** the node solve matches the analytic Nash; PIMC yields a belief-averaged mixed strategy.
- **Questions:** Why is a single scalar leaf value *exploitable* (what does the gadget fix)? Why does PIMC
  commit *strategy fusion*? Why is a rollout "policy-bound not sim-bound," and when to bootstrap vs roll out?
- **Code-change output:** "how would I wire a 1-ply matrix-game solve into the prober as inference-time search,
  using the win-prob head as the payoff?"

## Exercise 4 — A basic aux-loss head on a shared trunk

- **Goal:** add a supervised aux head reading the shared trunk, trained on a label; verify it (a) learns,
  (b) enriches the shared representation, (c) is leak-safe and OFF-byte-identical.
- **Context:** the pattern behind your belief / win-prob / value-dist heads **and** the public-info value head
  (`design_public_info_value.md`) — the next real build. Understand this and the public-value integration is
  mechanical.
- **Read:** your own `belief_head` / `win_prob_head` code (the in-repo pattern is the best reference);
  Jaderberg et al. 2016, *UNREAL* (auxiliary tasks in RL); the "interpretability by construction" section of
  the concept glossary; the soundness rule in `pbs_value_functions_and_search.md` §9.
- **Do:** add a toy aux head predicting a *known* board feature (e.g., turn count, or spikes-count) on the
  trunk. Verify: it learns; coef-0 is byte-identical; the aux gradient flows into the trunk (or is detached per
  `--belief-grad-mode`); the RL value head still learns `V^π`.
- **Done when:** the aux head predicts its label, the trunk features shift, and the advantage stays on-policy.
- **Questions:** Why does an aux target on a *shared* trunk enrich the representation the RL critic reads? Why
  keep the RL value head on `V^π` and **never** wire `V^human`/search-value into GAE (the bias)? Shaping vs
  detached gradient — when each?
- **Code-change output:** "the public-info value head is exactly this with a human-replay outcome label — the
  minimal wiring diff, plus the defensive-AUC transfer test."

## Exercise 5 — The "nearest item in a latent" loss (metric / contrastive learning)

- **Goal:** implement the losses that pull a *predicted* latent toward the *correct/nearest* item in an
  embedding space, and understand the collapse problem: cosine-to-target (SimSiam-style) vs triplet vs InfoNCE.
- **Context:** your **latent belief head already does this** (predict a slot's refined token → cosine toward the
  stop-grad `pokemon_encoder` role-token of the *true* hidden mon, VICReg floor to avoid collapse). This
  exercise makes that machinery legible and generalizes it to the future **team/failure latent + KNN retrieval
  curriculum** (`design_curriculum_uncertainty.md`).
- **Read:** Schroff et al. 2015, *FaceNet* (triplet loss); van den Oord et al. 2018, *CPC* (InfoNCE); Chen et
  al. 2020, *SimCLR*; Chen & He 2021, *SimSiam*; Grill et al. 2020, *BYOL*; Bardes et al. 2021, *VICReg*
  (collapse prevention); and your own latent-belief head code.
- **Do:** on a toy embedding, implement (a) **cosine-to-target** regression (SimSiam-style: predictor +
  stop-grad target); (b) **triplet** (anchor / positive / negative, margin); (c) **InfoNCE** (contrast the
  positive against a batch of negatives). Then **reproduce latent collapse** (everything → one point) and
  **defuse it** with a VICReg variance floor / stop-grad asymmetry.
- **Done when:** you can pull a predicted vector into its target's neighborhood, and you can *cause and cure*
  collapse.
- **Questions:** Why does contrastive (InfoNCE/triplet) need *negatives* but SimSiam/BYOL doesn't (the
  stop-grad + predictor asymmetry)? What's the difference between "regress to the *exact* target" and "be
  *nearest among candidates*" (retrieval)? For a **team/failure latent + KNN**, which loss trains the embedding,
  and what's the query at retrieval time?
- **Code-change output:** "the embedding + loss for a team/failure latent so KNN retrieves the 'family of
  surprise' — how does it differ from the belief-latent cosine target?"

## Exercise 6 — Calibration vs discrimination (measurement)

- **Goal:** compute AUC, a reliability curve, and the Murphy decomposition on a predictor; internalize why
  *discrimination (AUC)*, not calibration, is the value-quality metric.
- **Context:** the star metric of the whole value-function thesis; you already have the AUC-by-style script
  (`tmp/calibration_explore.py`) and the public-value leakage guard (`tmp/pubval_experiment.py`).
- **Read:** the calibration/discrimination section of the concept glossary + §2 of the theory doc; any
  reliability-diagram / Brier-decomposition reference (Murphy 1973).
- **Do:** on the public-value predictor (or the critic traces), plot AUC, a reliability diagram, AUC by
  subgroup, and confirm the turn-1 ≈ 0.5 leakage guard.
- **Done when:** you can explain, with an example, why a *calibrated-but-non-discriminating* predictor produces
  zero advantage (`V ≈ const → A ≈ 0`), and why AUC is robust to the loss-enrichment confound.

---

## Reading triage (what depth, and why)

| Paper | Depth | Why |
|---|---|---|
| **Perolat 2022 (DeepNash/R-NaD)** + **Hennes 2020 (NeuRD)** + **Perolat 2021 (regularization)** | **DEEP / implement** | The *actionable* cluster — a modification of your RL; do Ex. 2. |
| Heinrich & Silver 2016 (NFSP) | medium | Simpler model-free-Nash ancestor; helps read R-NaD. |
| Zinkevich 2007 (CFR), Tammelin 2014 (CFR+) | idea / toy | You have the core; implement in Ex. 1, don't master the proofs unless you build CFR. |
| Moravčík 2017 (DeepStack), Brown 2020 (ReBeL) | idea / skim | The PBS + value-net + gadget concept (you have it); details only if you go *full sound PBS search* (not the workstation path). |
| Silver 2018 (AlphaZero) | idea | MCTS/PUCT context for Ex. 3. |
| Jaderberg 2016 (UNREAL) | idea | Aux-tasks-in-RL context for Ex. 4. |
| Schroff 2015, van den Oord 2018, Chen 2020, Chen & He 2021, Grill 2020, Bardes 2021 | idea / toy | Metric/contrastive losses; implement in Ex. 5. |
| Libratus, Player of Games, Student of Games, ESCHER, Deep CFR (Brown 2019), MCCFR (Lanctot 2009) | skim / skip | History / generalizations / CFR-route-only. |

**Suggested one/two-day block:** Ex. 1 (½ day) → Ex. 2 (1 day, the priority) → skim the ReBeL/DeepStack idea →
Ex. 4 + Ex. 5 as they map directly onto next builds (public-value head; team/failure latent). Ex. 3 and Ex. 6
as time allows. The payoff isn't just understanding — it de-risks the choice of *which* direction to spend
compute on (R-NaD regularization vs public-value aux vs curriculum), so pair every read with its code-change note.

---

## Copy-paste prompts to scaffold a notebook

Paste any block below into a capable coding LLM (ChatGPT with code-interpreter, Claude, Cursor, or a local
Jupyter/Colab kernel with an LLM). Each prompt is **self-contained** — it needs no gen3ai context — and asks for
a *runnable, well-commented, plot-producing* notebook so the output teaches rather than just runs. Easiest path:
**Google Colab** (zero setup, has numpy/matplotlib/torch). Do them in order; each says "continue the notebook."

### Prompt 0 — Exercise 1: Regret Matching + CFR

```
You are helping me learn game theory for RL by building a small, well-commented Jupyter notebook. Use ONLY
numpy and matplotlib. Make every cell runnable in Colab. Add short markdown explanations between cells.

Build "Exercise 1: Regret Matching & CFR":
(1) Implement REGRET MATCHING for a one-shot two-player zero-sum matrix game. Test on Rock-Paper-Scissors
    (row-player payoff: R>S, P>R, S>P; +1 win / -1 lose / 0 tie). Run self-play regret matching for both
    players for T=10000 iterations, accumulate each player's AVERAGE strategy, and PLOT: (a) the row player's
    CURRENT strategy over iterations (show it oscillating), (b) its AVERAGE strategy converging to (1/3,1/3,1/3).
    Print final averages and the exploitability (max gain from best-responding).
(2) Implement vanilla CFR for KUHN POKER (standard 3-card, 1-bet toy poker). Run T iterations, print the
    converged average strategy per infoset, and compare the bet/bluff/call frequencies to Kuhn's known analytic
    Nash (a 1-parameter family in alpha ∈ [0, 1/3]) — show they match.
Add markdown: WHY the average converges but the current strategy cycles; what the counterfactual reach
weighting does. Keep it minimal and readable, not production code.
```

### Prompt 1 — Exercise 2: R-NaD / NeuRD (the priority)

```
Continue the notebook with "Exercise 2: R-NaD / NeuRD — making self-play converge to Nash". numpy + matplotlib.
On Rock-Paper-Scissors (zero-sum), demonstrate this progression, EACH in its own cell with a plot of
exploitability (distance to Nash) vs iteration, plus a trajectory plot in strategy space:
(a) NAIVE softmax policy-gradient self-play (both players ascend the softmax policy on expected payoff) — show
    it CYCLES/orbits the Nash (exploitability does NOT go to 0).
(b) NeuRD: instead of the softmax policy gradient, ascend DIRECTLY ON THE LOGITS by the action advantage,
    i.e. logits[a] += lr * (Q[a] - expected_value). Show it still cycles in the unregularized game.
(c) Add REGULARIZATION toward a fixed reference policy pi_reg: subtract eta*(log pi[a] - log pi_reg[a]) from
    each action's advantage (KL toward reference). Show the current strategy now CONVERGES (last-iterate) to a
    regularized equilibrium biased toward pi_reg.
(d) Add the OUTER LOOP ("Nash dynamics"): every K inner steps set pi_reg <- current pi and continue. Show the
    sequence converges to the TRUE Nash (exploitability -> 0).
Add markdown: why (a) cycles; what regularization does (unique equilibrium, last-iterate convergence); why
re-anchoring reaches the true Nash (the self-consistent fixed point pi_reg == pi). Keep it minimal.
```

### Prompt 2 — Exercise 3: node solve + PIMC

```
Continue with "Exercise 3: basic search — a matrix-game node solve + PIMC". numpy + matplotlib.
(1) Solve a single simultaneous-move decision as a matrix game with regret matching (reuse Exercise 1). Test on
    this 2x2 (row payoffs): rows=[Stay,Switch], cols=[Attack,Coverage], payoff=[[0,3],[4,-1]]. Print the mixed
    Nash and game value; verify against the analytic answer (row: Stay 0.625 / Switch 0.375; col 0.5/0.5;
    value 1.5).
(2) PIMC toy: the opponent has one of 3 hidden "types" with prior [0.5,0.3,0.2], each giving a different 2x2
    payoff matrix (make some up). Implement PIMC: for each type, solve the perfect-info matrix game; average the
    resulting strategies weighted by the prior. Print the belief-averaged mixed strategy.
Add markdown: why PIMC's "strategy fusion" is NOT a sound Nash of the imperfect-info game (it assumes it will
learn the hidden type). Keep it minimal.
```

### Prompt 3 — Exercise 5: nearest-item-in-latent losses

```
Continue with "Exercise 5: nearest-item-in-a-latent losses". Use numpy, a tiny PyTorch model, matplotlib.
Toy setup: N=200 items, each with a fixed random 8-dim "true embedding". A small predictor MLP takes a NOISY
view of an item and must map it near that item's true embedding. Implement and compare, each in its own cell
with a plot of top-1 nearest-neighbor retrieval accuracy (on held-out items) over training:
(a) COSINE-TO-TARGET (SimSiam-style): loss = 1 - cosine(pred, stopgrad(true_embedding)).
(b) TRIPLET: anchor=pred, positive=true target, negative=a random other item; margin loss.
(c) InfoNCE: cross-entropy of cosine similarities of pred against a batch containing the positive + negatives.
Then DEMONSTRATE COLLAPSE: train cosine-to-target WITHOUT the stop-gradient (let the target embeddings also
train) and show all embeddings collapse to ~one point (plot per-dimension embedding std -> 0). FIX it with a
VICReg-style variance floor (hinge penalty keeping each dimension's std above a threshold).
Add markdown: why contrastive (triplet/InfoNCE) needs NEGATIVES but SimSiam/BYOL doesn't (stop-grad + predictor
asymmetry); the difference between "regress to the exact target" and "be nearest among candidates" (retrieval).
Keep it minimal.
```

### Prompts 4 & 6 (toy versions — the real ones belong in-repo)

Exercise 4 (aux head) and Exercise 6 (calibration) are best done **against your real trunk/traces in-repo**, but
here are standalone toy prompts if you want the mechanics first:

```
Aux-head toy: build a tiny 2-layer "trunk" MLP with TWO heads on the shared trunk — a main regression head and
an AUXILIARY head predicting a different (correlated) target. Train with total_loss = main + coef*aux. Show:
(a) coef=0 reproduces the main-only model bit-for-bit; (b) coef>0 shifts the SHARED trunk's features and can
improve the main head; (c) a "detached" variant where the aux gradient does NOT flow into the trunk. Plot main
test error vs coef. Add markdown on how a shared-trunk aux target enriches the representation the main head reads.
```
```
Calibration-vs-discrimination toy: generate synthetic (score, binary-outcome) data where you control both the
RANKING quality and the probability SCALE. Compute and plot: AUC (discrimination), a reliability diagram
(calibration), and the Brier score with its Murphy decomposition (reliability - resolution + uncertainty).
Show a predictor that is PERFECTLY CALIBRATED but has AUC=0.5 (predict the base rate for everything), and one
with high AUC but poor calibration (fixable by monotonic recalibration, which leaves AUC unchanged). Add
markdown on why discrimination — not calibration — is what makes an advantage signal.
```

**Tip:** after the LLM generates each cell, *run it and check the plot matches the "Done-when" in the exercise
above.* If the numbers/plots don't match the stated analytic answer (RPS→uniform, the 2×2 Nash→0.625/0.375,
collapse→std 0), tell the LLM the expected result and have it debug — that back-and-forth is where the learning
happens.
