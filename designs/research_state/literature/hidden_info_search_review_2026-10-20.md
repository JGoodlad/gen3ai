# Search under hidden information: what the literature says works at our scale, and what it predicts for Gen 3 OU (as of 2026-10-20)

> A dated literature review, from web research on 2026-10-19/20 plus a reading of our own search ledger. It is a
> snapshot and does not get updated; a later review gets a new dated file. The always-current verdict lives in
> [`../../endstate/search_ledger.md`](../../endstate/search_ledger.md), which carries this review's summary and its
> ranked experiment list.
>
> **Evidence tags.** **VERIFIED** = read in the primary source (the paper, its arXiv / DOI text, or the code at a named
> commit), with a section, table or file pointer. **UNVERIFIED** = from a secondary source, or not found. **MY
> JUDGEMENT** = an inference of mine that no source states. Ladder numbers of bots are their authors' self-reports.
> Every claim about OUR stack points at the doc that owns the number; this file holds no measurement.
>
> The owner's priority (P0.1, 2026-10-11 / 2026-10-17): "what search needs". The scope rule: search is at most ONE ply,
> offline (`search_ledger.md` Scope).

---

## 0. Summary, in plain language

**Where we are** (`search_ledger.md` §1, §5.1; `design_world_sampler.md` §6.1). An oracle one-ply rollout search
(every hidden fact known) beats the policy by ≈ +0.07–0.08 win probability per decision. Honest search, in worlds
drawn from Smogon usage, does not beat the policy at all (IMP ≈ 0, or slightly negative along X26). Mixing the true
world into the honest worlds at weight p, the 8-rollout leaf needs p ≈ 0.2–0.3 before it pays; the network's own
belief is worth p ≈ 0.01. The network already knows the never-seen SPECIES (0.31 of its mass on the exact set) but
not the MOVES.

**What the literature says, in five points.**

1. **One-ply determinized search is the regime where the classic PIMC failures are smallest.** Strategy fusion
   needs a tree in which our later choices differ by world; a single shared root choice has none. Non-locality (the
   opponent steering play with information we lack) is the remaining error, and in its INFERENCE form it is the
   lever: the opponent's past choices are evidence about its hidden team. Card-game programs that weight sampled
   worlds by a model of the opponent's play gained large, replicated margins (Skat: Buro et al. 2009; Solinas et al.
   2019; Rebstock et al. 2019). Nothing in the literature suggests a deeper or more principled search would rescue
   a sampler that is wrong.
2. **The literature asks for a CALIBRATED belief, not for weight on the exact truth** (MY JUDGEMENT, from the
   theory and three results). PIMC averages values over the posterior; if the posterior is right, the argmax is the
   Bayes decision for the information the sampler conditions on, even when the posterior is diffuse. Every Pokémon
   search bot that beats its policy samples from a distribution that is right by construction: Wang 2024 used
   Showdown's own random-battle generator, Jaxcalibur its network's predictions over the random-battle set list, the
   VGC bots open team sheets. Ataraxos (Stratego, 2025) samples from a belief network trained on the self-play truth.
   In Skat, putting all weight on the true world did NOT beat good inference (Rebstock et al. 2019). Our dose curve
   mixes the truth into a MISCALIBRATED base (Smogon usage, against opponents drawn from our training pool and bots).
   So "p ≈ 0.2–0.3 on the truth" measures what a wrong prior costs, not what a right but diffuse one would need. One
   qualifier: a sampler that ignores evidence the policy reads (the opponent's past choices) can lose to the policy
   even when calibrated on everything else (point 1). Testable: L2 (free), L4.
3. **Our honest leaf is an argmax over about nine noisy means, and the optimizer's curse punishes that** (Smith &
   Winkler 2006). Eight binary outcomes per action tie 60 % of the time, and averaging over worlds shrinks the true
   differences between actions. Every modern sampled-world search ANCHORS to its policy: piKL (Jacob et al. 2022), the
   magnet of MMD (Sokota et al. 2023), SPARTA's deviation threshold (Lerer et al. 2020). In Ataraxos, removing the anchor
   took the search from +123 Elo over its policy to 362 Elo BELOW it. That is our symptom: honest search at or below
   the policy. Our "the prior hurts" read (N5) was made in the ORACLE world, where the leaf is right. On the fully
   honest base the anchor is unmeasured, and measuring it costs zero playouts (L1).
4. **The sound imperfect-information search methods (DeepStack, ReBeL, Student of Games) are outside our regime.**
   They carry a belief over every private state: 1,326 hands in hold'em against roughly 10²⁰ or more teams here
   (`design_value_discrimination.md` §14.2), and ReBeL and Student of Games say so in their limitation sections. The
   methods for huge information sets SAMPLE. Obscuro (fog-of-war chess) samples uniformly from an ENUMERATED consistent
   set of ≤ 10⁶ positions, which we cannot enumerate, and solves a small game in which the opponent knows each
   sampled state. Ataraxos samples from a learned belief and takes one anchored update; that is our shape, minus the
   anchor and minus a set-level learned belief. DeepNash did not search at all.
5. **Pokémon practice agrees.** Where a search bot wins, its belief is nearly free: a published set list (random
   battles), team preview (Gen 5 onward), or open team sheets (VGC). In the PokéAgent format without team preview
   (Gen 1 OU, NeurIPS 2025) the champion was a search-free RL agent, while Foul Play's MCTS won Gen 9 OU, which has team
   preview. Metamon, with no search, drew Foul Play in Gen 3 OU. Gen 3 OU (no preview, no set list) is the hardest
   case for search in the field, and our measurements match what the literature predicts there.

**What it predicts for Gen 3 OU** (MY JUDGEMENT):
- Search pays first where the belief is cheap: late in the game, when all six opponent mons are seen (our k = 6
  bucket is GATE-WEAK, `search_ledger.md` §4).
- It pays next when the acting rule is anchored to the policy, and when the sampler is calibrated to the opponent
  POPULATION (a ladder-team model) and weighted by the opponent's own choices.
- The equilibrium form (the Bayesian matrix) comes after that. Against a fixed opponent it can only lose a little; it
  pays against adaptive ones.

**The ranked experiments** (§10 gives each one's prediction, cost and decision rule). They are numbered L1–L6 so they are
not confused with the hidden-info study's red-team experiments E1–E7:

| rank | experiment | tests | cost |
|---|---|---|---|
| L1 | a policy ANCHOR in the acting rule (piKL / Gumbel scale / SPARTA threshold) on the BANKED honest rows, λ fitted on Tier 1 and read held out on X26 and the study's E2 bank | the optimizer's curse; Ataraxos's anchor | zero playouts, CPU-minutes |
| L2 | a descriptive amendment to the RUNNING stage 1, posted before its read: does an arm's IMP follow how likely it makes the truth (log-likelihood) rather than its exact-world mass? | calibration vs exactness | zero; time-critical |
| L3 | the opponent-action likelihood screen: does the truth out-rank the honest worlds under the opponent's own policy? | inference from play (Skat, GIB, Hanabi) | ≈ 1 agent-day, forward passes only |
| L4 | a calibrated-by-construction bank: opponents whose teams are DRAWN by F's own sampler, then O and F on ≈ 300 states | does a right but diffuse prior suffice? | ≈ 1–2 agent-days + 10–20 CPU-h |
| L5 | Long et al.'s three PIMC properties on our banked rows | where Gen 3 OU sits on the PIMC map | zero playouts, descriptive |
| L6 | the Bayesian one-turn matrix with a best-response referee | equilibrium vs best response | 1–2 agent-days, after L1 or a sampler pays |

---

## 1. Our regime: the numbers that place us on each method's map

Every "inside / outside the regime" verdict below is read against this table. Each number is owned by the doc named.

| quantity | value | owner |
|---|---|---|
| search shape | ONE ply: every legal action (≤ 9 in gen-3 singles) is played out in K = 8 sampled worlds, to the end (LEAF-K8) or 4 turns then the critic (TRUNC-4-K8); the opponent replies with the network's greedy action, in its own honest view of the world | `search_ledger.md` Scope, §2 W3–W4; `src/main/value_search/playout.py` |
| oracle ceiling | +0.081 / +0.071 win probability per decision over the policy's pick (LEAF-K8, 15M) | `search_ledger.md` §1, W1 |
| honest search | +0.006 [−0.004, +0.017] (Tier 1, F); along X26 25M → 225M, IMP_F −0.012 / −0.022 / −0.004 / −0.011 | §1 |
| weight the search needs on the TRUE world (dose curve, mixed into F) | p ≈ 0.2–0.35 at LEAF-K8; on the continuous 4-turn leaf the curve has no step and covers half its rise by p ≈ 0.15 | §5.1 |
| weight our best belief is worth | TODAY ≈ 0.01 at every X26 anchor | §5.1 |
| exact-world mass of every candidate sampler | ≤ 0.0025 (the never-seen SPECIES set: 0.31 under the network at 225M; every seen mon's hidden MOVES right: ≤ 0.035) | `design_world_sampler.md` §6.1 |
| where the gap lives | the opponent's TEAM: never-seen mons first (s_U 0.31, n_U 0.82), the seen mons' moves a necessary minority (n_M 0.42–0.58); items / abilities / spreads ≈ 0.003 alone | §5.2 |
| ties | the 8-world honest argmax is TIED on 60 % of states | §5.1 |
| who the states' opponents are | our training-pool teams, played by scripted bots or our own snapshots (never a ladder population) | `design_world_sampler.md` §2, §7 F3–F4 |
| cost | ≈ 20–60 CPU-s per state to build honest worlds; ≈ 2.6 CPU-s per TRUNC-4-K8 decision | §6.1, W3 |
| hidden space | ~10²¹–10³⁸ team configurations per side (an ESTIMATE) | `design_value_discrimination.md` §14.2 |

---

## 2. Determinization (perfect-information Monte Carlo, PIMC) and its failure modes

**The method.** Sample worlds consistent with what we have seen, solve or play out each as a perfect-information
game, and choose the action whose average value is best. GIB did this in bridge with "a set D of deals consistent with
both the bidding and play of the deal thus far", usually 50 deals (Ginsberg 2001, JAIR 14:303–358, DOI
[10.1613/jair.820](https://doi.org/10.1613/jair.820), arXiv [1106.0669](https://arxiv.org/abs/1106.0669); VERIFIED,
Alg. 3.0.1 and §3). GIB constrained its deals with its own bidding module and weighted them by a Bayes-rule estimate
of the opponents' card play (VERIFIED, §3). So inference from the opponent's actions was part of PIMC from the start.

**The two named failures** (Frank & Basin 1998, Artificial Intelligence 100(1–2):87–123, DOI
[10.1016/S0004-3702(97)00082-9](https://doi.org/10.1016/S0004-3702(97)00082-9); primary text NOT read, definitions
VERIFIED as quoted by Long et al. 2010 and Cowling et al. 2012):
- **Strategy fusion:** PIMC "(incorrectly) believes it can use a different strategy in each world, whereas in reality
  there are situations (information sets) which consist of multiple perfect information scenarios" (Long et al.).
  It causes an error only when some moves have anti-correlated values in one subtree and another move is reliably
  better (Long et al., VERIFIED).
- **Non-locality:** "a node's value may depend on other regions of the game tree not contained within its subtree,
  primarily due to the opponent's ability to direct the play towards regions of the tree that he knows … are
  favorable for him, using private information that he possesses but we do not" (Long et al., VERIFIED).
- Both errors persist "irrespective of the number of hypothetical worlds examined" (Long et al.; Cowling et al.;
  VERIFIED).

**When PIMC works** (Long, Sturtevant, Buro & Furtak, AAAI 2010, "Understanding the success of perfect information
Monte Carlo sampling in game tree search", DOI [10.1609/aaai.v24i1.7562](https://doi.org/10.1609/aaai.v24i1.7562),
[AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/7562); VERIFIED, Methodology and Figs 3–6). Three
properties of a game's tree predict PIMC's error:
- **Leaf correlation** (the probability that sibling terminal nodes have the same payoff): PIMC "is at its worst when
  leaf node correlation is low".
- **Bias** (how much the game favours one player): a small effect.
- **Disambiguation factor** (how fast a player's information set shrinks as play proceeds): non-monotone. PIMC gains
  most over random play when hidden information is revealed quickly, and approaches optimal as it approaches 1.
- Measured on Skat and Hearts: correlation 0.8 to nearly 1.0, disambiguation ≈ 0.6, which places Skat about 0.1 below
  Nash on the synthetic map (±1 payoffs). In real Skat endgames PIMC lost ≈ 0.063 per deal to CFR on average. The paper says games between these extremes are untested, and that an opponent that models
  PIMC's predictability is not addressed.

**What it requires.** A sampler whose worlds are distributed like the real posterior (GIB, Kermit and every later
program spend most of their effort there, §3), a perfect-information solver or a value function per world, and a game
in Long et al.'s good region.

**Our regime: INSIDE, and the one-ply restriction removes most of the method's named error** (MY JUDGEMENT).
- Strategy fusion needs OUR later choices to differ by world. At one ply we choose one action for all worlds, and in
  each playout our side acts on its own simulated observation (`design_world_sampler.md` §3.6). So there is no fusion
  at the root, and none inside the playout beyond what the network's own policy does.
- Non-locality remains in two forms. (a) Value: the opponent in each world knows its own team; that is correct, since it
  does. (b) Inference: the opponent's PAST choices were made with its true team, so they are evidence about the world,
  and an unweighted sampler ignores it. That is the lever of §3.
- Long et al.'s properties for Gen 3 OU are unmeasured. Many actions in a long game lead to the same result (high
  correlation, MY JUDGEMENT), and the team is revealed gradually (our G falls with the turn, `search_ledger.md` §4), so a
  mid-range disambiguation factor is plausible. Both are measurable from banked rows (L5).
- So the classic failures do not explain our null. The sampler and the leaf's noise are the remaining suspects (§3,
  §6).

**αμ and recursive PIMC (beyond one ply, for completeness).**
- αμ (Cazenave & Ventos, arXiv [1911.07960](https://arxiv.org/abs/1911.07960), a preprint; VERIFIED) keeps one
  outcome per world in Pareto fronts and plays the same move in every world, fixing fusion. At M = 1 it IS PIMC.
  In bridge declarer play it changed the move in only 124–388 of 13,000 positions and gained 0.8–1.8 points of
  success rate, at ≈ 13× PIMC's time with its optimisations.
- Recursive Monte Carlo (IIMC; Furtak & Buro, IEEE CIG 2013, DOI
  [10.1109/CIG.2013.6633646](https://doi.org/10.1109/CIG.2013.6633646); primary text NOT read, UNVERIFIED) calls PIMC
  inside the playouts.
- Both are multi-ply and outside the owner's scope rule. Their lesson for us: fixing fusion buys little where PIMC is
  already strong.

**The cheapest test in our stack:** L5, Long et al.'s properties on our banked Phase 1 rows (zero playouts).

---

## 3. Opponent modelling as belief evidence: weighting worlds by the opponent's own choices

This is the literature's best-replicated lever for a PIMC searcher, and the one aimed at the class our sampler lacks.

| work | how worlds are weighted | what it bought | VERIFIED |
|---|---|---|---|
| **Kermit** (Buro, Long, Furtak & Sturtevant, IJCAI 2009, "Improving state evaluation, inference, and search in trick-based card games", [IJCAI page](https://www.ijcai.org/Abstract/09/236)) | P(world \| bids) from tables over high-level hand features, learned from human games; sample consistent worlds, weight, renormalise | inference for both roles 996 vs 779 points per 36-game series without inference (std ≈ 50); robust against a very different opponent (XSkat). The authors call a deterministic P(move \| world), as in GIB, "brittle" | Table 1 |
| **Solinas, Rebstock & Buro**, AAAI 2019, "Improving search with supervised learning in trick-based card games", arXiv [1903.09604](https://arxiv.org/abs/1903.09604) | a network predicts each card's location from the move history; p(world) ∝ Π over cards | +8 to +11 points per game over no inference, +3 to +5 over Kermit's tables; 3.1× slower per move; the gain comes from the CARD-PLAY history (the ablation without it is much weaker) | tournament tables, 2,500 matches |
| **Rebstock, Solinas, Buro & Sturtevant**, IEEE CoG 2019, "Policy based inference in trick-taking card games", arXiv [1905.10911](https://arxiv.org/abs/1905.10911) | each world weighted by the reach probability of the opponents' past actions under a learned human policy, η(s \| I) = Π π(h, a) | +2.3 / +0.6 (n.s.) / +1.6 over card-location inference, +10.4 over none (suit); ≈ 5× slower. **A "cheating" Kermit with all weight on the TRUE world did WORSE than card-location inference in two of three game types** (−3.25 suit, −8.49 grand; +9.82 null): "higher TSSR does not always mean better play" | Eq. 1, Alg. 1, tournament tables |
| **GIB** (above) | Bayes-rule card probabilities from the assumption the player would have played a card it held | — | §3 |
| **Bayes' Bluff** (Southey et al., UAI 2005, arXiv [1207.1411](https://arxiv.org/abs/1207.1411)) | posterior over the opponent's STRATEGY, by importance-weighting 1,000 prior samples | concentrates within < 20 hands in Leduc; in Texas hold'em 200 hands are not enough | results section |
| **SPARTA** (Lerer, Hu, Foerster & Brown, AAAI 2020, arXiv [1912.02318](https://arxiv.org/abs/1912.02318)) | EXACT belief by filtering every possible hand through the known blueprint policy; **one-ply** search, deviates from the blueprint only when the gain exceeds 0.05 | Hanabi 24.08 → 24.53 (SAD blueprint); a guarantee "except for an error term that decays in the number of Monte Carlo rollouts", which needs the blueprint to be exactly the other player's policy | Table 1, §3–4 |
| **Learned Belief Search** (Hu, Lerer, Brown & Foerster, arXiv [2106.09086](https://arxiv.org/abs/2106.09086); venue not confirmed) | an autoregressive belief network trained on blueprint self-play samples hands; inconsistent samples are filtered | 55–91 % of SPARTA's gain at 4.6–35.8× less compute; full-length rollouts WORSE than bootstrapped partial ones (the belief is "brittle" off its training distribution) | Table 2 |

**What it requires.** A model of how the opponent chooses, good enough that its likelihood ratios are informative;
enough choices observed; a proposal set large enough to contain good worlds (re-weighting cannot create one).

**Our regime: INSIDE, and it targets the class we lack** (MY JUDGEMENT, from the table and our §1).
- An unused move leaves no protocol line, but the choices the opponent made instead of it do. A Skarmory that has not
  used Spikes in five turns of free switches is evidence. That is exactly Rebstock's η.
- Our network can act from the opponent's seat, so the cheapest model exists. On our states from games against our
  own snapshots it is nearly the true opponent (optimistic); against scripted bots it is mis-specified; on the ladder
  only a human model (a Metamon-style behaviour clone) is right (`design_world_sampler.md` §4 (f)).
- Rebstock's cheating-Kermit result says putting all weight on the truth was not the same as playing better. It is
  one game and three contracts. Our oracle world does beat the policy (+0.08), because our referee judges in the true
  world with the same greedy continuation. So it is a caution about the TARGET (calibrated weights, not maximal
  weight on the truth), not a contradiction of our oracle reads.

**The cheapest test:** L3, the likelihood screen (§10). Does the TRUE world out-rank the honest worlds under the
opponent's own policy, and by how much does resampling a larger proposal set by that weight raise the exact-world
mass?

---

## 4. Information-set MCTS

**The method** (Cowling, Powley & Whitehouse 2012, "Information set Monte Carlo tree search", IEEE TCIAIG
4(2):120–143, DOI [10.1109/TCIAIG.2012.2200894](https://doi.org/10.1109/TCIAIG.2012.2200894),
[open version](https://eprints.whiterose.ac.uk/75048/); VERIFIED). One tree over information sets instead of one per
world; a new determinization per iteration restricts which branches are legal. "The sharing of information between
determinizations also addresses the problem of strategy fusion." Non-locality is explicitly NOT addressed: it
"requires techniques such as opponent modeling and calculation of belief distributions".

**What it found** (VERIFIED, 10,000 iterations ≈ 1 s per decision):
- Lord of the Rings: The Confrontation: ISMCTS beats determinized UCT, attributed to the whole budget going into one
  deeper tree, not to fusion ("strategy fusion is not a major factor in this game").
- Phantom (4,4,4): only the multiple-observer variant beats determinization, and with more than 1.5 s per move it falls
  behind, because ISMCTS assumes the opponent sees our hidden information (pessimistic).
- Dou Di Zhu: on a par (42.3 % vs 43.6 %; a cheating player 56.5 %). ISMCTS wins where cheating helps and loses where
  it does not, because the tree's branching is ≈ 10× larger.
- Its authors' summary: where information sets have many legal moves "and the effect of strategy fusion is not so
  clear, ISMCTS offers no immediate benefit".
- The determinizations-vs-iterations trade-off barely matters in Dou Di Zhu (Powley, Whitehouse & Cowling, AISB 2011, as
  cited by Cowling 2012; VERIFIED there). The often-cited IEEE CIG 2011 Dou Di Zhu paper (DOI
  [10.1109/CIG.2011.6031993](https://doi.org/10.1109/CIG.2011.6031993)) was NOT read (UNVERIFIED).

**What it requires.** A multi-ply tree, a cheap simulator, and a sampler (it re-determinizes every iteration, so it
needs the same sampler PIMC does).

**Our regime: OUTSIDE the scope rule, and equivalent to what we do at one ply** (MY JUDGEMENT).
- At one ply, the information-set tree has one node (our root), so ISMCTS and our search coincide
  (`design_world_sampler.md` §3.6).
- Wang 2024's MCTS sampled "one possibility for all unknown opponent information at the start of each MCTS trajectory"
  into ONE shared tree (thesis §3.2.2; VERIFIED in `designs/references/wang2024_pokemon_rl.pdf`). That is the
  single-observer ISMCTS shape, with a calibrated sampler (§7).
- Its lesson for us is the same as PIMC's: it does not fix a wrong sampler, and it fixes non-locality not at all.

**The cheapest test:** none at one ply (the methods coincide). Re-open only if the scope rule is lifted.

---

## 5. Public-belief-state methods: DeepStack, Libratus, depth-limited solving, Pluribus, ReBeL, Student of Games

| method | mechanism | requires | compute (VERIFIED unless marked) | its own scaling statement |
|---|---|---|---|---|
| **DeepStack** (Moravčík et al., Science 356:508–513, 2017, DOI [10.1126/science.aam6960](https://doi.org/10.1126/science.aam6960), arXiv [1701.01724](https://arxiv.org/abs/1701.01724)) | continual re-solving with a counterfactual value net whose input is each player's RANGE (1,326 hands, bucketed to 1,000) | a range over every private state, for both players | value-net data: > 175 core-years (turn) + ≈ ½ GPU-year (flop); < 5 s per decision on one GTX 1080 | indirect: the input IS the range |
| **Libratus / safe subgame solving** (Brown & Sandholm, NIPS 2017, arXiv [1705.02955](https://arxiv.org/abs/1705.02955); Science 2018, DOI [10.1126/science.aao1733](https://doi.org/10.1126/science.aao1733), primary text NOT read) | re-solve a subgame with a guarantee that exploitability is "no higher than the blueprint's" | a blueprint equilibrium of an abstraction; the opponent's range | 100 CPUs in the 2017 match (Pluribus paper; a press release's 1,400 cores conflicts, UNVERIFIED) | — |
| **Depth-limited solving** (Brown, Sandholm & Amos, NeurIPS 2018, arXiv [1805.08195](https://arxiv.org/abs/1805.08195)) | at the depth limit the opponent CHOOSES among k = 4 continuation strategies (blueprint, biased to fold / call / raise), which "forces an agent to be robust" | an approximate-equilibrium blueprint | < 1,000 core-hours and 16 GB (vs ≈ 2M core-hours for Baby Tartanian8); beat it +6 ± 5 mbb/g, where naive depth-limited search lost −10 ± 8 | — |
| **Pluribus** (Brown & Sandholm, Science 365:885–890, 2019, DOI [10.1126/science.aay2400](https://doi.org/10.1126/science.aay2400)) | blueprint + depth-limited search with k = 4 continuation strategies | as above | blueprint 12,400 core-hours (≈ $144); play on two CPUs, ≈ 20 s per hand | — |
| **ReBeL** (Brown, Bakhtin, Lerer & Gong, NeurIPS 2020, arXiv [2007.13544](https://arxiv.org/abs/2007.13544)) | public belief states (a common-knowledge distribution over states) + depth-limited CFR + a PBS value net; provably converges in two-player zero-sum | the PBS | data generation on 90 DGX-1 (8 × V100 each) | §9: the input "grows linearly with the number of infostates in a public state. This is intractable in games such as Recon Chess that have strategic depth but very little common knowledge" |
| **Student of Games** (Schmid et al., Science Advances 9, 2023, DOI [10.1126/sciadv.adg3256](https://doi.org/10.1126/sciadv.adg3256), arXiv [2112.03178](https://arxiv.org/abs/2112.03178)) | growing-tree CFR + a counterfactual value-and-policy net over PBSs | enumeration of the information states per public state | "a similar amount of TPU resources" to AlphaZero's 3,500 actors | "requires enumerating the information states per public state, which can be prohibitively expensive in some games; this might be approximated by a generative model that samples world states" |

Student of Games beat a determinization MCTS baseline (PimBot) in Scotland Yard, whose hidden position space has 199
nodes (VERIFIED, supplement), and says determinization methods "are not guaranteed to converge to an optimal strategy".

**Our regime: OUTSIDE, by many orders of magnitude.** Each needs a belief over every private state of a public
state: 1,326 hands in hold'em against roughly 10²⁰ or more teams here. ReBeL and Student of Games say so in their own
limitation sections. Student of Games names our route as the way around it: a generative model that SAMPLES world
states. A "factored ReBeL" over per-mon marginals is open research (`design_value_discrimination.md` §14.4).

**What transfers at one ply** (MY JUDGEMENT):
- **The multi-valued leaf** (depth-limited solving, Pluribus). Our leaf assumes the opponent plays the network's greedy
  continuation. Letting the opponent choose among a few continuations (greedy, a sampled one, an older snapshot) at
  the 4-turn leaf would make a pick robust to the continuation model. It matters against adaptive opponents, the same
  place the matrix matters (§7, L6). It needs new playouts; ranked last.
- **The safety notion.** Safe subgame solving is relative to a blueprint. Our analogue is "never worse than the
  policy", which is what a KL anchor to the policy buys (§6, L1).

---

## 6. Methods for huge information sets: Obscuro, update-equivalence search, Ataraxos (Stratego), DeepNash

These are the methods built for games where the public belief state cannot be held. They are the closest
precedents for us.

**Knowledge-limited subgame solving and Obscuro** (Zhang & Sandholm, "Subgame solving without common knowledge",
NeurIPS 2021, arXiv [2106.06068](https://arxiv.org/abs/2106.06068); Zhang & Sandholm, "General search techniques
without common knowledge for imperfect-information games, and application to superhuman Fog of War chess", arXiv
[2506.01242](https://arxiv.org/abs/2506.01242), ICLR 2026 per the proceedings listing, venue page not opened; VERIFIED
from the arXiv text, §3, §5, App. C–E):
- The program keeps the FULL set P of positions consistent with its observations ("usually |P| ≤ 10⁶") and samples a
  subset I ⊆ P "at random", "no more than a few hundred positions" (MinInfosetSize = 256), UNIFORMLY. Weighting
  the sample "using a model of opponent behavior" is named as future work.
- At newly sampled states "the opponent is assumed to have perfect information". The subgame keeps states reachable
  within second-order knowledge and is solved with predictive CFR+ in a growing tree; leaves are Stockfish 14's
  evaluation; the final move is purified to the top ≤ 3 actions. It is "not always game-theoretically sound in theory
  … but … often sound in practice".
- Compute: a 6-core desktop for the human matches; average tree 1.07M nodes, depth 10.7 plies.
- Results: 97 % against humans rated 1450–2006 (100 games); 80 % against the top-rated human (20 games).
- Its own limit (§5): "Obscuro stores at all times the entire set of possible states in memory … one can imagine even
  more complex games on which they will not work directly."

**Update-equivalence search** (Sokota, Farina, Wu, Hu, Wang, Kolter & Brown, ICLR 2024, "The update equivalence
framework for decision-time planning", arXiv [2304.13138](https://arxiv.org/abs/2304.13138); VERIFIED): decision-time
search as one damped self-play update from sampled histories. It "need not rely on public information", which
"facilitates scalability to games with large amounts of non-public information". In Hanabi it used 10,000 samples from a
learned belief model (≈ 2 s per decision) and matched multi-agent SPARTA at two orders of magnitude less compute; its
adversarial form (MMDS) used 10–100 particles in small games. No result shows a wrong or very diffuse belief is safe.

**Ataraxos, superhuman Stratego with test-time search** (Sokota, Vinitsky, Hu, Fan, Kolter & Farina, arXiv
[2511.07312](https://arxiv.org/abs/2511.07312), v1 2025-11 "Superhuman AI for Stratego using self-play reinforcement
learning and test-time search", v2 2026-10 "Scalable decision making for games of imperfect information"; a Nature DOI
[10.1038/s41586-026-11036-y](https://doi.org/10.1038/s41586-026-11036-y) matched BY TITLE only, UNVERIFIED; VERIFIED from
the arXiv text, Methods, S3.7, Extended Data Table 1):
- **The search is our shape.** A learned BELIEF network (a transformer trained on every position of self-play, i.e.
  on the truth) samples the opponent's piece identities; ≈ 1,000 rollouts of 40 plies spread over (legal move × sampled
  configuration); the averaged values feed ONE magnetic-mirror-descent step, π = argmax ⟨q̂, π⟩ − α·KL(π, ρ) −
  β·KL(π, π_θ), and the move is sampled from it.
- **Its ablations are the two numbers that matter most to us:**

  | variant | Elo | per move (H100) |
  |---|---|---|
  | no search (the move network) | 2095 | ≈ 0.004 s |
  | the evaluation setting (learned belief, 40 ply, 1,000 rollouts) | 2218 | ≈ 1.26 s |
  | 200 rollouts | 2189 | ≈ 0.50 s |
  | UNIFORM belief instead of the learned one | 2148 | ≈ 1.07 s |
  | **no reverse-KL to the move network** | **1733** | — |

  Without the KL anchor, "the search overfits to idiosyncrasies of the move network", and it plays 362 Elo WORSE
  than not searching at all. A sharp learned belief is worth ≈ 70 Elo over a uniform one.
- Cost: training 16 H100s for a week + 4 H100s for 4 days for the belief network, "less than $8,000". Results: 15–1–4
  against a top human.
- Its own limit: the improvement "is ultimately bounded because it is mimicking a single update step".

**DeepNash** (Perolat et al., Science 378:990–996, 2022, DOI
[10.1126/science.add4679](https://doi.org/10.1126/science.add4679), arXiv [2206.15378](https://arxiv.org/abs/2206.15378);
VERIFIED): model-free R-NaD and NO search, because "Stratego computationally challenges all existing search
techniques as the search space becomes intractable". 84 % of 50 ranked Gravon games, 3rd on the 2022 ranking, on 768 +
256 TPU nodes. The Ataraxos paper notes the Gravon opponents "were far from the level of top humans".

**What they require.**
- Obscuro: an ENUMERABLE consistent set (≤ 10⁶). Our set is not enumerable: OUTSIDE.
- Update-equivalence / Ataraxos: a belief sampler TRAINED on the self-play truth (calibrated for the self-play
  population by construction) and a KL anchor to the policy. Compute: ours is far smaller, but so is our search.
  INSIDE in shape.
- DeepNash: no search; our knock-in forks found the same thing from the other side, that information is not a policy
  lever at our maturity (`search_ledger.md` §1).

**Our regime** (MY JUDGEMENT). Ataraxos is the in-regime template, and we have built half of it:
- the search shape (sampled worlds × every legal move, value-net leaves) is ours;
- the belief: ours is a hand-built Smogon sampler, not a network trained on the truth. Our network's X5 species head
  IS trained on the truth and is sharp on our pool (0.31 exact-set mass), but its MOVE belief is not, and no network
  sampler of whole sets exists (`design_world_sampler.md` §4 (d));
- **the anchor: we do not have it.** Our acting rule is a plain argmax over the leaf's means (`search_ledger.md` W4). In
  the ORACLE world that is right (the leaf is right; the Gumbel prior at the `mctx` defaults hurt, N5). In honest worlds
  Ataraxos's no-KL row is exactly our symptom: search that plays below its own policy.

**The cheapest tests:** L1 (the anchor, zero playouts) and stage 1's NS arm (a network-species sampler, already
running).

---

## 7. Simultaneous moves

- Regret-matching or Exp3 selection makes MCTS converge to an approximate Nash equilibrium in perfect-information
  simultaneous-move games (Lisý, Kovařík, Lanctot & Bošanský, NIPS 2013, "Convergence of Monte Carlo tree search in
  simultaneous move games", arXiv [1310.8613](https://arxiv.org/abs/1310.8613); VERIFIED). The negative result for
  UCB / decoupled UCT ("does not converge to Nash equilibrium, even in a simple one-stage simultaneous move game") is
  Shafiei et al.'s, as cited there (NOT read; UNVERIFIED in its primary).
- Magnetic mirror descent (Sokota et al., ICLR 2023, arXiv [2206.05825](https://arxiv.org/abs/2206.05825); VERIFIED,
  abstract) converges linearly to a quantal-response equilibrium and works as deep-RL self-play.
- piKL (Jacob et al., "Modeling strong and human-like gameplay with KL-regularized search", arXiv
  [2112.07544](https://arxiv.org/abs/2112.07544), ICML 2022 per the project's earlier pass; VERIFIED, objective and
  Theorem 1) maximises Q·π − λ·KL(π ‖ τ) toward an anchor τ, with exploitability bounded by λβ. The paper does not say
  "noisy values"; Ataraxos's ablation is the empirical case for the anchor (§6).

**Our regime.** Our one-ply Bayesian matrix (`design_world_sampler.md` §5: one mixed strategy for us, a best reply per
sampled world, a small LP) is the correct one-turn form for hidden opponent types. The literature agrees with the
doc's own reading: against a FIXED opponent a best response is already optimal, so the matrix pays only against
adaptive opponents, and "a matrix over worlds worth 1 % of the truth solves the wrong game". Foul Play's decoupled UCB
has no convergence guarantee in simultaneous moves.

**The cheapest test:** L6, after a sampler pays.

---

## 8. Pokémon-specific work

| system | format (hidden information) | how it gets the opponent's hidden information | search | what search adds | source |
|---|---|---|---|---|---|
| **Foul Play** (pmariglia; current code read at [`8f541f80`](https://github.com/pmariglia/foul-play/tree/8f541f80f91f960603673796559f796b10ccf0fb), 2026-10-03, and poke-engine [`f7b147f1`](https://github.com/pmariglia/poke-engine/tree/f7b147f13637ceecc1b0d1e9172ec232af4d45d0)) | all singles gens | **sets:** a team dataset (`data.foulplay.cc`) set consistent with what was seen, 75 % of the time when one exists, else Smogon set stats with moves drawn by usage (`fp/search/standard_battles.py` `_sample_pokemon`); **never-seen species:** the top 50 by mean teammate co-occurrence with the seen species (`predict_team_likelihood`); **constraints from mechanics:** speed ranges, Choice Scarf, reverse damage checks, Hidden Power type (`fp/battle/inference.py`). Worlds are EQUAL weight (1 / n); no use of the opponent's choices | per world, poke-engine MCTS with decoupled UCB1 per side, chance nodes by instruction probability, leaf `sigmoid(eval − root_eval)` (`src/mcts.rs` `selection`, `rollout`, `ucb1`); root visit shares summed across worlds, then sampled among moves ≥ 75 % of the best (`fp/search/main.py`). Defaults: 1 world × 100 ms (`--search-parallelism 1`, `--search-time-ms 100`), doubled early | not reported. Won the PokéAgent Gen 9 OU track 50–14 in the final (VERIFIED, arXiv [2603.15563](https://arxiv.org/abs/2603.15563)). PRE-ERA here: our 75M policy scored 0.388 against it (`search_ledger.md` W10). MY JUDGEMENT: in each world poke-engine holds both teams, so Foul Play's simulated opponent sees OUR hidden information | code (VERIFIED) |
| **Wang 2024** (MIT MEng thesis, [DSpace 1721.1/153888](https://dspace.mit.edu/handle/1721.1/153888)) | Gen 4 random battles | Showdown's OWN team generator, with rejection against known traits (10 attempts, then forced): the exact prior of the format | MCTS, one tree, one world sampled per trajectory; 1,000–2,000 rollouts per 10 s; the opponent modelled by the network | MCTS+NN beat NN alone 80.9 %, which the author calls "inflated" ("the MCTS always knows exactly what the NN will do"); peak #8, 1693 Elo | thesis §3.2.2, §4.2 (VERIFIED, local PDF) |
| **Jaxcalibur** ([write-up](https://jaxcalibur.github.io/), Aug 2026) | Gen 9 random battles (a published set list) | "possible 'worlds' … using its network's predictions" | pUCT with regret-mixed simultaneous selection, depth 4, 20,480 rollouts over 32 worlds. The opponent's search is truncated after a "surprising" own action, "to prevent search from leaking hidden information … to the simulated opponent" | "search adds 100–150 Elo over its base policy"; peak 2557, #1 (self-reported) | VERIFIED (write-up) |
| **PokéChamp** (Karten, Nguyen & Jin, ICML 2025, PMLR 267:29205–29222, arXiv [2503.04094](https://arxiv.org/abs/2503.04094)) | Gen 9 OU, Gen 8 random battles | Showdown usage statistics for move pools, spreads and items, plus an LLM estimate of the hidden attack / defence stats from the battle history (§4.1) | depth-limited minimax; the LLM samples our actions, models the opponent's and scores leaves | 84 % against the strongest rule-based bot, 76 % against the best LLM bot; a projected 1300–1500 Elo | VERIFIED (abstract, §4.1, Table 2) |
| **Metamon** (Grigsby, Xie, Sasek, Zheng & Zhu, arXiv [2504.04395](https://arxiv.org/abs/2504.04395); RLC 2025 per the 2026-10-18 survey, UNVERIFIED here) | Gen 1–4 OU (no team preview) | to build the TRAINING data, unrevealed parts of replays are filled from "usage data and the revealed teams of similar replays"; the agent itself infers in its sequence model | **none** ("select moves without explicit search of any kind") | — ; drew "the best version" of Foul Play in Gens 3 and 4; top-300 appearances in Gen 3 OU | VERIFIED (§ on reconstruction, results) |
| **PokéAgent Challenge** (Karten, Grigsby et al., NeurIPS 2025 competition, arXiv [2603.15563](https://arxiv.org/abs/2603.15563)) | Gen 1 OU and Gen 9 OU | — | — | Gen 1 OU (no team preview, "greater effective hidden information") won by PA-Agent, a Metamon-based RL agent with NO search; Gen 9 OU (team preview) won by Foul Play's MCTS. "Specialist RL and search methods dominated LLM approaches" | VERIFIED (paper text) |
| **VGC bots** (mikumiku37, Nessie123, Wolfe_Alt6767, autumn 2026) | Gen 9 doubles, open team sheets | near-exact: team sheets and published tournament teams | one-turn simultaneous-move matrices over 16 sampled worlds, regret matching (or double oracle) | not stated | `showdown_bots_landscape_2026-10-18.md` §2 (self-reported) |
| **Sarantinos 2022** (arXiv [2212.13338](https://arxiv.org/abs/2212.13338)) | Gen 8 | — | a 2-turn payoff matrix per node, dominated strategies pruned, a best response to a regret-minimising predictor | rank 33 (self-reported) | `design_value_discrimination.md` §14.4 (VERIFIED there, 2026-10-11) |

**What the Pokémon record says** (MY JUDGEMENT).
- **Every reported search gain comes with a nearly free, CALIBRATED belief**: the format's own generator (Wang), its
  set list plus a learned predictor (Jaxcalibur, +100–150 Elo), team sheets (VGC), team preview giving the species (Foul
  Play's Gen 9 OU win).
- **Where the hidden information is greatest, search-free RL wins or ties**: Gen 1 OU's PokéAgent champion; Metamon's
  draw with Foul Play in Gen 3 OU. Our own PRE-ERA reads agree (Foul Play W10 0.388, and later 0.490, NOT DETECTED).
  One tournament, many confounders: a direction, not a measurement.
- **Foul Play is PIMC with a hand evaluator, a set dataset and mechanical inference.** It uses no inference from the
  opponent's CHOICES, equal-weight worlds and, by default, ONE world. It is strong mostly where its datasets are good.
  Our measured R-EVAL says its evaluator is not what binds us (N13).
- **Leakage**: Jaxcalibur truncates the simulated opponent's search after a surprising own action. Our playouts do not
  need this: the opponent's continuation acts on its own honest view of the world (`playout.py`: "Both sides play the
  forward's greedy action" from their own rows). Foul Play's worlds hold both full teams.

---

## 9. What the literature predicts for Gen 3 OU, method by method

| method | requires | our numbers vs its regime | prediction for Gen 3 OU (MY JUDGEMENT) | cheapest test in our stack |
|---|---|---|---|---|
| PIMC, one ply (ours) | a posterior-distributed sampler; high leaf correlation | INSIDE; no fusion at one ply; Long's properties unmeasured | pays once the sampler is right for the opponent population; earliest late in the game | L5 (properties), L4 (calibration) |
| PIMC + inference from opponent choices (Kermit, PI, GIB) | an opponent policy model; observed choices; a large proposal set | INSIDE; the model exists (our network from the opponent's seat), not wired | the largest single honest lever on the MOVES, strongest against opponents like our policy | L3 |
| acting rule with a policy anchor (piKL, MMD, Ataraxos, SPARTA's threshold) | the policy's prior; a tuned λ | INSIDE; zero cost; measured only in the oracle world | turns honest IMP from ≤ 0 to ≥ 0, and possibly clear of 0 | L1 |
| learned belief sampler (LBS, Ataraxos) | a belief network trained on the truth | PARTLY: species yes (0.31), sets no | species alone buys little (FN ≈ F); a set-level learned sampler would be needed | stage 1's NS arm (running) |
| ISMCTS, αμ, IIMC | a multi-ply tree | OUTSIDE (scope rule); identical to ours at one ply | nothing at one ply | — |
| DeepStack / ReBeL / Student of Games / Libratus | a belief over every private state | OUTSIDE by ~17 orders of magnitude | not applicable | — |
| depth-limited multi-valued leaf (Pluribus) | k continuation strategies | INSIDE in shape | robustness against adaptive opponents only | after L6 |
| Obscuro (KLUSS) | an enumerable consistent set (≤ 10⁶) | OUTSIDE (not enumerable) | — | — |
| simultaneous-move matrix (VGC bots, §5 of the sampler doc) | a sampler worth its worlds | INSIDE in shape; premature | ≥ 0 against fixed opponents, pays against adaptive ones | L6 |
| no search (DeepNash, Metamon) | — | where we are today | the baseline search must beat | — |

---

## 10. The ranked experiments

Ranked by decision value per unit cost. Each is a proposal, not registered; whoever runs one pre-registers it (root
rule 8: thresholds fixed before the data, a bound within rounding of its boundary is not clear).

**L1. A policy anchor in the acting rule, on the BANKED honest rows (zero playouts; CPU-minutes).**
- *Prediction* (optimizer's curse; piKL; Ataraxos's no-KL row): honest search's argmax over 8-world means picks noise;
  π′(a) ∝ π(a)·exp(Q(a)/λ) at a moderate λ turns IMP_F from ≤ 0 to ≥ 0, possibly clear of 0. In the ORACLE rows the
  best λ is small (consistent with N5).
- *Data*: Tier 1's F / FN / O rows (both 15M policies; per-action, per-world outcomes and the root logits are stored:
  `measurements/hidden_info_tier1_2026-10-13/t1_data.py`, `~/gen3ai_archive/hidden_info_tier1/`), the hidden-info study's E2 bank (competent opponents), and the
  maturity battery's LEAN rows at X26.
- *Rules*: fit λ (and the alternatives: Gumbel's c_scale, SPARTA's "deviate only if Q(a) − Q(top1) > δ") on Tier 1,
  read held out on the X26 LEAN rows and the study's E2 bank. PAYS iff the held-out lower bound > +0.0005 at both LEAF-K8 and
  TRUNC-4-K8.
- *Decision*: PAYS ⇒ the anchor becomes the honest search's default, and every honest "search does not beat the
  policy" read is re-stated as "the argmax does not". NULL ⇒ the honest leaf carries no decision information beyond the
  policy: the sampler binds, not the acting rule.

**L2. Stage 1's read against the calibration prediction (zero cost; TIME-CRITICAL: before stage 1 is unblinded).**
- *Prediction*: an arm's IMP follows how likely its worlds make the truth (stage 0's level-1 log-likelihood, a proper
  score), not its exact-world mass q_SM.
- *Action*: post it as a descriptive amendment to `measurements/world_sampler_stage1_2026-10-19/PREREGISTRATION.md`
  BEFORE the read; afterwards it is post hoc. Note: "only C_pool pays" does not separate exactness from calibration,
  because C_pool is both the only exact and the only CALIBRATED sampler for a pool-opponent bank.

**L3. The opponent-action likelihood screen (≈ 1 agent-day build + forward passes; zero playouts).**
- *Prediction* (Kermit, PI, GIB): weighting worlds by Π_t π_opp(observed action_t | the opponent's view in world w)
  ranks the TRUE world above the honest worlds, increasingly with the number of opponent decisions, most for MOVES.
- *Build*: the world replay emits the opponent side's rows (`design_world_sampler.md` §4 (f), §5 table); the network at
  the bank's pin scores them.
- *Read*: the truth's rank among {truth} ∪ F's worlds (an AUC; Rebstock's TSSR), by opponent stratum (bots / own
  snapshots); and the exact-world mass after resampling 8 of 64 proposals by weight.
- *Decision*: a clear AUC > 0.5 on the bot stratum ⇒ build stage 1c (worlds resampled by weight, with playouts).
  Near 0.5 ⇒ the opponent's choices carry no usable evidence at our policy's maturity; drop candidate (f).

**L4. A calibrated-by-construction bank (≈ 1–2 agent-days + ≈ 10–20 CPU-h).**
- *Question*: does honest search pay when the sampler IS the opponent population's distribution, while putting ≈ 0
  weight on the exact truth?
- *Design*: harvest states from X26 (225M) self-play in which the OPPONENT's team is DRAWN by F's own legal sampler
  (so F's prior is the true prior, as in Wang's random battles); run O and F at LEAF-K8 and TRUNC-4-K8 on ≈ 300 states
  with the Tier-1 referee.
- *Prediction* (calibration): IMP_F clearly > 0 there, though its exact-world mass stays ≈ 0. If IMP_F ≤ 0 even here,
  the bottleneck is the leaf's noise or the evidence F ignores (the opponent's choices, which the policy reads but F
  does not), not the prior. Read with L1 and L3 to separate them.
- *Hazard*: the network plays off its training distribution (random Smogon teams) on both sides; read as a test of the
  search theory, never as a strength claim.

**L5. Long et al.'s properties on our banked rows (zero playouts; descriptive).**
- Leaf correlation from Phase 1's oracle rows (how often sibling actions give the same outcome on the same dice
  seed), bias from the outcome rate, disambiguation from the reveal count per turn in the eval traces.
- *Use*: places Gen 3 OU on Long et al.'s map, and says whether a residual PIMC error is expected even with a perfect
  sampler. Descriptive only.

**L6. The Bayesian one-turn matrix with a best-response referee (1–2 agent-days; AFTER a sampler or L1 pays).**
- `design_world_sampler.md` §5 as written. *Prediction*: ≈ 0 against the greedy referee, positive on the
  best-response referee and against exploiters. A Pluribus-style multi-valued leaf is its natural extension.

**Not proposed:** ISMCTS, αμ, IIMC, ReBeL, Student of Games, DeepStack-style value nets and Obscuro. Each is either
multi-ply (outside the scope rule) or needs a belief over every private state.

---

## 11. FINDINGS (hazards, and what could not be verified)

- **F1. The dose curve's "p ≈ 0.2–0.3 on the truth" is not a literature-backed requirement** (MY JUDGEMENT). It
  mixes the truth into a miscalibrated base. The literature asks for calibrated weights, and in Skat all weight on the
  truth was not best (Rebstock et al. 2019). Quoting p\* as "what search needs" may point the sampler programme at an
  unreachable target. L4 and L2 test this.
- **F2. Our "the prior hurts" (N5) is an ORACLE-world result**, measured at the `mctx` default scale. Nothing measured
  says it holds in honest worlds; the pilot's older builder read GUMBEL∘LEAF-K8 +0.013 against ARGMAX +0.007 on end2
  honest (descriptive, `value_disc_pilot_2026-10-11/README.md`). The search module's default (argmax) is therefore
  unvalidated for honest search. L1.
- **F3. Primary texts NOT read:** Frank & Basin 1998 (paywalled; definitions taken from Long 2010 and Cowling 2012),
  Whitehouse et al. CIG 2011, Furtak & Buro 2013, the Libratus Science paper, Shafiei et al.'s UCB result. Venues NOT
  confirmed: LBS (arXiv only), αμ (preprint), Obscuro's ICLR 2026 (from a listing), Ataraxos's Nature DOI (title match
  only), Metamon's RLC 2025, piKL's ICML 2022 (from the project's earlier pass).
- **F4. Every Pokémon bot strength figure is self-reported**, except PokéChamp's and Metamon's papers and the PokéAgent
  tournament. The PokéAgent and Metamon paper texts were read through a page summariser (WebFetch), so their quotes may
  be paraphrased; the arXiv ids are the evidence. The Gen 1 / Gen 9 contrast in PokéAgent is one tournament with many confounders.
- **F5. Foul Play's opponent sees our team inside each world** (poke-engine holds both full sides). That is MY JUDGEMENT
  from the code, not stated by the author; it matters when comparing its search with ours (ours does not leak).
- **F6. A miscited trade-off:** the determinizations-vs-iterations finding for Dou Di Zhu is in Powley et al. (AISB 2011),
  not the CIG 2011 paper often cited for it.
- **F7. No paper found tests PIMC with a belief worth ~1 % of the truth.** Every method here assumes a belief that is
  right in distribution (exact enumeration, the format's generator, or a network trained on the self-play truth).
  Our regime is below every published operating point; the predictions above are extrapolations.

---

## Sources

The links in each section are the primary sources. Code read: Foul Play at `8f541f80` and poke-engine at `f7b147f1`
(shallow clones in a scratch directory, not committed). The two literature-verification passes of 2026-10-19 read the
paper texts through arXiv, AAAI, IJCAI, the White Rose repository and author-posted PDFs; DOIs were checked against
Crossref.
