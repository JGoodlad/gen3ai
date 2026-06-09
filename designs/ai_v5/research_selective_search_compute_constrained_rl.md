# Selective Search-as-Policy-Improvement into PPO for gen3ai — Technical Research Report

## 1. Executive summary & verdict

**The idea is theoretically sound, has a precise name, and is the natural next step on your own roadmap — but as literally proposed ("MCTS on the worst states, fold the result into the PPO update") it is only *partially* feasible on your stack today, and the naive version carries a real risk of making the policy *worse*, not better.** Lead with this distinction:

- **The good news.** What you described is textbook **Expert Iteration (ExIt)** / generalized policy iteration with search as the policy-improvement operator — the same operator AlphaZero distills. The "selective" twist is legitimate and has direct precedent (Targeted Search Control / Go-Exploit, Value-of-Computation metareasoning). The PPO integration seam is clean: your `InstrumentedMaskablePPO.train()` already vendors the loss sum verbatim, so an auxiliary distillation loss drops in. Your **worst-state selector is essentially free** — every signal you need (advantage magnitude, value error, TD residual) is already sitting in the rollout buffer the instant `collect_rollouts` returns. And critically, **the sim *can* fork**: Showdown's `Battle.toJSON()`/`fromJSON()`+`restart()` is production-tested every turn, so the feared "replay-from-scratch × sims × states" cost blowup does **not** apply to the recommended native-clone path.

- **The single biggest risk.** Gen-3 OU is **simultaneous-move, stochastic, and imperfect-information**. The clean "MCTS gives a provably better target" guarantee holds only for two-player *perfect-information sequential* games. Here, standard UCT is unsound, you need determinization (PIMC), and determinization **provably suffers strategy fusion and non-locality** — it manufactures an over-confident, "I know which set the opponent has" target that cannot represent the mixed strategy a 50/50 read demands. **Distilling that biased target into PPO could make your policy *more* exploitable.** This is the load-bearing hazard, and it is precisely worst where you intend to apply it: your prober selects states where the critic is most wrong, but the search bootstraps on that same wrong critic. A second, gen3ai-specific bias compounds it (§11): if the search models the opponent with your own frozen pool net — the natural and cheapest choice — the distilled target is a *best response to that one opponent*, which Wang 2024 explicitly observed "weakens the agent's performance against players who play differently from the neural network."

- **The recommendation.** Do **not** build full in-the-loop ISMCTS first. The honest cost/benefit and your own profiling (rollout/opponent-forward bound, GPU ~86% idle, CPU saturated) point to a staged plan:
  1. **First, run the cheapest *falsifying* experiment** (§12): use the prober's existing intervention-sweep + a tiny offline determinized search to test whether *any* small-budget search measurably beats the current policy at the selected hard states. If it doesn't beat the raw net there, the whole idea is dead and you've spent days, not months.
  2. If it passes, build **Option C-then-A**: a shallow, revealed-only, determinized 1–2-ply expectimax/Gumbel search run **off the hot path** in an eval-worker-style subprocess, distilled as a **separate auxiliary cross-entropy(policy)+MSE(value) loss** (never as fake PPO transitions), with the **critic-independent outcome-ambiguity selector**, not value-error.

This is a "yes, but stage it and verify the premise before paying for it" verdict. The most decisive single data point is that the closest published work in this exact domain — Wang's MIT thesis, your own foundational reference — paired PPO with MCTS and *deliberately declined* the training-time fold-back you propose, on your exact compute grounds (§6). That does not mean your idea is wrong; it means the premise is unproven enough that someone already chose not to bet on it, and you should verify it cheaply before building.

**One reframe that changes how to read this whole report (§6.5): your idea is not net-new — it is a sharper re-sequencing of work you have already designed.** Your own `designs/ai_v6`–`ai_v8` roadmap already commits to MCTS as a policy-improvement operator, already specifies the *exact* native-clone fork primitive this report calls the linchpin (`Battle.fromJSON(root.toJSON()); fork.prng = new PRNG()` in `designs/ai_v6/impl_step5_mcts.md`), and already designs the determinization world-sampler (the ai_v6 team-completion model) that §7 flags as the missing piece. So the linchpin is confirmed *three* ways (Showdown source verified in-tree, your own bridge design, and the workflow's adversarial verifier), and much of what §7/§9 treat as "net-new plumbing" is in fact already on your books. The genuinely novel contribution of your proposal — **selectivity** — is the one thing the v6/v7/v8 plan does *not* have, and it is exactly the lever that could pull v8-quality search targets forward onto the v6 JS bridge without waiting for the Rust sim. Read §6.5 before §9; it changes the build-vs-reuse math.

---

## 2. The idea, restated precisely

In RL terms, you proposed a **selective Expert-Iteration loop grafted onto on-policy PPO**:

1. **Collect** an on-policy PPO rollout as today (`MaskablePPO` + your dual-head transformer extractor).
2. **Select** the N "worst / most-novel / most-wrong" decision states the current policy just produced (you have a forensic prober that already ranks these).
3. **Search** at those N states only — run MCTS / look-ahead to compute an *improved* local policy distribution π_search(s) and value v_search(s), spending extra compute *selectively* rather than at every decision.
4. **Distill** the search outputs back into the network as part of the PPO update, so the apprentice net learns the expert's improved targets at the hard moments.
5. **Cadence**: do this per batch, inspired by Cursor's "fresh checkpoint every few hours" real-time-RL loop.

Formally: you are constructing an improved policy π'(s) = Search(π_θ, V_θ, s) at a *subset* S_hard of states, and training π_θ → π' by supervised imitation on S_hard while running standard PPO elsewhere. This is **ExIt with a state-selection (metareasoning) front-end and a PPO apprentice**. The policy-improvement theorem permits improving on a *subset* of states and still getting a (weak) overall improvement, so "selective" is not a hack — it is a valid restriction of generalized policy iteration.

Two non-obvious specification choices are already implied and matter enormously: **what "improved target" means in an imperfect-info simultaneous-move game** (§5), and **how the off-policy search target enters an on-policy PPO objective** (§9, §11).

---

## 3. What this is called (taxonomy & theory)

### The canonical name: Expert Iteration

The umbrella name is **Expert Iteration (ExIt)**, from Anthony, Tian & Barber, *"Thinking Fast and Slow with Deep Learning and Tree Search"* (NeurIPS 2017). ExIt decomposes RL into **planning** (tree search produces an improved local plan at a state) and **generalisation** (a neural net imitates those plans). The apprentice net guides the search; the search output becomes a supervised target the net trains toward. Your "run search to get a better target, then train the net toward it" is *exactly* this loop — the only twist is doing it selectively.

### The famous instance: AlphaZero, MCTS as π' = MCTS(π)

AlphaGo Zero / AlphaZero is the best-known instantiation: **MCTS is explicitly a policy-improvement operator**, and the net is trained by distilling the search visit-count distribution and the search-backed value. AlphaGo Zero's loss is literally L = (z − v)² − πᵀ log p: an MSE value term toward the search-backed outcome plus a cross-entropy policy term toward the MCTS visit distribution. Your "improved policy distribution + value back into the update" is structurally identical to AlphaZero target construction; PPO plays the role of the supervised distillation step.

### Why search yields a better target — the formal answer (Grill et al. 2020)

The rigorous "why" is **Grill et al. 2020, *"Monte-Carlo Tree Search as Regularized Policy Optimization"*** (ICML 2020). They prove AlphaZero's visit-count distribution approximates the solution of

> π̄ = argmax_y [ qᵀy − λ·KL(π_θ ‖ y) ]

i.e. it maximizes the search-estimated action values **while staying close (in KL) to the net's prior**. That is precisely a **regularized / trust-region policy-improvement step**: the search policy is provably an improvement over the prior *to the extent the search Q-values are accurate*, and never strays arbitrarily far from the prior. Two consequences bite directly on your design:

- **Their gains are largest at LOW simulation budgets** — exactly the compute-starved regime you must operate in. This is a genuine point *for* the idea.
- They show raw visit counts are a *noisy* target at low budget (information lag, integer-ratio discretization noise) and recommend distilling the **soft regularized target π̄ directly**, computed from the search Q-values, not the integer visit counts. **This is a non-negotiable correctness requirement for your low-sim setting** (see §9).

### The convergence container and the Newton-step intuition

- **Dual Policy Iteration** (Sun et al. 2018) is the general theoretical container: alternate a fast reactive net and a slow search policy, each improving the other; ExIt and AlphaZero are special cases, with convergence guarantees under the alternating-optimization framework — *provided* the search policy is a genuine local improvement and the net imitates it with bounded error.
- **Bertsekas** frames AlphaZero's online search as a **Newton step on the Bellman operator**. His stated lesson: *"the major determinant of the quality of the on-line policy is the Newton step performed on-line, while off-line training plays a secondary role."* Because Newton's method has fast (superlinear) local convergence, even a cheap online search corrects large offline-net error — strong support for your intuition that **a little well-placed search is high-leverage.** Caveat: Newton-step superlinearity is a *perfect-information* result; the imperfect-info, simultaneous-move setting voids the clean guarantee (§5), so read this as motivating intuition, not a transferable theorem.

### The "closest named precedent" for re-searching stored states: MuZero Reanalyze

The single closest named precedent for "re-run search on stored states to manufacture fresh targets" is **MuZero Reanalyze**: MuZero periodically re-runs MCTS on *already-collected* trajectories with the *latest* network to regenerate improved policy/value targets, decoupling search-target-generation from environment interaction. This is the mechanistic heart of your idea — selectively re-searching stored states and distilling — and is worth studying directly, because it is the production-grade version of "fold re-search into the learner."

### The selective part: metareasoning and targeted search

Choosing *which* states to spend search on is the **Value of Computation (VOC)** / rational-metareasoning problem (Russell & Wefald). The direct ML precedents:

- **Targeted Search Control / Go-Exploit** (Trudeau & Buro, AAMAS 2023) samples self-play start states from an *archive of states of interest* and gets better sample efficiency than uniform AlphaZero. **Its own premise is your design** — but it carries the caveat that *bites*: search is an effective improvement operator only where value estimates are *accurate*, and your "worst/most-wrong" states are precisely where the critic is *least* accurate.
- **Value-of-Computation MCTS** (Sezener & Dayan; VOI-MCTS, UAI 2017) — allocate search by value-of-information.
- **Epistemic MCTS / E-MCTS** (ICLR 2025) — search where the value net is *most uncertain*. This is the cleaner selector than realized value-error (see §8).

### Relation to prioritized replay & active learning

The selection step is structurally **prioritized replay** (Schaul et al. 2015) plus an **active-learning acquisition function** (BALD / query-by-committee). Both literatures supply the warnings in §8.

---

## 4. The Cursor real-time-RL connection

**Cursor's real-time RL is fast on-policy RL in production — explicitly *not* search or MCTS.** The blog describes serving checkpoints to production, aggregating user responses as reward, updating weights, running CursorBench, and redeploying — a loop of about five hours, letting them ship an improved Composer checkpoint multiple times a day. The stated reason is **staying on-policy**: keeping the data distribution matched to the current weights, so the model being trained is the same model that generated the data. **There is no mention of search, MCTS, look-ahead, or distilling a search policy anywhere.**

**Where the analogy holds:** the *cadence/freshness* intuition. Re-deriving targets frequently from the current policy is valuable, and on-policy data avoids the heavy-tailed importance weights that destabilize PPO under staleness.

**Where it breaks:** the *mechanism*. Cursor's lever is **data freshness** (improve the data you learn *from*). Your lever is **a better target** (improve the target you learn *toward*, via search). These are **orthogonal axes** that can coexist but solve different problems. Note also that **your launcher already provides Cursor's freshness lever** — the ~3h restart cadence is structurally the same "fresh checkpoint every few hours" loop. So Cursor justifies the *cadence intuition only*; it provides zero support for the search mechanism. Treat the Cursor citation as motivation for "do it per-batch / frequently," not as evidence the search part works.

---

## 5. Why Pokémon makes vanilla MCTS hard

Gen-3 OU violates **all three** of vanilla AlphaZero-MCTS's assumptions. Each violation has a known, separable fix.

### (a) Imperfect information → determinization / PIMC / ISMCTS

An imperfect-info state has **no well-defined scalar value**, so naive depth-limited MCTS is unsound. The cheap practical answer is **Perfect-Information Monte Carlo (PIMC) / determinization**: sample several concrete opponent teams from priors and search each as if perfect-info, then aggregate. **Your obs layer already carries the exact likelihood weights** (Smogon usage stats, ability/item/HP/hidden-power priors) that determinization needs.

But PIMC is **provably flawed** via two failure modes (Cowling/Powley/Whitehouse 2012):
- **Strategy fusion**: *"an AI agent can obviously not make different decisions from different states in the same information set... however, different decisions can be made in different determinizations."* This makes the agent over-confident that it will "know" the opponent's set, and it **cannot represent a mixed strategy** (the right answer to a 50/50 read).
- **Non-locality**: some determinizations are vanishingly unlikely because the opponent steers play away from them.

**Information Set MCTS (ISMCTS)** searches one tree of information sets, structurally removing one form of strategy fusion at modest extra cost; **Smooth UCT** (Heinrich & Silver, IJCAI 2015) is a lightweight middle ground that actually *converges* toward equilibrium where plain UCT diverges. The heavy, game-theoretically-rigorous route is CFR/MCCFR + sound depth-limited search over belief states — **ReBeL** (Brown et al. 2020), DeepStack, Student of Games — which is **almost certainly overkill** against your bot/self-play ladder.

**Is PIMC even appropriate here? You can answer this a priori.** Long et al. (AAAI 2010) give three measurable predictors — **leaf correlation, bias, disambiguation** — and PIMC works when info is revealed fast (high **disambiguation**). Pokémon has high disambiguation (sets reveal as moves/items fire, which you already track via incoming-damage/item-consumption beliefs), which is *why* FoulPlay's determinization works in gen9. **Caveat for you specifically:** gen3ou uses hand-built teams with **no team preview**, so disambiguation is lower than in random battles — measure these three numbers on your logged battles before assuming PIMC suffices. (This caveat is *measurable*, not asserted: the §12 experiment can estimate disambiguation from your logged battles' reveal curves before you commit to PIMC.)

### (b) Stochastic transitions → sample rolls / damage-roll grouping

Chance nodes / expectimax are exact but the branching from damage rolls × accuracy × crit × secondary procs explodes. **Do NOT learn the chance model** — Stochastic MuZero solves a problem you don't have (no simulator). You *have* Showdown. The pragmatic answer, proven by FoulPlay's **damage-roll grouping** (cluster rolls by whether they cause a faint, average each group, sum the likelihoods), is to exploit the simulator and collapse strategically-equivalent rolls, plus **open-loop MCTS** (resample stochastic outcomes per simulation and aggregate). **Important**: do *not* reuse the real PRNG seed in rollouts — reseed each rollout (`battle.resetRNG` / `>reseed`) so the search estimates over the roll distribution instead of overfitting this game's realized RNG.

### (c) Simultaneous moves → Decoupled UCT (DUCT)

Treating a simultaneous turn as sequential leaks information that doesn't exist and is exploitable. Lisý/Lanctot/Bowling (NeurIPS 2013) **prove UCT does not converge to Nash in simultaneous-move games and degrades over time**; soundness requires an ε-Hannan-consistent (Exp3 / regret-matching) selector. The empirical winner, though, is **Decoupled UCT (DUCT)** (Tak/Lanctot/Winands, CIG 2014): each player keeps independent UCB statistics and picks independently at a node — *theoretically unsound but the best performer in practice and what every real Pokémon bot ships.*

### The affordability lever: Gumbel few-sim

**Gumbel MuZero / "Policy improvement by planning with Gumbel"** (Danihelka et al., ICLR 2022) is the tool that makes selective few-sim search affordable. With Gumbel-Top-k + Sequential Halving you get a policy-improvement *target* at very few simulations, and your **11-action space is an unusually good fit** — with ≤10 legal actions you can consider the *entire* root (m = k), with no sampling-without-replacement penalty.

**Honest caveat on Gumbel (verification downgraded this):** the improvement is **in expectation** and **conditional on action-values being correctly evaluated** — the paper states this caveat verbatim, twice. That precondition fails exactly where you'd invoke it (worst-states = worst critic). Furthermore, *all* Gumbel experiments are deterministic, perfect-information, sequential (Go/chess/Atari); its "stochastic" extension is a noisy-root *bandit*, **not** stochastic transitions, and it never addresses simultaneous moves or hidden info. So Gumbel's clean guarantee **does not transfer to Gen-3 OU as written** — it degrades to "improvement only insofar as a determinized, regret-aware, simultaneous-move search produces accurate Q-values." With ≤11 actions, the marginal value of the Gumbel machinery over simply *ranking all root actions by a 1-ply evaluation* is also correspondingly smaller — which is itself an argument for the cheap Option C below.

---

## 6. Pokémon prior art

### Wang 2024 — the closest, and it *rejected your exact core* (now verified from the thesis)

Wang's MIT thesis, *"Winning at Pokémon Random Battles Using Reinforcement Learning"* (2024) — your own foundational reference (`designs/references/wang2024_pokemon_rl.pdf`) — pairs a **PPO self-play actor-critic with MCTS**, peaking at rank 8 (1693 Elo) on the gen4randombattles ladder, the best known non-human result for that format. **It used MCTS at *inference only* and explicitly rejected training on MCTS results on slow-simulator grounds — your exact compute profile.** This is the most important single data point: someone in this domain, with the same constraint, declined to do precisely the "fold search into training" step you propose, citing sim speed.

This is no longer a hedge — the thesis PDF states it verbatim (Ch. 3, Methods):

> "However, our approach diverges from that of AlphaZero in that **MCTS is not used to train the neural network**. Instead, the neural network is trained via PPO, then MCTS is used purely at inference time as a policy improvement operator. **This was done because simulating the environment is very slow**, compared to a game like chess; generating gameplay using MCTS would not likely lead to enough samples for a neural network to converge, given the computational constraints of the present work."

Two further details from the thesis matter for your design specifically:

- **Opponent modeling (§3.2.1):** Wang models the opponent inside MCTS with the trained NN policy and flags the exact failure mode you inherit — *"this... weakens the agent's performance against players who play differently from the neural network."* That is the same single-opponent-overfit risk your frozen-pool-net opponent model creates (see §11).
- **Wang himself notes the visit-count argmax is wrong when the optimal play is mixed (§5.2.1)** — *"in some situations the optimal strategy is to randomize... one action just barely edging out the other"* — independent confirmation of the strategy-fusion / mixed-strategy hazard in §5.

**Important caveat, retained:** Wang's results are *random battles* (sets drawn from a known distribution), so its hidden-info problem is *milder* than gen3ou's hand-built teams, and absolute Elo numbers are not comparable to yours. Notably, Wang's hidden-info sampling *relies on Showdown's known random-team generation procedure* (rejection-sampling valid sets) — a luxury gen3ou does not have, which is exactly the unrevealed-roster gap in §7.

### FoulPlay — the working existence proof for all three fixes

FoulPlay (Mariglia) placed **#1 in the Gen-9-OU battling track of the NeurIPS 2025 PokeAgent Challenge** (the challenge writeup confirms "FoulPlay in Gen 9 OU ... eventually victorious," via "an independent RL and search approach"). It stitches all three adaptations together:
- **Imperfect info**: likelihood-weighted determinization from Smogon usage stats + scraped teams + replays, *pruning* impossible sets from revealed evidence and refining stat beliefs from observed damage rolls;
- **Simultaneous moves**: **DUCT** (root-parallelized MCTS modified with Decoupled UCB);
- **Stochasticity**: **damage-roll grouping** (which maps directly onto your `gen3_incoming_damage_v2` faint-bucketing).

Three load-bearing caveats: (1) FoulPlay's MCTS is guided by a **hand-written evaluation function, not a learned value net** — it proves search is feasible, not that pairing it with a PPO net as a *training-target generator* is. (2) It **abandoned expectiminimax for MCTS+DUCT** because expectiminimax beyond ~5 turns timed out — so the strongest paradigm is MCTS, not expectiminimax. (3) It uses a **custom Rust engine (poke-engine) returning reversible instruction lists** rather than copying state, *explicitly because "Pokémon states can become quite large and costly to copy."* That is a direct primary-source signal that **the simulator state-copy/re-roll, not the evaluator, is the search cost driver** — and a hint that Showdown's native clone may be too slow for high-volume rollouts. (Note: specific finals scores / peak-ELO numbers for FoulPlay were *not* in the challenge writeup and are deliberately not stated here.)

### PokeChamp — even depth-limited search wins big

PokeChamp (NeurIPS 2024; ICML 2025 spotlight) is a **depth-limited minimax** LLM-driven agent that wins **76% vs all existing AIs and 84% vs the best heuristic bots**, reaching Elo ~1500 / top-10% on the human ladder. This is strong evidence that **shallow look-ahead delivers large practical gains in Pokémon** — directly supporting your cheap Option C. (Earlier draft language called this "one-step minimax"; the source describes *depth-limited* minimax, so treat "shallow," not "one-ply," as the supported claim.)

### Future Sight AI — real-time expectiminimax in the live client

A live Showdown bot running **depth-2 simultaneous-move expectiminimax within the turn window**, reportedly reaching roughly top-5% OU — another existence proof that shallow real-time search is feasible *at inference*. (Treated as a secondary, lower-confidence data point: it is a community bot, not a peer-reviewed result.)

### Metamon — the no-search baseline

*"Human-Level Competitive Pokémon via Scalable Offline RL with Transformers"* scales offline RL with transformers, **no search**, reaching top-10%. This is your "is search even worth it?" baseline, and it is sobering: a strong network with *zero* search already reaches the same ballpark as PokeChamp's searched Elo, so the marginal value of search over a better net is an open empirical question, not a given.

### Does selective-search-into-PPO already exist for Pokémon?

**No.** None of these published Pokémon systems injects search-improved policy/value as a **per-batch auxiliary distillation target during online PPO**. Wang's search is inference-only (and explicitly rejected for training); FoulPlay/Future Sight are search-with-hand-eval; Metamon is no-search. **Your idea is a genuine, sensible extension** — but it inherits every imperfect-info search caveat those works faced, and the one group that got closest chose not to do the training-time fold-back.

---

## 6.5 How this maps onto your OWN ai_v6 / v7 / v8 roadmap (read this before §9)

The grounding pass read `src/` but not `designs/ai_v6`–`ai_v8`. It should have: **your own roadmap already designs ~80% of this idea**, and reading it changes the build-vs-reuse calculus and *independently corroborates the linchpin.*

### What you've already designed

| Roadmap chapter | What it specifies | Relation to your proposal |
|---|---|---|
| **ai_v6 §step5 (`impl_step5_mcts.md`)** | MCTS as an **inference-time** policy-improvement operator: PUCT tree (`Q + α·P^β·√M/(N+1)`), `V_θ` leaf eval, `max_depth` 3–8 (deliberately *shallow* for the stochastic-variance reason §5 gives), DUCT-style per-worker stats, `F[s]` faint-count pruning, **PIMC via a team-completion model**, and a persistent Node `sim_bridge.js` whose fork is **`Battle.fromJSON(root.toJSON()); fork.prng = new PRNG()`** with `>advance` root-sync and reseeded rollouts. | This is the **search engine** your proposal needs, already specified down to the protocol. It confirms the §7 linchpin (native clone, reseed) and the §5 stochastic/PIMC fixes *in your own hand.* |
| **ai_v6 §step3–4 (team completion)** | A masked-slot **team-completion model** that samples full opponent teams from revealed slots — the PIMC world-sampler. | This is **exactly the fix** for the "unrevealed-roster gap" §7 flagged as the determinization weakness. You already plan to build the thing that closes it. |
| **ai_v6 §step1 (replay collection)** | Replay-collection daemon — **already landed**. | Supplies real ladder/human games to *train* the completion model and to mine hard states. |
| **ai_v7 (cheap MCTS in training)** | Add **K=3, max_depth=1 action sampling to PPO data collection** so both sides play better actions during rollouts (~+30 ms/decision on the JS bridge). | This is the *uniform-compute* sibling of your idea: search at **every** decision but very shallowly. |
| **ai_v8 (Rust sim)** | PyO3 Rust sim at ~50k rollouts/turn → **full MCTS on every training-time action**. | The *uniform-compute, deep* version — gated on a multi-month sim rewrite. |

### Where your proposal is genuinely new: selectivity (the metareasoning axis)

Your roadmap currently has only two training-time search settings, both **uniform over decisions**: *shallow-everywhere* (v7) and *deep-everywhere-but-needs-Rust* (v8). **Your proposal adds the axis neither has — spend the search budget *non-uniformly*, concentrating it on the worst-N states.** That is precisely the Value-of-Computation / targeted-search contribution from §3 (Go-Exploit, E-MCTS), and it is the lever that could let you run **deeper / more-determinized search than v7's K=3/depth-1 on the *same* JS bridge**, by paying for it at tens of states per batch instead of every decision — i.e. pulling some of v8's target quality forward without the Rust rewrite. **Frame it as "v7.5": selective deep search on the v6 engine, slotted between v7 (cheap-uniform) and v8 (deep-uniform).**

### One mechanism distinction the roadmap blurs — and your phrasing picks the harder branch

There are two different ways to "feed MCTS back into RL," and they are not the same:

- **MCTS-improves-the-data (ai_v7's framing).** MCTS chooses the action *actually executed* in the PPO rollout. The buffer now contains MCTS(π_θ) actions, not π_θ actions. This improves the *behavior* policy (closer to Cursor's data-freshness lever) — but it makes the rollout **off-policy w.r.t. the network**, so PPO's importance ratio is technically wrong unless you treat π_MCTS as the behavior policy. The v7 doc does not address this; **§11's off-policy hazard applies directly and the research is more rigorous than the roadmap here.**
- **MCTS-as-distillation-target (your phrasing: "feed the policy distribution + value back").** Keep the PPO rollout **on-policy**, and add a **separate auxiliary CE(π_θ, π̄) + MSE(V_θ, v_search) loss** at the selected states only (§9 Option A, AlphaZero/SAVE/PPG-style). This is the cleaner, ExIt-correct branch and the one this report recommends — it is *not* what ai_v7 specifies, so your idea is a real refinement of your own plan, not a duplicate of it.

When you write up the v6/v7 docs next, reconcile them with §9/§11 of this report: the "off-policy contamination," "distill π̄ not visit counts," "PopArt-normalize v_search," and "stall/forfeit-horizon" guardrails are missing from the current roadmap docs and are load-bearing.

### Net effect on the recommendation

Less is net-new than §7 implies: the **fork bridge**, **PIMC sampler**, and **replay mining** are already designed (one is already built). What remains genuinely net-new is (a) the **selection front-end** (cheap, §8 — nearly free from the rollout buffer), (b) the **aux-distillation loss + guardrails** (§9/§11), and (c) the **async off-hot-path service** (§10, templated on your eval-worker pool). And the §12 falsifier has **double ROI**: it does not just de-risk *this* proposal — it de-risks the v6/v7 MCTS bet you were going to make anyway, on the same JS bridge, for a few days of work.

---

## 7. Feasibility on gen3ai's stack

### The linchpin — sim state cloning: VERDICT FEASIBLE (verified)

The feared blocker is *not* a blocker. Showdown's sim has **native, production-tested serialization**: `Battle.toJSON()` → `State.serializeBattle`, `Battle.fromJSON()` → `State.deserializeBattle`, plus `restart()` to resume (`deps/pokemon-showdown/sim/battle.ts`, `sim/state.ts`). The round-trip is **asserted byte-identical *every turn*** by the sim's own test suite (`test/sim/misc/state.js`) — it forks a node, reconstructs from JSON, and continues playing identically. The PRNG seed is part of the snapshot, so forks diverge only by the actions you feed. Adversarial verification tried four ways to refute this (vestigial API? proposed-only? cost blowup? bridge can't reach it?) and **all four failed** — the verdict is **supported, high confidence**. Notably, this was driven by a real GitHub request (Issue #5270 → PR #5427) to add `dumpstate`/`loadstate` *specifically to enable MCTS*. **Confirmed in-tree on this checkout** (`git submodule update --init`): `sim/battle.ts:318 toJSON()`, `:322 static fromJSON()`, `:1968 restart()` (with the comment *"Deserialized games should use restart()"*); `sim/state.ts:61 serializeBattle` / `:84 deserializeBattle`; `sim/battle-stream.ts:22 battle: Battle` (the public field to clone from). **And independently confirmed in your own design** — `designs/ai_v6/impl_step5_mcts.md` already specifies the fork as `Battle.fromJSON(sessions.get(src).battle.toJSON()); fork.prng = new PRNG()` with `>advance` root-sync (see §6.5). The linchpin is as solid as a pre-build claim gets.

Key cost fact: **native clone is O(state size) (~sub-ms to low-ms in V8), NOT a from-scratch re-sim.** The "replays-from-scratch × sims × states" blowup applies *only* to the inputLog-replay fallback (O(turns-to-node) per fork), which you **do not need**. The in-process `BattleStream` even exposes the live `Battle` as a public field (`stream.battle`), so you can clone without round-tripping the text protocol.

**The real linchpin is orthogonal to cloning: imperfect information.** `toJSON()` returns the **omniscient ground-truth** state — true opponent EVs/item/ability and the future RNG seed. If you fork from *that*, your search **cheats**. Sound search requires you to (a) **determinize** — construct a *plausible* full state from beliefs rather than clone the true one (so the integration is "instantiate a fresh consistent battle at the current board," more work than a raw clone), and (b) **reseed** each rollout. The current bridge (`src/utils/bridge/local_sim_bridge.js`) **deliberately never reads sim state** ("the whole point is to feed poke-env the protocol stream"), so this needs **new, additive** bridge plumbing (SNAPSHOT/FORK/DROP commands managing a map of cloned `Battle` objects), and a concurrency decision (the persistent child holds one `BattleStream`; forking N nodes means N live `Battle` objects or a dedicated search subprocess).

**Open measurement (do not skip):** per-clone wall-cost at gen3ou state size is *inferred* sub-ms-to-low-ms from state size + V8, **not benchmarked**. FoulPlay's deliberate choice of a reversible-instruction Rust engine *over* Showdown clone is a primary-source warning that native clone may be too slow for high-volume rollouts. A 10-line micro-benchmark (`toJSON`/`fromJSON` in a loop on a mid-game gen3ou battle) should pin the real number *before* you size a sim budget.

### Determinization from existing priors: PARTLY feasible (verified)

You already have ~70% of the determinization machinery. For a **revealed** opponent mon you know species + public base stats + current HP%/status/boosts/types, and you must sample hidden item, ability, 4-move set, EV/nature spread. **Strong per-species marginal priors exist for all four** (`gen3_data.priors.{ability,items,moves,spreads,hidden_power}`), spread priors are **joint** (nature + all 6 EVs as one weighted unit, so the stat block samples coherently), and you have a working "sample a plausible set, compute an outcome" engine (`incoming_damage.py` + `incoming_damage_encoder.py`) plus a per-episode Bayesian-narrowing posterior (`HiddenPowerTracker`).

**Two real gaps:** (1) **the unrevealed-roster problem** — gen3ou has no team preview, so 1–5 of the opponent's 6 slots are blank with *no identity prior currently derived* (the Teammates co-occurrence + Checks-and-Counters data **exist in raw `gen3_smogon_stats.json` but are never derived into the facade**). **This is the exact gap your ai_v6 team-completion model (§6.5) is designed to fill** — so it is a *planned* prerequisite, not an unforeseen blocker; the offline §12 falsifier can run revealed-mon-only and defer it. (2) **Move priors are marginal** P(move in set) (sum ≈ 2.3, not 4), so independently sampling a 4-move set yields incoherent/illegal sets without a coupling step. **Implication: determinized worlds are faithful only for the *revealed* mons today** — which is fine for the single-attacker-vs-our-team case your model already reasons hardest about, and the highest-leverage place to search. (Contrast Wang: random battles let him *rejection-sample valid full sets* from Showdown's own generator; you have no such generator, which is exactly why the unrevealed-roster gap is gen3ou-specific.)

A further opportunity: the `HiddenPowerTracker` pattern (a per-episode posterior that narrows as moves/damage/speed are observed) generalizes to item/ability/spread, so determinized samples deep in a battle could be much tighter than the raw Smogon marginal. That is net-new state to thread through the env, but it would materially improve sample quality late-game where the hard decisions cluster.

### Worst-moment selection: FEASIBLE and cheap (verified)

There are two disjoint paths, and the distinction is the whole story:
- The **offline prober** (`src/main/prober/` + `BattleRecorder`) computes rich signals (value error, ΔV, TD residual, intervention sweeps, gradient saliency) but runs *only* in the frozen-snapshot eval subprocess, and several signals require re-running the model — **far too expensive per transition.** Reuse it as a **template/formula source**, not a drop-in online ranker.
- The **online path is the rollout buffer itself.** After `collect_rollouts`, `InstrumentedMaskablePPO.train()` holds the full batch in numpy — observations, actions, masks, values, log_probs, advantages, returns, rewards, episode_starts. **Verified locally**: the TD residual δ = r + γV(s′) − V(s) is literally computed and discarded inside `RolloutBuffer.compute_returns_and_advantage` (`stable_baselines3/common/buffers.py`), and `returns = advantages + values` there too. So **advantage magnitude, value error |returns − values|, and the TD residual are pure-numpy reductions over arrays already in memory, with zero extra battles and zero extra forward passes.** Policy entropy costs one extra batched masked forward (or skip it and use `old_log_prob` as a confidence proxy).

**One gap:** no novelty / state-visitation signal exists anywhere — "most-novel" selection (RND, embedding distance) would be net-new work.

### The PPO integration seam: clean (verified)

`InstrumentedMaskablePPO.train()` vendors `MaskablePPO.train` verbatim, with the loss assembled as `loss = policy_loss + ent_coef·entropy_loss + vf_coef·value_loss` (`src/agents/training/instrumented_ppo.py`). An auxiliary distillation term adds at that sum. Your dual-head extractor + `Gen3DualHeadMaskablePolicy.evaluate_actions` (`src/agents/model/policy.py`) already exposes **independent policy and value features**, so the two aux targets route to their natural heads with no architecture change, and the action mask applies to π_search exactly as it does to the live policy. The off-hot-path execution pattern also exists: `spawn_eval_workers` freezes a snapshot, `Popen`s `eval_worker`, and the parent polls non-blocking and merges per-shard JSON (`src/agents/training/eval_callback.py`) — the natural template for an **async search service**.

### The obs-build perf gate, if you capture forks at every decision

If the in-the-loop variant snapshots the bridge `BattleStream` at *every* decision (to have a fork handle ready when a state is later selected), that capture lands on the **env-worker hot path**, which is exactly the path the project's mandatory obs-build benchmark protects. Per `src/agents/observation/CLAUDE.md`, obs build is ≈88% of per-turn CPU and any change there must run `obs_build_benchmark.py` / `trainer_turn_benchmark.py` before/after with no meaningful regression. A `toJSON()` per decision (even sub-ms) is additive CPU on a CPU-saturated box and *must* be benchmarked under that gate, not assumed free. The cheaper alternative is to *not* pre-capture: snapshot lazily only the few states the post-rollout selector actually picks — but that requires the env to retain enough state to re-instantiate them, which the rollout buffer alone does not (see the next subsection).

### The one feasibility gate that remains hard

The rollout buffer stores the obs *vector* but **not a resettable game state** — re-deriving a searchable sim from a 3391-dim obs is lossy/impossible. So online MCTS on selected states requires **snapshotting the bridge `BattleStream` at the decision points** (new plumbing), which means the *env*, not just the trainer, must capture fork handles at every decision (or lazily re-instantiate, with the same plumbing). This is the real engineering gate for the in-the-loop variant, and it is what couples the idea to the obs-build perf gate above.

---

## 8. Defining "worst N"

This is where the literature is unusually pointed, and where the naive choice is actively harmful.

### Candidate signals and their failure modes

| Signal | Cost | Verdict |
|---|---|---|
| **Critic value-error** `|returns − values|` | free (in buffer) | **AVOID as primary selector** |
| **Advantage magnitude** `|A|` | free (in buffer) | usable, but partly critic-coupled |
| **Policy entropy** | 1 forward | "uncertain tossup" — fine as a secondary axis |
| **Outcome ambiguity** `p(1−p)` | free-ish | **recommended** (critic-independent) |
| **Committee disagreement** (pool/checkpoints) | forwards you already run | **recommended** (epistemic signal, nearly free) |
| **Loss outcome** (lost battle) | coarse/lagged online | weak proxy, batch-boundary issues |
| **Novelty / RND** | net-new | complement only, dodges stochastic trap |

### The critic-error feedback-loop pitfall — the named hazard, confirmed

Your stated worry is **theoretically confirmed**. **Actor-PER** (Saglam et al. 2022) *proves* the policy gradient computed on high-TD-error transitions **diverges from the true gradient**, because the critic is least reliable exactly where TD-error is largest — *"actor networks should be trained with low TD error transitions."* Reliability-Adjusted PER names the exact loop: *"an imprecise estimate initiates a feedback loop where suboptimal actions, favored by the inaccurate critic, can be reinforced."* For your MCTS design this is **doubly dangerous** because the search bootstraps on the same critic at its leaves — a critic blind spot both *selects* the state and *corrupts* the search target there.

**Pokémon makes it worse.** PER's own authors warn TD-error priority is amplified by *"noise spikes when rewards are stochastic"* (crits, damage rolls, accuracy misses) and by bootstrapping. This is the **aleatoric trap**: value-error mining selects high-*aleatoric*-variance states (irreducible randomness), not high-*epistemic* (reducible) ones — the precise distinction active-learning theory (BALD) tells you to respect. Your own memory notes the OHKO-belief critic is noisy and the policy under-switches on surprise OHKOs — value-error selection would steer search straight onto that aleatoric noise.

### The recommended cheap signal

For an **on-policy PPO batch**, the right unit is **problem/state-level outcome ambiguity**: prioritize states where the model wins some rollouts and loses others, priority ω = p(1−p), maximized at p = 0.5. It is **critic-independent** (computed from rollout outcomes the batch already produced), and a mixed-outcome state is *both* genuinely decisive *and* where extra search most changes the action distribution. Augment it with **committee disagreement** across your existing frozen self-play pool snapshots + recent checkpoints (a BALD-style epistemic signal you can compute almost for free, since you already run those forwards in eval, and the prober already does intervention sweeps).

**Estimability caveat:** the p(1−p) signal wants *repeated/near-duplicate states* in the batch to estimate a per-state success rate; in a self-play battle each exact 3391-dim state is seen roughly once per batch. So in practice use the *realized GAE advantage magnitude* or *committee disagreement* as the per-state proxy, or coarsen "state" to a recurring matchup bucket, rather than literally estimating p(1−p) per unique state. Measure this before relying on it.

**Three non-negotiable guardrails** (from the PER literature): (1) a **stochastic floor** (PER α<1) so selection can't collapse onto a recurring pathology like your documented under-switching; (2) keep an **unprioritized fraction** of the batch (Actor-PER); (3) always benchmark against a **uniform-random-N control** — calibrated/clever acquisition functions can *lose* to random.

---

## 9. Concrete proposed designs, RANKED

> **Recommended path up front: C → A → B.** Start at Option C (verify the premise cheaply, near-zero search cost). Graduate to Option A only if C's shallow search measurably beats the raw net at the selected states. Reach for Option B (learned model) only if real-sim search proves empirically too slow under training contention — which the cloning verdict suggests it won't, at shallow depth. Do not start with full in-the-loop ISMCTS.

### Option C — cheap approximations (RECOMMENDED FIRST)

**What it is.** No tree. Three flavors, cheapest first:
1. **Pure hard-example upweighting, NO search** — just upweight the selected outcome-ambiguous states in the PPO loss. Captures the "spend the learning signal where it matters" intuition at *zero* search cost.
2. **One-step value-based action relabeling** — for each selected state, evaluate all ≤10 legal actions with a shallow determinized 1-ply expectimax over a handful of sampled damage/speed-tie/secondary-proc outcomes + the value net, then distill toward the resulting soft π̄.
3. **Shallow 1–2-ply DUCT expectimax** with damage-roll grouping and K determinizations.

**What it buys.** PokeChamp shows depth-limited minimax wins 76%/84% vs AIs/heuristic bots — *most of the search value lives in the first plies.* Hamrick et al. 2021 found shallow trees with simple rollouts often match deep search, and planning's main payoff is *better training targets* — exactly your use. With ≤11 actions, ranking all root actions by a 1-ply eval is cheap and degenerates the Gumbel machinery's advantage away anyway.
**Cost.** Lowest. Flavor 1 is free. Flavors 2–3 need the determinization + fork plumbing but at depth 1–2 only.
**Risk.** Low. Shallow search is less likely to amplify strategy-fusion bias than deep determinized lines.
**Effort.** Low-to-medium. Flavor 1 is a few days; flavor 2 needs the offline determinizer + a fork command.

### Option A — real-sim selective ISMCTS + ExIt aux-distillation (the "full" version)

**What it is.** At the N selected states, run **determinized DUCT/ISMCTS** over forked `Battle` clones (native `toJSON`/`fromJSON`), opponent modeled by your frozen pool net's own distribution, damage-roll-grouped stochastic outcomes, K determinizations from priors. Distill π̄ + v_search as a separate aux loss.
**What it buys.** The strongest target; reaches 10+ turns on promising lines (FoulPlay's depth). The closest thing to "real AlphaZero in Pokémon."
**Cost.** Highest. Each searched state costs K × sims × depth sim-plies + clones, on a CPU-saturated box. The determinization + fork concurrency plumbing is real engineering.
**Risk.** Highest. Strategy fusion + non-locality bias the target; deep determinized lines are *most* exposed to it; simultaneous-move DUCT is unsound; the unrevealed-roster gap means deep lines past the revealed mon are poorly grounded; and the frozen-opponent model makes the target a best-response to one opponent (§11).
**Effort.** High — multi-week, correctness-critical.

### Option B — learned-model (Stochastic / Gumbel MuZero) selective search

**What it is.** If real-sim cloning proves too slow under contention, search over a *learned* model instead.
**What it buys.** Search decoupled from sim throughput; MuZero Reanalyze is the canonical "re-search stored states" engine.
**Cost.** Very high upfront (you'd train a dynamics model you don't currently have).
**Risk.** High and **largely wasted** — you *have* a fast exact simulator; learning chance (Stochastic MuZero) solves a problem you don't have. The learned model would be *least* accurate at exactly the rare hard states you target.
**Effort.** Highest. **Not recommended** given the verified-feasible native clone.

---

## 10. Compute-budget analysis

### Where the cost actually goes

FoulPlay's primary source is decisive: their custom engine returns **reversible instructions rather than copying state** *because "Pokémon states can become quite large and costly to copy."* So **the simulator fork/re-roll is the marginal cost driver of the search subsystem** — not the NN evaluation. (Subtle framing point: in your *current training*, py-spy shows ~70% of busy per-worker CPU is the *opponent NN forward* and the JS-sim round-trip is only ~16% of blocked time — the NN dominates *today*. It is only in the *contemplated search* that sim-fork/re-roll becomes the *new* marginal cost, because the NN pass is already cheap and on the hot path while forking is net-new. Do not state "the simulator is the bottleneck" as a property of current training — it is a property of the search subsystem specifically.)

### The arithmetic against your actual budget (this is a stacked estimate, not a measurement)

The numbers below are an **illustrative stacked estimate** built from the project's documented FPS and the bridge's measured per-step latency — *not* a measured search cost. Each factor (N, sims, depth, K, per-ply cost) is a planning assumption; treat the conclusion as "plausibly 0.3–2× the rollout budget depending on choices," and pin the real per-ply and per-clone costs with the §12 / micro-benchmark before committing.

- **Rollout budget:** ~1489 FPS at n_envs=64 (your `CLAUDE.md`). A 2048×64 ≈ 131,072-transition batch takes **~88s of rollout wall-clock**.
- **A modest search:** N=64 states × 32 sims × depth~4 × K=4 determinizations ≈ **33k sim-plies/batch.**
- At the bridge's measured **~6.1 ms/step** (which includes IPC + obs + forward, so a *bare* sim ply is cheaper), that's **~196s — 2.2× the entire rollout budget.** Even at an optimistic sub-1ms *native clone* it's **~33s, a ~37% throughput hit.**

The honest, sobering reading. The verification verdict on cost is **partially-supported**: "selective is cheaper than full search" is arithmetically trivial (~2000× fewer searched states than searching every decision) but nearly tautological — it says selective beats the most expensive possible thing, not that it is cheap in absolute terms. Gumbel makes per-state cost low *in principle*, but **"affordable" against your specific saturated-CPU budget is not established**, and holds *only* if you (i) build the fast clone primitive (doesn't exist today), (ii) keep N genuinely small (tens), (iii) use shallow depth, and (iv) run it off the rollout critical path.

### The structural fit: run search off the hot path

This is where your existing infra saves you. The box is **rollout/latency-bound with the GPU ~86% idle** — so added search competes for the **saturated CPU, not the idle GPU**. The right shape is your own `--async-rollout` (`AsyncSubprocVecEnv`, drain-safe RPC, overlaps CPU stepping with GPU forward) + the **non-blocking work-stealing eval-worker** pattern (`eval_sharding`). An **async search service** that ingests the prober's ranked states, searches on spare cores / the `--eval-device`, and returns improved (π_search, v_search) pairs to a small replay buffer is the natural home. **But** beware: if search is async, the targets may **age out of the on-policy window**, reintroducing the staleness Cursor avoids — handle as a decoupled distillation phase (PPG-style), not as PPO transitions (§11).

---

## 11. Risks & failure modes

| Risk | Why it bites here | Mitigation |
|---|---|---|
| **Off-policy contamination of PPO** | Search targets are off-policy by construction (what the policy *should* have done, not what it did); feeding them through the PPO ratio breaks importance sampling and can cause catastrophic KL collapse. | **Never inject as PPO transitions.** Add a *separate* supervised aux loss `L_aux = w_π·CE(π_θ, π̄) + w_v·MSE(V_θ, v_search)` on selected states only, AlphaZero/SAVE-style. Borrow PPG's decoupled auxiliary phase. Small **adaptive** w_aux + a hard **KL cap** so the aux term can't leave PPO's trust region in one step. |
| **Distilling raw visit counts** | At your low sim budgets, integer visit counts are a *noisy* target (Grill 2020). | Distill the **soft regularized target π̄** (softmax over search-Q tempered toward the net's prior), not visit counts, not one-hot argmax. SAVE's ablation: distill the **ranking**, not raw values. |
| **Determinization bias / strategy fusion** | PIMC manufactures an over-confident "I know the opponent's set" target that can't represent a mixed strategy — distilling it makes the policy **more exploitable on 50/50 reads**. *This is the single biggest theory-to-practice risk* (Wang §5.2.1 independently flags the same mixed-strategy gap). | Prefer ISMCTS over PIMC where affordable; keep search **shallow** (less exposed); the **sound** version of this exact fold-back loop is **ReBeL** (search over belief states), the reference if soundness shows up empirically. **Mitigate by validation, not theory** — A/B the distilled target vs the un-searched target on the ladder. |
| **Search target overfits to the frozen pool opponent** | Modeling the opponent inside search with your own frozen pool net (the cheap, natural choice) makes π_search a *best response to that one opponent*. Wang 2024 observed this verbatim: it "weakens the agent's performance against players who play differently from the neural network." Distilling it can sharpen the policy against the pool while making it worse against off-distribution play (e.g. the heuristic bots, or a human). | Diversify the search opponent: sample across *several* pool sentinels / recent checkpoints per determinization, not one fixed net; or mix in a stochastic/uniform opponent. A/B the distilled policy against the **bot suite and a held-out opponent**, not just `win_rate_vs_pool` (which is already pinned near 50% by the promotion gate and will not reveal this regression). |
| **Simultaneous-move exploitability** | DUCT is theoretically unsound; UCT provably diverges from Nash. | Use DUCT (empirical best) but *measure* exploitability via self-play; escalate to regret-matching/Smooth UCT only if the self-play pool starts exploiting the distilled policy. |
| **Selection bias / feedback loop** | Critic-error selection picks the critic's own blind spots; the search bootstraps on that same critic → the loop relocates into the *target*. | Select on **critic-independent outcome ambiguity** + committee disagreement (§8). Give selected states **enough sims/depth** that the target is less critic-dependent than the critic itself — else you've just relocated the loop. |
| **Stall/forfeit-loss horizon mismatch** | gen3ai's anti-stall design makes a `turn ≥ cap` timeout a **forfeit-LOSS terminal** (`--draw-penalty`), and the no-progress clock charges a tax. A short-horizon search that doesn't see the looming cap will value a stall line as neutral/positive while the *real* terminal is a loss — so it can manufacture a target that endorses exactly the PP-exhaustion / heal-war stalls the reward redesign is trying to kill. | The search rollout must inherit the *same* terminal/timeout semantics as training (timeout → loss, no-progress tax), or be depth-capped well short of the stall regime. Validate searched targets specifically on the stall battles the project already logs. |
| **Stale targets** | Async search returns targets after the on-policy window. | Decoupled PPG aux phase tolerates mild staleness; cap target age; re-search (Reanalyze-style) with the current net if a target is too old. |
| **Search buffer must survive a launcher restart** | The launcher restarts the child every ~3h (and on crash); an in-RAM buffer of {state, π_search, v_search} pairs and any in-flight async search jobs are lost across that boundary, and the resume contract assumes a clean checkpoint. A half-applied search buffer at restart risks a corrupted/asymmetric update. | Treat the search buffer as ephemeral and **fully drained before each checkpoint/SIGTERM** (mirror the bridge child's "die → crash → restart" contract, not in-place recovery); or persist it alongside the checkpoint with the same version/`git_hash` gating. Do not let a partially-consumed buffer straddle a restart. |
| **Value target must match PopArt normalization** | The run uses PopArt value normalization (now standard); the critic head outputs *normalized* values. A raw `v_search` (in real return units) distilled with MSE against the normalized head would fight PopArt's running σ/μ and corrupt the value scale PopArt just stabilized. | Pass `v_search` through PopArt's *same* normalization (current μ, σ) before the MSE, or distill in normalized space end-to-end. This is a small but easy-to-miss correctness requirement given PopArt is on by default. |
| **Selection collapse / coverage starvation** | Per-batch worst-N can grind the same recurring pathology every batch, starving coverage. | PER α<1 stochastic floor + unprioritized batch fraction + periodic uniform-random-N control. |
| **Search adds no value at hard states** | At tiny budgets over a stochastic simultaneous-move node, the "improvement" can be within its own variance → a noisy target no better than the prior. | **The §12 falsifying experiment.** Verify offline before wiring anything in. |

---

## 12. Recommended next step — the single cheapest falsifying experiment

**Before building any training integration, run a pure-offline test of the one premise the whole idea rests on: *does a small-budget determinized search measurably beat the current policy at the selected hard states?*** If it doesn't, every downstream component is wasted effort. This is exactly the premise Wang 2024 declined to bet on for training — so verify it cheaply rather than inheriting his assumption or rejecting it blind.

Concretely, reusing what you already have:

1. **Pick states.** Use the prober's existing ranking to pull ~100–300 hard decision states from *lost* battles — but rank them by the **critic-independent outcome-ambiguity / committee-disagreement** signal (§8), *and* keep a uniform-random control set of equal size.
2. **Search them offline.** For each state, run a **shallow (1–2 ply) determinized expectimax/DUCT** over a handful of K determinizations sampled from your existing priors (revealed-mon-only is fine), with damage-roll grouping, opponent = your frozen pool net's distribution, reseeded rolls. Use the native `Battle.toJSON`/`fromJSON` clone (or, fastest to prototype, the prober's intervention-sweep harness, which already re-runs alternative actions). This needs **no training loop, no PPO changes, no async service.**
3. **Measure four things.** (a) Does π_search *disagree* with π_θ at these states (if not, search is inert)? (b) When it disagrees, does the searched action **win more often** in a few hundred reseeded bridge rollouts from that state (the ground-truth improvement test)? (c) Is the win-rate lift on the **hard-selected** set meaningfully larger than on the **random-control** set (does selection beat random)? (d) **As a free side-product, estimate Long et al.'s disambiguation** from the reveal curves of these battles, to sanity-check that PIMC is even appropriate for gen3ou (low team-preview disambiguation is the main reason it might not be).
4. **Decision rule.** If searched actions don't beat the policy's actions at the selected states — or don't beat them by more than at random states — **stop.** The target is noise around the prior and distilling it will at best do nothing, at worst inject strategy-fusion bias. If they *do* win materially more, you have an empirical green light to build Option C's distillation, with a quantified expected lift to size the compute budget against, and a measured per-ply/per-clone cost to plug into §10's arithmetic.

This experiment is days of work, reuses the prober + bridge you already trust, touches zero training code, and directly falsifies the proposal's core assumption.

---

## Appendix A — Annotated bibliography

**Expert Iteration / search-as-policy-improvement theory**
- Anthony, Tian & Barber, *Thinking Fast and Slow with Deep Learning and Tree Search* (ExIt), NeurIPS 2017 — https://arxiv.org/abs/1705.08439 ; https://proceedings.neurips.cc/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html
- Grill et al., *Monte-Carlo Tree Search as Regularized Policy Optimization*, ICML 2020 — https://arxiv.org/abs/2007.12509 ; https://proceedings.mlr.press/v119/grill20a.html ; full text https://ar5iv.labs.arxiv.org/html/2007.12509
- Sun, Gordon, Boots, Bagnell, *Dual Policy Iteration*, NeurIPS 2018 — https://arxiv.org/abs/1805.10755 ; https://papers.nips.cc/paper/7937-dual-policy-iteration
- Bertsekas, *Newton's Method for RL and MPC* — https://www.sciencedirect.com/science/article/pii/S2666720722000157 ; https://www.mit.edu/~dimitrib/Newton'sMethodforRLMPC.pdf ; *Lessons from AlphaZero* http://www.athenasc.com/Lessons.html
- AlphaGo Zero as policy improvement (inference.vc) — https://www.inference.vc/alphago-zero-policy-improvement-and-vector-fields/
- AlphaZero — Silver et al., *Mastering Chess and Shogi by Self-Play* — https://arxiv.org/abs/1712.01815
- MuZero (incl. Reanalyze) — Schrittwieser et al. — https://arxiv.org/abs/1911.08265
- Sutton & Barto policy improvement theorem (Ch. 4) — https://lcalem.github.io/blog/2018/09/24/sutton-chap04-dp

**Selective / targeted search & metareasoning**
- Trudeau & Buro, *Targeted Search Control in AlphaZero* (Go-Exploit), AAMAS 2023 — https://arxiv.org/abs/2302.12359 ; https://dl.acm.org/doi/10.5555/3545946.3598720
- Sezener & Dayan, *Static and Dynamic Values of Computation in MCTS* — https://arxiv.org/abs/2002.04335
- *MCTS using Batch Value of Perfect Information*, UAI 2017 — https://auai.org/uai2017/proceedings/papers/37.pdf
- *E-MCTS: Deep Exploration by Planning with Epistemic Uncertainty*, ICLR 2025 — https://openreview.net/forum?id=zrCybZXxC8 ; *Epistemic MCTS* — https://arxiv.org/abs/2210.13455

**Few-simulation / affordable search**
- Danihelka et al., *Policy improvement by planning with Gumbel* (Gumbel MuZero), ICLR 2022 — https://openreview.net/forum?id=bERaNdoegnO ; PDF https://openreview.net/pdf?id=bERaNdoegnO ; mctx https://github.com/google-deepmind/mctx
- Hamrick et al., *On the role of planning in model-based deep RL*, ICLR 2021 — https://arxiv.org/abs/2011.04021
- Hamrick et al., *Combining Q-Learning and Search with Amortized Value Estimates* (SAVE), 2019 — https://arxiv.org/abs/1912.02807 ; https://ar5iv.labs.arxiv.org/html/1912.02807
- Anthony et al., *Policy Gradient Search: Online Planning and Expert Iteration without Search Trees*, 2019 — https://arxiv.org/abs/1904.03646 ; https://ar5iv.labs.arxiv.org/html/1904.03646

**Imperfect-info / stochastic / simultaneous-move search**
- Cowling, Powley, Whitehouse, *Information Set Monte Carlo Tree Search*, 2012 — https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf ; journal version https://www.sciencedirect.com/science/article/pii/S0004370214001052
- Long, Sturtevant, Buro, Furtak, *Understanding the Success of Perfect Information Monte Carlo Sampling*, AAAI 2010 — https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf ; https://ojs.aaai.org/index.php/AAAI/article/view/7562
- Heinrich & Silver, *Smooth UCT Search in Computer Poker*, IJCAI 2015 — https://www.ijcai.org/Proceedings/15/Papers/084.pdf
- Lisý, Lanctot, Bowling, *Convergence of MCTS in Simultaneous Move Games*, NeurIPS 2013 — https://arxiv.org/pdf/1310.8613
- Tak, Lanctot, Winands, *MCTS Variants for Simultaneous Move Games*, CIG 2014 — https://www.mlanctot.info/files/papers/cig14-smmctsggp.pdf
- Lisý, Lanctot, Bowling, *Online Outcome Sampling MCCFR*, AAMAS 2015 — https://mlanctot.info/files/papers/aamas15-iioos.pdf
- *Algorithms for Computing Strategies in Two-Player Simultaneous Move Games* — https://dke.maastrichtuniversity.nl/m.winands/documents/sm-journal.pdf
- Brown et al., *ReBeL: Combining Deep RL and Search for Imperfect-Information Games*, 2020 — https://arxiv.org/pdf/2007.13544
- *Student of Games* — https://arxiv.org/pdf/2112.03178 ; *Search in Imperfect Information Games* survey — https://arxiv.org/pdf/2111.05884
- Antonoglou et al., *Stochastic MuZero*, ICLR 2022 — https://openreview.net/pdf?id=X6D9bAHhBQ1

**Pokémon prior art**
- Wang, *Winning at Pokémon Random Battles Using Reinforcement Learning* (MIT 2024) — https://dspace.mit.edu/handle/1721.1/153888 ; local copy `designs/references/wang2024_pokemon_rl.pdf` (Ch. 3 verbatim: MCTS inference-only, *not* used to train, on slow-sim grounds; §3.2.1 opponent-modeling weakness; §5.2.1 mixed-strategy gap)
- Mariglia, *Foul Play* writeup — https://pmariglia.github.io/posts/foul-play/ ; repo https://github.com/pmariglia/foul-play ; engine https://github.com/pmariglia/foul-play/blob/main/ENGINE.md
- *The PokeAgent Challenge* (NeurIPS 2025) — https://arxiv.org/html/2603.15563v2 ; competition page https://pokeagent.github.io/ ; PDF http://sethkarten.ai/data/NeurIPS_2025_PokeAgent_Challenge.pdf
- *PokeChamp: an Expert-level Minimax Language Agent* (NeurIPS 2024; ICML 2025 spotlight) — https://arxiv.org/abs/2503.04094 ; https://github.com/sethkarten/pokechamp
- Metamon, *Human-Level Competitive Pokémon via Scalable Offline RL with Transformers* — https://arxiv.org/html/2504.04395v1 ; https://github.com/UT-Austin-RPL/metamon

**Selection / prioritization / active learning**
- Schaul et al., *Prioritized Experience Replay*, 2015 — https://arxiv.org/abs/1511.05952 ; https://ar5iv.labs.arxiv.org/html/1511.05952
- Saglam et al., *Actor Prioritized Experience Replay*, 2022 — https://arxiv.org/abs/2209.00532
- *Prioritized Replay for RL Post-training* — https://arxiv.org/abs/2601.02648 ; AERO — https://arxiv.org/abs/2602.14338 ; *Not All Rollouts Are Useful* — https://arxiv.org/abs/2504.13818
- *Uncertainty Prioritized Experience Replay*, RLC/RLJ 2025 — https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_45.pdf ; *Reliability-Adjusted PER* — https://arxiv.org/abs/2506.18482
- BALD overview — https://www.emergentmind.com/topics/bayesian-active-learning-by-disagreement-bald ; *Epistemic Uncertainty Sampling* — https://arxiv.org/abs/1909.00218 ; *When Active Learning Fails* — https://arxiv.org/abs/2511.17760
- Random Network Distillation (Burda et al. 2018) — https://www.emergentmind.com/topics/random-network-distillation-rnd

**PPO integration / distillation / on-policy cadence**
- AlphaGo Zero loss derivation — https://julien-vitay.net/deeprl/src/4.5-AlphaGo.html
- *Policy or Value? Loss Function and Playing Strength in AlphaZero-like Self-play* — https://liacs.leidenuniv.nl/~plaata1/papers/CoG2019.pdf
- Cobbe et al., *Phasic Policy Gradient*, 2020 — https://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf ; CleanRL docs https://docs.cleanrl.dev/rl-algorithms/ppg/
- *VLA Post-Training via Action-Chunked PPO and Self Behavior Cloning* (adaptive BC-aux weight) — https://arxiv.org/html/2509.25718
- PPO (clipped surrogate / collapse) — https://en.wikipedia.org/wiki/Proximal_policy_optimization

**Cursor / async RL**
- Cursor, *Improving Composer through real-time RL* — https://cursor.com/blog/real-time-rl-for-composer ; https://mlq.ai/news/cursor-improves-composer-ai-using-real-time-reinforcement-learning/
- *AsyncFlow* — https://arxiv.org/abs/2507.01663 ; *Stable Asynchrony Variance-Controlled Off-Policy RL* — https://arxiv.org/abs/2602.17616 ; *A-3PO* — https://arxiv.org/html/2512.06547

**gen3ai stack (verified local references)**
- Sim clone primitive: `deps/pokemon-showdown/sim/battle.ts` (`toJSON`/`fromJSON`/`restart`), `sim/state.ts`, `test/sim/misc/state.js` (per-turn round-trip assert), `sim/battle-stream.ts` (`.battle` public field), `sim/prng.ts` (`resetRNG`/`>reseed`); GitHub Issue #5270 → PR #5427 (added `dumpstate`/`loadstate` for MCTS)
- PPO seam: `src/agents/training/instrumented_ppo.py` (vendored `train`, loss sum), `src/agents/model/policy.py` (`Gen3DualHeadMaskablePolicy.evaluate_actions`, independent π/v features)
- Rollout-buffer signals: `stable_baselines3/common/buffers.py` (`RolloutBuffer.compute_returns_and_advantage`: TD δ computed+discarded, `returns = advantages + values`)
- Selection/eval infra: `src/main/prober/engine.py`, `src/agents/training/eval_callback.py` (`spawn_eval_workers`), `src/agents/training/battle_recorder.py`
- Action space / bridge: `src/agents/action/constants.py` (`ACTION_SPACE_SIZE = 11`), `src/utils/bridge/local_sim_bridge.js` (deliberately no sim-state read), `src/utils/bridge/README.md` (~6.1 ms/step bridge)
- Determinization priors: `src/agents/gen3_data/priors.py`, `data/pokemon/gen3_smogon_stats.json` (Teammates/Checks data present but underived), `src/agents/training/hidden_power_tracker.py`, `src/agents/observation/incoming_damage.py`
- Perf gate + PopArt + anti-stall couplings: `src/agents/observation/CLAUDE.md` (obs-build benchmark gate), project memory `project_popart.md` (PopArt value normalization, now standard), `project_anti_stall_fix.md` / `project_markovian_reward_design.md` (timeout→forfeit-loss terminal, no-progress clock)
