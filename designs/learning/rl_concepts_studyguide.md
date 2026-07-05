# RL / Interpretability Concepts — Study Guide

A brush-up reference for the concepts behind the gen3ai diagnosis and roadmap. For each: the
**intuition**, the **technical** version, and **where it bit us** (the connection to this project).
Companion to `designs/ai_v8/design_curriculum_uncertainty.md` and `design_opponent_system_parity.md`.

---

## Part 1 — The PPO objective and the learning signal (the spine)

**Policy gradient**
- *Intuition:* nudge the policy to make good-outcome actions more likely and bad ones less likely, weighted by how much better/worse than expected they turned out.
- *Technical:* maximize `J(θ)=E_τ[Σ γ^t r_t]`; gradient `∇J=E[∇log π_θ(a|s)·A(s,a)]`. Ascend the log-prob of actions in proportion to their advantage.
- *Where it bit us:* everything — "why is it archetype-blind" is "what does this gradient incentivize."

**Value function / critic / baseline**
- *Intuition:* a running estimate of "how good is this position" — a yardstick to judge whether an action beat expectations, not the goal itself.
- *Technical:* `V^π(s)=E[Σγ^t r_t | s]`. Subtracting it as a baseline reduces variance without bias (baseline invariance: `E[∇log π · b(s)]=0`).
- *Where it bit us:* the critic's blindness to defensive positions (AUC≈0.5) is the central diagnostic.

**Advantage & GAE**
- *Intuition:* "how much better was this action than average from here." The actual learning signal.
- *Technical:* `A(s,a)=Q(s,a)-V(s)`. GAE: `Â_t=Σ_l (γλ)^l δ_{t+l}`, `δ_t=r_t+γV(s_{t+1})-V(s_t)`. λ trades bias (trust critic) vs variance (trust returns).
- *Where it bit us:* the deepest idea in the whole thread.

**The advantage-signal principle (the key one)**
- *Intuition:* if the outcome doesn't depend on what you do — win regardless (floor) or lose regardless (unwinnable) — there's nothing to learn; good and bad actions look identical.
- *Technical:* action-independent outcome → `Q(s,a)≈V(s)` ∀a → `A≈0` → `∇J≈0`.
- *Where it bit us:* defensive floor, unwinnable generalist games, luck floor, slow ramp, defensive blindness — all this one mechanism.

**Reward shaping & PBRS**
- *Intuition:* you can add hints to the reward, but a safe class of hints can't change the optimal policy, only how fast you find it.
- *Technical:* PBRS `F(s,s')=γΦ(s')-Φ(s)` is advantage-invariant (`A_shaped=A_base`); speeds value-learning, can't bias behavior.
- *Where it bit us:* can't make it heal/play-defensive via a material potential; the lever is opponents, not reward.

**Entropy regularization**
- *Intuition:* pay the policy to stay uncertain so it keeps exploring.
- *Technical:* add `β·H(π(·|s))`; higher β = flatter distribution = more exploration.
- *Where it bit us:* "the only lever we really have" for the specialist; entropy collapse = one axis of "too finely trained."

---

## Part 2 — Why learning stalls (the pathologies)

**Two kinds of "stuck"** — optimization local minima vs objective fixed points (self-play Nash where no gradient improves you *given the opponent distribution*). Different cures. The plateau is the latter; "train longer" can't escape it.

**Effective rank vs capacity** — capacity = width (128); effective rank = # principal components carrying the variance (~3–5). Big gap = impoverished representation despite abundant room. (Your "it's plenty big, why can't it represent this" correction.)

**Rank collapse / implicit under-parameterization** — TD bootstrapping (`r+γV_θ(s')` targets from θ) self-distills and *shrinks* feature rank over training (Kumar et al.), correlating with worse performance. Leading candidate for the 3–5 dims — a pathology, not "task is low-dim."

**Plasticity loss / primacy bias / dormant neurons** — over training, units go dormant, rank falls, the net loses the ability to fit new targets (Nikishin, Sokar, Lyle, Dohare). Fixes: resets, LayerNorm, weight decay.

**Simplicity bias / implicit regularization** — SGD prefers the simplest function that fits; the objective's needs set realized complexity, not the architecture's allowance. Why 123 dims sit idle.

**Shrink-and-perturb / resets / ReDo** — `w←αw+ε` (soft reboot); reset dormant neurons (ReDo) or last layers (primacy-bias fix). The honest "reset probabilities" answer.

**Double descent & grokking (why they don't apply)** — supervised, fixed-dataset phenomena; RL has a moving distribution and nothing fixed to memorize. The *hope* (plateau can break) is real but via the S-curve, not double descent.

**S-curves / takeoff dynamics** — sigmoidal RL curves: slow (A≈0, win nothing) → rapid (wins → A informative → self-reinforcing) → saturate. Your "once it beats bots it grows fast" hypothesis; fixed opponents saturate it.

---

## Part 3 — Curriculum & self-play

**ACL** — task-selection as its own optimization; a teacher picks tasks to maximize the student's learning.

**Zone of proximal development / frontier** — max info where outcome variance is highest, win-prob ≈ 0.5 (advantage variance ∝ `p(1-p)`). Target contested, not hardest. (ai_v7_05 maxed difficulty → stalled.)

**Regret vs difficulty** — sample by "how much better could I have done" (learnable gap), winnability-gated, not raw difficulty.

**Learning progress** — the empirical time-derivative of performance per style; value-*free*; distinguishes contested (rising) from unwinnable (flat). The escape from the circularity.

**PFSP / self-play auto-curriculum** — sample opponent i ∝ `f(p_i)` on *empirical* win-rate; variance-peaked `p(1-p)` targets contested. Self-play = opponent scales with you → always ~50% → auto-curriculum. Why ai_v7_02 ramped and ai_v7_05 didn't.

**PLR** — revisit levels by learning potential (TD-error/regret); RL analog of prioritized replay over tasks.

**Group DRO / CVaR** — optimize the *worst group* (`min_g E_{D_g}`) or the worst α-tail, not the average. The objective-level fix for the offense bias.

**The curriculum circularity & value-free escape** — a value-driven difficulty estimate inherits the critic's blindness (circular). Escape: empirical win-rate / learning progress (never query the critic); the value function co-improves as a consequence. State-level targeting stays circular → deferred.

---

## Part 4 — Uncertainty & ensembles

**Epistemic vs aleatoric** — epistemic = "haven't seen enough" (reducible); aleatoric = genuine randomness (dice, irreducible). ("Compute deterministic / learn epistemic / distribution aleatoric.")

**Deep ensembles & diversity** — N nets from different inits; disagreement ≈ uncertainty. Diversity weakest→strongest: init < data < architecture < explicit repulsion. Your "physically different" instinct is right.

**Randomized Prior Functions / Bootstrapped DQN** — member = `f_θ(x)+β·p_i(x)`, `p_i` a *fixed random* prior, different per member (Osband). Disagree by construction in unvisited regions (= surprise). Cheap, principled. The "better than a naive ensemble" answer.

**Epinets** — a small `epinet` conditioned on index z approximates an ensemble's epistemic spread cheaply (Osband 2023).

**Disagreement exploration / curiosity / RND** — intrinsic reward ∝ ensemble disagreement (Pathak) or ∝ error vs a fixed random target (RND). Value-free exploration bonus.

**MC Dropout (weak here)** — dropout at test time ≈ variational Bayes (Gal); low functional diversity, noises the critic; shared trunk → inherits the blind trunk → falsely confident on defensive. "Diversity must reach the trunk."

---

## Part 5 — Game theory & imperfect information

**Simultaneous-move / matrix games / Nash / mixed strategies** — both move at once → often no single best move; the solution is a mixed strategy (RPS-like). A turn is a K×M payoff matrix; solve for mixed Nash (LP / regret matching).

**Luck floor / variance / skill-band compression** — high variance (dice + team draw) gives a weak player a ~10% win floor vs a strong one and a ~90% ceiling; skill moves you *within* the band. "Bots beat the generalist ~10%, we ~6%" = the luck floor, not skill.

**Imperfect info / belief states** — hidden team/moves → reason over a distribution; the sufficient statistic is the *belief*, not the state. Your belief heads are this.

**CFR / Deep CFR** — self-play + per-infoset counterfactual regret minimization → average strategy converges to ε-Nash in 2p zero-sum imperfect-info. Deep CFR generalizes with nets.

**ReBeL / Player of Games** — "AlphaZero for imperfect info": RL + search over *public belief states* → Nash. The target design for training-time search.

**PIMC / determinization** — sample hidden info, solve perfect-info, average. Cheap; flawed (strategy fusion, non-locality). Wang thesis used it for team completion.

**Value of Information (VoI)** — `E_o[max_a U(a,o)] - max_a E_o[U(a,o)]`. Falsify a "give the model X" feature by its VoI before building. Killed the opp-action head (~0.03 mon).

---

## Part 6 — Search & planning

**MCTS / PUCT / AlphaZero** — selectively grow a tree, guided by a net. PUCT `U=Q + c·P·√ΣN/(1+N)` balances exploit vs explore. AlphaZero trains on search-improved targets.

**MuZero vs AlphaZero** — MuZero *learns* the dynamics and plans in latent space (rules unknown/expensive); AlphaZero searches the *real* rules. With a fast exact sim → AlphaZero-with-a-real-model, not MuZero.

**Expert Iteration (ExIt) / search-as-teacher / distillation** — slow search makes better decisions → train the fast net to imitate → repeat. Your search-teacher / OPD machinery.

**Rollout-to-terminal vs value-bootstrap** — MC return: unbiased under the rollout policy, high-variance, policy-dependent. Bootstrap `V_θ`: low-variance, biased. Rollouts for *training targets*, bootstrap for *search leaves*. (The "sim is µs but policy is ms" correction.)

**CRN (common random numbers)** — replay counterfactual actions under the *same* dice → luck cancels in the difference. The prober's `falsify`/`lookahead`; the deterministic sim's superpower.

**AWR / KL distillation** — AWR: `-E[exp(A/β)·log π]` (advantage-weighted imitation). KL: `KL(π_target‖π_θ)` (match the whole distribution). Search-teacher uses AWR toward A*; OPD uses KL toward π'.

---

## Part 7 — Representation & interpretability

**Superposition & polysemanticity** — nets cram more features than neurons by overlapping directions → neurons fire for several things. Little superposition in a low-rank (3–5-dim) trunk → SAEs a poor fit.

**Sparse autoencoders (SAEs)** — learn an over-complete sparse dictionary `z=ReLU(W_e x)`, `x̂=W_d z`, L1 penalty, dim(z)≫dim(x); each active unit ideally one concept. Best where there's real superposition.

**Probing (linear / concept probes)** — train a tiny classifier on frozen activations to predict concept X; accuracy/AUC = decodability. Cheap, direct.

**Calibration vs discrimination (AUC / reliability)** — calibration: "70% means it happens 70%." Discrimination (AUC = P(score_win>score_loss)): "do higher predictions mean better outcomes." You can be calibrated but useless (predict the base rate → AUC 0.5). The critic is calibrated on defensive but AUC≈0.5 = predicts the mean because it can't read the position.

**Interpretability by construction** — design in interpretable variables as supervised heads (belief, win-prob, value-dist, damage) instead of reverse-engineering. Better fit than SAEs for hand-engineered obs.

**DIAYN / skill discovery / mutual information** — discover styles with no reward by maximizing `I(state;z)` between a latent skill z and behavior via a discriminator. Label-free style discovery.

**Attention interpretability** — in a transformer over named tokens (your mons), read which mon attends to which. Unusually legible; the cheap interp target.

---

## Part 8 — Architecture & conditioning

**Transformers / attention / tokens / CLS pooling** — tokens (mons + history + global) attend `softmax(QK^T/√d)V`; a learned CLS query pools to a fixed vector. Your ~20-token trunk.

**FiLM (Feature-wise Linear Modulation)** — a context (archetype) retunes features: `γ(c)⊙h+β(c)`. Cheap conditional computation → give the defensive "mode" its own effective params.

**MoE / routing / shared-expert isolation** — router gates over experts, top-k active; a shared always-on expert holds common structure so routed experts specialize (DeepSeek). Learned experts ≠ human categories.

**Multi-task / negative transfer / gradient interference** — joint objectives can conflict (negative gradient cosine on shared weights). The `--belief-grad-mode detached` rationale; the OPD interference fingerprint.

**Compute-vs-learn principle** — compute exact physics (differentiable damage op) rather than learning it; only learn the uncertain (belief) part. The DamageOperator + the "does it need to be differentiable" debate.

**Marginalization & Jensen** — average the *outcome* over the belief (`E[f(X)]=ΣP(x)f(x)`), not `f(E[X])`; for convex f (KO thresholds) mean-field is biased (Jensen). The nature/EV quadrature.

**Team/failure embeddings & KNN retrieval** — embed teams/failures so similar ones are near; retrieve k-NN of a problem to drill the family. Better: embed the *failure* `(state, mechanism)`, not the roster. Your "family of surprise" idea.

---

## Part 9 — Domain-specific (Pokémon RL)

**Belief / opponent modeling** — posterior over hidden team/moves/stats: revealed = certain, unrevealed = prior ⊕ learned. Consumed by the op/policy.

**Archetype / style competence gradient** — confident with offense, tentative with defensive/mixed, but confidence ≠ competence. Turn-1 P(win) offense>balance>defensive (confidence); realized wins inverted (floor-carried); AUC offense≫defensive (reads offense, blind to defensive).

**Offset vs interaction conditioning** — does the team shift everything (offset) or change how the board is weighed (interaction)? ~60% offset / 40% interaction for value.

---

## The one-paragraph synthesis

PPO maximizes **average return**; the **advantage** is the only learning signal. Under an offense-heavy,
floor-carried distribution the objective is satisfiable with a **low-rank board summary**, so — helped by
**simplicity bias** and **TD-bootstrapping rank collapse** — the critic stays blind to defensive positions
(**AUC≈0.5**) while 90% of its **capacity** sits idle. The archetype **floor zeros the advantage** on defensive
play, so the policy never learns a defensive game plan; the same floor makes "winning" mostly the **luck floor**,
and mixing in **unwinnable** games starves the ramp. The escape is not more capacity or more steps (same
**fixed point**) — it's changing the **objective**: restore the advantage with **contested opponents** (a
**learning-progress**-driven, **value-free** curriculum, since a value-driven one is **circular**), aim it with
**balanced/DRO** exposure, give it capacity to hold multiple plans via **FiLM/routing**, detect where to practice
with a **randomized-prior ensemble** (value-free uncertainty + exploration), retrieve the **family of surprise**
via a **failure latent + KNN**, and — the strongest break-in — inject **exogenous data (human replays)** that's
value-independent. Verify with **AUC-by-style** and **fraction-of-games-at-~50%**, not raw win-rate.
