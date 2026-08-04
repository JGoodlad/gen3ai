# Imperfect-Information Search, PBS, and Value Functions — A Study Guide

An offline-study companion covering the whole thread on value functions, Public Belief States (PBS),
Counterfactual Regret Minimization (CFR), depth-limited search, and how they connect to breaking the
gen3ai plateau. Written to be read on its own. Sibling to `rl_concepts_studyguide.md` (the broad glossary);
this one is the focused narrative on **why the value function is the hub and how imperfect-information
search improves it.**

---

## 0. The organizing thesis (read this first)

Everything below hangs on one claim: **the value function is the limiting factor, because it determines the
advantage, and the advantage is the only learning signal.**

The chain:
```
better V  →  better advantage A = Q − V  →  better policy gradient  →  better policy
        →  better data (states visited)  →  better V targets  →  better V   (loop)
```

Two refinements that make this precise:
- **V and advantage are *coupled*, not sequential.** V is blind on defensive positions *because* the floor
  zeroed the advantage (no gradient trained it); the advantage is ~zero there *because* V is blind (can't
  discriminate → `Q(s,a) ≈ V(s)`). You can't fix one from inside the loop — you must **inject value signal
  from outside** (better inputs, better targets, better capacity, exogenous data) to start it turning.
- **Discrimination, not calibration, is the currency** (§2). A V that predicts the *mean* everywhere is
  calibrated and useless — it produces zero advantage.

The behavioral symptom, in one line: **"trivially okay in matchups, not great at games"** = V reads the
*local/immediate* board (offense AUC ≈ 0.63) but is *blind to long-horizon/positional* value (defensive
AUC ≈ 0.50). Local readability → sane trades. Positional blindness → no multi-turn game plan → generic
reactive play. **Give V positional discrimination → positional advantage appears → the policy learns to
plan.** That is the whole project, viewed through the value function.

---

## 0.5 Why our model doesn't reason in discrete space (amortization vs deliberation)

*Added after an owner observation: "our model doesn't want to reason in discrete space — humans
don't marginalize, they think about specific moves and simulate how it would go. It seems to use
its capacity to make heuristics it likes." That observation arrives, from architecture intuition,
at exactly where the L1–L4 falsification tree arrived from measurement. This section states the
mechanism, because it is the motivation for everything below.*

### Every gradient we give it is smooth

It is not a preference the model developed — it is the only thing we have ever trained. Walk the
pipeline looking for one term that rewards having simulated a **specific discrete line**: the move
belief is a distribution; the `DamageOperator` **marginalizes** over candidates; the spread belief
marginalizes over natures/EVs; `V(s)` is an expectation; PPO's advantage is an expectation. There
is no such term anywhere. We built a sophisticated apparatus for computing expectations and then
observe that the model reasons in expectations.

### The formal shape: search is marginalization at depth

| Regime | Procedure |
|---|---|
| **Mean-field** | collapse the belief → evaluate once |
| **Marginalize** | branch over *worlds* → evaluate each → collapse the results |
| **Search** | branch over *actions and worlds* → evaluate each → collapse → **recurse** |

We do depth-0 marginalization well. "Simulating specific moves" is depth ≥ 1 with **discrete
branching at the nodes**. Note this is not an argument against marginalizing — proper planning
marginalizes at the **leaves**; the defect is marginalizing **early**, before branching.

### The inversion worth carrying

**Humans don't marginalize because they can't.** Nobody computes an expectation over ~400
candidate movesets; a human picks the two or three most plausible and simulates those to a
conclusion. That is **few-sample-deep** versus our **full-marginal-shallow**. As an *estimator*
the human method is worse (high variance, biased by which lines came to mind). As a *decision
procedure* it can be much better **in this game specifically, because the deciding quantities are
thresholds** — KO/not, outspeed/not, status lands/not. Averaging is precisely the operation that
destroys threshold structure (the Jensen argument of
[[marginalization_and_uncertainty]], applied one level up): a discrete simulation preserves *"in
this line I am dead"*; an expectation smears it into *"moderately threatened."* Under a fixed
compute budget these are different points on a tradeoff, and the thresholded payoff structure
argues for the human allocation.

### Why a fixed-depth network structurally cannot do it

Simulation is **sequential composition** — apply transition, re-evaluate, repeat — and a
fixed-depth forward pass composes a bounded number of times, each step learned as its own slab of
computation. An unbounded search does not amortize into a fixed forward pass in general
(fixed-depth bounded-precision nets sit in a bounded complexity class; tree search does not).
AlphaZero is the honest acknowledgment: **the network is the amortized part, MCTS is the
deliberation, and the network is trained on the deliberation's output** (expert iteration). The
network never learns to search — it learns to predict what searching *would have concluded*.
**Without a search operator somewhere in the loop, nothing generates the deliberate answers to
amortize**; the net can only compile the average of what its own shallow policy already did.

### "It spends capacity on heuristics" — the correction

The evidence says it is **not spending capacity on heuristics; it is declining to spend capacity
at all.** `rank/trunk_pr` is **24 → 35 of 128 and rising**; eight offline probes concluded **"no
obvious representational hole"**; the critic's rank ~3 is *appropriate* (one scalar out;
`value_pooled` AUC 0.833 vs the policy's 384 dims at 0.835). The capacity is idle because
heuristics already minimize the loss it is given — gradient starvation
([[shortcut_learning_and_feature_delivery]]) one level up.

**The RL-specific amplifier, and it is nastier than the supervised version:** the policy **shapes
its own state distribution**. It can avoid entering positions its heuristic cannot evaluate, and
then never receives the gradient that would expose the failure. A supervised learner is stuck with
its dataset; a policy can quietly stop visiting its own counterexamples. That is a self-confirming
loop and a plausible mechanism for a **flat** fixed point rather than a slow one
([[project_plateau_research_2026_06_25]]).

### The pattern across every null we have: information, never computation

| Lever | Result | Varied |
|---|---|---|
| L1 obs / accumulation channels | ❌ all AUC ~0.5 | information |
| L2 critic calibration | ❌ calibrated once the eval-quota confound was defeated | information |
| L3 opponent-action ORACLE | ❌ VoI ~0.03 mon, p=0.53, zero win/loss skew | information |
| K9/K10 physics into the trunk | ❌ null 3-for-3 | information *routing* |
| pubval (v43) | ❌ | *target* |
| belief heads | collapsed to chance in the best-ever run | information |

**Every one varies what the model is TOLD. Not one varies how much it COMPUTES per decision.**
And the closed tree's residual, L4, is named in our own record as *"genuine multi-turn play
strength."*

**Information nulls do not refute the computation lever.** A chess engine has *perfect
information* and still gains on the order of 1000 Elo from search. "Knowing the reply" and "being
able to roll out N plies" are unrelated quantities; the VoI oracle measured the first at ~0.03
mon and is silent on the second. (The L3 memory flags this itself — "a one-turn re-roll can't
fully exclude arbitrary-depth anticipation" — and guards it with the zero win/loss skew. That
guard is real evidence against **one-ply** mattering, and no evidence about depth.)

### The lever that fits the no-runtime-search constraint — already built, never run

On-model search is owner-ruled-out. That rules out search **at inference**, not search **in the
loop**: `better-line` (CRN-anchored beam over the critic, clone-and-branch search server),
`--search-teacher` (ExIt/AWR), and `--opd-coef` (full-distribution on-policy distillation) all
exist. Search runs at *training* time as a supervision oracle; the deployed model stays one
forward pass. That is the AlphaZero split exactly, and it is **the named survivor in the closed
L1–L4 tree** ("an offline search-teacher distilled into the net"). See
[[on_policy_self_distillation]] for the bits-per-decision argument and the minimal first
experiment. Note also that **v51 already delivers depth-0 discrete structure** — every action has
its own seat and its own consequence cells; the gap is *depth*, not *discreteness*.

### Honest counterweights (do not skip these)

1. **~30% of losses are attributable LUCK.** High-variance stochastic games reward depth *less* —
   variance swamps the depth advantage. Pokémon is not chess here, and the deliberation ceiling is
   lower than the chess analogy implies.
2. **~64% is UNATTRIBUTED = UNKNOWN**, and only **~6% is proven policy-reducible**. Reasoning
   about deliberation headroom is reasoning into a measurement gap. (Never re-inflate this into a
   "team-draw / unwinnable" claim — [[feedback_no_circular_unwinnable_claims]].)
3. **Wang 2024 is the counter-counterweight** — Gen 4 PPO + MCTS, 78.6% → 90.8% vs Heuristic: the
   closest direct evidence that lookahead pays *in Pokémon specifically*, and it is large.

**The cheap decisive measurement, before committing to a search-teacher run:** tighten the
`unattributed` bucket with `falsify-scan` + `calibration` (which splits it into
`critic_overvalued` vs `lost_position`), then run `better-line` offline on the worst decisions and
ask *"does depth-3 search find a materially better line here?"* Zero training cost, existing
tooling. **No better line at our worst decisions ⇒ the discrete-reasoning thesis is falsified
cheaply. Better lines ⇒ the teacher targets are already in hand.**

---

## 1. The advantage-signal principle (the foundation)

**The policy gradient.** Maximize `J(θ)=E_τ[Σ γ^t r_t]`. The score-function identity gives
`∇J = E[Σ_t ∇log π(a_t|s_t) · R(τ)]`. Two *exact* (unbiased) refinements clean it up:
1. **Reward-to-go:** replace `R(τ)` with `G_t = Σ_{k≥t} γ^{k−t} r_k` (an action can't affect the past).
2. **Baseline:** for any `b(s)`, `E_a[∇log π(a|s)·b(s)] = 0`. The variance-minimizing baseline is `V(s)`,
   giving `G_t − V(s_t)`, a one-sample estimate of the **advantage** `A(s,a) = Q(s,a) − V(s)`.

So the learning signal is **relative**: not "was the outcome good" but "was this action better than the
alternatives from here."

**The core principle.** If the outcome distribution doesn't depend on the action at `s`, then
`Q(s,a) = V(s)` for all `a`, so `A ≡ 0`, so the per-state gradient is 0. **No advantage → no learning**,
regardless of learning rate or visit count. This has two faces:
- **Winning floor:** win regardless (stall vs a weak bot) → `A ≈ 0`.
- **Losing floor:** lose regardless (from-scratch vs a 1998-ELO opponent) → `A ≈ 0`.

**Advantage variance ∝ p(1−p).** Treat the terminal reward as Bernoulli(`p`). Its variance is `p(1−p)`,
maximal at `p = 0.5`. Since the advantage derives from the outcome, its **signal-to-noise peaks where you
win ~half the time** — the "contested frontier." This is the mathematical statement of curriculum:
sample where `p ≈ 0.5`, not where difficulty is highest.

**GAE and why V quality gates learning.** In practice you estimate `A` with the critic and bootstrap:
`Â_t = Σ_l (γλ)^l δ_{t+l}`, `δ_t = r_t + γV(s_{t+1}) − V(s_t)`. `λ=0` fully trusts the critic (low variance,
high bias); `λ=1` is Monte-Carlo (unbiased, high variance). Key consequence: **if V is uninformative in a
region, `A = Q − V` inherits that flatness** — a critic that predicts the mean for every defensive position
produces `A ≈ 0` there *even when the position is decisive*, silently switching off the policy gradient
exactly where it's blind.

**Why reward shaping can't rescue it.** Potential-based shaping `F(s,s') = γΦ(s') − Φ(s)` is
**advantage-invariant** (`A_shaped = A_base`): it speeds value-*learning* but cannot change which actions
have positive advantage → cannot change behavior. The only way to create advantage is to make the outcome
genuinely depend on the action — i.e., **contested opponents**, not reward tweaks.

**The five symptoms, one cause:** defensive blindness, the ~10% luck floor, the slow ramp when mixing in
unwinnable games, the S-curve takeoff, and the curriculum target are all `A ≈ 0 ⇔ no learning`, with
`Var(A) ∝ p(1−p)`.

---

## 2. Calibration vs discrimination (the value function's two properties)

A probabilistic prediction has **two orthogonal virtues:**
- **Calibration:** "when you say 70%, does it happen 70%?" — `P(Y=1 | p̂=q) = q`. The reliability diagram.
- **Discrimination:** can you *rank*? — measured by **AUC** `= P(p̂(X⁺) > p̂(X⁻))`. 0.5 = no ranking, 1.0 =
  perfect. AUC depends only on the *order* of scores (invariant to any monotonic transform).

The killer example — **our critic:** predict the base rate `p̄` for *every* input. Then it's perfectly
calibrated (`P(Y=1|p̂=p̄)=p̄`) yet **AUC = 0.5** — zero discrimination. It has learned the *average* defensive
win-rate and *nothing about which specific positions win*.

**The asymmetry:** calibration is *fixable post-hoc* (Platt/isotonic recalibration — a monotonic map, leaves
AUC unchanged); **discrimination cannot be recalibrated in** — it's an *information* property, obtainable only
by learning better features. So AUC measures *whether the model has the information*; calibration measures
*whether it's reported at the right scale*.

**Murphy decomposition:** `Brier = Reliability − Resolution + Uncertainty` (calibration error − discrimination
+ irreducible base-rate variance `p̄(1−p̄)`). The predict-the-mean critic has zero reliability error and zero
resolution — least effort that stays calibrated.

**Why AUC was our robust metric:** the eval traces were loss-enriched (base-rate distorted), so *level* metrics
(win-rate, calibration, Brier) are biased — but AUC is *rank-based*, invariant to that reweighting.

**The bridge to §1:** the advantage `A = Q − V` is meaningful only if V **varies across states in a way that
tracks real outcome differences** — i.e., only if V **discriminates**. A calibrated-but-non-discriminating V
gives `V ≈ const → A ≈ 0`. **Discrimination is what manufactures advantage.** Hence the acceptance metric for
the whole program is **AUC-by-style**, not win-rate (floor-confounded) and not calibration.

---

## 3. Imperfect information: why the state isn't a sufficient statistic

In perfect-information games (chess/Go) the **state** is a sufficient statistic — a position's value depends
only on the position. Imperfect information breaks this:

- **Information sets (infosets):** histories a player can't distinguish are grouped; the player must play one
  strategy across all of them (you don't know the opponent's hidden team → many true states look identical).
- **The value depends on *beliefs*, which depend on the *strategy*.** How you'd play with a strong hidden team
  affects what the opponent infers, which affects the value of your situation. Nodes are *coupled through
  beliefs* — you can't evaluate one in isolation.
- Two classic failures of naive approaches: **strategy fusion** (pretending you'll "know" your hidden info
  later — the PIMC flaw) and the need for **balanced mixed strategies** (bluff/protect across hidden states).

The sufficient statistic is not the state but the **belief** — a distribution over hidden variables.

---

## 4. Public Belief States (PBS)

**The move:** change what the "state" is. Everything public is common knowledge (the battle log). Given the
public history and a fixed strategy, both players can compute — as **common knowledge** — the distribution
over each player's private info.

> A **Public Belief State (PBS)** = public history + the common-knowledge joint distribution over each
> player's private information.

**Why it's magic:** the PBS *is* a sufficient statistic, and over PBSs the game is *perfect-information*
(both know the PBS). So AlphaZero's machinery — value/policy net + search + self-play bootstrapping — applies,
with PBSs in the role of states. **AlphaZero is the special case where every private distribution is a point
mass.** ReBeL strictly generalizes it.

**The price:** a PBS lives in a continuous high-dimensional space (it's a distribution), and its value is a
**vector** — a value per infoset (per hidden config), not a scalar. `V(PBS)` maps a belief to a vector of
infoset-values.

**The Pokémon belief-collapse gift:** unlike poker (hand hidden to the end), Pokémon *reveals* private info as
the game proceeds (moves as used, mons as switched in). So by mid-game the belief is near a point mass → the
PBS is nearly perfect-information → search is *cheap and sound exactly where games are decided.* You pay full
belief cost mostly in the opening. Gen 3 also has **no team preview**, so the belief starts as a clean usage
prior and monotonically narrows.

---

## 5. CFR — the engine

**The atom: regret matching on a one-shot game.** Play repeatedly; define **regret** for action `a` =
how much better you'd have done always playing `a`: `r^t(a) = u(a,·) − u(σ^t,·)`; accumulate `R^T(a)=Σ r^t(a)`;
set next strategy proportional to *positive* accumulated regret:
`σ^{T+1}(a) = R^{T,+}(a) / Σ_b R^{T,+}(b)` (uniform if none positive), `R^+ = max(R,0)`.

Two theorems:
- **(Blackwell / Hart–Mas-Colell)** average regret → 0 at `O(1/√T)`.
- **(Zero-sum folk theorem)** if *both* players' average regret → 0, their **time-averaged strategies
  converge to a Nash equilibrium**.

Burn in: **the *current* strategy oscillates; the *average* strategy converges.** Accumulate `σ̄`.

**Lifting to game trees: the counterfactual trick.** Split the reach probability of a history:
`π^σ(h) = π^σ_i(h) · π^σ_{-i}(h)` (your own contribution × chance+opponents). Define the **counterfactual value**
of infoset `I`, weighting by the *opponents'/chance's* reach (as if you'd steered toward `I`):
`v_i(σ,I) = Σ_{h∈I} π^σ_{-i}(h) Σ_z π^σ(h→z) u_i(z)`. **Why counterfactual reach?** An infoset you currently
avoid would otherwise get negligible regret updates; weighting by the counterfactual reach gives every infoset
credit *as if you'd tried to reach it*, so the per-infoset regrets bound the global regret. Then run regret
matching per infoset on `r^t(I,a) = v_i(σ^t, I→a) − v_i(σ^t, I)`. In 2p0s the reach-weighted average strategy
→ Nash.

**Variants (vanilla is intractable):**
- **MCCFR** — sample trajectories for unbiased regret estimates (rollout instead of full-tree sweep).
- **CFR+** — floor regrets each step + linear averaging; ~an order of magnitude faster (solved HU limit poker).
- **Deep CFR** — replace the regret/strategy *tables* with *nets* that generalize across infosets. The direct
  precursor to ReBeL (which adds search + value bootstrapping on public belief states).

Mental model: **regret matching : matrix game :: CFR : game tree :: Deep CFR : too-big tree :: ReBeL : Deep CFR
+ search + value net.**

---

## 6. Worked example — solving one simultaneous-move node

A Pokémon 50/50. My mon can **Stay** (attack) or **Switch** (pivot); opp can **Attack** or **Coverage**.
My payoff matrix (zero-sum; opp gets `−A`):

```
          Opp Attack   Opp Coverage
Stay          0            +3
Switch       +4           -1
```

No row/column dominates → no pure equilibrium (a genuine mixup). **Analytic Nash** (via indifference):
Me (Stay 0.625, Switch 0.375), Opp (Attack 0.5, Coverage 0.5), value +1.5. Now regret matching finds it:

Both start uniform. Per iteration: compute each action's value vs the opponent's current strategy; regret =
value(a) − value(current mix); accumulate; next strategy ∝ positive cumulative regret; accumulate for the average.

- **Iter 1** (both 0.5/0.5): my V(Stay)=V(Switch)=1.5 → regrets 0 → stay uniform. Opp regrets (−0.5,+0.5) →
  cumulative (−0.5,+0.5) → next Opp **(0,1)** (pure Coverage).
- **Iter 2** (Me 0.5/0.5, Opp 0/1): my V(Stay)=3, V(Switch)=−1, mix 1 → regrets (+2,−2) → cumulative (2,−2) →
  next Me **(1,0)**. Opp cumulative (−1.5,+0.5) → **(0,1)**.
- **Iter 3** (Me 1/0, Opp 0/1): my regrets (0,−4) → Me stays **(1,0)**. Opp vs Me-Stay: values (0,−3), regrets
  (+3,0) → cumulative (1.5,0.5) → next Opp **(0.75,0.25)** — starts mixing.

Running **average** strategies: after iter 3, Me ≈ (0.67, 0.33) — already near Nash 0.625/0.375; Opp lags at
(0.17, 0.83) and climbs toward 0.5/0.5 with more iterations. **The two phenomena:** current iterates bounce
between pure strategies; the *time-average* converges to Nash (at `O(1/√T)` — hence CFR+). This regret-matching
solve is what runs **at every simultaneous-move node** of the search.

---

## 7. Depth-limited search + the gadget (value-consistency)

**Why perfect-info search is easy and imperfect-info search is hard.** In chess, a leaf value is a
self-contained number: the opponent's best play below it is baked in. In imperfect info, **a player's optimal
subgame strategy depends on their strategy *outside* it, through the beliefs.** If you fix a leaf value and
solve the subgame above it, a *real* opponent will best-respond *below* the leaf and *choose which hidden hands
to route toward* it — so a fixed value assumes the opponent is "frozen" into equilibrium continuation. It isn't.
Your solve underestimates the opponent → the strategy is **exploitable.**

Two consequences: **(1)** the leaf value must be a **vector** (per opponent infoset), not a scalar averaged over
the belief; **(2)** the solve must be robust to the opponent best-responding below the leaf.

**The gadget game.** To re-solve a subgame *safely*, augment it: at the root, give the **opponent** a choice —
for each of their boundary infosets, **enter** the subgame or **opt out** for a fixed payoff equal to their
**counterfactual value** (what they were guaranteed by the surrounding solution). Those opt-out numbers are the
per-infoset leaf values (from the value net). Solving this gadget with CFR forces the subgame strategy to
concede the opponent **no more than their guarantee anywhere** — exactly the condition to splice into a full
equilibrium **without increasing exploitability.**

Analogy: a negotiation where you must guarantee the counterparty at least their **BATNA** at *every* information
state; the gadget hard-codes each BATNA as a take-it-or-leave-it outside option, so your strategy can't rely on
them "playing along" below the horizon.

**"Value consistency"** = the values you *use* at the leaves equal what the opponent can *actually achieve*
there against your solution. The gadget restores it by making leaf values the opponent's *optimizable outside
options* rather than passive constants. This is what makes a *local* solve (one subgame + a value net) yield
*globally* sound play — the imperfect-info analog of "trust the value net at the leaf" (DeepStack's continual
re-solving; carried into ReBeL). It's also the exact line between **PIMC** (scalar per-determinization leaf
values, no gadget → strategy-fused, exploitable) and **sound search**.

---

## 8. ReBeL (Recursive Belief-based Learning)

The AlphaZero loop, transplanted to PBSs:
1. At the current PBS, run a **depth-limited subgame search** whose local solver is **CFR** (find a *mixed*
   Nash over coupled infosets), not minimax.
2. At the leaves, **bootstrap with the learned `V(PBS)`** (vector) — made sound by the **gadget**.
3. Search outputs an **improved mixed policy** + **improved infoset values**.
4. **Train** the value net toward those values; **distill** the policy toward the search strategy.
5. **Step** the game; update the PBS by **Bayes** on the public action; recurse (hence *recursive*).

In self-play this converges to approximate **Nash** — the correct solution concept for a simultaneous-move
imperfect-info game — rather than the *average-return fixed point* that PPO alone converges to (the archetype-
blind, exploitable policy). **What maps to Pokémon:** your belief heads = (part of) the PBS; your critic +
win-prob head = the `V`; the Rust sim = affordable search; your search-teacher/OPD = the distillation loop.
**What's hard:** simultaneous moves (each node's local solve is the §6 matrix game), the huge private space
(needs abstraction/sampling), and the expensive value forward (needs a distilled light leaf net).

---

## 9. Search & value: rollout vs bootstrap, CRN, and the soundness rule

**Rollout-to-terminal vs value-bootstrap.** A Monte-Carlo rollout is unbiased *under the rollout policy* but
high-variance and *only as good as that policy*; a value-net bootstrap is low-variance but biased. Crucially the
"sim is microseconds" is misleading: a *meaningful* rollout step = sim + a *policy decision* (~ms for the net),
so rollouts are **policy-bound, not sim-bound**. Use rollouts for **training targets** (unbiased), bootstrap for
**search leaves**. Near-terminal (short games, exact reward, less-reliable critic) favors more rollout.

**CRN (common random numbers):** replay counterfactual actions under the *same* dice → luck cancels in the
*difference* → sharply lower comparison variance. The deterministic sim's superpower for counterfactuals.

**The soundness rule — three different V's:**
- **`V^π`** (PPO critic): value of the *current* policy. Trained toward observed returns. The correct GAE
  **baseline and bootstrap**.
- **`V^{π*}` / search value:** value under *equilibrium/optimal* play. Used as the *search-leaf bootstrap* and
  a *distillation target*.
- **`V^{human}`** (§10): value under *human* play.

**Do not substitute the search or human value into GAE.** As a pure baseline (`λ=1`) any state-function is
unbiased, but GAE at `λ<1` uses V as a **bootstrap** (the `γV(s')` term estimates the *continuation*), and that
must be the *current-policy* continuation — plugging in `V^{π*}` or `V^{human}` **biases the advantage**. So:
keep the on-policy `V^π` for the advantage; use the other values as **auxiliary targets / policy distillation**,
never as the GAE bootstrap. (This is the "value-only unsound" note — search V biases GAE.) **Two values, two
roles.**

---

## 10. PBS *without* the belief: the public-information value function

**The idea.** Strip the explicit belief distribution out of the PBS and keep a value on **public information
only**: `V(public) = E_{hidden ~ P(hidden|public)}[V(full)]` — the **belief-marginalized value**, with the
belief absorbed *implicitly into the weights*, learned from data. A scalar, not the per-infoset vector.

**Why it's the cleanest exogenous break-in.** The public log + outcome is *exactly what a human replay
contains* → you can train `V(public-state-at-turn-t) → who won` by pure **supervised regression** over millions
of human games. That makes it:
1. **Value-*independent* of the RL loop** (supervised on real outcomes, not your own bootstrap) → it injects
   value signal from *outside* the circular "blind V ⟷ zero advantage" trap. The thing that starts the loop.
2. **A teacher of *positional* discrimination** — human games are decided by multi-turn positional play, and the
   outcome labels reward exactly that → it learns to tell a winning defensive position from a losing one (the
   AUC-0.50 gap; the "not great at games" fix).
3. **Universal / transferable** — the public board (revealed mons, HP, hazards, weather, revealed moves) is a
   *common language* across all teams/matchups, unlike a team-specific policy.
4. **The concrete Metamon lever** — offline learning on human replays broke *this exact plateau* (arXiv
   2504.04395); the public-info value head is that result, instantiated.

**How to use it (clean + sound):** add it as an **auxiliary head on the *shared trunk***, supervised on
replays. To reduce that loss the trunk must learn *positional-value features*; the RL critic reads the same
enriched trunk → its `V^π` head now has discriminative features → non-zero advantage on positional decisions →
the policy learns game plans. **Leak-free and unbiased:** the RL value head still learns `V^π` (only the shared
representation is improved), so the advantage stays on-policy. Do **not** wire `V^{human}` into GAE (§9).

**The two honest limits:** (1) it's a **scalar** (belief-marginalized), not the per-infoset **vector**, so it
**cannot power gadget-sound search alone** — it's a value *signal*, not a PBS substitute *inside* search;
(2) it's `V^{human}`, not `V^π`, so it's a representation-shaper/target, not the live baseline. Quality is
bounded by replay strength (gen3ou ladder median ~1242 → rating-filter hard).

**Logistics — an early win, offline, credits-friendly:** parse the `/replays` corpus (137k gen3ou), rating-
filter, extract `(public_state, turn) → outcome`, train a supervised win-prob net on the *public/revealed*
obs subset (reuses the existing pipeline). No Rust sim, no live-training contention, embarrassingly parallel.
Doubles as a universal position evaluator for the prober/curriculum.

---

## 11. Improving the value function — the early-wins ladder

Every topic, re-sorted by its *value-function* lever, cheap-first:

**Tier A — cheap, current architecture:**
1. **Belief-condition the critic** — route the existing belief heads into `V` → discriminative features for
   hidden-info positions → lower advantage variance. *Better inputs.*
2. **Enable the distributional value aux loss** (the v29 head exists in `read_only`) — a richer target than a
   scalar; a known cure for value-learning instability/rank collapse. *Better target + anti-collapse.*
3. **Plasticity hygiene** — LayerNorm + weight decay (± light resets) → fights TD-bootstrapping rank collapse →
   lets V recruit the dimensions it needs. *Better capacity use.*
4. **AUC-by-style** — the ruler (discrimination = what makes advantage). Drive everything by watching defensive
   AUC 0.50 → 0.65.

**Tier B — the outside signal (the real break-in):**
5. **Public-info human-replay value head** (§10) — the value-independent, positional, universal aux target.
   Offline. The one to build first.
6. **PBS-lite / selective 1-ply value targets** — sample K opponent configs from the belief, evaluate, aggregate
   → a belief-marginalized value target; or use the prober's `lookahead` search-improved `V(s')` on crater
   moments. Injects positional discrimination the sparse terminal reward can't. Seed of the search-teacher.

**Tier C — the ceiling-raisers (regime change):**
7. **Archetype conditioning (FiLM) on the critic** — lets V compute a *different* value per style (a defensive-V
   that isn't overwritten by the offense-V). Needs balanced data under it.
8. **Contested curriculum + DRO** — restores the advantage gradient the RL-native way (§1). Value-free driver
   (learning progress), since a value-driven curriculum is circular.
9. **Full search-value distillation** (post-Rust-sim, at scale).

**Soundness rule throughout (§9):** search/human values shape the *representation and policy*; the on-policy
`V^π` stays the advantage baseline.

---

## 12. Feasibility / compute (what's single-machine tractable)

The decisive distinction: **inference-time search** (search only when playing, on a fixed net) vs
**training-time search** (search generates targets, distill, repeat). They differ ~2 orders of magnitude.

- **Tier 0 — inference-time search:** *demonstrably* single-machine (Wang thesis: Pokémon MCTS, ~10 s/decision,
  one workstation; DeepStack: gadget re-solving on one GPU, ~3 s/decision). A free strength boost for
  eval/analysis, no training change.
- **Tier 1 — selective search-as-teacher (distillation):** tractable on one machine — search only the
  crater/high-regret decisions, distill. Your existing scaffolding + Rust sim + a distilled leaf net. The
  realistic route to a stronger *trained* net. (Searching *every* decision is 1–2 orders slower than PPO;
  searching ~1% and distilling is a small multiplier.)
- **Tier 2 — full ReBeL from scratch:** cluster territory (ReBeL used tens–hundreds of GPUs; Libratus a
  supercomputer). On a high-powered *single* box (8 GPUs) a warm-started, selective, depth-limited *refinement*
  is feasible in weeks; from-scratch full-game PBS is gated by **belief engineering**, not FLOPs.

**Bottlenecks are the value-net leaf evals + the sim + the belief representation — not the CFR arithmetic.** The
ai_v8 plan (Rust sim, distilled light leaf net, belief-collapse late-game) attacks exactly these. **Compute
caveat:** the plateau is *regime-bound, not compute-bound* — more of the same recipe reaches the same fixed
point faster, not higher. Spend compute on *algorithm experiments in parallel*, not on scaling one run.

---

## 13. Synthesis + ordered plan

**The value function is the hub; discrimination is its currency; almost everything above is, from the right
angle, a way to feed V better inputs, better targets, or better capacity so its advantage stops being zero
exactly where the interesting games are decided.**

- The **advantage-signal principle** (§1) says learning happens only where the outcome depends on the action,
  peaking at `p≈0.5`.
- **Discrimination** (§2) is what makes V's advantage non-zero; **AUC-by-style** is the ruler.
- **Imperfect info / PBS** (§3–4) says the true sufficient statistic is the belief; the value is a vector.
- **CFR + the gadget** (§5–7) is how you *solve* a belief-conditioned subgame *soundly*; **ReBeL** (§8) is the
  full learning loop that targets **Nash** instead of PPO's flat average-return fixed point.
- **The soundness rule** (§9): keep `V^π` for the advantage; use search/human values as targets, not the GAE
  bootstrap.
- **The public-info value function** (§10) is the *cheap, exogenous, universal* slice of PBS — trainable on
  human replays — that breaks the bootstrap and teaches positional discrimination.

**Ordered plan (mostly pre-Rust-sim):**
1. **Public-info human-replay value head** (§10) + **belief-condition the critic** + **distributional aux +
   LayerNorm** (§11 Tier A/B) — one architecture pass; re-run watching **defensive AUC**. The "improve V" bundle,
   mostly parts you already own.
2. **PBS-lite / selective search value targets** — inject positional discrimination on crater moments.
3. **Then** the regime changes (archetype-FiLM-critic, contested curriculum, human-replay offline RL) scale it;
   the Rust sim + credits make the search-value version cheap; full ReBeL is the eventual capstone.

Fix V's positional discrimination and the "trivially okay in matchups, not great at games" ceiling is what breaks.
