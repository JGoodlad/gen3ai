# Design — the Q head, the opponent pointer, and fixed-mass belief tokens

**Status: DESIGN (owner-directed, 2026-09-27).** The one built piece is X4a's DETACHED A and B
heads, which ride along in the X26 baseline (`gen3_ridealong_heads_v1`, config v126, 2026-09-30;
§5.4). The joint Q head, α-as-flat-pointer and the fixed-mass tokens are not built. Experiment rows X3–X6 and X11 in
[`../research_state/EXPERIMENT_BACKLOG.md`](../research_state/EXPERIMENT_BACKLOG.md); build rows T2–T4
in [`../ops/TASK_BACKLOG.md`](../ops/TASK_BACKLOG.md). **Updated 2026-09-30 after the X4 pre-read**
(`a831190b`, `54fd5a6b`; [`../research_state/measurements/x4_preread/READOUT.md`](../research_state/measurements/x4_preread/READOUT.md)).
**Owner, 2026-09-30: only additions of AT MOST ONE PLY that can be subsampled are in scope now.**
One-ply V-bootstrapped Q̂ is kept for its measured uses only: an inference-time one-ply lookahead,
and diagnostics. Monte-Carlo playout labels (the grounded option) and X6 are DEFERRED, and X4 moves
down (§5.0, §9). The cost model is re-measured (§8.1), and the turn mechanics are corrected at the
source (§5.1a).

**Deletion pass L4 (2026-10-02):** the extractor-built per-action win-probability head of config v107
(`q_winprob_mode`) and the counterfactual-label training that fed it were DELETED, so nothing in the tree
trains a joint Q head today; `QWinProbHead` survives only as the shared-scorer class the detached
ride-along A head is built from (§5.4). A joint Q head here is a NEW build.

🚨 **ALWAYS-CURRENT WHILE BEING IMPLEMENTED (owner, 2026-09-27).** This doc is the spec the builds
follow. Any implementation that differs from it UPDATES this doc in the same commit, stating what
changed and why. [`../ARCHITECTURE.md`](../ARCHITECTURE.md) remains the doc of record for what the
model IS once a piece lands; this doc is the doc of record for what it is MEANT to become.

---

## 0. Why

The research bottleneck is the **leaf**: six win-probability heads sat on the null as search leaves
(mirror win rate 0.485–0.507, UNDERSTANDING §4.x). They are calibrated BETWEEN games and blurry WITHIN
one — and search compares siblings within one game. The whole design separates **calibration** (who is
winning) from **discrimination** (which move is better) and trains each on the comparison where its
signal is largest and everyone else's noise cancels. The three-axis variance measurement motivates it:
the opponent's choice explains ~60% of next-state variance, dice ~27%, our own choice ~10%.

Principles (owner): **discrete-first** (concrete, game-grounded entities; one OTHER entity carries the
marginal of everything unnamed); **measure before adopting**; **pool memorization is a success
milestone** — on-pool metrics first, off-pool second.

---

## 1. The board — fixed-mass belief tokens (T0)

**X5 STANDS ALONE (orchestrator, 2026-09-30).** This section was specified bundled with the Q head
(X4). X4 is now deferred (§5.0, §9). The tokens add no ply, since they are an architecture change
over the same observation, so they stay in scope under the owner's one-ply rule and land on their own
at the North Star 1 retrain boundary. They are judged on intent prediction, OTHER rates, the on-pool
belief metrics (X8) and a strength guard. Their value to the Q interaction term (I) is read later, if
X4 returns.

**Build spec: [`design_x5_belief_tokens.md`](design_x5_belief_tokens.md) (PROPOSAL, 2026-10-03, for owner review).** It refines this section: presence by iterative capped πps (Σ = k exactly) instead of `min(1, k·q)`, and the production sequence is 61 tokens, not 29, so X5 takes it to 62 (OTHER_move re-uses the active's E5 seat). Its §3.9 lists every departure and why.

Two groups of concrete entity tokens replace today's blob belief slots. **Token budgets are fixed;
each group's probability MASS equals the real count, OTHER included.**

| group | tokens | mass | revealed token weight | hypothesis weight | OTHER |
|---|---|---|---|---|---|
| **their team** | 6 (revealed + species hypotheses) **+ 1 OTHER_species** = 7 | **6** | 1 | presence probability | leftover mass |
| **their active's moves** | 6 seats (revealed + move hypotheses) **+ 1 OTHER_move** = 7 | **4** | 1 (pinned) | presence probability | leftover mass |

Future: the team budget may grow to 7–8 hypothesis tokens with the mass still 6 (more alternatives,
same meaning).

**Mass by construction.** Per unseen slot, one categorical `q(species)` = softmax(Smogon prior
conditioned on revealed teammates ⊕ a learned delta), revealed species excluded (Species Clause).
Presence of species s among the k unseen = `min(1, k·q(s))`. The (6 − r) hypothesis tokens are the
top species by presence; **OTHER_species = k − Σ hypothesis weights**. The group sums to 6 exactly,
whatever the belief's errors. Moves are the same with E10's prior ⊕ a learned delta and mass 4 (the
existing E4 seats already work this way: revealed moves pinned at 1). **Never renormalize the
hypotheses to absorb OTHER** — OTHER's mass is the calibrated "how blind are we" meter.

**OTHER is an ENTITY, not a scalar.** Its embedding = a learned "the rest" vector + the mass-weighted
average of the dex embeddings of the out-of-list candidates (the one place averaging is correct: OTHER
*is* the marginal). It is a pointer candidate and a Q column (§4). **One mask:** when a group's OTHER
mass is 0 (all six revealed; all four moves revealed) the OTHER token does not exist — masked in
attention (key-padding), in the opponent pointer, and as a Q column.

**Hypotheses are alternatives, not teammates.** Each hypothesis token enters attention with a
**log-weight attention bias**. Unrevealed slots are exchangeable (gen 3 has no team preview), so the
species hypotheses are ONE pool for the unseen remainder, not per-slot guesses.

**Correlation.** With revealed teammates: through the conditioned prior. Among the unseen: irrelevant
to the per-turn switch-in (one mon comes in); handled by re-conditioning on each reveal, and jointly by
determinized worlds in search. Upgrade: the learned roster and set mixtures (X12).

**Board shape** stays 12 team entity tokens + the OTHER tokens (≈ 29 → ≈ 31 tokens). E5's per-mon
tail tokens are the natural seed for OTHER_move.

---

## 2. Reasoning (T1)

Unchanged modules (`DamageOperator`, edge cells, `TeamTransformer`), now reading concrete hypotheses
instead of blobs: the op and the edges compute physics for each hypothesis; attention weights each by
its log-weight bias.

---

## 3. The opponent pointer (T2) — replaces α and β

**One flat candidate list** of concrete entities:

- their 6 move seats + OTHER_move;
- switch → each revealed bench mon;
- switch → each species hypothesis;
- OTHER_species.

**One shared scorer**, mirroring ours: `h_b = tanh(proj(token_b ⊕ cells_b) + ctx(their_cls, latent))`
(64-d), logit `w·h_b`, plus the candidate's **log belief weight as a logit bias**. One softmax = **α**
over the whole list. Illegal options are unrepresentable (masked), as β does today.

**Labels:** the candidate the opponent actually chose (one id). Today's masked belief misses
(`opp_intent/alpha_mask_rate`) become **OTHER labels**, so α(OTHER) is calibrated rather than counted.
OTHER (a move not in the list) and SWITCH-class errors (a switch to the wrong target) stay distinct.
Only a CHOICE is a label (the existing semantics: drags, forced replacements, Encore overrides are
masked; a called move is labelled as its caller).

---

## 4. The Q decomposition (T2/T3)

**In PROBABILITY space**, a two-player dueling / two-way ANOVA decomposition:

> **Q(s, a, b) = V(s) + A(s, a) + B(s, b) + I(s, a, b)**

| term | name | centring | meaning |
|---|---|---|---|
| **V** | baseline | — | P(win); the existing `value_cls` win head |
| **A** | our main effect (advantage) | Σ_a π(a)·A = 0 | how much better our move a is than our average |
| **B** | opponent main effect | Σ_b α(b)·B = 0 | how much their move b helps or hurts us on average |
| **I** | interaction | zero along both axes (double-centred) | the part that depends on the pair |

Consequences, by construction: Q(s,a) = Σ_b α Q(s,a,b) = V + A; V = the π × α average. **Our choice
depends only on A + I; theirs only on B + I; V is pure level.** A soft [0,1] bound (a mild penalty)
keeps cells in range — check early how often they leave it.

**Readouts.**
- **Ours:** a SIBLING MLP on the same per-action inputs as the policy pointer but its own weights and the
  VALUE context — `h^Q_a = tanh(proj_Q(token_a ⊕ cells_a) + ctx_Q(latent_v))` — so Q does not compete with
  the policy's 64-d readout. **Q(s,a) is not the pointer logit** (the logit is an uncalibrated PPO
  preference); log `corr(policy logit, A)` as a policy–value consistency meter.
- **Theirs:** the opponent pointer's `h_b` (shared with α: one opponent representation serves
  prediction, B and I).
- **I** = bilinear `h^Q_aᵀ W h_b` (low rank); a per-cell MLP is the fallback if matrix completion fails
  its early check (§5).
- No new CLS token: V uses `value_cls`; `h^Q_a` is state-conditioned through `latent_v`.

**Columns** = the flat opponent candidate list (§3), OTHER included.

---

## 5. Training

| part | label | loss |
|---|---|---|
| V | the real game outcome | BCE (calibration) |
| A | **DEFERRED (§5.0):** Monte-Carlo playout outcomes of sibling actions: same state, same dice (CRN), the opponent's sampled root action held fixed, then the current policy on both sides to terminal | paired-difference regression (MSE): A(a₁) − A(a₂) onto the outcome difference — LINEAR in the labels (§5.1) |
| I | **DEFERRED (§5.0):** playout outcomes per (a, b) | paired-difference regression WITHIN a column (cancels game AND opponent-choice noise; linear, §5.1) |
| B | outcome contrasts across their candidates | regression |
| α | their real action (OTHER included) | cross-entropy |
| belief masses | the true team and moves | the belief losses |

### 5.0 Label scope — at most ONE PLY now; playout labels DEFERRED (owner, 2026-09-30)

**Decision (owner, 2026-09-30, after the X4 pre-read).** *"Anything that needs a search or playout to
terminal to produce value is DEFERRED and LOWER PRIORITY. Choosing the playout ratio/coverage is
nuanced, and we are not taking that on now. In scope for now: only additions of AT MOST ONE PLY that
can be subsampled."*

**In scope now — one-ply V-bootstrapped Q̂, ONLY for its measured uses.** Q̂(s, a) = V(s′_a) at our
next decision (the terminal reward where the game ends first), subsampled:
- (a) an **inference-time one-ply lookahead**: its argmax beats the policy's by **+0.04 to +0.06**
  truth value on decisive turns, under the greedy root-opponent model (variant M, below);
- (b) **diagnostics** (the pre-read's reader, `src/main/policy_spectrum/qhat.py`).

**It is NOT an anti-starvation target and NOT a distillation target:** the pre-read shows V is blind
to the moves the policy starves. The owner's scope admits Q̂ for its measured uses only, so it is not
a training label for the Q head's A / I terms either. X4's counterfactual-label training therefore
has no current label source, and X4 as a whole moves down (§9).

**DEFERRED — the GROUNDED option: Monte-Carlo playout OUTCOMES.** Branch the action from the live
state, then play to terminal with the **current policy on both sides** (T = 1), M = 1 playout per
branch, with the same dice shared across a decision's branches (CRN). The label is the outcome in
probability units: win 1, loss 0, ties on V's own convention. A single playout outcome is an
unbiased, noisy sample of Q^π(s, a) under the current policy, and the M = 1 / linear-consumer rule
(§5.1) averages that noise out across updates. A V-bootstrapped label carries V's within-turn
blindness instead, which is a bias that no amount of averaging removes. This is the option to take
up when playout labels come back into scope. It is not the current plan: it needs a playout to
terminal, and its coverage (§8.1) is the nuanced choice the owner deferred. X6 (V trained on the same
playouts) is deferred with it.

**Why V-bootstrapped labels are shown blind** — the pre-read, 1,600 Lane S ground-truth turns, three
checkpoints (K2 final, N0 final, C_fix final; variant M, 95 % battle-clustered bootstrap;
[`READOUT.md`](../research_state/measurements/x4_preread/READOUT.md),
[`readout_tables.md`](../research_state/measurements/x4_preread/readout_tables.md)):
- **Blind to the moves the policy starves.** A starved near-best move (π < 1 %, truth within 0.1 of
  the best) lands in Q̂'s top-2 at **0.20–0.25**, against **~0.27 by chance**. The near-best moves
  the policy FEEDS land there at 0.44–0.48. As a target, softmax(Q̂/τ) puts **0.12–0.19** of its mass
  on the starved moves, against **0.17–0.18 for a UNIFORM floor**. It feeds them no more than an
  entropy floor does, and does not pick out which ones.
- **Weak within-turn ordering:** Spearman ρ **0.23–0.24** against the truth.
- **The paired error is still large.** 68–71 % of V's error variance is a per-turn offset, which
  cancels in a difference. What remains has RMS **0.34–0.36** between two actions of one turn: about
  **3.5× the ε = 0.1 near-best margin** and ~2.7× the dice noise floor (0.13).
- **The dice are not the limit, V is.** Q̂ on 1 seed per action reads the same as on 32 (K2 ρ 0.206
  → 0.226, regret 0.313 → 0.305). Repeating the dice cannot fix a V-scored label.
- **V is optimistic:** mean V 0.66–0.68 against a win rate of 0.58–0.59 on the exact successors,
  ECE 0.07–0.09.
- **The one positive result, which sets use (a).** Q̂'s ARGMAX beats the policy's argmax by
  **+0.038 / +0.056 / +0.039** truth value on decisive turns (every interval excludes 0), and by
  +0.064 to +0.073 on turns with a starved near-best move. That holds only when the root opponent is
  modelled the way the truth models it, i.e. greedy (variant M). With the opponent held at its
  recorded action (variant R) the decisive-turn gain is −0.007 to +0.020, not detected.
- **The truth is a GREEDY continuation, not Q^π** (F-X4-1, §5.3). The deferred playout labels would
  estimate Q under the current STOCHASTIC policy. The two targets differ; the pre-read's numbers are
  about V's ordering, not about what a playout label would score.

**Nonlinear uses read the LEARNED Q head** (unchanged, §5.1), whenever one is trained.

**Branch labels (the sampling recipe) — DEFERRED with the playout labels (above).** Kept as the
design for when they return. Generated INSIDE the training loop on a subsample, from the M5 core's
live in-memory state (§8.1 prices it); no side factory. Each branch is one playout. The layers were
called "one-ply" / "two-ply" when a branch was scored by V at the successor; they are named by what
they vary now.
- **The A layer (broad; was "one-ply"):** every legal action (~7) on a subsample of decisions, the
  opponent's sampled root action held fixed. With every legal row labelled, the row inclusion
  probability is 1: no row-selection bias to correct.
- **The pair layer (narrow; was "two-ply"), for I and B:** only on a narrow TARGETED subsample
  (decisive turns, §5.2). Here the (a, b) cell count binds (~49 to ~200 for the full matrix, each now a
  whole playout), so rows are chosen by **Gumbel top-k** at T ≈ 2 (k ≈ 3–4) plus one uniformly random
  legal move, and columns are the top candidates by α plus, with small probability, one below-cutoff
  action ∝ α labelled as the OTHER column. ~20–25 cells per labelled root; §8.1 prices it.
- States: the subsample is weighted toward close calls (policy top-2 near), high **EVPI** and decisive
  turns, plus a uniform share.
- **One playout per branch (M = 1), shared dice across siblings** (common random numbers), the
  opponent's actually-sampled root action held fixed — §5.1. §5.1a states how each decision type is
  handled.
- **Racing / sequential halving is an ALLOCATION device for search (§6) and offline studies only.**
  It needs repeated draws per arm, which the in-loop labels do not take (§5.1); where it runs, labels
  come from every rollout, never from the racing statistics (no winner's curse).
- Every label stores its **inclusion probability** (the state's subsample weight, and the row / column
  selection probability where Gumbel top-k or α-sampling chose it), the candidate-generator version,
  the candidate set and the opponent class — the linear losses weight by it. Accuracy is measured on a
  small **uniformly sampled** held-out cell set.
- The real played pair is labelled with its outcome for free; a real opponent action outside the list
  is labelled by identity AND as OTHER.
- Early check: does the bilinear head's filled-in matrix agree with FULLY labelled matrices on a
  held-out set (matrix completion)?

### 5.0a The ride-along A and B heads — BUILT (X4a, 2026-09-30; X26)

What is built differs from §4–§5. It is the in-scope floor: zero ply, labels PPO already produces, and
every input stop-grad. The code is `agents/model/ridealong_heads.py`, and
`designs/model/readouts_and_value_routes.md` is the doc of record for it.
- **A:** K = 5 bootstrapped members (per-state hash masks + randomized priors), each with
  `QWinProbHead`'s shared-scorer shape over the pointer head's own per-action tokens, with
  `value_pooled` as the context (this doc's `h^Q_a` with `value_pooled` in place of `latent_v`,
  which the extractor cannot see). Each is centred under π (stop-grad). The label is the GAE advantage
  of the action TAKEN (MSE on that single entry, linear per §5.1). The members' spread per action is
  the per-action epistemic read (X25: "build A/B as a bootstrapped ensemble from the start").
- **B:** K = 5 members over **α's current support**: their believed move seats, scored from the
  seat's concrete MOVE ID through the head's own embedding, plus one SWITCH column from the context.
  Each is centred under α (stop-grad). The label is the same advantage, at the column of the
  opponent's actual action (α's own label, `match_seats_to_move_num`; misses are masked). **This
  is the simple pre-X5 parameterisation, to be RE-BASED onto §3's flat pointer (seats + switch
  targets + OTHER) when X5 lands.** Its declared limit: a move outside the believed seats is not a
  label, so B is conditional on the opponent choosing a listed option. There is no OTHER mass, and
  B cannot tell switch targets apart.
- **Identification without counterfactual branches.** A gen-3 move turn is simultaneous, so a ⟂ b
  given s. With A centred under π and B centred under α, E[adv | s, a] = A(s, a) and E[adv | s, b]
  = B(s, b). Two MARGINAL regressions on the ONE observed label identify both main effects, and
  there is no I term. The bias that remains is the label's: GAE at the run's λ is V-bootstrapped, so
  A and B inherit V's within-turn blindness (§5.0). That is why X26 reads A against the pre-read's
  Q̂ as a floor, not a target.
- **Gradient routing:** DETACHED, the §5.2 "detached probe" in its simplest form. It is on-policy
  and in the loop, with no replay re-encoding and no probe-refit gap. The heads have their own
  optimizer, are outside PPO's grad clip, and use a private RNG. So they cannot move what the
  run learns, and `ridealong_update_test` pins this bit-for-bit. The shared-trunk main arm of §5.2
  is still the X4 plan.
- **Q = V + A + B** is a derived meter (`ridealong/q_out_of_range`, and the reader's
  Brier(V + A + B) at the played pair), never a training target and never a decision input.

### 5.1 Label policy — M = 1, breadth over repeats (owner, 2026-09-30)

**No per-decision repeat knob.** Each branch gets ONE opponent/dice sample (M = 1), and the compute a
repeat would cost goes to MORE DISTINCT DECISIONS. The owner's question — why not take one sample and
let the average form across many decisions and PPO updates — is right whenever the label is used
LINEARLY, so the design commits to it:

- **Labels are unbiased single samples.** For the deferred playout labels (§5.0), each label is one
  playout outcome. Common random
  numbers share the dice across a decision's branches, and the opponent's actually-sampled root action
  is held fixed. A gen-3 move turn is SIMULTANEOUS, so the opponent's root choice is independent of
  ours. Each branch is therefore an unbiased counterfactual draw of Q^π(s, a), with the opponent's real
  root policy and the current policy on both sides afterwards. Every LATER opponent choice is sampled
  inside the playout, never held from a record: the dice differ from the record, so a recorded choice
  does not apply. The per-decision-type rules, with the corrected mechanics, are in §5.1a.
- **M = 1 is confirmed for the dice (X4 pre-read, 2026-09-30).** A V-scored Q̂ on 1 dice seed per
  action reads the same as on 32. V's error, not the dice, limited it. A playout label is noisier than a
  V-scored one: a binary outcome has variance ≤ 0.25 in probability units. That variance is exactly what
  the linear consumers and the adaptive batch average out (the next bullets), and CRN cancels its shared
  part in every paired difference. The audit fraction (below) measures how much is left.
- **Every CONSUMER of a raw label is LINEAR in it**: MSE regression of the Q head (the paired
  differences of §5's A and I rows included), or an all-action advantage multiplying log π. SGD across
  updates then does the averaging: the expected gradient equals the gradient at the true mean.
- **Any NONLINEAR use reads the LEARNED Q HEAD, never a raw label** — a softmax / argmax / Gumbel
  top-k distillation target, a best-action pick, clipping. A max of noisy means is biased upward (the
  winner's curse; M5 Lane S found exactly this bias in its ground truth — UNDERSTANDING §4.4's
  caveats). The head is the amortised average.
- **The extra gradient noise from M = 1 is the adaptive-batch controller's job** (the noise scale
  grows, so K grows — `--adaptive-batch policy`, `design_learner_recipe.md`), not per-decision repeats.
- **A small AUDIT fraction** may draw a second independent sample, only to MEASURE label variance — a
  diagnostic, never a training input.
- **Rejected: "M = 1–4 repeats per decision."** With linear consumers a repeat only lowers the variance
  of a label the SGD average already de-noises, while a new decision adds a new STATE — coverage the
  average can never manufacture. Breadth is the better use of the same compute.

### 5.1a Turn mechanics and how a branch treats each decision type (verified 2026-09-30)

Verified in `deps/pokemon-showdown` @ `e0551883f` by the X4 pre-read
([`mechanics.md`](../research_state/measurements/x4_preread/mechanics.md) has file:line for every
answer). Gen 3 inherits the base `sim/battle.ts` / `sim/side.ts` turn loop unchanged
(`data/mods/gen3/scripts.ts:2` → gen 4 → gen 5).
- **After a SINGLE faint, only the fainted side chooses, MID-TURN in gen ≤ 3** (`checkFainted` after
  every action, `sim/battle.ts:2861–2864`, then `makeRequest('switch')` at `:2933`; the other side gets
  `wait`, `:1453–1457`). The opponent's NEXT-TURN choice is made after it sees our replacement
  (`endTurn` → `makeRequest('move')`, `:1797`), so it must be modelled, not held. **Confirmed.**
- **A DOUBLE faint is a SIMULTANEOUS blind choice.** One `makeRequest('switch')` issues `forceSwitch`
  to both sides, and nothing resolves until `allChoicesDone()` (`:1420–1428`, `:3081–3090`).
  **Corrects** this spec's earlier "sequential after a faint".
- **A mid-turn replacement can face an opponent move that is ALREADY LOCKED in the queue.** The rest
  of the turn's queue is saved and resumed after the replacement (`commitChoices`, `:3021–3040`).
  Baton Pass (`selfSwitch: 'copyvolatile'`, `data/moves.ts:1113`) is the same case: the opponent's
  move lands on whatever comes in, and it cannot change it. There is no reply to model, only an
  action the forced-switch observation cannot see.
- **Hidden trap:** a switch into a `maybeTrapped` request is REJECTED and re-requested with
  `trapped: true` (`sim/side.ts:966–978`), so one decision can carry two commands (F-X4-2, §5.3).

The table is written for the deferred playout labels. A one-ply branch (the in-scope lookahead)
treats the root the same way, then stops at our next decision and is scored by V there. The
locked-move row is exactly the case that a V score cannot see.

| decision at the branch point | what the branch does |
|---|---|
| **free move turn** (simultaneous) | vary our action; hold the opponent's actually-sampled root action fixed across the siblings (CRN dice); then play out |
| **our forced replacement after a SINGLE faint** (the opponent waits) | vary our replacement. The opponent has no root decision to hold; any move of its already LOCKED in the queue stays in the cloned sim state and executes in every branch. Its next-turn choice is sampled by the current policy inside the playout, after it sees our switch-in |
| **DOUBLE faint** (simultaneous blind replacements) | treat it like a free turn: hold the opponent's actually-sampled replacement fixed across our branches |
| **mid-turn replacement facing a LOCKED move** (incl. Baton Pass) | as the single-faint row: the locked move is part of the cloned state and plays out. A playout label accounts for it. A V-scored label at the forced-switch observation CANNOT, because the observation does not show the locked move, which is one more reason the labels are playouts |
| **the opponent's own later decisions** (its replacements, every later turn) | sampled from the current policy inside the playout; never held from the record |
| **our switch is rejected (hidden trap)** | the branch follows the sim: the re-request is answered by the current policy inside the playout, and the label is stored with a `trapped_retry` flag so a reader can separate those rows |

### 5.2 Placement

- **(DEFERRED with the playout labels, §5.0.) The A layer on a subsample; the pair layer only on a
  narrow targeted one** (decisive turns — a turn with at least one dominated action, UNDERSTANDING
  §4.4). Precedent: KataGo's playout-cap
  randomisation (Wu 2019: a small share of positions get the full search and train the policy, the
  rest are cheap) and Gumbel MuZero's policy improvement from few simulations (Danihelka et al. 2022).
  With playout labels, both layers' coverage is set by the sizing study (§8.1).
- **The tie to starvation (CORRECTED 2026-09-30).** M5 Lane S found the ai_v14 lineage's sharpening
  is mostly STARVATION of near-best moves (the starved share of decisive turns ~0.33 → ~0.50;
  UNDERSTANDING §4.4). A PPO gradient reaches only the action sampled; a counterfactual label gives an
  UNSAMPLED action a gradient.
  - This spec used to say that ANY such label complements X23 (policy sharpness) against starvation.
    **For one-ply labels from THIS V, the X4 pre-read contradicts it.** Q̂ puts a starved near-best
    move in its top-2 at chance (0.20–0.25 vs ~0.27). A Q̂ target feeds the starved moves no more than
    a uniform floor does (§5.0). Its only gain, a sharper argmax, pushes further in the direction that
    produced the starvation.
  - **X23 (entropy / temperature) stays the CHEAP starvation lever.** It feeds the starved moves as
    much as a V-scored Q̂ would, at no label cost.
  - **Monte-Carlo outcome labels are the GROUNDED lever (DEFERRED, §5.0),** because they carry the
    truth signal: a starved move that wins more often gets a higher label on average, whatever V thinks of it.
    Whether they actually un-starve the policy is X4's to measure; it is not assumed.
- **Soundness rule (standing, the search-as-teacher lesson; the search-teacher TRAINING consumer itself is deleted, deletion pass L3, 2026-10-02 — the lesson stands):** Q and search values train the POLICY
  side or the Q head, NEVER PPO's GAE critic target. A search value is the IMPROVED policy's value;
  regressing the GAE critic toward it biases every advantage (`../ai_v6/design_search_teacher.md`).
  X6's branch successors may train V only as on-policy values of off-path states (the continuation is
  the current π, never a max), with its off-path calibration meter. The deferred playouts of §5.0 are
  exactly such continuations: when they return, their outcomes would train V at the branch successors
  (X6) as well as label Q.

**Gradient routing (owner: heads + trunk shared).** The main arm trains the Q heads into the trunk. A
**detached probe copy** (stop-grad, observation-based replay re-encoded through the current trunk, fast
heads / slow trunk, a probe-refit gap at checkpoints, a random-trunk control probe, small probe size)
runs alongside as the baseline for "what PPO alone learns". **Guards:**
- trunk gradient cosines: cos(∇L_Q, ∇L_V), cos(∇L_Q, ∇L_policy), and per A/B/I;
- V calibration (BCE, ECE) non-inferior to the control;
- strength and entropy held.
**The opponent pointer's gradient mode is `label_only`** (owner, 2026-09-27; X20): α's label loss SHAPES the trunk (route B on — a richer trunk), but PPO and the Q losses can NOT reach α through its publication (route C cut). The failure this prevents is MOTIVATED COGNITION: a head whose output feeds action selection can be bent by the policy/Q gradient into predicting what justifies the preferred action ("they won't switch, so I attack"). α must stay a truthful predictor, scored only against real opponent actions. Meter: α's calibration SPLIT BY OUR CHOSEN ACTION — P(stay) must not inflate when we attack. Today's α runs `detached` (route B cut); `label_only` is a built, non-default mode. **Whatever α's mode, the α weights used for Q's CENTRING and MARGINALIZATION are always stop-grad**: a Q loss that could move α would "explain away" value errors by shifting the opponent model, breaking the decomposition's identifiability and the V-vs-table gap meter. PPO/Q may still shape α's hidden `h_b` (arm iii of X20). History: the 2026-09-1x search-era program already planned "contrastive, sibling-differenced, opponent-marginalized MC labels" for the win-prob head (ledger ~line 5980); this design's ranking losses are its successor.

**Critic data at scale (with M5).** Lane G's COMPLETE-GAME COLLECTOR (`program_rust_core.md` order
constraint 6, BUILT 2026-09-30; `designs/training/rust_collector.md`) supplies this. It replaced the
delayed-label buffer this section specified on 2026-09-27. A game's rows wait in the collector's
arena until the game ENDS, then take their complete-game GAE and their own outcome as the win label.
The update fires on a sample count, and the one game that straddles it is split, never dropped. Every
row is labelled once. No incomplete game is rejected, which would bias toward short games, and no row
is dropped at a window edge, which biased the old fixed-window path toward short games too. Branch successors training V (AlphaGo-style; X6, on the outcomes of the
playouts of §5.0, with an off-path calibration meter) is DEFERRED with those playouts (owner,
2026-09-30).

### 5.3 Label construction hazards (F-X4-1..3, found by the X4 pre-read)

Each one bit the pre-read's reader (`src/main/policy_spectrum/qhat.py`). Each will bite any label
builder that reads a banked battle log or the Lane S ground truth.
- **F-X4-1 — the Lane S truth used a GREEDY root opponent on 90 % of turns.** On 1,441 of the 1,600
  truth turns the opponent's root request was still open at the branch point. The truth answered it
  with the continuation's greedy choice, not the recorded one. A reader scored against that truth must
  model the root opponent the same way (the pre-read's variant M); holding the recorded action (variant
  R) disagrees with the truth on those turns by construction, and read lower on every measure. The
  truth's whole continuation is greedy on both sides, so it is NOT Q^π under the stochastic policy the
  playout labels estimate (§5.0).
- **F-X4-2 — hidden-trap switch retries in banked logs.** A switch into a `maybeTrapped` request is
  rejected and re-requested (`sim/side.ts:966–978`), so the recorded log holds TWO commands from one
  side for ONE decision: the rejected switch, then the retry. A reader that maps commands 1:1 onto
  decisions misaligns every later decision of that battle. One banked turn had one; the pre-read moves
  the pair whole (`opp_cmds_after`, `recorded_root_log`). In live labels it is the `trapped_retry` row
  of §5.1a.
- **F-X4-3 — `rec_action` is empty on the untraced side.** In an exploiter's games only one side is
  traced, and `rec_action` is `None` on the other. A label builder must take the played action from
  the decision's tokens (as `qhat.py` does), or skip the row. It must never default it.


---

## 6. Uses

- **Search:** Q as the leaf and the prior; Q(s,a,b) is a ready payoff matrix. A later arm runs
  Gumbel sequential halving over our actions with Q as prior/leaf.
- **One-ply V-bootstrapped Q̂ (Q̂ = V(successor)) — IN SCOPE NOW, for its two measured uses only
  (owner, 2026-09-30, §5.0):** (a) an **inference-time one-ply lookahead**, where its argmax beats the
  policy's by +0.04 to +0.06 truth value on decisive turns, with the root opponent modelled greedy
  (the pre-read's variant M); and (b) **diagnostics** (`src/main/policy_spectrum/qhat.py`). **Never an
  anti-starvation target, never a distillation target**: V is blind to the starved moves, and Q̂
  feeds them no more than a uniform floor does.
- **The exploit ↔ robust dial:** best-respond to α, or play the matrix's Nash strategy; set per state
  by α(OTHER) and the V-vs-table gap (trust α where it has earned it).
- **Meters:** α(OTHER) calibration per group; the V − (α-weighted table average) gap (opponent-model
  calibration in win-probability units — reported, never forced to zero); **EVPI(s)** =
  Σ_b α(b)·max_a Q(s,a,b) − max_a Σ_b α(b)·Q(s,a,b) (prediction states); corr(policy logit, A); the
  on-pool belief metrics (mass on truth by reveal count, exact-team identification, OTHER calibration,
  learned gain over the Smogon prior) with off-pool as a secondary column.

---

## 7. The omniscient twin — OFFLINE ONLY

A **detached, separate network** (own parameters, no gradient into the public model, never on the
decision path) fed by a training-only key `obs_true` from the Rust core's engine state: the opponent's
true mons through OUR full-information entity encoder with a side flag (a symmetric board), hidden
counters included (sleep/confusion/Toxic/lock/Encore-Taunt-Disable turns, exact HP, PP). Same V/A/B/I
heads. Run at CHECKPOINTS in background slots (≈ 0% training cost). Meters, via the search's
determinized worlds: reducible error V − E_belief[V*] vs irreducible Var_belief(V*); a value-of-
information map (where scouting pays). **Never a training target for the public policy or V**
(strategy fusion; the 09-10 `truevalue` arm was +0.08 optimistic and NOT DETECTED on discrimination).

---

## 8. Cost

- A/B/I heads: < 1% of the forward (they read tokens the forward already computes). The real cost is
  the counterfactual LABELS. The deferred playout labels (§5.0) would cost branches per decision ×
  the playout length × the labelled-state share, priced in §8.1, and run INSIDE the training loop
  on a subsample (not in background slots). The in-scope one-ply lookahead is ~7 successors plus 7
  forward rows per decision it runs on.
- Fixed-mass tokens: ≈ 2 more attention tokens; a small model-version bump (no observation change —
  hypotheses are built in the forward from dex tables).
- Omniscient twin: offline, ≈ 0% of training.
- Rule: anything that feeds training is judged on strength per GPU-hour with its cost included;
  anything diagnostic runs off the training critical path; any instrument costing > ~2% of
  training throughput moves offline or is cut.

### 8.1 The label cost model on the M5 core (re-measured 2026-09-30, the X4 pre-read)

Each figure is tagged **MEASURED** (with its source) or **ESTIMATED** (arithmetic on the measured
ones). **The GPU side is NOT measured**: no figure below says what a batched GPU forward of the
playout rows costs, nor the step-time share in the live loop. The M5 sizing study measures both.

| quantity | figure | tag · source |
|---|---|---|
| one successor on the playout step path (sim + fold + encode) | **133 µs** at 1 torch thread (135 µs at 4) | MEASURED · X4 pre-read bench, K2, CPU, box load 1.0–3.8 (warn, never stretch); `../research_state/measurements/x4_preread/READOUT.md` "Cost", rows `~/gen3ai_archive/x4_preread/bench_K2_S8_t{1,4}.jsonl` |
| one successor as the search TREE builds it (step-built rows) | ~232 µs per thread | MEASURED · `../research_state/measurements/m5_laneI/PROGRESS.md` Unit 4 (search's depth-1 ply row). Not the label path |
| one policy / value forward, per row, on CPU | **946 µs** at 1 torch thread (549 µs at 4), **~7× the sim** | MEASURED · X4 pre-read bench (same rows) |
| one policy / value forward, per row, batched on the GPU | **NOT MEASURED** | the sizing study |
| the training loop at the production shape (95/5 self-play, 48 envs, complete-game collector, T2) | **3,780 [3,650, 3,941] trainee decisions / s**, ~0.5 ms CPU per decision (496 µs) | MEASURED · Lane G, `../research_state/measurements/m5_laneJ/results/throughput_production_sp95_n48.json` (arm `rust_serial_keyed`) |
| the box | **16 cores** | MEASURED · same file, `cpu_count` |
| **one PLAYOUT label** (A layer, §5.0): every legal action (≈ 7) × the remaining decisions (≈ half the mean game length of ~25 turns, ≈ 12) | **≈ 80–90 policy decisions per labelled root decision**, per side. Both sides play the current policy, so ≈ 160–180 forward rows. A loop decision already carries one opponent forward, so one playout step costs about one loop decision | ESTIMATED · the ~25-turn mean is the orchestrator's figure, not measured here; decisive-turn roots sit at their own depth, not the game's midpoint |
| the A layer at **1 %** coverage (≈ 38 roots / s) | **≈ 3,000–3,400 extra decisions / s per side = +0.8–0.9× the loop's own rate**. Sim ≈ 0.4–0.5 core (floor; the opponent's encode adds up to ~2×). Forwards on CPU ≈ 6 cores. If the loop is decision-bound, trainee throughput falls to ≈ 2,000–2,100 / s | ESTIMATED |
| the A layer at **2 %** coverage | **≈ 6,000–6,800 / s = +1.6–1.8×**. Sim ≈ 0.8–0.9 core; CPU forwards ≈ 11–13 cores. Decision-bound trainee rate ≈ 1,350–1,450 / s | ESTIMATED |
| the A layer at **5 %** coverage | **≈ 15,000–17,000 / s = +4.0–4.5×**. Sim ≈ 2.0–2.3 cores; CPU forwards ≈ 29–32 cores, beyond the box. Decision-bound trainee rate ≈ 690–760 / s | ESTIMATED |
| the pair layer (I and B): ~20–25 (a, b) cells per labelled root (Gumbel top-k rows × α columns, §5.0) | ≈ 1,600–2,250 decisions per root. At **0.1 %** coverage ≈ +1.6–2.3× the loop's rate: far costlier than the A layer; coverage well under 0.1 %, or truncated playouts once X6 licenses them | ESTIMATED |
| a V-scored one-ply Q̂ (IN SCOPE: the inference-time lookahead and diagnostics only, §5.0) | **~98 ms** per decision at S = 8 on 1 thread (~65 ms at 4); **~12 ms at S = 1** | MEASURED at S = 8 · pre-read bench; S = 1 ESTIMATED from it |

**Reading the estimates.** CPU forwards are infeasible beyond about 1 % coverage: 946 µs a row
against a 16-core box. The playout rows must ride the GPU inference tier (T2), batched with the
loop's own. The "decision-bound" trainee rates are the UPPER bound on the slowdown: batched GPU
forwards of playout rows may be cheaper per row than the loop's average decision. The labelling cost
is a FORWARD-PASS budget, not a sim budget. The sim side stays at a few cores even at 5 %.

**Against the August factory.** The 2026-08-21 counterfactual label factory reached **1.7% coverage on
4 cores**; full coverage would have needed **~230 cores**, because **91% of its time was PREFIX
REPLAY** — each arm re-played the battle from turn 1 (arm ≈ 4.78 + 0.853·turn ms, ~26 ms at the mean
turn, before prefix sharing's measured 2.91×) (ledger 2026-08-21 · the counterfactual cost model; the
`cf_audit` command it produced). The M5 core holds the LIVE in-memory state, so a branch is a clone of
the state the loop is already at — no replay — and one successor costs ~133 µs.

**Conclusion.** Playout labels are DEFERRED (owner, 2026-09-30, §5.0); this is their price for when
they return. They would be generated INSIDE the loop (§5), never in a side factory. They are
≈ 12× the sim steps and ≈ 25× the forward rows of a V-scored one-ply label per labelled root (≈ 7
successors and 7 rows), so coverage is a real trade:
1 % coverage roughly doubles the policy-decision work (ESTIMATED). The sizing study measures the GPU
side. Choosing the A layer's coverage is the nuanced call the owner deferred, and the pair layer
would run far below it.

---

## 9. Staging

**Owner, 2026-09-30: only additions of AT MOST ONE PLY that can be subsampled are in scope now.**
Anything that needs a search or a playout to terminal to produce a value is DEFERRED and LOWER
PRIORITY (§5.0).

1. **X3** (no training): policy top-N prior + Foul Play's hand eval as the leaf — is discrimination
   the missing piece?
2. **In scope now (at most one ply, subsampled):**
   - the one-ply V-bootstrapped Q̂ as an **inference-time lookahead** and as a **diagnostic** (§5.0,
     §6), never an anti-starvation or distillation target;
   - **X4a**, the A/B ride-along on the labels PPO already produces (zero extra ply). **BUILT
     2026-09-30 (§5.0a) and runs in the X26 baseline**, with B on α's support until X5;
   - **X23** (entropy / temperature, zero ply) stays the starvation lever (§5.2);
   - **X5**, the fixed-mass belief tokens (§1), zero ply, STANDALONE. It is no longer bundled with
     X4 (orchestrator, 2026-09-30), and it lands at the North Star 1 retrain boundary.
3. **DEFERRED, lower priority:**
   - **X4 as a whole moves DOWN.** Its counterfactual-label training has no in-scope label source:
     one-ply Q̂ is not a training label (§5.0), and the grounded option, Monte-Carlo playout OUTCOMES,
     needs a playout to terminal. When it returns, it is ready after the M5 switch and the M5 sizing
     study (`program_rust_core.md` order constraint 5), on top of the fixed-mass tokens (X5, landed
     on their own by then) and the flat opponent pointer, then the Q heads (detached probe + shared
     main arm), judged on the leaf rows with V and strength guards;
   - **X6** (branch successors train V through an on-policy continuation to terminal), for the same
     reason. When playouts return, X6 rides on the same outcomes, and it is a prerequisite for any
     V-bootstrapped TRAINING label (V's within-turn discrimination is the measured bottleneck);
   - the Gumbel-search arm, then expert iteration (X15). Distillation was the only BUILT route to X15 and is DELETED (deletion pass L3, 2026-10-02); a Rust port is ~1-2 agent-days if X15 is scheduled (`distill_mask` becomes a per-episode host key from `TeamStager`; teachers are frozen T2 slots or a learner-side forward).
4. **X11** (the omniscient twin) as background infrastructure whenever a checkpoint is read.

## 10. Open questions (resolve by measurement, record the answer here)

- The bilinear rank vs a per-cell MLP (the matrix-completion check).
- The OTHER embedding: learned-only vs learned + out-of-list average.
- The team hypothesis budget: 6 vs 7–8 (read OTHER_species mass early in games).
- The weight of the branch-ranking losses into the trunk, set by the gradient-cosine guard.
- Shared vs detached α.
- DEFERRED with the playout labels (owner, 2026-09-30): the A layer's playout coverage (1 %, 2 %,
  5 %, …) against the sizing study's GPU forward cost (§8.1); whether the pair layer is affordable at
  all; and the bot slice (the 5 % non-self-play games), whether to label it with the bot continuing
  on its side or skip it, since the "current policy on both sides" rule (§5.0) is written for
  self-play.

---

## Decision record

Owner decisions are marked **(owner)**. `L…` is the ledger line as `ledger_index.md` lists it.

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-09-27 | The value side's end state **(owner-directed)** | Q(s,a,b) = V + A + B + I in probability space, double-centred under π × α; calibration (V) separated from discrimination (A, I) | Another scalar win-prob head (six sat on the null as search leaves, mirror 0.485–0.507) | §0, §4; `556eb4f8` |
| 2026-09-27 | Principles **(owner)** | Discrete-first entities with one OTHER entity; measure before adopting; pool memorization a SUCCESS milestone (on-pool first) | Blob belief slots | §0, §1 |
| 2026-09-27 | Opponent model | One flat pointer α over concrete candidates, OTHER as entities, masked when impossible; belief misses become OTHER labels | Separate α and β heads; masking misses (`opp_intent/alpha_mask_rate`) | §3; TASK_BACKLOG T4 |
| 2026-09-27 | Our readout | A SIBLING MLP on the policy's per-action inputs with its own weights and the value context | Reusing the policy pointer logit (an uncalibrated PPO preference) | §4 |
| 2026-09-27 | Gradient routing **(owner)** | Q heads trained into the shared trunk, with a detached probe as the baseline and gradient-cosine guards | — (the detached probe runs alongside as the baseline) | §5 |
| 2026-09-27 | α's gradient mode **(owner)** | `label_only`: α's label loss shapes the trunk (route B on); PPO and Q cannot bend α (route C cut) | Letting the policy / Q gradient reach α (motivated cognition: "they won't switch, so I attack") | §5; EXPERIMENT_BACKLOG X20 (`503872c6`) |
| 2026-09-27 | Critic data at scale — **SUPERSEDED 2026-09-30** by Lane G's complete-game collector (below) | A DELAYED-LABEL BUFFER back-fills outcomes, every row once | Rejecting incomplete games (biases toward short games) | §5 |
| 2026-09-27 | Omniscient twin | OFFLINE only, own parameters, never a training target | Training the public V on true values (the 09-10 `truevalue` arm: +0.08 optimistic, NOT DETECTED on discrimination) | §7 |
| 2026-09-27 | Priority **(owner)** | Q first in the research queue; memorization chipped in parallel | Root-causing memorization first | EXPERIMENT_BACKLOG header (`93745a66`) |
| 2026-09-27 | V calibration **(owner)** | Fix under-dispersion NATIVELY (X21); post-hoc temperature / Platt only as an isolation control | Post-hoc recalibration as the end state | EXPERIMENT_BACKLOG X21 (`503872c6`) |
| 2026-09-30 | Label cost model on the M5 core — **SUPERSEDED 2026-09-30** by the re-measured cost model below (the labels are playouts now; successor 133 µs, not 232) | One-ply labels for every legal action generated INSIDE the training loop on a subsample (~6 cores at 100%, ~0.3 core at 5%, ESTIMATED from a MEASURED 232 µs successor and 3,780 decisions / s); two-ply only on a 1–3% targeted subsample | A side label factory (August: 1.7% coverage on 4 cores, ~230 cores for full, 91% prefix replay); two-ply at full coverage (~40–170 cores) | §8.1; m5_laneI PROGRESS Unit 4; m5_laneJ `throughput_production_sp95_n48.json`; ledger 2026-08-21 |
| 2026-09-30 | Label repeats **(owner)** | NO per-decision repeat knob: M = 1 opponent/dice sample per branch (CRN dice, the opponent's sampled action held fixed), compute to MORE DISTINCT DECISIONS; every raw-label consumer LINEAR; nonlinear uses read the learned Q head; M = 1 noise → the adaptive-batch controller; an audit fraction's second draw measures variance only | M = 1–4 repeats per decision (with linear consumers breadth is the better use of the same compute; a max over raw labels carries the winner's curse Lane S found) | §5.1; UNDERSTANDING §4.4 |
| 2026-09-30 | A and I losses (reconciles §5 with the M = 1 decision) | Paired-difference REGRESSION (MSE) of A(a₁) − A(a₂) / within-column I differences onto the CRN label difference | The pairwise BT ranking loss this doc specified on 2026-09-27: on a single noisy draw its target is P(label₁ > label₂), a nonlinear function of the label, not the difference of means — SGD's averaging would not recover A. The pairing (noise cancellation) is kept | §5 table, §5.1 |
| 2026-09-30 | Label recipe (reconciles §5) — **SUPERSEDED IN PART 2026-09-30**: each branch is now a playout, not a V-scored successor (label source, below); the row / column / inclusion-probability rules stand | One-ply labels EVERY legal row (inclusion probability 1); Gumbel top-k rows + α columns with stored inclusion probabilities kept for two-ply; inclusion probabilities now also carry the state-subsample weight; racing / sequential halving kept for search and offline studies only | "fresh labels" and in-loop racing (both need repeated draws per arm, which the M = 1 decision rejects); Gumbel top-k at one ply (every row is affordable there) | §5, §8.1 |
| 2026-09-30 | Placement — the X23 clause **SUPERSEDED 2026-09-30** (starvation, below) | One-ply broad, two-ply narrow on decisive turns (precedent: KataGo playout-cap randomisation, Wu 2019; Gumbel MuZero, Danihelka et al. 2022); counterfactual labels give unsampled actions a gradient, complementing X23 against starvation; Q / search values never train PPO's GAE critic target | Uniform two-ply; search values as the critic target (biases the advantage — the search-as-teacher lesson) | §5.2; UNDERSTANDING §4.4; `../ai_v6/design_search_teacher.md` |
| 2026-09-30 | X4 readiness | M5 successors and T2 are built; X4 is ready after the M5 switch and the sizing study | "after M5's successors and T2" (both now built) | §9; EXPERIMENT_BACKLOG X4 |
| 2026-09-30 | **Label scope (owner)** | Only additions of AT MOST ONE PLY that can be subsampled are in scope now; anything that needs a search or a playout to terminal to produce a value is DEFERRED and lower priority. One-ply V-bootstrapped Q̂ is kept ONLY for its measured uses: an inference-time one-ply lookahead (argmax +0.04 to +0.06 truth value over the policy's on decisive turns, greedy root opponent, variant M) and diagnostics. It is NOT an anti-starvation or distillation target, nor a Q-head training label. Monte-Carlo playout OUTCOMES (current policy both sides to terminal, M = 1, CRN) are recorded as the GROUNDED option for when labels return, and are not the current plan. X4 as a whole moves down | The orchestrator's same-day proposal to make playout outcomes X4's training labels now (choosing the playout ratio / coverage is nuanced; not taken on now). Q̂ = V(successor) as a training label (the earlier 2026-09-30 recipe above): V-bootstrapped labels are shown BLIND. Starved near-best in Q̂ top-2 0.20–0.25 vs ~0.27 chance; within-turn ρ ~0.23; paired RMS error 0.34–0.36, ~3.5× the 0.1 margin; S = 1 reads like S = 32, so V is the limit, not the dice; a bias that averaging cannot remove | §5.0, §6, §9; `../research_state/measurements/x4_preread/READOUT.md`; `a831190b`, `54fd5a6b` |
| 2026-09-30 | **X6 DEFERRED (owner)** | X6 (branch successors train V through an on-policy continuation to terminal) is deferred with the playout labels, for the same reason. When playouts return it rides on the same outcomes (the current π's value of off-path states, never a max), and it is a prerequisite for any V-bootstrapped TRAINING label | The orchestrator's same-day proposal to move X6 up and co-land it with X4 | §5.0, §5.2, §9; EXPERIMENT_BACKLOG X6 |
| 2026-09-30 | **Starvation** (orchestrator; the owner may revise) | X23 (entropy / temperature) stays the CHEAP starvation lever; MC-outcome labels are the GROUNDED one, because they carry the truth signal (DEFERRED with them, owner 2026-09-30; X4 measures whether they un-starve; not assumed) | "Counterfactual labels complement X23 against starvation" for any label: for one-ply labels from THIS V the measurement contradicts it. A softmax(Q̂/τ) target puts 0.12–0.19 of its mass on starved moves, vs 0.17–0.18 for uniform | §5.2; READOUT "Verdict" |
| 2026-09-30 | **Cost model re-measured** (orchestrator) | MEASURED: successor 133 µs, CPU forward 946 µs a row at 1 thread (~7× the sim), loop 3,780 decisions / s. ESTIMATED: a playout label ≈ 7 actions × ≈ 12 remaining decisions ≈ 80–90 policy decisions per side; the A layer at 1 / 2 / 5 % coverage ≈ +0.8–0.9× / +1.6–1.8× / +4.0–4.5× the loop's decision rate. CPU forwards are infeasible beyond ~1 %, so the rows ride T2. The GPU forward cost is NOT measured; the sizing study sets coverage | The one-ply V-scored model (~6 cores at 100 %, ~0.3 core at 5 %, from a 232 µs successor, which was the search TREE's step-built row) | §8.1; x4_preread READOUT "Cost"; m5_laneJ `throughput_production_sp95_n48.json` |
| 2026-09-30 | **Turn mechanics corrected** (verified at source) | After a SINGLE faint only the fainted side chooses, mid-turn in gen ≤ 3, and the opponent's next-turn reply must be modelled (confirmed). A DOUBLE faint is a SIMULTANEOUS blind choice. A mid-turn replacement (incl. Baton Pass) can face an opponent move ALREADY LOCKED in the queue. The branch rules per decision type are §5.1a's table | "Sequential after a faint; the opponent's reply must be modelled" for every faint (this spec's §5.1 until 2026-09-30) | §5.1a; `../research_state/measurements/x4_preread/mechanics.md` (`sim/battle.ts:2861–2864`, `:2933`, `:1420–1428`, `:3021–3040`, `:3081–3090`) |
| 2026-09-30 | **Critic data at scale: the complete-game collector** (brings §5.2 in line with `program_rust_core.md`) | Lane G's complete-game collector (order constraint 6, owner 2026-09-29; BUILT 2026-09-30) is the critic-data mechanism: rows wait in its arena until their game ends, get complete-game GAE and their own outcome, and the sample-count trigger splits (never drops) the one straddling game, so every row is labelled once | The delayed-label buffer this doc specified on 2026-09-27 (SUBSUMED: K10(b), a bolt-on beside fixed n_steps windows; the collector retires the windows themselves) | §5.2; `program_rust_core.md` order constraint 6 and its 2026-09-29 Lane G collector row; `../training/rust_collector.md` |
| 2026-09-30 | **X5 unbundled from X4** (orchestrator) | The fixed-mass belief tokens (§1, X5) STAND ALONE: a zero-ply architecture change, in scope under the owner's one-ply rule, landing at the North Star 1 retrain boundary; judged on intent NLL / calibration, OTHER rates, on-pool belief metrics and a strength guard; the I-term readout waits for X4's return | Keeping X5 "bundled with X4" (it would inherit X4's deferral although it adds no ply) | §1, §9; EXPERIMENT_BACKLOG X5 |
| 2026-09-30 | **The ride-along A / B heads (X4a) — BUILT** (owner's baseline ask, 2026-09-30: "establish a baseline just with the ensemble of value heads … and the Q, A and B values") | DETACHED heads on labels PPO already produces. A = per-action, centred under π, MSE on the taken action's GAE advantage. B = over α's current support (seats by move ID + SWITCH), centred under α, MSE on the same advantage where the opponent's action is named. Both are identified by marginal regressions because the turn is simultaneous. Each is a K = 5 bootstrapped ensemble with randomized priors. No I term; Q = V + A + B is a meter. B is to be RE-BASED onto X5's flat pointer | (i) Deferring B until X5 (the backlog's stated prerequisite): rejected, because marginal identification needs only a centring distribution and α exists today; the limit (no OTHER, conditional on a listed option) is declared, not faked. (ii) One joint regression A(a) + B(b) onto the advantage: equivalent in expectation, but it couples the two heads' errors and doubles the per-row noise in each, so it was rejected. (iii) Training into the shared trunk (§5.2's main arm): out of scope for a baseline that must learn exactly what production learns | §5.0a; `agents/model/ridealong_heads.py`; `ridealong_heads_test` / `ridealong_update_test`; EXPERIMENT_BACKLOG X4a / X26 |
| 2026-09-30 | **Label construction hazards** | F-X4-1 (Lane S truth answered an open root opponent GREEDY on 90 % of turns), F-X4-2 (hidden-trap switch retries: two commands for one decision in banked logs), F-X4-3 (`rec_action` empty on the untraced side), each with its rule | — | §5.3; `src/main/policy_spectrum/qhat.py` |
