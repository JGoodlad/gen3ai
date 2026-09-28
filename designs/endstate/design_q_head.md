# Design — the Q head, the opponent pointer, and fixed-mass belief tokens

**Status: DESIGN (owner-directed, 2026-09-27). Not built.** Experiment rows X3–X6 and X11 in
[`../research_state/EXPERIMENT_BACKLOG.md`](../research_state/EXPERIMENT_BACKLOG.md); build rows T2–T4
in [`../ops/TASK_BACKLOG.md`](../ops/TASK_BACKLOG.md).

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
| A | branch rollouts of sibling actions, same state, same dice | pairwise ranking: BT logit = A(a₁) − A(a₂) |
| I | branch rollouts per (a, b) | ranking WITHIN a column (cancels game AND opponent-choice noise) |
| B | outcome contrasts across their candidates | regression |
| α | their real action (OTHER included) | cross-entropy |
| belief masses | the true team and moves | the belief losses |

**Branch labels (the sampling recipe).**
- States: weighted toward close calls (policy top-2 near) and high **EVPI**, plus a uniform share.
- Our rows: **Gumbel top-k** at T ≈ 2 (k ≈ 3–4), plus one uniformly random legal move.
- Their columns: the top candidates by α, plus with small probability one below-cutoff action ∝ α
  rolled out as the OTHER column.
- Rollouts: the CURRENT stochastic policies (T = 1) on both sides, **shared dice across siblings**.
- **Racing / sequential halving for ALLOCATION only**; labels from every rollout (or fresh ones) — never
  the racing statistics (no winner's curse).
- Every label stores its **inclusion probability**, the candidate-generator version, the candidate set
  and the opponent class. Accuracy is measured on a small **uniformly sampled** held-out cell set.
- The real played pair is labelled with its outcome for free; a real opponent action outside the list
  is labelled by identity AND as OTHER.
- Early check: does the bilinear head's filled-in matrix agree with FULLY labelled matrices on a
  held-out set (matrix completion)?

**Gradient routing (owner: heads + trunk shared).** The main arm trains the Q heads into the trunk. A
**detached probe copy** (stop-grad, observation-based replay re-encoded through the current trunk, fast
heads / slow trunk, a probe-refit gap at checkpoints, a random-trunk control probe, small probe size)
runs alongside as the baseline for "what PPO alone learns". **Guards:**
- trunk gradient cosines: cos(∇L_Q, ∇L_V), cos(∇L_Q, ∇L_policy), and per A/B/I;
- V calibration (BCE, ECE) non-inferior to the control;
- strength and entropy held.
**The opponent pointer's gradient mode is `label_only`** (owner, 2026-09-27; X20): α's label loss SHAPES the trunk (route B on — a richer trunk), but PPO and the Q losses can NOT reach α through its publication (route C cut). The failure this prevents is MOTIVATED COGNITION: a head whose output feeds action selection can be bent by the policy/Q gradient into predicting what justifies the preferred action ("they won't switch, so I attack"). α must stay a truthful predictor, scored only against real opponent actions. Meter: α's calibration SPLIT BY OUR CHOSEN ACTION — P(stay) must not inflate when we attack. Today's α runs `detached` (route B cut); `label_only` is a built, non-default mode. **Whatever α's mode, the α weights used for Q's CENTRING and MARGINALIZATION are always stop-grad**: a Q loss that could move α would "explain away" value errors by shifting the opponent model, breaking the decomposition's identifiability and the V-vs-table gap meter. PPO/Q may still shape α's hidden `h_b` (arm iii of X20). History: the 2026-09-1x search-era program already planned "contrastive, sibling-differenced, opponent-marginalized MC labels" for the win-prob head (ledger ~line 5980); this design's ranking losses are its successor.

**Critic data at scale (with M5).** A DELAYED-LABEL BUFFER keeps unfinished games' rows and back-fills
the outcome when the game ends (every row once; no rejection of incomplete games — that would bias
toward short games). Branch successors may also train V (AlphaGo-style; X6), with an off-path
calibration meter.

---

## 6. Uses

- **Search:** Q as the leaf and the prior; Q(s,a,b) is a ready payoff matrix. A later arm runs
  Gumbel sequential halving over our actions with Q as prior/leaf.
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

- A/B/I heads: < 1% of the forward (they read tokens the forward already computes); the real cost is
  the branch LABELS, set by k × rollouts × the labelled-state share, run in background slots after M5.
- Fixed-mass tokens: ≈ 2 more attention tokens; a small model-version bump (no observation change —
  hypotheses are built in the forward from dex tables).
- Omniscient twin: offline, ≈ 0% of training.
- Rule: anything that feeds training is judged on strength per GPU-hour with its cost included;
  anything diagnostic runs off the training critical path; any instrument costing > ~2% of
  training throughput moves offline or is cut.

---

## 9. Staging

1. **X3** (no training): policy top-N prior + Foul Play's hand eval as the leaf — is discrimination
   the missing piece?
2. **X4 + X5**, after M5's successors and T2: the fixed-mass tokens and the flat opponent pointer, then
   the Q heads (detached probe + shared main arm), judged on the leaf rows with V and strength guards.
3. **X6** (branch successors train V), then the Gumbel-search arm, then expert iteration (X15).
4. **X11** (the omniscient twin) as background infrastructure whenever a checkpoint is read.

## 10. Open questions (resolve by measurement, record the answer here)

- The bilinear rank vs a per-cell MLP (the matrix-completion check).
- The OTHER embedding: learned-only vs learned + out-of-list average.
- The team hypothesis budget: 6 vs 7–8 (read OTHER_species mass early in games).
- The weight of the branch-ranking losses into the trunk, set by the gradient-cosine guard.
- Shared vs detached α.

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
| 2026-09-27 | Critic data at scale | A DELAYED-LABEL BUFFER back-fills outcomes, every row once | Rejecting incomplete games (biases toward short games) | §5 |
| 2026-09-27 | Omniscient twin | OFFLINE only, own parameters, never a training target | Training the public V on true values (the 09-10 `truevalue` arm: +0.08 optimistic, NOT DETECTED on discrimination) | §7 |
| 2026-09-27 | Priority **(owner)** | Q first in the research queue; memorization chipped in parallel | Root-causing memorization first | EXPERIMENT_BACKLOG header (`93745a66`) |
| 2026-09-27 | V calibration **(owner)** | Fix under-dispersion NATIVELY (X21); post-hoc temperature / Platt only as an isolation control | Post-hoc recalibration as the end state | EXPERIMENT_BACKLOG X21 (`503872c6`) |
