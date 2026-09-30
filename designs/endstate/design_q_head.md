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
| A | branch labels of sibling actions, same state, same dice (CRN), the opponent's sampled action held fixed | paired-difference regression (MSE): A(a₁) − A(a₂) onto the label difference — LINEAR in the labels (§5.1) |
| I | branch labels per (a, b) | paired-difference regression WITHIN a column (cancels game AND opponent-choice noise; linear, §5.1) |
| B | outcome contrasts across their candidates | regression |
| α | their real action (OTHER included) | cross-entropy |
| belief masses | the true team and moves | the belief losses |

**Branch labels (the sampling recipe).** Generated INSIDE the training loop on a subsample, from
the M5 core's live in-memory state (§8.1 prices it); no side factory.
- **One-ply (the broad layer):** every legal action (~7) on a broad subsample of decisions — each
  branch is a successor (clone, advance one turn, fold, encode) scored by a value-only forward. With
  every legal row labelled, the row inclusion probability is 1: no row-selection bias to correct.
- **Two-ply (the narrow layer):** only on a narrow TARGETED subsample (decisive turns, §5.2); here the
  branch count (~49 to ~200) binds, so rows are chosen by **Gumbel top-k** at T ≈ 2 (k ≈ 3–4) plus one
  uniformly random legal move, and columns are the top candidates by α plus, with small probability,
  one below-cutoff action ∝ α labelled as the OTHER column.
- States: the subsample is weighted toward close calls (policy top-2 near), high **EVPI** and decisive
  turns, plus a uniform share.
- **One sample per branch (M = 1), shared dice across siblings** (common random numbers), the
  opponent's actually-sampled action held fixed — §5.1. Deeper continuations (rollout to terminal
  under the CURRENT stochastic policies, T = 1, both sides) stay a valid, costlier label form.
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

### 5.1 Label policy — M = 1, breadth over repeats (owner, 2026-09-30)

**No per-decision repeat knob.** Each branch gets ONE opponent/dice sample (M = 1), and the compute a
repeat would cost goes to MORE DISTINCT DECISIONS. The owner's question — why not take one sample and
let the average form across many decisions and PPO updates — is right whenever the label is used
LINEARLY, so the design commits to it:

- **Labels are unbiased single samples.** Common random numbers share the dice across a decision's
  branches, and the opponent's actually-sampled action is held fixed. A gen-3 turn is SIMULTANEOUS, so
  the opponent's choice is independent of ours and each branch is an unbiased counterfactual draw of
  Q(s, a) under the opponent's real policy. At a forced replacement after a faint the turn is
  SEQUENTIAL — the opponent's next choice is made after it sees ours — so no recorded opponent action
  can be held fixed across our branches; its reply must be MODELLED (inside the successor's value at
  one ply; from α / the opponent policy in a deeper continuation).
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

### 5.2 Placement

- **One-ply on a broad subsample; two-ply only on a narrow targeted one** (decisive turns — a turn with
  at least one dominated action, UNDERSTANDING §4.4). Precedent: KataGo's playout-cap randomisation
  (Wu 2019: a small share of positions get the full search and train the policy, the rest are cheap)
  and Gumbel MuZero's policy improvement from few simulations (Danihelka et al. 2022).
- **The tie to starvation.** M5 Lane S found the ai_v14 lineage's sharpening is mostly STARVATION of
  near-best moves (the starved share of decisive turns ~0.33 → ~0.50; UNDERSTANDING §4.4). A PPO
  gradient reaches only the action sampled; a counterfactual label gives an UNSAMPLED action a
  gradient, so these labels attack the same failure as X23 (policy sharpness) from the other side —
  complements, not substitutes.
- **Soundness rule (standing, the search-as-teacher lesson):** Q and search values train the POLICY
  side or the Q head, NEVER PPO's GAE critic target. A search value is the IMPROVED policy's value;
  regressing the GAE critic toward it biases every advantage (`../ai_v6/design_search_teacher.md`).
  X6's branch successors may train V only as on-policy values of off-path states (the continuation is
  the current π, never a max), with its off-path calibration meter.

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
  the branch LABELS, set by branches per decision × the labelled-state share — priced in §8.1, and run
  INSIDE the training loop on a subsample (not in background slots).
- Fixed-mass tokens: ≈ 2 more attention tokens; a small model-version bump (no observation change —
  hypotheses are built in the forward from dex tables).
- Omniscient twin: offline, ≈ 0% of training.
- Rule: anything that feeds training is judged on strength per GPU-hour with its cost included;
  anything diagnostic runs off the training critical path; any instrument costing > ~2% of
  training throughput moves offline or is cut.

### 8.1 The label cost model on the M5 core (2026-09-30)

Each figure is tagged **MEASURED** (with its source) or **ESTIMATED** (arithmetic on the measured
ones; the step-time share is not yet read in the live loop — the sizing study reads it).

| quantity | figure | tag · source |
|---|---|---|
| one successor — clone, advance one turn, fold, encode | **~232 µs CPU per thread**, in process (4,314 / s; release, one thread) | MEASURED · `../research_state/measurements/m5_laneI/PROGRESS.md` Unit 4 (search's depth-1 ply row) |
| the training loop at the production shape (95/5 self-play, 48 envs, complete-game collector, T2) | **3,780 [3,650, 3,941] trainee decisions / s**, **~0.5 ms CPU per decision** (496 µs) | MEASURED · Lane G, `../research_state/measurements/m5_laneJ/results/throughput_production_sp95_n48.json` (arm `rust_serial_keyed`) |
| the box | **16 cores** | MEASURED · same file, `cpu_count` |
| one-ply, every legal action (~7), M = 1, on 100% of decisions | ≈ 7 × 232 µs ≈ **1.6 ms CPU per decision ≈ 6 cores** at 3,780 / s, plus ~7 value-only forward rows per decision (roughly a **15–25% longer step**) | ESTIMATED |
| one-ply at 5% coverage (decisive turns) | ≈ **0.3 core** — under one core, effectively free | ESTIMATED |
| two-ply (~49 to ~200 branches) at full coverage | ≈ 11–46 ms CPU per decision ≈ **40–170 cores** — infeasible; on a **1–3% targeted subsample** ≈ 0.4–5 cores | ESTIMATED |

**Against the August factory.** The 2026-08-21 counterfactual label factory reached **1.7% coverage on
4 cores**; full coverage would have needed **~230 cores**, because **91% of its time was PREFIX
REPLAY** — each arm re-played the battle from turn 1 (arm ≈ 4.78 + 0.853·turn ms, ~26 ms at the mean
turn, before prefix sharing's measured 2.91×) (ledger 2026-08-21 · the counterfactual cost model; the
`cf_audit` command it produced). The M5 core holds the LIVE in-memory state, so a branch is a clone of
the state the loop is already at — no replay — and one successor costs ~232 µs.

**Conclusion.** One-ply labels for every legal action fit INSIDE the training loop on a broad
subsample for a few cores and a modest step-time share; two-ply fits only on a narrow targeted
subsample. The labels are generated in the loop (§5), not in a side factory.

---

## 9. Staging

1. **X3** (no training): policy top-N prior + Foul Play's hand eval as the leaf — is discrimination
   the missing piece?
2. **X4 + X5** — M5's successors and T2 are BUILT, so X4 is ready after the M5 switch (training on the
   Rust env core) and the M5 sizing study (`program_rust_core.md` order constraint 5), with the labels
   costed in §8.1: the fixed-mass tokens and the flat opponent pointer, then the Q heads (detached
   probe + shared main arm), judged on the leaf rows with V and strength guards.
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
| 2026-09-30 | Label cost model on the M5 core | One-ply labels for every legal action generated INSIDE the training loop on a subsample (~6 cores at 100%, ~0.3 core at 5%, ESTIMATED from a MEASURED 232 µs successor and 3,780 decisions / s); two-ply only on a 1–3% targeted subsample | A side label factory (August: 1.7% coverage on 4 cores, ~230 cores for full, 91% prefix replay); two-ply at full coverage (~40–170 cores) | §8.1; m5_laneI PROGRESS Unit 4; m5_laneJ `throughput_production_sp95_n48.json`; ledger 2026-08-21 |
| 2026-09-30 | Label repeats **(owner)** | NO per-decision repeat knob: M = 1 opponent/dice sample per branch (CRN dice, the opponent's sampled action held fixed), compute to MORE DISTINCT DECISIONS; every raw-label consumer LINEAR; nonlinear uses read the learned Q head; M = 1 noise → the adaptive-batch controller; an audit fraction's second draw measures variance only | M = 1–4 repeats per decision (with linear consumers breadth is the better use of the same compute; a max over raw labels carries the winner's curse Lane S found) | §5.1; UNDERSTANDING §4.4 |
| 2026-09-30 | A and I losses (reconciles §5 with the M = 1 decision) | Paired-difference REGRESSION (MSE) of A(a₁) − A(a₂) / within-column I differences onto the CRN label difference | The pairwise BT ranking loss this doc specified on 2026-09-27: on a single noisy draw its target is P(label₁ > label₂), a nonlinear function of the label, not the difference of means — SGD's averaging would not recover A. The pairing (noise cancellation) is kept | §5 table, §5.1 |
| 2026-09-30 | Label recipe (reconciles §5) | One-ply labels EVERY legal row (inclusion probability 1); Gumbel top-k rows + α columns with stored inclusion probabilities kept for two-ply; inclusion probabilities now also carry the state-subsample weight; racing / sequential halving kept for search and offline studies only | "fresh labels" and in-loop racing (both need repeated draws per arm, which the M = 1 decision rejects); Gumbel top-k at one ply (every row is affordable there) | §5, §8.1 |
| 2026-09-30 | Placement | One-ply broad, two-ply narrow on decisive turns (precedent: KataGo playout-cap randomisation, Wu 2019; Gumbel MuZero, Danihelka et al. 2022); counterfactual labels give unsampled actions a gradient, complementing X23 against starvation; Q / search values never train PPO's GAE critic target | Uniform two-ply; search values as the critic target (biases the advantage — the search-as-teacher lesson) | §5.2; UNDERSTANDING §4.4; `../ai_v6/design_search_teacher.md` |
| 2026-09-30 | X4 readiness | M5 successors and T2 are built; X4 is ready after the M5 switch and the sizing study | "after M5's successors and T2" (both now built) | §9; EXPERIMENT_BACKLOG X4 |
