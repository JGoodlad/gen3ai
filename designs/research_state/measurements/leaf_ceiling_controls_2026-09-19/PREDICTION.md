# PRE-REGISTRATION — IS THE 0.545 `top1|top2` LEVEL THE HEADS, THE LABELS, OR THE GAME?

*Registered 2026-09-19, landed on main BEFORE the first battle of either control. Two controls on
the EXACT banked forks of [`offline_leaf_fit_2026-09-18/`](../offline_leaf_fit_2026-09-18/README.md)
— 5,040 held-out contested CRN forks, 14,570 successor states, 3,795 non-tied pairs. CPU only
(`CUDA_VISIBLE_DEVICES=""`), `nice 15`, rust sim bridge, ports 9700–9799, everything under
`/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_control/`, **nothing under `models/`**. A training arm
holds the GPU and is not touched; 8000/8001 untouched.*

---

## 1. The question, and why it is the one left

Twelve learned win-prob heads read **0.567–0.578** pooled on this instrument, and the pooled number
decomposes (POST-HOC, `pairtype.json`) into **`top1|rand` 0.5938 · `top2|rand` 0.5752 ·
`top1|top2` 0.5450**. The last column is the one a re-ranking search leaf actually consumes, and it
is the worst. Six levers have been varied against it and none moved it, so the remaining accounts
are not about the LEARNER:

* **H — the learned heads are missing something a heuristic has.** A hand-written evaluator would
  then beat 0.545 on the same pairs.
* **N — the level is not the learner's.** A hand evaluator lands inside the head's interval.
* **L — a LABEL-NOISE floor.** Each banked label is ONE rollout from the successor. Averaging the
  label over K fresh dice raises every scorer.
* **G — the GAME.** The policy's own two best moves are genuinely near-tied in win probability, so
  there is almost nothing at this depth for a value function to re-rank; averaging raises nothing
  and the single-rollout order is already close to the truth.

H/N and L/G are independent axes; the read returns a cell.

## 2. 🚨 THE ARITHMETIC THAT MAKES A NAKED BAR VACUOUS — registered before the data

A pairwise-accuracy bar on a SINGLE-ROLLOUT binary label has a ceiling that is NOT 1.0, and it is
low. Let the two siblings' true win probabilities be `p_a > p_b`. A pair is non-tied with
probability `p_a(1−p_b) + p_b(1−p_a)`, and among those an **ORACLE that knows `p_a` and `p_b`
exactly** is right with probability

> `ACC_oracle(K=1) = p_a(1−p_b) / [ p_a(1−p_b) + p_b(1−p_a) ]`

At `p_a = 0.55, p_b = 0.50` that is **0.550**. At a 2-point gap it is **0.520**. **So a scorer
reading 0.545 on `top1|top2` is consistent with it being a near-perfect estimator of a population
whose typical sibling gap is ~4–5 points** — and no amount of learning can raise it. The same holds
for a K-averaged label with a different ceiling: the averaged label is itself noisy
(`sd ≈ sqrt(p(1−p)/K)`), so at K = 8 and a 5-point true gap an oracle reads only ≈ **0.58**.

**Consequence, registered:** the instruction's bar *"head's `top1|top2` above 0.60 under averaged
labels ⇒ branch L"* is read against a MEASURED CEILING at the same K, never against 0.60 in the
abstract (memory: *matched-NOISE control before peer-vs-peer agreement*; rule 25's width clause).
Two ceiling instruments are registered below, and both are reported whatever the headline says.

## 3. What is measured

### 3.1 CONTROL 1 — the hand evaluator, on the EXACT banked successors

`hand_capture.py` replays every banked branch under the SAME common random numbers (the record's own
resolved start seed, both sides greedy, the same substitute action at the same turn) and reads the
board at the first LIVE decision off **both** players' battle objects at once:

| readout | source | what it is |
|---|---|---|
| **ONE-SIDED** | the trainee's own `LiveView` (`battle.strict_view().live`) | our full side + what WE have seen of the opponent — the information the head itself had |
| **OMNISCIENT** | the OPPONENT player's `LiveView` (in ITS view, `live.ours` is its complete team) | the opponent's TRUE bench: species, exact HP, status, alive count |

Both players are in-process, so this is one instant of one battle. Scorers, all evaluated as
`V = the score`, from OUR side's perspective:

| scorer | form |
|---|---|
| **PBRS-1S** | the PRODUCTION potential composition, called on the one-sided `LiveView`: `Φ_mat + Φ_status + Φ_hazard + Φ_boost + Φ_opp_boosts + Φ_roar` (`agents.training.reward_potentials`, imported not re-implemented) — the hand-written evaluator the SHAPED critic regressed |
| **PBRS+BELIEF-1S** | the same plus `Φ_belief` (expected surviving material discounted by the active mon's outspeed-adjusted incoming P(KO)) — the one potential that prices an imminent KO |
| **MAT-1S** | the simple one: `(Σ our hp_frac − Σ opp hp_frac) + 1.25·(our alive − opp alive)`, opponent unrevealed slots counted full-HP-alive |
| **MAT-OMNI** | the same with the opponent's TRUE team HP / alive / status |
| **PBRS-OMNI** | `Φ_mat` and `Φ_status` recomputed on the true opponent side, the public terms unchanged |
| **TYPE-1S / TYPE-OMNI** | MAT plus a type-matchup term over the opponent's REVEALED / TRUE alive team — the only term for which omniscience is not nearly free in gen 3 |
| **arm W's head** | `sigmoid(win_prob_logits)` on the banked successor obs — the published instrument, re-read on the same pairs |

🚨 **The join is PROVEN, not assumed.** The re-run recomputes the successor obs with the banked
builder's own capture discipline and compares it **element-wise** to `succ_<tag>.npy[local]`. A
mismatch means the CRN did not reproduce or the row→successor index is wrong. The rate is reported
per shard and every row carries its own `obs_match`; that is strictly stronger than re-running the
OUTCOME, which is one bit. A sample of forks is additionally run to a TERMINAL with no forfeit and
its outcome compared to the banked one.

### 3.2 CONTROL 2 — rollout-averaged labels

`reroll_labels.py` re-rolls each branch **K = 8** times with FRESH post-divergence dice
(`replay_counterfactual(post_t_seed=s_k)`: the PREFIX keeps the recorded dice, so every re-roll
starts from the IDENTICAL pre-fork state), both sides greedy. **CRN within a re-roll index**: for
re-roll `k` the three branches share one seed, so the k-th `top1` and the k-th `top2` differ in
exactly the action. Seeds are `fresh_seeds(K, salt=battle:inv:seed)` — reproducible, and
`fresh_seeds(4)` is a strict prefix of `fresh_seeds(8)`, so a smaller K is a SUB-SAMPLE of the same
dice. A capped line scores 0.5 (`cf_producer.rollout_outcome_score`, imported).

**The K seeds are independent of the banked label** (which used the realized stream,
`post_t_seed=None`), so the single-vs-averaged order-disagreement rate is honest rather than
deflated by a shared realization.

**The two ceiling instruments, both registered:**

1. **SPLIT-HALF self-agreement.** Label A = mean of re-rolls {1,3,5,7}, label B = mean of
   {2,4,6,8}. On a pair, the rate at which `sign(A_a − A_b)` equals `sign(B_a − B_b)` is what a
   K = 4 label can agree with ITSELF about. **No scorer read against a K = 4 label can exceed it**,
   and it is measured on the same forks with the same dice budget.
2. **The moment-estimated true-gap distribution.** `Var(observed gap) − E[per-branch noise var]`
   estimates `Var(p_a − p_b)`; the per-branch noise variance is estimated from the K rollouts
   themselves. From the recovered gap distribution, `ACC_oracle(K)` of §2 is integrated at K = 1 and
   K = 8 and printed beside every measured number.

### 3.3 The read

The three PAIR-TYPE columns for every scorer, under BOTH label regimes, bootstrapped over FORKS
(2,000 resamples, `refit.boot_ci`), on the identical pairs; plus the paired Δ of each hand evaluator
against arm W's head (`refit.paired_delta`). Rule 25's indexing clause is executed by
[`../exploiter_discrimination_2026-09-18/verify_indexing.py`](../exploiter_discrimination_2026-09-18/verify_indexing.py),
invoked by path.

## 4. THE REGISTERED BARS

| bar | statement |
|---|---|
| **BAR 1 — branch H** | the BEST hand evaluator's `top1|top2` CI **lower bound > 0.5450** (arm W's head's point estimate), i.e. above it by CI ⇒ **H**; a CI containing 0.5450 ⇒ **N** |
| **BAR 2 — branch L (literal)** | arm W's head's `top1|top2` against the K = 8 averaged label **> 0.60** ⇒ **L** |
| **BAR 3 — branch L (matched)** | the head's averaged-label `top1|top2` is **below the split-half self-agreement ceiling by a paired CI clear of zero** ⇒ headroom exists that the head is not using ⇒ **L**; if it sits AT the ceiling ⇒ **G** |
| **BAR 4 — branch G** | every scorer's `top1|top2` stays **≤ 0.58** under averaged labels AND the single-vs-averaged order agreement is **high** (≥ 0.85 on pairs non-tied under both) ⇒ **G** |
| **BAR 5 — the instrument** | `obs_match` ≥ 99 % of captured branches, and the to-terminal outcome check ≥ 95 % agreement. **Below either, the read is INCONCLUSIVE and reported as such** |
| **BAR 6 — the mirror battery is NOT RUN**, on instruction |

## 5. THE PREDICTIONS

| # | prediction |
|---|---|
| **P1** | **The cell is N × G.** No hand evaluator clears BAR 1, and the head's averaged-label `top1|top2` does not clear 0.60 |
| **P2** | arm W's head reproduces **0.5450 ± 0.002** on `top1|top2` and **0.5723 ± 0.002** pooled on the banked single labels — the same forks, the same pairs, so this is an identity check, not a measurement |
| **P3** | the best hand evaluator reads **0.50–0.57** on `top1|top2`, i.e. at or BELOW the head. A one-ply material swing is exactly what a 75M-step value head already contains |
| **P4** | the hand evaluators do markedly BETTER on `top1|rand` than on `top1|top2` — the same MIXTURE the head shows. Registered range for the best hand evaluator on `top1|rand`: **0.55–0.65** |
| **P5** | 🚨 **OMNISCIENCE IS NEARLY FREE IN GEN 3 FOR THIS EVALUATOR FAMILY and I register it as a near-null in advance.** An UNREVEALED opponent mon has never been on the field, so it is at full HP, alive and unstatused — which is exactly what the one-sided `Φ_mat` already assumes. The omniscient and one-sided MAT scores will differ only by HP-percent rounding: **mean \|Δ\| < 0.1 HP-units, and `top1|top2` accuracies within 0.01**. The TYPE variants are the only pair where omniscience can move anything |
| **P6** | the K = 8 averaged label RAISES the head's `top1|top2` — registered range **0.55–0.62** — but **does not clear 0.60**, and lands **at or just below** the split-half ceiling |
| **P7** | the SPLIT-HALF self-agreement ceiling on `top1|top2` is **0.55–0.70** at K = 4. If it is near 0.60, then 0.60 is not a bar a scorer can clear at this dice budget and BAR 2 is vacuous — which is the reason §2 was registered |
| **P8** | the moment-estimated typical sibling gap `E|p_a − p_b|` on `top1|top2` is **0.02–0.10**, and the derived `ACC_oracle(K=1)` for that population is **0.52–0.58** — i.e. **the head at 0.545 is at or near the single-rollout oracle**, which IS branch G stated quantitatively |
| **P9** | the single-vs-averaged ORDER disagreement rate on `top1|top2` pairs is **HIGH — 0.30–0.45** (a single rollout is a very noisy read of a 5-point gap). This is the label-noise rate the instruction asks for, and a high value is NOT by itself branch L: it is equally what branch G predicts, and only the ceiling instruments separate them |
| **P10** | `obs_match` ≥ 99.5 % and the to-terminal outcome agreement ≥ 98 % — the fork builder is deterministic under CRN and the 2026-09-18 read measured 462/462 |
| **P11** | the `rand` branch re-rolls cap more often than `top1`'s (the 2026-09-18 read measured 37× on the realized stream); registered as a DIRECTION, not a magnitude |
| **P12** | the mean K = 8 label for `top1` exceeds that for `top2` on average (the policy's own ordering is right on average), by **0.005–0.05** — a direct measurement of what the top-2 gap is WORTH, which nothing in this campaign has ever measured |

## 6. Declared deviations, and what could make this read wrong

* **The sample for control 2 is ~750–1,000 forks, not all 5,040** — K = 8 × 3 branches is 24
  rollouts per fork and the box carries a training arm. Forks are drawn by a SEEDED SHUFFLE over
  forks whose three branches all have a successor, and the selection **does not read any outcome**.
  The loop is FORK-OUTER / K-INNER, so a wall-clock stop yields FEWER FORKS AT FULL K rather than
  all forks at a ragged K; a ragged K would weight forks unequally in the mean without saying so.
  The instruction's "stratified across the three columns" is satisfied by construction — every
  sampled fork contributes to all three pair types.
* **The `top1|top2` cell has ~1,167 non-tied pairs in the full set and proportionally fewer in the
  control-2 subset.** Intervals there are ~1.7× the pooled ones, and the subset's ~2.3× wider
  again. Any "not detected" from control 2 is reported WITH its width.
* **One frozen policy.** Arm W. Nothing here says a differently-trained trunk behaves the same.
* **The hand evaluators are NOT a search bot.** No damage calculation, no lookahead, no move
  selection — they are static board scores of the kind PBRS uses. A Foul-Play-style *searching*
  heuristic is a different and much more expensive object and is explicitly out of scope; a null
  here does not refute that one.
* **`Φ_belief` is the only term that reads the incoming-damage model**, and it is the one hand term
  with real machinery behind it. If any hand scorer beats the head it will be that one.
* Probe runs of both scripts (≤ 4 forks, ≤ 18 rollouts) were executed BEFORE this registration to
  validate the capture and the reseed path; their outputs are under `.../leaf_control/probe*` and
  are not part of any reported number.

## 7. What each outcome would mean

| cell | reading |
|---|---|
| **N × G** (predicted) | the leaf question is ANSWERED: at one ply, the policy's two best moves are near-tied in truth, a value function has ~nothing to re-rank, and the campaign's 0.545 was never a statement about learning. The road is not a better leaf but a DEEPER or WIDER one — more plies, or a Q head that amortizes the ply instead of re-deriving it |
| **N × L** | the level is a dice budget, not a representation: K-rollout leaves or K-averaged targets are the road, and the rust search driver is what it was built for |
| **H × ·** | a heuristic has something the twelve heads do not, and the finding is what it is — the largest single surprise available here |
| INCONCLUSIVE | BAR 5 fails; the join to the banked successors is not proven and no number leaves this directory |
