# Training — THE FORK ARM (`--fork-fraction`, `gen3_fork_v1`)

**Owner doc.** `src/agents/training/CLAUDE.md` keeps the heading and the rules an agent must know
before touching the arm; everything else is here, and this file carries the same always-current
obligation as the leaf — update it in the same pass as the code.

---

## 1. WHY — one measurement, and the conclusion it forces

`designs/research_state/measurements/paired_refit_discrimination_2026-09-14/` measured, for the
first time, the property a one-ply search consumes: does `sign(V(s'_a) − V(s'_b))` agree with which
of two successors **one move apart** actually wins, on the **same dice**?

| row | pairwise accuracy | read |
|---|---|---|
| the promoted win-prob critic (`ctrl10M@10M`) | **0.5169 [0.4800, 0.5524]** | **a coin.** The CI straddles 0.50 |
| a FROZEN trunk, only `WinProbHead`'s four tensors refit, plain BCE on counterfactual successors | **0.6032 [0.5690, 0.6374]** | **+0.0863 [+0.0384, +0.1347], DETECTED** |
| the same rows plus a pairwise RANKING term (coef 0.1 / 0.3 / 1.0) | −0.0107 [−0.0249, +0.0028] | **NOT DETECTED**, negative on points |

Pairwise accuracy is a **rank** statistic — invariant to any monotone recalibration — so a frozen
trunk reaching 0.60 means the ordering was already inside `value_pooled` and the on-policy PPO
stream simply never asked the head for it. **The loss form is not the lever; the DATA is.**

The data the refit used does not exist in a training rollout: a rollout visits exactly **one**
successor per decision. This arm manufactures it. At a contested decision the battle is forked, the
branches are played to a terminal by the current policy, and their transitions go into the **same
PPO buffer**. No new loss, no auxiliary objective, and **no ranking term** — closed as a lever at
this depth.

⚠️ Part C of the same record plugged the refit head in as a SEARCH LEAF and it did **not** pay
(−0.0125 on both rows, NOT DETECTED). That is why the arm puts forks into TRAINING rather than
shipping the offline refit: the producer is the thing to change.

---

## 2. THE THREE BRANCHES, and why the random one is the point

Three facts about the policy fell out of 5,076 CRN forks **before any head was fitted**:

* top-1 and top-2 are **outcome-interchangeable** on identical dice — **0.7082 vs 0.7078**, a gap
  of 0.0004. A two-branch fork spends its whole simulation budget on a distinction the outcome
  cannot see.
* a uniformly random legal alternative wins **0.6795** — throwing the decision away costs
  **2.9 pp**.
* in **4.5 % [3.99, 5.16]** of forks that random alternative beat **BOTH** policy candidates.

That 4.5 % is the **blind-spot rate** and it is where the new information is: it is the only branch
whose outcome the policy's own ordering did not predict. `--fork-branches 2` is the CONTROL that
isolates it, and `fork/random_wins` is the meter that says whether the third branch earned its
simulation.

🚨 **79.7 % of branch PAIRS are TIED** (same outcome) — the structural tax on any terminal-outcome
sibling signal. `fork/tie_rate` publishes the per-fork version every rollout; a run whose tie rate
climbs is buying less, whatever its fork count says.

---

## 3. THE SELECTOR — `gen3_fork_select_v1`, inherited verbatim

`agents/training/fork_arm.py`. A decision is forkable when **all** of:

| condition | source |
|---|---|
| `win_mask == 1` — the episode TERMINATED in this buffer (which is also what makes the state replayable: the `__RECON__` record is written at episode END) | `win_prob_rollout.eligible_mask` |
| a per-decision reconstruction HANDLE was captured (`turns >= 0`) | `gen3_winprob_rollout_target_v1`, shared |
| `MIN_LABELABLE_TURN <= turn <= 40` | `cf_producer` / the offline builder's `--max-turn` |
| a MOVE ROUND with `>= 3` legal actions | `cf_producer_sampler.is_move_round`, `--min-legal 3` |

and then the CONTESTED test: the policy's **top-2 masked-logit gap** is at or below the
`--fork-contested-gap` **QUANTILE of this rollout's own candidate pool** (0.40 — the offline
builder's `--gap-quantile` default, taken verbatim so the arm forks the population the 0.5169
baseline was measured on). Ties are broken tightest-gap-first.

🚨 **A QUANTILE, not an absolute gap.** The gap's SCALE moves as a run's logits sharpen, so a fixed
threshold would fork 40 % of decisions early and ~0 % late — the treatment would anneal itself off
in silence.

⚠️ **On a FRESH network the quantile degenerates, and that is honest rather than broken.** An
untrained policy's masked logits are near-identical, so most gaps are ~0, the 0.40 quantile lands at
~0 and the `<=` admits the whole pool. Measured in the smoke: `gap_threshold` 0.0 on the first
rollout and 0.00195 on the second, with every pooled candidate selected. The fork COUNT is then set
by `--fork-max-per-battle` and the fraction, not by contestedness — which is the right behaviour
(every decision really is contested at that point) but means `fork/gap_threshold` is the tag to read
before quoting an early rollout's population.

🚨 **THE SELECTOR DOES NOT READ V, and `--fork-contested-absv` is OFF by default.** The offline
builder's own words: *"|V−0.5| is RECORDED per decision but never selects — selecting on V would
make the held-out read partly a measurement of the selector."* The arm's registered endpoint is a
held-out pairwise accuracy **against that baseline**, so turning the `absv` arm on forfeits the
comparison. The knob exists because the registration named it; it is off.

**The pool.** Scoring every eligible row would mean a policy forward over most of a 98,304-row
buffer. Instead a candidate POOL of `n_forks / quantile` rows (floor 32) is drawn UNIFORMLY over
episode SLICES — at most `--fork-max-per-battle` (default **1**) per slice, because two forks of one
game share a prefix, an opponent and a team draw and are far more correlated than two forks of
different games. The quantile is taken over that pool.

**The random branch** is drawn from `legal \ {top1, top2}` with a DECISION-KEYED sha256 — the
offline builder's draw, verbatim, and for its reason: a probability-ordered draw would spend the
branch on what the policy already covers.

---

## 4. COMMON RANDOM NUMBERS — `--fork-crn` (default `dice_and_draws`)

🚨 **A concrete, testable account of the `cf_q_labels` null.** That factory pairs the sim **dice**
and leaves both sides sampling at temperature 1.0 (its own recorded caveat); the offline forks were
**greedy on both sides**, which is the only reason 104 determinism re-runs came back identical. *A
label factory that pairs the dice but not the policy draws may be teaching the head noise.*

The fork arm cannot take the offline dataset's way out — its branches must be played in the ecology
the training actor plays in (temperature 1.0 on both sides), because the transitions enter the PPO
buffer. So the draws are **paired** instead of removed.

| setting | what the branches share |
|---|---|
| `dice` | ONE sim seed for the whole line (`replay_counterfactual`'s default, `post_t_seed=None`). The `cf_q_labels` regime, kept as the CONTROL |
| **`dice_and_draws`** (default) | that, **plus** both players' policy sampling streams seeded IDENTICALLY per branch — `RLPlayer`'s `policy_seed` (`gen3_policy_sample_rng_v1`) — so the k-th decision of every branch consumes the SAME uniform |

Three properties make the draw-pairing exact rather than approximate, and each is closed rather
than noted:

* **the prefix consumes nothing** — `install_scripted_prefix` returns a passthrough order for every
  scripted decision and never reaches `_predict_best_action`, so two branches start their streams
  aligned AT the fork, not at turn 1.
* **one uniform per decision, whatever the state** — `torch.multinomial(cat.probs, 1, True, gen)`
  draws exactly one sample from an 11-way categorical every call, so the streams cannot slip by a
  state-dependent amount.
* **the two SIDES get different seeds** — sharing one would couple the trainee's coin to the
  opponent's, a correlation that is not in the training ecology and would be a second confound
  smuggled in under a fix for the first.

**VERIFIED, not asserted.** `fork_crn_sim_test.py` (`sim`) plays real bridge battles in three cells:
same sim seed + same streams ⇒ **byte-identical protocol**; same sim seed + different streams ⇒ the
lines diverge (the `cf_q_labels` shape reproduced); different sim seed ⇒ diverge (the control that
proves the comparison can see a change).

---

## 5. THE MASK RULE — one rule, and it is UNIFORM

🚨 **The FORK STEP is excluded from the clipped policy term for EVERY branch, the top-2 included.**

The alternative was to mask only `rand` (whose action the policy actively down-ranked) and let the
clip handle the top-2 (whose true log-prob is recorded). That is rejected because **an exclusion
criterion that depends on WHICH branch a row came from re-weights the policy gradient by the branch
mix**: masking only `rand` would leave `top1` and `top2` as the only fork-step rows in the policy
term, i.e. would silently up-weight the policy's own two candidates at exactly the contested states
the arm selects for — an on-policy re-weighting shipped under a flag whose declared job is to buy
VALUE data. The uniform rule costs almost nothing: at 3 branches it removes 3 rows per fork from a
term that keeps every one of the ~25 post-fork rows the same fork contributes.

The fork step stays **fully in the value terms** — its return is its own branch's — which is the
whole point of playing the branch.

**Carrier:** the `fork_pg_m` obs Dict key, a per-row multiplier that survives
`RolloutBuffer.get()`'s shuffle aligned to its own row (the mechanism `win_row_w` uses). Declared
ONLY when the flag is on. The env emits a 1.0 placeholder on every COLLECTED row.

🚨 **The masked term is RENORMALISED, never just zeroed:**

```
policy_loss = -( min(pl1, pl2) * m ).sum() / m.sum().clamp(min=1)
```

A masked `.mean()` over the full row count would shrink the policy term by the masked fraction —
i.e. silently lower the effective policy learning rate by a number that moves with the fork rate.

---

## 6. THE PREFIX IS COUNTED ONCE

A branch's rows begin **AT** the fork step. The turns before it are the parent episode's, they are
already in the buffer as the parent's own rows, and they are identical across the branches by
construction (that is what CRN means). Injecting them per branch would put the same transition in
the objective `branches` times and multiply the prefix's weight by the fork rate.

The fork STATE itself does appear once per branch, with a **different action** each time. That is
the exploring start, not a duplicate: those rows differ in the action, the return and the successor
— and none of them contributes a policy gradient (§5).

---

## 7. THE VALUE TARGET — ordinary GAE, plain BCE, NO ranking term

A branch is a complete episode from the fork step to a terminal, so its advantages and returns are
`RolloutBuffer.compute_returns_and_advantage`'s recursion with no bootstrap
(`fork_buffer.gae`, cross-checked against SB3's own arrays in `fork_buffer_test`). `win_target` is
the branch's own outcome bit and `win_mask` is 1.

🚨 **Why the arm REFUSES any critic but `winprob`.** Under `--critic winprob` the reward stream is
the **terminal win indicator alone** (`--terminal-indicator`, `--victory-value 1.0`,
`--no-hand-shaping`), so a branch's ENTIRE reward sequence is reconstructible from its outcome bit —
which is the only reason `fork_buffer.branch_rewards` can build one outside the env. Under `shaped`
a per-turn reward is a PBRS/bias composition the env's `RewardManager` folds from a `TurnDelta` that
no branch has, and the injected rows would silently carry zero reward, i.e. would teach the critic
that a third of the buffer is inert.

**No ranking term**, ever, at this depth: −0.0107 [−0.0249, +0.0028], NOT DETECTED, negative on
points at every coefficient tested.

---

## 8. WHERE THE ROWS COME FROM, and what the parent does

| piece | file |
|---|---|
| selector, meters, cost model, the flag constants | `agents/training/fork_arm.py` |
| the CRN seed derivation | `agents/training/fork_crn.py` |
| the child that PLAYS the branches | `agents/training/fork_worker.py` |
| the parent fan-out + the batched scoring + the row assembly | `agents/training/fork_driver.py` |
| the fill table, the GAE, `ForkRolloutBuffer` | `agents/training/fork_buffer.py` |
| the rollout hook | `agents/training/fork_callback.py` |
| the gates | `fork_arm_test.py`, `fork_buffer_test.py`, `fork_crn_test.py`, `fork_callback_test.py`, `fork_flags_test.py`, `fork_crn_sim_test.py` (`sim`) |

**Values and log-probs are computed in the PARENT**, by the live policy, in one batched forward.
Not an optimisation: the parent's weights ARE the weights that collected the buffer, so
`old_log_prob` is exactly the behaviour policy's and the PPO ratio is 1.0 at the first epoch, which
is what the clipped objective assumes. A log-prob from a child's re-loaded snapshot would be that
number only up to serialisation, and an importance ratio does not survive "only up to".

**The substitute is handed to `replay_counterfactual` as a CALLABLE** `(player, battle) -> str`
(`gen3_fork_v1` extends `install_scripted_prefix` to accept one, default-identical). A buffer row
knows its action as an INDEX and the index→choice mapping is a function of the LIVE legal set;
resolving it through the player's own `action_to_order` at the divergence decision is the only way
to be sure the branch sends what the mask said it could. A callable that raises is deliberately NOT
swallowed — a swallowed failure would fall through to replaying the RECORDED move and silently
produce a line that is not the counterfactual the caller asked for.

**Row 0 is the fork step** (its obs comes from the scripted path's `on_scripted_decision` hook, so
it describes the fork state and not the state after it); **row 1 is the successor `s'`** — the state
a one-ply search would score and the state the pairwise endpoint is about. A branch that ends AT the
fork turn has no successor and reports one row rather than a fabricated one.

**A CAPPED branch is DROPPED.** A 250-turn cap is decided by SEAT and not by the position
(`counterfactual._battle_outcome`), so it never enters a sibling pair and never scores 0.5.

### The fill table, and why an unknown key REFUSES

A branch is played by two `RLPlayer`s, not by a `Gen3Env`, so the capture yields `observation` and
`action_mask` and nothing else — while the buffer's obs Dict may carry a dozen flag-gated LABEL
keys. `fork_buffer.FILL` declares, per key, what an injected row honestly holds; a key that is not
in the table makes the arm REFUSE at setup, naming the key. The alternative is a heuristic
("anything ending in `_mask` is zero"), and a heuristic keeps working — silently and wrongly — the
first time someone adds a key it happens to match.

🚨 **The rule for a label a branch cannot supply is NOT SCORED, never a guess.** Every privileged
belief label is a fact about the opponent's team read from `battle2` inside the env; a branch has no
env, so those rows are masked out of their losses. The cost is supervision on those rows; the
alternative cost is a belief head trained on fiction.

Four flags are **REFUSED alongside the arm** rather than filled, each by a `combination_checks` row:
`--value-true-team` (the extractor's value route RAISES on a missing `opp_true_team`, and a zero
block would be a fabricated privileged input), `--win-prob-dense-aux` (its targets are the
episode's end-of-battle facts, which for a branch this process never held), `--win-prob-strata-weight`
(it prices rows by `win_margin`, which the env's reward manager computes — every injected row would
carry the 0.0 fill and land in one stratum, making the delivered strata dose a function of the fork
rate), and anything declaring `defensive_opportunity` / `bait_opportunity` / `distill_mask`.

### The buffer

`ForkRolloutBuffer` is a MIXIN over whichever buffer the algorithm selected (today
`MaskableDictRolloutBuffer`). Raising `buffer_size` to make room would resize the NEXT rollout's
arrays; padding to a whole multiple of `n_envs` would put fabricated transitions in the objective.
So the rows live beside the collected ones in their own flat arrays and `get()` yields minibatches
over the concatenation. `reset()` drops them, an empty fork set makes `get()` upstream's generator
**called**, and a shape the concatenation cannot take is a REFUSAL at injection rather than a torch
stack error three frames into the loss.

---

## 9. THE ECOLOGY APPROXIMATION — the arm's largest declared caveat

A training `__RECON__` record carries the resolved seed, both packed teams and the committed
choices, and **nothing that says which policy sat on the other side**. So a branch is played against
a **self-like** opponent (the current snapshot), not the episode's real one — right for the ~90 %
self-play share of the training mixture and biased for the rest, the same direction
`win_prob_rollout` documents. `fork/branch_share` prices how much of the buffer that population
substitution occupies and `fork/bot_share` names how much of it replaced a BOT. An injected row's
`opp_class` is labelled **POOL** for that reason: the class tag says who the row was played against,
and it was not the parent's opponent.

---

## 10. COST

```
fork sim steps  ≈  forks × branches × mean_remaining_decisions
```

against the `n_steps × n_envs` decisions the trainee itself makes. `fork/sim_steps_share` computes
it from the MEASURED per-branch decision counts every rollout — a lever whose cost is not on the
dashboard is a lever that gets left on.

At `--fork-fraction 0.02` with 3 branches on a production buffer (2,048 × 48 = 98,304 decisions)
the fraction asks for **1,966 forks**; at the smoke's measured ~35 live decisions per branch that
is `1966 × 3 × 35 / 98,304 ≈ **2.1×**` a plain run's own simulation.

🚨 **BUT THE ROW BUDGET BINDS FIRST, AND ABOVE ~0.008 THE FRACTION IS INERT.** The injection is
capped at one buffer's worth of rows, and a fork contributes ~125 of them (3 branches × ~42
decisions, measured), so the budget allows **~790 forks** — well below the 1,966 the fraction asks
for. The delivered numbers at `--fork-fraction 0.02` are therefore ~790 forks, `fork/branch_share`
≈ 0.5 and `fork/sim_steps_share` ≈ **0.85×**, not 2.1×. Read **`fork/requested` against
`fork/forks`** and **`fork/row_budget` against `fork/rows_per_fork`**: a gap between the first pair
means the ceiling is the budget's and raising the fraction will change nothing but the first
rollout's wasted simulation. Raising the ceiling means raising `ROW_BUDGET_MULTIPLE`, and that is a
MEMORY decision — a branch row is a full observation, so one extra buffer's worth is ~1 GB at the
production shape on a box that is also carrying a GPU run.

Three bounds keep a fat-fingered fraction from wedging a run, and all three are PUBLISHED rather
than silent:

* `MAX_FORKS_PER_ROLLOUT` — a hard cap on the fork count.
* `ROW_BUDGET_MULTIPLE = 1.0` — the injected rows may at most DOUBLE the buffer. A branch row is a
  full observation (2,501 float32 plus every label key), so an unbounded injection is an unbounded
  memory bill on a box that is also carrying a GPU run. Over budget, forks are dropped **WHOLE** —
  never partially, because a half-injected fork would put one branch of a sibling pair in the
  objective and not the other — and `fork/dropped_forks` counts them.
* 🚨 **the ASK is bounded by the MEASURED rows-per-fork**, because **a fork dropped at the row
  budget has already been PLAYED**. The branches are simulated before their rows can be counted, so
  the row cap alone bounds MEMORY and not COST: a run at a fraction the budget cannot carry would
  pay the full simulation bill for rows it then throws away (measured in the smoke: **30 of 85
  forks dropped in one rollout**). The previous rollout's realised `fork/rows_per_fork` therefore
  caps this one's ask. MEASURED and not modelled — a branch's length is the run's own episode
  length, which moves over a run and is exactly what a banked constant would get wrong. The first
  rollout pays the drop, once. The drop path stays as the BACKSTOP: an estimate must never be the
  only thing between the buffer and an unbounded injection.

---

## 11. THE `fork/` TB FAMILY

The prefix is dropped in the table. **An absent `fork/*` family means `--fork-fraction 0.0`** (or
the arm disabled itself, which announces itself once) and nothing else.

| scalar | what it says |
|---|---|
| **`rate`** | **forks per BATTLE** — forks / episodes started in this buffer. (`fraction` is the per-DECISION number and is published beside it) |
| **`branch_share`** | the share of the rows the PPO objective sees that came from a BRANCH rather than from the trainee's own play. The honest headline for the population substitution of §9 |
| **`tie_rate`** | share of forks whose branches ALL share one outcome — the structural tax, measured. The offline PAIR version read 79.7 % |
| **`random_wins`** | **THE BLIND-SPOT RATE**: share of 3-branch forks where the random branch beat BOTH policy candidates. Offline: 4.5 % [3.99, 5.16]. At 0 the third branch is not earning its simulation |
| **`pairwise_acc`** | **THE DIRECT DISCRIMINATION METRIC** — `sign(V(s'_a) − V(s'_b))` vs the outcome on non-tied pairs. The same quantity the offline baseline read at 0.5169. ⚠️ IN-SAMPLE (these are the states the arm trained on); the registered endpoint is a HELD-OUT read on fresh forks. A run whose `pairwise_acc` never leaves 0.5 is a run whose treatment is not landing |
| **`pairwise_pairs`** | how many non-tied pairs that mean is over — read it before reading the mean |
| **`sim_steps_share`** | **THE COST.** Branch decisions / the trainee's own collection decisions |
| `fraction` / `branches` / `crn_draws` | the configuration, echoed so a plot is self-describing (`crn_draws` 1 ⇒ `dice_and_draws`) |
| `eligible` / `pool` / `gap_threshold` | the selector's own state: how many rows were forkable, how many were scored, and where the quantile landed |
| `requested` / `forks` / `failed` / `records_missing` | asked for, completed, lost, and lost specifically to a pruned `cf_records` ring |
| `injected_rows` / `masked_rows` / `dropped_forks` | rows added, fork steps masked out of the policy term, forks dropped whole at the row budget |
| `rows_per_fork` / `row_budget` | the MEASURED mean rows a fork contributes, and the budget it is bounded by. `rows_per_fork` is what caps the NEXT rollout's ask; a rising one on a lengthening run is the tell that the fork count is about to fall |
| `seconds` / `worker_failures` / `branches_capped_frac` | the STALL this added to the training loop, killed children, and the share of branches that hit the 250-turn cap (those are dropped) |
| `bot_share` | share of forks whose PARENT episode faced a bot — i.e. how much of the treatment replaced a weaker opponent with a self-like one |

---

## 12. OPERATING IT

**Both requirements are refused at launch, not discovered at runtime:** `--critic winprob` (§7) and
`--cf-records` (a fork replays its episode, and the replayable record lives in the ring that flag
switches on).

🚨 **RAISE `--cf-records-keep`.** The ring is pruned GLOBALLY to the newest N while a production
rollout finishes ~2,400 episodes, so at the default 512 the forks that DO resolve are the rollout's
LATE ones — a **selection bias**, not just a shortfall. `fork/records_missing` is the tell; the
refusal text names the flag. Set it above `n_envs × n_steps / mean_episode_length`.

The registered launch is `ctrl10M`'s argv plus `--fork-fraction 0.02 --fork-branches 3 --seed 1001
--cf-records --cf-records-keep 4096`, i.e. the arm on top of the exact configuration whose 0.5169
baseline it is being read against.

**Registered endpoints, in order** (from the measurement's orchestrator note):

1. **held-out pairwise accuracy** on FRESH contested forks vs `ctrl10M`'s 0.517 — the bar is that
   the CI clears **0.60**, the offline refit's level, since a head trained on-stream should at least
   match an offline refit.
2. the mirror battery at **≥ 400 pairs** with a CONTEMPORANEOUS control. 🚨 The 100-pair L2 cell is
   RETIRED: its own same-configuration replicate moved **+0.040 [−0.010, +0.081]** between windows,
   which is wider than every effect it was being used to read.
3. `cond.opp_class_auc.t4_10` as the guard.

**What to watch in the first hour:** `fork/rate` > 0 and `fork/records_missing` near 0 (the ring is
big enough); `fork/sim_steps_share` at the size the cost model predicted; `fork/tie_rate` (how much
of the treatment carries any ordering at all); `fork/pairwise_acc` with `fork/pairwise_pairs` beside
it. **Kill conditions are the run's usual ones** — this arm adds a large simulation cost and a
population substitution, so `signal/stall_rate` and the entropy trace are read as they always are.

---

## 13. THE SMOKE, and what it measured

`--debug --steps 10000 --fork-fraction 0.05 --cf-records --cf-records-keep 4096` on CPU, one
`DummyVecEnv` at `n_steps` 2048, `--use-bridge rust`. **Training complete**, 0 `worker_failures`,
0 `records_missing`.

| rollout | `rate` | `branch_share` | `tie_rate` | `random_wins` | `pairwise_acc` (pairs) | `sim_steps_share` | `dropped_forks` | `rows_per_fork` |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.98 | 0.500 | 0.567 | 0.136 | 0.583 (12) | **7.64** | **48** | 125 |
| 2 | 0.25 | 0.492 | 0.750 | 0.000 | 0.500 (8) | **2.02** | 2 | 132 |
| 3 | 0.21 | 0.472 | 0.467 | 0.200 | 0.250 (16) | **1.65** | 0 | 127 |

🚨 **The cost meter and its bound, both working.** The first rollout asks for more forks than the row
budget can carry and PAYS for them (48 dropped, 7.64× the trainee's own simulation); the measured
`rows_per_fork` then bounds every ask after it and the realised share settles at **1.65–2.0×** — the
size the cost model predicts for a production launch at `--fork-fraction 0.02` with 3 branches
(1,966 forks × 3 × ~35 decisions / 98,304 ≈ **2.1×**). `branch_share` sits at the 0.5 the row budget
allows, i.e. the injection is doubling the buffer, which is the arm's designed maximum.

⚠️ **`pairwise_acc` here is noise and must be read as such** — 8–16 pairs a rollout on a network
10k steps old. It is the METER that is being smoked, not the quantity. 🚨 `--debug` exercises a STRICTLY SMALLER surface than a real launch
(no forkserver, no compile preload, no warm start), so the smoke proves the arm FIRES and costs what
the model says — it does not prove the launch layer.
