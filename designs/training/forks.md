# Training — THE FORK ARM (`--fork-fraction`, `gen3_fork_v1`)

**Owner doc.** `src/agents/training/CLAUDE.md` keeps the heading and the rules an agent must know
before touching the arm; everything else is here, and this file carries the same always-current
obligation as the leaf — update it in the same pass as the code.

> **STATUS (deletion pass L5, 2026-10-02).** The PYTHON-core fork arm described in §§1–13 is
> **DELETED** (L4 made it unreachable; L5 removed the code: `fork_callback.py`, `fork_driver.py`,
> `fork_worker.py`, `fork_crn.py`, the `ForkRolloutBuffer` and the branch-row builder in `fork_buffer.py`,
> `cf_records.py`, the decision-time handle capture in the wrapper / env factory / async collector /
> `WinProbLabelCallback`, and `compile_trainer.eager_extractor`; the callable-substitute seam in
> `utils/bridge/counterfactual.py` went with it). What the Rust port imports is KEPT: `fork_arm.py`
> (selector, branch actions, meters) and `fork_buffer.FILL` / `unfillable_keys` / `refusal_text`. The
> `--fork-fraction > 0` on `--env-core python` refusal row (`fork_python_core_unavailable`) stays until
> the Python core itself is deleted, so the flag cannot be a silent no-op there. Before L5 the arm was
> **UNREACHABLE**: it replayed its episode from the `<run>/cf_records/` ring, the ring and its flag
> (`cf-records`) were deleted with the cf training half, and `--fork-fraction > 0` on
> `--env-core python` is refused (combination row `fork_python_core_unavailable`). Its code is deleted
> in unit L5 (done). The **live arm is the Rust fork port (§14)**, which replays the core's own finished input
> log and needs no ring. §§1–13 stay as the record of what was measured and built (the measurement in
> §1, the draw pairing, the cost model and the endpoints all carry to the Rust port); wherever they name
> the ring, `fork/records_missing` or the ring's cap, read it as history of the Python arm.

---

## 1. WHY — one measurement, and the conclusion it forces

`designs/research_state/measurements/paired_refit_discrimination_2026-09-14/` measured, for the
first time, the property a one-ply search consumes: does `sign(V(s'_a) − V(s'_b))` agree with which
of two successors **one move apart** actually wins, on the **same dice**?

| row | pairwise accuracy | read |
|---|---|---|
| the promoted win-prob critic (`ctrl10M@10M`) | ~~0.5169 [0.4800, 0.5524]~~ **0.5872 [0.5526, 0.6225]** (CORRECTED 2026-09-16: the 0.5169 was an indexing artifact in `refit.py`; `fork_arm_read_2026-09-16/`) | **a modest ranker, not a coin.** |
| a FROZEN trunk, only `WinProbHead`'s four tensors refit, plain BCE on counterfactual successors | **0.6032 [0.5690, 0.6374]** | vs the CORRECTED baseline **+0.0160 [+0.0000, +0.0313], NOT DETECTED** |
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


> **STATUS 2026-09-16 — the arm ran and read NOT DETECTED** (`ai_v13_03_fork`, `fork_arm_read_2026-09-16/`): 0.5754 vs 0.5757 on 10,080 fresh forks, paired Δ −0.0003 [−0.0148, +0.0136]; guard within floor; 400-pair battery no dividend. The delivered dose was ~1,060 forks/rollout with zero drops, governed by `fork/gap_threshold` (→1.99), not the row budget — the §on-cost prediction that the budget binds at ~790 was WRONG. Forks move the state distribution (+31 % non-tied pairs) without moving the head. **The line is CLOSED at this depth; `--fork-fraction` stays 0 = OFF.**

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
| `win_mask == 1` — the episode TERMINATED in this buffer (which is also what makes the state replayable: the `__RECON__` record is written at episode END) | `fork_arm.eligible_mask` |
| a per-decision reconstruction HANDLE was captured (`turns >= 0`) | the per-decision handle capture `WinProbLabelCallback` performs for the fork arm |
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

**VERIFIED, not asserted** (the Python arm's gate, `fork_crn_sim_test.py`, was deleted in L5; the Rust port's is `rust_rollout/fork_crn_integration_test.py`, §14.8). It played real bridge battles in three cells:
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
`RolloutBuffer.get()`'s shuffle aligned to its own row (the mechanism the deleted `win_row_w` anchor weight used). Declared
ONLY when the flag is on. The env emits a 1.0 placeholder on every COLLECTED row.
Under `--compile-trainer` (torch 2.8) the key is part of the compile region R1's DECLARED signature
from startup (it is in the env's obs space; the injected rows make a ragged last micro-batch, which
runs eager by declaration) — verified 2026-10-01 by `r1_declared_levers_test`'s `fork` row
(`compile_flags.md` "R1's DECLARED LEVERS").

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
(the Python arm's `fork_buffer.gae`, cross-checked against SB3's own arrays — deleted in L5; the Rust pass uses `store.game_gae`, §14.5). `win_target` is
the branch's own outcome bit and `win_mask` is 1.

🚨 **Why the arm REFUSES any critic but `winprob`.** Under `--critic winprob` the reward stream is
the **terminal win indicator alone** (`--terminal-indicator`, `--victory-value 1.0`,
the terminal-only reward), so a branch's ENTIRE reward sequence is reconstructible from its outcome bit —
which is the only reason a branch's reward can be built outside the env (the Python arm's `branch_rewards`, deleted in L5; the Rust pass applies the core's indicator rule, §14.5). Under `shaped`
a per-turn reward is a PBRS/bias composition the env's `RewardManager` folds from a `TurnDelta` that
no branch has, and the injected rows would silently carry zero reward, i.e. would teach the critic
that a third of the buffer is inert.

**No ranking term**, ever, at this depth: −0.0107 [−0.0249, +0.0028], NOT DETECTED, negative on
points at every coefficient tested.

---

## 8. WHERE THE ROWS COME FROM, and what the parent does

> ⚠️ **§8–§13 describe the PYTHON-core implementation, DELETED in deletion pass L5** (2026-10-02) now
> that the capability is ported (§14). They are the record of what was built and measured; the files
> named below no longer exist except where marked KEPT. The RULES in §2–§7 are shared by both.

| piece | file |
|---|---|
| selector, meters, cost model, the flag constants | `agents/training/fork_arm.py` (KEPT — the Rust port imports it) |
| the CRN seed derivation | `fork_crn.py` (DELETED, L5) |
| the child that PLAYS the branches | `fork_worker.py` (DELETED, L5) |
| the parent fan-out + the batched scoring + the row assembly | `fork_driver.py` (DELETED, L5) |
| the fill table | `agents/training/fork_buffer.py` (KEPT: `FILL`, `unfillable_keys`, `refusal_text`; the GAE, the branch-row builder and `ForkRolloutBuffer` DELETED, L5) |
| the rollout hook | `fork_callback.py` (DELETED, L5) |
| the gates | KEPT: `fork_arm_test.py`, `fork_buffer_test.py` (the fill table), `fork_flags_test.py` (the flag surface); DELETED: `fork_crn_test.py`, `fork_callback_test.py`, `fork_crn_sim_test.py` (`sim`) |

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

The arm's refusals are THREE `combination_checks` rows: it needs the winprob critic, it is unavailable on the python core (`fork_python_core_unavailable`, deletion pass L4), and it refuses `--win-prob-strata-weight`
(it prices rows by `win_margin`, which the env's reward manager computes — every injected row would
carry the 0.0 fill and land in one stratum, making the delivered strata dose a function of the fork
rate). An obs key the fill table does not know is refused at setup, never guessed (`fork_buffer`'s fill table; the `REFUSE_KEYS` set that held `distill_mask` was deleted with distillation, deletion pass L3). The privileged true-team and dense-aux flags the arm once also refused are deleted (deletion pass L2).

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
the (deleted) R-rollout target's ecology note documented. `fork/branch_share` prices how much of the buffer that population
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
| `requested` / `forks` / `failed` / `records_missing` | asked for, completed, lost, and lost specifically to a pruned `cf_records` ring (Python arm only) |
| `injected_rows` / `masked_rows` / `dropped_forks` | rows added, fork steps masked out of the policy term, forks dropped whole at the row budget |
| `rows_per_fork` / `row_budget` | the MEASURED mean rows a fork contributes, and the budget it is bounded by. `rows_per_fork` is what caps the NEXT rollout's ask; a rising one on a lengthening run is the tell that the fork count is about to fall |
| `seconds` / `worker_failures` / `branches_capped_frac` | the STALL this added to the training loop, killed children, and the share of branches that hit the 250-turn cap (those are dropped) |
| `bot_share` | share of forks whose PARENT episode faced a bot — i.e. how much of the treatment replaced a weaker opponent with a self-like one |

---

## 12. OPERATING IT

**The Python-core operating recipe is HISTORY (deletion pass L4).** The Python arm needed a replayable
record per forked episode, which lived in a ring (flag `cf-records`, pruned globally to the newest N by
`cf-records-keep`); at the default 512 against a production rollout of ~2,400 finished episodes the forks
that DID resolve were the rollout's LATE ones — a selection bias, with `fork/records_missing` as the
tell — so the registered launch raised the cap to 4096: `ctrl10M`'s argv plus `--fork-fraction 0.02
--fork-branches 3 --seed 1001` and the ring flags, i.e. the arm on top of the exact configuration whose
0.5169 baseline it is being read against. None of that can be launched now: the ring flags are gone and
the Python core refuses the arm. A Rust-core launch needs `--critic winprob` (§7) and the §14.7 refusals
only.

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

(Python core, run before deletion pass L4 — not reproducible now.) `--debug --steps 10000 --fork-fraction 0.05` with the ring flags (`cf-records`, `cf-records-keep 4096`) on CPU, one
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

---

## 14. THE RUST CORE PORT (`gen3_fork_rust_v1`) — DECLARED, OFF, DEFERRED

> **STATUS 2026-10-01 — BUILT, DECLARED, OFF; readiness only.** The owner: *"build in the infrastructure so that we can use
> the existing template … and then defer it. We're only on one ply with things that can be
> subsampled."* Playouts to a terminal are DEFERRED by the one-ply scope rule, so `--fork-fraction`
> stays **0 = OFF** on both env cores; no recipe, baseline or arm turns it on. The port exists so the
> tested rules above survive the M5 DELETION PASS, which removes the Python env core (§1–§13 are its
> implementation — **deleted in L5**).

### 14.1 Reused vs new

| piece | Rust core | from |
|---|---|---|
| the battle at the fork, branched with its dice | `search::playout` (`Game::replay` → `Game::branch`), one SearchCore handle per concurrent fork | Lane I, unchanged |
| the episode's input log | `core.finished()` — every ended episode's `InputLog` script, read in the op it ended | Lane H, unchanged |
| branch policy forwards | T2 (`svc.submit` / `flush`), the trainee's CURRENT slot and the parent's opponent slot | T2, unchanged |
| the draws | `keyed_draw` (stream 0 trainee, stream 1 opponent), keyed by the PARENT's (run seed, env, episode) and the decision's frame index | Lane G, unchanged |
| GAE, the arena, the complete-game FIFO | `store.game_gae`, `RowStore`, `GameLog.completed` | Lane G, unchanged |
| selector, branch actions, meters, FILL table, branch reward rule | `fork_arm.*`, `fork_buffer.FILL` | the Python arm's modules, KEPT verbatim (the arm's other halves are deleted, L5) |
| the PG mask + renormalisation in the learner | `fork_pg_m` → `MicroStatic.fork_pg_mask` (R1 declares it) | unchanged |
| **new: the fork PASS** | `agents/training/rust_rollout/fork.py` | — |
| **new: one additive FFI row** | `rust_env_playout_pending_n` — each pending decision's frame index `n` (the core's `dec_n`) | — |

### 14.2 Where the pass runs, and why not a post-collect hook

`RustCollector.collect` is `play (until the trigger fires) → FORK → fill`. The pass is a declared
step of the collector's own phase table (`COLLECT_PHASES`), present only when the flag is on.

The Python arm injects into a buffer it then GROWS. The Rust learner's buffer is a fixed
`[D / N, N]` (every micro-batch full, one compiled graph, nothing allocated after the freeze), so
branch rows cannot be added to it; they COMPETE for the same `D`. A branch is a complete game from the
fork step, so it enters the completed-game FIFO like any other game, inserted RIGHT AFTER ITS PARENT:
a branch is trained in the same update as its parent whenever the parent is (the Python arm's
semantics), and the split/carry rules are the FIFO's. A post-collect hook in `loop.py` would see an
already-filled buffer, so it could only push branches into the NEXT update. The owned loop's own
table is therefore unchanged; its post-collect order (labels → fork → PBRS → frozen-φ) is the Python
core's.

**Dose.** At `branch_share` s an update's `D` rows are `(1−s)·D` own rows and `s·D` branch rows. The
Python arm DOUBLED the update instead. Per row the dose is the same, but the update cadence per own
decision rises by 1/(1−s). `fork/branch_share` reads it.

### 14.2b 🚨 DEPARTURES FROM THE PYTHON ARM — the 2026-09-16 read does NOT transfer unchanged

The NOT DETECTED read (`ai_v13_03_fork`, the status box above) was taken on the PYTHON arm. Two
choices make this port a different treatment, and a reader must not carry that verdict over to it:

1. **Branch rows COMPETE for the update's `D` instead of being added on top.** With forks ON, an
   update sees `(1 − branch_share)·D` own-policy decisions, not `D`. The Python arm doubled the
   buffer, so its updates kept every own decision. The dose per row and the data mix both differ.
   `fork/branch_share` and `fork/own_rows` say by how much.
2. **The branch opponent is the parent's REAL policy opponent where it can be (§14.4).** The Python
   arm substituted a self-like opponent for EVERY fork. Here the substitution covers only the
   remainder: bot parents, external routes and pool slots reloaded since the parent's episode.
   `fork/opp_substituted` publishes that share each pass, so the ecology of a Rust-core read is
   measurable rather than assumed.

### 14.3 CRN on the Rust core

| setting | dice | draws |
|---|---|---|
| `dice` | the battle's own PRNG from the branch point (`seeds = [null]`: `Game::branch(None)` clones the engine with its stream) — the SAME for every branch | each branch keyed on its own stream (`FORK_STREAM_BASE + 2·branch + side`), so the draws are NOT paired: the control |
| **`dice_and_draws`** (default) | the same | **the PARENT'S keys**: trainee = (parent's segment run seed, stream 0, env, episode, `n`); opponent = (…, stream 1, …, `n`), where `n` is the decision's frame index in ITS branch |

🚨 **The identity this buys, and the gate that holds it.** The keyed draw is a pure function of
(key, log-probs). So a branch that takes the parent's own action, against the parent's own opponent
at the same weights, IS the parent: the same commands, rows, protocol and outcome. The sides keep
streams 0 and 1, so they never share a uniform. The k-th post-fork decision of every branch carries
the same key, so it consumes the same uniform. The prefix consumes nothing: the replay feeds the
log's commands by token. `n` is read from the playout (`rust_env_playout_pending_n`), never counted
host-side, because a decision a later feed REPLACES within one write moves `n` without the host
answering it. **Requires `--opponent-sampling keyed`**: the `generator` mode's per-env
`torch.Generator` cannot be replayed per branch, so it is refused.

### 14.4 The branch's opponent

| parent's route | branch's p2 | `opp_class` | counted as |
|---|---|---|---|
| policy (pool / stable / exploiter), the slot still serving the model it served when the parent's episode started | **that slot**, at the route's sampling (greedy, or the route's temperature) | the parent's class | `fork/opp_real` |
| policy, but the slot was reloaded since; or a BOT; or an external route | the CURRENT TRAINEE on stream 1 at T = 1 — §9's self-like approximation | POOL (`fork_buffer.FILL`) | `fork/opp_substituted`, `fork/bot_share` |

On the Rust core the §9 caveat therefore shrinks to the bot share (plus rare pool reloads). On the
Python core it covered every fork.

### 14.5 Rows

- **Row 0, the fork step:** the parent's arena row (observation, mask), RE-SERVED by the current
  trainee slot, so `V(s)` and `log π(a_branch|s)` are the current version's and the PPO ratio is 1 at
  the first epoch.
- **Rows 1+:** the playout's own rows (the TRAINING observation path, program §6c), served in the
  same flushes as the opponents'.
- **Every row:** labels per `fork_buffer.FILL` (NOT SCORED); `fork_pg_m` 0 on row 0 and 1 after
  (collected rows carry 1.0); `episode_start` 1 on row 0; the policy `version` that served it; the
  parent's env and episode. The reward is the core's indicator rule (`victory_value` on a win, else 0;
  a non-indicator terminal is refused). Advantages and returns come from `store.game_gae` over the
  branch, with no bootstrap, and `win_target` is the branch's outcome bit.
- **A branch is DROPPED** when it is CAPPED: p1's stall forfeit (decided by seat), `max_turns`, or
  more rows than `max_game_rows`. **A fork is DROPPED WHOLE** when it leaves fewer than two scorable
  branches, or when its rows would pass the fork row budget.

### 14.6 Declared sizes (acquired at startup, never grown)

| size | value | why |
|---|---|---|
| playout handles | `FORK_CONCURRENCY` = 32, each `max_branches = --fork-branches`, `max_nodes = 1` | forks played in lockstep waves, one T2 flush per wave step |
| rows per wave step | `2 × branches × handles` = 192 at 3 branches | checked against T2's `max_rows_per_flush` at startup (refused, not split) |
| fork rows live in the arena | `ROW_BUDGET_MULTIPLE (1.0) × target` | added to the arena capacity; the measured `rows_per_fork` caps the next ask, and the drop is the backstop (§10) |
| forks per pass | `MAX_FORKS_PER_ROLLOUT` | §10 |

The selection pool, threshold and random branch are §3's, over the games completed since the last
pass. The pool rows are scored by the CURRENT trainee slot in one batched T2 forward, as the Python
arm scores them with the live policy.

### 14.6b Telemetry on the Rust core — the §11 family, per PASS, plus four

The pass publishes §11's `fork/*` family with these readings:
- `rate` is forks per game that ENDED since the last pass;
- `branch_share` is the pass's injected rows over its own rows plus injected rows;
- `records_missing` counts ended games with no input log read;
- `requested` is the forks SELECTED and played; `failed` is the forks dropped whole.

Four tags are new:

| scalar | what it says |
|---|---|
| `asked` | what the fraction asked for, after the measured `rows_per_fork` cap. A gap to `requested` means the candidate POOL held fewer games than the ask (`--fork-max-per-battle` per game) |
| `own_rows` | the trainee's own rows in the games the pass drew from |
| `opp_real` / `opp_substituted` | forks played against the parent's REAL policy opponent, and the share played against the self-like substitute (§14.2b, §14.4) |

### 14.7 Refusals (combination checks)

`--fork-fraction` leaves the `--env-core rust` unported list. Under `--env-core rust` it requires:
- `--opponent-sampling keyed` (§14.3);
- `--rollout-trigger complete_game` (the window fill is the parity schedule).

The ring was a Python-core requirement only; the Rust core's finished logs replace it, and the Python
core's arm is now refused outright (`fork_python_core_unavailable`, deletion pass L4). The other fork
refusals hold: `winprob` and strata weight.

### 14.8 Gates (each FAILS on revert)

| gate | holds |
|---|---|
| (a) `rust_rollout/fork_crn_integration_test.py` | a branch taking the parent's action reproduces the parent byte for byte (commands, every row, mask, action, frame index, outcome). Same dice with other draw streams DIVERGES; a reseeded dice stream DIVERGES. Mutation-checked: swapping the trainee/opponent streams, keying on `n + 1`, or moving the run seed each FAILS it |
| (b) `rust_rollout/fork_test.py` | the prefix appears once (branch rows begin at the fork step; the FIFO holds the parent's rows once); every branch's row 0 has `fork_pg_m` 0 and the learner's PG term is renormalised by `m.sum()`; the advantages and returns equal a hand-built GAE over the branch's outcome; the refusals under a non-winprob critic, generator sampling and the window trigger |
| (c) OFF is a no-op | flag off ⇒ no fork object, no `fork_pg_m` key, no extra arena rows: the K9 learner golden and the stage-1 differential are unchanged; a collector-level pin asserts the off path never touches the pass |
| (d) lifecycle | `fork_crn_integration_test::…lifecycle_stays_clean` (collect = play → fork → fill over four updates: rows injected, `fork_pg_m` 0 on fork steps, the `fork/*` family, every core + T2 `*_after_freeze` 0); `r1_declared_levers_test`'s `fork_rust` row (R1's declared signature carries `fork_pg_mask` and its key; 0 compiles after the lock); a CPU `--debug` real run (§14.10) |

### 14.9 Estimate (scoping, 2026-10-01)

**1.8 agent-days (1.4–2.4): GO** (the scoping estimate; the build landed inside it). The build is
- the pass: ~0.7;
- the collector / store / build / checks wiring: ~0.3;
- the FFI row: ~0.1;
- gates (a)–(d): ~0.5;
- docs and the gate runs: ~0.2.

The largest risk is (a)'s byte identity, because every component of the key must line up. The test
exists to find exactly that.

### 14.10 The CPU smoke (2026-10-01), and what it does NOT cover

`--debug --steps 8192 --env-core rust --arch production --fork-fraction 0.05 --device cpu
--rust-env-profile selfcheck` on the build commit, one env, a 2,048-row update. **Training
complete**: 8 passes, every update's core and T2 lifecycle check clean, 0 `records_missing`.

| pass | `own_rows` | `asked` → `requested` → `forks` | `dropped_forks` | `injected_rows` | `branch_share` | `tie_rate` | `pairwise_acc` (pairs) | `sim_steps_share` | `rows_per_fork` | `seconds` |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2,070 | 103 → 61 → 24 | **37** | 1,980 | 0.489 | 0.625 | 0.444 (18) | 1.88 | 82.6 | 24.1 |
| 2 | 57 | 2 → 1 → 1 | 0 | 162 | 0.740 | 1.000 | — (0) | 5.53 | 162 | 1.8 |
| 3 | 1,890 | 12 → 12 → 12 | 0 | 1,640 | 0.464 | 0.750 | 0.400 (5) | 1.99 | 136 | 23.2 |
| 4 | 409 | 15 → 3 → 3 | 0 | 708 | 0.634 | 0.333 | 0.500 (4) | 3.47 | 236 | 9.2 |
| 5 | 1,330 | 8 → 8 → 8 | 0 | 752 | 0.362 | 0.875 | 0.000 (2) | 1.13 | 94 | 7.3 |
| 6 | 1,310 | 21 → 16 → 16 | 0 | 1,530 | 0.537 | 0.562 | 0.714 (14) | 2.29 | 95.4 | 10.0 |
| 7 | 517 | 21 → 8 → 8 | 0 | 888 | 0.632 | 0.375 | 0.600 (10) | 3.38 | 111 | 7.4 |
| 8 | 1,180 | 18 → 15 → 14 | 1 | 1,120 | 0.486 | 0.714 | 0.625 (8) | 1.88 | 79.6 | 13.3 |

**What the smoke shows:**
- **The first pass pays the row-budget drop once.** It dropped 37 of 61 played forks. After that the
  measured `rows_per_fork` caps the ask, as in §10.
- **Updates come faster.** Branch rows fill the FIFO, so an update can fire after only 57 new own
  rows (pass 2), and `branch_share` exceeds 0.5 on such a pass. That is the §14.2b dose departure,
  measured.
- **`pairwise_acc` is noise here**, as in §13.

**What it does NOT cover:**
- **The real-opponent path.** `--debug` plays bots, so `opp_substituted` = 1 throughout. The choice
  between the real slot and the substitute is pinned by `fork_test`. A branch played on a pool slot
  has not run end to end.
- **GPU R1.** `--debug` bypasses `--compile-trainer`, and no GPU run was made. R1's declaration with
  `fork_pg_mask` is held by `r1_declared_levers_test`'s `fork_rust` row (dynamo's eager backend on
  CPU, 0 compiles after the lock). Its compiled signature is the Python arm's `fork` row minus the
  ragged tail.
- **Production scale.** Waves at N = 48 or more, and the stall as a share of a production update,
  are unmeasured.

### 14.11 🚨 BEFORE ENABLING — the checklist that lifts the owner's deferral

The port is readiness, not a validated treatment. **No argv sets `--fork-fraction > 0` on
`--env-core rust` until every item below is done and recorded here**, each with a link to its
evidence. Whoever lifts the owner's deferral (2026-10-01, the one-ply scope rule) works from this
list; an item may not be waived silently.

1. **A compiled CUDA R1 proof with fork rows.** Run a real GPU launch under `--compile-trainer`,
   outside `--debug` (which bypasses it), with branch rows in the update and `fork_pg_m` 0 on
   their fork steps. The pass bar is 0 compiles and 0 rejections after the lock
   (`compile/recompiles_after_lock` 0) and a clean canary. Today R1's declaration is held only by
   `r1_declared_levers_test`'s `fork_rust` row, on dynamo's eager backend on CPU (§14.10).
2. **An end-to-end run of the REAL-OPPONENT path, outside `--debug`.** A branch must play the
   parent's POOL slot (`fork/opp_real` > 0), and the identity gate (a) must be repeated with a
   policy-route parent. Today the real-versus-substitute choice is pinned only by `fork_test`.
   Every end-to-end run so far played bots or a self-like external p2.
3. **A DECLARED cap on `branch_share` per update.** §14.10 measured `branch_share` above 0.5 on a
   pass and an update firing on 57 own-policy rows. That makes the dose and the update cadence
   functions of the fork rate, not of the flag.
   - **Proposed default:** cap a pass's injected rows at its own rows
     (`injected_rows ≤ own_rows`, i.e. `branch_share ≤ 0.5` per pass). This is the Python arm's
     designed maximum, which doubled the buffer and never exceeded half.
   - **Mechanism:** forks over the cap are dropped WHOLE, counted as `fork/dropped_forks`.
   - **Startup refusal:** a fractional `ROW_BUDGET_MULTIPLE` above it is refused at startup.
   - **Required gate:** an update must never hold more branch rows than own rows. It must fail on
     revert.
4. **Production-scale COST at the adopted N.** Measure the wave stall (`fork/seconds`) as a share of
   an update's wall, plus `fork/sim_steps_share`, at the sizing study's adopted N and production
   update size. Read them beside `rust_env/*` so the host loop's own cost is the denominator. Neither
   was measured beyond one CPU env.
5. **A REGISTERED arm, compared against X26.** Pre-register the endpoint (held-out pairwise accuracy
   on fresh forks, §12) and the comparator. The comparator is the X26 ride-along baseline (the first
   GPU run after the M5 switch) by registry name, at matched budget and dose (`main.dose`). Record the
   §14.2b departures in the registration. The 2026-09-16 NOT DETECTED read is not the comparator.
