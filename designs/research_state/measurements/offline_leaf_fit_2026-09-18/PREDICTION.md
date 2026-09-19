# PRE-REGISTRATION — THE OFFLINE LEAF FIT: a value head fitted against a FROZEN policy on branched, decorrelated self-play

*Registered 2026-09-18, and landed on main BEFORE the first battle of this read is played. Every
number in the record that follows is scored against this file in its §8.*

---

## 1. The question, and why this design and not another

Eleven win-prob heads, across two 10M runs, two 75M+ runs, a fork arm trained on contested forks
and an exploiter narrowed to one team and one opponent, all rank two states one move apart at
**0.568–0.578** pairwise accuracy under shared dice. Coverage, width, loss form, training data and
distribution have each been varied; none moved the level
(`paired_refit_discrimination_2026-09-14/`, `fork_arm_read_2026-09-16/`,
`exploiter_discrimination_2026-09-18/`; UNDERSTANDING §4.x). A head at 0.575 is not a usable search
leaf — nine of those heads have a `SEARCH HARMS` or no-dividend battery to show for it.

`UNDERSTANDING.md` names what is left: *"the offline fit against a frozen policy is the last
untried design."* Every head so far was trained by PPO **on its own moving policy's own visited
states**, where the value target is a moving bootstrap and the state distribution is the policy's
own. AlphaGo's value network was trained the other way: **one position per self-play game, taken
after a single uniformly-random move, labelled by the game's final outcome under a FROZEN policy.**
The two properties that recipe buys — a STATIONARY target and DECORRELATED positions — are exactly
the two this campaign has never varied.

**The question: does an offline fit against a frozen policy on branched, decorrelated positions
clear the 0.575 ceiling?**

## 2. The frame

| | |
|---|---|
| **frozen policy** | `models/ai_v13_02_flywheel_winprob/final_model.zip` — **arm W**, step **75,005,952**, config_version **119**, `ARCH_SIGNATURE` `gen3_critic_route_wave_v1`, `--critic winprob`, `win_prob_mode shaping`, `--ent-coef 0.05`, `eval_sentinel_greedy: true`, a FRESH generalist run (no fork parent, no team pin) |
| **the control head** | that checkpoint's OWN `WinProbHead`, scored on the SAME forks and the SAME pairs — the only comparator this read takes a paired delta against |
| **the trunk** | FROZEN. `value_pooled` and the pre-pool 12-token memory are forwarded ONCE, offline, under `no_grad`; every fit is a small MLP on a fixed matrix. No trunk parameter moves and no gradient reaches one |
| **regime** | CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, rust sim bridge, ports 9700–9799 reserved (the bridge is in-process and uses none), everything under `/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit/`, **nothing written under `models/`**, a live training arm holds the GPU and is not touched, 8000/8001 untouched |

## 3. The datasets

### 3.1 TRAIN — the branched AlphaGo set (`branch_forks.py`)

Self-play games of the frozen policy against ITSELF, both seats **GREEDY** (the run's recorded
`eval_sentinel_greedy: true` regime; greedy on both sides is also what makes a branch a
deterministic function of its action, which is the whole basis of the CRN pairing). Teams are the
full 719-team pool on both sides — arm W is a generalist and has no pin.

At a decision drawn **UNIFORMLY** over the game's eligible `move_selection` decisions (turn 1–60,
≥2 legal actions) — not on the policy's logit gap, not on `|V − 0.5|`, not on the outcome; the
selector reads no head — **ONE uniformly-random legal move** is played, drawn from
`legal \ {top1}` with a decision-keyed rng, and the frozen policy plays the line to a terminal on
the record's own dice stream. The position right after that move, and the game's outcome, are the
datum. The **SIBLING** is the position after the policy's own `top1` from the same pre-fork state
under the SAME dice.

**Target: 6,000 self-play games, 3 branch points per game ⇒ ~18,000 forks / ~36,000 rollouts.**
Whatever the clock delivers is reported as delivered, with its wall and its shard geometry; the
floor below which this read is reported as UNDERPOWERED rather than as a result is **8,000 forks**.

### 3.2 READ — the contested set (`forks.py`, VERBATIM BY PATH)

A **SEPARATE** tree of the same frozen policy, different seed, different games. Forks built by
`paired_refit_discrimination_2026-09-14/forks.py`, invoked by path and unmodified, at flags
byte-identical to `fork_arm_read_2026-09-16` and `exploiter_discrimination_2026-09-18` except the
seed: `--forks-per-battle 4 --gap-quantile 0.40 --min-legal 3 --min-turn 2 --max-turn 40`, three
branches (`top1` / `top2` / `rand`). **Target ~5,040 forks / 15,120 rollouts.** The fits never see
these battles, these positions or these labels.

### 3.3 GUARD frame

A third small tree of arm W at 74,000,016 with the DEFAULT bot roster and 3 pool sentinels
(`--games 80 --sentinels 3` ⇒ 1,200 battles), which is what `cond.opp_class_auc.t4_10` needs: a
bots-free tree has ONE opponent class and the AUC is undefined, and `refit.conditioning_frame`
also needs a `plan.json`, which a LIVE eval tree does not carry.

## 4. The fits

All on the frozen trunk's tensors, split **80/20 BY BATTLE** (two forks of one battle share a
prefix; splitting by fork leaks), Adam `lr 1e-3`, batch 1024, ≤20,000 steps, validation every 25
steps, early stop at patience 40 — `refit.py`'s recipe, reused.

| | head | input | data | loss |
|---|---|---|---|---|
| **(a)** | `WinProbHead`, **WARM-STARTED** from the checkpoint's own | `value_pooled` | the **RAND-branch** positions only (the strict one-label-per-game AlphaGo set) | BCE |
| **(b)** | the same, warm-started | `value_pooled` | **BOTH** branches of the paired siblings | BCE |
| **(c)** | **WIDER** (LN → 256 → ReLU → 256 → ReLU → 1), fresh-init | `value_pooled` | both branches | BCE |
| **(d)** | the same wider head | **PRE-POOL 12 team tokens (12×128, fainted zeroed, flattened) ⊕ `value_pooled`** = 1,664 dims | both branches | BCE |
| **(e)** | the same wider head | `value_pooled` | both branches | BCE + a pairwise ranking term at **0.3** over the fork's own sibling pairs |
| (a0) | `WinProbHead` **FRESH-INIT** | `value_pooled` | (a)'s data | BCE |
| (c0) | `WinProbHead` **FRESH-INIT** | `value_pooled` | (c)'s data | BCE |

(a0) and (c0) are **matched-init controls**, not registered cells: (a)/(b) are warm-started and
(c)/(d)/(e) structurally cannot be, so a wide-head null would otherwise be confounded with cold
start.

🚨 **`vf_features` IS NOT THE PRE-POOL TENSOR, and the task's (d) as literally written is
information-vacuous.** The extractor returns
`activation(value_projection(value_pre_norm(vf_combined)))`, and since the critic-route deletion
wave **`vf_combined` IS `value_pooled`** (`src/agents/model/projection.py`) — so `vf_features` is a
deterministic function of `value_pooled` and carries nothing it lacks. The genuine "more
information" tap is the **12-token memory the value CLS query attends over**, captured at the
`cls_pool.value_cls_attn` seam so it carries every value-route TOKEN injection the pool itself
sees, and **concatenated with `value_pooled`** so the v89 route contributions that are added AFTER
the pool are not silently dropped. (d)'s input is therefore a strict SUPERSET of (c)'s, which is
what makes "(d) > (c) ⇒ the pooling discards ranking information" a readable claim. Registered as
a substitution, before the data.

## 5. THE BARS

**The primary metric** is held-out **pairwise accuracy** on the contested read set:
`sign(V(s'_a) − V(s'_b))` against the shared-dice outcome on non-tied sibling pairs, with a
**bootstrap over FORKS** (three pairs inside one fork share a prefix and are not three
observations). It is rank-based, so no monotone re-calibration can produce a gain.

| bar | statement |
|---|---|
| **BAR 1 — THE REGISTERED PASS** | a fit's **CI LOWER BOUND > 0.60**. Rule 25's instrument width at ~4,000 pairs is ±0.015, so this needs a point estimate of roughly ≥ 0.615 — a full instrument-width clear of the four-head 0.568–0.578 band |
| **BAR 2 — the paired delta** | that fit's paired Δ against the ORIGINAL head on the SAME pairs, CI clear of zero |
| **BAR 3 — not bought with calibration** | the passing fit's **ECE** is not worse than the original head's by more than **0.03**, and its **Brier** is reported beside it. A ranking gain paid for with a wrecked probability is reported as such |
| **BAR 4 — the conditioning guard** | `cond.opp_class_auc.t4_10` (label 1 = scripted bot, the published sign) for the passing fit stays within **0.10** of the original head's. A fit that destroys between-opponent conditioning has traded one axis for another and the trade is named |
| **BAR 5 — the battery, only if BAR 1 fires** | ≥400 paired mirror games, the fitted head as the leaf via `--leaf-head`, the ORIGINAL head as the CONTEMPORANEOUS control in the same window (rule 23): **L2 paired win rate with its CI**, the `grid` cell, and separation at matched realized K |
| **the indexing clause** | rule 25's second clause EXECUTED, not asserted: `exploiter_discrimination_2026-09-18/verify_indexing.py` by path, re-deriving the ORIGINAL head's accuracy on the read set with independent code, matching to < 1e-6, with ten pairs printed |

🚨 **Only WinProbHead-shaped fits (a / b / a0 / c0) are battery-eligible** — `--leaf-head` loads a
bare state_dict whose keys must match the checkpoint's `win_head`. If a WIDE fit (c / d / e) is the
one that passes, **the battery is NOT RUNNABLE without an `src/` change** and that will be reported
as the obstacle, not worked around.

## 6. The branches, registered

* **(a) A fit clears 0.60 AND the battery pays** ⇒ the leaf exists; the search-teacher path opens.
* **(b) Clears 0.60 on the metric, the battery does not pay** ⇒ ranking on branched states does not
  transfer to search-time states — the two populations come apart, and the metric stops being a
  proxy for leaf usefulness.
* **(c) No fit clears 0.60** ⇒ the ceiling is **not** the target's stationarity or the state
  distribution's correlation either. What remains to be named in that case: **rollout-averaged
  leaves** (a leaf that is N frozen-policy rollouts rather than one forward), **hand evaluators as
  controls** (does a Foul-Play-style heuristic outrank 0.575 on the same forks? — which would
  separate "the task is hard" from "the head is bad"), and **per-action Q leaves**
  (`q_winprob_head`, built and OFF). The leaf question CLOSES at this depth.
* **(d) (d) beats (c) with its CI clear of zero** ⇒ the value CLS pool discards ranking information
  — an ARCHITECTURE finding, independent of whether either clears 0.60.

## 7. THE PREDICTIONS — falsifiable, before the data

| # | prediction |
|---|---|
| **P1** | **The registered verdict is branch (c).** No fit's CI lower bound clears 0.60. Prior: five distinct levers have moved nothing; the frozen target is the sixth |
| **P2** | The **ORIGINAL** head on the contested read set reads **0.55–0.60**, reproducing the four-head band (0.5679 / 0.5746 / 0.5746 / 0.5781). If it does NOT, the read set is not comparable to the banked ones and the whole read is reported as such |
| **P3** | **(a)** reads **0.555–0.605**, ≈ 0.578; its paired Δ vs the original is **−0.01 to +0.03**, ≈ +0.005, **NOT DETECTED**. A warm-started head refitted on 18k off-policy labels moves a little and not far |
| **P4** | **(b)** reads within **0.015** of (a). The pairing adds positions, not information: the `top1` sibling's label IS the recorded game's own outcome under CRN, so the paired set doubles the rows while adding ~0 independent labels |
| **P5** | **(c)** reads **0.53–0.60** — at or slightly BELOW (a), because it is fresh-init on ~36k rows against a head that had 75M steps of trunk-coupled training. (c0), the matched-init control, reads within 0.02 of (c) |
| **P6** | **(d)** reads within **0.02** of (c), Δ(d−c) **NOT DETECTED** — branch (d) does not fire. The pre-pool tensor is 13× wider and the fit is more likely to overfit 36k rows than to find ranking information the pool threw away. ⚠️ Registered as the prediction most likely to be wrong, and the one with the highest value if it is |
| **P7** | **(e)** reads within **0.02** of (c). The ranking term was already null on recorded on-policy states (2026-09-14); here the pairs are the design, but only ~10–20 % of them are non-tied, so the term sees **1,500–3,500** pairs — thin |
| **P8** | The branched set's **non-tied pair rate is 8–20 %**, materially BELOW the contested set's ~19 %: a uniformly-drawn decision is usually not contested, so a random move usually does not change the outcome |
| **P9** | The **`rand` branch's stall-cap rate EXCEEDS `top1`'s** — a random move can walk the frozen policy into a 250-turn loop that its own top-1 line avoids. The exclusion of capped branches is therefore NOT symmetric across branches, and the per-branch rate is reported |
| **P10** | The branched set's **base rate is 0.45–0.55**: self-play, one seat, both sides the same policy |
| **P11** | Zero determinism failures over ≥ 200 CRN re-runs across the two builders; **< 2 %** of branch rollouts stall-capped overall |
| **P12** | The guard `cond.opp_class_auc.t4_10` for the ORIGINAL head reads **0.60–0.80** (the published online-head values are 0.723 / 0.730); every fit stays within 0.10 of it — BAR 4 is met by all cells, because none of them is trained on anything that would remove opponent conditioning from `value_pooled` |
| **P13** | The battery is **NOT RUN**, because P1 says no fit clears BAR 1 |

**The prior this reads against, stated so the result cannot be fitted to it:** on this instrument,
**every** registered lever in this campaign has come back inside ±0.015 of 0.575, and the ONE
detected effect (`exploiter_discrimination_2026-09-18`, +0.0301) turned out to be the comparator
falling rather than the arm rising. A 0.60 lower bound is a large ask, and predicting it will not
be met is the honest prior — which is exactly why the bar and this prediction are written down
before any fork exists.

## 8. Declared deviations from the task as briefed

1. **6,000 games × 3 branch points, not 20,000 games × 1.** Tree generation runs at ~0.8–1.2
   games/s on this contended box (a live training arm holds the GPU and ~16 cores), so 20,000
   self-play games is ~5 h of tree generation before a single fork is built. Three forks per game
   buys the position count at a third of the games. **The cost is correlation**: three positions
   from one game share a prefix. It is mitigated, not removed — the split is **by battle**, and
   every CI is a bootstrap **over forks**, never over pairs or rows. Reported as a limitation.
2. **The uniform draw excludes `top1`.** `rand ~ U(legal \ {top1})` rather than `U(legal)`, so the
   two siblings are always different actions — a draw landing on `top1` would give fits (b) and (e)
   a degenerate pair. The positions are correspondingly slightly further off-policy than AlphaGo's.
3. **(d) is fitted on the pre-pool 12-token memory ⊕ `value_pooled`, not on `vf_features`** — §4.
4. **The guard frame is a generated bots+sentinels tree, not the read tree** — §3.3.
5. **A ~20-battle / 12-fork PLUMBING SMOKE was run before this registration** to validate the
   shadow-run self-play posing, the branched builder and the fit driver, and to measure throughput.
   Its forks are DISCARDED and are not scored; no endpoint was read from it.

## 9. What would make this read WRONG rather than null

Stated now so the record cannot be rescued after the fact:

* the ORIGINAL head reading outside 0.55–0.60 on the read set (P2) ⇒ the read set is not the
  instrument the reference levels are on, and no comparison may be made;
* a determinism failure ⇒ the CRN pairing is not a pairing and the branches are three samples;
* the indexing verification failing to reproduce to < 1e-6 ⇒ the 2026-09-14 artifact class is back;
* the pre-pool tap capturing a constant or wrongly-shaped tensor ⇒ (d) is a fit on a bias term and
  its null says nothing (`fit_heads.py` REFUSES on both, before any fit runs);
* fewer than 8,000 branched forks delivered ⇒ reported as UNDERPOWERED, not as branch (c).
