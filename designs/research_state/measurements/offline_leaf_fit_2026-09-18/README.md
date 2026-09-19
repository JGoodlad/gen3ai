# DOES AN OFFLINE FIT AGAINST A FROZEN POLICY CLEAR THE 0.575 SIBLING-RANKING CEILING? — AlphaGo's value-network recipe, on arm W

*Measured 2026-09-19 02:55 – 08:40 UTC · **6,000 frozen-policy self-play games** (arm W posed as its
own sole sentinel, both seats greedy), **17,904 branched AlphaGo forks / 35,808 rollouts** as the
training set, **5,040 fresh contested CRN forks / 15,120 rollouts** from a SEPARATE tree as the
held-out read, **seven heads fitted on a frozen trunk** · CPU only (`CUDA_VISIBLE_DEVICES=""`),
`nice 15`, rust sim bridge, nothing written under `models/` · **zero errors, 462/462 determinism
re-runs identical, 59 of 50,928 branch rollouts stall-capped (0.116 %)** · a training arm (the
ERA-1 fold) held the GPU throughout and was not touched; ports 8000/8001 untouched.*

**Pre-registered in [`PREDICTION.md`](PREDICTION.md), landed on main at `a78d4b30` before the first
battle of this read was played.** Scored in §8.

---

## 1. VERDICT — branch (c): the ceiling is not the target's stationarity either. And the pooled 0.575 turns out to be a MIXTURE of three very different difficulties, which is the more useful result.

**1 — NO FIT CLEARS THE BAR, AND NOT ONE COMES CLOSE.** Seven heads — a warm-started `WinProbHead`
on the strict one-label-per-game AlphaGo set, the same on the paired siblings, a 2-layer 256-wide
head on `value_pooled`, that head on the PRE-POOL 12-token memory, that head plus a pairwise
ranking term, and two matched-init controls — read **0.5673 to 0.5760** on 3,795 held-out non-tied
contested pairs. The registered bar was a **CI lower bound > 0.60**; the highest lower bound
achieved is **0.5613**. **BAR 1 FAILS on all seven.** Every paired Δ against the original head is
**NOT DETECTED**, the largest being **+0.0037 [−0.0050, +0.0117]**.

**2 — THE ORIGINAL HEAD READS 0.5723 [0.5583, 0.5866] ON THIS FRESH SET,** reproducing the
four-head 0.568–0.578 band on a state set nobody had built before (P2 held). The instrument is the
one the published levels are on. **Twelve heads now sit on that level.**

**3 — WHAT THE OFFLINE FIT ACTUALLY BOUGHT IS CALIBRATION, AND IT BOUGHT A LOT OF IT.** On the same
14,570 branch rows: **ECE 0.0721 → 0.0133–0.0167** (a 4–5× improvement), **Brier 0.1927 → 0.1825–0.1838**,
`mean_V` **0.5539 → 0.482–0.492** against a base rate of **0.4836**. 🚨 **The frozen critic is
over-confident by 7.0 points on its own self-play forks, and one pass of offline BCE on branched
labels removes essentially all of it while leaving the RANKING exactly where it was.** That is the
calibration-vs-resolution split, measured on one head in one afternoon: **stationary targets on
decorrelated positions fix the offset and do not touch the resolution.** It is the same verdict the
G0 bias map returned in August by a different route.

**4 — 🚨 THE POOLED 0.575 IS A MIXTURE OF THREE PAIR TYPES WITH A 0.049 SPREAD, AND THE HARDEST ONE
IS THE ONE A SEARCH LEAF ACTUALLY FACES.** Decomposed on the identical forks
([`pairtype.json`](pairtype.json), POST-HOC):

| pair | non-tied | the ORIGINAL head | the best fit (d) |
|---|---|---|---|
| **`top1` vs `rand`** — a played move against a random legal one | 1,312 | **0.5938 [0.5739, 0.6142]** | 0.6014 [0.5813, 0.6219] |
| `top2` vs `rand` | 1,316 | 0.5752 [0.5549, 0.5957] | 0.5722 [0.5522, 0.5922] |
| **`top1` vs `top2`** — the policy's own two best, at a CONTESTED decision | 1,167 | **0.5450 [0.5236, 0.5666]** | 0.5518 [0.5302, 0.5733] |

**The head separates "good move" from "random move" at 0.59, and its own top-1 from its own top-2
at 0.545.** The published ceiling is the average of a task it half-does and a task it barely does.
**And the second is the leaf's actual job** — a search that re-ranks successors is deciding between
candidates the policy already likes, which is exactly the `top1|top2` column.

**5 — THE SAME HEADS READ 0.615 ON BRANCHED PAIRS.** On 981 held-out non-tied pairs of the
BRANCHED population (held-out BATTLES, [`posthoc.json`](posthoc.json), POST-HOC), the original head
reads **0.6157 [0.5925, 0.6397]** and the fits **0.6035–0.6198** — all above what any of them reads
on the contested set. **The level is a property of the STATE SET, not of the head.** Together with
§1.4: the 0.575 number is "how well can this head rank the states a CONTESTED-gap selector picks",
and a contested-gap selector picks, by construction, the states where the policy itself cannot
tell the candidates apart.

**6 — THE PRE-POOL TENSOR DOES NOT BEAT `value_pooled`, so BRANCH (d) DOES NOT FIRE.** Fitted on
the 12-token memory the value CLS query attends over (12×128, fainted slots zeroed) CONCATENATED
with `value_pooled` — a strict superset, 1,664 dims — fit (d) reads **0.5760 [0.5613, 0.5903]**
against (c)'s **0.5713**, paired **Δ +0.0047 [−0.0034, +0.0133] NOT DETECTED**. It is the best of
the seven and it is not detected. **No evidence the value CLS pool discards ranking information;
also no evidence it does not, at this sample.**

**7 — THE RANKING TERM IS THE WORST CELL, for the third time.** (e) reads **0.5673**, paired against
(c) **−0.0040 [−0.0087, +0.0008] NOT DETECTED** — the same null the 2026-09-14 refit returned on
recorded on-policy states and the 2026-09-16 fork arm returned on trained-in forks. Here the pairs
were the DESIGN and it still did not help.

**8 — THE GUARD IS CLEAN.** `cond.opp_class_auc.t4_10` = **0.7384** for the original head,
**0.7069–0.7331** for the seven fits — every one inside the registered ±0.10, the worst deviation
**0.0315**. No fit traded opponent conditioning for anything. **BAR 4 MET by all seven.**

**9 — THE BATTERY WAS NOT RUN**, on the registered ground: BAR 5 was conditional on BAR 1 and BAR 1
failed on every cell (P13 held).

## 2. The frame

| | |
|---|---|
| **frozen policy** | `models/ai_v13_02_flywheel_winprob/final_model.zip` — **arm W**, step **75,005,952**, config_version **119**, `ARCH_SIGNATURE` `gen3_critic_route_wave_v1`, `--critic winprob`, `win_prob_mode shaping`, `--ent-coef 0.05`, `eval_sentinel_greedy: true` (RECORDED, not declared), a FRESH generalist — no fork parent, no team pin, so both seats draw the full 719-team pool |
| **the control head** | that checkpoint's OWN `WinProbHead`, scored on the SAME forks and the SAME 3,795 non-tied pairs |
| **the trunk** | FROZEN and not in the graph. `value_pooled`, the pre-pool memory and `V0` are forwarded ONCE under `no_grad`; every fit is a small MLP on a fixed matrix. No trunk parameter moved |
| **builder / scorer** | the contested read set is `paired_refit_discrimination_2026-09-14/forks.py` invoked **BY PATH and UNMODIFIED**; the read is `refit.read_head` / `refit.paired_delta`; the alignment assertion is `fork_arm_read_2026-09-16/score_forks.branch_index`; rule 25's indexing clause is `exploiter_discrimination_2026-09-18/verify_indexing.py`, also by path. [`branch_forks.py`](branch_forks.py) IMPORTS the CRN machinery (`run_branch`, `load_model`, `score_batch`, `battle_files`, `_sha`) from the 2026-09-14 builder and changes only the SELECTION rule |
| regime | CPU only, `nice 15`, rust bridge, everything under `/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit/`, **nothing under `models/`** |
| the box | contended for most of the window (the ERA-1 fold on the GPU, load 13–25 on 16 cores), idle for the last hour. Walls are DESCRIPTORS |

### 2.1 The declared deviation, as executed — the SELF-PLAY posing

`eval_trace_gen` draws sentinels from the run's own `snapshots/`, whose newest is 72,000,000, not
the frozen 75,005,952 final. A shadow source run therefore carries arm W's `model_config.json` and
`metadata.json` byte-for-byte, `eval_traces/step_75005952/snapshot.zip` → arm W's `final_model.zip`,
and `snapshots/snapshot_000075005952.zip` → **the same file**, with arm W's own `model_config.json`
beside it. With `--sentinels 1 --include-current-snapshot --opponents ,` the cycle's sole opponent
IS the frozen policy. **The artifact confirms it: the generated manifests record
`sentinel_steps: [75005952]`, one opponent, capture ALL, `eval_sentinel_greedy: true` source
`recorded`, and the self-play win rates are 0.4908 (train) and 0.5023 (eval)** — a mirror of a
policy against itself, as it must be.

## 3. The datasets, and the checks that ran before any head was fitted

| | **TRAIN — the branched AlphaGo set** | **READ — the contested set** |
|---|---|---|
| tree | 6,000 self-play games, seed 20260918, 62.9 min | 1,300 self-play games, seed 20260919, 13.9 min |
| games complete | **6,000 / 6,000** | **1,300 / 1,300** |
| self-play win rate | 0.4908 | 0.5023 |
| selector | **UNIFORM over eligible `move_selection` decisions** (turn 1–60, ≥2 legal); 3 per game | contested: bottom 40 % of the top-2 masked-logit gap; 4 per game |
| branches | `top1` + `rand ~ U(legal \ top1)` | `top1` + `top2` + `rand` |
| forks · rollouts | **17,904 · 35,808** | **5,040 · 15,120** |
| shards · wall | 6 · 143.7–155.6 min | 6 · 59.7–68.3 min |
| **errors** | **0** | **0** |
| **determinism re-runs · disagreements** | **360 · 0** | **102 · 0** |
| branches with no successor | 836 (2.3 %) | 519 (3.4 %) |
| branches stall-capped (EXCLUDED) | **38 (0.11 %)** | **21 (0.14 %)** |
| branch rows scored | 34,890 | 14,570 |
| pairs · **non-tied** | 17,256 · **4,935** | 14,439 · **3,795** |
| tie rate | 0.7140 | 0.7372 |
| **base rate** | **0.4749** | **0.4836** |
| median fork turn | 17 | 17 |
| median top-2 logit gap | **0.863** | (shard thresholds **0.593–0.647**) |
| median depth fraction through the game | **0.500** | — |

**CHECK 1 — the CRN pairing is asserted, not assumed.** 462 branches were re-run a second time
across the two builders and compared: **0 disagreements.** With both sides greedy on one dice
stream a branch is a deterministic function of its action, which is what makes two (or three)
branches of a fork a paired comparison rather than independent samples.

**CHECK 2 — the alignment assertion.** `score_forks.branch_index` recomputes the branch→successor
index independently of `build_table` and the read REFUSES on any disagreement. It reports the
condition under which the 2026-09-14 defect bites: **14,224 of 14,570 read rows** sit at a position
different from their successor index (max offset 31).

**CHECK 3 — rule 25's second clause, EXECUTED against hand-scored pairs.**
[`verify_indexing.py`](../exploiter_discrimination_2026-09-18/verify_indexing.py), invoked by path,
re-derives the whole metric from the raw `forks_s*.jsonl` and `succ_s*.npy` with its own reader,
pair enumeration and agreement rule, and prints ten pairs in full
([`verify_indexing.json`](verify_indexing.json)). It reproduces **0.5723320158** against the read's
**0.5723320246** — a gap of **8.8e-9**, float32-mean precision (an indexing difference would be of
order 1/n).

**CHECK 4 — the pre-pool tap is not vacuous.** `fit_heads.py` REFUSES before any fit if the tap's
shape is not `(N, 12, 128)` or if it is constant across states. Measured: shape `(N, 12, 128)`,
**25.03 % of tokens key-masked to zero** (≈3 of 12 slots fainted on an average successor, which is
the right order for a mid-game position), **mean per-dimension sd 0.755**. The tap is live.

**CHECK 5 — the guard frame is the published one.** `guard_frame` re-implements
`refit.conditioning_frame` in order to capture the pre-pool tensor alongside; it is cross-checked
against `refit.conditioning_frame` itself on the same seed and the report records
`frame_matches_refit_conditioning_frame: true` (960 frames, 720 bot / 240 sentinel, label 1 = bot,
the published sign).

### 3.1 Three facts about the BRANCHED population, before any head is fitted

| on identical dice | |
|---|---|
| `top1` wins | **0.4928** |
| `rand` (a uniformly random legal alternative) wins | **0.4571** |
| **cost of throwing the decision away** | **3.57 pp** |
| `top1` stall-cap rate | **0.006 %** (1 of 17,904) |
| **`rand` stall-cap rate** | **0.207 %** (37 of 17,904) — **37× higher** |
| non-tied rate on the matched `(top1, rand)` pair | **0.2860** (branched) vs **0.2726** (contested) |

🚨 **The cap exclusion is NOT symmetric across branches** (P9, registered): a random move walks the
frozen policy into a 250-turn loop 37× more often than its own top-1 line does. At 0.2 % the
induced selection is far too small to move any endpoint here, but a future read at a deeper fork
budget must carry it.

**And the matched decidability inverts the registered prediction:** restricted to the SAME
`(top1, rand)` pair, a uniformly-drawn decision's random move changes the outcome **more** often
(0.286) than a CONTESTED decision's does (0.273). **The contested-gap selector does not select for
outcome relevance**; it selects for the policy's own indifference, which is a different thing.

## 4. THE PRIMARY READ

`sign(V(s'_a) − V(s'_b))` against the shared-dice outcome on non-tied pairs; `V` is the head's own
`sigmoid(logit)`, which under `--critic winprob` IS the critic. **Bootstrap over FORKS**, 2,000
resamples. All seven fits and the original are scored on the **identical 3,795 non-tied pairs**.

| head | input | data | **pairwise accuracy** | paired Δ vs original | ECE | Brier | sep ratio | mean V |
|---|---|---|---|---|---|---|---|---|
| **original** (arm W's own) | — | — | **0.5723 [0.5583, 0.5866]** | — | 0.0721 | 0.1927 | 1.349 | 0.5539 |
| **(a)** `WinProbHead` WARM | `value_pooled` | rand-branch only | 0.5713 [0.5569, 0.5853] | −0.0011 [−0.0083, +0.0063] **ND** | **0.0141** | 0.1833 | 1.446 | 0.4834 |
| **(b)** `WinProbHead` WARM | `value_pooled` | both branches | 0.5681 [0.5540, 0.5823] | −0.0042 [−0.0113, +0.0029] **ND** | **0.0147** | 0.1831 | 1.419 | 0.4820 |
| **(c)** wide 256 | `value_pooled` | both branches | 0.5713 [0.5562, 0.5858] | −0.0011 [−0.0088, +0.0063] **ND** | **0.0167** | 0.1830 | 1.416 | 0.4859 |
| **(d)** wide 256 | **PRE-POOL ⊕ pooled** (1,664) | both branches | **0.5760 [0.5613, 0.5903]** | +0.0037 [−0.0050, +0.0117] **ND** | 0.0321 | 0.1838 | **1.491** | 0.4624 |
| **(e)** wide 256 + rank@0.3 | `value_pooled` | both branches | 0.5673 [0.5526, 0.5818] | −0.0050 [−0.0116, +0.0017] **ND** | **0.0140** | 0.1836 | 1.438 | 0.4833 |
| (a0) `WinProbHead` FRESH | `value_pooled` | rand-branch only | 0.5731 [0.5590, 0.5873] | +0.0008 [−0.0064, +0.0079] **ND** | **0.0133** | 0.1833 | 1.437 | 0.4919 |
| (c0) `WinProbHead` FRESH | `value_pooled` | both branches | 0.5708 [0.5559, 0.5849] | −0.0016 [−0.0096, +0.0062] **ND** | **0.0159** | 0.1825 | 1.449 | 0.4871 |

*base rate 0.4836 on every row. **ND** = NOT DETECTED.*

**Two more registered contrasts:** Δ(d − c) = **+0.0047 [−0.0034, +0.0133] NOT DETECTED**;
Δ(e − c) = **−0.0040 [−0.0087, +0.0008] NOT DETECTED**.

**Against the registered bars**

| bar | outcome |
|---|---|
| **BAR 1 — a fit's CI lower bound > 0.60** | **FAILED on all seven.** Best lower bound **0.5613** (fit d), 0.039 short |
| **BAR 2 — that fit's paired Δ clear of zero** | **Does not arise** (no fit passed BAR 1). For completeness, no Δ is clear of zero |
| **BAR 3 — ECE not worse by > 0.03** | **MET with enormous room and in the opposite direction:** every fit is **0.045–0.059 BETTER** than the original |
| **BAR 4 — guard within 0.10** | **MET by all seven.** Worst deviation 0.0315 (fit d) |
| **BAR 5 — the battery** | **NOT RUN**, as registered: conditional on BAR 1 |

**Which branch? (c), cleanly.** No fit clears 0.60; (d) does not beat (c); the battery is not
reached. **A stationary target on decorrelated positions is not what bounds this metric either.**

### 4.1 Four levers, measured separately, and each one null on ranking

* **WARM vs FRESH init:** (a) 0.5713 vs (a0) 0.5731; (c0) 0.5708. 75M steps of trunk-coupled head
  training is worth **nothing** on this metric over a head fitted from scratch in 1.1 seconds on
  the same frozen features. **The head is not where the information is.**
* **ONE label per game vs BOTH siblings:** (a) 0.5713 vs (b) 0.5681, |Δ| 0.0032 — P4 held. The
  `top1` sibling's label IS the recorded game's own outcome under CRN, so the paired set doubles
  the rows while adding ≈0 independent labels, and it reads that way.
* **CAPACITY:** a 2-layer 256-wide head (≈99k params) reads exactly what the 17k-param bottleneck
  reads. **Not capacity.**
* **INPUT WIDTH:** 13× more input (the 12 pre-pool tokens) buys +0.005, not detected.

## 5. What this says about the question the campaign asked

**The leaf question closes at this depth, and it closes with a sharper statement than "0.575".**

Six levers have now been varied against this metric — coverage (`ctrl10M` vs 75M), training-state
distribution (the fork arm), policy narrowness (the exploiter), loss form (the ranking term, three
times), head capacity and input width (here), and now **target stationarity and position
decorrelation** (here). **None has moved it.** Twelve heads read 0.567–0.578 on a contested fork
set.

Three claims, stated as claims:

1. **The bound is not in the head, the loss, the data distribution or the target's stationarity.**
   A head fitted from scratch in one second on frozen features matches a head trained for 75M
   steps, and every structural variation lands inside one interval. Whatever limits this is
   upstream of the readout — it is what `value_pooled` CONTAINS about a successor state, or it is
   the irreducible noise in a single outcome label at this game's variance.
2. **"The head cannot rank siblings" was too coarse, and §1.4 replaces it.** The head ranks a
   played move against a random one at **0.594**, and its own top-1 against its own top-2 at
   **0.545** — and only the second is what a search leaf is for. A re-ranking search asks the
   critic to break ties the policy has already declared nearly even, and that is the column where
   the critic is barely above chance. **This is why every battery has read no dividend**: the
   dividend requires exactly the sub-task the head is worst at, and the pooled metric was hiding
   how much worse.
3. **An offline frozen-policy fit is nevertheless a cheap, real calibration instrument.** ECE
   0.072 → 0.013 and a 7-point over-confidence removed, for ~4.5 h of CPU and a 1-second fit, with
   the opponent-conditioning guard intact. If a future use needs a well-calibrated `P(win)` from an
   existing checkpoint — a risk-modulation probe, a PBRS potential source, a label for a search
   prior — this recipe delivers it and needs no GPU and no retraining. It just does not deliver a
   leaf.

**What remains, named as the registration required.** With target stationarity eliminated, the
open moves on the leaf are, in cost order:

* **A HAND EVALUATOR AS A CONTROL, on these exact forks.** A Foul-Play-style heuristic scored on
  the banked 5,040-fork set would separate *"the task is hard at this noise"* from *"the learned
  head is bad"* — the single most informative thing left, it costs no training, and **nothing in
  this campaign has ever measured a non-learned baseline on this instrument.** If a heuristic also
  reads 0.545 on `top1|top2`, the ceiling is the game and the leaf question is answered.
* **ROLLOUT-AVERAGED LEAVES.** A leaf that is N frozen-policy rollouts from the successor rather
  than one forward pass. The labels here are single-rollout outcomes; the `top1|top2` column's
  0.545 may be a label-noise floor rather than a representation floor, and averaging attacks
  exactly that. It is what the rust search driver was built for.
* **THE PER-ACTION Q HEAD** (`q_winprob_head`, built and OFF), which amortizes the one-ply leaf
  instead of re-deriving it from a state value — and is scored on the `top1|top2` column directly.
* ⚠️ **What is NOT worth another arm:** another value-head objective read through the pooled
  metric. Six levers, twelve heads, one interval.

🚨 **The honest limits.** (i) §1.4 and §1.5 are **POST-HOC** — they were computed after the
registered read and are reported as decompositions of it, not as registered findings; the pair-type
split has ~1,200–1,300 non-tied pairs per cell and its intervals are correspondingly ~1.7× wider
than the pooled one. (ii) This read is **one frozen policy**. Arm W is a 75M generalist trained with
`--critic winprob`; nothing here says a differently-trained trunk's `value_pooled` would behave the
same. (iii) Three forks per game, not one — the AlphaGo recipe's decorrelation is weakened, and
though the split is by BATTLE and every CI bootstraps over FORKS, a strict one-per-game dataset was
not built.

## 6. Hazards and findings about the instruments

Every one of these is a finding.

1. 🚨 **`vf_features` IS NOT A PRE-POOL TENSOR, and a fit on it would have been vacuous.** The
   extractor returns `activation(value_projection(value_pre_norm(vf_combined)))`, and since the
   critic-route deletion wave **`vf_combined` IS `value_pooled`** (`src/agents/model/projection.py`
   returns `pi_combined, value_pooled`). So `vf_features` is a deterministic function of
   `value_pooled` with no information it lacks, and "fit the wide head on `vf_features`, which has
   more information" describes a cell that cannot exist. Caught by reading the extractor before
   writing the fit, declared in `PREDICTION.md` §4 before any data, and substituted with the
   genuine tap. **The general lesson: a tensor's NAME in a design note is not its position in the
   graph, and `(pi_features, vf_features)` in particular sounds pre-pool and is post-pool.**
2. 🚨 **A LIVE eval tree cannot host the conditioning guard.** `refit.conditioning_frame` reads
   `plan.json` for the opponent KINDS, and a live cycle writes no `plan.json` — only a generated
   one does. A bots-free tree cannot host it either (one opponent class ⇒ the AUC is undefined,
   which is how it failed on the exploiter's slice). Both are properties of the tool, not of the
   run, and both cost a separate small generated tree (5.5 min here). Worth knowing before
   registering a guard on a frame that cannot carry it.
3. 🚨 **A CONTESTED-GAP SELECTOR IS NOT AN OUTCOME-RELEVANCE SELECTOR.** On the matched
   `(top1, rand)` pair the contested set is LESS decidable (0.273) than the uniform-depth set
   (0.286). The selector picks states where the POLICY is indifferent, which is correlated with —
   but not the same as — states where the OUTCOME is undetermined. Every published level on this
   metric is on the contested population and inherits this.
4. 🚨 **A POOLED RANK METRIC OVER HETEROGENEOUS PAIR TYPES HIDES A 0.049 SPREAD.** §1.4. Reporting
   0.575 for a year-long campaign's central instrument concealed that two of its three constituent
   tasks are meaningfully different in difficulty, and that the leaf-relevant one is the worst.
   Every future read on this instrument should print the three columns.
5. **The `rand` branch stall-caps 37× more often than `top1`** (0.207 % vs 0.006 %), so the
   capped-branch exclusion is not symmetric across branches. Immaterial at this rate; material at
   a deeper or longer fork budget.
6. **A `--leaf-head` swap is shape-locked to `WinProbHead`.** `install_leaf_head` requires a bare
   state_dict whose keys match the checkpoint's `win_head`, so the wide fits (c/d/e) have no
   runnable battery. Registered as an obstacle before the data; it did not bind, because no fit
   passed. Only `leafhead_{a,b,a0,c0}.pt` were written in the battery-loadable form.
7. **The box was contended for most of the window** (the ERA-1 fold on the GPU, load 13–25 on 16
   cores) and idle for the last hour. Every wall here is a descriptor; the measurement itself is
   timing-independent, which is what the 462 determinism re-runs assert.
8. **Nothing under `src/` was changed by this read**, so no test or gate obligation attaches. Every
   instrument it calls is invoked by path, unmodified; the new code is in this directory.

## 7. Files

| | |
|---|---|
| [`PREDICTION.md`](PREDICTION.md) | the registration, landed on main at `a78d4b30` before the first battle |
| [`run_trees.sh`](run_trees.sh) · [`tree_manifest_{train,eval,guard}.json`](tree_manifest_train.json) | the three trees and the self-play posing, with the manifests as `eval_trace_gen` wrote them |
| [`branch_forks.py`](branch_forks.py) · [`run_forks.sh`](run_forks.sh) | the AlphaGo branched builder (the 2026-09-14 CRN machinery imported, the SELECTION rule replaced) and the driver that also invokes the contested builder by path |
| [`fork_meta_branch.json`](fork_meta_branch.json) · [`fork_meta_eval.json`](fork_meta_eval.json) | the twelve shards' own metadata — selectors, gap thresholds, determinism checks, walls |
| [`fit_heads.py`](fit_heads.py) · [`run_fit.sh`](run_fit.sh) · [`fit_report.json`](fit_report.json) | the seven fits, the held-out read, the guard, and every descriptor |
| [`run_verify.sh`](run_verify.sh) · [`verify_indexing.json`](verify_indexing.json) | rule 25's indexing clause, executed by the 2026-09-18 verifier — 8.8e-9 agreement plus ten hand-scored pairs |
| [`posthoc.py`](posthoc.py) · [`posthoc.json`](posthoc.json) | POST-HOC: the same heads on held-out BRANCHED pairs (0.60–0.62), and the matched `(top1, rand)` decidability |
| [`pairtype.py`](pairtype.py) · [`pairtype.json`](pairtype.json) | POST-HOC: the contested read decomposed by PAIR TYPE — the 0.049 spread |
| [`run_battery.sh`](run_battery.sh) | the battery as it WOULD be run — kept unrun, on the registered ground (§4) |
| not committed | the three trees, the two fork sets and the fitted head weights, under `/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit/` |

## 8. The registered predictions, scored

`PREDICTION.md`, landed at `a78d4b30` before the first battle. **Twelve of thirteen held.**

| # | registered | outcome |
|---|---|---|
| **P1** | the verdict is **branch (c)**; no fit's CI lower bound clears 0.60 | **HELD.** Best lower bound 0.5613 |
| **P2** | the ORIGINAL head reads **0.55–0.60** on the read set | **HELD.** 0.5723 [0.5583, 0.5866] |
| **P3** | (a) reads **0.555–0.605**, ≈0.578; Δ vs original **−0.01 to +0.03**, NOT DETECTED | **HELD on both clauses.** 0.5713; Δ −0.0011 [−0.0083, +0.0063] |
| **P4** | (b) within **0.015** of (a) | **HELD.** 0.0032 |
| **P5** | (c) reads **0.53–0.60**, at or below (a); (c0) within **0.02** of (c) | **HELD.** 0.5713 (equal to (a) to 4 dp); (c0) 0.5708, 0.0005 apart |
| **P6** | (d) within **0.02** of (c), Δ(d−c) NOT DETECTED — branch (d) does not fire | **HELD.** +0.0047 [−0.0034, +0.0133] |
| **P7** | (e) within **0.02** of (c) | **HELD.** −0.0040; the registered "thin pairs" caveat also held — the term saw 3,954 |
| **P8** | the branched set's non-tied rate is **8–20 %**, materially BELOW the contested set's ~19 % | **REFUTED in both clauses, and the direction is a finding (§6.3):** 28.6 % branched, and on the MATCHED `(top1, rand)` pair it is 0.286 against the contested set's 0.273 — the branched set is MORE decidable, not less |
| **P9** | the `rand` branch's cap rate EXCEEDS `top1`'s | **HELD, and by 37×** (0.207 % vs 0.006 %) |
| **P10** | the branched base rate is **0.45–0.55** | **HELD.** 0.4749 |
| **P11** | zero determinism failures over ≥200 re-runs; **< 2 %** capped | **HELD on both.** 462 re-runs, 0 disagreements; 59 of 50,928 = **0.116 %** |
| **P12** | the original guard reads **0.60–0.80**; every fit within **0.10** | **HELD.** 0.7384; worst deviation 0.0315 |
| **P13** | the battery is **NOT RUN** | **HELD** |

**The one refutation, P8, is the one that produced a finding** — and it is the thread that leads to
§1.4, which is the read's most useful output and was not registered at all. **Everything that WAS
registered came back null**, which is what a sixth consecutive null on a saturated instrument looks
like; the reason to report it loudly rather than quietly is that it eliminates the design
`UNDERSTANDING.md` named as the last untried one.

## 9. Ready-to-append ledger paragraph

> ### 🎯 DOES AN OFFLINE FIT AGAINST A FROZEN POLICY CLEAR THE 0.575 SIBLING-RANKING CEILING? — AlphaGo's recipe on arm W: **NO, branch (c), all seven fits inside one interval** — 🚨 **but the pooled 0.575 decomposes into a 0.049 SPREAD across pair types, and the LEAF-RELEVANT one (`top1` vs `top2`) is the WORST at 0.545**, while the offline fit buys a **4–5× CALIBRATION improvement** (ECE 0.072 → 0.013) and no resolution (2026-09-19)
>
> Record `designs/research_state/measurements/offline_leaf_fit_2026-09-18/` (`PREDICTION.md` landed
> on main at **`a78d4b30`** before the first battle). **6,000 frozen-policy self-play games of arm W
> (`ai_v13_02_flywheel_winprob` @ 75,005,952, posed as its own sole sentinel through a shadow run,
> both seats greedy, full pool, self-play win rate 0.4908), 17,904 branched AlphaGo forks / 35,808
> rollouts as the TRAIN set (one uniformly-random legal move at a uniformly-random depth + the
> policy's own top-1 sibling under CRN), and 5,040 fresh CONTESTED CRN forks / 15,120 rollouts from
> a SEPARATE tree as the held-out READ; seven heads fitted on a FROZEN trunk; CPU only, zero errors,
> 462/462 determinism re-runs identical, 59 of 50,928 rollouts stall-capped (0.116 %), nothing
> written under `models/`.**
>
> **THE REGISTERED READ IS A CLEAN NULL.** On 3,795 held-out non-tied contested pairs the seven fits
> read **0.5673–0.5760** against the original head's **0.5723 [0.5583, 0.5866]**; the registered bar
> was a **CI lower bound > 0.60** and the best achieved is **0.5613**. **BAR 1 FAILS on all seven**
> and every paired Δ is NOT DETECTED (largest **+0.0037 [−0.0050, +0.0117]**, fit d). Δ(d−c) =
> **+0.0047 [−0.0034, +0.0133] NOT DETECTED** — the PRE-POOL 12-token memory (12×128 ⊕ `value_pooled`,
> 1,664 dims, captured at the `cls_pool.value_cls_attn` seam) does not beat `value_pooled`, so
> **branch (d) does not fire**. Δ(e−c) = **−0.0040** — the ranking term is the WORST cell for the
> third time. 🚨 **A head fitted FROM SCRATCH in 1.1 s on frozen features matches the head trained
> for 75M steps** (0.5731 vs 0.5723), and a 6× wider head matches the 17k-param bottleneck: **the
> bound is not the head, the loss, the data distribution, the capacity, the input width, or the
> target's stationarity.** Twelve heads now read 0.567–0.578. **Branch (c): the offline frozen-policy
> fit — the design `UNDERSTANDING.md` named as the last untried one — is ELIMINATED.**
>
> 🚨 **WHAT THE FIT DID BUY IS CALIBRATION, AND A LOT OF IT.** On the same rows: **ECE 0.0721 →
> 0.0133–0.0167**, **Brier 0.1927 → 0.1825–0.1838**, `mean_V` **0.5539 → 0.482–0.492** against a base
> rate of **0.4836** — the frozen critic is **over-confident by 7.0 points on its own self-play
> forks** and one pass of offline BCE on branched labels removes essentially all of it with the
> ranking untouched and `cond.opp_class_auc.t4_10` intact (**0.7384 → 0.707–0.733**, worst deviation
> 0.0315, BAR 4 met by all seven). **Stationary targets on decorrelated positions fix the OFFSET and
> do not touch the RESOLUTION** — the G0 bias map's August verdict, reproduced by a different route.
> As an instrument this is cheap and reusable: ~4.5 h of CPU, a 1-second fit, no GPU, no retraining.
>
> 🚨 **AND THE POOLED METRIC WAS HIDING ITS OWN STRUCTURE (POST-HOC, on the identical forks).** Split
> by PAIR TYPE, the original head reads **`top1` vs `rand` 0.5938 [0.5739, 0.6142]** · **`top2` vs
> `rand` 0.5752** · **`top1` vs `top2` 0.5450 [0.5236, 0.5666]** — a **0.049 spread**. **The head
> separates a played move from a random one at 0.59 and its own two best moves at 0.545, and only
> the second is what a search leaf is for**: a re-ranking search breaks ties the policy has already
> declared nearly even. **That is why every leaf battery has read no dividend, and the pooled 0.575
> concealed how much worse the leaf-relevant column is.** Consistently, the SAME heads read
> **0.6035–0.6198** on 981 held-out BRANCHED pairs (original **0.6157 [0.5925, 0.6397]**) — **the
> level is a property of the STATE SET, not of the head.** Every future read on this instrument
> should print the three columns. ⚠️ Post-hoc, ~1,200–1,300 non-tied pairs per cell, intervals ~1.7×
> wider than the pooled one.
>
> **ONE PREDICTION OF THIRTEEN WAS REFUTED, and it is the one that produced a finding.** P8 registered
> that the branched set would be LESS decidable than the contested set; on the MATCHED `(top1, rand)`
> pair it is **MORE** (0.286 vs 0.273). 🚨 **A CONTESTED-GAP SELECTOR IS NOT AN OUTCOME-RELEVANCE
> SELECTOR** — it selects for the POLICY's indifference, and every published level on this metric is
> on that population and inherits it. Other instrument findings: 🚨 **`vf_features` is NOT a pre-pool
> tensor** — since the critic-route deletion wave `vf_combined IS value_pooled`, so `vf_features` is a
> deterministic function of it and the literal "fit on `vf_features` for more information" cell cannot
> exist; caught by reading `projection.py` before writing the fit and declared in the registration.
> 🚨 **A LIVE eval tree cannot host the conditioning guard** (`refit.conditioning_frame` needs a
> `plan.json`, which only a GENERATED cycle writes; a bots-free tree has one opponent class and the
> AUC is undefined) — the guard needs its own small generated tree, 5.5 min. The `rand` branch
> stall-caps **37×** more often than `top1` (0.207 % vs 0.006 %), so the capped-branch exclusion is
> not symmetric — immaterial at this rate, material at a deeper fork budget. `--leaf-head` is
> shape-locked to `WinProbHead`, so the wide fits have no runnable battery; it did not bind because
> **the battery was NOT RUN**, as registered, BAR 5 being conditional on BAR 1.
>
> **WHAT REMAINS ON THE LEAF, in cost order, with target stationarity now eliminated:** (1) **a HAND
> EVALUATOR as a control on the banked 5,040 forks** — nothing in this campaign has ever measured a
> non-learned baseline on this instrument, it costs no training, and if a heuristic also reads ~0.545
> on `top1|top2` the ceiling is the GAME and the leaf question is answered; (2) **rollout-averaged
> leaves** (the labels here are single-rollout outcomes, so 0.545 may be a label-noise floor rather
> than a representation floor — what the rust search driver was built for); (3) **the per-action
> `q_winprob_head`** (built and OFF), scored on the `top1|top2` column directly. ⚠️ **What is NOT
> worth another arm: a seventh value-head objective read through the pooled metric.** Instrument note:
> rule 25's indexing clause was EXECUTED — the 2026-09-18 `verify_indexing.py` invoked by path
> reproduces **0.5723320158** against **0.5723320246** (gap **8.8e-9**, float32-mean precision) with
> ten hand-scored pairs printed; the pre-pool tap carries its own vacuity REFUSAL (shape and
> non-constancy) and the guard frame is asserted equal to `refit.conditioning_frame`'s. Declared
> deviations: three forks per game rather than one (tree throughput; split by BATTLE, every CI
> bootstrapped over FORKS), `rand ~ U(legal \ top1)` so the sibling pair is never degenerate, and the
> self-play posing via a shadow run. Tag: **MEASURED (MAJOR) · BRANCH (c) — the OFFLINE FROZEN-POLICY
> FIT IS ELIMINATED, twelve heads on 0.567–0.578 · the fit buys CALIBRATION (ECE 0.072 → 0.013) and
> NOT RESOLUTION · 🚨 the pooled 0.575 is a MIXTURE and the leaf-relevant `top1|top2` column is 0.545
> · a from-scratch 1-second head matches a 75M-step one · the contested selector is not an
> outcome-relevance selector**.
