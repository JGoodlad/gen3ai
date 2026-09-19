# IS THE 0.545 `top1|top2` LEVEL THE HEADS, THE LABELS, OR THE GAME? — two controls on the exact banked forks

*Measured 2026-09-19 11:55 – 14:05 UTC · **CONTROL 1**: all **5,040 banked held-out contested CRN
forks** re-run and captured, **14,601 / 14,601 successor observations reproduced ELEMENT-WISE**,
thirteen scorers on the identical 3,795 non-tied pairs · **CONTROL 2**: **1,002 forks × 8 fresh
post-divergence dice × 3 branches = 24,048 rollouts, 0 errors, 32 stall-capped (0.13 %)** ·
CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, rust sim bridge, nothing written under `models/`;
a training arm held the GPU throughout and was not touched, ports 8000/8001 untouched.*

**Pre-registered in [`PREDICTION.md`](PREDICTION.md), landed on main at `a72b8b21` before the first
battle of either control.** Scored in §9.

---

## 1. VERDICT — the cell is **N × L**, and branch G is REFUTED on its mechanism

**1 — 🚨 NEITHER THE POLICY NOR THE CRITIC IS SEEING A REAL, LARGE DIFFERENCE. The policy's own
top-1 is worth `+0.0079 [−0.0025, +0.0188]` of win probability over its top-2 — NOT CLEAR OF ZERO —
while the two successors' true values differ by `E|gap| ≈ 0.177` (sd 0.222).** Measured on 1,002
forks × 8 rollouts. Branch **G said the policy's two best moves are near-tied in truth and a search
leaf has nothing to re-rank; that is REFUTED.** They are near-tied *on average* and far apart
*case by case* — which is the definition of a re-ranking opportunity. It is large, and nothing in
this campaign is taking it.

**2 — THE AXIS IS N: NO HAND EVALUATOR BEATS THE HEAD ON THE LEAF COLUMN.** Nine non-learned
scorers — the PRODUCTION PBRS potential composition, that plus the incoming-KO belief potential,
bare material+HP, a type-matchup variant, and their omniscient twins — read **0.5219 to 0.5416** on
`top1|top2` against arm W's **0.5450**. **BAR 1 FAILS on all nine**: every interval contains 0.5450
and every point estimate is below it; every paired Δ is negative and NOT DETECTED.

**3 — 🚨 BUT A HAND-WRITTEN POTENTIAL SUM MATCHES A 75M-STEP CRITIC ON THE PUBLISHED POOLED METRIC,
AND BEATS IT ON A COLUMN.** `PBRS_BELIEF_1S` reads **0.5714 [0.5581, 0.5847]** pooled against the
head's 0.5723 — **Δ −0.0009 [−0.0152, +0.0136] NOT DETECTED** — and on `top2|rand` **0.6007
[0.5806, 0.6200]** against 0.5752, **Δ +0.0255 [+0.0043, +0.0461] DETECTED**. The mechanism is
measured: Spearman(head, `PBRS_1S`) = **0.768** over 14,570 successors, and their across-state AUC
against the outcome is **0.789 vs 0.763**. **The head is a slightly better material counter and
essentially nothing else.**

**4 — 🚨 THE SINGLE-ROLLOUT LABEL IS WRONG ABOUT THE ORDER 39.7 % OF THE TIME on `top1|top2`**
(single-vs-averaged order agreement **0.6034**; 0.6375 pooled). **BAR 4's branch-G clause FAILS.**
So the published 0.545 is not a measurement of resolution — it is a measurement of resolution
*through* a label that is close to a coin flip about which sibling was better. Formally: **46.7 %
of the variance in the observed sibling-label difference is label noise** (observed 0.0925,
noise 0.0432, true 0.0493).

**5 — 🚨 AND A FOUR-ROLLOUT LEAF BEATS THE 75M-STEP CRITIC, ON A MATCHED-NOISE READ.** One K = 4
label, two scorers, the same pairs: an INDEPENDENT 4-rollout estimate of the very same successors
reads **0.6172 [0.5995, 0.6354]** pooled against the head's **0.5607** — **Δ +0.0566 [+0.0282,
+0.0831] DETECTED**, and DETECTED on both `rand` columns too (+0.0717, +0.0678). **That is
branch L**: headroom exists and it is reachable with four rollouts. ⚠️ **On `top1|top2` alone it is
+0.0275 [−0.0176, +0.0705] NOT DETECTED** — the leaf column is where four rollouts also run out of
resolution.

**6 — THE LITERAL BAR 2 FAILS AND WAS REGISTERED AS VACUOUS-UNTIL-CEILINGED.** The head's
`top1|top2` against the K = 8 averaged label is **0.5610 [0.5313, 0.5913]** — it ROSE from 0.5450,
and it does not clear 0.60. But **0.60 was never the ceiling**: the measured ceiling on that column
is a split-half agreement of **0.6303**, and an 8-rollout estimate scored on the banked SINGLE
labels reads **0.5811 [0.5390, 0.6259]** where the head reads 0.5405 on the same pairs. Scoring
against a 0.60 bar without measuring that would have called a clean result a failure.

**7 — POST-HOC, TWO MORE MIXTURES IN THE PUBLISHED NUMBER.** Split by DEPTH: 19.4 % of successors
sit at the fork's OWN turn (a KO forced a replacement), only 76.5 % of forks have all three
branches at one depth, and the head reads **0.6457 on mixed-depth pairs against 0.5571 on matched
ones**. **The genuinely hard cell — the policy's two best moves, compared at the same depth — is
`0.5355 [0.5117, 0.5583]`, and all thirteen scorers sit in 0.514–0.546 there.**

**8 — 🚨 OMNISCIENCE IS EXACTLY FREE IN GEN 3 FOR THIS EVALUATOR FAMILY** (P5, registered in
advance): over 14,601 successors the one-sided and referee readouts agree on the opponent's alive
count **14,601 / 14,601**, on its tempo-status count **14,601 / 14,601**, and on its HP sum to a
mean **|Δ| 0.0074 of ~6.0 (0.12 %)** — and not because the opponent is revealed (**1.94 of 6** mons
are unrevealed at the average successor). **The hidden information in gen 3 is IDENTITY and
MOVESET, not MATERIAL.**

## 2. The frame

| | |
|---|---|
| **the policy** | `models/ai_v13_02_flywheel_winprob/final_model.zip` — **arm W**, step **75,005,952**, `--critic winprob`, a fresh generalist over the full 719-team pool. FROZEN; nothing was trained in this read |
| **the states** | the **banked** 5,040 contested CRN forks of [`offline_leaf_fit_2026-09-18/`](../offline_leaf_fit_2026-09-18/README.md) — `top1` + `top2` + `rand` per fork, one dice stream, both seats greedy, under `/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit/forks_eval` |
| **the metric** | `sign(V(s'_a) − V(s'_b))` against the branch outcome on non-tied pairs, bootstrapped over **FORKS** (2,000 resamples). `refit.load_forks` / `build_table` / `read_head` / `paired_delta` / `boot_ci` are imported **BY PATH** and unmodified; so is `score_forks.branch_index` (the alignment assertion) and `fit_heads.forward_feats` (the frozen trunk) |
| **the second fork set** | `fork_arm_read_2026-09-16/` was NOT reused: its forks are the FORK ARM's own training-state distribution — a different population — and its rows live in `rows_*.jsonl.gz` in a different schema. One population with all thirteen scorers was the better trade, and it is declared |
| regime | CPU only, `nice 15`, rust bridge (`strings … \| grep gen3ai-wt` → 0: the 2026-09-09 worktree-path contamination class does not apply), everything under `.../tmp/leaf_control/`, **nothing under `models/`** |

## 3. THE INSTRUMENT IS PROVEN, NOT ARGUED — six checks, all clean

| check | result |
|---|---|
| **the re-run reproduces the banked successor OBSERVATION element-wise** | **14,601 / 14,601 (100.0 %)**, **0 mismatches**, over all six shards. This is the join. A hand score computed on a different state than the head saw would not be a control, and nothing is taken on trust: the CRN replay recomputes the 2501-dim vector and it is `array_equal` to `succ_<tag>.npy[local]` |
| **the re-run reproduces the banked OUTCOME** | **138 / 138** forks run to a terminal with no forfeit returned the banked win/loss |
| **the opponent's own view was live at every capture** | **14,601 / 14,601**, and its `battle.turn` equalled ours on **14,601 / 14,601** — the omniscient half is one instant of one battle, not two runs stitched together |
| **rule 25's indexing clause, EXECUTED** | [`../exploiter_discrimination_2026-09-18/verify_indexing.py`](../exploiter_discrimination_2026-09-18/verify_indexing.py), invoked by path with its own reader and pair enumeration, reproduces **0.5723320158** against **0.5723320246** — a gap of **8.79e-9**, float32-mean precision ([`verify_indexing.json`](verify_indexing.json)) |
| **arm W's head reproduces its published levels exactly** | **0.5723 pooled · 0.5938 `top1\|rand` · 0.5752 `top2\|rand` · 0.5450 `top1\|top2`** — identical to `offline_leaf_fit_2026-09-18/pairtype.json` to four decimals on all four |
| errors · capture failures · re-roll errors | **0 · 0 · 0** (24,048 re-rolls, 32 stall-capped = 0.13 %, scored 0.5 each) |

**BAR 5 MET, with room.**

## 4. CONTROL 1 — the hand evaluator as a leaf

### 4.1 What was scored

Nine hand scorers, none of them learned, all read from the board at the successor decision.
**The potentials are the PRODUCTION ones** — `agents.training.reward_potentials` imported and
called, never re-implemented; `_opp_spikes` is borrowed from `RewardBiasTerms`, which is what
`Gen3RewardManager` itself resolves.

| scorer | what it is |
|---|---|
| `PBRS_1S` | Φ_mat + Φ_status + Φ_hazard + Φ_boost + Φ_opp_boosts + Φ_roar on the one-sided `LiveView` — **the hand-written evaluator the shaped critic regressed** |
| `PBRS_BELIEF_1S` | the same plus Φ_belief (expected surviving material discounted by the active mon's outspeed-adjusted incoming P(KO)) — the only term with a damage model behind it |
| `BELIEF_1S` | Φ_belief alone |
| `MAT_1S` | `2.0·(Σ our hp − Σ opp hp) + 1.25·(our alive − opp alive)`, unrevealed opponent slots full-HP-alive |
| `TYPE_1S` | `MAT` plus twice a best-STAB type-matchup edge over the opponent's REVEALED alive team |
| `*_OMNI` | the same with the opponent's **TRUE** team (HP, alive, status, and for `TYPE_OMNI` its unrevealed species) read off the OPPONENT player's own `LiveView` at the same instant |

Beside them: arm W's head, and three 2026-09-18 offline fits loaded from their banked weights —
`fit_d_wide_prepool` (the best of the seven), `fit_c_wide_pooled`, `fit_a_winprob_warm`.

### 4.2 The table

### single-rollout labels — pairwise accuracy [95 % CI over forks]

| scorer | POOLED | rand\|top1 | rand\|top2 | top1\|top2 |
|---|---|---|---|---|
| **headW** | **0.5723 [0.5583, 0.5866]** | **0.5938 [0.5739, 0.6142]** | **0.5752 [0.5549, 0.5957]** | **0.5450 [0.5236, 0.5666]** |
| fit_c_wide_pooled | 0.5713 [0.5562, 0.5858] | 0.5945 [0.5738, 0.6146] | 0.5760 [0.5553, 0.5962] | 0.5398 [0.5178, 0.5620] |
| fit_d_wide_prepool | 0.5760 [0.5613, 0.5903] | 0.6014 [0.5813, 0.6219] | 0.5722 [0.5522, 0.5922] | 0.5518 [0.5302, 0.5733] |
| fit_a_winprob_warm | 0.5713 [0.5569, 0.5853] | 0.5922 [0.5725, 0.6120] | 0.5821 [0.5614, 0.6031] | 0.5356 [0.5146, 0.5576] |
| MAT_1S | 0.5632 [0.5503, 0.5762] | 0.5716 [0.5527, 0.5911] | 0.5790 [0.5592, 0.5976] | 0.5360 [0.5157, 0.5568] |
| PBRS_1S | 0.5677 [0.5543, 0.5810] | 0.5766 [0.5568, 0.5964] | 0.5866 [0.5666, 0.6057] | 0.5364 [0.5159, 0.5576] |
| **PBRS_BELIEF_1S** | **0.5714 [0.5581, 0.5847]** | 0.5728 [0.5525, 0.5922] | **0.6007 [0.5806, 0.6200]** | **0.5368 [0.5157, 0.5565]** |
| BELIEF_1S | 0.5491 [0.5367, 0.5617] | 0.5541 [0.5355, 0.5722] | 0.5684 [0.5499, 0.5866] | 0.5219 [0.5054, 0.5385] |
| TYPE_1S | 0.5416 [0.5276, 0.5553] | 0.5412 [0.5211, 0.5608] | 0.5498 [0.5291, 0.5687] | 0.5330 [0.5128, 0.5540] |
| MAT_OMNI | 0.5625 [0.5495, 0.5755] | 0.5705 [0.5513, 0.5896] | 0.5790 [0.5590, 0.5977] | 0.5347 [0.5146, 0.5559] |
| PBRS_OMNI | 0.5672 [0.5537, 0.5806] | 0.5758 [0.5563, 0.5957] | 0.5874 [0.5672, 0.6068] | 0.5347 [0.5140, 0.5562] |
| PBRS_BELIEF_OMNI | 0.5704 [0.5570, 0.5838] | 0.5728 [0.5525, 0.5922] | 0.5992 [0.5794, 0.6185] | 0.5351 [0.5138, 0.5546] |
| TYPE_OMNI | 0.5441 [0.5301, 0.5582] | 0.5438 [0.5231, 0.5636] | 0.5467 [0.5271, 0.5659] | 0.5416 [0.5208, 0.5627] |

### paired Δ vs arm W's head, same pairs

| scorer | POOLED | rand\|top1 | rand\|top2 | top1\|top2 |
|---|---|---|---|---|
| fit_c_wide_pooled | −0.0011 [−0.0088, +0.0063] ND | +0.0008 [−0.0110, +0.0134] ND | +0.0008 [−0.0110, +0.0120] ND | −0.0051 [−0.0177, +0.0079] ND |
| fit_d_wide_prepool | +0.0037 [−0.0050, +0.0117] ND | +0.0076 [−0.0049, +0.0212] ND | −0.0030 [−0.0159, +0.0094] ND | +0.0069 [−0.0068, +0.0203] ND |
| fit_a_winprob_warm | −0.0011 [−0.0083, +0.0063] ND | −0.0015 [−0.0135, +0.0109] ND | +0.0068 [−0.0048, +0.0180] ND | −0.0094 [−0.0217, +0.0027] ND |
| MAT_1S | −0.0091 [−0.0245, +0.0061] ND | −0.0221 [−0.0429, −0.0006] **DET** | +0.0038 [−0.0184, +0.0250] ND | −0.0090 [−0.0341, +0.0163] ND |
| PBRS_1S | −0.0046 [−0.0194, +0.0101] ND | −0.0171 [−0.0373, +0.0036] ND | +0.0114 [−0.0098, +0.0321] ND | −0.0086 [−0.0324, +0.0158] ND |
| **PBRS_BELIEF_1S** | **−0.0009 [−0.0152, +0.0136] ND** | −0.0210 [−0.0415, +0.0000] ND | **+0.0255 [+0.0043, +0.0461] DET** | −0.0081 [−0.0310, +0.0159] ND |
| BELIEF_1S | −0.0232 [−0.0396, −0.0065] **DET** | −0.0396 [−0.0628, −0.0171] **DET** | −0.0068 [−0.0292, +0.0149] ND | −0.0231 [−0.0479, +0.0027] ND |
| TYPE_1S | −0.0307 [−0.0473, −0.0141] **DET** | −0.0526 [−0.0762, −0.0281] **DET** | −0.0255 [−0.0503, −0.0012] **DET** | −0.0120 [−0.0373, +0.0136] ND |
| MAT_OMNI | −0.0099 [−0.0254, +0.0051] ND | −0.0232 [−0.0439, −0.0018] **DET** | +0.0038 [−0.0184, +0.0251] ND | −0.0103 [−0.0357, +0.0150] ND |
| PBRS_OMNI | −0.0051 [−0.0199, +0.0093] ND | −0.0179 [−0.0380, +0.0030] ND | +0.0122 [−0.0091, +0.0332] ND | −0.0103 [−0.0340, +0.0142] ND |
| PBRS_BELIEF_OMNI | −0.0020 [−0.0163, +0.0127] ND | −0.0210 [−0.0415, +0.0000] ND | +0.0239 [+0.0029, +0.0449] **DET** | −0.0099 [−0.0328, +0.0145] ND |
| TYPE_OMNI | −0.0282 [−0.0453, −0.0120] **DET** | −0.0499 [−0.0736, −0.0254] **DET** | −0.0285 [−0.0536, −0.0036] **DET** | −0.0034 [−0.0285, +0.0231] ND |

### 4.3 What it says

**BAR 1 FAILS ON ALL NINE, so the axis is N.** The best hand scorer on `top1|top2` is
`PBRS_BELIEF_1S` at **0.5368 [0.5157, 0.5565]** — its interval CONTAINS arm W's 0.5450 and its
point estimate is BELOW it. Every one of the nine paired Δ on that column is negative and NOT
DETECTED, the largest being `BELIEF_1S` at −0.0231 [−0.0479, +0.0027]. **Nothing a heuristic knows
separates the policy's own two best moves better than the head does.**

**The pooled and `top2|rand` results are the surprise.** A sum of six hand-written potentials, with
no training at all, is statistically indistinguishable from a 75M-step win-prob critic on the
campaign's headline metric, and DETECTABLY better at "the policy's second-best move vs a random
legal one". The head keeps `top1|rand` (hand −0.0171 to −0.0526, four of them DETECTED): it knows
its own best move is good better than a material count does, and that is all it knows.

**Why, measured** ([`posthoc_corr.json`](posthoc_corr.json)): over the 14,570 scored successors,
Spearman(head, `PBRS_1S`) = **0.768**, and the across-state AUC against the branch outcome is
**headW 0.789 vs PBRS_1S 0.763 vs MAT_1S 0.749**. **Across states both are strong; within a fork,
between two states one move apart, both collapse to ~0.54.** What distinguishes siblings is not
material, and 75M steps did not teach the head anything that is.

### 4.4 Omniscience is free — exactly, not approximately

[`posthoc_omni.json`](posthoc_omni.json); P5 registered in advance.

| quantity, one-sided minus TRUE | over 14,601 successors |
|---|---|
| opponent **alive count** | **exactly equal on 14,601 / 14,601 (100 %)** |
| opponent **tempo-status count** (par/slp/frz) | **exactly equal on 14,601 / 14,601** |
| opponent **HP sum** (of ~6.0) | mean \|Δ\| **0.0074** (0.12 %), sd 0.0063, **max 0.036** — percent rounding, nothing else |
| and not because the opponent is revealed | **1.94 of 6** opponent mons unrevealed at the average successor; 627 successors have only ONE revealed |

An unrevealed mon has never been on the field, so it is at full HP, alive and unstatused — exactly
what the one-sided Φ_mat already assumes for a declared-but-unseen slot. Every `*_OMNI` scorer
lands within 0.001–0.007 of its one-sided twin on every column. Two consequences worth carrying:
a "hidden-information floor" on a gen-3 value function cannot be a floor about MATERIAL, and the
cheapest omniscient referee readout buys a value function nothing at all.

### 4.5 POST-HOC — the pooled metric is a mixture in DEPTH too

**19.4 % of successors (2,826 / 14,601) sit at the SAME turn as the fork**: the branch KO'd
something and the forced replacement re-opened a decision inside the fork turn. **Only 76.5 % of
forks have all three branches' successors at the same turn** — so nearly a quarter of pairs compare
states at DIFFERENT depths, and a depth asymmetry is a KO marker.

| arm W's head | same-depth pair | different-depth pair |
|---|---|---|
| POOLED | **0.5571 [0.5413, 0.5729]** (3,143 nt) | **0.6457 [0.6141, 0.6806]** (652 nt) |
| `top1\|rand` | 0.5710 (1,070 nt) | 0.6942 (242 nt) |
| `top2\|rand` | 0.5637 (1,059 nt) | 0.6226 (257 nt) |
| **`top1\|top2`** | **0.5355 [0.5117, 0.5583]** (1,014 nt) | 0.6078 (153 nt) |

**The genuinely hard cell — the policy's own two best moves, compared at the SAME depth — is
0.5355, and all thirteen scorers sit in 0.514–0.546 there with overlapping intervals.** The
published 0.5723 is an average over "one branch got a KO" (0.65) and "neither did" (0.56), and over
three pair types on top of that.

## 5. CONTROL 2 — rollout-averaged labels, and the ceilings

**1,002 forks × K = 8 fresh post-divergence dice × 3 branches = 24,048 rollouts, 0 errors,
32 capped (0.13 %).** CRN within a re-roll index (the k-th `top1` and the k-th `top2` share one
seed); the K seeds are independent of the banked realized stream.

### 5.1 The K-averaged read

| scorer | POOLED | rand\|top1 | rand\|top2 | top1\|top2 |
|---|---|---|---|---|
| **headW** | **0.5801 [0.5592, 0.6005]** | 0.5904 [0.5600, 0.6197] | 0.5870 [0.5563, 0.6162] | **0.5610 [0.5313, 0.5913]** |
| fit_c_wide_pooled | 0.5718 [0.5515, 0.5926] | 0.5888 [0.5587, 0.6169] | 0.5667 [0.5366, 0.5961] | 0.5582 [0.5281, 0.5892] |
| fit_d_wide_prepool | 0.5661 [0.5451, 0.5865] | 0.5855 [0.5558, 0.6156] | 0.5583 [0.5267, 0.5873] | 0.5527 [0.5234, 0.5831] |
| fit_a_winprob_warm | 0.5764 [0.5553, 0.5965] | 0.5953 [0.5661, 0.6237] | 0.5777 [0.5462, 0.6075] | 0.5536 [0.5245, 0.5853] |
| MAT_1S | 0.5518 [0.5324, 0.5701] | 0.5717 [0.5452, 0.5987] | 0.5439 [0.5155, 0.5730] | 0.5379 [0.5104, 0.5666] |
| PBRS_1S | 0.5555 [0.5358, 0.5745] | 0.5700 [0.5429, 0.5979] | 0.5541 [0.5251, 0.5840] | 0.5407 [0.5118, 0.5696] |
| PBRS_BELIEF_1S | 0.5507 [0.5307, 0.5698] | 0.5513 [0.5233, 0.5787] | 0.5667 [0.5384, 0.5960] | 0.5323 [0.5044, 0.5611] |
| BELIEF_1S | 0.5375 [0.5192, 0.5562] | 0.5375 [0.5117, 0.5637] | 0.5583 [0.5318, 0.5842] | 0.5148 [0.4911, 0.5394] |
| TYPE_1S | 0.5195 [0.4991, 0.5382] | 0.5309 [0.5026, 0.5593] | 0.5228 [0.4935, 0.5522] | 0.5028 [0.4740, 0.5313] |
| MAT_OMNI | 0.5504 [0.5308, 0.5690] | 0.5684 [0.5422, 0.5957] | 0.5439 [0.5155, 0.5730] | 0.5370 [0.5098, 0.5663] |
| PBRS_OMNI | 0.5535 [0.5339, 0.5723] | 0.5651 [0.5382, 0.5929] | 0.5541 [0.5251, 0.5840] | 0.5397 [0.5103, 0.5687] |
| PBRS_BELIEF_OMNI | 0.5498 [0.5299, 0.5689] | 0.5513 [0.5233, 0.5787] | 0.5650 [0.5359, 0.5945] | 0.5314 [0.5030, 0.5599] |
| TYPE_OMNI | 0.5232 [0.5031, 0.5422] | 0.5366 [0.5090, 0.5669] | 0.5186 [0.4892, 0.5478] | 0.5129 [0.4836, 0.5419] |

*2,996 scorable pairs; 1,747 non-tied pooled, 541 on `top1|top2`.*

**The head RISES under averaged labels — 0.5450 → 0.5610 on `top1|top2`, 0.5723 → 0.5801 pooled —
and does not clear 0.60. BAR 2 FAILS.** Every hand evaluator falls FURTHER behind the head under
averaged labels (0.5028–0.5407 on `top1|top2`): the noise the single label carried was flattering
them.

### 5.2 🚨 THE MATCHED-NOISE READ — one K = 4 label, two scorers, the same pairs

The decisive instrument, and the only one with no modelling in it at all. `ROLLOUT_Khalf` is the
mean of re-rolls 0–3; the LABEL is the mean of re-rolls 4–7. Independent dice, same states, so
whatever `ROLLOUT_Khalf` reads is what **a value function of four-rollout quality** reads on this
label — and the head is read against the identical label on the identical pairs.

| scorer | POOLED | rand\|top1 | rand\|top2 | top1\|top2 |
|---|---|---|---|---|
| **ROLLOUT_Khalf (4 rollouts)** | **0.6172 [0.5995, 0.6354]** | **0.6275 [0.6026, 0.6541]** | **0.6245 [0.5984, 0.6527]** | **0.5980 [0.5705, 0.6241]** |
| **headW** | 0.5607 [0.5384, 0.5829] | 0.5558 [0.5225, 0.5899] | 0.5567 [0.5236, 0.5911] | 0.5705 [0.5363, 0.6034] |
| fit_c_wide_pooled | 0.5621 [0.5385, 0.5844] | 0.5757 [0.5447, 0.6085] | 0.5455 [0.5114, 0.5794] | 0.5650 [0.5316, 0.5976] |
| fit_d_wide_prepool | 0.5510 [0.5280, 0.5734] | 0.5777 [0.5460, 0.6118] | 0.5253 [0.4906, 0.5605] | 0.5496 [0.5154, 0.5819] |
| PBRS_1S | 0.5431 [0.5218, 0.5642] | 0.5329 [0.5015, 0.5641] | 0.5476 [0.5166, 0.5780] | 0.5496 [0.5180, 0.5828] |
| MAT_1S | 0.5341 [0.5128, 0.5546] | 0.5269 [0.4952, 0.5577] | 0.5324 [0.5032, 0.5637] | 0.5441 [0.5122, 0.5747] |
| **ROLLOUT − head** | **+0.0566 [+0.0282, +0.0831] DET** | **+0.0717 [+0.0346, +0.1093] DET** | **+0.0678 [+0.0286, +0.1083] DET** | **+0.0275 [−0.0176, +0.0705] ND** |

*1,450 non-tied pooled, 454 on `top1|top2`.*

**BAR 3 is MET pooled and on both `rand` columns: headroom exists, it is DETECTED, and four
rollouts reach it. That is branch L.** ⚠️ **On `top1|top2` it is NOT DETECTED** — four rollouts run
out of resolution on exactly the column a re-ranking leaf consumes, at n = 454.

### 5.3 The ceilings, and the recovered gap

| quantity | POOLED | rand\|top1 | rand\|top2 | top1\|top2 |
|---|---|---|---|---|
| split-half agreement (K/2 vs K/2) | 0.6425 | 0.6433 | 0.6529 | **0.6303** |
| → implied oracle acc @K = 4 *(cross-check only, §7.4)* | 0.7670 | 0.7677 | 0.7765 | 0.7553 |
| observed var of the label difference | 0.0974 | 0.0988 | 0.0997 | 0.0925 |
| — of which LABEL NOISE | 0.0422 | 0.0420 | 0.0416 | **0.0432 (46.7 %)** |
| **recovered TRUE-gap sd** | **0.2349** | 0.2384 | 0.2411 | **0.2220** |
| **E\|true gap\|** | **0.1874** | 0.1902 | 0.1924 | **0.1771** |
| model oracle @K = 1 *(cross-check only)* | 0.6702 | 0.6723 | 0.6739 | 0.6623 |
| **single-vs-averaged ORDER AGREEMENT** | **0.6375** (n=582) | 0.6275 (n=204) | 0.6765 (n=204) | **0.6034** (n=174) |

**The label-noise rate the instruction asked for is `1 − 0.6034 = 0.397` on `top1|top2`** — a
single rollout puts the two siblings in the wrong order two times in five.

### 5.4 The K-mean AS A SCORER, on the banked SINGLE labels — the model-free ceiling

The K = 8 mean is the best available estimate of each successor's true value, and its dice never
reproduce the banked realized stream, so there is no shared-noise inflation. Read on exactly the
metric the campaign publishes, on the same subset:

| scorer | POOLED | rand\|top1 | rand\|top2 | top1\|top2 |
|---|---|---|---|---|
| **ORACLE_Kfull (8 rollouts)** | **0.6125 [0.5830, 0.6430]** | 0.6111 [0.5667, 0.6563] | 0.6412 [0.6035, 0.6802] | **0.5811 [0.5390, 0.6259]** |
| ORACLE_Khalf (4 rollouts) | 0.6006 [0.5716, 0.6294] | 0.6047 [0.5662, 0.6463] | 0.6333 [0.5975, 0.6727] | 0.5586 [0.5172, 0.6036] |
| **headW** | 0.5359 [0.5032, 0.5702] | 0.5812 [0.5324, 0.6291] | 0.4902 [0.4444, 0.5346] | 0.5405 [0.4928, 0.5882] |
| PBRS_1S | 0.5893 [0.5588, 0.6203] | 0.6368 [0.5896, 0.6809] | 0.5608 [0.5184, 0.6058] | 0.5721 [0.5248, 0.6191] |
| MAT_1S | 0.5809 [0.5518, 0.6109] | 0.6239 [0.5775, 0.6690] | 0.5490 [0.5062, 0.5932] | 0.5721 [0.5274, 0.6181] |

*711 non-tied pooled, 222 on `top1|top2`. ⚠️ This subset is ~5× smaller than the full set and its
per-column head numbers wander accordingly — `headW` reads 0.4902 on `top2|rand` here against
0.5752 on all 1,316 pairs. **The row to read is `ORACLE_Kfull` vs `headW` within this table**, not
either against the §4.2 numbers.*

**On the published metric, an 8-rollout estimate reads 0.6125 pooled where the head reads 0.5359 on
the same pairs. 0.545 is not the label's ceiling — the ceiling is around 0.61 — and the head is not
at it.**

### 5.5 🚨 What the policy's own preference is worth

| quantity, mean K = 8 label over 1,002 forks | |
|---|---|
| `top1` | 0.4936 |
| `top2` | 0.4857 |
| `rand` | 0.4637 |
| **`top1` − `top2`** | **+0.0079 [−0.0025, +0.0188] — NOT CLEAR OF ZERO** |
| `top1` − `rand` | +0.0299 [+0.0181, +0.0417] — clear |
| **E\|true gap\| between `top1` and `top2`** | **0.1771** |

**The policy's own argmax is worth about eight tenths of a point of win probability over its own
runner-up, indistinguishable from zero — while the two successors' true values differ by about
eighteen points in absolute terms.** Those two facts together are the finding of this read: **there
is a great deal to re-rank between the policy's two best moves, and the policy's ordering is
carrying almost none of it.**

## 6. What this says about the campaign's question

**The leaf question does NOT close as "the game has nothing in it". It closes as "the metric was
measuring a coin flip, and the signal it was trying to see is real, large, and unclaimed."**

1. **The 0.545 was never a resolution measurement.** 46.7 % of the variance in the sibling-label
   difference is label noise, the single label mis-orders `top1|top2` 39.7 % of the time, and an
   8-rollout estimate of the same states reads 0.58–0.61 on the same metric. **Every published
   level on this instrument — 0.517, 0.575, 0.587, the twelve heads at 0.567–0.578 — is compressed
   toward 0.5 by the label, and the compression was never quantified until now.**
2. **The learner was never the binding constraint, and now neither is the heuristic.** Six levers
   moved nothing; a hand-written potential sum matches the 75M-step head pooled and beats it on a
   column; and both collapse together on `top1|top2`. **A static board score — learned or written —
   is the wrong class of object for this task.**
3. **A ROLLOUT LEAF IS A DIFFERENT CLASS AND IT PAYS.** Four rollouts beat the head by
   **+0.057 [+0.028, +0.083]** pooled, DETECTED. This is the first thing in the campaign to
   out-rank the critic on this instrument. ⚠️ And on `top1|top2` alone four rollouts are NOT
   detected ahead of it — so the next question is **how many rollouts the leaf column needs**, which
   is a dose curve, not a new objective.
4. **The named next moves, in cost order, with this read's evidence attached:**
   * **A K-ROLLOUT LEAF at increasing K on the `top1|top2` column.** Everything needed exists —
     `reroll_labels.py` is the harness and the rust search driver is the fast path. The registered
     question is the smallest K at which `ROLLOUT_K − head` is DETECTED on `top1|top2`; it is
     +0.0275 [−0.018, +0.071] at K = 4.
   * **THE PER-ACTION Q HEAD** (`q_winprob_head`, built and OFF), which amortizes the ply instead of
     re-deriving it from a state value — and should now be scored **on the same-depth `top1|top2`
     cell, against K-averaged labels**, never on the pooled single-label metric.
   * **RE-READ THE TWELVE HEADS' 0.567–0.578 AGAINST THE CEILING.** The levels are not wrong; their
     INTERPRETATION was. Nothing needs re-running — the correction is a divisor.
   * ⚠️ **What is NOT worth another arm:** a thirteenth static value head, and any further read on
     this instrument that does not print the three pair-type columns, the depth split, and the
     label's own ceiling.

## 7. Hazards and findings about the instruments

Every one of these is a finding.

1. 🚨 **A PAIRWISE-ACCURACY BAR ON A SINGLE-ROLLOUT LABEL IS VACUOUS UNTIL THE LABEL'S OWN CEILING
   IS MEASURED — and here it is 0.61, not 1.0.** The arithmetic was registered before the data
   (`PREDICTION.md` §2): an ORACLE knowing both siblings' true win probabilities reads
   `p_a(1−p_b)/[p_a(1−p_b)+p_b(1−p_a)]`, which is 0.550 at a five-point gap. Measured here at
   0.6125 pooled with an 8-rollout estimator. **Every future read on this instrument must carry a
   contemporaneous rollout ceiling, the way a contention-scaled timeout carries its factor.**
2. 🚨 **A POOLED PAIRWISE METRIC HIDES A SECOND MIXTURE: DEPTH.** §4.5. 19.4 % of successors are at
   the fork's own turn (a KO forced a replacement), only 76.5 % of forks have all three branches at
   one depth, and the head reads 0.646 on mixed-depth pairs against 0.557 on matched ones. The
   campaign's `top1|top2` 0.545 is really **0.5355 at matched depth**. **Print the depth split
   beside the three pair-type columns.**
3. 🚨 **OMNISCIENCE IS EXACTLY FREE FOR A MATERIAL / STATUS / HAZARD / BOOST READOUT IN GEN 3.**
   §4.4 — alive counts and tempo-status counts match the referee's on 14,601 of 14,601 successors,
   HP to 0.12 %. An "omniscient control" of that shape cannot fail to be null; the budget belongs
   on IDENTITY and MOVESET.
4. ⚠️ **THE SPLIT-HALF CEILING'S INVERSION IS BIASED IN BOTH DIRECTIONS AND IS REPORTED AS A
   CROSS-CHECK, NOT A HEADLINE.** `A = q² + (1−q)²` inverts to `q = (1+√(2A−1))/2` only for a
   homogeneous `q`; the inverse is concave, so heterogeneous gaps make it an OVER-estimate (Jensen),
   while scoring an exact half-label tie as 0.5 pulls `A` toward 0.5 and makes it an UNDER-estimate.
   The same caveat applies to the parametric `model oracle @K = 1`, which assumes a normal gap
   around a base rate of 0.5 and over-reads the measured 8-rollout number by ~0.06. **The
   matched-noise half-label read (§5.2) and the K-mean-as-scorer read (§5.4) have neither problem —
   one label, two scorers, one pair set — and the branch call rests on those two alone.**
5. 🚨 **THE AVERAGED-LABEL POPULATION IS NOT THE SINGLE-LABEL POPULATION.** 26 % of pairs are
   non-tied under two single rollouts against 58 % under two K-means, so "averaging raised the
   accuracy" is not a statement until both numbers are read on the same forks. Every comparison in
   §5 is within one fork set, and §5.4 carries an explicit small-n warning because its
   per-column head numbers wander by up to 0.09 against the full set.
6. 🚨 **A MOMENT-RECOVERED VARIANCE NEEDS `ddof=1`.** `np.var(ddof=0)` estimates `p(1−p)(K−1)/K`,
   so `Var(K-mean)` is `s²_unbiased / K`; the biased form under-states the label noise by `(K−1)/K`
   and inflates every recovered true gap by the same factor. Caught before the read and fixed in
   `score_controls.py`; at K = 8 it would have moved the noise share from 46.7 % to 40.9 %.
7. **A hand evaluator is cheap enough that there was never a reason not to run one.** Nine scorers,
   one CPU pass of 10 minutes over six shards, no training — and the answer to a question six GPU
   arms had left open. It had not been run because the campaign kept varying the LEARNER; the
   control that mattered was varying the SCORER CLASS.
8. **`ForfeitBattleOrder` on the decision after the capture is a safe ~3× saving** on a
   capture-only pass: the line ends at ~turn T+1 instead of ~turn 55, the banked outcome is already
   on file, and the 138 to-terminal forks confirm the CRN line is unchanged. The right shape for any
   future "read the board at the successor" pass.
9. **A fork-OUTER / K-INNER loop is how a rollout-averaging pass should be written.** A wall-clock
   stop then yields FEWER FORKS AT FULL K rather than all forks at a ragged K, and a ragged K
   weights forks unequally in the mean without saying so. The pass is also resumable on
   `fork_line`, which cost four lines.
10. **Nothing under `src/` was changed by this read**, so no test or gate obligation attaches; every
    instrument it calls is invoked by path, unmodified, and the new code is in this directory. The
    `sim_bridge` binary was checked for the 2026-09-09 worktree-path contamination class
    (`strings … | grep gen3ai-wt` → 0).

## 8. Files

| | |
|---|---|
| [`PREDICTION.md`](PREDICTION.md) | the registration, landed on main at `a72b8b21` before the first battle |
| [`hand_capture.py`](hand_capture.py) · [`run_hand.sh`](run_hand.sh) | CONTROL 1 — the CRN re-run, the dual-`LiveView` capture, the element-wise obs assertion, the forfeit |
| [`reroll_labels.py`](reroll_labels.py) · [`run_reroll.sh`](run_reroll.sh) | CONTROL 2 — the K-rollout re-rolls, CRN within a re-roll index, fork-outer/K-inner and resumable |
| [`score_controls.py`](score_controls.py) · [`run_score.sh`](run_score.sh) · [`score.json`](score.json) | the read: thirteen scorers, four columns, both label regimes, all four ceilings, the depth split |
| [`posthoc_omni.py`](posthoc_omni.py) · [`posthoc_omni.json`](posthoc_omni.json) | POST-HOC: how much information omniscience adds, and the successor-depth census |
| [`posthoc_corr.py`](posthoc_corr.py) · [`posthoc_corr.json`](posthoc_corr.json) | POST-HOC: the head-vs-hand rank correlation and the across-state AUCs |
| [`make_table.py`](make_table.py) | renders `score.json` into this file's tables — no number above was retyped by hand |
| [`verify_indexing.json`](verify_indexing.json) | rule 25's indexing clause, executed by the 2026-09-18 verifier |
| [`hand_meta.json`](hand_meta.json) · [`reroll_meta.json`](reroll_meta.json) | the twelve shards' own metadata — counts, checks, walls |
| not committed | the captured boards and the 24,048 re-roll outcomes, under `/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_control/` |

## 9. The registered predictions, scored

`PREDICTION.md`, landed at `a72b8b21` before the first battle. **Eight of twelve held; three of the
four misses are findings.**

| # | registered | outcome |
|---|---|---|
| **P1** | the cell is **N × G** | **HALF.** N HELD; **G REFUTED** — the true sibling gap is large (`E\|gap\|` 0.177) and the matched-noise read DETECTS headroom, so the cell is **N × L** |
| **P2** | the head reproduces 0.5450 ± 0.002 and 0.5723 ± 0.002 | **HELD exactly** (0.5450 / 0.5723, four decimals) |
| **P3** | the best hand evaluator reads **0.50–0.57** on `top1\|top2`, at or below the head | **HELD.** 0.5368, below 0.5450 |
| **P4** | the best hand evaluator reads **0.55–0.65** on `top1\|rand`, markedly better than on `top1\|top2` | **HELD.** 0.5766 (`PBRS_1S`) vs 0.5364 |
| **P5** | omniscience is nearly free: mean \|Δ\| < 0.1 HP-units, accuracies within 0.01 | **HELD, and more strongly than registered** — alive and status counts EXACTLY equal on 14,601/14,601, HP mean \|Δ\| **0.0074**, accuracies within 0.007 |
| **P6** | K = 8 labels raise the head's `top1\|top2` into **0.55–0.62** without clearing 0.60 | **HELD.** 0.5610 [0.5313, 0.5913] |
| **P7** | the split-half ceiling is **0.55–0.70** at K = 4 | **HELD.** 0.6303 agreement on `top1\|top2` (0.6425 pooled) |
| **P8** | `E\|p_a − p_b\|` is **0.02–0.10** and `ACC_oracle(K=1)` is **0.52–0.58**, i.e. the head is near-oracle | **REFUTED, and it is the read's central finding.** `E\|gap\|` = **0.1771**, and an 8-rollout estimator reads **0.5811** on `top1\|top2` (0.6125 pooled) against the head's 0.5405 on the same pairs. The gap is far larger than registered and the head is NOT near-oracle |
| **P9** | the single-vs-averaged order disagreement on `top1\|top2` is **0.30–0.45** | **HELD.** 0.397 |
| **P10** | `obs_match` ≥ 99.5 %, to-terminal outcome agreement ≥ 98 % | **HELD at 100 % and 100 %** (14,601/14,601; 138/138) |
| **P11** | the `rand` branch caps more often than `top1`'s | **REFUTED on the re-rolls.** Cap rates `top1` 0.00162 / `top2` 0.00137 / **`rand` 0.00100** — the ORDER INVERTS under fresh post-divergence dice, against 37× the other way on the realized stream. At 32 caps in 24,048 rollouts nothing turns on it, but the 2026-09-18 asymmetry does not generalise off the realized stream |
| **P12** | mean K-label `top1` − `top2` is **+0.005 to +0.05** | **HALF — and the miss is the finding.** +0.0079 is inside the range, but its CI **[−0.0025, +0.0188] STRADDLES ZERO**: the policy's own argmax is not detectably better than its runner-up |

**BAR 1 FAILED on all nine (→ N) · BAR 2 FAILED (0.5610 < 0.60) · BAR 3 MET pooled and on both
`rand` columns, NOT MET on `top1|top2` (→ L) · BAR 4's branch-G clause FAILED (order agreement
0.6034, not ≥ 0.85) · BAR 5 MET at 100 % · BAR 6: the battery was NOT RUN, on instruction.**

## 10. Ready-to-append ledger paragraph

> ### 🎯 IS THE 0.545 `top1|top2` SIBLING-RANKING LEVEL THE HEADS, THE LABELS, OR THE GAME? — **N × L: no hand evaluator beats the head, but 🚨 THE SINGLE-ROLLOUT LABEL MIS-ORDERS THE PAIR 39.7 % OF THE TIME and an 8-rollout estimate of the same states reads 0.61 where the head reads 0.54** — the campaign's metric was measuring a coin flip, the true sibling gap is **E|gap| = 0.177**, the policy's own top-1 is worth **+0.0079 [−0.0025, +0.0188] NOT CLEAR OF ZERO** over its top-2, and **a FOUR-ROLLOUT leaf beats the 75M-step critic by +0.057 [+0.028, +0.083] DETECTED** (2026-09-19)
>
> Record `designs/research_state/measurements/leaf_ceiling_controls_2026-09-19/` (`PREDICTION.md`
> landed on main at **`a72b8b21`** before the first battle). Two controls on the EXACT banked 5,040
> held-out contested CRN forks of `offline_leaf_fit_2026-09-18/`, arm W
> (`ai_v13_02_flywheel_winprob` @ 75,005,952) frozen. **CONTROL 1: every banked branch re-run under
> the same dice and the board read at the first live decision off BOTH players' battle objects at
> once — 14,601 / 14,601 successor observations reproduced ELEMENT-WISE, 0 mismatches, 138/138
> to-terminal outcomes reproduced. CONTROL 2: 1,002 forks × K = 8 fresh post-divergence dice × 3
> branches = 24,048 rollouts, 0 errors, 32 capped (0.13 %).** CPU only, nothing under `models/`.
>
> **THE HAND-EVALUATOR AXIS IS N.** Nine non-learned scorers — the PRODUCTION PBRS potential
> composition (`agents.training.reward_potentials`, imported not re-implemented), that plus the
> incoming-KO belief potential, bare material+HP, a type-matchup variant, and their omniscient twins
> — read **0.5219–0.5416** on `top1|top2` against arm W's **0.5450**; **BAR 1 FAILS on all nine**,
> every interval contains 0.5450 and every paired Δ is negative and NOT DETECTED. 🚨 **But a
> hand-written potential sum MATCHES a 75M-step critic on the published pooled metric** —
> `PBRS_BELIEF_1S` **0.5714 [0.5581, 0.5847]** vs **0.5723**, Δ **−0.0009 [−0.0152, +0.0136] NOT
> DETECTED** — **and BEATS it on `top2|rand`**, **0.6007** vs **0.5752**, Δ **+0.0255 [+0.0043,
> +0.0461] DETECTED**. Mechanism measured: **Spearman(head, Φ-sum) = 0.768** over 14,570 successors
> and across-state **AUC 0.789 vs 0.763** — the head is a slightly better material counter and
> essentially nothing else; across states both are strong, within a fork both collapse to ~0.54.
>
> 🚨 **THE LABEL IS THE STORY.** The single-rollout label puts the two siblings in the WRONG ORDER
> **39.7 %** of the time on `top1|top2` (order agreement 0.6034; 0.6375 pooled), and **46.7 % of the
> variance in the observed sibling-label difference is label noise** (observed 0.0925 / noise 0.0432
> / true 0.0493). **Read against a K = 8 averaged label the head rises 0.5450 → 0.5610 [0.5313,
> 0.5913]** — BAR 2's literal >0.60 FAILS, but 0.60 was registered in advance as vacuous until the
> label's own ceiling is measured (`PREDICTION.md` §2: an ORACLE knowing both true win probabilities
> reads only `p_a(1−p_b)/[p_a(1−p_b)+p_b(1−p_a)]`, **0.550 at a five-point gap**). **Measured
> ceiling: the K = 8 rollout MEAN used as a SCORER on the banked SINGLE labels reads 0.6125 [0.5830,
> 0.6430] pooled and 0.5811 [0.5390, 0.6259] on `top1|top2`, where the head reads 0.5359 and 0.5405
> on the identical pairs.** **Every published level on this instrument — 0.517, 0.575, 0.587, the
> twelve heads at 0.567–0.578 — is compressed toward 0.5 by the label, and the compression had never
> been quantified.**
>
> 🚨 **A FOUR-ROLLOUT LEAF BEATS THE 75M-STEP CRITIC — the first thing in this campaign to do so.**
> Matched-noise read (one K = 4 label, two scorers, the same pairs; the scorer is an INDEPENDENT
> 4-rollout mean of the same successors): **0.6172 [0.5995, 0.6354] vs the head's 0.5607, Δ +0.0566
> [+0.0282, +0.0831] DETECTED** pooled, and DETECTED on both `rand` columns (+0.0717, +0.0678).
> ⚠️ **On `top1|top2` alone it is +0.0275 [−0.0176, +0.0705] NOT DETECTED** at n = 454 — four
> rollouts also run out of resolution on the leaf column, so the open question is the DOSE (the
> smallest K at which it detects), not a new objective. **Branch L: the headroom is real and
> reachable; branch G is REFUTED on its mechanism.**
>
> 🚨 **AND THE POLICY'S OWN PREFERENCE IS WORTH NOTHING MEASURABLE.** Mean K = 8 label: `top1`
> 0.4936, `top2` 0.4857, `rand` 0.4637 — **`top1` − `top2` = +0.0079 [−0.0025, +0.0188], NOT CLEAR
> OF ZERO**, while `top1` − `rand` = +0.0299 [+0.0181, +0.0417] is clear. Yet the two successors'
> true values differ by **E|gap| = 0.1771** (recovered sd 0.2220). **There is a great deal to
> re-rank between the policy's two best moves and the policy's ordering carries almost none of it** —
> which is the quantitative replacement for "the top-2 are near-tied": near-tied ON AVERAGE, far
> apart CASE BY CASE.
>
> **POST-HOC, TWO MORE MIXTURES.** 🚨 **DEPTH:** 19.4 % of successors sit at the FORK'S OWN TURN (a
> KO forced a replacement inside the fork turn) and only **76.5 %** of forks have all three branches
> at one depth, so ~a quarter of pairs compare states one ply apart; the head reads **0.6457
> [0.6141, 0.6806] on mixed-depth pairs against 0.5571 [0.5413, 0.5729] on matched ones**, and **the
> genuinely hard cell — the policy's two best moves at the SAME depth — is 0.5355 [0.5117, 0.5583]**,
> where all thirteen scorers sit in 0.514–0.546. 🚨 **OMNISCIENCE IS EXACTLY FREE IN GEN 3 for a
> material/status/hazard/boost readout** (registered in advance as P5): over 14,601 successors the
> one-sided and referee views agree on the opponent's ALIVE COUNT **14,601/14,601** and on its
> TEMPO-STATUS COUNT **14,601/14,601**, and on its HP sum to a mean **|Δ| 0.0074 of ~6.0 (0.12 %)** —
> and not because the opponent is revealed (**1.94 of 6** mons unrevealed at the average successor).
> **The hidden information in gen 3 is IDENTITY and MOVESET, not MATERIAL**, so a hidden-information
> floor on a gen-3 value function cannot be a floor about material, and an "omniscient control" of
> that shape cannot fail to be null.
>
> **EIGHT OF TWELVE PREDICTIONS HELD; three of the four misses are findings.** P8 registered
> `E|gap|` 0.02–0.10 with the head near-oracle and is **REFUTED** (0.177, and the head is 0.04–0.08
> short of an 8-rollout estimator) — the read's central result. P12's CI straddles zero. P11 —
> the `rand` branch caps MORE often — **inverts under fresh post-divergence dice** (`top1` 0.00162 /
> `top2` 0.00137 / `rand` 0.00100), so the 2026-09-18 37× asymmetry is a property of the REALIZED
> stream and does not generalise. Instrument notes: rule 25's indexing clause was EXECUTED (the
> 2026-09-18 `verify_indexing.py` by path reproduces **0.5723320158** against **0.5723320246**, gap
> **8.79e-9**); ⚠️ the split-half ceiling's `q = (1+√(2A−1))/2` inversion and the parametric
> normal-gap oracle are **cross-checks only** — the inversion is biased UP by Jensen over
> heterogeneous gaps and DOWN by tie-scoring, and the model over-reads the measured 8-rollout number
> by ~0.06 — so the branch call rests solely on the two model-free reads; 🚨 a moment-recovered
> label-noise variance needs **`ddof=1`** (`np.var(ddof=0)` under-states it by `(K−1)/K`, which would
> have moved the noise share from 46.7 % to 40.9 %); 🚨 the averaged-label population is NOT the
> single-label population (26 % non-tied vs 58 %), so every comparison is read within one fork set.
> **WHAT REMAINS:** (1) a **K-ROLLOUT LEAF DOSE CURVE on the same-depth `top1|top2` cell** — find the
> smallest K at which `ROLLOUT_K − head` detects; `reroll_labels.py` is the harness and the rust
> search driver the fast path; (2) the per-action **`q_winprob_head`** (built and OFF), scored on
> that cell against K-averaged labels rather than the pooled single-label metric; (3) **re-read the
> twelve heads' 0.567–0.578 against the ceiling** — the levels are right and their interpretation was
> not, and the correction is a divisor, not a re-run. ⚠️ **NOT worth another arm: a thirteenth static
> value head, or any read on this instrument that does not print the three pair-type columns, the
> depth split, and the label's own contemporaneous rollout ceiling.** Tag: **MEASURED (MAJOR) ·
> N × L — THE HAND EVALUATOR DOES NOT BEAT THE HEAD, BUT THE LABEL MIS-ORDERS THE PAIR 39.7 % OF THE
> TIME · the metric's ceiling is 0.61, not 1.0 · a 4-ROLLOUT LEAF BEATS THE 75M-STEP CRITIC (+0.057
> DETECTED) · the policy's top-1 is worth +0.008 [−0.003, +0.019] over its top-2 while their true
> values differ by 0.177 · omniscience is EXACTLY free in gen 3 for material**.
