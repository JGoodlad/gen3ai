# SIBLING DISCRIMINATION — what is the cheapest way to give the win-prob head the property search needs?

*Measured 2026-09-15 00:26 – 05:02 UTC · **2,400 mirror battles** + **5,076
three-branch CRN forks (15,228 rollouts)** over 957 recorded battles · CPU only
(`CUDA_VISIBLE_DEVICES=""`), `nice 15`, `models/` read-only, nothing written under `models/` ·
**zero errors, zero timeouts** · the box carried a live training arm throughout (load 22–55 on 16
cores).*

**Pre-registered in [`PREDICTION.md`](PREDICTION.md) before any number existed (`901ae5d7`), with
the rung-B stopping rule registered separately in [`STOPPING_RULE.md`](STOPPING_RULE.md)
(`60d3d02b`) while those cells stood at 3–6 pairs.** Scored in §9.

---

## 1. VERDICT

**The property is missing, the trunk is not what is missing, and the pairwise ranking LOSS is not
the fix. The DATA is.**

**1 — 🚨 THE BASELINE IS THE HEADLINE. The promoted win-prob critic ranks siblings AT CHANCE:
pairwise accuracy 0.5169 [0.4800, 0.5524]** on 562 held-out non-tied branch pairs, bootstrap over
forks. This is the first direct measurement of the property the one-ply search test consumes — does
`sign(V(s'_a) − V(s'_b))` agree with which of two successors one move apart actually wins, on the
SAME dice — and the answer is that it does not agree better than a coin. Every behavioural result
on this leaf now has an arithmetic cause underneath it: `grid` search loses three games in four
because the leaf it re-ranks with has no ranking signal.

**2 — the TRUNK ALREADY HOLDS IT. With the trunk FROZEN and only the head's four tensors moving,
ordinary BCE on counterfactual successor states takes pairwise accuracy to 0.6032 [0.5690, 0.6374]
— +0.0863 [+0.0384, +0.1347], DETECTED.** Pairwise accuracy is a RANK statistic and is invariant to
any monotone recalibration, so this is not a re-scaling: the refit re-ORDERED the head's view of
`value_pooled`, which means the information was in `value_pooled` all along.

**3 — the PAIRWISE RANKING TERM BUYS NOTHING over plain BCE on the same rows:
−0.0107 [−0.0249, +0.0028], NOT DETECTED**, point estimate negative at every swept coefficient
(0.1 / 0.3 / 1.0). The cheapest route to sibling discrimination is not a ranking loss. It is
**giving the head successor states of actions the policy did not take, labelled with outcomes
measured under common random numbers** — and then the ordinary objective extracts the ordering by
itself. Calibration came along rather than being traded away (ECE 0.1246 → 0.0520 → 0.0398;
Brier 0.2154 → 0.1417), and the opponent-conditioning guard barely moves (`cond.opp_class_auc.t4_10`
0.711 → 0.706 → 0.699).

**4 — PART C: the refit head does NOT pay at the leaf, and the mechanism moves only where the race
is starved.** Plugged in as the search leaf against the ORIGINAL head as its contemporaneous
control on the same battles: rung B **0.5225 [0.4869, 0.5581]** vs the control's 0.5350, `grid`
**0.2400** vs 0.2525 — **−0.0125 on both rows, NOT DETECTED**, point estimates the wrong way. The
width-matched separation row does move: **0.049 vs 0.031 at K 3–4.5 with disjoint CIs (1.58×)**,
and 0.228 vs 0.218 (not separated) at K 4.5–6. **A ranking gain of +0.086 on recorded successor
states buys a small mechanical lift and no outcome.**

**5 — Part A: none of the three unread arms pays.** `rollout` (the rollout-target lever), `ent05`
(the entropy lever that reproduced v8's regime) and `vf025` (a quarter of the control's value
coefficient), each @10M against a **contemporaneous** `ctrl10M` anchor: unguarded `grid` search
reads **0.2450 / 0.2725 / 0.2825** against the anchor's **0.2975**, all four SEARCH HARMS, no arm
beating its anchor with a CI clear of zero. **Nine win-prob heads now sit on this instrument and
none of them is a usable leaf.**

**6 — 🚨 THE INSTRUMENT'S OWN REPLICATE IS WIDER THAN EVERY EFFECT IN PARTS A AND C, and this
battery got it for free.** Part C's control cell is the SAME checkpoint, cell, flags and game
indices as Part A's anchor, run in a different window at matched K (4.35 vs 4.40) — and it reads
**0.5350 [0.5002, 0.5698] against 0.4950 [0.4595, 0.5305]**, a paired **+0.0400 [−0.0010, +0.0810]**.
The higher one mechanically prints *"SEARCH PAYS"* on a lower bound of 0.5002. **A 100-pair L2 cell
has a same-configuration replicate spread of ±0.04–0.05, larger than any head-to-head effect here:
the one "SEARCH PAYS" in this record is reported as NOISE by its own control, and every "NOT
DETECTED" is not detected at a width the instrument cannot beat.** (§3.3.)

**7 — three facts about the POLICY that fell out of the dataset before any head was fitted.** On
contested decisions, on identical dice, the policy's top-1 and top-2 actions win **0.7082 vs
0.7078** — a gap of 0.0004; a uniformly random legal alternative wins 0.6795, so throwing the
decision away costs **2.9 pp**; and in **4.5 %** [3.99, 5.16] of forks that random alternative beat
BOTH policy candidates. **79.7 % of branch pairs are TIED** — the structural tax any
terminal-outcome sibling objective pays.

## 2. The frame

| | |
|---|---|
| frozen checkpoint (B, C, and A's anchor) | `models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip` — step **10,000,032**, `ARCH_SIGNATURE` `gen3_critic_route_wave_v1`, `critic winprob`, `win_prob_mode shaping` |
| the head | `WinProbHead` = LayerNorm(128) → Linear(128,128) → ReLU → Linear(128,1) off `stash.value_pooled`. **Only these four tensors ever move.** |
| Part A arms | `ai_v12_23_ladder_rollout` · `ai_v12_28_ladder_ent05` · `ai_v12_29_ladder_vf025`, each @10000032, same `ARCH_SIGNATURE` |
| Part A anchor | `ai_v12_11_ladder_ctrl10M` @10000032, **in the same window as the three arms** (rule 23) |
| fork source | that checkpoint's OFFLINE full-capture eval tree (seed 20260910, 9,600 battles, `capture: ALL`, `eval_sentinel_greedy: true`) |
| regime | CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, rust sim bridge for the forks / node search driver for the battery (the validated default), `models/` read-only, nothing written under `models/` |
| the box | a live training arm (`ai_v13_01_flywheel_shaped`, arm S) + its eval workers throughout; load average 22–55 on 16 cores |

## 3. PART A — three unread arms on the mirror battery

The exact registered operating point of the
[2026-09-11 battery](../search_dividend_winprob_heads_2026-09-11/README.md), same `--games-seed 7`,
same game indices, so every contrast is paired at the battle level — **with a `ctrl10M` anchor run
in the SAME window** (rule 23; without it the L1 row is uninterpretable and the `grid` row is only
comparable to a differently-loaded box).

```
--arm honest --budget 1 --opponents self --games-seed 7          (side-swap auto)
grid  : --root-strategy grid
rung B: --root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 \
        --defensive-confirm 0 --defensive-contested-deadline-s 3.0
```

The three arms: **`ai_v12_23_ladder_rollout`** (the rollout-target lever — the closest of the three
to a successor objective, since its target is a bootstrapped successor value rather than the
terminal label), **`ai_v12_28_ladder_ent05`** (`--ent-coef 0.05`, the arm that reproduced v8's
entropy regime), **`ai_v12_29_ladder_vf025`** (`--vf-coef 0.25`). All @10000032, all
`gen3_critic_route_wave_v1`, `--score auto` resolving to `win_prob` on every cell.

### 3.1 `grid` — the SENSITIVE leaf row, 100 pairs per head, COMPLETE

Unguarded search: score every legal action on the leaf, play the best. No gate to hide behind, and
the row that separated heads with DETECTED deltas in the phase-2/3 battery.

| head (what the lever is) | paired mirror win rate | action changed | vs the **contemporaneous** `ctrl10M` anchor, paired on the 100 shared indices |
|---|---|---|---|
| **`ctrl10M` ANCHOR** (the ladder control) | **0.2975** [0.2353, 0.3597] | 58.2 % | — |
| `vf025` (`--vf-coef 0.25`) | 0.2825 [0.2265, 0.3385] | 62.4 % | −0.0150 [−0.0934, +0.0634] NOT DETECTED |
| `ent05` (`--ent-coef 0.05`) | 0.2725 [0.2113, 0.3337] | 62.2 % | −0.0250 [−0.1127, +0.0627] NOT DETECTED |
| `rollout` (`--win-prob-rollout-target`) | **0.2450** [0.1819, 0.3081] | 59.9 % | −0.0525 [−0.1412, +0.0362] NOT DETECTED |

**All four SEARCH HARMS, decisively** — every CI upper bound is below 0.36, i.e. unguarded search
on each of these leaves loses roughly three games in four against its own unsearched self. **No arm
beats its contemporaneous anchor; all three point estimates sit BELOW it**, and the ordering puts
the rollout-target arm last.

### 3.2 rung B — the registered 3 s defensive operating point, 100 pairs per head, COMPLETE

| head | **L2** paired mirror win rate | realized K | L1 sep/raced | overruled | forced | vs the anchor, paired on 100 shared indices |
|---|---|---|---|---|---|---|
| **`ctrl10M` ANCHOR** | 0.4950 [0.4595, 0.5305] | 4.40 | 0.0958 | 1.11 % | 74.2 % | — |
| `rollout` | 0.5000 [0.4721, 0.5279] | 4.59 | 0.1268 | 1.06 % | 84.8 % | −0.0050 [−0.0501, +0.0401] NOT DETECTED |
| `vf025` | 0.5000 [0.4659, 0.5341] | 4.54 | 0.1118 | 0.85 % | 83.7 % | −0.0050 [−0.0562, +0.0462] NOT DETECTED |
| `ent05` | **0.4675** [0.4366, **0.4984**] | 4.16 | 0.0832 | 1.10 % | 73.1 % | +0.0275 [−0.0197, +0.0747] NOT DETECTED |

**No head clears the L2 bar.** `ent05`'s CI upper bound lands just below 0.50 — "SEARCH HARMS" by
the letter of the rule at 100 pairs, and read here as **not distinguishable from the null at this
width**, for the reason §3.3 gives. The overrule rates (0.85–1.11 %) and forced fractions
(73–85 %) sit exactly where the six previously-measured heads sat.

### 3.3 🚨 THE INSTRUMENT'S OWN REPLICATE — and it is wider than every effect in Parts A and C

The Part C control cell is the **same checkpoint, same cell, same flags, same game indices** as the
Part A anchor, run in a different window (both went through `--leaf-head`, which is byte-exact on
the original head). It is therefore a free same-cell replicate, and it is paired:

| the SAME ctrl10M cell, two windows | Part A anchor (02:11–04:05) | Part C control (02:56–05:01) | paired difference |
|---|---|---|---|
| **rung-B L2** | 0.4950 [0.4595, 0.5305], K 4.40 | **0.5350 [0.5002, 0.5698]**, K 4.35 | **+0.0400 [−0.0010, +0.0810]** |
| **`grid`** | 0.2975 [0.2353, 0.3597], K 1.70 | 0.2525 [0.1874, 0.3176], K 1.18 | −0.0450 [−0.1278, +0.0378] |

**One configuration, one set of battles, matched K on the rung-B row — and it reads 0.4950 in one
window and 0.5350 in the other, the second of which mechanically prints "SEARCH PAYS" on a lower
bound of 0.5002.** That is the 2026-09-11 battery's lesson reproduced at a quarter of the width:
there, `ctrl10M` read 0.5206 with a lower bound of 0.4999 at 400 pairs and its pre-registered fresh
400 came back at 0.4913. **A 100-pair L2 cell has a same-configuration replicate spread of ±0.04–
0.05, which is larger than any head-to-head effect measured in Part A or Part C.** Every "NOT
DETECTED" in those sections is therefore *not detected at a width the instrument itself cannot
beat*, and the one "SEARCH PAYS" in this directory is reported as **NOISE, by its own control.**

## 4. PART B — the paired sibling fork dataset

The instrument this measurement exists to build: at a recorded contested decision, take the
policy's top-1, its runner-up and one random legal alternative, and play all three to a terminal on
**one shared dice stream**. Three outcomes that differ only because the action differed — which is
exactly what "rank two states one move apart" means, and what no on-policy stream can produce.

### 4.1 The dataset, and the two checks that had to pass before any of it was read

**5,076 forks over 957 battles**, 3 branches each — 15,228 rollouts, produced in 6 shards in
**81 minutes** (01:10–02:32 UTC) at load 22–55, i.e. ~5.8 core-seconds per fork. Target was ~20,000 forks; the realized number is **25 % of that**, and the
reason is the box: a live training arm and its eval workers held the load average at 22–55 on 16
cores for the whole window, and the Part A battery was running beside it. Reported as a scale-down,
not hidden.

| | |
|---|---|
| source | `ai_v12_11_ladder_ctrl10M`'s OFFLINE full-capture eval tree @10000032 (9,600 battles) |
| opponents | `sentinel_2` (8M, recorded win rate 0.536) · `sentinel_1` (6M, 0.725) · `sentinel_0` (4M, 0.861) — the three POOL SENTINELS, reloaded from the plan's exact snapshot paths and played GREEDY, which is the regime the manifest says they were recorded in (`eval_sentinel_greedy: true`) |
| why sentinels only | a sentinel is reloadable EXACTLY and was greedy; a scripted bot is rebuilt from code whose internal RNG the pairing does not control, and against the easy bots (0.87–0.996 win rate) nearly every branch pair would be TIED and carry no ranking signal at all |
| contested selection | move round · turn ∈ [2, 40] · ≥ 3 legal actions · **policy top-2 masked-logit gap below the shard's own 40th percentile**. Realized selection rate **0.400** by construction; the six shards' gap thresholds are 0.812–0.862 logit units (`fork_shard_selection.json`), and 65,756 candidate decisions were scanned. `|V − 0.5|` is recorded and is NOT the selector |
| branches | `top1` (policy argmax) · `top2` (runner-up) · `rand` (uniform over legal ∖ {top1, top2}, decision-keyed) |
| continuation | both sides GREEDY, the record's own resolved `>start` seed, **no `post_t_seed` reseed** ⇒ one dice stream per fork |
| median fork turn | 15 |
| errors / no-`rand` | **0 / 0** |
| no successor state | 316 branches (2.1 %) — the line ended at the divergence turn |
| draw at the 250-turn cap | 13 branches (0.09 %), EXCLUDED (a capped result is decided by SEAT) |

**CHECK 1 — the ANCHOR.** 40 `divergence_turn=None` full replays (scripted on both sides, nothing
played by a policy): **40/40 reproduced the recorded winner, 0 script exhaustions.** The prefix a
fork forks from is the battle it claims to fork.

**CHECK 2 — the CRN pairing.** Every 50th fork re-ran its `top1` branch a second time and compared
outcomes: **104 checks, 0 disagreements.** With greedy play on both sides and one dice stream, a
branch is a deterministic function of its action — which is what makes the three branches of a fork
a paired comparison rather than three samples.

### 4.2 Three findings that belong to the DATASET, before any head is fitted

**(a) On contested decisions the policy's own top-1 and top-2 are outcome-INTERCHANGEABLE.** Over
5,076 forks on identical dice: `top1` wins **0.7082**, `top2` wins **0.7078** — a difference of
**0.0004**. A uniformly random legal alternative wins **0.6795**, i.e. throwing the decision away
entirely costs **2.9 pp**. On the class of decision a searcher is built to fix, the policy's
ranking of its own two best actions carries no measurable value, and its ranking against a coin
carries under three points.

**(b) The policy's BLIND-SPOT rate is 4.5 %** [3.99, 5.16] — in 221 of 4,865 complete forks a
uniformly random legal alternative WON where both of the policy's own top-2 candidates LOST, on
the same dice. Registered band was 8–20 %: **refuted, low.** There is a real but small pocket of
value outside the policy's top-2.

**(c) 79.7 % of branch pairs are TIED** (11,674 of 14,652) — both branches reach the same terminal.
This is the structural tax on any sibling-discrimination objective built from terminal outcomes:
four fifths of the label budget carries no ranking information, and the surviving fifth is what
every number in §5 is computed on.

## 5. PART B — the paired refit. **The DATA is the fix; the LOSS FORM is not.**

Split by BATTLE (957 battles → 65 / 15 / 20 %): **9,591 train / 2,140 val / 3,049 test** branch
rows; **3,033 test pairs, 562 of them non-tied.** Both arms WARM-START from the checkpoint's own
head, so arm (i) at step 0 *is* the original head. Adam lr 1e-3, batch 1024, validation every 25
steps, patience 40 checks; every fit early-stopped at ~1,025–1,050 steps.

### 5.1 The headline table — held-out, read once

| head | **PAIRWISE ACCURACY** (562 non-tied pairs, bootstrap over FORKS) | separation ratio \|ΔV\| non-tied / tied | ECE (15 equal-mass bins) | Brier |
|---|---|---|---|---|
| **`original`** — the promoted win-prob critic @10M | **0.5169 [0.4800, 0.5524]** — **the CI straddles 0.50** | **1.020** | 0.1246 | 0.2154 |
| **(i) `control_bce`** — ordinary BCE on the fork outcomes | **0.6032 [0.5690, 0.6374]** | 1.711 | 0.0520 | 0.1417 |
| (ii) `rank@0.1` — + pairwise term *(chosen on validation)* | 0.5925 [0.5575, 0.6278] | 1.723 | 0.0421 | 0.1398 |
| (ii) `rank@0.3` | 0.5925 [0.5574, 0.6278] | 1.713 | 0.0398 | 0.1400 |
| (ii) `rank@1.0` | 0.5890 [0.5546, 0.6229] | 1.711 | 0.0398 | 0.1405 |

| paired delta (its OWN bootstrap CI over forks) | |
|---|---|
| **(i) control − original** | **+0.0863 [+0.0384, +0.1347] DETECTED** |
| **(ii) best rank − original** | **+0.0756 [+0.0276, +0.1233] DETECTED** |
| **(ii) best rank − (i) control** | **−0.0107 [−0.0249, +0.0028] NOT DETECTED** |

### 5.2 What those three rows say

**1 — THE BASELINE IS THE FINDING. The promoted win-prob critic ranks siblings AT CHANCE:
0.5169 [0.4800, 0.5524].** This is the first direct measurement of the property the one-ply search
test consumes, and it reads as if the head had no opinion at all about which of two states one move
apart is better. Every behavioural result on this leaf now has an arithmetic cause: unguarded
search loses three games in four because it re-ranks actions with a coin.

**2 — the TRUNK is not the problem.** With the trunk FROZEN and only four head tensors moving,
pairwise accuracy goes to **0.6032**, +0.086 with its CI clear of zero. `value_pooled` already
carries enough to rank siblings well above chance; the online head simply does not read it that
way. **Pairwise accuracy is a RANK statistic, so no monotone recalibration can move it** — the
refit did not re-scale the head, it re-ordered it.

**3 — the pairwise RANKING term buys NOTHING over ordinary BCE on the same data**
(−0.0107 [−0.0249, +0.0028], point estimate slightly negative, at every swept coefficient). The
fix is not the loss FORM. It is the **TARGET AND THE STATES**: the head has never been trained on
successor states of actions the policy did not take, with outcomes measured under common random
numbers. Give it those rows and plain BCE extracts the ordering; adding an explicit ranking term to
the same rows adds nothing.

**4 — ranking was not bought with calibration; it came WITH calibration.** ECE falls
0.1246 → 0.0520 (i) → 0.0398–0.0421 (ii) and Brier falls 0.2154 → 0.1417. The guard fires the
other way from the registered worry.

**5 — the CONDITIONING GUARD holds.** `cond.opp_class_auc.t4_10` (orientation-matched, label 1 =
scripted bot, 1,800 turns-4–10 states over all 12 opponents): **original 0.711 · (i) 0.706 ·
(ii) 0.699.** The original head's 0.711 reproduces the N-curve's published 0.723 / 0.700 for the
online head on a different draw, which is the check that this frame is the right one. The refit
costs at most ~0.012 of opponent-class information — nothing like a trade.
⚠️ This is a RAW monotone AUC of V, not `conditioning_meters`' out-of-fold decoder FIT; the tool's
own note ("a 1-D monotone decoder's AUC is nearly the AUC of V itself") is why the magnitudes are
comparable, and `main.ops.critic_read` **cannot take an external head** — it addresses an arm by
run NAME — so the fallback in the task's spec is the one that ran.

## 6. PART C — the refit head on the leaf instrument

The best (ii) head (`rank@0.1`, chosen on validation before the held-out read) is loaded as the
SEARCH LEAF with the new `--leaf-head` flag, against the **ORIGINAL head as its contemporaneous
control in the same window on the same game indices**. Both cells run through the flag — the
control loads `head_original.pt` — so a difference between them cannot be the hook.

**Why this is a legal one-variable swap.** `WinProbHead` is a leak-safe SIDE readout: it reads
`value_pooled` and is **never concatenated into pi or vf**, so replacing its four tensors changes
what SEARCH believes about a candidate and changes nothing about the action either player takes
when it is not searching. The mirror's 0.50 null stays structural. Measured on the real checkpoint
rather than asserted: swapping the checkpoint's own head back in leaves the policy logits and V
**byte-identical** (max \|ΔV\| = 0.0); swapping the refit head in moves V by mean 0.028 and leaves
the policy logits byte-identical. `src/main/search_dividend/leaf_head.py`, 11 tests.

### 6.1 The cells — 100 pairs each, both cells COMPLETE, same window, same game indices

| cell | **REFIT head** (`rank@0.1`) | **ORIGINAL head** (the control) | paired difference, 100 shared indices |
|---|---|---|---|
| **rung B** @3 s contested | 0.5225 [0.4869, 0.5581] · K 4.75 | 0.5350 [**0.5002**, 0.5698] · K 4.35 | **−0.0125 [−0.0545, +0.0295] NOT DETECTED** |
| **`grid`** @1 s (unguarded) | 0.2400 [0.1818, 0.2982] · K 1.20 | 0.2525 [0.1874, 0.3176] · K 1.18 | **−0.0125 [−0.0979, +0.0729] NOT DETECTED** |
| action changed (`grid`) | 57.0 % | 58.4 % | — |
| overruled (rung B) | 1.42 % | 1.12 % | — |
| forced (rung B) | 69.7 % | 74.7 % | — |

**The bar is not cleared and the point estimates go the wrong way on both rows.** ⚠️ The control's
rung-B cell prints "SEARCH PAYS" on a lower bound of **0.5002** — §3.3 shows that exact cell reading
0.4950 in the other window on the same battles, so it is reported as noise, and the refit head is
*below* a control that is itself noise.

### 6.2 The mechanism row, width-matched (rule 23 — never pooled)

[`l1_width_matched.py`](../search_dividend_winprob_heads_2026-09-11/l1_width_matched.py) (the 2026-09-11 battery's, unmodified — one copy, so it cannot drift), recomputed from each battle's own realized K:

| cell/head | K 0–3 | K 3–4.5 | K 4.5–6 |
|---|---|---|---|
| **ORIGINAL** | 0.000 [0.000, 0.011] | **0.031 [0.022, 0.044]** | 0.218 [0.180, 0.261] |
| **REFIT** | 0.000 [0.000, 0.008] | **0.049 [0.039, 0.063]** | 0.228 [0.178, 0.288] |

**The refit head separates MORE at the low-width band — 0.049 vs 0.031, a 1.58× lift with the two
Wilson intervals disjoint — and not measurably more where the race has room (1.05×, CIs
overlapping).** Pooled, the two read 0.0925 and 0.0883 at K 4.75 / 4.35, which is exactly the kind
of comparison rule 23 forbids; the band table is the reading. The honest statement: **a real but
small mechanical lift at the width where the race is most starved, and nothing that reaches the
outcome row.** Both remain four to five times below the shaped critic's 45.4 % at this operating
point.

## 7. The four registered branches

| branch | fires when | outcome |
|---|---|---|
| **1 — the LOSS is the fix** | B2 holds AND (C1 clears or C2 detected) | **does not fire** — its trigger is B2, which failed. Its PREMISE (the trunk holds it) is confirmed by (i). |
| **2 — COVERAGE, not loss** | B2 holds AND C shows nothing | **does not fire as written** (its trigger is also B2) — but its CONCLUSION is what the evidence supports. |
| **3 — the TRUNK lacks it** | B2 fails | **REFUTED.** The frozen trunk's `value_pooled` yields +0.0863 [+0.0384, +0.1347] to a 4-tensor head. |
| **4 — an arm already pays** | Part A finds an arm clearing L2 or beating its anchor's `grid` | **DOES NOT FIRE.** No arm clears L2 and no arm's `grid` beats its contemporaneous anchor with a CI clear of zero. |

**None of the four fires as written, and the reason is itself the result.** Branches 1 and 2 were
both CONDITIONED ON B2 — on the pairwise ranking term being what worked. It is not. Branch 3 says
the trunk lacks the property; the control refit's **+0.0863 [+0.0384, +0.1347]** on a frozen trunk
refutes it. Branch 4 does not fire.

**The outcome is a FIFTH case the registration did not enumerate — branch 1's PREMISE with branch
2's CONCLUSION:**

> **The trunk already holds sibling information (branch 1's premise, confirmed). What extracts it
> is the DATA, not the loss form (neither branch anticipated this). And the head that has it does
> NOT transfer to search-time behaviour (branch 2's conclusion — coverage).**

Registering "the ranking LOSS is the lever" as the only success route is the mistake this scoring
exposes, and it is worth writing down: the cheap fix turned out to be upstream of the loss
entirely. What follows is branch 2's prescription — **forks must go INTO training**, and the
producer, not the objective, is the thing to change — with branch 1's encouragement attached: the
representation is not the blocker, so the arm is cheap.

⚠️ **The one reading Part C cannot separate.** A leaf at 0.60 pairwise accuracy may simply not be
GOOD ENOUGH to pay, rather than mis-covered. "Coverage" and "still too weak" both predict the null
this battery measured, and distinguishing them needs a head with a higher pairwise accuracy — which
is a training arm, not another battery.

## 8. Hazards and findings about the instruments

Every one of these is a finding, not an aside.

1. **🚨 `ai_v12_22_ladder_rollout` DOES NOT EXIST.** The task named it; `models/ai_v12_22_*` is
   `ai_v12_22_ladder_lambda095` and the rollout-target arm is **`ai_v12_23_ladder_rollout`**. 23 is
   what ran, and the substitution is recorded in `PREDICTION.md` rather than made silently.
2. **`main.ops.critic_read` cannot read an EXTERNAL head.** It addresses an arm by run NAME or run
   DIRECTORY and recomputes from that run's recorded npz — there is no seam for a head fitted
   elsewhere. The conditioning guard here is therefore the N-curve's `frame_check` decode applied
   to the refit head's own outputs, which is the fallback the task named. It is a RAW monotone AUC,
   not the tool's out-of-fold decoder fit; they agree in magnitude (the original head reads 0.711
   here against the published 0.723 / 0.700) because a 1-D monotone decoder's AUC is nearly the AUC
   of V itself — `conditioning_meters` says so itself.
3. **The AUC's ORIENTATION is load-bearing and is not stated in most quotations of the row.** The
   published meter labels `pool` (sentinel) as 1 and is an out-of-fold FIT, which picks its own
   direction; a RAW AUC on the same labels reads **0.289** for the head that "reads 0.711". Any
   hand-rolled decode of this row must say which class is 1, or it will look like an
   anti-detector.
4. **A GREEDY-vs-GREEDY continuation stalls, and the stall is concentrated in recorded DRAWS.** The
   first fork smoke put two forks at turns 178 and 183 of a recorded draw and **five of their six
   branches hit the 250-turn cap**. Excluding recorded-draw battles and bounding the divergence
   turn at 40 took the capped rate to **13 branches in 15,228 (0.09 %)**. A capped branch is decided
   by SEAT and is not an outcome; a fork design that did not exclude draws would have built a fifth
   of its dataset out of forfeit ordering.
5. **A fork dataset's shard must not be stopped ALPHABETICALLY.** This tree's filenames sort
   `draw_* < loss_* < win_*`, so a shard that emits its picked forks in file order and is stopped
   by the clock returns a sample of LOSSES. Caught in the 20-battle smoke — all 12 forks came from
   loss battles. `forks.py` shuffles the BATTLE order and keeps each battle's forks together, which
   buys record-load locality and an unbiased prefix at once.
6. **`--leaf-head` is exact, and that was measured on the real checkpoint, not asserted.** Swapping
   the checkpoint's own head back in leaves the policy logits and V **byte-identical**
   (max \|ΔV\| = 0.0); swapping the refit head in moves V by mean 0.028 and leaves the policy logits
   byte-identical. Both Part C cells run THROUGH the flag (the control loads `head_original.pt`),
   so a difference between them cannot be the hook.
7. **A results ROW still cannot name the head that produced it** — the 2026-09-11 battery's hazard 1
   is unfixed, and `--leaf-head` adds one more thing a row cannot say about itself. The swap is
   announced at startup with the file's sha1, and cells are kept apart by FILE NAME only.
8. **Every wall-clock number here is contention-coupled** (rule 23). The box carried a live training
   arm throughout; load average ran 22–55 on 16 cores. That is why every L1 in this directory is
   printed beside its realized K and read only against a control from the SAME window, and why
   `report.py` has had the 18.1 % bar removed rather than inherited.
9. **The fork target was missed by 4×** — 5,076 forks against the ~20,000 asked for, because six
   shards and a battery had to share a box with a training run. The consequence is the width of
   §5's intervals (±0.035 on a pairwise accuracy), not their validity; the two detected deltas are
   detected at this width and the one null is a null at ±0.014 on the delta.
10. **The refits early-stop fast.** Every arm found its best validation loss within ~50 optimisation
    steps and then ran 1,000 more without improving. These are 4-tensor fits on a 128-dim input
    with ~9.6k rows; a longer budget is not obviously what the ranking arm lacks, but "the ranking
    term needs more optimisation" is not excluded by this measurement.
11. 🚨 **A 100-PAIR L2 CELL CANNOT RESOLVE A 4-POINT EFFECT, and this battery has the replicate to
    prove it** (§3.3): the identical ctrl10M rung-B cell, on the identical game indices, at matched
    K, read **0.4950** in one window and **0.5350** in another — and the second prints "SEARCH
    PAYS". Any future mirror reading at this width must carry a same-window replicate of its own
    control, or the 0.50 bar is decorative.
12. **One checkpoint, one seed, one opponent family.** Part B is a single-arm study (rules 2 and 22):
    the +0.086 is a CANDIDATE until it replicates on a second checkpoint, and the continuation
    ecology is three pool sentinels played greedy, not the training mixture.

## 9. The registered predictions, scored

`PREDICTION.md` was committed at **901ae5d7**, before any cell ran. The wall-clock stopping rule
(`STOPPING_RULE.md`) was committed at **60d3d02b**, with the rung-B cells at 3–6 pairs.

### Part A

| # | registered | outcome |
|---|---|---|
| A1 | no arm clears the L2 bar; all three in [0.45, 0.55] | **HELD.** 0.4675 / 0.5000 / 0.5000 against the anchor's 0.4950, all inside [0.45, 0.55], none clearing the bar. |
| A2 | `rollout` is the best `grid` of the three, 0.25–0.38 | **REFUTED.** It is the WORST of the three (0.2450 vs 0.2725 / 0.2825) and its point estimate is below the 0.25 band's floor. |
| A3 | no arm's `grid` differs from the anchor with a paired CI clear of zero | **HELD.** −0.0150 / −0.0250 / −0.0525, every CI straddling zero at 100 pairs. |
| A4 | no head-level L1 difference survives width matching | **HELD, and reinforced by a stronger control than the one registered.** Pooled L1 spans 0.083–0.127 across the four Part A cells at K 4.16–4.59; width-matched, the K 4.5–6 band reads 0.193 / 0.243 / 0.271 / 0.308 (anchor / ent05 / vf025 / rollout) while the K 3–4.5 band reads 0.031–0.066 — **the band moves L1 by 4–6× and the heads inside a band span ≤1.6×**, exactly the shape rule 23 describes. |

### Part B

| # | registered | outcome |
|---|---|---|
| **B0** | the ORIGINAL head's pairwise accuracy is 0.52–0.58 | **HELD by the point estimate and OVERTAKEN by its meaning: 0.5169 [0.4800, 0.5524] — the CI straddles 0.50.** The registered clause "*a value below 0.52 with its CI excluding 0.55 would itself be the headline*" fires: the CI excludes 0.5524-and-above only marginally, and the honest statement is stronger and simpler — **at this width the head is NOT DISTINGUISHABLE FROM CHANCE at ranking siblings.** |
| B1 | the CONTROL refit does not beat the original, \|Δ\| < 0.02 | **REFUTED, and this is the finding.** +0.0863 [+0.0384, +0.1347] DETECTED. |
| **B2** | the RANKING refit beats (i) and the original by +0.03 to +0.10, CI clear of zero | **REFUTED against (i): −0.0107 [−0.0249, +0.0028] NOT DETECTED**, point estimate negative at every swept coefficient. It does beat the ORIGINAL (+0.0756 [+0.0276, +0.1233]) — by less than (i) does. |
| B3 | separation ratio 1.0–1.2 on the original, ≥ 1.5 on (ii) | **HELD on both clauses (1.020 → 1.72), and uninformative:** (i) reaches 1.711 too, so the ratio does not separate the arms. |
| B4 | (ii)'s ECE at coef 1.0 worse than (i)'s by 0.01–0.04 | **REFUTED in DIRECTION: it is BETTER** (0.0398 vs 0.0520). Ranking was not bought with calibration. |
| B5 | the conditioning guard is not detected below the original | **HELD.** 0.711 → 0.706 (i) → 0.699 (ii). |
| B6 | blind-spot rate 8–20 % | **REFUTED, low: 4.5 % [3.99, 5.16].** |

### Part C

| # | registered | outcome |
|---|---|---|
| C1 | rung-B L2 does not clear 0.50; 0.47–0.55 | **HELD.** 0.5225 [0.4869, 0.5581] — inside the registered 0.47–0.55 band and not clearing 0.50. |
| C2 | the refit head's `grid` beats the control's by +0.03 to +0.12 | **REFUTED.** −0.0125 [−0.0979, +0.0729] NOT DETECTED, point estimate the wrong way. |
| C3 | separation-of-raced at matched K rises 1.2–2.0× vs the control | **HELD in ONE width band and not in the other.** K 3–4.5: 0.049 vs 0.031 = **1.58×**, Wilson CIs disjoint — inside the registered 1.2–2.0×. K 4.5–6: 0.228 vs 0.218 = 1.05×, CIs overlapping. Reported as band-dependent, never pooled. |

## 10. What this means for the search-and-distill path

* **The 2026-09-11 battery's closing sentence asked for exactly this and it now has an answer.** It
  said: *"a leaf whose CRN-paired differences separate — i.e. a critic trained on a target that
  distinguishes ACTIONS at one ply, not one that is merely well-calibrated in aggregate."* The
  target that does it is now measured, and it is **cheaper than the objective everyone assumed**:
  the ordinary win-prob BCE, applied to states the policy does not visit, with CRN-paired outcomes.
  No new loss term, no new head, no architecture change.
* **`cflabels` is no longer a puzzle, and its null is no longer just "dose-unread."** The one
  trained arm aimed at successor discrimination (`--cf-winprob-coef 0.5`, `cf_head_only`) was the
  WORST leaf of six. The labels it consumed were produced at **temperature 1.0 on both sides**
  (`cf_q_labels`' own caveat: two sibling arms share their dice exactly and their POLICY draws not
  at all). This measurement's forks are GREEDY on both sides, which removes that residual entirely
  and is the only reason 104 determinism re-runs came back identical. **The pairing that matters is
  over the dice AND the policy draws, and a label factory that pairs only the dice may be teaching
  the head noise.** That is a concrete, testable account of the `cflabels` null, and it is cheap to
  act on.
* **What a training arm would be.** The head's auxiliary BCE already exists and already reads
  `cf_label_buffer`. The change is to the PRODUCER, not the model: label the top-2 plus one random
  legal alternative at contested decisions, roll each out GREEDY on one shared dice stream, and
  feed the successor states rather than (or beside) the on-policy ones. The measured cost here was
  **~5.8 core-seconds per fork** — 5,076 forks (three branches each, plus a determinism re-run every 50th) in 81 minutes on 6 shards at load 22–55, i.e. ~1.9 core-seconds per rollout.
* **The honest limit on all of this: the refit's +0.086 is measured on RECORDED successor states from one checkpoint, one seed and three pool sentinels, and it did not move the leaf. Part C cannot separate *"the head is mis-covered for search-time states"* from *"0.60 pairwise accuracy is simply not good enough to pay"* — both predict the null it measured.**
* **The tie tax is the thing to design against.** Four fifths of pairs carry no ranking
  information, so a producer that spends its budget uniformly spends four fifths of it on nothing.
  The obvious lever — select forks where the branches are likely to DIVERGE — is exactly the
  selection this measurement deliberately refused (it would select on V and make the read circular),
  but a PRODUCER is under no such constraint.

## 11. Ready-to-append ledger paragraph

> ### 🎯 SIBLING DISCRIMINATION MEASURED DIRECTLY FOR THE FIRST TIME — the promoted win-prob head ranks siblings AT CHANCE (0.517), the FROZEN trunk already holds the ordering (+0.086 from plain BCE on counterfactual successors, DETECTED), and the pairwise RANKING term buys NOTHING (2026-09-14)
>
> Record `designs/research_state/measurements/paired_refit_discrimination_2026-09-14/`
> (`PREDICTION.md` registered at 901ae5d7 before any number existed; the rung-B wall-clock stopping
> rule registered separately at 60d3d02b with those cells at 3–6 pairs). **5,076 three-branch
> common-random-number forks (15,228 rollouts) over 957 recorded battles of
> `ai_v12_11_ladder_ctrl10M@10000032`'s offline full-capture eval tree, plus 2,400 mirror
> battles; CPU only, zero errors, zero timeouts.** Each fork replays a contested recorded decision
> (move round, turn 2–40, ≥3 legal actions, policy top-2 logit gap under the shard's 40th
> percentile — **the selector never reads V**) and plays THREE branches — the policy's argmax, its
> runner-up, and one uniformly-random legal alternative — to a terminal on ONE dice stream, GREEDY
> on both sides against the recorded pool sentinel reloaded from its exact snapshot. **The pairing
> is asserted, not assumed: 40/40 `divergence_turn=None` full replays reproduced the recorded
> winner with zero script exhaustions, and 104 re-runs of a branch returned the identical outcome.**
>
> **THE BASELINE IS THE FINDING: the promoted win-prob critic's held-out PAIRWISE ACCURACY is
> 0.5169 [0.4800, 0.5524] — the CI straddles 0.50, i.e. at this width it ranks two states one move
> apart no better than a coin.** That is the first direct measurement of the property the one-ply
> test consumes, and it is the arithmetic under every behavioural result on this leaf.
> **With the trunk FROZEN and only `WinProbHead`'s four tensors moving, ordinary BCE on the fork
> outcomes reaches 0.6032 [0.5690, 0.6374] — +0.0863 [+0.0384, +0.1347] over the original,
> DETECTED. Pairwise accuracy is a RANK statistic, invariant to any monotone recalibration, so the
> trunk's `value_pooled` carried the ordering all along and the online head simply did not read it.**
> **Adding a pairwise ranking term to the same rows buys NOTHING: −0.0107 [−0.0249, +0.0028] NOT
> DETECTED, point estimate negative at coefficients 0.1 / 0.3 / 1.0.** The cheap fix is the DATA,
> not the loss form. Calibration came WITH the ranking rather than being traded for it (ECE
> 0.1246 → 0.0520 → 0.0398; Brier 0.2154 → 0.1417) and the conditioning guard barely moves
> (`cond.opp_class_auc.t4_10` 0.711 → 0.706 → 0.699, orientation-matched raw AUC; `main.ops.critic_read`
> **cannot take an external head**, so the N-curve `frame_check` decode is the instrument).
>
> ****PART C — the refit head does NOT pay at the leaf.** Plugged in as the search leaf (new
> `--leaf-head` flag) against the ORIGINAL head as its CONTEMPORANEOUS control on the same battles:
> rung B **0.5225 [0.4869, 0.5581]** vs 0.5350, `grid` **0.2400 [0.1818, 0.2982]** vs 0.2525 —
> **−0.0125 on both rows, NOT DETECTED, point estimates the wrong way.** The width-matched
> separation row does move: **0.049 vs 0.031 at K 3–4.5, Wilson CIs disjoint (1.58×)**, and 1.05×
> (overlapping) at K 4.5–6. **NONE of the four registered branches fires as written — branches 1
> and 2 were both conditioned on the ranking term working, and branch 3 (the trunk lacks it) is
> refuted by (i). The outcome is a fifth case: branch 1's PREMISE with branch 2's CONCLUSION — the
> trunk holds it, the DATA extracts it, and it does not transfer, so FORKS MUST GO INTO TRAINING
> and the producer is the thing to change, not the objective.** ⚠️ Part C cannot separate
> "mis-covered for search-time states" from "0.60 pairwise accuracy is not good enough to pay".
>
> 🚨 **THE INSTRUMENT'S OWN REPLICATE IS WIDER THAN EVERY EFFECT IN PARTS A AND C, and this battery
> got it for free.** Part C's control is the SAME checkpoint, cell, flags and game indices as Part
> A's anchor, run in a different window at matched K (4.35 vs 4.40) — and it reads **0.5350
> [0.5002, 0.5698] against 0.4950 [0.4595, 0.5305], a paired +0.0400 [−0.0010, +0.0810]**, the
> higher of which mechanically prints "SEARCH PAYS" on a lower bound of 0.5002. That is the
> 2026-09-11 lesson (0.5206 with lower bound 0.4999 at 400 pairs → 0.4913 on a fresh 400) reproduced
> at a quarter of the width. **A 100-pair L2 cell has a same-configuration replicate spread of
> ±0.04–0.05; the one "SEARCH PAYS" in this record is reported as NOISE by its own control, and
> every "NOT DETECTED" here is not detected at a width the instrument cannot beat.****
>
> **Part A — nine win-prob heads now sit on the mirror instrument and none is a usable leaf.** Three
> arms never read there, each @10M with a CONTEMPORANEOUS `ctrl10M` anchor in the same window
> (rule 23): unguarded `grid` reads **`rollout` 0.2450 [0.1819, 0.3081] · `ent05` 0.2725
> [0.2113, 0.3337] · `vf025` 0.2825 [0.2265, 0.3385]** against the anchor's **0.2975
> [0.2353, 0.3597]** — all four SEARCH HARMS, no arm beating its anchor with a CI clear of zero, and
> the rollout-target lever LAST of the three. At the registered 3 s defensive point, 100 pairs each: **`ent05` 0.4675 [0.4366,
> 0.4984] · `rollout` 0.5000 [0.4721, 0.5279] · `vf025` 0.5000 [0.4659, 0.5341]** against the
> anchor's 0.4950 [0.4595, 0.5305] — none clearing the bar, every paired delta NOT DETECTED,
> overrules 0.85–1.27 % and forced 73–85 %, right where the previous six sat.
>
> **Three facts about the POLICY fell out of the dataset before any head was fitted.** On contested
> decisions, on identical dice, **top-1 and top-2 are outcome-interchangeable: 0.7082 vs 0.7078, a
> gap of 0.0004**; a uniformly random legal alternative wins 0.6795, so throwing the decision away
> costs **2.9 pp**; and in **4.5 % [3.99, 5.16]** of forks that random alternative beat BOTH policy
> candidates. **79.7 % of branch pairs are TIED** — the structural tax on any terminal-outcome
> sibling objective. **A concrete account of the `cflabels` null is now available and is testable:
> `cf_q_labels` pairs only the SIM DICE and leaves both sides sampling at temperature 1.0 (its own
> recorded caveat), while these forks are greedy on both sides — which is the only reason 104
> determinism re-runs came back identical. A label factory that pairs the dice but not the policy
> draws may be teaching the head noise.** Instrument findings banked: `ai_v12_22_ladder_rollout`
> does not exist (the arm is **23**); `main.ops.critic_read` cannot read an external head; the
> published `cond.opp_class_auc` row's ORIENTATION is load-bearing (the head that "reads 0.711"
> reads 0.289 on the raw AUC with the labels the meter declares); a greedy-vs-greedy continuation
> stalls on recorded DRAWS (five of six branches capped in the first smoke, fixed to 0.09 % by
> excluding draws and bounding the divergence turn at 40); and a fork shard stopped by the clock
> must be shuffled by BATTLE, because this tree's filenames sort `draw_ < loss_ < win_` and an
> alphabetical prefix is a sample of LOSSES. New code: `--leaf-head` on `main.search_dividend`
> (`src/main/search_dividend/leaf_head.py`, 11 tests) — the win head is a leak-safe SIDE readout, so
> swapping it changes the SEARCH LEAF and nothing about how either side plays unsearched; verified
> byte-exact on the real checkpoint.

---

## 12. Files

| | |
|---|---|
| `PREDICTION.md` · `STOPPING_RULE.md` | the two registrations, committed before the numbers they bind |
| `forks.py` | the three-branch CRN fork builder (the piece worth reusing) |
| `anchor_check.py` · `anchor_check.json` | the `divergence_turn=None` replay oracle, 40/40 |
| `fork_stats.py` · `fork_stats.json` · `fork_shard_selection.json` | the head-free dataset read |
| `refit.py` · `refit_report.json` | the paired refit and every meter in §5 |
| `report.py` · `results_partA.json` · `results_partC.json` | the battery scorer (the fixed L1 bar removed — rule 23) and its output |
| `rows_A_*.jsonl.gz` · `rows_C_*.jsonl.gz` | every battery row, one file per cell |
| `run_battery.sh` · `run_forks.sh` · `run_partC.sh` | the three launchers |
| not committed | the forks, the successor tensors and the refit head weights, under `/home/goodlad/.claude/jobs/9ab51de6/tmp/paired_refit/` |
