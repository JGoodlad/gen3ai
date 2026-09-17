# PREDICTION — the FORK ARM's registered read (`ai_v13_03_fork`)

**Committed before a single primary number exists.** The fork datasets this read is computed on
had not been built when this file was written; the two offline eval trees it needs were still
generating. Everything below — the frame, the two deviations, the metric definitions, the bars,
the numeric predictions and the four branches — is fixed here and scored, unchanged, in
`README.md` §9.

---

## 0. What is being read, and against what

`ai_v13_03_fork` is the arm registered in ledger *2026-09-14 · SIBLING DISCRIMINATION MEASURED
DIRECTLY* and built per the amendment *THE FORK ARM IS BUILT* (`a5a649a5`, `gen3_fork_v1`, config
v120). It puts **contested-state forks into the PPO buffer**: at a contested decision the battle is
forked into three branches (policy top-1, its runner-up, one uniformly-random legal alternative),
all three are played to a terminal under **common random numbers on the dice AND both sides' policy
draws** (`--fork-crn dice_and_draws`), the fork step is masked out of the clipped policy term for
every branch, the shared prefix is counted once, and the branches' transitions go into the same
buffer under plain BCE with **no ranking term**. It completed at **10,027,008** steps
(ledger `e2172333`): dose DELIVERED at ~1,060 forks/rollout with the row budget never binding,
`fork/sim_steps_share` 1.48, and a **train-frame** `fork/pairwise_acc` of **0.5817** (final
median-10) — a rule-20 training-frame descriptor, never the read.

**The registered endpoints, in the order the registration set them:**

1. held-out **pairwise accuracy on fresh contested forks**, against `ctrl10M`'s level, bar = the
   arm's CI clears **0.60** (the offline refit's level);
2. the **mirror battery at ≥ 400 pairs** with a contemporaneous control (rule 25);
3. **`cond.opp_class_auc.t4_10`** as the guard.

## 1. 🚨 A DEFECT IN THE REGISTERED COMPARATOR, FOUND BEFORE THIS READ BEGAN

The registration's bar is *"vs `ctrl10M`'s 0.517"*. **That number is an indexing artifact and this
read must not be scored against it.**

`refit.py` (the 2026-09-14 record) builds its branch table with
`b_idx.append(br["succ"])` — an index into the *successor* array — and then evaluates the ORIGINAL
head as `V["original"] = V0`, where `V0` is laid out by **successor index**, while
`read_head` indexes it by **branch-row index** (`V[pairs["a"]]`). Every refit head is scored
correctly (`head_V(h, tab["pooled"])` is per branch row); only the original is not. The two index
spaces coincide only while no branch that HAS a successor is dropped, and 24 were (13 stall-capped,
the rest non-binary outcomes). Recomputed on the surviving dataset
(`/home/goodlad/.claude/jobs/9ab51de6/tmp/paired_refit/forks`, 5,040 forks / 14,803 successors /
14,780 branch rows): `b_idx == arange` **fails from row 1,880 onward, maximum offset 23**.

Re-scoring the **identical** held-out split (seed 20260914, the same 562 non-tied test pairs) with
the original head correctly indexed:

| | as published | correctly indexed |
|---|---|---|
| pairwise accuracy | 0.5169 [0.4800, 0.5524] | **0.5872 [0.5526, 0.6225]** |
| separation ratio | 1.020 | 1.793 |
| ECE / Brier | 0.1246 / 0.2154 | **0.0287 / 0.1412** |

**The promoted win-prob critic does NOT rank siblings at chance.** Its CI clears 0.50, it sits
inside the refit's 0.6032 [0.5690, 0.6374] interval, and its calibration was already where the
refit's was. This is recorded here, before the fork arm's own numbers exist, because it changes
what this read can conclude: **the control for the fork arm is `ctrl10M`'s head scored on the SAME
fresh forks by the SAME code path**, which is immune to the defect by construction. The 0.517 is
quoted below only to mark the registered bar; it is not a comparator.

The full recheck, including what it does to the refit's `+0.0863` and to the calibration claim, is
`recheck_paired_refit.py` / `recheck_paired_refit.json` and is reported in README §8. **Nothing in
`paired_refit_discrimination_2026-09-14/` is edited** — a record is a record.

## 2. The frame, and TWO DEVIATIONS, both declared here

| | |
|---|---|
| **arm** | `models/ai_v13_03_fork/snapshots/snapshot_000010000032.zip` — step **10,000,032** |
| **control** | `models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip` — step **10,000,032** |
| both | `ARCH_SIGNATURE` `gen3_critic_route_wave_v1`, `critic winprob`, `win_prob_mode shaping`, `eval_sentinel_greedy: true` |
| state sources | each arm's OWN offline full-capture 800-game eval tree (`main.ops.eval_trace_gen … --games 800 --sentinels 3 --concurrency 1`), seeds **20260910** and **20260911**, 12 opponents × 800 = 9,600 battles per draw |
| fork continuation opponents | that tree's own three POOL SENTINELS, reloaded from the plan's exact snapshot paths, played GREEDY (the regime the manifest records) |
| regime | CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice`, rust sim bridge, `models/` READ-ONLY, nothing written under `models/`, forks/trees under `/home/goodlad/.claude/jobs/9ab51de6/tmp/fork_read/` |

**DEVIATION 1 — the arm is read at `snapshot_000010000032.zip`, not `final_model.zip`
(10,027,008).** Three reasons, and they are not cosmetic. (i) The control is at 10,000,032, so this
is an exactly step-matched comparison rather than one 26,976 steps (0.27 % of the run) apart.
(ii) The sentinel count is set by how many pool snapshots lie BELOW the read step: at 10,000,032 the
fork arm has exactly **three** (4,000,032 / 6,000,000 / 8,000,016), which are the **same three
steps** as `ctrl10M`'s, so both trees carry an identical 3-sentinel + 9-bot frame; read at
10,027,008 the arm would carry **four** sentinels and `critic_read`'s guard would compare a 13-cell
frame to a 12-cell one. (iii) It is the arm's final PROMOTED node, the one the bank line's SmallRL
anchor read. The 26,976-step gap is recorded as the deviation it is.

**DEVIATION 2 — the `ctrl10M` fork dataset is built from the seed-**20260911** draw, not the
seed-20260910 one the 2026-09-14 refit used.** The registration asks for **FRESH** contested forks;
the 20260910 tree is the tree that record's baseline was measured on. Both arms' fork datasets come
from the **same** seed-20260911 draw, so the two are matched. The 20260910 draw is generated for
both arms anyway and is the guard's second draw.

## 3. The primary metric — stated precisely enough to be falsifiable

For each fork (a recorded contested decision) three branches are played; each branch that reaches a
terminal contributes a **successor state** (the obs at the first LIVE decision after the fork) and
an **outcome bit**. For a pair of branches (a, b) of the SAME fork with different outcomes,
the head is correct iff `sign(V(s'_a) − V(s'_b)) == sign(outcome_a − outcome_b)`.

* **PAIRWISE ACCURACY** = that agreement rate over non-tied pairs. It is a **rank** statistic —
  invariant to any monotone recalibration — which is why it cannot be bought with calibration.
* **The CI is a bootstrap over FORKS**, never over pairs: three pairs inside one fork share a
  prefix and are not three observations. 2,000 resamples, `refit.boot_ci` verbatim.
* `V` is the head's own `sigmoid(win_prob_logits)` on the successor state — the `--v-column
  win_probs` tensor. Under `--critic winprob` it IS the critic (`max|values − win_probs| = 0`).
* A branch is EXCLUDED if it has no successor (the line ended at the divergence turn), hit the
  250-turn stall cap (a capped result is decided by SEAT, not the position), or has a non-binary
  outcome.
* **The whole dataset is read** — there is no train/val/test split, because nothing is fitted.
* 🚨 **The alignment defect of §1 is asserted, not assumed:** the scorer refuses unless every head's
  `V` vector has one entry per BRANCH ROW, and a regression check asserts
  `V_row == V_succ[b_idx]`.

**Both directions are read, and the two effects are separated:**

| | scored by the FORK ARM's head | scored by `ctrl10M`'s head |
|---|---|---|
| forks from the **fork arm's** own states | ON-OWN (the primary) | CROSS |
| forks from **`ctrl10M`'s** own states | CROSS | ON-OWN (the control's own level) |

The **head effect** is the paired Δ down a column-pair within one state set (same forks, same
pairs, two heads — paired at the fork). The **state-distribution effect** is the difference between
the two rows for one head, and is NOT paired (different forks); it is reported as two intervals,
never as a delta with a CI.

## 4. The registered bars

| # | bar | as registered |
|---|---|---|
| **BAR 1 (PRIMARY)** | the fork arm's pairwise-accuracy CI on ITS OWN states **clears 0.60** | verbatim from the registration |
| **BAR 1′ (the comparator that works)** | the **paired Δ** (fork arm head − `ctrl10M` head) on the fork arm's own forks, CI clear of zero ⇒ DETECTED | replaces "vs 0.517", which §1 voids |
| **BAR 2 (battery, rule 25)** | ≥ 400 paired mirror games per cell; L2 Wilson lower bound **> 0.50** = DIVIDEND, against a CONTEMPORANEOUS `ctrl10M` cell in the same window on the same game indices; plus separation-of-raced at **matched realized K** | verbatim |
| **GUARD** | `cond.opp_class_auc.t4_10`, fork arm vs `ctrl10M`, `main.ops.critic_read` on both 800-game draws against `hp800_floor_v6.json` / `hp800b_floor_v6.json` | verbatim |
| **calibration side-condition** | ECE and Brier on the same states: a ranking gain bought with a calibration loss is reported as such | verbatim |

## 5. The numeric predictions

Scored in README §9. Each is a band, and a band that contains everything is not a prediction.

**Primary**

| # | prediction |
|---|---|
| P1 | the fork arm ON ITS OWN states reads pairwise accuracy in **[0.59, 0.66]**, point estimate ≈ 0.62, and its CI **DOES clear 0.60** — stated as the coin-flip it is: I give this ~50 %, because the train-frame meter reads 0.5817 and a recorded-state offline read has never come in above its own train-frame meter |
| P2 | the paired Δ (fork arm − `ctrl10M`) on the fork arm's own forks is **+0.01 to +0.06** and **IS DETECTED** (~3,000 non-tied pairs gives a Δ CI of roughly ±0.02) |
| P3 | the same Δ on `ctrl10M`'s forks (CROSS for the arm) is **positive and smaller**, +0.00 to +0.04, and I do NOT predict it is detected |
| P4 | the **state-distribution** effect is small: for either head, |own-states − cross-states| < **0.04** |
| P5 | `ctrl10M`'s ON-OWN level lands in **[0.55, 0.62]**, i.e. reproduces the CORRECTED 0.5872 of §1 on a fresh draw and NOT the published 0.517 |

**Dataset descriptors**

| # | prediction |
|---|---|
| P6 | tie rate **0.76–0.83** per branch pair (the 2026-09-14 dataset's 0.797) on BOTH state sets |
| P7 | blind-spot rate (a random legal alternative beats both policy candidates) **3–7 %** on both, reproducing 4.5 % [3.99, 5.16] |
| P8 | top-1 and top-2 remain outcome-interchangeable on both arms: |win rate(top1) − win rate(top2)| < **0.01** |

**Calibration**

| # | prediction |
|---|---|
| P9 | the fork arm's ECE on its own successor states is **0.02–0.08** and its Brier **0.12–0.18**; the ranking result is NOT bought with calibration, i.e. the arm's ECE is not worse than `ctrl10M`'s by more than **0.03** |

**Guard**

| # | prediction |
|---|---|
| P10 | `cond.opp_class_auc.t4_10`, fork arm vs `ctrl10M`, MATCHED frame: **|Δ| < 0.0245** (the imported floor), NOT DETECTED, and the sign is the SAME on both draws |

**Battery (part 2, run only after part 1 is landed)**

| # | prediction |
|---|---|
| P11 | rung-B L2 for the fork arm at 400 pairs lands in **[0.48, 0.53]** and its Wilson lower bound **does NOT** clear 0.50 — NO DIVIDEND |
| P12 | `grid` (unguarded) for the fork arm lands in **[0.22, 0.42]** — SEARCH HARMS, as it has for all nine heads on this instrument |
| P13 | separation-of-raced at matched K (band 3–4.5) is **1.0–1.8×** the contemporaneous control's, and I do NOT predict the Wilson intervals are disjoint |
| P14 | the paired Δ (arm − contemporaneous control) on rung-B L2 is NOT DETECTED at 400 pairs, |Δ| < 0.04 |

**Descriptors that are NOT endpoints** (rule 20 / rule 25): the train-frame `fork/pairwise_acc`
0.5817; H_end 0.5052 against `ctrl10M`'s 0.7473 at the same `--ent-coef 0.02` (a single draw each,
noted, not read); the SmallRL anchor 0.530 [0.433, 0.625] at 10M with **no 10M comparator cell in
the campaign** (so no deficit is claimable); strength on the 4-node ladder, which is **UNREADABLE**
at this depth (the 2026-09-14 external-anchor campaign measured the least-connected 10M node moving
−42.4 Elo when two external edges were added).

## 6. The four branches, fixed here

| branch | fires when | conclusion |
|---|---|---|
| **(a)** | BAR 1 cleared **AND** the battery pays (BAR 2) | **forks-into-training produce a LEAF** — the producer change transfers all the way to search-time behaviour |
| **(b)** | BAR 1 cleared, battery does NOT pay | **ranking on recorded states, no transfer at this level** — the same wall Part C hit, now with a head trained rather than refitted |
| **(c)** | BAR 1 not cleared but the arm is **above the control by a CI clear of zero** (BAR 1′) | **PARTIAL** — forks moved the head, by less than the offline refit's level |
| **(d)** | the arm is **not above the control** | **forks did not teach the head**, and the train-frame 0.5817 is the puzzle: a recorded-state read and an on-stream meter are then measuring different things |

Branches (a)–(c) are not mutually exclusive with §1's correction: if the corrected control is
itself near 0.59, then BAR 1 can be cleared with a Δ of ~0.01 and the honest reading is **(c)'s
substance under (a)/(b)'s label** — that case is reported as such and not smoothed over.

## 7. What would make me abandon this read

* a determinism check that disagrees (the CRN pairing is asserted, not assumed: every 50th fork
  re-runs one branch and the shard REFUSES on a mismatch);
* an anchor check that fails to reproduce a recorded winner on a `divergence_turn=None` full
  replay;
* a capped-branch rate above ~1 % (the 2026-09-14 build reached 0.09 % by excluding recorded draws
  and bounding the divergence turn at 40);
* either eval tree coming in with a shortfall, a regime mismatch, or a sentinel count ≠ 3.

Any of these is reported as a refusal, not worked around.
