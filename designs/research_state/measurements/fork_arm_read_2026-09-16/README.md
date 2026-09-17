# THE FORK ARM'S REGISTERED READ — did putting contested-state forks into the PPO buffer teach the win-prob head to rank siblings?

*Measured 2026-09-17 00:17 – 05:10 UTC · **10,080 three-branch common-random-number forks
(30,240 rollouts)** over two policies' own offline eval trees, **two fresh 9,600-battle
full-capture eval trees** generated for the arm, the conditioning guard on both draws, and a
**3,200-battle mirror battery at 400 paired games per cell** ·
CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, rust sim bridge, `models/` READ-ONLY, nothing
written under `models/` · **zero errors, zero determinism failures** · the GPU was free and
untouched; ports 8000/8001 untouched.*

**Pre-registered in [`PREDICTION.md`](PREDICTION.md) at `55cf08a7`, before a single fork existed.**
Scored in §9.

---

## 1. VERDICT — branch (d), and the reason is not the one the branch anticipated

**The fork arm's head does NOT rank siblings better than the control's. And the registered target it
was supposed to beat never existed.**

**1 — 🚨 THE REGISTERED COMPARATOR IS AN INDEXING ARTIFACT.** The registration's bar is *"vs
`ctrl10M`'s 0.517"* — the 2026-09-14 finding that the promoted win-prob head *"ranks siblings AT
CHANCE."* That number comes from scoring the original head with a **successor-indexed** `V` read at
**branch-row** positions; every refit head in the same script is scored correctly. On the banked
dataset the two index spaces diverge from branch row 1,880 and **12,900 of 14,780 rows are
mis-scored**. Re-scored on the identical held-out split, the identical 562 non-tied pairs:
**0.5872 [0.5526, 0.6225], not 0.5169** — the CI clears 0.50, it sits inside the refit's own
interval, and its ECE is **0.0287, not 0.1246**. *The promoted win-prob critic was never at chance.*

**2 — and with the baseline corrected, the 2026-09-14 record's two DETECTED rows both collapse.**
The refit's headline gain **+0.0863 [+0.0384, +0.1347] DETECTED** becomes
**+0.0160 [+0.0000, +0.0313] NOT DETECTED**; *"calibration came WITH the ranking"* inverts — the
refit made ECE **worse** (0.0287 → 0.0520). What survives untouched is the one row computed between
two correctly-scored heads: the pairwise ranking term buys nothing, −0.0107 [−0.0249, +0.0028].

**3 — THE PRIMARY READ. On 5,040 fresh contested forks from its OWN eval-trace states, the fork
arm's head reads pairwise accuracy 0.5754 [0.5608, 0.5897]. The registered bar was that the CI
CLEAR 0.60. It does not — the whole interval lies below it.**

**4 — and it is not above the control on any state set.** Paired, on the same forks and the same
pairs: **−0.0003 [−0.0148, +0.0136]** on the arm's own states and **+0.0067 [−0.0096, +0.0231]** on
the control's — **both NOT DETECTED**, one point estimate on each side of zero, at a Δ width of
about ±0.015. **BAR 1 fails and BAR 1′ fails: branch (d).**

**5 — the puzzle branch (d) reserved for itself DISSOLVES, and a different one replaces it.** The
branch named the train-frame `fork/pairwise_acc` of **0.5817** as the thing to explain if the
offline read came in low. It does not need explaining: the offline read of **0.5754** on
recorded states reproduces the on-stream meter to within 0.006. **The number that never reproduced
is the 0.517 the arm was aimed at.** Both heads sit at ~0.575 and always did; the arm's live meter
looked like progress only against a comparator that was arithmetic.

**6 — the treatment DID change something, and it is the STATE DISTRIBUTION, not the head.** The
fork arm's own contested states carry markedly more ranking signal per fork: tie rate **0.745 vs
0.806**, **3,745 non-tied pairs vs 2,849 (+31 %)** from the same 5,040 forks, and a blind-spot rate
of **5.53 % [4.93, 6.20] vs 4.55 % [4.01, 5.16]** with the Wilson intervals disjoint. Its contested
selector also bites at a far wider absolute gap (40th-percentile top-2 logit gap **1.45–1.63**
against the control's **0.81–0.89**), which is the same sharpening its entropy descriptor shows.
**Training on forks moved the policy's own decision population; it did not move the head's ability
to order two states one move apart.**

**7 — THE BATTERY DOES NOT PAY EITHER.** At 400 paired mirror games with a contemporaneous
control in the same window (rule 25): rung-B **L2 0.4844 [0.4570, 0.5117] against the control's
0.5019 [0.4781, 0.5257]**, paired **−0.0175 [−0.0531, +0.0181] NOT DETECTED**, neither lower bound
clearing 0.50; `grid` **0.2925 vs 0.2731**, both SEARCH HARMS. **BAR 2 is not met.** The mechanism
row does move, and it splits by width: the arm separates **0.80×** the control's rate where the
race is most starved (K 3–4.5, intervals disjoint) and **1.13–1.32×** from K 4.5 upward (all
disjoint) — **more separation at width, no outcome**, which is the shape Part C found for the
offline refit head.

**8 — the GUARD holds.** `cond.opp_class_auc.t4_10`, matched frame, draw 1 (seed 20260910):
**arm 0.7224 vs control 0.7097, Δ +0.0127 [−0.0021, +0.0277], WITHIN FLOOR** — the arm reads the
opponent's class slightly *better*, inside the imported floor. Draw 2 in §6.

## 2. The frame

| | |
|---|---|
| **arm** | `models/ai_v13_03_fork/snapshots/snapshot_000010000032.zip` — step **10,000,032**, pin `a5a649a5`, `gen3_fork_v1`, config v120, `--critic winprob`, `--ent-coef 0.02`, `--fork-crn dice_and_draws` |
| **control** | `models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip` — step **10,000,032** |
| both | `ARCH_SIGNATURE` `gen3_critic_route_wave_v1`, `win_prob_mode shaping`, `eval_sentinel_greedy: true` |
| **fork state sources** | each policy's OWN offline full-capture eval tree, draw seed **20260911**, `--games 800 --sentinels 3 --concurrency 1`, 12 opponents × 800 = **9,600 battles**, `capture: ALL` |
| **frames are matched by construction** | both trees carry the SAME 12 opponents and the SAME three sentinel STEPS — 4,000,032 / 6,000,000 / 8,000,016 — because each run's snapshot cadence is identical. Neither tree fell short: 9,600 traces each, `exit=0`, 76.6 min and 77.8 min |
| fork continuation opponents | those three pool sentinels, reloaded from the plan's exact snapshot paths, played GREEDY (the regime the manifest records) |
| regime | CPU only, `nice 15`, rust sim bridge (`POKESIM_SIM_BRIDGE_BIN`), forks/trees under `/home/goodlad/.claude/jobs/9ab51de6/tmp/fork_read/`, **nothing under `models/`** |
| the box | idle at the start (load 0.4); the two tree generations, then the two fork builds, carried it to load 8–20 on 16 cores. **No training run, no launcher, no server on 8000/8001 was started, touched or killed.** |

**The two declared deviations** (`PREDICTION.md` §2), both held to: the arm is read at its final
**promoted snapshot 10,000,032** rather than `final_model.zip` (10,027,008) — step-matched to the
control and 3-sentinel matched, where the final model would have carried four; and the `ctrl10M`
fork set is built from the **fresh seed-20260911 draw**, not the seed-20260910 tree the 2026-09-14
refit was measured on.

## 3. The datasets — and the two checks that had to pass before any head was scored

The builder, the contested selector, the CRN discipline and the pairwise metric are the
2026-09-14 record's, **reused by import, never copied**
([`run_forks.sh`](run_forks.sh) invokes `paired_refit_discrimination_2026-09-14/forks.py`;
[`score_forks.py`](score_forks.py) imports `refit.read_head` / `refit.paired_delta` /
`refit.boot_ci`), so they cannot drift from the instrument the corrected 0.5872 is measured on.

| | **arm's states** (`ai_v13_03_fork`) | **control's states** (`ctrl10M`) |
|---|---|---|
| forks / rollouts | 5,040 / 15,120 | 5,040 / 15,120 |
| shards · wall | 6 · 42–47 min | 6 · 77–84 min |
| errors · forks with no random branch | **0 · 0** | **0 · 0** |
| **determinism re-runs · disagreements** | **102 · 0** | **102 · 0** |
| branches with no successor | 280 (1.85 %) | 283 (1.87 %) |
| branches stall-capped at 250 turns (EXCLUDED) | **14 (0.09 %)** | **21 (0.14 %)** |
| branch rows scored | 14,809 | 14,806 |
| branch pairs · non-tied | 14,694 · **3,745** | 14,673 · **2,849** |
| **tie rate** | **0.7451** | **0.8058** |
| contested gap threshold (the shard's own 40th pct) | **1.4526 – 1.6310** | **0.8115 – 0.8889** |
| median fork turn | 14 | 15 |
| opponent mix (sentinel 0 / 1 / 2) | 1,689 / 1,595 / 1,756 | 1,681 / 1,584 / 1,775 |
| base rate (branch win rate) | 0.6096 | 0.7022 |

**CHECK 1 — the CRN pairing is asserted, not assumed.** Every 50th fork re-ran one branch and
compared outcomes: **204 checks across the two datasets, 0 disagreements.** With both sides greedy
on one dice stream a branch is a deterministic function of its action, which is what makes three
branches of a fork a paired comparison rather than three samples.

**CHECK 2 — the alignment assertion, which exists because of §1.** `score_forks.py` recomputes the
branch→successor index independently of `build_table` and REFUSES the read on any disagreement.
It also reports the condition under which the 2026-09-14 defect bites: on these datasets
**14,751 and 14,614 rows sit at a position different from their successor index (max offset 31)** —
i.e. had the same code path been used, essentially the whole read would have been mis-scored.

### 3.1 Three facts about the two POLICIES, before any head is scored

| on identical dice, contested decisions | **arm** | **control** |
|---|---|---|
| `top1` wins | 0.6178 | 0.7154 |
| `top2` wins | 0.6238 | 0.7107 |
| **\|top1 − top2\|** | **0.0060** | **0.0047** |
| `rand` (a uniformly random legal alternative) wins | 0.5892 | 0.6822 |
| cost of throwing the decision away | **2.86 pp** | **3.32 pp** |
| **blind-spot rate** (`rand` beats BOTH policy candidates) | **5.53 % [4.93, 6.20]** | **4.55 % [4.01, 5.16]** |

**Top-1 and top-2 remain outcome-interchangeable on both policies** — the 2026-09-14 dataset's
0.0004 gap reproduces as 0.006 and 0.005, both inside the registered 0.01. The random branch still
costs about 3 pp on both, and the fork arm's blind-spot rate is **higher** than the control's with
the two Wilson intervals disjoint: training on forks did not close the pocket of value outside the
policy's own top-2; if anything it widened it.

🚨 **The two win-rate columns are NOT comparable as strength.** Each policy plays its OWN three
pool sentinels, which are its own earlier snapshots. A lower absolute win rate against one's own
sentinels means the sentinels are relatively stronger, not that the policy is weaker.

## 4. THE PRIMARY READ — held-out pairwise accuracy, both heads on both state sets

`sign(V(s'_a) − V(s'_b))` against the shared-dice outcome on non-tied pairs; `V` is the head's own
`sigmoid(win_prob_logits)` (the `--v-column win_probs` tensor, which under `--critic winprob` IS the
critic). **Bootstrap over FORKS**, 2,000 resamples — three pairs inside one fork share a prefix and
are not three observations. The whole dataset is read; nothing is fitted, so nothing is held back.

| forks from → | scored by **FORK ARM's head** | scored by **`ctrl10M`'s head** | paired Δ (arm − ctrl), same pairs |
|---|---|---|---|
| **the arm's own states** (3,745 non-tied pairs) | **0.5754 [0.5608, 0.5897]** ← *the PRIMARY* | 0.5757 [0.5614, 0.5893] | **−0.0003 [−0.0148, +0.0136] NOT DETECTED** |
| **the control's own states** (2,849 non-tied pairs) | 0.5746 [0.5590, 0.5920] | **0.5679 [0.5515, 0.5843]** | **+0.0067 [−0.0096, +0.0231] NOT DETECTED** |

**Against the registered bars**

| bar | outcome |
|---|---|
| **BAR 1 — the arm's CI clears 0.60 on its own states** | **FAILED, and not narrowly**: the entire interval [0.5608, 0.5897] lies below 0.60 |
| **BAR 1′ — the paired Δ vs the control is clear of zero** | **FAILED on both state sets**, at a Δ width of ±0.014 / ±0.016 |
| the registered "vs `ctrl10M`'s 0.517" | **VOID** — §1. The control's own level on these fresh forks is **0.5679 [0.5515, 0.5843]**, reproducing the corrected 0.5872, not the published 0.517 |

### 4.1 The two effects, separated

* **HEAD effect ≈ 0.** Paired on identical forks: −0.0003 and +0.0067, neither clear of zero, the
  two point estimates on opposite sides of it.
* **STATE-DISTRIBUTION effect ≈ 0 for a head's ranking.** The arm's head reads 0.5754 on its own
  states and 0.5746 on the control's (0.0008 apart); the control's head reads 0.5679 on its own and
  0.5757 on the arm's (0.0078 apart). Both are far inside the registered 0.04. **These two rows are
  NOT paired — different forks — and are reported as two intervals, never as a delta with a CI.**
* The state distributions themselves differ a great deal (§3): the arm's forks yield 31 % more
  non-tied pairs. **More decidable pairs, the same accuracy on them.**

### 4.2 Calibration — the ranking was not bought, because there is no ranking gain to buy

| | mean V | base rate | separation ratio (\|ΔV\| non-tied / tied) | ECE (15 equal-mass bins) | Brier |
|---|---|---|---|---|---|
| arm head on **arm** states | 0.6689 | 0.6096 | 1.423 | 0.0597 | 0.1848 |
| ctrl head on **arm** states | 0.6651 | 0.6096 | 1.496 | 0.0565 | 0.1894 |
| arm head on **ctrl** states | 0.7050 | 0.7022 | 1.597 | **0.0188** | 0.1460 |
| ctrl head on **ctrl** states | 0.7289 | 0.7022 | 1.700 | 0.0301 | 0.1473 |

The side-condition is satisfied vacuously: the arm's ECE is 0.0032 worse on one state set and
0.0113 better on the other, both far inside the registered 0.03, and there is no ranking gain that
could have been traded for either. 🚨 **ECE is not comparable ACROSS the two state sets** — the
base rates differ by 9 pp (0.610 vs 0.702) and the arm's own states are the harder population for
both heads (Brier 0.185–0.189 against 0.146–0.147). The comparison that means something is within
a row-pair, down a column.

**The separation ratio is the one row where the control leads on every cell** (1.496 vs 1.423 and
1.700 vs 1.597). It is a magnitude statistic, not a rank statistic, and it is not a registered
endpoint; it says the control's head spreads its values slightly more between decidable siblings
while agreeing with the outcome no more often.

## 5. What this says about the arm

**Forks went into the buffer exactly as designed, and the head came out where it went in.** The
dose was delivered (~1,060 forks/rollout, budget never binding, `fork/sim_steps_share` 1.48, ledger
`e2172333`), the CRN regime was the paired one, the mask rule was uniform, the prefix was counted
once — and on the property the arm exists to buy, the arm and its control are indistinguishable at
a width of ±0.015.

**What the arm's ~44 % of wall-clock bought is visible in the POLICY, not the critic**: sharper
logits (the contested gap threshold roughly doubles), a lower tie rate, a higher blind-spot rate,
and — from the bank line — `H_end` 0.5052 against the control's 0.7473 at the same `--ent-coef
0.02`. A policy that concentrates faster generates more decidable siblings per fork. It did not
generate a head that orders them better.

🚨 **And the honest limit on all of it: with the baseline corrected, the gap this arm was built to
close was ~0.01–0.02 wide, not ~0.09.** The 2026-09-14 record's own corrected numbers say a frozen
trunk refit on exactly this kind of data moves pairwise accuracy by +0.0160 [+0.0000, +0.0313],
NOT DETECTED. **An arm sized to close a gap that is inside its own instrument's width cannot
report a detection, whatever it does** — and this read cannot separate *"forks do not teach the
head"* from *"there was ~0.016 available and this experiment could not resolve it."* Both predict
what was measured.

## 6. THE GUARD — `cond.opp_class_auc.t4_10`

`main.ops.critic_read ai_v13_03_fork --control ai_v12_11_ladder_ctrl10M --step 10000032`, on both
offline 800-game draws, each against that draw's own v6 floor file, quota-matched before any
frame-sensitive row was labelled.

| draw | frame | arm | control | Δ (arm − control) | CI | verdict |
|---|---|---|---|---|---|---|
| **20260910** | MATCHED · battle 797/330/9 (9,551 battles, 21 seeds) | +0.7224 | +0.7097 | **+0.0127** | [−0.0021, +0.0277] | **WITHIN FLOOR** |
| **20260910** | MATCHED · decoder 897/371/9 | +0.7213 | +0.7086 | +0.0127 | [−0.0019, +0.0270] | WITHIN FLOOR |
| **20260911** | MATCHED · battle 797/366/9 (9,531 battles, 21 seeds) | +0.7218 | +0.7131 | **+0.0086** | [−0.0058, +0.0244] | **WITHIN FLOOR** |
| **20260911** | MATCHED · decoder 897/412/9 | +0.7209 | +0.7131 | +0.0078 | [−0.0066, +0.0222] | WITHIN FLOOR |

**The registered guard row is WITHIN FLOOR on both draws, with the SAME sign both times**
(+0.0127 and +0.0086, the arm marginally ahead). **Nothing in the conditioning block is degraded by
the fork treatment.**

**Sibling rows, reported because they are on the same read and because one of them splits.**

| row | draw 20260910 | draw 20260911 |
|---|---|---|
| `cond.opp_class_auc.t1` | +0.0094 [−0.0096, +0.0285] NOT DETECTED (0.5124 vs 0.5030) | +0.0058 [−0.0130, +0.0243] WITHIN FLOOR (0.4945 vs 0.4887) |
| `cond.opp_class_auc.t1_3` | −0.0036 [−0.0197, +0.0130] WITHIN FLOOR (0.5755 vs 0.5791) | **−0.0179 [−0.0351, −0.0005]** NOT DETECTED (0.5628 vs 0.5808) |

⚠️ **`t1_3` SPLITS across the two draws** — inside the floor on one, outside it and negative on the
other, at a per-draw spread (0.0143) comparable to the effect. That is the shape rule 19 assigns to
a row at its eval-draw noise and rule 21 calls **NOT CONFIRMED, never refuted**; it is not the
registered row, and it is recorded rather than promoted. The turn-1 row sits at ~0.50 on draw 2 on
both sides — a decode that has nothing to read that early, as it has on every previous arm.

## 7. THE MIRROR BATTERY at 400 pairs (part 2, rule 25)

Run **after part 1 was landed** (`e90ce7e4`), so a battery failure could never block or colour the
primary read. The 2026-09-11 registered operating point, unchanged, `--games-seed 7`, so these
cells join the nine heads already on this instrument — with the fork arm and a **CONTEMPORANEOUS
`ctrl10M` control in the same window, in the same 4-shard geometry, on the same 400 game
indices**. 3,200 battles, **0 unfinished, 0 errors**, 1–5 ties per cell. The box was otherwise
idle (load 0.18 at launch).

### 7.1 The outcome rows

| cell | **fork arm** | **`ctrl10M`** (contemporaneous) | paired Δ (fork − ctrl), 400 shared indices |
|---|---|---|---|
| **rung B** @3 s contested — **L2** | **0.4844 [0.4570, 0.5117]** | 0.5019 [0.4781, 0.5257] | **−0.0175 [−0.0531, +0.0181] NOT DETECTED** |
| **`grid`** @1 s (unguarded) | 0.2925 [0.2607, 0.3243] | 0.2731 [0.2432, 0.3031] | **+0.0194 [−0.0237, +0.0624] NOT DETECTED** |
| action changed (`grid`) | 62.3 % | 57.9 % | — |
| overruled (rung B, all decisions) | 3.45 % | 2.53 % | — |
| forced (rung B) | 71.5 % | 76.5 % | — |
| realized **K worlds** (rung B) | **6.19** | **5.68** | *not matched — see §7.2* |

**BAR 2 IS NOT MET. Neither cell's L2 lower bound clears 0.50, and the fork arm's point estimate is
BELOW its own contemporaneous control.** `grid` is SEARCH HARMS on both heads, as it has been on
every one of the eleven win-prob heads now on this instrument. **No dividend; branch (a) does not
fire, and part 1 had already settled the primary against it.**

The control's own cell is a useful replicate: it reads **0.5019** here where the 2026-09-11 battery
read `ctrl10M` at **0.5206** on its first 400 pairs and **0.4913** on its pre-registered fresh 400.
Three 400-pair cells of one configuration spanning 0.491–0.521 is the width this instrument
actually has, and it brackets everything measured above.

### 7.2 The mechanism, width-matched (rule 23 — never pooled)

🚨 **The pooled L1 must NOT be read here: the two cells did not race at the same width.** The fork
arm's rung-B cell realized **K = 6.19 worlds** against the control's **5.68** — the arm's leaf is
cheaper per world for its own policy — so its pooled separation-of-raced (0.2751 vs 0.2401) is
exactly the comparison rule 23 forbids. `l1_width_matched.py` (the 2026-09-11 battery's, unmodified)
recomputes it inside bands of realized width:

| L1 = separated / raced | K 0–3 | K 3–4.5 | K 4.5–6 | K 6–8 | K 8+ |
|---|---|---|---|---|---|
| **`ctrl10M`** | — | **0.095 [0.082, 0.109]** | 0.254 [0.236, 0.273] | 0.339 [0.320, 0.358] | 0.257 [0.229, 0.287] |
| **fork arm** | 0.051 [0.031, 0.083] | **0.076 [0.065, 0.089]** | 0.291 [0.272, 0.311] | 0.383 [0.366, 0.401] | 0.339 [0.316, 0.363] |
| ratio (fork / ctrl) | — | **0.80×** | 1.15× | 1.13× | **1.32×** |

**The mechanism row SPLITS BY WIDTH, and the split has a sign change in it.** Where the race is
most starved (K 3–4.5) the fork arm separates **LESS** than its control, 0.80×, Wilson intervals
disjoint. From K 4.5 upward it separates **MORE** — 1.15× / 1.13× / 1.32×, every pair of intervals
disjoint. **More separation at width did not become an outcome**: the same cell's L2 is the one
that reads 0.4844 against 0.5019.

🚨 **And the absolute level is not comparable to any earlier campaign.** These cells raced at
K 5.7–6.2 on an idle box against the 2026-09-11 battery's 4.85–5.48 and the 2026-09-14 battery's
4.16–4.59, and their L1 is correspondingly 0.24–0.28 against 0.083–0.150. That is rule 23's width
effect at full size, and it is why the only reading taken here is between two cells from the same
window.

## 8. THE RECHECK OF THE 2026-09-14 RECORD, in full

[`recheck_paired_refit.py`](recheck_paired_refit.py) ·
[`recheck_paired_refit.json`](recheck_paired_refit.json). The banked fork dataset
(5,040 forks / 14,803 successors / 14,780 branch rows) is re-scored on the **identical** held-out
split — same seed, same battles, same 562 non-tied test pairs — with the original head's `V`
indexed by BRANCH ROW instead of by successor.

**The defect.** `refit.py` builds `b_idx` as an index into the successor array and produces
`tab["pooled"]` / `tab["V0"]` correctly, one entry per branch row. Every refit head is then scored
as `head_V(h, tab["pooled"])` — per branch row, correct. The original head alone is scored as
`V["original"] = V0`, which is per SUCCESSOR, while `read_head` addresses it by branch-row index.
The two spaces agree only until the first branch that HAS a successor but is dropped
(stall-capped, or a non-binary outcome); 24 such rows exist, the first divergence is at branch row
**1,880**, and the maximum offset is **23**. **12,900 of 14,780 rows are mis-scored.**

| held-out read, 562 non-tied pairs | pairwise accuracy | sep ratio | ECE | Brier |
|---|---|---|---|---|
| `original` — **as published** | 0.5169 [0.4799, 0.5524] | 1.020 | 0.1246 | 0.2154 |
| **`original` — corrected** | **0.5872 [0.5526, 0.6225]** | **1.793** | **0.0287** | **0.1412** |
| `control_bce` (unaffected) | 0.6032 [0.5690, 0.6374] | 1.711 | 0.0520 | 0.1417 |
| `rank@0.1` (unaffected) | 0.5925 [0.5575, 0.6278] | 1.723 | 0.0421 | 0.1398 |

| the record's three headline deltas | as published | **corrected** |
|---|---|---|
| control BCE − original | **+0.0863 [+0.0384, +0.1347] DETECTED** | **+0.0160 [+0.0000, +0.0313] NOT DETECTED** |
| best rank − original | +0.0756 [+0.0276, +0.1233] DETECTED | **+0.0053 [−0.0083, +0.0169] NOT DETECTED** |
| best rank − control | −0.0107 [−0.0249, +0.0028] NOT DETECTED | −0.0107 [−0.0249, +0.0028] NOT DETECTED *(unchanged — both heads correctly scored)* |

**What changes, stated as claims:**

1. **"The promoted win-prob head ranks siblings AT CHANCE (0.517)" is REFUTED.** It reads
   0.5872 [0.5526, 0.6225] on that record's own data, and 0.5679 [0.5515, 0.5843] on a fresh draw
   here. Its CI clears 0.50 in both.
2. **"The frozen trunk already holds the ordering — +0.086 from plain BCE, DETECTED" is NOT
   SUPPORTED at the registered standard.** The corrected delta is +0.0160 with a lower bound of
   +0.0000. The trunk conclusion may still be right — the head does read `value_pooled` well — but
   the *evidence for it* was the size of the gap, and the gap is ~5× smaller.
3. **"Calibration came WITH the ranking rather than being traded for it" INVERTS.** The original
   head's ECE is 0.0287; the control refit's is 0.0520 and the ranking refit's 0.0421. The refits
   made calibration **worse**.
4. **"The pairwise RANKING term buys NOTHING" STANDS, untouched** — it is the one delta taken
   between two correctly-scored heads.
5. **UNDERSTANDING rule 25 quotes "the promoted head 0.517"** as one of its three reference points.
   That clause needs amending. *(This record does not edit `UNDERSTANDING.md` or `ledger.md`; §11
   is the paragraph to append.)*
6. **`designs/training/forks.md` §1 reproduces the same table** and its "a coin. The CI straddles
   0.50" row. Not edited here for the same reason; the amendment belongs with the ledger entry.

**Nothing in `paired_refit_discrimination_2026-09-14/` is edited.** A record is what was believed
at the time; the correction lives here and in the ledger.

## 9. The registered predictions, scored

`PREDICTION.md`, committed at `55cf08a7` before the first fork.

| # | registered | outcome |
|---|---|---|
| **P1** | the arm ON ITS OWN states reads **[0.59, 0.66]**, ≈ 0.62, and its CI **clears 0.60** | **REFUTED on both clauses.** 0.5754 [0.5608, 0.5897] — below the band and entirely below 0.60. |
| **P2** | paired Δ (arm − ctrl) on the arm's forks is **+0.01 to +0.06, DETECTED** | **REFUTED.** −0.0003 [−0.0148, +0.0136], NOT DETECTED, point estimate negative. |
| **P3** | the same Δ on the control's forks is positive, +0.00 to +0.04, **not** predicted detected | **HELD** (and uninformatively so): +0.0067 [−0.0096, +0.0231], inside the band, not detected. |
| **P4** | the state-distribution effect: \|own − cross\| < **0.04** for either head | **HELD, with room.** 0.0008 (arm head) and 0.0078 (control head). |
| **P5** | the control's ON-OWN level is in **[0.55, 0.62]** — the corrected 0.5872, not the published 0.517 | **HELD.** 0.5679 [0.5515, 0.5843]. |
| **P6** | tie rate **0.76–0.83** on both state sets | **HELD on the control (0.8058), REFUTED on the arm (0.7451)** — below the band, and the direction is a finding (§1.6). |
| **P7** | blind-spot rate **3–7 %** on both | **HELD.** 5.53 % and 4.55 %, with the two intervals disjoint from each other. |
| **P8** | top-1/top-2 interchangeable, \|Δ\| < **0.01** | **HELD on both.** 0.0060 and 0.0047. |
| **P9** | the arm's ECE **0.02–0.08**, Brier **0.12–0.18**; not worse than the control's by > 0.03 | **HELD on the arm's states** (0.0597 / 0.1848 — Brier 0.0048 over the band's top, called as held at the band's edge and reported) **and on the control's** (0.0188 / 0.1460). The arm is 0.0032 worse on one set, 0.0113 better on the other. |
| **P10** | the guard: \|Δ\| < **0.0245**, NOT DETECTED, same sign on both draws | **HELD on all three clauses.** +0.0127 [−0.0021, +0.0277] and +0.0086 [−0.0058, +0.0244], both WITHIN FLOOR, both positive. |
| **P11** | rung-B L2 for the arm in **[0.48, 0.53]**, lower bound NOT clearing 0.50 — NO DIVIDEND | **HELD on both clauses.** 0.4844 [0.4570, 0.5117]. |
| **P12** | `grid` for the arm in **[0.22, 0.42]** — SEARCH HARMS | **HELD.** 0.2925 [0.2607, 0.3243], upper bound far below 0.50. |
| **P13** | separation-of-raced at matched K (band 3–4.5) is **1.0–1.8×** the control's, Wilson intervals **not** predicted disjoint | **REFUTED on both clauses, and in the opposite direction:** 0.076 vs 0.095 = **0.80×**, intervals **disjoint**. The arm separates MORE than the control only from K 4.5 upward (1.15× / 1.13× / 1.32×, all disjoint) — a band-dependent split the prediction did not anticipate. |
| **P14** | the paired Δ on rung-B L2 is NOT DETECTED, \|Δ\| < **0.04** | **HELD.** −0.0175 [−0.0531, +0.0181]. |

**Two predictions were wrong in the way that matters: P1 and P2, which were the read.** P1 was
registered at ~50 % by its own text and the read came in below even the train-frame meter's level;
P2 predicted a detected head effect and the point estimate is negative.

## 10. Hazards and findings about the instruments

Every one of these is a finding.

1. 🚨 **A successor-indexed vector read at branch-row positions produces a number that looks like
   a null.** The 2026-09-14 original-head row is the case (§1, §8), and the failure mode is
   dangerous precisely because it is quiet: mis-indexing a monotone value function against nearby
   rows does not produce garbage, it produces **0.517** — the most publishable possible artifact,
   an exact null on a rank statistic. `score_forks.py` now recomputes the index independently and
   REFUSES on disagreement.
2. 🚨 **The condition for that defect is the NORMAL case, not an edge case.** On these two fresh
   datasets, 14,751 and 14,614 of ~14,800 rows sit at a position different from their successor
   index. Any future reuse of `refit.py`'s scoring path is mis-scored by default.
3. **A contested-gap QUANTILE selects the same FRACTION of decisions at very different absolute
   gaps.** The arm's 40th-percentile top-2 logit gap is 1.45–1.63 against the control's 0.81–0.89.
   That is the selector behaving as designed (`forks.md` §3: a fixed threshold would anneal the
   treatment off), but it means the two datasets' "contested" populations are defined relative to
   two different policies and are **not the same states**. This is exactly why the read is taken
   in both directions.
4. **Each policy's forks are continued against its OWN pool sentinels.** The ecology differs
   between the two datasets by construction (a sentinel is a run's own earlier snapshot), so the
   absolute branch win rates — 0.618 vs 0.715 — are NOT a strength comparison and are never quoted
   as one.
5. **A base-rate difference of 9 pp makes ECE incomparable across state sets.** Both heads read
   materially better-calibrated on the control's forks than on the arm's; that is a property of the
   population, not of either head.
6. **`main.ops.eval_trace_gen`'s `<run>@<step>` form worked here**, resolving through
   `eval_traces/step_10000032/snapshot.zip` — the OPS DEFECT banked in `e2172333` is specific to
   `main.anchors`' `resolve_model_ref` path, not to every `@step` spelling. Worth knowing before
   the amendment is written narrowly or broadly.
7. **The sentinel-count trap is real and was avoided by the deviation, not by luck.** Read at
   `final_model.zip` (10,027,008) the arm would have carried FOUR sentinels below its step against
   the control's three, and `critic_read`'s frame would have been 13 cells against 12. The
   registered "`final_model.zip`, 10,027,008" is not compatible with a matched guard.
8. **The corrected effect size is inside the instrument's own width.** +0.0160 [+0.0000, +0.0313]
   for a frozen-trunk refit is the best prior estimate of what fork data can buy this head, and a
   ±0.015 Δ instrument cannot resolve it. **No arm aimed at this target can report a detection at
   this width** — which is a design finding about the next arm, not about this one.
9. **Nothing under `src/` was changed by this read**, so no test or gate obligation attaches. All
   analysis code lives in this directory.

## 11. Ready-to-append ledger paragraph

> ### 🎯 THE FORK ARM'S REGISTERED READ — forks into the PPO buffer did NOT teach the head to rank siblings (branch d: −0.0003 [−0.0148, +0.0136] paired, NOT DETECTED, and the CI does not clear the registered 0.60) — and 🚨 **the registered comparator was an INDEXING ARTIFACT: the promoted win-prob head never ranked siblings at chance (0.5872, not 0.5169), which collapses BOTH of the 2026-09-14 record's DETECTED rows** (2026-09-16)
>
> Record `designs/research_state/measurements/fork_arm_read_2026-09-16/` (`PREDICTION.md`
> registered at `55cf08a7` before a single fork existed, including the defect below, which was
> found first). **10,080 fresh three-branch common-random-number forks (30,240 rollouts) over two
> policies' own offline full-capture eval trees, plus two new 9,600-battle trees generated for the
> arm; CPU only, zero errors, 204/204 determinism re-runs identical, 0.09 %/0.14 % stall-capped.**
>
> 🚨 **THE COMPARATOR FIRST. `paired_refit_discrimination_2026-09-14/refit.py` scores the ORIGINAL
> head with a successor-indexed `V` read at BRANCH-ROW positions; every refit head in the same
> script is scored correctly. On that record's own banked dataset the two index spaces diverge from
> branch row 1,880 and 12,900 of 14,780 rows are mis-scored.** Re-scored on the identical held-out
> split and the identical 562 non-tied pairs, the promoted win-prob critic reads
> **0.5872 [0.5526, 0.6225], not 0.5169** — the CI clears 0.50 — with **ECE 0.0287, not 0.1246**.
> **"The promoted head ranks siblings AT CHANCE" is REFUTED**, and the record's two DETECTED rows
> collapse with it: the frozen-trunk BCE refit's **+0.0863 [+0.0384, +0.1347] DETECTED becomes
> +0.0160 [+0.0000, +0.0313] NOT DETECTED**, the ranking refit's +0.0756 becomes +0.0053, and
> **"calibration came WITH the ranking" INVERTS** (the refits made ECE worse, 0.0287 → 0.0520 →
> 0.0421). **What STANDS untouched is the one delta taken between two correctly-scored heads: the
> pairwise ranking term buys nothing, −0.0107 [−0.0249, +0.0028].** The failure mode is worth its
> own line: mis-indexing a monotone value function against nearby rows does not produce garbage, it
> produces **an exact null on a rank statistic**. ⚠️ **UNDERSTANDING rule 25 and
> `designs/training/forks.md` §1 both quote the 0.517 and need amending.** Nothing in the
> 2026-09-14 record is edited — a record is what was believed at the time.
>
> **THE ARM'S READ, against that corrected frame.** `ai_v13_03_fork` at its final promoted node
> **10,000,032** (the declared deviation from `final_model.zip` at 10,027,008: step-matched to the
> control, and matched at THREE sentinels where the final model would carry four) against
> `ai_v12_11_ladder_ctrl10M@10000032`, both heads scored on BOTH policies' fork sets:
> **on its own states the arm reads 0.5754 [0.5608, 0.5897] — the registered bar was that the CI
> CLEAR 0.60 and the whole interval is below it — and the paired Δ against the control on the same
> forks and the same pairs is −0.0003 [−0.0148, +0.0136], with +0.0067 [−0.0096, +0.0231] on the
> control's forks. BOTH NOT DETECTED, the two point estimates on opposite sides of zero. BRANCH
> (d): forks did not teach the head.** The puzzle branch (d) reserved — the train-frame
> `fork/pairwise_acc` of 0.5817 — **dissolves**: the offline recorded-state read of 0.5754
> reproduces the on-stream meter to within 0.006. **The number that never reproduced is the 0.517
> the arm was aimed at**; both heads sat at ~0.575 all along.
>
> **THE TREATMENT DID MOVE SOMETHING, AND IT IS THE STATE DISTRIBUTION.** From the same 5,040
> forks the arm's own contested states yield **3,745 non-tied pairs against the control's 2,849
> (+31 %)**, a tie rate of **0.745 vs 0.806**, a blind-spot rate of **5.53 % [4.93, 6.20] vs
> 4.55 % [4.01, 5.16]** (Wilson intervals disjoint), and a contested-gap threshold of **1.45–1.63
> against 0.81–0.89** — the selector's 40th percentile bites at roughly twice the absolute logit
> gap, the same sharpening `H_end` 0.5052 vs 0.7473 shows. **Training on forks moved the policy's
> decision population; it did not move the head's ability to order two states one move apart.**
> Top-1 and top-2 stay outcome-interchangeable on both policies (|Δ| 0.0060 / 0.0047) and the
> random branch still costs ~3 pp (2.86 / 3.32).
>
> **THE BATTERY DOES NOT PAY EITHER (rule 25, ≥400 pairs, run only after part 1 landed at
> `e90ce7e4`).** 3,200 battles, 0 unfinished, the 2026-09-11 operating point with a
> CONTEMPORANEOUS `ctrl10M` control in the same window on the same 400 game indices:
> **rung-B L2 0.4844 [0.4570, 0.5117] against the control's 0.5019 [0.4781, 0.5257], paired
> −0.0175 [−0.0531, +0.0181] NOT DETECTED**, neither lower bound clearing 0.50; unguarded `grid`
> **0.2925 [0.2607, 0.3243] vs 0.2731 [0.2432, 0.3031]**, paired +0.0194 NOT DETECTED, **both
> SEARCH HARMS** — eleven win-prob heads on this instrument and none is a usable leaf.
> 🚨 **The two cells did NOT race at the same width (K 6.19 vs 5.68), so the pooled L1 is the
> comparison rule 23 forbids; width-matched, the mechanism row SPLITS WITH A SIGN CHANGE:** the arm
> separates **0.80×** the control's rate at K 3–4.5 and **1.15× / 1.13× / 1.32×** at K 4.5–6 / 6–8 /
> 8+, **every one of those four bands with the Wilson intervals disjoint**. More separation at
> width, no outcome — the shape Part C found for the offline refit head, now reproduced by a
> TRAINED head. (These cells raced at K 5.7–6.2 on an idle box against 4.2–5.5 in the two previous
> batteries, with L1 0.24–0.28 against 0.083–0.150; no cross-campaign L1 comparison is available.)
> The control's own 400-pair cell reads 0.5019 here where the 2026-09-11 battery read 0.5206 and
> then 0.4913 on a fresh 400 — a 0.491–0.521 span that brackets every number in this paragraph.
>
> **THE GUARD HOLDS.** `cond.opp_class_auc.t4_10`, matched frame, arm vs control, draw 20260910:
> **+0.7224 vs +0.7097, Δ +0.0127 [−0.0021, +0.0277], WITHIN FLOOR**; draw 20260911 **+0.7218 vs
> +0.7131, Δ +0.0086 [−0.0058, +0.0244], WITHIN FLOOR** — same sign on both draws, the arm
> marginally ahead. ⚠️ The sibling `t1_3` SPLITS (−0.0036 WITHIN FLOOR on one draw,
> −0.0179 [−0.0351, −0.0005] on the other), the shape rule 19 assigns to a row at its eval-draw
> noise; it is not the registered row. Both trees are
> 9,600 traces, zero shortfall, 12 opponents with the SAME three sentinel steps (4,000,032 /
> 6,000,000 / 8,000,016), quota-matched before any frame-sensitive row was labelled.
>
> 🚨 **AND THE DESIGN FINDING THAT OUTLASTS THE ARM: with the baseline corrected, the gap this arm
> was built to close is ~0.016 wide, not ~0.086 — inside the ±0.015 width of the instrument that
> reads it.** The corrected frozen-trunk refit (+0.0160 [+0.0000, +0.0313]) is the best available
> estimate of what fork data can buy this head, and no arm aimed at it can report a detection at
> this width. **This read therefore cannot separate "forks do not teach the head" from "there was
> ~0.016 available and this experiment could not resolve it"** — both predict what was measured.
> Descriptors, not endpoints: `fork/pairwise_acc` 0.5817 (rule 20, live frame); `H_end` 0.5052 vs
> `ctrl10M`'s 0.7473 at the same `--ent-coef 0.02`, a single draw each; the SmallRL anchor 0.530
> [0.433, 0.625] — **and the 10M comparator cell was built the same evening** (ledger `6cc9f397`:
> three win-prob controls read 0.440 / 0.380 / 0.330, an **eleven-point three-seed floor**), so the
> arm's +0.090 over the best control is **INSIDE the floor — no fire, NOT separated**; and STRENGTH is **UNREADABLE**
> here — the two 4-node ladders read 1995.4 ± 15.8 against 2018.7 ± 16.7 on six dense pairs each,
> and the 2026-09-14 external-anchor campaign moved exactly this class of node by −42.4 Elo by
> adding two edges. Instrument notes: `eval_trace_gen`'s `<run>@<step>` resolves through
> `eval_traces/step_N/snapshot.zip` and does NOT share `main.anchors`' snapshot defect; reading the
> arm at `final_model.zip` would have given it FOUR sentinels against the control's three and made
> the guard's frame unmatched. Tag: **MEASURED (MAJOR) · fork arm branch (d), NOT DETECTED · the
> registered comparator REFUTED as an indexing artifact · two 2026-09-14 DETECTED rows collapse ·
> the ranking-term null STANDS · state distribution moved, the head did not · battery NO DIVIDEND
> at 400 pairs with the L1 splitting by width and changing sign · guard WITHIN FLOOR**.

## 12. Files

| | |
|---|---|
| [`PREDICTION.md`](PREDICTION.md) | the registration, committed at `55cf08a7` before any fork existed |
| [`run_forks.sh`](run_forks.sh) | the two fresh fork datasets — invokes the 2026-09-14 builder by path, never a copy |
| [`score_forks.py`](score_forks.py) · [`score.json`](score.json) | the primary read: two heads × two state sets, the paired deltas, every descriptor |
| [`recheck_paired_refit.py`](recheck_paired_refit.py) · [`recheck_paired_refit.json`](recheck_paired_refit.json) | §8 — the 2026-09-14 record re-scored with the original head correctly indexed |
| [`run_guard.sh`](run_guard.sh) | `main.ops.critic_read` on both offline draws against their v6 floors |
| [`run_battery.sh`](run_battery.sh) · [`battery_report.json`](battery_report.json) · [`battery_deltas.json`](battery_deltas.json) · `rows_<cell>__<head>.jsonl.gz` | the 400-pair mirror battery (§7), scored by the 2026-09-14 `report.py` and the 2026-09-11 `l1_width_matched.py`, both unmodified; every row archived |
| `guard_hp800_critic_read.json` · `guard_hp800b_critic_read.json` | the two guard reads, as `critic_read` wrote them |
| not committed | the forks, the successor tensors and the two eval trees, under `/home/goodlad/.claude/jobs/9ab51de6/tmp/fork_read/` and `.../tmp/hp_eval/hp800{,b}/ai_v13_03_fork/` |
