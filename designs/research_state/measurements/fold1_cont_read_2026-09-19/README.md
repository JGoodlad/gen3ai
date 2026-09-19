# THE CONVERGED-ENDPOINT READ OF THE ERA-1 FOLD — `ai_v13_08_fold1_cont`, 2026-09-19

**The fold ran to convergence and the registered question is whether the last 25 % of the teaching
moved any row.** `ai_v13_08_fold1_cont` is a FORK of `ai_v13_07_fold1` at 81,100,800 carrying the
**identical distill block** (its `model_config.json` differs from the fold's in **no field at all**,
verified here by diff), pin `6eb9c776`, config v119 / `gen3_critic_route_wave_v1`, `--fork-lr 2.8e-5
--fork-lr-freeze` (realized dose **4.272e-9 = 0.20× the v8 reference**, `main.dose` re-read here),
`--distill-coef 0.1761`, both teachers in the stable pool each piloting its own pinned team. It ran
to **87,097,344 = +12,091,392 cumulative post-fork**, and 🚨 **its distill STOP RULE FIRED at
82,673,664 = +7.667M** — the event the run existed to obtain.

Every bar below was fixed in [`PREDICTION.md`](PREDICTION.md) and **committed before any registered
number in this directory existed** (commit `e59e71d0`, 07:54 PT; the first battle started 07:58).
Instruments, opponents, manifests, seeds, concurrency, cell sizes and the decision rule are
**REUSED VERBATIM** from the +6M read, [`fold1_read_2026-09-19/`](../fold1_read_2026-09-19/), so
the two reads are one series.

**Sign convention: `Δ = fold path − arm W`.** Positive means ahead of the frozen parent.
**The rule, fixed in advance:** OUTSIDE THE FLOOR iff **(a)** `|Δ| > floor` **AND** **(b)** the Δ's
own 95 % CI excludes the floor POINT; otherwise WITHIN FLOOR at n = 2. 🚨 **ONE PAIR BOUNDS A
FLOOR; IT DOES NOT ESTIMATE ONE** (rules 19/22); WITHIN FLOOR is never "equivalent" (rule 6).

---

## 0. THE VERDICT TABLE — every registered row, finding beside floor

| # | row | **finding** `Δ = fold path − arm W` + CI | **floor** | (a) | (b) | **verdict** |
|---|---|---|---|---|---|---|
| **1a** | untaught meter, **+7.59M** (the STOP POINT) | **+2.81 pp** [−2.31, +9.37], 4 of 8 | 3.69 (imported) | ❌ | ❌ | **WITHIN FLOOR at n = 2** |
| **1b** | untaught meter, **+9.09M** | **+5.13 pp** [−1.06, +11.31], 6 of 8 | 3.69 (imported) | ✅ | ❌ | **WITHIN FLOOR at n = 2** |
| **1c** | untaught meter, **+12.09M** | **+5.75 pp** [+1.19, +11.06], **7 of 8** | 3.69 (imported) | ✅ | ❌ | **WITHIN FLOOR at n = 2** *(CI clear of zero)* |
| **2a** | per-slice, **BIG-5** (balance), **+7.59M** | **−0.0712** [−0.1196, −0.0224] | **0.0475** (measured, same cell) | ✅ | ❌ | **WITHIN FLOOR** — and **NEGATIVE** |
| **2b** | per-slice, **BIG-5**, **+12.09M** | **−0.0525** [−0.1010, −0.0036] | 0.0475 | ✅ | ❌ | **WITHIN FLOOR** — and **NEGATIVE** |
| **2c** | per-slice, **DDTAR** (offense), **+7.59M** | **+0.1250** [+0.0762, +0.1730] | **0.0850** (measured, same cell) | ✅ | ❌ | **WITHIN FLOOR at n = 2** |
| **2d** | 🚨 **per-slice, DDTAR, +12.09M** | 🚨 **+0.1538** [+0.1051, +0.2013] | 0.0850 | ✅ | ✅ | 🚨 **OUTSIDE THE FLOOR** |
| **3** | `SmallRL` greedy, away, 100 games, +12.09M | **−0.070** [−0.204, +0.067] | 0.090 | ❌ | ❌ | **WITHIN FLOOR at n = 2** |
| **4** | ladder at matched snapshot COUNT | — | 25.2 Elo | — | — | 🚨 **NOT RUNNABLE — ZERO promotions again, n = 0** |
| 5 | collateral | *row 1 IS the collateral read* | — | — | — | `main.exploitability` **NOT APPLICABLE** (§5) |

### 🚨 ROW 2d IS THE FIRST REGISTERED ROW IN THIS FOLD PROGRAMME TO CLEAR ITS BAR

On the **DDTar** slice at the converged endpoint the fold path is **+0.1538 [+0.1051, +0.2013]**
over its zero-head-start parent against a **0.0850 seed floor measured on that very cell**. Clause
(a) passes at 1.8× the floor; clause (b) passes because the interval's lower end, +0.1051, is above
the floor point. **Nothing in the +6M read cleared anything.**

⚠️ **And it is ONE cell, at ONE of three depths, on ONE arm, with seniority unresolved.** Rules
19/22: a single-arm detection is a **CANDIDATE** until its own seed replicate agrees, and more steps
on the same arm are not a replicate.

### 🚨 NO REGISTERED BRANCH IS MET — and that is reported, not repaired

The GO fixed three branches. **The result satisfies none of them**, and the registration's only
tie-break covered the case where (b) and (c) BOTH hold, which is the opposite of what happened:

| branch | its condition | met? |
|---|---|---|
| **(a) THE FOLD PAID AT CONVERGENCE** | a row clears its bar at **≥ 2 of the 3** new points | ❌ — DDTar clears at **1 of 3** (+12.09M only) |
| **(b) EXTRACTION SATURATED BEFORE AGREEMENT DID** | the **+3M peak stands** and later points are flat/down | ❌ — **the +3M peak does NOT stand on either slice**; DDTar's maximum IS the endpoint |
| **(c) NOT DETECTED AT CONVERGENCE** | nothing clears **and** there is no trajectory | ❌ — something clears, and there is a trajectory |

**The honest description, in the registration's own vocabulary:** *the fold paid on ONE of its two
taught slices, at the ENDPOINT only, and the trajectory on that slice is RISING rather than
saturated.* It is nearest to (a) and misses only the ≥ 2-of-3 clause — and the reason it misses is
itself the finding: at +7.59M and +9.09M the DDTar effect was **+0.1250** and **+0.1138**, large but
not large enough for clause (b); it cleared at +12.09M because **it grew after the stop rule fired**.

### What this read does NOT license

* **It does not say the fold recipe works.** One arm, one slice of two, one depth of three
  (rules 19/22), against a floor bounded by a single pair.
* **It does not separate EXTRACTION from SENIORITY** — and the confound is now **twelve million
  steps**, twice what it was at +6M. `ai_v13_09_wcont` is the arm that splits them; it is LIVE and
  was **not read** (§8).
* **It does not say the off-slice row moved.** +5.75 pp is WITHIN an imported floor.
* **It does not price the missing CONTINUATION CONTROL.** Every delta is against a FROZEN parent.
* **It says nothing about folds in general**, at another dose, with other teachers, or at 277M.
* 🚨 **It does not license reading the DDTar rise as a TREND.** Six points on one arm is a
  description of that arm (the standing short-window refusal, 5-for-5 on this campaign).

---

## 1. 🚨 THE WITHIN-FOLD TRAJECTORY — the deliverable, all three rows in one frame

Every cell below is **paired on the same games**: the untaught columns come from ONE invocation of
eight refs over the same 8 teams × 200 games; the slice columns from one manifest, one opponent, one
seed, 800 games per cell.

| depth (cumulative post-fork) | run | untaught pp | untaught Δ vs arm W | Big-5 rate | Big-5 Δ | DDTar rate | DDTar Δ |
|---|---|---:|---|---:|---|---:|---|
| **+1.00M** | `ai_v13_07_fold1` | 47.62 | **+1.44** [−0.44, +3.19] | 0.4675 | −0.0250 [−0.0737, +0.0239] | 0.5375 | +0.0675 [+0.0185, +0.1160] |
| **+3.00M** | `ai_v13_07_fold1` | 51.56 | **+5.38** [+1.63, +9.19] | 0.4675 | −0.0250 [−0.0737, +0.0239] | **0.5938** | +0.1238 [+0.0749, +0.1717] |
| **+6.09M** | `ai_v13_07_fold1` | 51.31 | **+5.13** [+1.06, +9.00] | 0.4313 | −0.0612 [−0.1097, −0.0124] | 0.5375 | +0.0675 [+0.0185, +0.1160] |
| — | 🚨 **FORK CROSSING** → `ai_v13_08_fold1_cont` @ 81,100,800 | — | — | — | — | — | — |
| **+7.59M** *(the STOP POINT)* | `ai_v13_08_fold1_cont` | **49.00** | **+2.81** [−2.31, +9.37] | **0.4213** | −0.0712 [−0.1196, −0.0224] | 0.5950 | +0.1250 [+0.0762, +0.1730] |
| **+9.09M** | `ai_v13_08_fold1_cont` | 51.31 | **+5.13** [−1.06, +11.31] | 0.4225 | −0.0700 [−0.1184, −0.0212] | 0.5837 | +0.1138 [+0.0649, +0.1619] |
| **+12.09M** | `ai_v13_08_fold1_cont` | **51.94** | **+5.75** [+1.19, +11.06] | 0.4400 | −0.0525 [−0.1010, −0.0036] | 🚨 **0.6238** | 🚨 **+0.1538** [+0.1051, +0.2013] |

*arm W — the frozen PARENT, +0: untaught **46.19** pp · Big-5 **0.4925** · DDTar **0.4700**.*
*W_b — the seed-floor arm, +0: untaught **49.88** pp · Big-5 **0.5400** · DDTar **0.5550**.*
*t1 Big-5 on its own pin **0.5763** · t2 DDTar on its own pin **0.4775**.*

### 1.1 What the trajectory says, row by row

**Did the last 25 % of the teaching move any row?** The GO's question, answered three ways:

| row | +6.09M → +12.09M | reading |
|---|---|---|
| **untaught (off-slice)** | 51.31 → 51.94, **Δ = +0.62 pp** [−2.75, +3.88], 5 of 8 teams | **NO.** The off-slice row is FLAT across the last half of the budget — inside its own noise, inside every floor. It moved in the first 3M and never again. |
| **Big-5 (taught, balance)** | 0.4313 → 0.4400, **Δ = +0.0087** [−0.0398, +0.0572] | **NO.** Flat, and NEGATIVE against the parent at **every one of the six depths**. |
| **DDTar (taught, offense)** | 0.5375 → 0.6238, **Δ = +0.0863** [+0.0379, +0.1340] | 🚨 **YES.** The only row that moved, and the move is what carries 2d over its bar. |

**Was the +3M peak the maximum?** **NO, on either slice — and for opposite reasons.**

* **Big-5** peaks at **+1.00M / +3.00M (0.4675, an exact tie)** and the endpoint is 0.0275 below it.
  The slice the fold drew MORE of its taught signal from is the slice it never improved on: its best
  depth is its first.
* **DDTar** peaks at **the ENDPOINT (0.6238)**. Its shape is a double rise —
  0.5375 → 0.5938 → 0.5375 → 0.5950 → 0.5837 → 0.6238 — with the dip sitting exactly on the +6.09M
  point the earlier read stopped at.

🚨 **THE +6M READ STOPPED ON A LOCAL MINIMUM OF THE DDTAR ROW.** Its headline — *"the fold peaked at
+3M and fell by +6M on BOTH taught teams"* — was true of the two points it had, and the third point
reverses it: +7.59M returns to 0.5950, within 0.0012 of the +3M value it had "fallen from". ⚠️ This
is the SIXTH time on this campaign a short window would have been read as a direction and been
wrong, and it is recorded as such rather than as a correction of the earlier read, which reported
exactly what its own two points said.

### 1.2 The dip at the fork crossing, and the refusal attached to it

**Every row is at or near its worst at +7.59M**, the first continuation checkpoint: untaught
**49.00** (the lowest of the four post-+1M points, Δ vs arm W falling to +2.81 on 4 of 8 teams) and
Big-5 **0.4213** (its global minimum). DDTar is the exception — it is *up* there.

⚠️ **This read cannot say whether that is the FORK or the STEPS.** The +6.09M → +7.59M leg crosses
the `ai_v13_07_fold1` → `ai_v13_08_fold1_cont` boundary: the fork re-seeds the optimizer path,
re-pins the LR (to the same frozen 2.8e-5), re-seeds the pool (from the fold — the same 20 arm-W
files, verified) and opens a new event file. The matchup hash `0a7b730a4d` is unchanged, so it is
**not** a rule-15 opponent-regime boundary — but it is a discontinuity, it is the only leg in the
table that carries one, and the untaught dip is the only non-monotone feature of that column.
**Stated, not resolved.**

---

## 2. ROW 1 — THE UNTAUGHT METER (off-slice; this IS the collateral read)

`python -m main.untaught_meter`, registry opponent `untaught_meter_opponent`
(= `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` @24,000,000), the untaught-8 manifest in
its canonical order, **200 games/team, `--seed 0`, concurrency 1, 8 single-concurrency shards, ALL
EIGHT REFS IN ONE INVOCATION**. **12,800 battles, 0 TIMEOUTS (0.0 %)** — rule 12's 25 %
INCONCLUSIVE threshold nowhere near.

### 2.1 ✅ ALL FIVE REGISTERED REPRODUCTION CHECKS PASS, EXACTLY, ON THE PER-TEAM ROWS

| ref | banked | **here** | Δ | per-team rows identical? |
|---|---:|---:|---:|---|
| arm W | 46.19 pp (739/1600) | **46.19 pp (739/1600)** | **0.00** | ✅ all 8 |
| W_b | 49.88 pp (798/1600) | **49.88 pp (798/1600)** | **0.00** | ✅ all 8 |
| fold @ +1M | 47.62 pp (762/1600) | **47.62 pp (762/1600)** | **0.00** | ✅ all 8 |
| fold @ +3M | 51.56 pp (825/1600) | **51.56 pp (825/1600)** | **0.00** | ✅ all 8 |
| fold @ +6.09M | 51.31 pp (821/1600) | **51.31 pp (821/1600)** | **0.00** | ✅ all 8 |

**This is the fourth independent confirmation that the meter is deterministic at seed 0 /
concurrency 1, and the first on five refs at once.** The 3.69 pp floor re-measures at exactly
**−3.69 pp** on the identical games. The five new levels are therefore comparable to the banked
three not by argument but because they are **the same games**.

### 2.2 The levels, and every team

| ref | level | wins/finished | `U_61590463` | `U_92832108` | `U_ce35b736` | `U_9909f2e9` | `U_9d5f8458` | `U_f7ba5702` | `U_90b94599` | `U_dbf81d8e` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fold @ +1M | 47.62 | 762 / 1600 | 50.00 | 43.50 | 53.00 | 40.00 | 51.50 | 44.50 | 47.00 | 51.50 |
| fold @ +3M | 51.56 | 825 / 1600 | 44.50 | 44.50 | 58.50 | 39.00 | 56.00 | 52.50 | 63.00 | 54.50 |
| fold @ +6.09M | 51.31 | 821 / 1600 | 48.50 | 37.50 | 60.50 | 45.50 | 56.00 | 54.50 | 60.50 | 47.50 |
| **cont @ +7.59M** | **49.00** | 784 / 1600 | 42.50 | 36.50 | 56.50 | 43.00 | 50.00 | 47.00 | 70.00 | 46.50 |
| **cont @ +9.09M** | **51.31** | 821 / 1600 | 42.50 | 43.00 | 54.00 | 45.50 | 62.00 | 53.50 | 67.50 | 42.50 |
| **cont @ +12.09M** | **51.94** | 831 / 1600 | 50.50 | 41.00 | 53.50 | 42.50 | 63.00 | 50.00 | 67.00 | 48.00 |
| **arm W** (parent) | 46.19 | 739 / 1600 | 48.50 | 39.00 | 49.50 | 38.00 | 47.00 | 48.00 | 48.00 | 51.50 |
| **W_b** (other seed) | 49.88 | 798 / 1600 | 58.00 | 53.00 | 42.50 | 48.00 | 46.00 | 48.00 | 51.50 | 52.00 |

### 2.3 The paired, team-clustered contrasts (rule 10, 20,000 draws, ONE shared index set)

| contrast | Δ pp | CI95 | teams favouring first | verdict |
|---|---:|---|---:|---|
| +7.59M − arm W | **+2.81** | [−2.31, +9.37] | 4 of 8 | WITHIN FLOOR (a ❌) |
| +9.09M − arm W | **+5.13** | [−1.06, +11.31] | 6 of 8 | WITHIN FLOOR (a ✅, b ❌) |
| **+12.09M − arm W** | **+5.75** | **[+1.19, +11.06]** | **7 of 8** | WITHIN FLOOR (a ✅, b ❌) — **CI clear of zero** |
| +7.59M − W_b | −0.87 | [−8.81, +7.69] | 3 of 8 | — |
| +9.09M − W_b | +1.44 | [−6.56, +9.56] | 4 of 8 | — |
| +12.09M − W_b | +2.06 | [−4.81, +9.44] | 4 of 8 | — |
| **arm W − W_b — THE FLOOR, re-measured here** | **−3.69** | [−8.19, +0.81] | 2 of 8 | reproduces exactly |
| **+12.09M − +6.09M — the last 50 % of the budget** | **+0.62** | [−2.75, +3.88] | 5 of 8 | — |
| +12.09M − +3M | +0.37 | [−3.06, +3.75] | 4 of 8 | — |
| +7.59M − +6.09M (crosses the fork) | −2.31 | [−5.31, +1.56] | **1 of 8** | — |
| +12.09M − +7.59M | +2.94 | [−0.44, +6.69] | 5 of 8 | — |

⚠️ **The off-slice row's own maximum is the ENDPOINT (51.94 pp)**, but the +6.09M → +12.09M delta is
**+0.62 pp inside a ±3.3 pp interval**. The honest statement is *flat*, not *rising*.

⚠️ **THE CIs WIDEN ON THE CONTINUATION POINTS** — half-widths ≈ 5.8, 6.2 and 4.9 pp against ≈ 3.8–4.0
on the fold's own points, because the per-team spread grew. Clause (b) was near-unsatisfiable before;
on these points it is further out of reach still, and that is arithmetic, not evidence.

### 2.4 🚨 The clause that binds, registered in advance as a property of the bar

`PREDICTION.md` §2 recorded, before any game, that the untaught contrast's team-clustered CI has a
half-width of ≈ 3.8–4.0 pp against a 3.69 pp floor, so clause (b) can pass only at **|Δ| ≳ 7.5 pp**.
**Every untaught verdict here is WITHIN FLOOR by that rule and the rule stands.** What follows is
what followed at +6M: **a fold effect of ~5 pp off-slice is not resolvable by this instrument at
this n against this floor, and buying more GAMES will not fix it** (rule 19 — the dominant component
is RUN-level; the honest lever is a second fold ARM).

### 2.5 ⚠️ Arm W is still the LOWER of the two parent seeds

Against W_b the same three contrasts are **−0.87 / +1.44 / +2.06 pp**, not +2.81 / +5.13 / +5.75.
The defensible statement is *the fold path ends ahead of BOTH win-prob seeds off-slice, by 2.1 to
5.8 pp, against a 3.7 pp seed floor* — the same shape, and the same caveat, the +6M read attached to
its own numbers.

---

## 3. ROW 2 — PER-SLICE PILOTING: THE MATCHED-EXTRACTION ROW

Every ref pilots the SAME pinned taught team against the SAME fixed opponent
(`untaught_meter_opponent` @24,000,000 — **not** arm W's pool sentinels, because a sentinel is the
trainee's OWN snapshot and arm W's differ from W_b's, floor-read hazard F-G) drawing the SAME
800-long pool-team sequence under the SAME dice. **8,000 battles run here, 0 TIMEOUTS**, plus four
declared imports per team. Team pins `450fb83c20` (Big-5) and `6212de2e8c` (DDTar).

### 3.1 ✅ THE IMPORT WARRANT — arm W reproduces EXACTLY on both cells

| cell | banked (the +6M read) | **here** | reproduces? |
|---|---:|---:|---|
| arm W, Big-5 | 394 / 800 | **394 / 800** | ✅ |
| arm W, DDTar | 376 / 800 | **376 / 800** | ✅ |

Registered in `PREDICTION.md` §3.2 before the first battle: *if arm W does not return 394/800 and
376/800, every import is VOID and this row is INVALID, not patched.* It did. `fold_p3M`,
`fold_p6M`, `W_b`, `t1` and `t2` are therefore taken from
`fold1_read_2026-09-19/out/fold_slice_read.json` — the same games — and are labelled **IMPORTED** in
every table below, saving 6,400 battles that would have returned identical counts.

⚠️ **A DISCLOSED COINCIDENCE.** `fold_p1M`, measured here, returns **374/800** on Big-5 — the exact
count `fold_p3M` returned — and **430/800** on DDTar, the exact count `fold_p6M` returned. Both were
checked against the raw shard file (`out/slice/slice_C.json`, the only shard containing `fold_p1M`)
and are its own games, not an import collision. Two exact ties at 800 games is mildly surprising
(~2.3 % per pair) and is recorded rather than smoothed.

### 3.2 The BIG-5 slice (BALANCE) — `f6229d2c867e21d6`, teacher t1

| ref | win rate | Wilson 95 % | wins/800 | source |
|---|---:|---|---:|---|
| **t1 `ai_v13_05_exploit_big5starmie` — its OWN teacher** | **0.5763** | [0.5417, 0.6100] | 461 | IMPORTED |
| W_b | 0.5400 | [0.5054, 0.5743] | 432 | IMPORTED |
| **arm W — the PARENT** | **0.4925** | [0.4580, 0.5271] | 394 | measured here |
| fold @ +1M | 0.4675 | [0.4332, 0.5021] | 374 | measured here |
| fold @ +3M | 0.4675 | [0.4332, 0.5021] | 374 | IMPORTED |
| **cont @ +12.09M** | **0.4400** | [0.4060, 0.4746] | 352 | measured here |
| fold @ +6.09M | 0.4313 | [0.3973, 0.4658] | 345 | IMPORTED |
| cont @ +9.09M | 0.4225 | [0.3887, 0.4570] | 338 | measured here |
| **cont @ +7.59M** | **0.4213** | [0.3875, 0.4558] | 337 | measured here |
| t2 DDTar (a team it never trained on) | 0.2762 | [0.2464, 0.3082] | 221 | IMPORTED |

| contrast | Δ | Newcombe 95 % | floor | verdict |
|---|---:|---|---:|---|
| **+7.59M − arm W** | **−0.0712** | [−0.1196, −0.0224] | 0.0475 | WITHIN FLOOR (a ✅, b ❌) |
| +9.09M − arm W | −0.0700 | [−0.1184, −0.0212] | 0.0475 | WITHIN FLOOR (a ✅, b ❌) |
| **+12.09M − arm W** | **−0.0525** | [−0.1010, −0.0036] | 0.0475 | WITHIN FLOOR (a ✅, b ❌) |
| **arm W − W_b — THE SEED FLOOR, same cell** | **−0.0475** | [−0.0961, +0.0015] | — | — |
| **t1 − cont @ +12.09M — THE TEACHER'S CEILING** | **+0.1363** | [+0.0874, +0.1842] | — | the teacher stays far above |
| **t1 − arm W — the gap the fold was asked to close** | **+0.0838** | [+0.0349, +0.1321] | — | detected vs zero |
| +12.09M − +6.09M | +0.0087 | [−0.0398, +0.0572] | — | flat |

🚨 **THE BIG-5 SLICE IS NEGATIVE AT ALL SIX DEPTHS AND NEVER RECOVERS.** Twelve million steps of
distillation from a teacher that is **+0.0838 above the parent on this very team** left the student
**0.0525 BELOW** the parent and **0.1363 below the teacher**. The gap did not close; it was crossed
backwards at +6M and only partly retraced.

### 3.3 The DDTAR slice (OFFENSE) — `9eb3abdc52876a63`, teacher t2

| ref | win rate | Wilson 95 % | wins/800 | source |
|---|---:|---|---:|---|
| 🚨 **cont @ +12.09M** | 🚨 **0.6238** | [0.5897, 0.6567] | 499 | measured here |
| cont @ +7.59M | 0.5950 | [0.5606, 0.6285] | 476 | measured here |
| fold @ +3M | 0.5938 | [0.5593, 0.6273] | 475 | IMPORTED |
| cont @ +9.09M | 0.5837 | [0.5493, 0.6174] | 467 | measured here |
| W_b | 0.5550 | [0.5204, 0.5891] | 444 | IMPORTED |
| fold @ +1M | 0.5375 | [0.5029, 0.5718] | 430 | measured here |
| fold @ +6.09M | 0.5375 | [0.5029, 0.5718] | 430 | IMPORTED |
| **t2 `ai_v13_06_exploit_ddtar_spikes` — its OWN teacher** | **0.4775** | [0.4431, 0.5121] | 382 | IMPORTED |
| **arm W — the PARENT** | **0.4700** | [0.4356, 0.5046] | 376 | measured here |
| t1 Big-5 (a team it never trained on) | 0.3725 | [0.3397, 0.4065] | 298 | IMPORTED |

| contrast | Δ | Newcombe 95 % | floor | verdict |
|---|---:|---|---:|---|
| +7.59M − arm W | **+0.1250** | [+0.0762, +0.1730] | 0.0850 | WITHIN FLOOR (a ✅, b ❌) |
| +9.09M − arm W | **+0.1138** | [+0.0649, +0.1619] | 0.0850 | WITHIN FLOOR (a ✅, b ❌) |
| 🚨 **+12.09M − arm W** | 🚨 **+0.1538** | **[+0.1051, +0.2013]** | **0.0850** | 🚨 **OUTSIDE THE FLOOR** |
| **arm W − W_b — THE SEED FLOOR, same cell** | **−0.0850** | [−0.1334, −0.0360] | — | — |
| 🚨 **t2 − cont @ +12.09M — THE TEACHER'S CEILING** | 🚨 **−0.1463** | [−0.1939, −0.0977] | — | **the STUDENT is far above the TEACHER** |
| **t2 − arm W — the gap the fold was asked to close** | **+0.0075** | [−0.0413, +0.0563] | — | **NOT DETECTED — there was almost no gap** |
| **+12.09M − +6.09M** | **+0.0863** | [+0.0379, +0.1340] | — | the only row that moved |

🚨 **THE STUDENT ENDS 0.1463 ABOVE ITS OWN TEACHER ON THE TEACHER'S OWN PINNED TEAM** — and its
teacher was **+0.0075, NOT DETECTED**, over the parent on that team. **So the row that clears the
bar is the row where the teacher had essentially nothing to teach.** That is the sentence this
read's headline has to carry, and it is why 2d is a candidate rather than a verdict about
distillation: what the arm gained on DDTar, its teacher did not have.

### 3.4 🚨 THE SIGN SPLIT SURVIVED AND WIDENED — and it is the one thing seniority cannot explain

| | Big-5 (t1) | DDTar (t2) |
|---|---:|---:|
| the teacher's own edge over the parent on its team | **+0.0838** [+0.0349, +0.1321] | **+0.0075** [−0.0413, +0.0563] |
| the fold's edge over the parent, **+6.09M** | **−0.0612** | **+0.0675** |
| the fold path's edge over the parent, **+12.09M** | **−0.0525** | 🚨 **+0.1538** |
| the fold path vs its own teacher, +12.09M | −0.1363 (far below) | 🚨 **+0.1463 (far above)** |
| **Big-5 − DDTar** | **−0.1287** at +6M → **−0.2063** at +12.09M | |

**The split did not close with convergence; it nearly doubled.** And it is the strongest inference
this design supports:

> 🚨 **A SCALAR SENIORITY TERM CANNOT PRODUCE A SIGN SPLIT.** Both slices were played by the same
> checkpoint, at the same depth, against the same opponent, under the same dice. Whatever the
> +12,091,392 steps of ordinary progress are worth, they are worth *the same thing to both cells* —
> so the **−0.0525 / +0.1538 difference is not a seniority artefact.**

⚠️ **The honest limit of that argument.** It rules out a *scalar* seniority term, not a
*team-differential* continuation effect — a policy drifting for reasons unrelated to teaching could
still help on one archetype and hurt on another. **`ai_v13_09_wcont` measures exactly that**, on
these same two cells, and until it is read the magnitude of each slice's delta remains confounded
even though the split does not.

### 3.5 ⚠️ The gated share still ANTI-predicts, now from the other direction

At the fold's end t1/t2 gated 0.3051 / 0.1222 and the ledger recorded the ratio narrowing to ~1.2×
across the continuation (0.2635 / 0.2135 at +12M). **Big-5 — the slice with the consistently HIGHER
gated share at every point — is the slice the fold path never improved on**; DDTar, gating less,
is the one that cleared its bar. The +6M read logged this as a descriptor on two cells with two
candidate readings and adopted neither; **nothing here changes that**, and the second reading (*the
gate fires where the student is WRONG, not where it is learning*) now has the more natural fit.

### 3.6 The pooled row, SECONDARY — the trap was pre-declared and it did not fire this time

| ref | pooled over BOTH teams | Wilson 95 % |
|---|---:|---|
| W_b | 0.5475 | [0.5230, 0.5717] |
| **cont @ +12.09M** | **0.5319** | [0.5074, 0.5562] |
| fold @ +3M | 0.5306 | [0.5061, 0.5550] |
| cont @ +7.59M | 0.5081 | [0.4836, 0.5326] |
| cont @ +9.09M | 0.5031 | [0.4786, 0.5276] |
| fold @ +1M | 0.5025 | [0.4780, 0.5270] |
| fold @ +6.09M | 0.4844 | [0.4600, 0.5089] |
| **arm W** | **0.4813** | [0.4568, 0.5057] |

Pooled, the endpoint reads **+0.0506** over the parent. ⚠️ **That number is still wrong in kind**:
it averages a **−0.0525** and a **+0.1538**, each with a CI clear of zero, into one figure that
describes neither cell. At +6M the same pooling produced an exact **+0.003** null out of a real
split. **The per-team rows are the result; this table is a footnote** (rule 10).

---

## 4. ROW 3 — THE EXTERNAL ANCHOR: Metamon `SmallRL`, greedy, AWAY team set

Through the tool of record, `python -m main.anchors`, which starts and stops its own Showdown server
(**:9451**, inside this job's 9400–9499 band), verifies the greedy regime **per decision** on both
sides, and writes the Wilson interval itself. `SmallRL` ckpt 40, `VanillaAttention`, Metamon
`@0a00a759`, Showdown pin `e0551883f`, CPU, 100 games as two role-balanced half-cells.

| arm | win rate | Wilson 95 % | half-cells (ours-challenge / peer-challenge) |
|---|---|---|---|
| W_b (banked) | 0.590 | [0.492, 0.681] | 0.54 / 0.64 |
| `ai_v13_07_fold1` @ +6.09M (banked) | 0.540 | [0.443, 0.634] | 0.58 / 0.50 |
| arm S (banked, scale only) | 0.520 | [0.423, 0.615] | — |
| **arm W — the PARENT** (banked) | **0.500** | [0.404, 0.596] | — |
| **`ai_v13_08_fold1_cont` @ +12.09M** | **0.430** | **[0.337, 0.528]** | **0.38 / 0.48** |

| contrast | Δ | Newcombe 95 % | floor | (a) | (b) | verdict |
|---|---|---|---|---|---|---|
| **+12.09M − arm W** | **−0.070** | [−0.204, +0.067] | 0.090 | ❌ | ❌ | **WITHIN FLOOR at n = 2** |
| +12.09M − W_b | −0.160 | [−0.290, −0.022] | 0.090 | ✅ | ❌ | **WITHIN FLOOR at n = 2** |
| +12.09M − fold @ +6.09M | −0.110 | [−0.242, +0.028] | 0.090 | ✅ | ❌ | **WITHIN FLOOR at n = 2** |

**Instrument health, all green.** `status: OK`, 100/100 games, **0 ties, 0 games at the 250-turn
forfeit cap**, `regime_verified: true` with Metamon's `argmax_match_rate` **1.0000** over **2,648**
decisions and our `stochastic` kwarg `[false]`; `distinct_our_teams` 19 of 20; `model_loader: bare`,
`model_rung: explicit_zip`, `model_step` 87,097,344; mean 52.04 turns. Role split **−10 pp**
(0.38 / 0.48); at n = 50 per half nothing may be taken from it.

**This row was registered as DESCRIPTIVE and it behaved as registered** (P(clears) = 0.15; it did
not). ⚠️ The point estimate is the lowest of the five arms measured on this cell and it fell 0.110
from the fold's +6M value — **inside the floor, inside the interval, and on 100 games against a
run-level floor of 0.090 this cell cannot resolve it in either direction** (rule 19: more games are
the wrong lever).

---

## 5. ROW 4 — THE LADDER: 🚨 **n = 0 AGAIN, AT DOUBLE THE BUDGET**

The GO asked for the ladder and named the check: `eval/pool_snapshot_count`.

| | `ai_v13_08_fold1_cont` | `ai_v13_07_fold1` | arm W | W_b |
|---|---|---|---|---|
| `snapshot_ladder/` | **absent** | absent | present | present |
| `eval/pool_snapshot_count`, all cycles | **20.0** | 20.0 | — | — |
| snapshots the run **promoted itself** | **0** | 0 | 20 | 20 |
| pool contents | arm W's 26.0M → 72.0M, re-seeded **by name from the fold** | arm W's, auto-seeded | its own | its own |

**`eval/pool_snapshot_count` reads 20.0 at every one of the six post-fork eval cycles**, and each
cycle's five drawn sentinels are the same arm-W snapshots (72.0M, 54.0M, 44.0M, 36.0M, 26.0M).
A `snapshot_ladder` is a Bradley-Terry fit over a run's OWN promoted snapshots; across **12,091,392
steps the fold path promoted none**, so there is no node to fit and **no rating is quoted in either
direction**. Registered in `PREDICTION.md` §3.4 before any number.

### 5.1 The descriptor — 🚨 and the monotone fall BOTTOMED OUT

`win_rate_vs_pool`, read from the events and **sliced at the original fork step** (⚠️ a fork
inherits its parent's event files, and this path has TWO forks, so the cont's `tb/` carries arm W's
history AND the fold's — the `g7_ladder` trap, ledger `36f8f7eb`):

| step | +steps | `win_rate_vs_pool` | `win_rate_vs_bots` | `eval/mean_ep_len_vs_bots` |
|---:|---:|---:|---:|---:|
| 76,000,032 | +1.0M | 0.488 | 0.9238 | 26.75 |
| 78,000,000 | +3.0M | 0.446 | 0.9288 | 26.86 |
| 80,000,016 | +5.0M | 0.412 | 0.9338 | 27.92 |
| 82,000,032 | +7.0M | **0.372** | 0.9250 | 27.68 |
| 84,000,000 | +9.0M | **0.400** | 0.9300 | 28.73 |
| 86,000,016 | +11.0M | **0.402** | 0.9325 | 27.84 |

🚨 **The +6M read reported this series as falling monotonically 0.488 → 0.372 and it does not
continue to fall.** It bottoms at 0.372 and recovers to 0.400 / 0.402. Every point is against an
**IDENTICAL, FROZEN** opponent set, which is what makes the series clean — and what makes the
reversal worth recording. ⚠️ **This is the seventh short-window direction this campaign would have
called wrong**; the standing rule (*report the number and its window; refuse the trend*) holds
again. No point reaches the 0.55 promotion gate. **DESCRIPTOR, no bar attached**; rule 5's
2026-09-07 regime boundary does not bite (every point post-boundary, regime RECORDED as inherited).

### 5.2 ⚠️ A second descriptor, and it runs the other way

The run's own eval also plays the **two teachers head-to-head** (`externals`, n = 100/cycle,
asymmetric teams, so **not** the piloting row of §3):

| +steps | vs t1 Big-5 | vs t2 DDTar |
|---:|---:|---:|
| +1.0M | 0.19 | 0.27 |
| +3.0M | 0.45 | 0.23 |
| +5.0M | 0.30 | 0.24 |
| +7.0M | 0.34 | 0.27 |
| +9.0M | 0.26 | 0.18 |
| +11.0M | 0.31 | **0.12** |

⚠️ **Do not read a trend here.** The t1 column swings 0.19 → 0.45 in one cycle, which at n = 100 is
inside the instrument. It is printed because it is the only in-run signal about the teachers and
because it **disagrees in kind** with §3.3: the fold path pilots t2's team far better than t2 does,
while losing to t2 more often head-to-head. **Piloting a team and beating its specialist are
different measurements**, and neither is evidence about the other.

---

## 6. ROW 5 — COLLATERAL, and 🚨 why `main.exploitability` WAS NOT RUN

**The untaught meter IS the collateral read** — row 1, the fold path piloting eight teams no teacher
in the fleet has trained on — and no second off-slice instrument was added.

**`main.exploitability` is NOT APPLICABLE and was declared so in `PREDICTION.md` §3.5 before the
first battle.** Its `--help` is unambiguous: *"Generation exploitability curve from
fleet_admission-schema artifacts. Bookkeeping only — no battles, no models."* `ai_v13_08_fold1_cont`
has produced no such artifact; the committed ones are each the output of a separate fleet-admission
battery carrying a fixed cross-era REFERENCE (`rev1final`) that row 2 does not have.

---

## 7. THE READING — what these four rows say together

### 7.1 One paragraph, every hedge in place

> *At its converged endpoint (+12.09M cumulative post-fork, 4.4M steps past the stop rule's fire),
> `ai_v13_08_fold1_cont` is **+5.75 pp on the untaught 8** against its frozen parent (CI clear of
> zero, 7 of 8 teams, WITHIN an imported 3.69 pp floor), **+0.1538 on the DDTar taught team —
> OUTSIDE a 0.0850 floor measured on that very cell, the first registered row in this programme to
> clear its bar** — **−0.0525 on the Big-5 taught team** (CI clear of zero, within its 0.0475
> floor), **−0.070 on the `SmallRL` away anchor** (within 0.090), and has **no ladder at all** at
> double the budget. **No registered branch is met.** The nearest description is that the fold paid
> on ONE of its two taught slices, at the ENDPOINT only, with the trajectory on that slice RISING
> rather than saturated — and that the row which cleared is the row whose teacher had
> **+0.0075, NOT DETECTED**, to teach.*

### 7.2 The three things the earlier read could not have known

1. 🚨 **THE +6M POINT WAS A LOCAL MINIMUM ON THE DDTAR ROW.** "Peaked at +3M and fell by +6M on
   BOTH taught teams" was true of two points; the third reverses it (+7.59M returns to 0.5950) and
   the endpoint is the maximum. The +6M read reported what its points said; the window was short.
2. 🚨 **AGREEMENT AND PILOTING CAME APART, THEN CAME BACK TOGETHER — ON ONE SLICE ONLY.**
   `teacher_agreement_on_slice` kept creeping (0.8200/0.8232 at +6.09M → **0.8360/0.8382** at the
   final step, the two-writer pair) while DDTar rose and Big-5 stayed flat and negative. **The
   apparent decoupling at +3M → +6M was a decoupling on one of the two slices**, and the slice it
   held on is the one where the teacher had nothing to add.
3. 🚨 **THE STOP RULE FIRED AT +7.67M AND THE DDTAR ROW GAINED +0.0288 AFTER IT** (0.5950 →
   0.6238). By its own meter the teaching was done; by the piloting row the largest single gain on
   that slice came afterwards. **Reported as an instrument observation** — the rule's conjunct is
   about AGREEMENT, and agreement is not what these cells measure. Changing the rule is not this
   read's call.

### 7.3 The distill meters, reproduced from the events (post-fork, stitched, both legs)

| row | +6.09M (the fold's end) | **+12.09M (the endpoint)** |
|---|---|---|
| `teacher_agreement_on_slice` | 0.8200 / 0.8232 | **0.8360 / 0.8382** |
| `on_slice_kl` | 0.6621 / 0.6903 | **0.4873 / 0.5075** |
| `collateral_kl_vs_parent` | 0.3855 / 0.4019 | 0.3929 / 0.4029 |
| `kl` (overall) | 0.4835 / 0.4928 | 0.4343 / 0.4483 |
| `t1_gated_frac` (Big-5) | 0.2717 / 0.2749 | 0.2791 / 0.3171 |
| `t2_gated_frac` (DDTar) | 0.1027 / 0.1337 | 0.1044 / 0.1464 |
| `stop_signal` | 0.0000 | **1.0000** |
| `n_teachers_active` | 2.0 | 2.0 |
| `train/learning_rate` | 2.8e-05 | **2.8e-05, flat on every post-fork point** |
| `train/entropy_loss` | −0.7371 / −0.7349 | −0.7201 / −0.7126 |

⚠️ **Each cell is a PAIR, not a number** — a restarted run's TB carries two values at its final step,
one per child event file (the +6M read's hazard D-N). Neither is picked.

🚨 **`on_slice_kl` FELL by a quarter** (0.66–0.69 → 0.49–0.51) while agreement rose: the policy
stopped moving on the taught slice. **`collateral_kl_vs_parent` is flat.** The arm converged in the
sense its own meters describe, and the DDTar piloting row still gained. **These are descriptors;
none carries a bar.**

### 7.4 What this read hands the continuation control

`ai_v13_09_wcont` — arm W resumed +12M with **no teachers** (`grep -c DISTILL` on its launcher log
returns 0), same frozen 2.8e-5 dose, same pool seeding, seed 1001, the same cumulative endpoint
87,005,952 — is live and **was not read**. What it now has to answer is sharper than when it
launched:

* **The untaught row does NOT need it to be read as flat.** +0.62 pp over the last 50 % of the
  budget is inside its own interval whatever the control says.
* **It DOES set the magnitude of both slice deltas**, which carry 12M steps of seniority.
* 🚨 **It CANNOT explain the sign split** (§3.4) unless a plain continuation moves the two cells in
  opposite directions — which is itself the measurement to make: **run the control on these two
  cells, at 800 games, same opponent, same dice.** That is the cheap decisive follow-up, and it is
  four cells (~1.5 h CPU), not an arm.
* ⚠️ **It cannot promote 2d from a CANDIDATE to a family verdict.** That needs a second fold ARM
  (rule 22 — a lever may change the variance it is read against), never more steps and never a
  control replicate.

---

## 8. WHAT WAS NOT DONE, AND WHY

* **`ai_v13_09_wcont` WAS NOT READ, TOUCHED OR OPENED.** It is live on the GPU. No checkpoint,
  event file or log of that run was accessed by this job.
* **The `--config auto` resolution was NOT re-run.** The registry config and each model's own differ
  in `config_version` (101 vs 119) but carry the same `arch_signature`, and `config_path` drives
  only the observation-family CHECK in `load_foreign_opponent` — the weights come from the zip. The
  floor read measured three 75M arms of this config under both resolutions and got levels identical
  to 0.01 pp. **NOT RUN, on budget, and declared.**
* **`fold_p3M`, `fold_p6M`, `W_b`, `t1` and `t2` were NOT re-run on the slice** — declared imports,
  warranted by the arm-W reproduction (§3.1).
* **No continuation control, no second fold seed, no ladder refit, no `main.exploitability`.**
* **No Foul Play cell, no Metamon HOME cell, no critic/calibration frame.** None is a registered row.
* **No code under `src/` was changed. Nothing was written under `models/`.** Other agents' worktrees
  (`leaf_control/`, `viewemit/`) were not touched.

---

## 9. HAZARDS — every one a finding

| # | hazard | why it matters, and what was done |
|---|---|---|
| **C-A** | 🚨 **NO REGISTERED BRANCH IS MET.** (a) needs ≥ 2 of 3 and got 1; (b) needs the +3M peak to stand and it stands on neither slice; (c) needs nothing to clear and something did. The registration's tie-break covered (b)+(c) both holding — the opposite case. | Reported as an uncovered outcome in §0, with the mechanical check of each clause printed, and the nearest description given in the registration's own vocabulary. **The branches are NOT rewritten after the fact** and no branch is declared met. |
| **C-B** | 🚨 **THE ROW THAT CLEARED IS THE ROW WHOSE TEACHER HAD NOTHING TO TEACH.** t2's own edge over the parent on its own pinned team is **+0.0075 [−0.0413, +0.0563], NOT DETECTED**, and the fold path ends **+0.1463 ABOVE t2**. | §3.3. So 2d cannot be read as "distillation transferred the teacher's skill" — the student exceeded the teacher by twice the teacher's own undetectable edge. What the arm gained on that cell, the teacher did not have. Stated beside the verdict, not after it. |
| **C-C** | 🚨 **THE SIGN SPLIT WIDENED (−0.1287 → −0.2063) AND A SCALAR SENIORITY TERM CANNOT PRODUCE IT.** Same checkpoint, same depth, same opponent, same dice on both cells. | §3.4. This is the strongest inference the design supports and it is stated with its limit: it rules out a scalar seniority term, **not** a team-differential continuation effect — which is a measurement `ai_v13_09_wcont` can make on these same two cells for ~1.5 h CPU. |
| **C-D** | 🚨 **THE +6M READ STOPPED ON A LOCAL MINIMUM OF THE DDTAR ROW**, and its "peaked at +3M, fell by +6M on both teams" is reversed by the next point (+7.59M → 0.5950). | §1.1. Recorded as the SIXTH vindicated short-window refusal on this campaign, **not** as an error in the earlier read, which reported exactly what its two points said. `win_rate_vs_pool`'s monotone fall also bottoms and recovers (§5.1) — the seventh. |
| **C-E** | 🚨 **CLAUSE (b) IS NEAR-UNSATISFIABLE ON ROW 1 AND IT GOT WORSE.** Half-widths on the continuation points are ≈ 5.8 / 6.2 / 4.9 pp against a 3.69 pp floor (the fold's own points ran ≈ 3.8–4.0), because the per-team spread grew. | Registered in `PREDICTION.md` §2 **before** the games, with the |Δ| ≳ 7.5 pp arithmetic. The rule STANDS and every untaught verdict is WITHIN FLOOR by it. A CI clear of ZERO is reported separately and is **not** a verdict. The lever is a second fold ARM (rules 19/22), never more games. |
| **C-F** | 🚨 **THE +6.09M → +7.59M LEG CROSSES A FORK BOUNDARY**, and it is the leg where the untaught row dips (49.00, 1 of 8 teams favouring) and Big-5 hits its global minimum. | §1.2. The fork re-seeds the optimizer path, re-pins the LR to the same frozen value, re-seeds the pool by name and opens a new event file; matchup `0a7b730a4d` is unchanged, so it is not a rule-15 regime boundary. **This read cannot say whether the dip is the FORK or the STEPS** and does not try. Marked on every trajectory row. |
| **C-G** | 🚨 **THE LADDER IS n = 0 AGAIN, AT DOUBLE THE BUDGET.** Zero promotions in 12,091,392 steps; `eval/pool_snapshot_count` is 20.0 at all six cycles. | Disclosed in `PREDICTION.md` §0 before any registered number, with the check named by the GO's own tag. Reported as a structural finding with the frozen-pool series as its descriptor (§5). |
| **C-H** | ⚠️ **POOLING STILL MISDESCRIBES THE SLICE ROW**, even though it did not manufacture a null this time: +0.0506 pooled, from −0.0525 and +0.1538. | Pre-declared as a trap in `PREDICTION.md` §2. The per-team rows are the result; the pooled table is a labelled footnote (rule 10, §3.6). |
| **C-I** | ⚠️ **EVERY FLOOR HERE EXCEPT THE TWO PER-SLICE ONES IS AN IMPORT FROM A 75M FRESH-ARM SEED PAIR ONTO A TWICE-FORKED FOLD.** A floor is a property of the DEPTH and the REGIME. | 3.69 pp sits between the meter's frozen-dose (1.66) and controller-live (4.27) fold floors, which is why the GO chose it. Labelled an IMPORT in `PREDICTION.md`, in §0 and in every row it bars. **The two per-slice floors were MEASURED on the same cells, same opponent, same 800 games** — and arm W's half of each was re-measured here and reproduces exactly. |
| **C-J** | ⚠️ **NO CONTINUATION CONTROL; SENIORITY IS NOW TWELVE MILLION STEPS.** Every delta is against a FROZEN parent. | §2.2's "the correction does not bite on OUR parents" is an import from ~28M parent depth and ~1M fold depth; this is 75M and +12M. Printed beside every delta, and the one inference that survives it is named (C-C). `ai_v13_09_wcont` is the arm; it is live and not read. |
| **C-K** | ⚠️ **ARM W IS STILL THE LOWER OF THE TWO PARENT SEEDS** — untaught 46.19 vs W_b's 49.88; Big-5 0.4925 vs 0.5400; DDTar 0.4700 vs 0.5550. | So every "ahead of its parent" number is against the weaker draw. Both contrasts are printed (§2.3, §2.5); against W_b the untaught row is +2.06, not +5.75. **On DDTar the endpoint is +0.0688 over W_b as well**, so 2d does not depend on the choice of seed — stated because it is the row that clears. |
| **C-L** | ⚠️ **A FORK'S TB CARRIES ITS PARENT'S HISTORY AND THIS PATH HAS TWO FORKS** — the cont's events carry arm W's history AND the fold's. | Every series in §5 and §7.3 is sliced at the ORIGINAL fork step (75,005,952) and the two legs are stitched on step, in `scripts/cont_run_rows.py` with the slice in its docstring (the `g7_ladder` trap, `36f8f7eb`). |
| **C-M** | ⚠️ **A RESTARTED RUN CARRIES TWO DISTINCT VALUES AT ITS FINAL STEP.** Every per-rollout distill row in §7.3 is a PAIR. | Neither is picked, in either column, so the +6M → +12.09M comparison is pair-to-pair. |
| **C-N** | ⚠️ **AN EXACT-TIE COINCIDENCE ON THE SLICE ROW.** `fold_p1M` returns 374/800 (= `fold_p3M`'s count) on Big-5 and 430/800 (= `fold_p6M`'s) on DDTar. | Checked against the raw shard file — `out/slice/slice_C.json`, the only shard containing `fold_p1M` — and they are its own games, not an import collision (§3.1). ~2.3 % per pair; recorded rather than smoothed. |
| **C-O** | ⚠️ **THE `externals` ROW AND THE PILOTING ROW DISAGREE IN KIND.** The fold path pilots DDTar's team far better than t2 does (+0.1463) while losing to t2 head-to-head more often as it trains (0.27 → 0.12). | §5.2. Piloting a team and beating its specialist are different measurements; both are printed, neither is evidence about the other, and the n = 100 swings in the t1 column are named as inside the instrument. |
| **(context)** | The box carried the **live GPU arm `ai_v13_09_wcont`** throughout plus up to 14 concurrent CPU battle workers; load average ran **29 to 49**. | Everything ran `CUDA_VISIBLE_DEVICES=""`, `nice 15`, `OMP_NUM_THREADS=1` for the anchor peer, from the MAIN checkout. **The GPU arm was never touched and never read**; no process this read did not start was signalled; :8000 and :8001 were never touched and the only server started was `main.anchors`' own on :9451, which it stopped itself. **20,900 battles, 0 timeouts (0.0 %)** — no row is contention-sensitive by construction (none is a width meter, rule 23), and the meters are deterministic at seed 0 / concurrency 1, which the exact reproduction of **five** refs on their per-team rows demonstrates under exactly this load. |

---

## 10. PROVENANCE

| | **arm W** (PARENT) | **W_b** (floor arm) | **`ai_v13_07_fold1`** | **`ai_v13_08_fold1_cont`** | t1 Big-5 | t2 DDTar |
|---|---|---|---|---|---|---|
| `lineage.role` | `fresh` | `fresh` | `fold` | **`fold`** | fork/exploiter | fork/exploiter |
| forked from | — | — | arm W @75,005,952 | **`ai_v13_07_fold1` @81,100,800** | arm W @75,005,952 | arm W @75,005,952 |
| steps | 75,005,952 | 75,005,952 | 81,100,800 | **87,097,344** (+12,091,392 cumulative) | 83,066,880 | 83,066,880 |
| pin | `6eb9c776` | `6eb9c776` | `6eb9c776` | **`6eb9c776`, ONE row** | `6eb9c776` | `6eb9c776` |
| config / arch | 119 / `gen3_critic_route_wave_v1` | same | same | same | same | same |
| ladder nodes | 20 | 20 | **0** | **0** | — | — |
| realized dose (`main.dose`) | 4.578e-08 | 5.035e-08 | 4.272e-09 | **4.272e-09** `[FROZEN; pinned 2.80e-05]`, **0.20×** | 3.815e-08 | 3.815e-08 |

* `ai_v13_08_fold1_cont`'s `model_config.json` differs from `ai_v13_07_fold1`'s in **no field at
  all**, and from arm W's in **exactly one** — `distill_target`, `"kl"` → `"action"`. Verified here
  by diff.
* `main.lineage` resolves both teachers to `final_model.zip @83,066,880`
  **`[rung=latest_txt rule=last_snapshot]`** — i.e. the run named bare directories and got each
  teacher's last snapshot. Recorded per rule 9: the teachers are quoted here from the RESOLVED file.
* Every slice and untaught ref resolved `[rung=explicit_zip rule=explicit_zip]`, checked in each
  shard log — no last-snapshot surprise on any comparator.

---

## 11. WHAT IS IN THIS DIRECTORY

| path | what |
|---|---|
| `PREDICTION.md` | the pre-registration — bars, branches, priors, the four disclosed pre-look facts, the checkpoint grid, the declared imports and their warrant; committed (`e59e71d0`) before any registered number existed |
| `README.md` | this note |
| `scripts/cont_untaught_delta.py` | row 1 — the paired team-clustered contrasts at six depths off the meter's own per-team rows, with the five reproduction checks. Paired procedure, 20,000 draws and seed VERBATIM from the +6M read |
| `scripts/cont_slice_read.py` | row 2 — the matched-extraction row: Wilson per cell, Newcombe per difference, the import warrant gate, the peak question, the divergence, the pooled footnote |
| `scripts/cont_away_anchor.py` | row 3 — the `main.anchors` away cell joined to arm W, W_b and the fold's +6M point |
| `scripts/cont_run_rows.py` | row 4 — the fold path's TB rows POST-FORK and stitched across both legs, the structural ladder finding, `eval/pool_snapshot_count` read two ways, the frozen-pool descriptor |
| `scripts/render_tables.py` | renders every markdown table above straight from `out/*.json` — no number here is hand-transcribed |
| `scripts/harness/run_untaught.sh` | the untaught invocation as executed — all EIGHT refs in ONE call, 8 workers, seed 0 |
| `scripts/harness/run_slice.sh` | the per-slice invocation as executed — three ref-shards × 2 workers, 800 games/cell |
| `scripts/harness/taught_slice_teams.json` | the TAUGHT-SLICE manifest, in teacher order (the order is the seed offset) |
| `scripts/harness/run_away_cell.sh` | the away cell as executed, on :9451 |
| `out/cont_untaught_delta.json` | every untaught level, per-team row, contrast, verdict and reproduction check |
| `out/cont_slice_read.json` | every per-slice cell, interval, floor, verdict, the peak and the import record |
| `out/cont_away_anchor.json` | the away cell with its regime instruments |
| `out/cont_run_rows.json` | the stitched post-fork TB series, the distill meters and the ladder structural record |
| `out/untaught/`, `out/slice/`, `out/anchors/` | the tools' own raw artifacts, unedited |

---

## 12. Ledger paragraph — ready to append (nothing in `ledger.md`, `UNDERSTANDING.md` or any design note was edited from here)

> ### 2026-09-19 · MEASUREMENT (MAJOR) · **THE ERA-1 FOLD AT CONVERGENCE — 🚨 THE FIRST REGISTERED ROW IN THIS PROGRAMME TO CLEAR ITS BAR (DDTar +0.1538 vs a 0.0850 MEASURED floor), on the slice whose teacher had +0.0075 (NOT DETECTED) to teach — while the OTHER taught slice is NEGATIVE at all six depths and the off-slice row is FLAT across the last 50 % of the budget; NO registered branch is met**
>
> `designs/research_state/measurements/fold1_cont_read_2026-09-19/`. Bars, branches, priors, the
> checkpoint grid and the declared imports pre-registered in `PREDICTION.md` and committed
> (`e59e71d0`, 07:54 PT) BEFORE the first battle (07:58) — including the four facts LOOKED AT while
> scoping and disclosed there (zero promotions, `pool_snapshot_count` 20, the ledger's meter table,
> the stop-state sidecar). **THE ARM.** `ai_v13_08_fold1_cont`, a FORK of `ai_v13_07_fold1` at
> 81,100,800 carrying the **identical distill block** (`model_config.json` differs from the fold's
> in NO field; from arm W's in exactly one, `distill_target` `kl`→`action` — verified by diff), pin
> `6eb9c776`, config v119, `--fork-lr 2.8e-5 --fork-lr-freeze`, dose **4.272e-9 = 0.20× the v8
> reference**, to **87,097,344 = +12,091,392 cumulative post-fork**; its stop rule FIRED at +7.667M.
> **THE RULE, copied verbatim from the +6M read so the two are one series: OUTSIDE THE FLOOR iff
> `|Δ| > floor` AND the Δ's CI excludes the floor POINT; else WITHIN FLOOR at n = 2 — one pair
> BOUNDS a floor (rules 19/22), and WITHIN FLOOR is never "equivalent" (rule 6).** `Δ = fold path −
> arm W`. CPU-only (`CUDA_VISIBLE_DEVICES=""`, `nice 15`), MAIN checkout, **nothing written under
> `models/`**, no `src/` file changed, only server `main.anchors`' own on :9451, **:8000/:8001 never
> touched**, and 🚨 **the live GPU control `ai_v13_09_wcont` never touched and NEVER READ**.
> **20,900 battles, 0 TIMEOUTS.** ✅ **ALL FIVE REGISTERED REPRODUCTION CHECKS PASS EXACTLY, ON THE
> PER-TEAM ROWS** — arm W 46.19 (739/1600), W_b 49.88 (798/1600), fold +1M/+3M/+6M 47.62/51.56/51.31
> — the fourth confirmation that the untaught meter is deterministic at seed 0 / concurrency 1, and
> the first on five refs at once; the 3.69 floor re-measures at exactly −3.69; arm W's two SLICE
> cells reproduce at **394/800** and **376/800**, which is the registered WARRANT for importing
> `fold_p3M`/`fold_p6M`/`W_b`/`t1`/`t2` rather than re-deriving 6,400 identical battles.
> **ROW 1 — UNTAUGHT (and it IS the collateral read).** Eight refs in ONE invocation, 12,800
> battles, so the six-point trajectory is PAIRED on one index set. Levels **47.62 / 51.56 / 51.31 /
> 49.00 / 51.31 / 51.94** at +1/+3/+6.09/+7.59/+9.09/+12.09M ⇒ Δ vs arm W **+1.44 / +5.38 / +5.13 /
> +2.81 [−2.31,+9.37] / +5.13 [−1.06,+11.31] / +5.75 [+1.19,+11.06] (7 of 8 teams)**. **All three
> new points WITHIN FLOOR** — clause (b) fails at every one, and 🚨 **it got HARDER: the
> team-clustered half-widths on the continuation points are ≈5.8/6.2/4.9 pp against the fold's own
> ≈3.8–4.0**, so the registered |Δ| ≳ 7.5 pp requirement is further out of reach; a CI clear of zero
> (+12.09M only) is reported separately and is NOT a verdict. Against W_b: −0.87 / +1.44 / +2.06.
> 🚨 **THE LAST 50 % OF THE BUDGET MOVED THIS ROW BY +0.62 pp [−2.75,+3.88] — FLAT.** **ROW 2 —
> PER-SLICE PILOTING.** 800 games/cell, same pinned teams, same third-party opponent
> (`untaught_meter_opponent`, NOT arm W's sentinels), Wilson per cell, Newcombe per difference
> (CONSERVATIVE under CRN). 🚨 **DDTAR at +12.09M: 0.6238, Δ = +0.1538 [+0.1051, +0.2013] against a
> 0.0850 seed floor MEASURED on that cell — clause (a) ✅ at 1.8×, clause (b) ✅ ⇒ OUTSIDE THE FLOOR.
> The first registered row this programme has cleared.** At +7.59M and +9.09M it was +0.1250 and
> +0.1138 (a ✅, b ❌): **it cleared because it GREW after the stop rule fired.** 🚨 **AND THE STUDENT
> ENDS +0.1463 [+0.0977,+0.1939] ABOVE ITS OWN TEACHER on that teacher's own pinned team — because
> t2's edge over the parent there is +0.0075 [−0.0413,+0.0563], NOT DETECTED.** So 2d is not
> "distillation transferred the teacher's skill"; what the arm gained, the teacher did not have.
> **BIG-5**: 0.4675 / 0.4675 / 0.4313 / 0.4213 / 0.4225 / 0.4400 ⇒ **NEGATIVE at all six depths**
> (−0.0250 / −0.0250 / −0.0612 / −0.0712 / −0.0700 / **−0.0525 [−0.1010,−0.0036]**), all WITHIN its
> 0.0475 measured floor, CIs clear of zero from +6.09M; the teacher stays **+0.1363** above and the
> **+0.0838** gap the fold was asked to close was crossed backwards and only partly retraced.
> 🚨 **THE SIGN SPLIT WIDENED: Big-5 − DDTar = −0.1287 at +6M → −0.2063 at +12.09M**, and 🚨 **a
> SCALAR seniority term cannot produce a sign split** — same checkpoint, same depth, same opponent,
> same dice on both cells — though a TEAM-DIFFERENTIAL continuation effect is not excluded and is
> exactly what `ai_v13_09_wcont` can measure on these two cells for ~1.5 h CPU. The gated share
> still ANTI-predicts (Big-5 gates more at every point and is the slice that never improved).
> Pooled, +0.0506 — averaging −0.0525 and +0.1538 into a figure describing neither cell; footnote
> only (rule 10). **THE REGISTERED PEAK QUESTION: the +3M peak does NOT stand on EITHER slice.**
> Big-5's maximum is its FIRST point (+1M / +3M tie at 0.4675); **DDTar's maximum IS the endpoint**.
> 🚨 **THE +6M READ STOPPED ON A LOCAL MINIMUM OF THE DDTAR ROW** — +7.59M returns to 0.5950, within
> 0.0012 of the +3M value it had "fallen from" — **the SIXTH vindicated short-window refusal on this
> campaign**, and not an error in the earlier read, which reported what its two points said. **ROW 3
> — ANCHOR.** `main.anchors --opponent metamon:SmallRL --regime greedy --teamset away --games 100`
> on :9451 (own server), `SmallRL` ckpt40 / Metamon `@0a00a759` / Showdown `e0551883f`: **0.430
> [0.337,0.528]** ⇒ **−0.070 [−0.204,+0.067] vs arm W ⇒ WITHIN FLOOR**, as registered
> (P(clears)=0.15). `status: OK`, **0 ties, 0 forfeits**, `regime_verified: true`, Metamon
> `argmax_match_rate` **1.0000** over 2,648 decisions, `distinct_our_teams` 19/20, role split −10 pp.
> **ROW 4 — LADDER n = 0 AGAIN, AT DOUBLE THE BUDGET.** No `snapshot_ladder/`; **zero promotions in
> 12,091,392 steps**; `eval/pool_snapshot_count` = **20.0 at all six cycles**, every drawn sentinel
> an arm-W snapshot, pool re-seeded BY NAME from the fold. 🚨 **And the monotone fall the +6M read
> recorded BOTTOMED OUT: `win_rate_vs_pool` 0.488 → 0.446 → 0.412 → 0.372 → 0.400 → 0.402 against
> the IDENTICAL FROZEN pool** — the SEVENTH short-window direction this campaign would have called
> wrong. Bots 0.9238 → 0.9325 over the same span. DESCRIPTORS, no bar. **ROW 5 — COLLATERAL.** The
> untaught meter IS it; `main.exploitability` NOT APPLICABLE (bookkeeping over a
> `fleet_admission` artifact this arm has none of), declared before the first battle. **THE
> READING.** 🚨 **NO REGISTERED BRANCH IS MET:** (a) needs ≥2 of 3 and got 1; (b) needs the +3M peak
> to stand and it stands on neither slice; (c) needs nothing to clear and something did. The
> registration's tie-break covered (b)+(c) both holding — the opposite case — **so the outcome is
> reported as uncovered, the branches are NOT rewritten, and the nearest description in the
> registration's own words is: the fold paid on ONE of two taught slices, at the ENDPOINT only,
> with that slice's trajectory RISING rather than saturated.** 🚨 **AGREEMENT AND PILOTING CAME
> APART AND CAME BACK TOGETHER ON ONE SLICE ONLY:** `teacher_agreement_on_slice` crept 0.8200/0.8232
> → **0.8360/0.8382** while `on_slice_kl` FELL a quarter (0.66–0.69 → 0.49–0.51) and
> `collateral_kl_vs_parent` went flat — the arm converged by its own meters and DDTar still gained
> **+0.0288 AFTER the stop rule fired.** The rule's conjunct is about AGREEMENT; agreement is not
> what these cells measure. Instrument observation; changing the rule is not this read's call.
> **HAZARDS, each a finding.** (1) no registered branch met; (2) the clearing row's teacher had
> nothing to teach; (3) the sign split widened and scalar seniority cannot make one; (4) the +6M
> read stopped on a local minimum (×2, with `win_rate_vs_pool`); (5) clause (b) near-unsatisfiable
> and WORSE on the new points; (6) the +6.09M→+7.59M leg CROSSES A FORK and is where every row dips
> — FORK or STEPS, not separable here; (7) ladder n = 0 again; (8) pooling still misdescribes the
> slice row; (9) every floor but the two per-slice ones is an IMPORT from a fresh-arm pair onto a
> twice-forked fold; (10) NO continuation control, seniority now 12M steps; (11) arm W is still the
> LOWER seed — **though DDTar's endpoint is +0.0688 over W_b too, so 2d does not depend on the seed
> choice**; (12) a fork's TB carries its parent's history and this path has TWO forks — every series
> sliced at 75,005,952 and stitched (`36f8f7eb`); (13) two exact-tie coincidences on the slice row,
> checked against the raw shard and disclosed; (14) the `externals` head-to-head row and the
> piloting row disagree in kind, and neither is evidence about the other. **WHAT IS NOT CLAIMED:**
> that the fold recipe works; that 2d is anything but a CANDIDATE at n = 1 (rules 19/22 — more steps
> on the same arm are NOT a replicate); that extraction is separated from seniority; that the
> off-slice row moved; that the DDTar rise is a TREND; that any branch was met. **WHAT
> `ai_v13_09_wcont` NOW HAS TO ANSWER, sharpened:** it sets the MAGNITUDE of both slice deltas, it
> cannot explain the SIGN SPLIT unless a plain continuation moves the two cells in opposite
> directions — **so run it on these two cells, 800 games, same opponent, same dice: four cells,
> ~1.5 h CPU, and the cheapest decisive follow-up available** — and it cannot promote 2d to a family
> verdict, which needs a second fold ARM. Tag: **MEASURED · THE ERA-1 FOLD AT CONVERGENCE · FIRST
> ROW EVER TO CLEAR — DDTar +0.1538 vs a 0.0850 measured floor, at the ENDPOINT only (1 of 3) ·
> the clearing slice's TEACHER had +0.0075 NOT DETECTED, and the student ends +0.1463 ABOVE it ·
> Big-5 NEGATIVE at all six depths · off-slice FLAT over the last 50 % (+0.62 pp) · the +3M peak
> stands on NEITHER slice · the +6M read stopped on a LOCAL MINIMUM (6th short-window refusal) ·
> `win_rate_vs_pool` bottomed and recovered (7th) · sign split WIDENED to −0.2063 and scalar
> seniority cannot make one · ladder n = 0 at double the budget · NO registered branch met ·
> 20,900 battles, 0 timeouts · five reproduction checks EXACT, per-team**.
