# THE SPLIT ARM — `ai_v13_11_split_lossoff`: which of the era-1 fold's FOUR levers carried its cost? 2026-09-20

**The arm that separates the DISTILLATION LOSS from the ECOLOGY has run.**
`ai_v13_11_split_lossoff` is **fold-1's exact argv with the loss OFF**: a FORK of
`ai_v13_02_flywheel_winprob` (**arm W**) at 75,005,952, built token-exactly from
`ai_v13_07_fold1`'s own `metadata.json original_command` with `--distill-coef 0.1761 → 0.0`, the
same frozen dose `--fork-lr 2.8e-5 --fork-lr-freeze` → **4.272e-9 = 0.20× the v8 reference**, the
same auto-seeded arm-W pool, seed 1001, pin `6eb9c776` in **ONE** row, config v119 /
`gen3_critic_route_wave_v1`, **+12,091,392 steps exactly to 87,097,344 — the control's and the fold
path's own endpoint step, to the step.** It keeps **every** ecology lever: `--distill-team-bias
0.4`, both exploiter specialists in `--stable-opponents`, and `--team-block-episodes 64`.

Every bar below was fixed in [`PREDICTION.md`](PREDICTION.md) and **committed (`afa01250`) before
the first battle**. Instruments, opponent, manifest ORDER (byte-identical file), seed, concurrency,
cell sizes, floors and the decision rule are **REUSED VERBATIM** from
[`fold1_read_2026-09-19/`](../fold1_read_2026-09-19/),
[`fold1_cont_read_2026-09-19/`](../fold1_cont_read_2026-09-19/) and
[`wcont_control_read_2026-09-20/`](../wcont_control_read_2026-09-20/), so the **four reads are ONE
series**. 🚨 **Every number of the control path and the fold path is IMPORTED, not re-derived**,
under the arm-W reproduction warrant registered in advance — and **that warrant HELD, EXACTLY, on
every per-team row of both invocations** (§2.1, §3.1).

**Three sign conventions, all labelled at every use.** `Δ_path = <path> − arm W`;
🚨 `Δ_ecology = split − control` (**THREE levers**: 40 % team bias + two specialists in the stable
pool + `--team-block-episodes 64` vs 1); 🚨 `Δ_loss = split − fold path` (**ONE lever**: the
distillation loss). **The rule, fixed in advance:** OUTSIDE THE FLOOR iff **(a)** `|Δ| > floor`
**AND** **(b)** the Δ's own 95 % CI excludes the floor POINT; otherwise WITHIN FLOOR at n = 2.
🚨 **ONE PAIR BOUNDS A FLOOR; IT DOES NOT ESTIMATE ONE** (rules 19/22); WITHIN FLOOR is never
"equivalent" (rule 6).

**12,800 battles, 0 TIMEOUTS (0.0 %)** — rule 12's 25 % INCONCLUSIVE threshold nowhere near.
Load average ran **16 → 36** throughout, with `ai_v13_12_plateau` block 1 live on the GPU.

---

## 0. THE HEADLINE, in one sentence

> 🚨 **At the matched endpoint step, the arm that got the teachers' TEAMS, the teachers' OPPONENTS
> and 64-episode team blocks — and none of the teachers' ACTIONS — sits on the CONTINUATION
> CONTROL, not on the fold: `Δ_ecology` is WITHIN FLOOR on 9 of 9 cells (3 rows × 3 depths) and is
> exactly `+0.0000` on the Big-5 cell at +12M, while `Δ_loss` is POSITIVE on 9 of 9 and OUTSIDE the
> floor on the Big-5 cell at all three depths (`+0.1600 / +0.1750 / +0.1650`).**
>
> 🚨 **AND THE REGISTERED BRANCH IS STILL `(e) UNCOVERED`, AT BOTH CLAUSE-CARRYING DEPTHS.**
> Branch (a) needs `|Δ_loss|` OUTSIDE on **≥ 2 of 3 rows** and it reaches **1 of 3** — because on
> the untaught row clause (b) is the near-unsatisfiable one this registration named in advance
> (`+6.69 pp` needs `≳ 7.5`), and because on the DDTar cell **the fold path had already matched the
> split**, so there was nothing there for the loss to have cost.

Three things follow, and only three:

1. 🚨 **The ECOLOGY bundle is exonerated on every row this read can see — including the FOURTH
   lever.** `--team-block-episodes 1 → 64`, which `f389fce4` found in fold-1's argv and in no
   registration before it, sits **inside** the bundle whose total effect is WITHIN FLOOR on 9 of 9
   cells. 64-episode blocks against a 40 % two-team bias was a named, untested candidate carrier;
   **at this dose, over 12.09M steps, it does not carry.**
2. 🚨 **The DIRECTION of every row is branch (a), and the CLAUSE COUNT is not.** Reported as
   **UNCOVERED** with every clause printed and no branch rewritten, exactly as `PREDICTION.md` §4
   instructed. The honest one-line reading is in §1.3.
3. ⚠️ **The ecology is not measurably ZERO either.** On the untaught row `Δ_ecology` runs
   **−0.62 → −1.87 → −3.06 pp**, monotone with depth, and the +12M CI **[−5.87, −0.13] EXCLUDES
   ZERO while staying inside the 3.69 floor.** A CI clear of zero is reported separately and **is
   not a verdict** (§2.3). That is a candidate small ecology cost that grows, and it needs a second
   arm, not more games.

---

## 1. THE DELIVERABLE — the 3 × 3 × 3 table, the two contrasts, the branch

### 1.1 THE 3 × 3 × 3 TABLE — path × depth × row

| depth | path | untaught pp | untaught Δ vs arm W | Big-5 rate | Big-5 Δ | DDTar rate | DDTar Δ |
|---|---|---:|---|---:|---|---:|---|
| +3M | **SPLIT `ai_v13_11_split_lossoff`** | 57.88 | **+11.69** [+9.19, +14.06] | 0.6275 | **+0.1350** [+0.0865, +0.1826] | 0.6525 | **+0.1825** [+0.1342, +0.2296] |
| +3M | control `ai_v13_09_wcont` | 58.50 | **+12.31** [+9.69, +15.56] | 0.6062 | **+0.1137** [+0.0650, +0.1617] | 0.6312 | **+0.1613** [+0.1127, +0.2087] |
| +3M | fold path | 51.56 | **+5.38** [+1.63, +9.19] | 0.4675 | **-0.0250** [-0.0737, +0.0239] | 0.5938 | **+0.1238** [+0.0749, +0.1717] |
| +6M | **SPLIT `ai_v13_11_split_lossoff`** | 58.56 | **+12.38** [+10.12, +14.37] | 0.6062 | **+0.1137** [+0.0650, +0.1617] | 0.6538 | **+0.1838** [+0.1355, +0.2308] |
| +6M | control `ai_v13_09_wcont` | 60.44 | **+14.25** [+11.69, +16.63] | 0.5950 | **+0.1025** [+0.0537, +0.1506] | 0.6388 | **+0.1688** [+0.1203, +0.2161] |
| +6M | fold path | 51.31 | **+5.13** [+1.06, +9.00] | 0.4313 | **-0.0612** [-0.1097, -0.0124] | 0.5375 | **+0.0675** [+0.0185, +0.1160] |
| +12M | **SPLIT `ai_v13_11_split_lossoff`** | 58.62 | **+12.44** [+10.69, +14.31] | 0.6050 | **+0.1125** [+0.0638, +0.1605] | 0.6362 | **+0.1663** [+0.1178, +0.2136] |
| +12M | control `ai_v13_09_wcont` | 61.69 | **+15.50** [+12.00, +18.88] | 0.6050 | **+0.1125** [+0.0638, +0.1605] | 0.6475 | **+0.1775** [+0.1291, +0.2247] |
| +12M | fold path | 51.94 | **+5.75** [+1.19, +11.06] | 0.4400 | **-0.0525** [-0.1010, -0.0036] | 0.6238 | **+0.1538** [+0.1051, +0.2013] |
|  +0  | arm W — the FROZEN PARENT | 46.19 | — | 0.4925 | — | 0.4700 | — |
|  +0  | W_b — the seed-floor arm | 49.88 | — | 0.5400 | — | 0.5550 | — |

*floors: untaught **3.69 pp** (IMPORTED) · Big-5 **0.0475** (measured on the cell) · DDTar **0.0850** (measured on the cell)*


#### THE TWO CONTRASTS — Δ_ecology (split − control, THREE levers) and Δ_loss (split − fold, ONE lever)

| depth | row | Δ_ecology | CI95 | floor | (a) | (b) | verdict | Δ_loss | CI95 | (a) | (b) | verdict |
|---|---|---:|---|---:|---|---|---|---:|---|---|---|---|
| +3M | untaught 8 | **-0.62** | [-3.31, +2.06] | 3.69 | ❌ | ✅ | WITHIN | **+6.31** | [+3.06, +9.88] | ✅ | ❌ | WITHIN |
| +3M | Big-5 slice | **+0.0212** | [-0.0263, +0.0687] | 0.0475 | ❌ | ❌ | WITHIN | **+0.1600** | [+0.1114, +0.2075] | ✅ | ✅ | **OUTSIDE** |
| +3M | DDTar slice | **+0.0212** | [-0.0257, +0.0681] | 0.085 | ❌ | ✅ | WITHIN | **+0.0587** | [+0.0113, +0.1059] | ❌ | ❌ | WITHIN |
| +6M | untaught 8 | **-1.87** | [-4.13, +1.00] | 3.69 | ❌ | ❌ | WITHIN | **+7.25** | [+3.19, +11.50] | ✅ | ❌ | WITHIN |
| +6M | Big-5 slice | **+0.0112** | [-0.0367, +0.0591] | 0.0475 | ❌ | ❌ | WITHIN | **+0.1750** | [+0.1263, +0.2225] | ✅ | ✅ | **OUTSIDE** |
| +6M | DDTar slice | **+0.0150** | [-0.0318, +0.0617] | 0.085 | ❌ | ✅ | WITHIN | **+0.1163** | [+0.0682, +0.1635] | ✅ | ❌ | WITHIN |
| +12M | untaught 8 | **-3.06** | [-5.87, -0.13] | 3.69 | ❌ | ❌ | WITHIN | **+6.69** | [+1.06, +11.44] | ✅ | ❌ | WITHIN |
| +12M | Big-5 slice | **+0.0000** | [-0.0478, +0.0478] | 0.0475 | ❌ | ❌ | WITHIN | **+0.1650** | [+0.1163, +0.2126] | ✅ | ✅ | **OUTSIDE** |
| +12M | DDTar slice | **-0.0112** | [-0.0581, +0.0357] | 0.085 | ❌ | ✅ | WITHIN | **+0.0125** | [-0.0348, +0.0597] | ❌ | ✅ | WITHIN |

#### The clause-by-clause record

**+12M** — PRIMARY -- all three paths on the identical step 87,097,344 ⇒ **(e) UNCOVERED**

| branch | clause | met? |
|---|---|---|
| `branch_a_THE_LOSS_IS_THE_CARRIER` | |Δ_ecology| WITHIN FLOOR on ALL THREE rows AND |Δ_loss| OUTSIDE THE FLOOR on >= 2 rows | ❌ |
| `branch_b_THE_ECOLOGY_IS_THE_CARRIER` | |Δ_loss| WITHIN FLOOR on ALL THREE rows AND |Δ_ecology| OUTSIDE THE FLOOR on >= 2 rows | ❌ |
| `branch_c_BOTH_CARRY` | |Δ_ecology| OUTSIDE on >= 2 AND |Δ_loss| OUTSIDE on >= 2, with Δ_ecology NEGATIVE (the split BETWEEN) on >= 2 of those | ❌ |
| `branch_d_THE_ECOLOGY_HELPED` | Δ_ecology POSITIVE and OUTSIDE THE FLOOR on >= 2 rows | ❌ |


*Δ_ecology carries THREE levers; Δ_loss carries ONE. Clause **(a)** = `|Δ| > floor`; clause **(b)**
= the CI excludes the floor POINT. Every cell's raw counts and intervals are in
`out/split_slice_read.json` and `out/split_untaught_delta.json`.*

### 1.2 🚨 THE BRANCHES — **(e) UNCOVERED**, at BOTH clause-carrying depths, which AGREE

**+3M** — ROBUSTNESS -- all three paths on the identical step 78,006,048 ⇒ **(e) UNCOVERED**

| branch | clause | met? |
|---|---|---|
| `branch_a_THE_LOSS_IS_THE_CARRIER` | |Δ_ecology| WITHIN FLOOR on ALL THREE rows AND |Δ_loss| OUTSIDE THE FLOOR on >= 2 rows | ❌ |
| `branch_b_THE_ECOLOGY_IS_THE_CARRIER` | |Δ_loss| WITHIN FLOOR on ALL THREE rows AND |Δ_ecology| OUTSIDE THE FLOOR on >= 2 rows | ❌ |
| `branch_c_BOTH_CARRY` | |Δ_ecology| OUTSIDE on >= 2 AND |Δ_loss| OUTSIDE on >= 2, with Δ_ecology NEGATIVE (the split BETWEEN) on >= 2 of those | ❌ |
| `branch_d_THE_ECOLOGY_HELPED` | Δ_ecology POSITIVE and OUTSIDE THE FLOOR on >= 2 rows | ❌ |

**THE BRANCH: (e) UNCOVERED**

*depths agree: True*


#### The split statistic (Big-5 minus DDTar), previewed here and read in §3.5 — DESCRIPTOR, no bar

| depth | split | control | fold path |
|---|---:|---:|---:|
| +3M | -0.0475 | -0.0476 | -0.1488 |
| +6M | -0.0701 | -0.0663 | -0.1287 |
| +12M | -0.0538 | -0.0650 | -0.2063 |

### 1.3 🚨 THE NEAREST DESCRIPTION IN THE REGISTRATION'S OWN VOCABULARY — and why (e) is honest

`PREDICTION.md` §4's standing instruction is followed literally: **every clause is printed, no
branch is rewritten, and the outcome is named UNCOVERED.** The nearest description the registration
itself licenses:

> **Branch (a) with its ECOLOGY clause met in full and its LOSS clause met on one row of three.**
> The split tracks the control on **every** row at **every** depth — `|Δ_ecology|` WITHIN FLOOR on
> 9 of 9 cells, and the Big-5 endpoint cell is not merely within the floor but **exactly equal**
> (0.6050 vs 0.6050, Δ = +0.0000). `Δ_loss` is **POSITIVE on all 9 cells** and clears the floor by
> both clauses on the Big-5 cell at **all three** depths. What it cannot do is clear on a second
> row, and the two reasons are both structural rather than evidential:

| row | why `Δ_loss` does not clear | pre-registered? |
|---|---|---|
| **untaught 8** | `+6.31 / +7.25 / +6.69 pp` — clause **(a)** passes at every depth (1.7–2.0× the floor) but clause **(b)** fails: the team-clustered CI half-width is ≈3.4–5.6 pp against a 3.69 pp floor, so the rule needs `\|Δ\| ≳ 7.5 pp`. | 🚨 **YES** — `PREDICTION.md` §2 recorded exactly this arithmetic before the first battle, and §4.1's prior put branch (a) at 0.35 partly because of it. |
| **DDTar slice** | `+0.0587 / +0.1163 / +0.0125` — at the ENDPOINT the fold path had **already reached 0.6238** against the split's 0.6362, so the residual is 0.0125 against an 0.0850 floor. **There is nothing on this cell for the loss to have cost.** | ⚠️ NO — the convergence read's `+0.1538` DDTar row was the one row the fold programme had ever cleared, and it is exactly the cell where the loss turns out not to have hurt. |

🚨 **The DDTar row now has its third and final reading.** Against arm W: fold `+0.1538`, split
`+0.1663`, control `+0.1775` — **all three OUTSIDE the 0.0850 floor by both clauses.** It is not
teaching (the control read established that), it is not ecology (`Δ_ecology` = −0.0112, WITHIN),
and it is not the loss (`Δ_loss` = +0.0125, WITHIN). **Every continuation of arm W at this dose
gains ~0.16 on this cell**, and that is the whole of it.

---

## 2. ROW 1 — THE UNTAUGHT METER (off-slice; this IS the collateral read)

`python -m main.untaught_meter`, registry opponent `untaught_meter_opponent`
(= `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` @24,000,000), the untaught-8 manifest
in its canonical order, **200 games/team, `--seed 0`, concurrency 1, 8 single-concurrency shards,
all four MEASURED refs in ONE invocation**. **6,400 battles, 0 TIMEOUTS.**

### 2.1 ✅ THE REGISTERED REPRODUCTION CHECK PASSES EXACTLY, ON THE PER-TEAM ROWS, AGAINST BOTH BANKED SOURCES

| ref | banked | **here** | Δ | per-team rows identical to the CONTROL read's? | to the FOLD reads'? |
|---|---:|---:|---:|---|---|
| **arm W** | 46.19 pp (739 / 1600) | **46.19 pp (739 / 1600)** | **0.00** | ✅ | ✅ |

**This is the sixth independent confirmation that the meter is deterministic at seed 0 /
concurrency 1**, and the first across five reads and five different ref lists. The 3.69 pp floor
re-measures at exactly **−3.69 pp** on the identical games. Every imported cell is therefore
comparable to the split's not by argument but because they are **the same games**
(`PREDICTION.md` §1.2's warrant, registered before the first battle). **This is the ONE re-derived
verification cell the GO named** (§3.5 of the registration), and it is re-derived twice.

### 2.2 The per-team rows at the endpoint — where the split sits, team by team

| team | arm W | **SPLIT** | control | fold path |
|---|---:|---:|---:|---:|
| `U_61590463` | 48.5 | **61.5** | 59.0 | 50.5 |
| `U_92832108` | 39.0 | **47.0** | 56.5 | 41.0 |
| `U_ce35b736` | 49.5 | **62.0** | 65.0 | 53.5 |
| `U_9909f2e9` | 38.0 | **55.5** | 61.5 | 42.5 |
| `U_9d5f8458` | 47.0 | **60.0** | 66.0 | 63.0 |
| `U_f7ba5702` | 48.0 | **62.0** | 67.0 | 50.0 |
| `U_90b94599` | 48.0 | **59.0** | 60.0 | 67.0 |
| `U_dbf81d8e` | 51.5 | **62.0** | 58.5 | 48.0 |

**8 of 8 teams favour the split over arm W** at every depth (`+11.69 / +12.38 / +12.44 pp`, all
OUTSIDE the floor by both clauses). **6 of 8** favour the split over the fold path at +12M; **2 of
8** favour it over the control.

### 2.3 ⚠️ HAZARD S-B — THE ECOLOGY ROW IS SMALL, NEGATIVE, MONOTONE, AND ITS ENDPOINT CI EXCLUDES ZERO

| depth | `Δ_ecology` (split − control) | CI95 | floor | verdict | CI excludes 0? |
|---|---:|---|---:|---|---|
| +3M | **−0.62 pp** | [−3.31, +2.06] | 3.69 | WITHIN FLOOR | no |
| +6M | **−1.87 pp** | [−4.13, +1.00] | 3.69 | WITHIN FLOOR | no |
| +12M | **−3.06 pp** | [−5.87, −0.13] | 3.69 | WITHIN FLOOR | 🚨 **yes** |

🚨 **A CI CLEAR OF ZERO IS NOT A VERDICT** — the registration says so in §2 and the rule is the
floor, not zero. But the shape is worth recording honestly: the ecology bundle costs *something*
off-slice, it is **below the floor at every depth**, and **it grows monotonically**. ⚠️ **Three
points on one arm is a DESCRIPTION, never a trend** (the standing short-window refusal), and the
+12M point is a single arm at n = 1. The honest lever is a second arm (rules 19/22), never more
games. **If the bundle is real at ~3 pp per 12M steps, the instrument that would see it is a
LONGER arm or a replicate, and nothing in this read resolves it.**

### 2.4 The other registered contrasts

| contrast | Δ pp | CI95 | teams favouring first |
|---|---:|---|---:|
| **split +12M − W_b** (the OTHER parent seed) | **+8.75** | [+3.44, +13.44] | 7 of 8 |
| split +6M − W_b | +8.69 | [+3.25, +13.62] | 6 of 8 |
| split +3M − W_b | +8.00 | [+3.19, +12.75] | 6 of 8 |
| arm W − W_b — THE FLOOR, re-measured here | −3.69 | [−8.19, +0.81] | 2 of 8 |
| the split's own +12M − +3M | +0.75 | [−1.81, +3.31] | 5 of 8 |
| the split's own +12M − +6M | +0.06 | [−2.38, +2.75] | 4 of 8 |

⚠️ **Arm W is the LOWER of the two parent seeds and it does not change the picture** (hazard S-E):
against W_b the split is **+8.00 / +8.69 / +8.75 pp**, still 2.2–2.4× the floor with every CI clear
of it.

🚨 **THE SPLIT IS THE MOST FRONT-LOADED ARM OF THE THREE.** `+11.69` of its final `+12.44 pp` is
there by +3M, and its last 50 % of the budget moves it **+0.06 pp** inside a ±2.6 pp interval. The
control moved +1.25 over the same leg and the fold path +0.62. **All three front-load; they
front-load to three different levels** (58.6 / 61.7 / 51.9).

---

## 3. ROW 2 — PER-SLICE PILOTING: THE TEAMS-AND-OPPONENTS-WITHOUT-THE-ACTIONS READ

Every ref pilots the SAME pinned taught team against the SAME fixed opponent
(`untaught_meter_opponent` @24,000,000 — **not** any pool sentinel, because a sentinel is the
trainee's OWN snapshot and the arms' differ, floor-read hazard F-G) drawing the SAME 800-long
pool-team sequence under the SAME dice. **6,400 battles run here, 0 TIMEOUTS**, plus the declared
imports. Team pins `450fb83c20` (Big-5) and `6212de2e8c` (DDTar); the manifest file is
**byte-identical** (md5 `60f735d1…`) to the three earlier reads'.

🚨 **THE SPLIT ARM DID TRAIN ON THESE TWO TEAMS** — 40 % bias, in 64-episode blocks, against two
opponents that pilot them — **with NO teacher loss.** The control never saw them with any bias.
That is exactly what this cell separates.

### 3.1 ✅ THE IMPORT WARRANT — arm W reproduces EXACTLY on BOTH cells

| cell | banked | **here** | reproduces? |
|---|---:|---:|---|
| arm W, Big-5 | 394 / 800 | **394 / 800** | ✅ |
| arm W, DDTar | 376 / 800 | **376 / 800** | ✅ |

Registered before the first battle: *if arm W does not return 394/800 and 376/800, every import is
VOID, this row is INVALID rather than patched, and the job STOPS.* It did not have to.

### 3.2 🚨 THE SPLIT CLEARS BOTH CELLS AT ALL THREE DEPTHS — SIX OF SIX, exactly as the control did

| cell | +3M | +6M | +12M | floor (measured on the cell) |
|---|---|---|---|---:|
| **Big-5 (balance)** | **+0.1350** ✅✅ | **+0.1137** ✅✅ | **+0.1125** ✅✅ | 0.0475 |
| **DDTar (offense)** | **+0.1825** ✅✅ | **+0.1838** ✅✅ | **+0.1663** ✅✅ | 0.0850 |

*(✅✅ = clause (a) and clause (b); every one is OUTSIDE THE FLOOR against arm W.)* **The control
cleared six of six; the split clears six of six; the fold path cleared one of six.**

### 3.3 🚨 THE TEACHERS' CEILING — the split is LEVEL with t1 on t1's own team, having never seen t1's actions

| contrast | Δ | Newcombe 95 % | reading |
|---|---:|---|---|
| **t1 Big-5 − SPLIT @ +12M** | **−0.0287** | [−0.0767, +0.0194] | 🚨 **LEVEL with the teacher**, NOT DETECTED against zero — **the identical number the control read got**, because the two arms land on the identical cell (484/800) |
| **t2 DDTar − SPLIT @ +12M** | **−0.1588** | [−0.2062, −0.1103] | 🚨 the split is FAR ABOVE the teacher |
| t1 Big-5 − arm W | +0.0838 | [+0.0349, +0.1321] | the gap the fold was asked to close |
| t2 DDTar − arm W | +0.0075 | [−0.0413, +0.0563] | **NOT DETECTED — there was almost no gap** |

🚨 **The `+0.0838` gap the fold was asked to close and crossed BACKWARDS, the split closed while
being fed the teacher's TEAM and the teacher's OPPONENT and none of its ACTIONS** — landing on the
exact cell (484/800) the control reached without any of that. **On this cell the ecology buys
nothing and costs nothing; the loss costs 0.165.**

### 3.4 The split's own slice trajectory is FLAT after +3M

| cell | +12M − +6M | +12M − +3M |
|---|---|---|
| Big-5 | −0.0012 [−0.0490, +0.0465] | −0.0225 [−0.0700, +0.0251] |
| DDTar | −0.0175 [−0.0642, +0.0293] | −0.0162 [−0.0630, +0.0306] |

**Everything the split buys on these two teams, it has bought by +3M** — the same shape the control
showed. ⚠️ Three points on one arm is a DESCRIPTION of this arm, never a trend.

### 3.5 THE SPLIT STATISTIC (Big-5 minus DDTar difference-of-deltas) — a DESCRIPTOR, no bar

| depth | **SPLIT** | control | fold path |
|---|---:|---:|---:|
| +3M | **−0.0475** | −0.0476 | −0.1488 |
| +6M | **−0.0701** | −0.0663 | −0.1287 |
| +12M | **−0.0538** | −0.0650 | −0.2063 |

🚨 **The split reproduces the CONTROL's split statistic to within 0.004–0.012 at every depth and
is nowhere near the fold path's.** It is POSITIVE on both cells, so **the fold's sign pattern
(DDTar +, Big-5 −) is absent** — the control read's branch (d) verdict holds for the split too, and
for the same reason: **the sign split belongs to the arm that went DOWN on Big-5, and that arm is
the one with the loss.** ⚠️ A difference of two independent differences on 800 games each; **no
interval is printed and no verdict is taken from it** — registered as a descriptor in advance.

### 3.6 The pooled row, SECONDARY — the trap was pre-declared and it does not fire

Pooled over both taught teams the split reads **0.6206** against arm W's **0.4813**, i.e.
**+0.1394**, against the control's +0.1449. ⚠️ It happens to describe both cells adequately this
time *because the two cells agree in sign* — the condition under which pooling is harmless and
which did NOT hold for the fold path (rule 10). **The per-team rows are the result; this is a
footnote.**

---

## 4. ROWS 3 AND 4 — the descriptors, both banked, neither carrying a bar

### 4.1 ENTROPY — the prior this read stated in advance, and the channel it agrees with

`H_end` at the identical endpoint step (banked in `f389fce4`, **not re-derived here**):

| arm | `H_end` | vs the 75M run-level floor 0.019 |
|---|---:|---|
| **SPLIT `ai_v13_11_split_lossoff`** | **0.9589 ± 0.0155** | — |
| control `ai_v13_09_wcont` | 0.9407 ± 0.0146 | split is **+0.0182** ≈ one sd |
| fold path | 0.7373 ± 0.0185 | split is **+0.2216** ≈ **12×** the floor |

🚨 **The registration stated this as its PRIOR and said in the same breath that entropy is a
descriptor, not this read's endpoint** (§4.1) — because a top-1 action-target distillation
concentrates the policy *by construction*, so the loss moving entropy is nearly a tautology. **The
two channels agreed anyway**, and the licensed conditional is exactly the one registered: *the arm
that lands with the control on entropy lands with it on piloting too, on 9 of 9 cells.* ⚠️ **No
mechanism is claimed. Three arms, one axis, entropy perfectly confounded with the distill block.**

### 4.2 🚨 THE PROMOTION RECORD — the split promoted SIX, the control FIVE, the fold path ZERO

| | **SPLIT** | control `ai_v13_09_wcont` | `ai_v13_07_fold1` / `_08_cont` |
|---|---|---|---|
| `snapshot_ladder/` | **present** | present | absent |
| snapshots the run **promoted itself**, post-fork | 🚨 **6** (76.0M, 78.0M, 80.0M, 82.0M, 84.0M, 86.0M) | **5** | **0** |
| pool generation at the end | **3** | — | seeded only |
| `win_rate_vs_bots` at 86,000,016 | **0.9513** | 0.9175 | — |
| `win_rate_vs_pool` at 86,000,016 | **0.6700** | 0.558 | 0.402 |

**Over the same 12,091,392 steps, against the same auto-seeded arm-W pool at the same 0.55 gate,
the split cleared the gate six times, the plain continuation five, and the fold path never once.**
⚠️ **No Elo is quoted from any of this** — six own nodes is far below the n ≥ 12 report floor and
by the end the three arms' pools are not the same object (hazard **S-G**: the fold path's pool was
FROZEN, so its later cycles face a fixed opponent set while the split's and the control's face a
strengthening one — the asymmetry runs against the two continuations). ⚠️ **The bot row is BLIND at
this depth** (hazard **S-I**, inherited from the control read's W-I): 0.9513 vs 0.9175 vs ~0.93 over
exactly the span in which the untaught meter separates the split from the fold by 6.7 pp and the
Big-5 cell by 0.165.

---

## 5. THE READING

### 5.1 One paragraph, every hedge in place

> *At matched depth, matched endpoint step, matched frozen dose and matched parent, the split arm
> `ai_v13_11_split_lossoff` — fold-1's argv with the distillation loss off and every ecology lever
> kept — is **+12.44 pp [+10.69, +14.31] on the untaught 8** against its frozen parent (8 of 8
> teams, OUTSIDE an imported 3.69 pp floor by both clauses), **+0.1125 [+0.0638, +0.1605] on the
> Big-5 taught team** and **+0.1663 [+0.1178, +0.2136] on DDTar** (both OUTSIDE floors measured on
> those very cells). Against the CONTINUATION CONTROL it is **−3.06 pp [−5.87, −0.13] off-slice,
> +0.0000 [−0.0478, +0.0478] on Big-5 and −0.0112 [−0.0581, +0.0357] on DDTar — WITHIN FLOOR on
> all three, and WITHIN on 9 of 9 cells across all three depths.** Against the FOLD PATH it is
> **+6.69 pp [+1.06, +11.44] off-slice (WITHIN, clause (b)), +0.1650 [+0.1163, +0.2126] on Big-5
> (OUTSIDE) and +0.0125 [−0.0348, +0.0597] on DDTar (WITHIN)**. The registered branch is **(e)
> UNCOVERED at both clause-carrying depths, which agree**: branch (a)'s ecology clause is met in
> full and its loss clause reaches 1 row of 3. **`ai_v13_11_split_lossoff` is one arm at n = 1
> (rules 19/22); this read cannot separate the three levers inside the ecology bundle, and no
> quantity here is a ladder rating.***

### 5.2 What changes in the research state (as a PROPOSAL — nothing was edited)

1. 🚨 **"The ecology" — bias + pool + team-block-64 — has a measured total effect and it is WITHIN
   FLOOR on every row.** The FOURTH lever `f389fce4` named is inside that bundle. The candidate
   carrier it proposed (*"64-episode blocks against a 40 % teacher-team bias is a plausible
   mechanism for the fold path's off-slice loss in its own right"*) is **not supported at this
   dose**, and the bundle's own residual (§2.3) is ≤ 3.06 pp against a 15 pp continuation effect.
2. 🚨 **The remaining candidate carrier is the DISTILLATION LOSS**, and the read's failure to put
   it OUTSIDE on ≥ 2 rows is a statement about the BAR and about the DDTar cell's headroom, not
   about the direction: `Δ_loss` is positive on **9 of 9** cells and clears on Big-5 at all three
   depths. **This is a CANDIDATE, at n = 1, not a family verdict.**
3. 🚨 **The DDTar row is closed as an attribution question** (§1.3): `+0.1538` (fold) / `+0.1663`
   (split) / `+0.1775` (control) — **every continuation of arm W at this dose gains ~0.16 there**,
   and neither the loss nor the ecology moves it.
4. ⚠️ **The control read's re-basing stands and is now sharper.** The fold's celebrated off-slice
   `+5.75 pp` is a third of the continuation's `+15.50` — and the split shows that **about 6.7 pp
   of the 9.75 pp shortfall travels with the LOSS and at most ~3.1 pp with the ecology**, with the
   ecology share inside its floor.

### 5.3 The cheapest decisive follow-ups, in order

* 🚨 **A SECOND SPLIT SEED, or a second CONTROL seed.** Rules 19/22: one arm per cell bounds
  nothing, and the ecology residual of §2.3 is exactly the size a run-to-run floor would settle.
  ~10 GPU-h each.
* 🚨 **THE ECOLOGY BUNDLE IS STILL THREE LEVERS.** If the residual matters, the cell that splits it
  is `--team-block-episodes 64` alone (bias off, pool off) — the one lever that has never appeared
  in any registration and the one this read leaves unidentified.
* **A LOSS-DOSE LADDER.** `--distill-coef` at 0.0 (this arm), 0.1761 (fold-1) and one point
  between, same ecology, same dose — the design that would turn `Δ_loss` from a candidate into a
  curve. Nothing in this series has varied the coefficient with everything else held.

---

## 6. WHAT THIS READ DOES NOT LICENSE

* 🚨 **It does not say "the distillation loss hurts."** It says the split−fold contrast is positive
  on 9 of 9 cells and clears the floor on 1 of 3 rows at n = 1, and the registered branch is
  UNCOVERED. **A candidate, not a verdict.**
* 🚨 **It does not separate the THREE levers inside the ecology bundle** — the 40 % bias, the two
  specialists in the pool, and `--team-block-episodes 64`. The bundle is exonerated as a BUNDLE.
* 🚨 **It does not make anything a family verdict** (rules 19/22). One split arm, one control arm,
  one fold path. Rule 2's warning applies in full.
* **It does not establish any floor.** Every bar is `|arm W − W_b|` at n = 2 — one pair BOUNDS,
  never estimates — and the untaught bar is additionally an IMPORT from a fresh-arm pair.
* **It does not separate the FORK BOUNDARY from the STEPS** (hazard S-F). The split and the control
  are each ONE continuous run; the fold path is two runs with a fork between, so every `Δ_loss` row
  carries that difference too. **Bounded, not resolved:** at +3M no path has crossed a fork and
  `Δ_loss` is already +6.31 pp and +0.1600.
* **It does not attribute anything to ENTROPY** (§4.1).
* **It does not read a TREND** off three points on one arm — including the monotone ecology row.
* 🚨 **It does not convert any pp into ladder STRENGTH.** `068534aa` measured the control's +15.50 pp
  untaught effect at **+0.039 externally, NOT DETECTED at 8,400 anchor battles** — the meter RANKS
  but does not SCALE (hazard **S-J**). No rating is quoted for any arm here, and **anchors were not
  run** (the GO excluded them).
* **It says nothing about folds in general**, at another dose, with other teachers, or at 277M.

---

## 7. HAZARDS — every one a finding

| # | hazard | why it matters, and what was done |
|---|---|---|
| **S-A** | 🚨 **THE REGISTERED BRANCH IS (e) UNCOVERED AT BOTH CLAUSE-CARRYING DEPTHS.** Branch (a)'s ecology clause is met in full (3 of 3 rows, 9 of 9 cells); its loss clause needs ≥ 2 rows OUTSIDE and reaches 1. | Reported as UNCOVERED in §1.2 with every clause mechanically printed and **no branch rewritten**, per `PREDICTION.md` §4. §1.3 names both reasons and marks which was pre-registered (the untaught clause-(b) arithmetic: YES) and which was not (the DDTar cell's headroom: NO). |
| **S-B** | ⚠️ **THE ECOLOGY ROW IS SMALL, NEGATIVE, MONOTONE, AND ITS +12M CI EXCLUDES ZERO** (−0.62 → −1.87 → −3.06 pp, CI [−5.87, −0.13]) while staying inside the 3.69 floor at every depth. | §2.3. A CI clear of zero is reported separately and **is not a verdict** (registered in §2 of the prediction). Recorded as a candidate small ecology cost that a second arm — not more games — would settle. |
| **S-C** | ⚠️ **INHERITED INERT DISTILL FLAGS.** `metadata.json` carries `distill_target: "kl"`, `distill_topk: 1`, `distill_beta: 1.0`, `distill_gate: "none"`, re-resolved from arm W's checkpoint config. | **INERT at `--distill-coef 0.0`**: `grep -c DISTILL` on both launcher logs is **0** and the run's TensorBoard contains **ZERO `distill/*` scalars**, so the term was ABSENT, not zero-weighted. Disclosed in `PREDICTION.md` §0.4 before any number. The same standing rule ("any flag the argv does not NAME re-resolves") that produced the control read's W-C. |
| **S-D** | ⚠️ **THE +6M DEPTHS DIFFER**, the split being 260,928 steps deeper than the control and 303,408 deeper than the fold. | 2.2 % of the post-fork budget, and it flatters the SPLIT on that row only. 🚨 **+3M and +12M are the SAME STEP NUMBER on all three paths** (78,006,048 and 87,097,344), **both carry clauses, and both reach the same branch** — no conclusion rests on the offset row. Named in `PREDICTION.md` §1.1 before the grid was run; the nearest-checkpoint convention was fixed there too. |
| **S-E** | ⚠️ **ARM W IS THE LOWER OF THE TWO PARENT SEEDS** (46.19 vs W_b's 49.88). | Both contrasts printed (§2.4). Against W_b the split is **+8.00 / +8.69 / +8.75 pp**, still 2.2–2.4× the floor with CIs clear of it. The result does not depend on the seed choice. |
| **S-F** | 🚨 **THE FORK-BOUNDARY ASYMMETRY.** The split and the control each ran ONE continuous +12M; the fold path is +6.09M then +5.99M with a FORK between, which re-seeds the optimizer path, re-pins the LR, re-seeds the pool by name and opens a new event file. | **Bounded, not resolved.** At **+3M neither the split nor the fold has crossed a fork** and `Δ_loss` is already **+6.31 pp and +0.1600** there, so the crossing cannot carry the bulk of the loss row. Stated with the part it can and cannot explain separated. |
| **S-G** | ⚠️ **THE THREE `win_rate_vs_pool` SERIES ARE NOT COMPARABLE IN KIND.** The fold path's pool was frozen (zero promotions); the split's and the control's GREW. | §4.2. The two continuations' later cycles face a strengthening opponent set and still read 0.670 / 0.558 against a 0.55 gate. DESCRIPTORS on all sides, no bar attached, **no Elo quoted**. |
| **S-H** | ⚠️ **CLAUSE (b), REGISTERED AS NEAR-UNSATISFIABLE ON ROW 1, BOUND EXACTLY WHERE THE REGISTRATION SAID IT WOULD.** `Δ_loss` is +6.31 / +7.25 / +6.69 pp against a bar that needs ≳ 7.5. | §1.3, §2.3. Recorded because the registration predicted it in writing before the first battle. The earlier WITHIN-FLOOR verdicts on 5-pp effects and this one on a 6.7-pp effect are the same arithmetic; the control read's 12–15 pp effects passed it without difficulty. |
| **S-I** | 🚨 **THE BOT ROW AND G7 ARE BLIND TO A DIFFERENCE THIS SIZE.** `f389fce4` banked bots 0.9513 and G7 clean for the split, indistinguishable from the other two paths. | §4.2. Over exactly the span in which the untaught meter separates the split from the fold by 6.7 pp and the Big-5 cell by 0.165. Inherited from the control read's W-I; **a bot row saturated at ~0.92–0.95 is not a competence meter at this depth**. |
| **S-J** | 🚨 **THE UNTAUGHT METER RANKS BUT DOES NOT SCALE.** `068534aa` (2026-09-20, 8,400 anchor battles) put the control's +15.50 pp at **+0.039 [−0.000, +0.079]** externally, NOT DETECTED. | Declared in `PREDICTION.md` §0.7 before the first battle. **Every pp in this file is a RANK on this instrument**, and no strength claim is made from any of them. **Anchors were not run here** — the GO excluded them. |
| **S-K** | 🚨 **ONE SPLIT ARM.** No seed replicate; rules 19/22 and rule 2. | Every verdict here is a **CANDIDATE**. §5.3 names the replicate as the first follow-up. |
| **S-L** | ⚠️ **THE ECOLOGY BUNDLE IS STILL THREE UNIDENTIFIED LEVERS**, one of which (`--team-block-episodes 64`) has never appeared in any registration before `f389fce4`. | §6, §5.3. The read exonerates the BUNDLE, not its parts; a bundle that is within floor as a whole may still contain a lever that helps and one that hurts. |
| **S-M** | ⚠️ **EVERY FLOOR EXCEPT THE TWO PER-SLICE ONES IS AN IMPORT FROM A 75M FRESH-ARM SEED PAIR ONTO A FORKED CONTINUATION.** A floor is a property of the DEPTH and the REGIME (rule 3). | Labelled an IMPORT in `PREDICTION.md` §1.3 and in every row it bars. The nearest floor in KIND is the gen-era G5 three-continuation-arm floor of **1.00 pp**, at a different parent and era. The two per-slice floors were MEASURED on the same cells and arm W's half of each re-measured here, reproducing exactly. |
| **(context)** | The box carried `ai_v13_12_plateau` block 1 on the GPU throughout, plus this job's 8 concurrent battle workers; **load average ran 16 → 36**. | Everything ran `CUDA_VISIBLE_DEVICES=""`, `nice 15`, `OMP_NUM_THREADS=1`, from the MAIN checkout, with the two invocations SEQUENTIAL to hold the 8-worker cap. **The GPU was never touched and never read.** **No server was started at all** — the meter is serverless and no anchor cell was run; **:8000 and :8001 were never touched**. Nothing was written under `models/`; no file under `src/` was changed. **12,800 battles, 0 timeouts (0.0 %)** — and the meter's determinism at seed 0 / concurrency 1 is demonstrated *under exactly this load* by arm W's exact per-team reproduction. |

---

## 8. PROVENANCE

| | **arm W** (PARENT) | **W_b** | **`ai_v13_11_split_lossoff`** (SPLIT) | `ai_v13_09_wcont` (CONTROL) | `ai_v13_07_fold1` | `ai_v13_08_fold1_cont` |
|---|---|---|---|---|---|---|
| `lineage.role` | `fresh` | `fresh` | **`fold`** | `fork` | `fold` | `fold` |
| forked from | — | — | **arm W @75,005,952** `[explicit_zip]` | arm W @75,005,952 | arm W @75,005,952 | `_07_fold1` @81,100,800 |
| teachers | — | — | **2, both resolved `final_model.zip` @83,066,880 `[latest_txt / last_snapshot]`** | **`[]` — none** | 2 | 2 |
| `--distill-coef` | — | — | 🚨 **0.0** | 0.0 | **0.1761** | 0.1761 |
| `--distill-team-bias` | — | — | **0.4** | absent | 0.4 | 0.4 |
| `--stable-opponents` | — | — | **both specialists** | absent | both | both |
| `--team-block-episodes` | **1** | 1 | 🚨 **64** | **1** | **64** | 64 |
| steps | 75,005,952 | 75,005,952 | **87,097,344** (+12,091,392) | 87,097,344 | 81,100,800 | 87,097,344 |
| pin | `6eb9c776` | `6eb9c776` | **`6eb9c776`, ONE row** | `6eb9c776`, ONE row | `6eb9c776` | `6eb9c776` |
| config / arch | 119 / `gen3_critic_route_wave_v1` | same | same | same | same | same |
| realized dose | 4.578e-08 | 5.035e-08 | **4.272e-09** `[FROZEN; pinned 2.80e-05]` | 4.272e-09 | 4.272e-09 | 4.272e-09 |
| matchup hash | `ef5242cffd` | — | **`0a7b730a4d` — fold-1's era hash** | `ef5242cffd` | `0a7b730a4d` | `0a7b730a4d` |
| post-fork promotions | — | — | 🚨 **6** | **5** | **0** | **0** |
| forks in the path | — | — | **0 (one continuous run)** | 0 | 1 | **2** |

* **The three-way argv token diff, computed from the three `original_command` strings:** `split` vs
  `fold-1` = `--distill-coef 0.1761 → 0.0`, `--steps`, `--run-name`, and `--distill-target action`
  DROPPED (**FORCED**, not a second lever: `combination_checks.distill_target_needs_coef` makes
  `action` + coef 0 a fatal refusal, and the drop re-resolves to the INHERITED inert `kl`).
  `split` vs `control` = `--distill-team-bias 0.4`, `--stable-opponents <both>`,
  `--team-block-episodes 64 vs 1` (plus the inert `--distill-teacher`/`--distill-topk`/
  `--distill-gate`/`--rank-tripwire` tokens), `--steps`, `--run-name`.
* Every ref in both invocations resolved `[rung=explicit_zip rule=explicit_zip]` — no last-snapshot
  surprise on any measured comparator (rule 9). The two TEACHERS resolve by `last_snapshot` to
  `final_model.zip` @83,066,880, the same files fold-1 used, and are IMPORTED rather than re-run.
* The split's `pin_history` is a **single row**, 75,005,952 → 87,097,344, `pin_source pin_commit`.
* `pool_seeded_from` = arm W's `snapshots/`, 20 files; by the end the pool is generation 3 and holds
  the split's own six promotions.

---

## 9. WHAT IS IN THIS DIRECTORY

| path | what |
|---|---|
| `PREDICTION.md` | the pre-registration — bars, the five branches with the (c)/(d) tie-break and the uncovered case registered in advance, priors, the checkpoint grid and its +6M convention, the declared imports, the ONE re-derived verification cell, and the seven disclosed pre-look facts; committed (`afa01250`) before the first battle |
| `README.md` | this note |
| `scripts/split_untaught_delta.py` | row 1 — the paired team-clustered contrasts at three matched depths for ALL THREE paths, `Δ_ecology` and `Δ_loss`, and the reproduction check that gates every import |
| `scripts/split_slice_read.py` | row 2 — Wilson per cell, Newcombe per difference, the import warrant gate, the split statistic, the teachers' ceiling, the pooled footnote |
| `scripts/split_branches.py` | the branch decision — joins both rows and evaluates the five registered clauses mechanically at +12M (primary) and +3M (robustness), with the SPLIT-BY-DEPTH guard |
| `scripts/render_tables.py` | renders every markdown table above straight from `out/*.json` — no number here is hand-transcribed |
| `scripts/harness/run_untaught.sh` | the untaught invocation as executed — four refs in ONE call, 8 workers, seed 0 |
| `scripts/harness/run_slice.sh` | the per-slice invocation as executed — four ref-shards × 2 workers, 800 games/cell |
| `scripts/harness/taught_slice_teams.json` | the TAUGHT-SLICE manifest in teacher order, **byte-identical** (md5 `60f735d1…`) to the three earlier reads' |
| `out/split_untaught_delta.json` | every untaught level, per-team row, contrast, both contrast rows, verdict and the reproduction check |
| `out/split_slice_read.json` | every per-slice cell, interval, floor, verdict, the split statistic and the import record |
| `out/split_branches.json` | the mechanical branch evaluation at all three depths |
| `out/untaught/`, `out/slice/` | the tool's own raw artifacts, unedited |

---

## 10. Ready-to-append ledger paragraph (nothing in `ledger.md`, `UNDERSTANDING.md` or any design note was edited from here)

> ### 2026-09-20 · MEASUREMENT (MAJOR) · **THE SPLIT ARM — 🚨 THE ECOLOGY BUNDLE IS EXONERATED: fold-1's argv with the LOSS OFF sits ON THE CONTINUATION CONTROL on 9 of 9 cells (3 rows × 3 depths), exactly +0.0000 on the Big-5 endpoint cell, and +6.69 pp / +0.1650 ABOVE the fold path — yet the registered branch is (e) UNCOVERED at BOTH clause-carrying depths, because Δ_loss clears the floor on ONE row of three**
>
> `designs/research_state/measurements/split_lossoff_read_2026-09-20/`. Bars, the five branches
> (a)–(e) with the (c)/(d) tie-break and the uncovered case registered **IN ADVANCE**, priors, the
> checkpoint grid, the declared imports and the ONE re-derived verification cell pre-registered in
> `PREDICTION.md` and committed (`afa01250`) BEFORE the first battle — including the seven facts
> LOOKED AT while scoping and disclosed there (the grid, the metadata, the **six post-fork
> promotions**, the inherited inert distill flags, the three-way argv token diff, the ledger's
> banked descriptors, and `068534aa`'s anchor limit). **THE ARM.** `ai_v13_11_split_lossoff`, a
> FORK of arm W `ai_v13_02_flywheel_winprob` at 75,005,952, built token-exactly from
> `ai_v13_07_fold1`'s own `original_command` with `--distill-coef 0.1761 → 0.0` (`--distill-target
> action` DROPPED — **FORCED** by `combination_checks.distill_target_needs_coef`, re-resolving to
> the inherited inert `kl`), **every ecology lever kept** (`--distill-team-bias 0.4`, both
> specialists in `--stable-opponents`, `--team-block-episodes 64`), the SAME frozen dose 4.272e-9 =
> 0.20× the v8 reference, the same auto-seeded pool, seed 1001, pin `6eb9c776` in ONE row, config
> v119, **one continuous +12,091,392 steps to 87,097,344 — the control's and the fold path's own
> endpoint step, to the step**; matchup `0a7b730a4d`, fold-1's era hash. **THE RULE, copied verbatim
> from all three predecessor reads so the FOUR are ONE series: OUTSIDE THE FLOOR iff `|Δ| > floor`
> AND the Δ's CI excludes the floor POINT; else WITHIN FLOOR at n = 2 — one pair BOUNDS a floor
> (rules 19/22), WITHIN FLOOR is never "equivalent" (rule 6).** Three conventions: `Δ_path = path −
> arm W`, 🚨 `Δ_ecology = split − control` (**THREE levers**: bias + pool + team-block-64),
> 🚨 `Δ_loss = split − fold path` (**ONE lever**: the distillation loss). CPU-only
> (`CUDA_VISIBLE_DEVICES=""`, `nice 15`, `OMP_NUM_THREADS=1`, ≤ 8 workers, the two invocations
> SEQUENTIAL), MAIN checkout, **nothing written under `models/`**, no `src/` file changed, **no
> server started at all**, **:8000/:8001 never touched, the GPU never touched** (`ai_v13_12_plateau`
> block 1 was live on it; load 16 → 36). **12,800 battles, 0 TIMEOUTS.** ✅ **THE REGISTERED
> REPRODUCTION CHECK — the ONE re-derived verification cell — PASSES EXACTLY ON THE PER-TEAM ROWS
> IN BOTH INVOCATIONS**: arm W 46.19 pp (739/1600) on the untaught 8 with all eight rows identical
> to BOTH banked sources, and **394/800 + 376/800** on the two slice cells — the **sixth**
> confirmation that the meter is deterministic at seed 0 / concurrency 1 and the WARRANT for
> importing the whole control path, the whole fold path, W_b and both teachers rather than
> re-deriving ~19,200 identical battles; the 3.69 floor re-measures at exactly −3.69. **ROW 1 —
> UNTAUGHT.** Split levels **57.88 / 58.56 / 58.62 pp** at +3M/+6M/+12M ⇒ Δ vs arm W **+11.69
> [+9.19,+14.06] / +12.38 [+10.12,+14.37] / +12.44 [+10.69,+14.31], 8 of 8 teams at EVERY depth,
> OUTSIDE by BOTH clauses** — against the control's +12.31 / +14.25 / +15.50 and the fold path's
> +5.38 / +5.13 / +5.75. Against W_b the split is +8.00 / +8.69 / +8.75, so it does not depend on
> arm W being the lower seed. 🚨 **Δ_ecology = −0.62 [−3.31,+2.06] / −1.87 [−4.13,+1.00] / −3.06
> [−5.87,−0.13] pp — WITHIN FLOOR at all three, but SMALL, NEGATIVE, MONOTONE, and the +12M CI
> EXCLUDES ZERO** (a CI clear of zero is NOT a verdict — the honest lever is a second arm).
> 🚨 **Δ_loss = +6.31 [+3.06,+9.88] / +7.25 [+3.19,+11.50] / +6.69 [+1.06,+11.44] pp — clause (a)
> passes at every depth, clause (b) fails at every depth**, exactly the near-unsatisfiable
> arithmetic the registration wrote down in advance (the bar needs ≳ 7.5 pp). The split is the most
> FRONT-LOADED of the three arms: +11.69 of its +12.44 is there by +3M and its last 50 % moves it
> +0.06 pp. **ROW 2 — PER-SLICE PILOTING, 800 games/cell, same pinned teams (`450fb83c20` /
> `6212de2e8c`), same third-party opponent, manifest byte-identical.** 🚨 **THE SPLIT CLEARS BOTH
> CELLS AT ALL THREE DEPTHS — SIX OF SIX against arm W, exactly as the control did (the fold path
> cleared one of six): Big-5 +0.1350 / +0.1137 / +0.1125 against a 0.0475 MEASURED floor; DDTar
> +0.1825 / +0.1838 / +0.1663 against 0.0850.** 🚨 **Δ_ecology on the slices: +0.0212 / +0.0112 /
> +0.0000 on Big-5 — the endpoint cell is 484/800 for BOTH arms, an EXACT tie — and +0.0212 /
> +0.0150 / −0.0112 on DDTar; all six WITHIN FLOOR.** 🚨 **Δ_loss on the slices: +0.1600 / +0.1750 /
> +0.1650 on Big-5, OUTSIDE by both clauses at ALL THREE depths; +0.0587 / +0.1163 / +0.0125 on
> DDTar, all WITHIN.** 🚨 **THE TEACHERS' CEILING: t1 Big-5 − split = −0.0287 [−0.0767,+0.0194], NOT
> DETECTED — the split is LEVEL with the teacher on the teacher's own pinned team, landing on the
> IDENTICAL cell the control reached, while having been fed the teacher's TEAM and the teacher's
> OPPONENT and none of its ACTIONS; t2 DDTar − split = −0.1588, the split far above.** 🚨 **AND THE
> DDTar ROW IS CLOSED AS AN ATTRIBUTION QUESTION: fold +0.1538, split +0.1663, control +0.1775, all
> three OUTSIDE the floor against arm W — every continuation of arm W at this dose gains ~0.16
> there, and neither the loss nor the ecology moves it.** Split statistic (Big-5 − DDTar): split
> −0.0538 ≈ control −0.0650, both far from the fold path's −0.2063, and the fold's sign pattern is
> ABSENT — the sign split belongs to the arm that went DOWN on Big-5. Pooled +0.1394; footnote only
> (rule 10). **ROWS 3–4 — DESCRIPTORS, BANKED, NOT RE-DERIVED.** `H_end` split **0.9589** lands WITH
> the control 0.9407 and 0.2216 ABOVE the fold 0.7373 (≈12× the 75M run-level floor) — registered as
> this read's PRIOR and explicitly not its endpoint; **the two channels agreed**, no mechanism
> claimed. 🚨 **PROMOTIONS: the split promoted SIX post-fork snapshots (76/78/80/82/84/86M) and has
> a `snapshot_ladder/`; the control FIVE; the fold path ZERO** — `win_rate_vs_pool` 0.670 vs 0.558
> vs 0.402 at 86,000,016, with the two continuations facing a GROWING pool and the fold path a
> frozen one. No Elo quoted (n = 6 ≪ 12). Bots 0.9513 — **BLIND at this depth**. **THE BRANCH.**
> 🚨 **(e) UNCOVERED at +12M AND at +3M, and the two depths AGREE; the branches are NOT rewritten.**
> (a) needs |Δ_ecology| WITHIN on all three rows — **met, 3 of 3, and 9 of 9 across depths** — AND
> |Δ_loss| OUTSIDE on ≥ 2 rows, **which reaches 1 of 3**; (b) needs |Δ_loss| WITHIN on all three
> (it is 2 of 3) plus ecology OUTSIDE on ≥ 2 (it is 0); (c) and (d) need ecology OUTSIDE on ≥ 2 (0).
> **Nearest description in the registration's own vocabulary: branch (a) with its ECOLOGY clause met
> in full and its LOSS clause met on one row of three** — the untaught row blocked by the
> pre-registered clause-(b) arithmetic, and the DDTar cell blocked because **the fold path had
> already matched the split there, so there was nothing for the loss to have cost.** **HAZARDS, each
> a finding.** (S-A) the branch is UNCOVERED and why, with which reason was pre-registered; (S-B)
> 🚨 the ecology row is small, negative, MONOTONE and its +12M CI excludes ZERO inside the floor;
> (S-C) inherited INERT distill flags (`distill_target kl`, `distill_topk 1`) — `grep -c DISTILL` =
> 0 and **ZERO `distill/*` scalars in TB**, so the term was ABSENT not zero-weighted; (S-D) the +6M
> depths differ, the SPLIT deeper by 260,928 / 303,408 steps — **+3M and +12M are the SAME STEP
> NUMBER on all three paths, both carry clauses, and both reach the same branch**; (S-E) arm W is
> the lower seed and it does not matter; (S-F) the fork-boundary asymmetry — **BOUNDED: at +3M
> neither the split nor the fold has crossed a fork and Δ_loss is ALREADY +6.31 pp and +0.1600**;
> (S-G) the three pool series are not comparable in kind; (S-H) clause (b) bound exactly where the
> registration said it would; (S-I) the bot row and G7 are BLIND; (S-J) 🚨 **the untaught meter
> RANKS but does not SCALE — `068534aa` put the control's +15.50 pp at +0.039 externally, NOT
> DETECTED over 8,400 anchor battles; no strength claim is made and anchors were not run**; (S-K)
> ONE split arm; (S-L) 🚨 **the ecology bundle is still THREE unidentified levers, one of which
> (`--team-block-episodes 64`) has never appeared in any registration before `f389fce4`**; (S-M)
> every floor but the two per-slice ones is an IMPORT. **WHAT IS NOT CLAIMED:** that "the
> distillation loss hurts" (the contrast is positive on 9 of 9 cells and clears on 1 of 3 rows at
> n = 1 — a CANDIDATE); that the three ecology levers are individually exonerated (**the BUNDLE
> is**); that anything is a family verdict; that any floor is established; that entropy explains
> anything; that three points are a trend; anything about ladder STRENGTH. **WHAT FOLLOWS.** (1)
> 🚨 **"The ecology" — bias + pool + team-block-64, including the FOURTH lever `f389fce4` named — has
> a measured total effect and it is WITHIN FLOOR on every row; the candidate carrier that entry
> proposed is NOT supported at this dose.** (2) 🚨 **The remaining candidate carrier is the
> DISTILLATION LOSS**, and about 6.7 pp of the fold's 9.75 pp off-slice shortfall travels with it
> against at most ~3.1 pp with the ecology. (3) The cheapest decisive follow-ups, in order: **a
> second SPLIT or CONTROL seed** (~10 GPU-h; the only thing that settles S-B), then
> **`--team-block-episodes 64` alone** (the one lever no registration has ever named), then **a
> `--distill-coef` DOSE LADDER** at fixed ecology — nothing in this series has varied the
> coefficient with everything else held. Tag: **MEASURED (MAJOR) · THE SPLIT ARM · the ECOLOGY
> BUNDLE (bias + pool + team-block-64) is WITHIN FLOOR on 9 of 9 cells, +0.0000 on the Big-5
> endpoint · the split is +6.69 pp / +0.1650 ABOVE the fold path · branch (e) UNCOVERED at BOTH
> depths (a's ecology clause met in full, its loss clause 1 row of 3) · the DDTar row is CLOSED —
> every continuation gains ~0.16 there · the split is LEVEL with teacher t1 on t1's own team ·
> promotions 6 / 5 / 0 · bots and G7 BLIND · 12,800 battles, 0 timeouts · reproduction EXACT,
> per-team, in BOTH invocations**.
