# THE CONTINUATION CONTROL — the era's G5 cell, `ai_v13_09_wcont`, 2026-09-20

**The arm that separates SENIORITY from EXTRACTION has run, and it answers the question the two
fold reads could not.** `ai_v13_09_wcont` is a FORK of `ai_v13_02_flywheel_winprob` (**arm W**, the
75M win-prob flywheel arm) at 75,005,952 **with NO TEACHERS** (`--distill-coef 0.0`;
`grep -c DISTILL` on its launcher log is **0** across all 9 h 43 m), the **same frozen dose**
`--fork-lr 2.8e-5 --fork-lr-freeze` → realized **4.272e-9 = 0.20× the v8 reference**, the same pool
auto-seeded from arm W, seed 1001, pin `6eb9c776` in **ONE** row, config v119 /
`gen3_critic_route_wave_v1`, **+12,000,000 steps exactly to 87,097,344 — the fold path's own
endpoint step, to the step.**

Every bar below was fixed in [`PREDICTION.md`](PREDICTION.md) and **committed (`e8ea5ba3`) before
the first battle**. Instruments, opponent, manifest ORDER, seed, concurrency, cell sizes and the
decision rule are **REUSED VERBATIM** from [`fold1_read_2026-09-19/`](../fold1_read_2026-09-19/)
and [`fold1_cont_read_2026-09-19/`](../fold1_cont_read_2026-09-19/), so the three reads are ONE
series. 🚨 **Every number of the fold path is IMPORTED, not re-derived**, under the arm-W
reproduction warrant registered in advance — and **that warrant HELD, exactly** (§2.1, §3.1).

**Two sign conventions, both labelled at every use.** `Δ_path = <path> − arm W` (positive = ahead
of the frozen parent); 🚨 `Δ_extract = fold path − control` (positive = the fold bought what the
plain continuation did not). **The rule, fixed in advance:** OUTSIDE THE FLOOR iff **(a)**
`|Δ| > floor` **AND** **(b)** the Δ's own 95 % CI excludes the floor POINT; otherwise WITHIN FLOOR
at n = 2. 🚨 **ONE PAIR BOUNDS A FLOOR; IT DOES NOT ESTIMATE ONE** (rules 19/22); WITHIN FLOOR is
never "equivalent" (rule 6).

**12,900 battles, 0 TIMEOUTS (0.0 %)** — rule 12's 25 % INCONCLUSIVE threshold nowhere near.

---

## 0. THE HEADLINE, in one sentence

> 🚨 **The plain continuation of arm W — no teachers, same dose, same steps, same endpoint — is
> `+15.50 pp` on the untaught 8 (8 of 8 teams, OUTSIDE an imported 3.69 pp floor by BOTH clauses),
> `+0.1125` on Big-5 and `+0.1775` on DDTar (both OUTSIDE their own MEASURED cell floors by both
> clauses), against its frozen parent — and the ERA-1 FOLD, at the same depth from the same parent
> at the same dose, is BELOW it on every one of those three rows: `−9.75 pp` off-slice (OUTSIDE the
> floor), `−0.1650` on Big-5 (OUTSIDE), `−0.0237` on DDTar (within).**

Two things follow, and only two:

1. 🚨 **"Our parents do not gain from ordinary continued training" (§2.2) is REFUTED for this era's
   75M win-prob parent**, and not marginally — **+15.50 pp against a 3.69 pp floor, 4.2×, 8 of 8
   teams, with the CI's lower end at +12.00.** The gen-era G5 cell read **−1.92 pp** against its own
   1.00 floor. This is a different parent in a different era and it behaves the opposite way.
2. 🚨 **Against the right baseline the era-1 fold did not merely fail to pay — it COST.** Its
   celebrated off-slice `+5.13/+5.75 pp` is **a third of what the parent would have gained by
   simply carrying on**, and the `+0.1538` DDTar row that was *"the first registered row this
   programme has ever cleared"* is **below the continuation's `+0.1775` on the same cell**.

⚠️ **What this does NOT say** is in §7, and the first entry is the one that matters: this read does
**not** isolate which part of the distill block did the damage. The fold path differs from the
control in the loss, in `--distill-team-bias 0.4`, **and in its opponent distribution** (two
exploiter specialists in the stable pool). Three levers, one contrast.


## 1. THE DELIVERABLE — the 2 × 3 × 3 table, and the verdicts

### THE 2 x 3 x 3 TABLE — path x depth x row

| depth | path | untaught pp | untaught Δ vs arm W | Big-5 rate | Big-5 Δ | DDTar rate | DDTar Δ |
|---|---|---:|---|---:|---|---:|---|
| **+3M** | **CONTROL** `ai_v13_09_wcont` | 58.50 | **+12.31** [+9.69, +15.56] | 0.6062 | **+0.1137** [+0.0650, +0.1617] | 0.6312 | **+0.1613** [+0.1127, +0.2087] |
| **+3M** | fold path | 51.56 | **+5.38** [+1.63, +9.19] | 0.4675 | **-0.0250** [-0.0737, +0.0239] | 0.5938 | **+0.1238** [+0.0749, +0.1717] |
| **+6M** | **CONTROL** `ai_v13_09_wcont` | 60.44 | **+14.25** [+11.69, +16.63] | 0.5950 | **+0.1025** [+0.0537, +0.1506] | 0.6388 | **+0.1688** [+0.1203, +0.2161] |
| **+6M** | fold path | 51.31 | **+5.13** [+1.06, +9.00] | 0.4313 | **-0.0612** [-0.1097, -0.0124] | 0.5375 | **+0.0675** [+0.0185, +0.1160] |
| **+12M** | **CONTROL** `ai_v13_09_wcont` | 61.69 | **+15.50** [+12.00, +18.88] | 0.6050 | **+0.1125** [+0.0638, +0.1605] | 0.6475 | **+0.1775** [+0.1291, +0.2247] |
| **+12M** | fold path | 51.94 | **+5.75** [+1.19, +11.06] | 0.4400 | **-0.0525** [-0.1010, -0.0036] | 0.6238 | **+0.1538** [+0.1051, +0.2013] |
| +0 | arm W — the FROZEN PARENT | 46.19 | — | 0.4925 | — | 0.4700 | — |
| +0 | W_b — the seed-floor arm | 49.88 | — | 0.5400 | — | 0.5550 | — |

*floors: untaught **3.69 pp** (IMPORTED) · Big-5 **0.0475** (measured on the cell) · DDTar **0.0850** (measured on the cell)*

### THE VERDICT TABLE

| # | row | finding | floor | (a) | (b) | verdict |
|---|---|---|---:|---|---|---|
| **1a** | untaught, CONTROL, +3M | **+12.31 pp** [+9.69, +15.56], 8 of 8 | 3.69 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **1b** | untaught, CONTROL, +6M | **+14.25 pp** [+11.69, +16.63], 8 of 8 | 3.69 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **1c** | untaught, CONTROL, +12M | **+15.50 pp** [+12.00, +18.88], 8 of 8 | 3.69 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **X+3M** | 🚨 EXTRACTION, untaught, +3M (fold − control) | **-6.94 pp** [-12.31, -1.56], 2 of 8 | 3.69 | ✅ | ❌ | **WITHIN FLOOR at n = 2** |
| **X+6M** | 🚨 EXTRACTION, untaught, +6M (fold − control) | **-9.12 pp** [-11.38, -6.50], 0 of 8 | 3.69 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **X+12M** | 🚨 EXTRACTION, untaught, +12M (fold − control) | **-9.75 pp** [-14.75, -3.75], 1 of 8 | 3.69 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **2** | per-slice BIG-5, CONTROL, +3M | **+0.1137** [+0.0650, +0.1617] | 0.0475 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **2** | per-slice BIG-5, CONTROL, +6M | **+0.1025** [+0.0537, +0.1506] | 0.0475 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **2** | per-slice BIG-5, CONTROL, +12M | **+0.1125** [+0.0638, +0.1605] | 0.0475 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **X** | 🚨 EXTRACTION, BIG-5, +3M (fold − control) | **-0.1387** [-0.1866, -0.0900] | 0.0475 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **X** | 🚨 EXTRACTION, BIG-5, +6M (fold − control) | **-0.1637** [-0.2114, -0.1150] | 0.0475 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **X** | 🚨 EXTRACTION, BIG-5, +12M (fold − control) | **-0.1650** [-0.2126, -0.1163] | 0.0475 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **2** | per-slice DDTAR, CONTROL, +3M | **+0.1613** [+0.1127, +0.2087] | 0.085 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **2** | per-slice DDTAR, CONTROL, +6M | **+0.1688** [+0.1203, +0.2161] | 0.085 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **2** | per-slice DDTAR, CONTROL, +12M | **+0.1775** [+0.1291, +0.2247] | 0.085 | ✅ | ✅ | **OUTSIDE THE FLOOR** |
| **X** | 🚨 EXTRACTION, DDTAR, +3M (fold − control) | **-0.0375** [-0.0850, +0.0102] | 0.085 | ❌ | ❌ | **WITHIN FLOOR at n = 2** |
| **X** | 🚨 EXTRACTION, DDTAR, +6M (fold − control) | **-0.1013** [-0.1488, -0.0530] | 0.085 | ✅ | ❌ | **WITHIN FLOOR at n = 2** |

### 🚨 THE BRANCHES — ROW 1 IS **UNCOVERED**, ROW 2 IS **(d)**

The GO fixed four branches in two independent pairs. **Row 2 lands on (d) as written. Row 1 lands
outside both of its clauses — and NOT in the middle the registration named: the control
OVERSHOT.**

| branch | its clause | met? |
|---|---|---|
| **(a) SENIORITY / CONTINUATION** | `\|Δ_extract\|` ≤ 3.69 pp at ≥ 2 of 3 depths, control positive there | ❌ — `\|Δ_extract\|` is **6.94 / 9.12 / 9.75 pp**, FAR LARGER than the floor, not smaller |
| **(b) EXTRACTION** | `\|Δ_path(control)\|` < 3.69 pp at ≥ 2 of 3 **AND** Δ_extract clears the floor | ❌ — the control is **+12.31 / +14.25 / +15.50**, nowhere near 0; Δ_extract does clear, in the **NEGATIVE** direction |
| **(c) TEAM-DIFFERENTIAL CONTINUATION** | the control's +12M signs match the fold's (DDTar +, Big-5 −) and its split ≤ −0.103 | ❌ — the control is **POSITIVE ON BOTH** slices; its split is −0.0650 |
| **(d) THE SPLIT IS THE FOLD's** | the control's Big-5 − DDTar ≥ 0, or the sign pattern is absent | ✅ — **the sign pattern is absent** |

🚨 **Row 1's outcome is reported as UNCOVERED and the branches are NOT rewritten.** The
registration's two clauses bracket a control at ~0 (extraction) and a control at ~+5 (seniority);
the registered "uncovered middle" was the gap *between* them. **What happened is outside the
bracket on the far side.** That is the honest label, and `PREDICTION.md` §4's standing instruction
— *"report as UNCOVERED and name which clause failed"* — is followed literally.

**The nearest description in the registration's own vocabulary:** *branch (a) with the sign
reversed — the fold's off-slice lean is not teaching, and it is not even the whole of continuation;
it is a FRACTION of continuation.*

🚨 **And the answer to the GO's branch-(a) rider is YES and then some:** *"'our parents do not gain
from continuation' is REFUTED for this era's 75M win-prob parent, and the fold added nothing
off-slice"* — the refutation holds at 4.2× the floor, and the fold did not add nothing, it
subtracted.

🚨 **The answer to branch (c)/(d)'s rider is the OPPOSITE of what the sign test alone says.** The
control does **not** reproduce the fold's sign split, so by the registered clause the split is the
fold's (d). But it is the fold's **because the fold is the arm that went DOWN on Big-5** — the
control is `+0.1125` there. The split is not a property the fold created on top of a shared
trajectory; it is the fold **losing 0.165 of a 0.11 continuation gain on one archetype and 0.024 on
the other.** The DDTar "first cleared row" is therefore **not teaching**, exactly as branch (c)
proposed — just not for branch (c)'s reason.

---

## 2. ROW 1 — THE UNTAUGHT METER (off-slice; this IS the collateral read)

`python -m main.untaught_meter`, registry opponent `untaught_meter_opponent`
(= `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` @24,000,000), the untaught-8 manifest in
its canonical order, **200 games/team, `--seed 0`, concurrency 1, 8 single-concurrency shards, all
four MEASURED refs in ONE invocation**. **6,400 battles, 0 TIMEOUTS.**

### 2.1 ✅ THE REGISTERED REPRODUCTION CHECK PASSES EXACTLY, ON THE PER-TEAM ROWS

| ref | banked | **here** | Δ | all 8 per-team rows identical? |
|---|---:|---:|---:|---|
| **arm W** | 46.19 pp (739 / 1600) | **46.19 pp (739 / 1600)** | **0.00** | ✅ |

**This is the fifth independent confirmation that the meter is deterministic at seed 0 /
concurrency 1**, and the first across four reads and four different ref lists. The 3.69 pp floor
re-measures at exactly **−3.69 pp** on the identical games. Every imported fold-path level is
therefore comparable to the control's not by argument but because they are **the same games**
(`PREDICTION.md` §1.2's warrant, registered before the first battle).

### 2.2 The contrasts against the OTHER parent seed, and against the control's own earlier self

| contrast | Δ pp | CI95 | teams favouring first |
|---|---:|---|---:|
| **control +3M − W_b** | **+8.62** | [+3.75, +13.12] | 6 of 8 |
| **control +6M − W_b** | **+10.56** | [+5.00, +16.25] | 7 of 8 |
| **control +12M − W_b** | **+11.81** | [+6.69, +17.12] | **8 of 8** |
| arm W − W_b — THE FLOOR, re-measured here | −3.69 | [−8.19, +0.81] | 2 of 8 |
| control's own +12M − +3M | +3.19 | [+0.94, +5.19] | 7 of 8 |
| control's own +12M − +6M | +1.25 | [−0.50, +3.19] | 3 of 8 |
| control's own +6M − +3M | +1.94 | [−0.63, +4.56] | 5 of 8 |

⚠️ **Arm W is the LOWER of the two parent seeds, and it does not matter here.** Against W_b the
control is **+8.62 / +10.56 / +11.81 pp** — still 2.3× to 3.2× the 3.69 floor with every CI clear
of it. **The result does not depend on the choice of seed** (hazard W-E).

🚨 **MOST OF THE CONTROL's OFF-SLICE GAIN IS THERE BY +3M** (+12.31 of the final +15.50), and its
own last 50 % of the budget moves it **+1.25 pp inside a ±1.8 pp interval** — the *same shape* the
fold path showed (+0.62 pp over its last 50 %). **Both arms front-load; they front-load to very
different levels.**

### 2.3 🚨 THE CLAUSE THAT WAS SUPPOSED TO BIND DID NOT

`PREDICTION.md` §2 recorded, before any game, that clause (b) is **near-unsatisfiable** on this row
— half-widths of ≈3.8–6.2 pp against a 3.69 pp floor mean it needs `|Δ| ≳ 7.5 pp`. **Every untaught
verdict in both earlier reads was WITHIN FLOOR by exactly that arithmetic.** Here the control's
effect is **12.3 to 15.5 pp**, so clause (b) passes at all three depths and the extraction row
passes at two of three. **The bar that could not resolve a 5 pp fold effect resolves a 15 pp
continuation effect without difficulty** — which is the clearest possible demonstration that the
earlier WITHIN-FLOOR verdicts were a statement about the SIZE of the fold effect and not an artefact
of an unusable rule.

---

## 3. ROW 2 — PER-SLICE PILOTING: THE TEAM-DIFFERENTIAL CONTINUATION READ

Every ref pilots the SAME pinned taught team against the SAME fixed opponent
(`untaught_meter_opponent` @24,000,000 — **not** arm W's pool sentinels, because a sentinel is the
trainee's OWN snapshot and arm W's differ from W_b's, floor-read hazard F-G) drawing the SAME
800-long pool-team sequence under the SAME dice. **6,400 battles run here, 0 TIMEOUTS**, plus the
declared imports. Team pins `450fb83c20` (Big-5) and `6212de2e8c` (DDTar).

### 3.1 ✅ THE IMPORT WARRANT — arm W reproduces EXACTLY on BOTH cells

| cell | banked | **here** | reproduces? |
|---|---:|---:|---|
| arm W, Big-5 | 394 / 800 | **394 / 800** | ✅ |
| arm W, DDTar | 376 / 800 | **376 / 800** | ✅ |

Registered before the first battle: *if arm W does not return 394/800 and 376/800, every import is
VOID and this row is INVALID, not patched.* It did.

### 3.2 🚨 THE CONTROL CLEARS BOTH CELLS AT ALL THREE DEPTHS — SIX OF SIX

| cell | +3M | +6M | +12M | floor (measured on the cell) |
|---|---|---|---|---:|
| **Big-5 (balance)** | **+0.1137** ✅✅ | **+0.1025** ✅✅ | **+0.1125** ✅✅ | 0.0475 |
| **DDTar (offense)** | **+0.1613** ✅✅ | **+0.1688** ✅✅ | **+0.1775** ✅✅ | 0.0850 |

*(✅✅ = clause (a) and clause (b); every one is OUTSIDE THE FLOOR.)* **Before this read, the fold
programme had cleared exactly ONE registered row in its entire history. The control clears six.**

### 3.3 🚨 THE TEACHERS' CEILING, MEASURED AGAINST THE ARM THAT NEVER SAW THEM

| contrast | Δ | Newcombe 95 % | reading |
|---|---:|---|---|
| **t1 Big-5 − control @ +12M** | **−0.0287** | [−0.0767, +0.0194] | 🚨 **the control is LEVEL with the teacher** on the teacher's own pinned team, NOT DETECTED against zero |
| **t2 DDTar − control @ +12M** | **−0.1700** | [−0.2172, −0.1216] | 🚨 **the control is FAR ABOVE the teacher** |
| t1 Big-5 − arm W | +0.0838 | [+0.0349, +0.1321] | the gap the fold was asked to close |
| t2 DDTar − arm W | +0.0075 | [−0.0413, +0.0563] | **NOT DETECTED — there was almost no gap** |

🚨 **The +0.0838 gap the fold was asked to close, and crossed BACKWARDS, the plain continuation
closed by simply carrying on.** The control ends 0.0287 short of t1 with a CI covering zero, having
never once trained on the Big-5 team with any bias and never seen t1's actions. The fold, which did
both for 12.09M steps, ends **0.1363 below t1**.

### 3.4 The control's own slice trajectory is FLAT after +3M

| cell | +12M − +6M | +12M − +3M |
|---|---|---|
| Big-5 | +0.0100 [−0.0379, +0.0579] | −0.0012 [−0.0490, +0.0465] |
| DDTar | +0.0087 [−0.0381, +0.0556] | +0.0162 [−0.0308, +0.0632] |

**Everything the continuation buys on these two teams, it has bought by +3M.** ⚠️ Three points on
one arm is a DESCRIPTION of this arm, never a trend (the standing short-window refusal, 7-for-7).

### 3.5 The pooled row, SECONDARY — the trap was pre-declared and it does not fire

Pooled over both taught teams the control reads **0.6262** against arm W's **0.4813**, i.e.
**+0.1449**. ⚠️ It happens to describe both cells adequately this time *because the two cells agree
in sign* — which is precisely the condition under which pooling is harmless and which did NOT hold
for the fold path (rule 10; the fold's +6M pooled row was an exact +0.003 null manufactured out of
−0.061 and +0.068). **The per-team rows are still the result; this is a footnote.**

---

## 4. ROW 3 — THE EXTERNAL ANCHOR: Metamon `SmallRL`, greedy, AWAY team set

Through the tool of record, `python -m main.anchors`, which starts and stops its own Showdown server
(**:9450**, inside this job's 9400–9499 band), verifies the greedy regime **per decision** on both
sides, and writes the Wilson interval itself. `SmallRL` ckpt 40, `VanillaAttention`, Metamon
`@0a00a759`, Showdown pin `e0551883f`, CPU, 100 games as two role-balanced half-cells.

| arm | win rate | Wilson 95 % | half-cells |
|---|---|---|---|
| 🚨 **`ai_v13_09_wcont` @ +12.09M — THE CONTROL** | **0.640** | **[0.542, 0.727]** | **0.58 / 0.70** |
| W_b (banked) | 0.590 | [0.492, 0.681] | 0.54 / 0.64 |
| fold `ai_v13_07_fold1` @ +6.09M (banked) | 0.540 | [0.443, 0.634] | 0.58 / 0.50 |
| arm S (banked, scale only) | 0.520 | [0.423, 0.615] | — |
| **arm W — the PARENT** (banked) | **0.500** | [0.404, 0.596] | — |
| fold path `ai_v13_08_fold1_cont` @ +12.09M (banked) | 0.430 | [0.337, 0.528] | 0.38 / 0.48 |

| contrast | Δ | Newcombe 95 % | floor | (a) | (b) | verdict |
|---|---|---|---|---|---|---|
| **control − arm W** | **+0.140** | [+0.003, +0.270] | 0.090 | ✅ | ❌ | **WITHIN FLOOR at n = 2** |
| **control − the fold path @ +12.09M** | **+0.210** | [+0.072, +0.337] | 0.090 | ✅ | ❌ | **WITHIN FLOOR at n = 2** |
| control − fold @ +6.09M | +0.100 | [−0.036, +0.231] | 0.090 | ✅ | ❌ | WITHIN FLOOR at n = 2 |
| control − W_b | +0.050 | [−0.084, +0.181] | 0.090 | ❌ | ❌ | WITHIN FLOOR at n = 2 |

**This row was registered as DESCRIPTIVE and it behaved as registered** (P(clears) = 0.15; it did
not). At 100 games the cell resolves ~±0.10 against a 0.090 run-level floor, so it cannot separate
these arms whatever they did — **but the control is the highest of the six arms ever measured on
this cell and the fold path is the lowest**, and the direction agrees with every other row.

### 4.1 ⚠️ HAZARD W-J — `regime_verified` reads FALSE, and the reason is NOT a regime failure

| half | decisions | argmax checked / matched | rate | our `stochastic` | their regime | `regime_check_ok` | peer error |
|---|---:|---|---:|---|---|---|---|
| `ours_challenge` | 2,677 | 2,677 / 2,677 | **1.0000** | `[false]` | greedy | ✅ true | — |
| `peer_challenge` | 2,961 | 2,961 / 2,961 | **1.0000** | `[false]` | greedy | ❌ false | 🚨 `RecursionError: maximum recursion depth exceeded in comparison` |

`main.anchors`' composite `regime_verified` is an AND over both halves' `regime_check_ok`, and a
peer's flag goes false if the peer process raises **anywhere** — including after its last decision,
in the results aggregation, which is what happened (`peer_rc: 1`, `peer_results` empty for that
half). **The per-decision instruments are what verify the regime, and they read MATCHED on BOTH
halves over 5,638 decisions.** Also: `status: OK`, 100/100 games, 0 ties, **1 game at the 250-turn
forfeit cap** (the earlier reads had 0), `distinct_our_teams` 19 of 20, `model_loader: bare`,
`model_rung: explicit_zip`, `model_step` 87,097,344, mean 51.37 turns, role split **−12 pp**
(at n = 50 per half nothing may be taken from it). Reported as a finding; the row's verdict does not
depend on it.

---

## 5. ROWS 4 AND 5 — the descriptors, and one of them is the loudest number in the file

* **entropy** `H_end` control **0.9447** (banked 0.9447, reproduces=True) vs fold path **0.7465** ⇒ gap **+0.1982** nats = **10.4x** the 75M run-level floor 0.019
* **promotions** control **5** post-fork ([78000000, 80000016, 82000032, 84000000, 86000016]), `snapshot_ladder/` present=True; fold path **0**, present=False

| step | +steps | control `win_rate_vs_pool` | control `win_rate_vs_bots` |
|---:|---:|---:|---:|
| 76,000,032 | +1.0M | **0.534** | 0.9175 |
| 78,000,000 | +3.0M | **0.590** | 0.9287 |
| 80,000,016 | +5.0M | **0.622** | 0.8912 |
| 82,000,032 | +7.0M | **0.588** | 0.9325 |
| 84,000,000 | +9.0M | **0.576** | 0.9187 |
| 86,000,016 | +11.0M | **0.558** | 0.9175 |

### 5.1 🚨 THE PROMOTION ASYMMETRY — the run's own evidence, disclosed pre-look

| | **control `ai_v13_09_wcont`** | `ai_v13_07_fold1` | `ai_v13_08_fold1_cont` |
|---|---|---|---|
| `snapshot_ladder/` | **present** | absent | absent |
| snapshots the run **promoted itself**, post-fork | 🚨 **5** (78.0M, 80.0M, 82.0M, 84.0M, 86.0M) | **0** | **0** |
| pool contents at the end | its own five + arm W's 36.0M → 72.0M (the five oldest seeds EVICTED) | arm W's 26.0M → 72.0M, untouched | the same 20 files again |
| `eval/win_rate_vs_pool`, post-fork | **0.534 / 0.590 / 0.622 / 0.588 / 0.576 / 0.558** | 0.488 / 0.446 / 0.412 | 0.372 / 0.400 / 0.402 |
| ever reached the 0.55 promotion gate? | **yes, at five of six cycles** | **never** | **never** |

**Over the same 12,091,392 steps, against the same auto-seeded arm-W pool at the same 0.55 gate,
the plain continuation cleared the gate five times and the fold path never once.** ⚠️ **No Elo is
quoted from either side** — a Bradley-Terry fit over five own nodes is far below the n ≥ 12 report
floor, and by the end the two arms' pools are not the same object.

⚠️ **And the two `win_rate_vs_pool` series are NOT comparable in kind.** The fold path's pool was
FROZEN (nothing promoted, so every point faces the identical 20 arm-W snapshots); the control's
pool GREW as it promoted, so its later cycles face a **strengthening** opponent set and still read
0.558–0.622. That asymmetry makes the control's series the harder one — it is noted in the control's
favour and no bar is attached to either (hazard W-G).

### 5.2 The entropy row — a DESCRIPTOR, reproduced, and deliberately not used

`H_end` (median of the last 20 post-fork points, `H = −train/entropy_loss`) reproduces the ledger's
banked **0.9447** exactly, against the fold path's **0.7465**: a **0.198-nat** gap, **10.4×** the
75M run-level entropy floor of 0.019. The control's own post-fork range is 0.918–1.047 across 117
points.

🚨 **The conditional statement `PREDICTION.md` §3.4 licensed in advance, and no more:** *the
higher-entropy arm is the one that pilots better off-slice, by 15.5 pp against its parent where the
lower-entropy arm manages 5.8.* **NO mechanism is claimed. Two arms cannot support an
entropy-to-piloting claim, entropy is perfectly confounded with the distill block itself, and the
direction is exactly what `--distill-target action --distill-topk 1` predicts for the entropy
alone.** It is printed because it is the only axis on which the two arms were already known to
differ, and because it now has a piloting row beside it.

### 5.3 ⚠️ HAZARD W-I — THE BOT ROW IS BLIND TO ALL OF THIS

The completion entry banked bots and G7 as **indistinguishable** between the two paths (control
0.891–0.933 vs fold path 0.925–0.934; G7 worst 1.083 vs 1.090, both under bar) and said so as a
descriptor. **That is the same 12.09M steps over which the untaught meter separates them by 9–10 pp
and the Big-5 cell by 0.165.** A bot row saturated at ~0.92 cannot see a difference of this size,
and neither can G7. **This is not a criticism of the banked table — it is the strongest available
demonstration that the in-run bot row is not a competence meter at this depth**, and it is why the
completion entry was right to call the untaught meter "the read that decides".

---

## 6. THE READING

### 6.1 One paragraph, every hedge in place

> *At matched depth, matched endpoint step, matched frozen dose and matched parent, the plain
> continuation `ai_v13_09_wcont` is **+15.50 pp [+12.00, +18.88] on the untaught 8** against its
> frozen parent (8 of 8 teams, **OUTSIDE** an imported 3.69 pp floor by both clauses), **+0.1125
> [+0.0638, +0.1605] on the Big-5 taught team** and **+0.1775 [+0.1291, +0.2247] on DDTar** (both
> **OUTSIDE** floors measured on those very cells), and **+0.140 on the `SmallRL` away anchor**
> (WITHIN 0.090). The era-1 fold path, from the same parent at the same dose to the same step, is
> **−9.75 pp [−14.75, −3.75] BELOW it off-slice (OUTSIDE the floor, 1 of 8 teams favouring the
> fold)**, **−0.1650 [−0.2126, −0.1163] below it on Big-5 (OUTSIDE)** and **−0.0237 [−0.0707,
> +0.0234] below it on DDTar (WITHIN)**. Row 1's registered branches are both missed — the control
> OVERSHOT the bracket they define — and row 2 lands on **(d)**, the split being the fold's because
> the fold is the arm that went down. **`ai_v13_09_wcont` is one arm at n = 1 (rules 19/22) and this
> read does not isolate which of the distill block's three levers — the loss, the 40 % team bias,
> or the two specialists in the opponent pool — produced the loss.***

### 6.2 What changes in the research state (as a proposal — nothing was edited)

1. 🚨 **§2.2 "our parents do not gain from a continuation" needs an era qualifier.** It is a
   statement about **gen-era R2ACTION parents at ~28M**, and it is **REFUTED at 4.2× its own
   imported floor for the 75M win-prob parent**. The frozen-parent baseline does **NOT** stand on
   this side of the era boundary, and every delta-against-a-frozen-parent taken on a 75M win-prob
   arm is inflated by something of the order of **+15 pp on the untaught meter over 12M steps**.
2. 🚨 **The convergence read's DDTar row — *"the first registered row this programme has ever
   cleared"* — is re-based and does not survive re-basing.** `+0.1538` against the frozen parent
   becomes `−0.0237` against the continuation. It still clears its floor against arm W; it clears
   nothing against the arm that answers "what would have happened anyway".
3. ⚠️ **The two earlier reads are not wrong; they were under-baselined, and they said so.** Both
   printed *"NO CONTINUATION CONTROL; every delta is against a FROZEN parent"* beside every number
   (their hazards C-J and §6.1), and both named this arm as the one that would fix it.
4. ⚠️ **What is NOT licensed** is "distillation hurts". §7.

### 6.3 The cheapest decisive follow-ups, in order

* 🚨 **A SECOND CONTROL SEED.** Rules 19/22: one continuation arm bounds nothing, and a +15 pp
  effect on a single arm is exactly the configuration rule 2 warns about. This is the only thing
  that turns any verdict here into a family claim. ~10 GPU-h.
* 🚨 **THE THREE-LEVER SPLIT.** The fold path differs from the control in the distill LOSS, in
  `--distill-team-bias 0.4`, **and in its opponent pool** (two exploiter specialists). The
  gen-era C1 cell (`--distill-coef 0` with the 40 % bias still ON) is the design that isolates the
  sampling axis; re-running it at 75M on this parent is one arm and would say whether the loss or
  the ecology is the carrier.
* **The same untaught cell on `ai_v13_08_fold1_cont`'s intermediate checkpoints is already banked**
  — no further battles are needed to place the fold path anywhere on this axis.

---

## 7. WHAT THIS READ DOES NOT LICENSE

* 🚨 **It does not say "distillation hurts".** The fold path carries **three** differences from the
  control, not one: the distill loss, `--distill-team-bias 0.4`, and **two exploiter specialists in
  its stable opponent pool** — a different opponent distribution for 12.09M steps. Any of the three
  could carry the whole gap. This read measures the **RECIPE**, not the loss.
* 🚨 **It does not make anything a family verdict** (rules 19/22). **One control arm.** A
  control-replicate floor bounds the CONTROL's run-to-run variance, not a lever's, and the replicate
  that promotes a detection is a replicate OF THE ARM. Rule 2's warning applies in full.
* **It does not establish any floor.** Every bar is `|arm W − W_b|` at n = 2 — one pair BOUNDS,
  never estimates — and the untaught and anchor bars are additionally IMPORTS from a fresh-arm pair.
* **It does not separate the FORK BOUNDARY from the STEPS** (hazard W-F) — but it bounds it (§8).
* **It does not attribute anything to ENTROPY** (§5.2).
* **It does not read a TREND** off three points on one arm.
* **It says nothing about folds in general**, at another dose, with other teachers, or at 277M; and
  nothing about ladder STRENGTH, on which no rating is quoted for any arm here.

---

## 8. HAZARDS — every one a finding

| # | hazard | why it matters, and what was done |
|---|---|---|
| **W-A** | 🚨 **ROW 1's REGISTERED BRANCHES ARE BOTH MISSED, AND NOT IN THE MIDDLE.** (a) needs `\|Δ_extract\|` ≤ 3.69 and got 6.9–9.8; (b) needs the control near 0 and got +12 to +15. | Reported as **UNCOVERED** in §1 with every clause mechanically printed, the direction of each failure named, and **no branch rewritten**. `PREDICTION.md` §4 registered this handling in advance precisely because the convergence read met no branch and had to improvise. The JSON's diagnostic distinguishes an OVERSHOOT from the registered middle. |
| **W-B** | 🚨 **THE EXTRACTION ROW IS NEGATIVE AND CLEARS THE FLOOR.** −9.12 and −9.75 pp off-slice (OUTSIDE), −0.1387/−0.1637/−0.1650 on Big-5 (OUTSIDE at all three depths). | Stated as the headline rather than buried: against the right baseline the fold did not fail to pay, it **cost**. Its DDTar row, the programme's only previously-cleared row, re-bases to −0.0237 (WITHIN). |
| **W-C** | ⚠️ **INHERITED INERT DISTILL FLAGS.** The control's `model_config.json` carries `distill_target: "kl"`, `distill_topk: 1`, `distill_beta: 1.0`, `distill_gate: "none"` — re-resolved from arm W's checkpoint config because the argv did not name them. | **INERT at `--distill-coef 0.0` with `teachers: []`** and `grep -c DISTILL` = 0 across the whole run, so no distill code path ran. Disclosed in `PREDICTION.md` §0.4 before any number, and named here because "any flag the argv does not NAME re-resolves" is the standing rule that makes a config diff between these two arms read `distill_target kl vs action` when the operative difference is `--distill-coef`. |
| **W-D** | ⚠️ **THE +6M DEPTHS DIFFER BY 42,480 STEPS, WITH THE CONTROL DEEPER.** | 0.35 % of the post-fork budget, and it flatters the control by a hair on that row only. **+3M and +12M are the SAME STEP NUMBER on both paths** (78,006,048 and 87,097,344) and show the same picture — +12.31 / +15.50 vs +5.38 / +5.75 — so no conclusion rests on the offset row. Named in `PREDICTION.md` §1.1 before the grid was run. |
| **W-E** | ⚠️ **ARM W IS THE LOWER OF THE TWO PARENT SEEDS** (untaught 46.19 vs W_b's 49.88; Big-5 0.4925 vs 0.5400; DDTar 0.4700 vs 0.5550). | Both contrasts are printed (§2.2). Against W_b the control is **+8.62 / +10.56 / +11.81 pp**, still 2.3–3.2× the floor with CIs clear of it. **The result does not depend on the seed choice.** |
| **W-F** | 🚨 **THE FORK-BOUNDARY ASYMMETRY.** The control ran **ONE continuous +12M** (3 launcher restarts, no fork); the fold path is +6.09M then +5.99M **with a FORK between**, which re-seeds the optimizer path, re-pins the LR, re-seeds the pool by name and opens a new event file. The fold path's untaught row dips exactly at that crossing (−2.31 pp, **1 of 8** teams). | **Bounded, not resolved.** At **+3M neither path has crossed a fork** — both are single-run continuations of arm W at the same frozen dose, differing only in the distill block — and the extraction gap is **already −6.94 pp off-slice and −0.1387 on Big-5** there. So the fork crossing cannot account for the bulk of the gap; it can account for part of the widening from −6.94 to −9.75. Stated, with the part it can and cannot explain separated. |
| **W-G** | ⚠️ **THE TWO `win_rate_vs_pool` SERIES ARE NOT COMPARABLE IN KIND.** The fold path's pool was frozen (zero promotions); the control's GREW as it promoted five times. | §5.1. The control's later cycles face a strengthening opponent set and still read 0.558–0.622 against a 0.55 gate — the harder measurement — so the asymmetry runs in the control's favour. DESCRIPTORS on both sides, no bar attached to either, and no Elo quoted. |
| **W-H** | ⚠️ **CLAUSE (b), REGISTERED AS NEAR-UNSATISFIABLE, PASSED EVERYWHERE THAT MATTERED.** | §2.3. Recorded because the registration said it would bind: the bar needs `\|Δ\| ≳ 7.5 pp` and the control's effect is 12–15 pp. The earlier WITHIN-FLOOR verdicts were about the SIZE of the fold effect, not an unusable rule. |
| **W-I** | 🚨 **THE BOT ROW AND G7 ARE BLIND TO A DIFFERENCE THIS SIZE.** Banked as "indistinguishable" between the two paths over exactly the span in which the untaught meter separates them by 9–10 pp. | §5.3. Not a criticism of the banked table — a demonstration that a bot row saturated at ~0.92 is not a competence meter at this depth. |
| **W-J** | ⚠️ **`regime_verified` READS FALSE ON THE ANCHOR, FOR A POST-GAME CRASH.** `peer_rc: 1`; the challenger half's peer raised `RecursionError` in its results aggregation **after** all 2,961 of its decisions had been argmax-checked. | §4.1. Both halves' per-decision instruments read `argmax_match_rate` **1.0000** over 5,638 decisions with our `stochastic` kwarg `[false]` and `their_regime: greedy`. The composite flag and the regime are separated in `out/wcont_away_anchor.json` rather than collapsed. **1 game hit the 250-turn forfeit cap** (the earlier reads had 0). The row is DESCRIPTIVE and its verdict does not depend on any of this. |
| **W-K** | 🚨 **ONE CONTROL ARM.** No seed replicate; rules 19/22 and rule 2. | Every verdict here is a **CANDIDATE**. §6.3 names the replicate as the first follow-up. |
| **W-L** | 🚨 **THREE LEVERS, ONE CONTRAST.** loss + team bias + opponent pool. | §7's first entry. The read measures the RECIPE. The C1-style cell (`--distill-coef 0`, bias ON) is the design that splits it. |
| **W-M** | ⚠️ **EVERY FLOOR EXCEPT THE TWO PER-SLICE ONES IS AN IMPORT FROM A 75M FRESH-ARM SEED PAIR ONTO A FORKED CONTINUATION.** A floor is a property of the DEPTH and the REGIME (rule 3). | Labelled an IMPORT in `PREDICTION.md` §1.3 and in every row it bars. The nearest floor in KIND is the gen-era G5 three-continuation-arm floor of **1.00 pp**, at a different parent and era; against **that** the control's +15.50 is 15×. The two per-slice floors were MEASURED on the same cells and arm W's half of each re-measured here, reproducing exactly. |
| **W-N** | ⚠️ **THE CONTROL'S GAIN IS FRONT-LOADED AND THE READ CANNOT SAY WHY.** +12.31 of the final +15.50 pp is there by +3M; both slice cells are at their level by +3M. | §2.2, §3.4. Three points on one arm is a description of that arm; no trend is read and no mechanism proposed. |
| **(context)** | The box carried another agent's CPU battery throughout plus this job's 14 concurrent battle workers; load average ran **12 → 49**. | Everything ran `CUDA_VISIBLE_DEVICES=""`, `nice 15`, `OMP_NUM_THREADS=1` for the anchor peer, from the MAIN checkout. **The GPU was never touched.** No process this read did not start was signalled; **:8000 and :8001 were never touched** and the only server started was `main.anchors`' own on :9450, which it stopped itself. Other agents' worktrees (`gatecurve/`, `matflip/`) and port bands were not touched. **12,900 battles, 0 timeouts (0.0 %)** — no row here is a width meter (rule 23), and the meters are deterministic at seed 0 / concurrency 1, which the exact reproduction of arm W on its per-team rows demonstrates under exactly this load. |

---

## 9. PROVENANCE

| | **arm W** (PARENT) | **W_b** (floor arm) | 🚨 **`ai_v13_09_wcont`** (CONTROL) | `ai_v13_07_fold1` | `ai_v13_08_fold1_cont` | t1 Big-5 | t2 DDTar |
|---|---|---|---|---|---|---|---|
| `lineage.role` | `fresh` | `fresh` | **`fork`** | `fold` | `fold` | fork/exploiter | fork/exploiter |
| forked from | — | — | **arm W @75,005,952** `[explicit_zip]` | arm W @75,005,952 | `ai_v13_07_fold1` @81,100,800 | arm W | arm W |
| teachers | — | — | 🚨 **`[]` — none** | 2 | 2 | — | — |
| steps | 75,005,952 | 75,005,952 | **87,097,344** (+12,091,392) | 81,100,800 | 87,097,344 | 83,066,880 | 83,066,880 |
| pin | `6eb9c776` | `6eb9c776` | **`6eb9c776`, ONE row** | `6eb9c776` | `6eb9c776` | `6eb9c776` | `6eb9c776` |
| config / arch | 119 / `gen3_critic_route_wave_v1` | same | same | same | same | same | same |
| realized dose | 4.578e-08 | 5.035e-08 | **4.272e-09** `[FROZEN; pinned 2.80e-05]` | 4.272e-09 | 4.272e-09 | 3.815e-08 | 3.815e-08 |
| matchup hash | `ef5242cffd` | — | **`ef5242cffd` — arm W's own, UNCHANGED** | `0a7b730a4d` | `0a7b730a4d` | — | — |
| post-fork promotions | — | — | **5** | **0** | **0** | — | — |
| forks in the path | — | — | **0 (one continuous run)** | 1 | **2** | 1 | 1 |

* Every ref in both invocations resolved `[rung=explicit_zip rule=explicit_zip]`, checked in each
  shard log — no last-snapshot surprise on any comparator (rule 9).
* The control's `pin_history` is a **single row**, 75,005,952 → 87,097,344, `pin_source pin_commit`.
* `pool_seeded_from` = arm W's `snapshots/`, 20 files; by the end the five oldest (26.0M–34.0M) have
  been evicted in favour of the control's own five promotions.

---

## 10. WHAT IS IN THIS DIRECTORY

| path | what |
|---|---|
| `PREDICTION.md` | the pre-registration — bars, both branch pairs, the uncovered middle registered in advance, priors, the checkpoint grid, the declared imports and their warrant, and the five disclosed pre-look facts; committed (`e8ea5ba3`) before the first battle |
| `README.md` | this note |
| `scripts/wcont_untaught_delta.py` | row 1 — the paired team-clustered contrasts at three matched depths for BOTH paths, the extraction row, the reproduction check that gates every import, and the mechanical branch clauses |
| `scripts/wcont_slice_read.py` | row 2 — the matched-extraction row: Wilson per cell, Newcombe per difference, the import warrant gate, the split statistic, the branch (c)/(d) clauses, the pooled footnote |
| `scripts/wcont_away_anchor.py` | row 3 — the `main.anchors` away cell joined to the five banked arms, with the per-half instrument-health record (hazard W-J) |
| `scripts/wcont_run_rows.py` | rows 4–5 — the control's TB rows POST-FORK (sliced at the fork step), the entropy descriptor, the promotion asymmetry, the `win_rate_vs_pool` series |
| `scripts/render_tables.py` | renders every markdown table above straight from `out/*.json` — no number here is hand-transcribed |
| `scripts/harness/run_untaught.sh` | the untaught invocation as executed — four refs in ONE call, 8 workers, seed 0 |
| `scripts/harness/run_slice.sh` | the per-slice invocation as executed — four ref-shards × 2 workers, 800 games/cell |
| `scripts/harness/taught_slice_teams.json` | the TAUGHT-SLICE manifest, in teacher order (the order is the seed offset), byte-identical to the two earlier reads' |
| `scripts/harness/run_away_cell.sh` | the away cell as executed, on :9450 |
| `out/wcont_untaught_delta.json` | every untaught level, per-team row, contrast, extraction row, verdict and the reproduction check |
| `out/wcont_slice_read.json` | every per-slice cell, interval, floor, verdict, the split and the import record |
| `out/wcont_away_anchor.json` | the away cell with its per-half regime instruments |
| `out/wcont_run_rows.json` | the post-fork TB series, the entropy row and the promotion record |
| `out/untaught/`, `out/slice/`, `out/anchors/` | the tools' own raw artifacts, unedited |

---

## 11. Ledger paragraph — ready to append (nothing in `ledger.md`, `UNDERSTANDING.md` or any design note was edited from here)

> ### 2026-09-20 · MEASUREMENT (MAJOR) · **THE CONTINUATION CONTROL — 🚨 OUR 75M WIN-PROB PARENT *DOES* GAIN FROM ORDINARY CONTINUED TRAINING, BY +15.50 pp ON THE UNTAUGHT 8 AGAINST A 3.69 FLOOR (8 of 8 teams), AND THE ERA-1 FOLD IS 9.75 pp BELOW IT; §2.2's "our parents do not gain from a continuation" is REFUTED for this parent, and the fold's only previously-cleared row RE-BASES TO −0.0237**
>
> `designs/research_state/measurements/wcont_control_read_2026-09-20/`. Bars, both branch pairs,
> the **uncovered middle registered IN ADVANCE**, priors, the checkpoint grid and the declared
> imports pre-registered in `PREDICTION.md` and committed (`e8ea5ba3`) BEFORE the first battle —
> including the five facts LOOKED AT while scoping and disclosed there (the grid, the metadata, the
> **promotion asymmetry**, the inherited inert distill flags, the ledger's banked descriptors).
> **THE ARM.** `ai_v13_09_wcont`, a FORK of arm W `ai_v13_02_flywheel_winprob` at 75,005,952 with
> **`teachers: []`**, `--distill-coef 0.0`, the SAME frozen dose 4.272e-9 = 0.20× the v8 reference,
> the same auto-seeded pool, seed 1001, pin `6eb9c776` in ONE row, config v119, **one continuous
> +12,000,000 steps to 87,097,344 — the fold path's own endpoint step, to the step**; matchup
> `ef5242cffd`, arm W's own, unchanged. **THE RULE, copied verbatim from both fold reads so the
> three are ONE series: OUTSIDE THE FLOOR iff `|Δ| > floor` AND the Δ's CI excludes the floor
> POINT; else WITHIN FLOOR at n = 2 — one pair BOUNDS a floor (rules 19/22), WITHIN FLOOR is never
> "equivalent" (rule 6).** Two conventions: `Δ_path = path − arm W`, `Δ_extract = fold path −
> control`. CPU-only (`CUDA_VISIBLE_DEVICES=""`, `nice 15`), MAIN checkout, **nothing written under
> `models/`**, no `src/` file changed, only server `main.anchors`' own on :9450, **:8000/:8001
> never touched, the GPU never touched**. **12,900 battles, 0 TIMEOUTS.** ✅ **THE REGISTERED
> REPRODUCTION CHECK PASSES EXACTLY ON THE PER-TEAM ROWS** — arm W 46.19 pp (739/1600) on the
> untaught 8 with all eight rows identical, and **394/800 + 376/800** on the two slice cells — the
> **fifth** confirmation that the meter is deterministic at seed 0 / concurrency 1 and the WARRANT
> for importing the whole fold path, W_b and both teachers rather than re-deriving 9,600 identical
> battles; the 3.69 floor re-measures at exactly −3.69. **ROW 1 — UNTAUGHT (and it IS the collateral
> read).** Control levels **58.50 / 60.44 / 61.69 pp** at +3M/+6M/+12M ⇒ Δ vs arm W 🚨 **+12.31
> [+9.69,+15.56] / +14.25 [+11.69,+16.63] / +15.50 [+12.00,+18.88], 8 of 8 teams at EVERY depth,
> OUTSIDE THE FLOOR by BOTH clauses at all three** — against the fold path's +5.38 / +5.13 / +5.75,
> every one of which was WITHIN FLOOR. Against W_b the control is **+8.62 / +10.56 / +11.81**, so
> the result does not depend on arm W being the lower seed. 🚨 **THE EXTRACTION ROW (fold path −
> control) is −6.94 [−12.31,−1.56] / −9.12 [−11.38,−6.50] / −9.75 [−14.75,−3.75] pp, OUTSIDE THE
> FLOOR at +6M and +12M, with 0–2 of 8 teams favouring the fold.** Most of the control's gain is
> there by +3M (+12.31 of +15.50); its last 50 % moves it +1.25 [−0.50,+3.19] — the same front-loaded
> shape the fold path showed, to a very different level. ⚠️ **Clause (b), registered as
> NEAR-UNSATISFIABLE (it needs |Δ| ≳ 7.5 pp), passed everywhere that mattered** — so the earlier
> WITHIN-FLOOR verdicts were about the SIZE of the fold effect, not an unusable bar. **ROW 2 —
> PER-SLICE PILOTING, 800 games/cell, same pinned teams, same third-party opponent.** 🚨 **THE
> CONTROL CLEARS BOTH CELLS AT ALL THREE DEPTHS — SIX OF SIX, every one OUTSIDE by both clauses:
> Big-5 +0.1137 / +0.1025 / +0.1125 against a 0.0475 MEASURED floor; DDTar +0.1613 / +0.1688 /
> +0.1775 against 0.0850.** Before this read the fold programme had cleared ONE registered row in
> its history. 🚨 **AND THE FOLD PATH IS BELOW THE CONTROL ON BOTH: Big-5 −0.1387 / −0.1637 /
> −0.1650 (OUTSIDE at all three depths), DDTar −0.0375 / −0.1013 / −0.0237 (all WITHIN).** 🚨 **So
> the convergence read's DDTar +0.1538 — "the first registered row this programme has ever cleared"
> — RE-BASES TO −0.0237 against the continuation.** 🚨 **THE TEACHERS' CEILING, measured against the
> arm that never saw them: t1 Big-5 − control = −0.0287 [−0.0767,+0.0194], NOT DETECTED — the plain
> continuation is LEVEL with the teacher on the teacher's own pinned team, having closed by simply
> carrying on the +0.0838 gap the fold was asked to close and crossed BACKWARDS; t2 DDTar − control
> = −0.1700, the control far above.** The control's slice cells are flat after +3M. Pooled +0.1449 —
> harmless here only because the two cells agree in sign, which is exactly what did NOT hold for the
> fold (rule 10); footnote. **ROW 3 — ANCHOR.** `metamon:SmallRL` greedy AWAY, 100 games on :9450:
> control **0.640 [0.542,0.727]** ⇒ **+0.140 [+0.003,+0.270] vs arm W, +0.210 [+0.072,+0.337] vs the
> fold path — both WITHIN the 0.090 floor**, as registered (P(clears)=0.15); the control is the
> highest and the fold path the lowest of the six arms measured on this cell. ⚠️ `regime_verified`
> reads FALSE for a POST-GAME `RecursionError` in the challenger half's peer aggregation
> (`peer_rc: 1`), while BOTH halves' per-decision instruments read `argmax_match_rate` **1.0000**
> over 5,638 decisions with our `stochastic` `[false]`; 1 game at the 250-turn cap. **ROWS 4–5 —
> DESCRIPTORS.** `H_end` reproduces the banked **0.9447** vs the fold path's 0.7465 (0.198 nats,
> 10.4× the 75M floor) — conditional statement only, no mechanism, entropy perfectly confounded with
> the distill block. 🚨 **THE PROMOTION ASYMMETRY: over the same 12.09M steps against the same
> auto-seeded arm-W pool at the same 0.55 gate, the CONTROL promoted FIVE snapshots and has a
> `snapshot_ladder/`; the fold path promoted ZERO and has none** — the control's own
> `win_rate_vs_pool` reads **0.534 / 0.590 / 0.622 / 0.588 / 0.576 / 0.558** against the fold path's
> 0.488 → 0.402, and the control's pool GREW while the fold's stayed frozen, so the control's series
> is the HARDER measurement. No Elo quoted for any arm (n = 5 ≪ 12). 🚨 **AND THE BOT ROW IS BLIND:
> the completion entry banked bots and G7 as INDISTINGUISHABLE between the two paths over exactly
> the span in which the untaught meter separates them by 9–10 pp — a bot row saturated at ~0.92 is
> not a competence meter at this depth.** **THE BRANCHES.** 🚨 **ROW 1 IS UNCOVERED AND THE
> BRANCHES ARE NOT REWRITTEN:** (a) needs |Δ_extract| ≤ 3.69 and got 6.9–9.8; (b) needs the control
> near 0 and got +12 to +15 — **the control OVERSHOT the bracket both clauses define, which is not
> the "uncovered middle" the registration named, and the JSON's diagnostic says so.** Nearest
> description in the registration's own words: *branch (a) with the sign reversed — the fold's
> off-slice lean is not teaching and not even the whole of continuation; it is a FRACTION of it.*
> **ROW 2 lands on (d) THE SPLIT IS THE FOLD's** — the control is POSITIVE on BOTH slices, so the
> sign pattern is absent — **but it is the fold's because the FOLD is the arm that went down**, and
> branch (c)'s rider holds for a different reason: the DDTar row is not teaching. **HAZARDS, each a
> finding.** (W-A) row 1's branches both missed, on the far side; (W-B) the extraction row is
> negative AND clears; (W-C) the inherited INERT distill flags (`distill_target kl`, `distill_topk
> 1`) re-resolved from arm W's config — inert at `--distill-coef 0.0` with no teacher, `grep -c
> DISTILL` = 0; (W-D) the +6M depths differ by 42,480 steps with the CONTROL deeper — **+3M and
> +12M are the SAME STEP NUMBER and show the same picture**; (W-E) arm W is the lower seed and it
> does not matter; (W-F) 🚨 **THE FORK-BOUNDARY ASYMMETRY — the control is ONE continuous run, the
> fold path has a FORK in the middle — BOUNDED, not resolved: at +3M neither path has crossed a
> fork and the gap is ALREADY −6.94 pp and −0.1387**, so the crossing cannot carry the bulk; (W-G)
> the two `win_rate_vs_pool` series are not comparable in kind (frozen vs growing pool), the
> asymmetry favouring the control; (W-H) clause (b) did not bind; (W-I) the bot row is blind;
> (W-J) the anchor's `regime_verified` flag; (W-K) ONE control arm; (W-L) 🚨 **THREE LEVERS, ONE
> CONTRAST**; (W-M) every floor but the two per-slice ones is an IMPORT; (W-N) the control's gain is
> front-loaded and the read cannot say why. **WHAT IS NOT CLAIMED:** 🚨 **that "distillation hurts"
> — the fold path differs from the control in the distill LOSS, in `--distill-team-bias 0.4` AND in
> its OPPONENT POOL (two exploiter specialists for 12.09M steps); any of the three could carry the
> whole gap, and this read measures the RECIPE, not the loss**; that anything is a family verdict
> (ONE control arm — rules 19/22 and rule 2); that any floor is established; that entropy explains
> anything; that three points are a trend; anything about ladder STRENGTH. **WHAT FOLLOWS.** (1) 🚨
> **§2.2 needs an era qualifier — the frozen-parent baseline does NOT stand on the 75M win-prob
> side, and every delta-against-a-frozen-parent on such an arm is inflated by something of order
> +15 pp on this meter over 12M steps.** (2) The cheapest decisive follow-ups, in order: **a SECOND
> control seed** (~10 GPU-h; the only thing that makes any of this a family claim), then **the
> three-lever split** — a C1-style cell (`--distill-coef 0` with the 40 % team bias still ON) at 75M
> on this parent, which says whether the LOSS or the ECOLOGY is the carrier. Tag: **MEASURED
> (MAJOR) · THE CONTINUATION CONTROL · our 75M win-prob parent GAINS +15.50 pp from plain
> continuation (8/8 teams, 4.2× the floor, both clauses) · §2.2 REFUTED for this parent · the era-1
> fold is −9.75 pp BELOW it off-slice and −0.1650 on Big-5, both OUTSIDE the floor · the DDTar
> "first cleared row" re-bases to −0.0237 · the control is LEVEL with teacher t1 on t1's own team
> having never seen it · control promoted 5 snapshots, the fold path 0 · row 1 UNCOVERED (the
> control OVERSHOT both clauses), row 2 (d) · bots and G7 BLIND · 12,900 battles, 0 timeouts ·
> reproduction EXACT, per-team**.
