# THE ERA-1 FOLD's REGISTERED READ — `ai_v13_07_fold1`, 2026-09-19

**The fold ran its full +6M and the registered question is whether it PAID.** `ai_v13_07_fold1` is a
FORK of `ai_v13_02_flywheel_winprob` (**arm W**, the 75M win-prob flywheel arm) at 75,005,952,
distilling from the era's **two win-prob exploiters** — `ai_v13_05_exploit_big5starmie` (**t1**,
BALANCE, its pinned team `f6229d2c867e21d6`) and `ai_v13_06_exploit_ddtar_spikes` (**t2**, OFFENSE,
`9eb3abdc52876a63`) — at `--fork-lr 2.8e-5 --fork-lr-freeze`, `--distill-team-bias 0.4`, with both
teachers in the stable pool each piloting its own pinned team. Pin `6eb9c776` throughout,
config v119 / `gen3_critic_route_wave_v1`, +6,094,848 steps to **81,100,800**.

Every bar below was fixed in [`PREDICTION.md`](PREDICTION.md) and **committed before any registered
number in this directory existed**. The floors are IMPORTED VERBATIM from
[`flywheel_wb_floor_read_2026-09-18/`](../flywheel_wb_floor_read_2026-09-18/) — untaught **3.69 pp**,
ladder **25.2 Elo**, `SmallRL` greedy away **0.090** — and every one of them is `|arm W − W_b|`, a
**75M FRESH-ARM seed pair**. This arm is a **FOLD**. 🚨 **A floor is a property of the DEPTH and the
REGIME (§3.3): every floor used here is an IMPORT and is labelled as one in the row it bars.**

🚨 **ONE PAIR BOUNDS A FLOOR; IT DOES NOT ESTIMATE ONE** (rules 19/22). There is no CI on a floor,
WITHIN FLOOR is never "equivalent" (rule 6), and a row that clears one has survived ONE replicate.

**Sign convention throughout: `Δ = fold − arm W`.** Positive means the fold is ahead of its parent.
**The decision rule, fixed in advance and copied from the floor read so the two are commensurable:**
OUTSIDE THE FLOOR iff **(a)** `|Δ| > floor` **AND** **(b)** the Δ's own 95 % CI excludes the floor
POINT; otherwise WITHIN FLOOR at n = 2.

---

## 0. THE VERDICT TABLE — every registered row, finding beside floor

| # | row | **finding** `Δ = fold − arm W` + CI | **floor** | clause (a) | clause (b) | **verdict** |
|---|---|---|---|---|---|---|
| **1a** | **untaught meter, +1M** | **+1.44 pp** [−0.44, +3.19] | 3.69 (imported) | ❌ | ❌ | **WITHIN FLOOR at n = 2** |
| **1b** | **untaught meter, +3M** | **+5.38 pp** [+1.63, +9.19], **7 of 8 teams** | 3.69 (imported) | ✅ | ❌ | **WITHIN FLOOR at n = 2** |
| **1c** | **untaught meter, +6M** | **+5.13 pp** [+1.06, +9.00], 5 of 8 teams | 3.69 (imported) | ✅ | ❌ | **WITHIN FLOOR at n = 2** |
| **2a** | **per-slice, BIG-5 (balance), +6M** | 🚨 **−0.0612** [−0.1097, −0.0124] | **0.0475** (measured HERE, same cell) | ✅ | ❌ | **WITHIN FLOOR at n = 2** — and the sign is **NEGATIVE** |
| 2b | per-slice, Big-5, +3M | −0.0250 [−0.0737, +0.0239] | 0.0475 | ❌ | ❌ | **WITHIN FLOOR at n = 2** |
| **2c** | **per-slice, DDTAR (offense), +6M** | **+0.0675** [+0.0185, +0.1160] | **0.0850** (measured HERE, same cell) | ❌ | ❌ | **WITHIN FLOOR at n = 2** |
| 2d | per-slice, DDTar, +3M | **+0.1238** [+0.0749, +0.1717] | 0.0850 | ✅ | ❌ | **WITHIN FLOOR at n = 2** |
| **3** | **ladder at matched snapshot COUNT** | — | 25.2 Elo | — | — | 🚨 **NOT RUNNABLE — the fold promoted ZERO snapshots (n = 0)** |
| **4** | **Metamon `SmallRL` greedy, away, 100 games** | +0.040 [−0.097, +0.175] | 0.090 | ❌ | ❌ | **WITHIN FLOOR at n = 2** |
| 5 | collateral | *row 1 IS the collateral read* | — | — | — | `main.exploitability` **NOT APPLICABLE** (§5) |

### 🚨 THE BRANCH: **(d) NOT DETECTED at +6M, with the fold demonstrably unconverged**

Every registered clause, mechanically:

| branch | its condition | met? |
|---|---|---|
| **(a) THE FOLD PAID** | untaught clears the floor UPWARD at ≥ 2 checkpoints **AND** per-slice gains clear | ❌ — untaught fails clause (b) at all three; the slice gains clear nothing |
| **(b) LOCAL TRANSFER** | per-slice gains clear, untaught within floor | ❌ — **no per-slice gain clears**, and one slice is NEGATIVE |
| **(c) THE LEAK** | untaught DOWN past the floor | ❌ — untaught is **UP**, at 1.4× the floor at two of three checkpoints |
| **(d) NOT DETECTED** | nothing clears | ✅ |

**And "nothing clears" is a bad description of what these numbers do.** Three things happened that
the registration did not predict and that no single verdict word carries:

1. 🚨 **THE OFF-SLICE ROW IS THE ONE THAT MOVED.** The fold is **+5.4 / +5.1 pp above its frozen
   parent on the untaught 8** at +3M and +6M — 1.4× the imported floor, on 7 of 8 and 5 of 8 teams,
   with both CIs clear of ZERO. It fails only the clause that asks the CI to exclude the floor
   POINT.
2. 🚨 **THE ON-SLICE ROWS SPLIT IN SIGN, AND THE SPLIT IS THE OPPOSITE OF THE ONE REGISTERED.** On
   the **Big-5** team — the teacher whose gated share ROSE 0.226 → 0.305 — the fold ends
   **0.061 BELOW its parent** (CI clear of zero). On the **DDTar** team — the teacher whose share
   FELL 0.210 → 0.122 — it ends **0.068 ABOVE** (CI clear of zero). Registered prediction:
   *Big-5 > DDTar*, at P = 0.60. **It is Big-5 − DDTar = −0.129.**
3. 🚨 **THE FOLD PEAKED AT +3M AND FELL BY +6M ON BOTH TAUGHT TEAMS** (Big-5 0.4675 → 0.4313,
   DDTar 0.5938 → 0.5375) **while `teacher_agreement_on_slice` kept RISING** (0.758 → 0.796 → 0.815 on the
   registered grid, and 0.820 / 0.823 at the final rollout — §7.1). The completion entry's headline — *"agreement was
   still climbing, so the budget stopped the fold and not its stop rule: it had not finished
   learning from these teachers"* — is now measured against piloting, and **agreement rising did
   not mean piloting improving.**

### What this read does NOT license

* **It does not say the fold hurt, or helped, off-slice.** +5.1 pp is WITHIN an imported floor and
  WITHIN floor is never "equivalent" (rule 6) and never a detection either.
* **It does not establish any floor.** Every bar is `|arm W − W_b|` at n = 2 — one pair bounds,
  never estimates (rules 19/22) — and the untaught bar is additionally an IMPORT from a fresh-arm
  pair onto a fold.
* **It does not separate EXTRACTION from SENIORITY.** The fold carries 6M steps arm W does not.
* **It does not price the missing CONTINUATION CONTROL.** Every delta is against a FROZEN parent.
* **It says nothing about folds in general**, at another dose, with other teachers, or at 277M.
* 🚨 **It does not say the recipe failed.** The fold is unconverged by its own stop rule's own
  meter, and one arm at n = 1 against an imported floor is the weakest evidential configuration
  this programme has.

---

## 1. ROW 1 — THE UNTAUGHT METER (off-slice; this IS the collateral read)

`python -m main.untaught_meter`, the win rate of a checkpoint **piloting** the fixed untaught-8
slice against ONE fixed opponent, cluster-bootstrapped over TEAMS. Registry opponent
`untaught_meter_opponent` (= `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` @24,000,000),
the untaught-8 manifest in its canonical order, **200 games/team, `--seed 0`, concurrency 1, 8
single-concurrency shards, all five refs in ONE invocation** so every ref saw the identical 8 teams
and the identical games. **8,000 battles, 0 TIMEOUTS (0.0 %)** — rule 12's 25 % INCONCLUSIVE
threshold nowhere near.

### 1.1 ✅ THE REPRODUCTION CHECK, AND IT IS EXACT ON BOTH COMPARATORS

| ref | banked 2026-09-18 | **here** | Δ | per-team rows identical? |
|---|---:|---:|---:|---|
| arm W | 46.19 pp (739/1600) | **46.19 pp (739/1600)** | **0.00** | ✅ **all 8 identical** |
| W_b | 49.88 pp (798/1600) | **49.88 pp (798/1600)** | **0.00** | ✅ **all 8 identical** |

**This is the third independent confirmation that the meter is deterministic at seed 0 /
concurrency 1** (arm W reproduced its 2026-09-16 level on 2026-09-18; both arms reproduce their
2026-09-18 levels here), and the first time the check has been run on the **per-team** rows rather
than the pooled level. **The 3.69 pp floor is re-measured here on the identical games at exactly
−3.69 pp.** Nothing in this row is a fresh draw of the comparators; they are the same games.

### 1.2 The levels, and every team

| ref | level | wins / finished | `U_61590463` | `U_92832108` | `U_ce35b736` | `U_9909f2e9` | `U_9d5f8458` | `U_f7ba5702` | `U_90b94599` | `U_dbf81d8e` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **fold @ +1M** | **47.62** | 762 / 1600 | 50.00 | 43.50 | 53.00 | 40.00 | 51.50 | 44.50 | 47.00 | 51.50 |
| **fold @ +3M** | **51.56** | 825 / 1600 | 44.50 | 44.50 | 58.50 | 39.00 | 56.00 | 52.50 | **63.00** | 54.50 |
| **fold @ +6M** | **51.31** | 821 / 1600 | 48.50 | 37.50 | **60.50** | 45.50 | 56.00 | 54.50 | 60.50 | 47.50 |
| **arm W** (parent) | **46.19** | 739 / 1600 | 48.50 | 39.00 | 49.50 | 38.00 | 47.00 | 48.00 | 48.00 | 51.50 |
| **W_b** (other seed) | **49.88** | 798 / 1600 | 58.00 | 53.00 | 42.50 | 48.00 | 46.00 | 48.00 | 51.50 | 52.00 |

*(arm S `ai_v13_01_flywheel_shaped`, for scale only, is 54.50 pp — not a comparator of this read.)*

### 1.3 The paired, team-clustered contrasts (rule 10, 20,000 draws, ONE shared index set)

| contrast | Δ pp | CI95 | teams favouring first | floor | verdict |
|---|---:|---|---:|---:|---|
| **fold +1M − arm W** | **+1.44** | [−0.44, +3.19] | 5 of 8 | 3.69 | WITHIN FLOOR |
| **fold +3M − arm W** | **+5.38** | [+1.63, +9.19] | **7 of 8** | 3.69 | WITHIN FLOOR (clause a ✅, b ❌) |
| **fold +6M − arm W** | **+5.13** | [+1.06, +9.00] | 5 of 8 | 3.69 | WITHIN FLOOR (clause a ✅, b ❌) |
| fold +1M − W_b | −2.25 | [−6.44, +2.56] | 2 of 8 | — | — |
| fold +3M − W_b | +1.69 | [−5.44, +8.69] | 5 of 8 | — | — |
| fold +6M − W_b | +1.44 | [−5.75, +8.75] | 4 of 8 | — | — |
| **arm W − W_b — THE FLOOR, re-measured here** | **−3.69** | [−8.19, +0.81] | 2 of 8 | — | reproduces exactly |
| the fold's own +6M − +1M (within-arm) | +3.69 | [−0.75, +8.13] | 5 of 8 | — | — |

### 1.4 🚨 THE EARLY OFF-SLICE HOLE DID NOT HAPPEN

§2.3's strongest fold law is that **every gen-era fold digs an early off-slice hole, ~3–4 pp by
+1M, regardless of teacher content** — measured three independent times (funded −3.12, unfunded
−3.28, K=6 −4.19, "indistinguishable"). The pre-registered prediction here put P(Δ at +1M is
negative) at **0.70** on the strength of it.

**Δ at +1M is +1.44 pp, and the fold is above its parent at every checkpoint measured.** No hole
appears at +1M, at +3M or at +6M.

⚠️ **Three differences from the cells that established the law, any of which could account for it,
and this read separates none of them:** the dose is **0.20× the v8 reference and FROZEN** (those
cells ran frozen-dose too, but at ~1M depth on a 28M parent); the parent is at **75M** rather than
~28M and is a `--critic winprob` arm; and the teachers are **8M-step forks of the parent itself**,
about as near-parent as a teacher can be, which is the condition §2.3 says lets a student climb out
of the hole. **The law is not refuted — it is not reproduced in this regime, at n = 1.**

### 1.5 🚨 The clause that binds, and the arithmetic that says it always would

Both +3M and +6M pass clause (a) at 1.4× the floor with CIs clear of zero, and fail clause (b).
That is worth stating as a property of the BAR:

> The untaught meter's team-clustered CI on a contrast at this n has a half-width of **≈ 3.8 to
> 4.0 pp**. Clause (b) asks the CI to exclude the 3.69 pp floor point, so it can only pass when
> `|Δ| − 3.8 > 3.69`, i.e. **|Δ| ≳ 7.5 pp**. A real effect of exactly the floor's own size is
> structurally unable to clear this rule on 8 teams × 200 games.

The floor read recorded clause (b) as **near-VACUOUS** on its entropy row (an OLS se over
autocorrelated points, which always excludes). Here it is the opposite failure mode — **near-UNSATISFIABLE**
— and both are the same underlying fact: clause (b) compares a SAMPLING interval to a RUN-LEVEL
quantity. It is reported, not worked around: **the registered rule is the registered rule and its
verdict stands.** What follows from it is that a fold effect of ~5 pp off-slice is not resolvable by
this instrument at this n against this floor, and buying more GAMES will not fix it (rule 19: the
component that dominates here is RUN-level, and the honest lever is a second fold arm).

---

## 2. ROW 2 — PER-SLICE PILOTING: THE MATCHED-EXTRACTION ROW

Every ref pilots the SAME pinned taught team against the SAME fixed opponent
(`untaught_meter_opponent` @24,000,000 — **not** arm W's pool sentinels, because a sentinel is the
trainee's OWN snapshot and arm W's differ from W_b's, floor-read hazard F-G) drawing the SAME
800-long pool-team sequence under the SAME dice. **800 games per cell, 12 cells, 9,600 battles,
0 TIMEOUTS.** Team pins `450fb83c20` (Big-5) and `6212de2e8c` (DDTar), stamped by the tool.

Wilson per cell; **Newcombe on every difference — CONSERVATIVE here**, because the games are paired
under CRN and the tool retains no per-battle outcome vector, so the pairing cannot be exploited.

### 2.1 The BIG-5 slice (BALANCE) — `f6229d2c867e21d6`, teacher t1

| ref | win rate | Wilson 95 % | wins/800 |
|---|---:|---|---:|
| **t1 `ai_v13_05_exploit_big5starmie` — its OWN teacher** | **0.5763** | [0.5417, 0.6100] | 461 |
| W_b | 0.5400 | [0.5054, 0.5743] | 432 |
| **arm W — the PARENT** | **0.4925** | [0.4580, 0.5271] | 394 |
| fold @ +3M | 0.4675 | [0.4332, 0.5021] | 374 |
| **fold @ +6M** | **0.4313** | [0.3973, 0.4658] | 345 |
| t2 DDTar (a team it never trained on) | 0.2762 | [0.2464, 0.3082] | 221 |

| contrast | Δ | Newcombe 95 % | floor on this cell | verdict |
|---|---:|---|---:|---|
| **fold +6M − arm W** | 🚨 **−0.0612** | [−0.1097, −0.0124] | **0.0475** | WITHIN FLOOR (a ✅, b ❌) |
| fold +3M − arm W | −0.0250 | [−0.0737, +0.0239] | 0.0475 | WITHIN FLOOR |
| **arm W − W_b — THE SEED FLOOR, same cell** | **−0.0475** | [−0.0961, +0.0015] | — | — |
| **t1 − fold +6M — THE TEACHER'S CEILING** | **+0.1450** | [+0.0961, +0.1929] | — | the teacher is far above the fold |
| **t1 − arm W — the gap the fold was asked to close** | **+0.0838** | [+0.0349, +0.1321] | — | detected vs zero |

🚨 **On the slice the fold drew MORE of its taught signal from, it ends BELOW its parent, and it got
worse between +3M and +6M** (−0.025 → −0.061). The teacher is 0.1450 above it; the fold closed
**none** of the 0.0838 gap and moved backwards through it.

### 2.2 The DDTAR slice (OFFENSE) — `9eb3abdc52876a63`, teacher t2

| ref | win rate | Wilson 95 % | wins/800 |
|---|---:|---|---:|
| **fold @ +3M** | **0.5938** | [0.5593, 0.6273] | 475 |
| W_b | 0.5550 | [0.5204, 0.5891] | 444 |
| **fold @ +6M** | **0.5375** | [0.5029, 0.5718] | 430 |
| **t2 `ai_v13_06_exploit_ddtar_spikes` — its OWN teacher** | **0.4775** | [0.4431, 0.5121] | 382 |
| **arm W — the PARENT** | **0.4700** | [0.4356, 0.5046] | 376 |
| t1 Big-5 (a team it never trained on) | 0.3725 | [0.3397, 0.4065] | 298 |

| contrast | Δ | Newcombe 95 % | floor on this cell | verdict |
|---|---:|---|---:|---|
| **fold +6M − arm W** | **+0.0675** | [+0.0185, +0.1160] | **0.0850** | WITHIN FLOOR (a ❌) |
| **fold +3M − arm W** | **+0.1238** | [+0.0749, +0.1717] | 0.0850 | WITHIN FLOOR (a ✅, b ❌) |
| **arm W − W_b — THE SEED FLOOR, same cell** | **−0.0850** | [−0.1334, −0.0360] | — | — |
| 🚨 **t2 − fold +6M — THE TEACHER'S CEILING** | **−0.0600** | [−0.1086, −0.0110] | — | **the FOLD is ABOVE its own teacher** |
| **t2 − arm W — the gap the fold was asked to close** | **+0.0075** | [−0.0413, +0.0563] | — | **NOT DETECTED — there was almost no gap** |

### 2.3 🚨 THE DIVERGENCE, AND IT RUNS THE OPPOSITE WAY

| | Big-5 (t1) | DDTar (t2) |
|---|---:|---:|
| the teacher's **gated share** in the fold, +1M → +6M | **0.226 → 0.305** (rose) | **0.210 → 0.122** (fell) |
| the **teacher's** own edge over the parent on its team | **+0.0838** [+0.0349, +0.1321] | **+0.0075** [−0.0413, +0.0563] |
| the **fold's** edge over the parent on that team, +6M | 🚨 **−0.0612** | **+0.0675** |
| the fold vs its own teacher | −0.1450 (far below) | 🚨 **+0.0600 (above)** |

**The registered prediction was that the Big-5 slice would gain more, at P = 0.60, because its
teacher's gated share rose 2.5× relative to the other's. The measured difference is
Big-5 − DDTar = −0.1287, i.e. the gated share ANTI-PREDICTS the slice gain on the one pair of
slices available.**

⚠️ **This is TWO cells and is a DESCRIPTOR, not a law.** It has at least two readings this design
cannot separate, and both are worth having on the record:

* **The gate fires where the student is WRONG, not where it is learning.** A gated row is one the
  teacher's action disagrees with the trainee's; a slice the student is failing produces more of
  them. On that reading a rising gated share is a symptom, not a dose.
* **The two teachers are not equally worth copying.** t2's own edge over the parent on its own team
  against a third-party opponent is **+0.0075, NOT DETECTED** — there was almost nothing to teach —
  while t1's is **+0.0838 and detected**. The fold beat t2 and fell far short of t1, which is what
  "copy a teacher who is barely better than you" and "copy a teacher who is much better than you"
  respectively look like when the copying does not work.

### 2.4 🚨 THE 0.740 HEADLINE DOES NOT TRANSFER, AND THIS IS THE ROW THAT SAYS SO

Both exploiters reached **0.740 vs their own parent on their own pinned team** (ledger `deddfcd5`),
a symmetry the completion entry recorded as "the same endpoint by different routes". **Against a
THIRD-PARTY opponent on the same pinned teams, that symmetry is gone:** t1 pilots Big-5 at
**+0.0838 over the parent**; t2 pilots DDTar at **+0.0075, not detected**. A best-response edge
against ONE target is not piloting skill, and a fold that distils from a best-response teacher
inherits whichever of the two it actually is.

**And a specialist is badly OOD off its own team:** t2 on the Big-5 team reads **0.2762** (−0.216
vs the parent), t1 on the DDTar team **0.3725** (−0.098). Both are far below the generalist parent
they were forked from. The narrowing that buys the vs-target number costs this much off the pin.

### 2.5 The pooled row, SECONDARY — and it is a textbook rule-10 inversion

| ref | pooled over BOTH teams | Wilson 95 % |
|---|---:|---|
| W_b | 0.5475 | [0.5230, 0.5717] |
| fold @ +3M | 0.5306 | [0.5061, 0.5550] |
| **fold @ +6M** | **0.4844** | [0.4600, 0.5089] |
| **arm W** | **0.4813** | [0.4568, 0.5057] |
| t1 Big-5 | 0.4744 | [0.4500, 0.4989] |
| t2 DDTar | 0.3769 | [0.3535, 0.4009] |

🚨 **Pooled, the fold reads +0.003 over its parent — a nothing — while the two per-team rows are
−0.061 and +0.068, each with a CI clear of zero.** The pooling cancels a real split into an exact
null. This is rule 10 (the team is the unit) demonstrating itself on 1,600 games, and it is why the
per-team rows above are the result and this table is a footnote.
---

## 3. ROW 3 — THE LADDER: 🚨 **NOT RUNNABLE AT n = 0, AND THAT IS THE FINDING**

The GO asked for the ladder at matched snapshot COUNT against arm W and W_b, on the fold's **own**
nodes only, and warned it would likely be "too few for a verdict — report as descriptor if n < 12".

**It is not too few. It is none.**

| | `ai_v13_07_fold1` | arm W | W_b |
|---|---|---|---|
| `snapshot_ladder/` directory | **absent** | present | present |
| snapshots in the pool | 20 | 20 | 20 |
| snapshots the run **promoted itself** | **0** | 20 | 20 |
| the pool's actual contents | **arm W's own 26.0M → 72.0M snapshots, auto-seeded at fork** | its own | its own |
| committed `ladder.json` newest | — | 2019.1 @72.0M | 2044.3 @74.0M |

A `snapshot_ladder` is a Bradley-Terry fit over a run's OWN promoted snapshots. **The fold promoted
nothing in 6,094,848 steps, so there is no node to fit and no rating to quote in either direction.**
This was disclosed in `PREDICTION.md` §0 before any registered number existed, precisely so that the
absence could not later be dressed up as "the ladder did not separate them".

### 3.1 The descriptor underneath it — and it is the cleanest series in this read

The fold's own `eval/win_rate_vs_pool`, read from the TensorBoard events and **sliced at the fork
step** (⚠️ a fork INHERITS its parent's event files, so the unsliced series is arm W's history — the
same trap that made `g7_ladder` structurally wrong on a fork, ledger `36f8f7eb`):

| step | +steps | `win_rate_vs_pool` | `win_rate_vs_bots` | `eval/mean_ep_len_vs_bots` |
|---:|---:|---:|---:|---:|
| 76,000,032 | +1.0M | **0.488** | 0.9238 | 26.75 |
| 78,000,000 | +3.0M | **0.446** | 0.9288 | 26.86 |
| 80,000,016 | +5.0M | **0.412** | 0.9338 | 27.92 |
| *82,000,032* | *+6.9M (`ai_v13_08_fold1_cont`, LIVE)* | ***0.372*** | — | — |

🚨 **Because the fold promoted nothing, every one of those points is against an IDENTICAL, FROZEN
opponent set** — arm W's own 20 late snapshots, seeded at the fork and never added to. A within-run
series against a fixed pool is unusually clean, and this one falls **monotonically, 0.488 → 0.372
over 6.9M steps**, never once reaching the 0.55 promotion gate. The fourth point is the already-live
continuation `ai_v13_08_fold1_cont`, which seeded its pool from the fold — i.e. **the same 20 files
again**, verified here — so the series is continuous across the two runs.

🚨 **And the two eval rows dissociate.** Against the **nine pinned bots** the fold IMPROVES
monotonically (0.9238 → 0.9338) over exactly the span in which it falls against **its parent's own
snapshots** (0.488 → 0.412). Getting better against a fixed weak reference while getting worse
against the parent's recent self is the signature of a policy moving somewhere the pool does not
reward — it is a DESCRIPTOR here, with no bar attached and no claim made from it.

⚠️ **What this descriptor is not.** `win_rate_vs_pool` is the row carrying the 2026-09-07
opponent-regime boundary (§3.2 rule 5). It does not bite here — every point is post-boundary, the
regime is RECORDED (`eval_sentinel_greedy` inherited from arm W's argv), and the comparison is
within one run against one frozen pool — but the number is **not** comparable to any pre-boundary
`win_rate_vs_pool` in the ledger, and no cross-run use is made of it.

---

## 4. ROW 4 — THE EXTERNAL ANCHOR: Metamon `SmallRL`, greedy, AWAY team set

Through the tool of record, `python -m main.anchors`, which starts and stops its own Showdown server
(**:9450**, inside this job's 9400–9499 band), verifies the greedy regime **per decision** on both
sides, and writes the Wilson interval itself. `SmallRL` ckpt 40 (13.9M), `VanillaAttention`, Metamon
`@0a00a759`, CPU, 100 games as two role-balanced half-cells. Showdown pin `e0551883f`.

| arm | win rate | Wilson 95 % | half-cells (ours-challenge / peer-challenge) |
|---|---|---|---|
| arm S `ai_v13_01_flywheel_shaped` (banked) | 0.520 | [0.423, 0.615] | — |
| **arm W — the PARENT** (banked) | **0.500** | [0.404, 0.596] | — |
| W_b (banked) | 0.590 | [0.492, 0.681] | 0.54 / 0.64 |
| **`ai_v13_07_fold1` @ +6M** | **0.540** | **[0.443, 0.634]** | **0.58 / 0.50** |

| contrast | Δ | Newcombe 95 % | floor | clause (a) | clause (b) | verdict |
|---|---|---|---|---|---|---|
| **fold − arm W** | **+0.040** | [−0.097, +0.175] | **0.090** | ❌ | ❌ | **WITHIN FLOOR at n = 2** |
| fold − W_b | −0.050 | [−0.183, +0.086] | 0.090 | ❌ | ❌ | **WITHIN FLOOR at n = 2** |

**Instrument health, all green.** `status: OK`, 100/100 games recorded, **0 ties, 0 games at the
250-turn forfeit cap**, `regime_verified: true` with Metamon's `argmax_match_rate` **1.0000** over
2,732 decisions and our own `stochastic` kwarg `[false]`; `distinct_our_teams` 19 of 20 (the same as
W_b's); `model_loader: bare`, `model_rung: explicit_zip`, `team_source_asymmetry: false`. Role split
**8 pp** (0.58 / 0.50) against W_b's 26 pp on the HOME cell — unremarkable, and at n = 50 per half
nothing may be taken from it.

**This row was registered as DESCRIPTIVE and it behaved as registered.** At 100 games the cell
resolves ~±0.10 against a 0.090 run-level floor, so it could not have separated a fold from its
parent whatever the fold did. The prediction filed in advance was P(clears) = 0.15; it did not.

---

## 5. ROW 5 — COLLATERAL, and 🚨 why `main.exploitability` WAS NOT RUN

**The untaught meter IS the collateral read.** That is what row 1 measures — the fold piloting eight
teams no teacher in the fleet has ever trained on — and no second off-slice instrument was added.

**`python -m main.exploitability` is NOT APPLICABLE to this arm, and it was declared so in
`PREDICTION.md` before the first battle rather than discovered afterwards.** Its own `--help` is
unambiguous: *"Generation exploitability curve from fleet_admission-schema artifacts. Bookkeeping
only — no battles, no models."* It consumes admission JSONs in generation order, and the four
committed ones (`designs/research_state/measurements/admission_artifacts/`) are each the output of a
separate **fleet-admission battery**: for every (arm, team) cell it plays the arm's TEACHER on the
team, plus seed-paired games with a fixed REFERENCE model (`rev1final`) **and** with the arm's
TARGET, which is what makes `net` (teacher − reference) and `ordered` (teacher − target) mean
anything. `ai_v13_07_fold1` has produced no such artifact.

⚠️ **And it would not have been cheap.** Those batteries run 800 games per arm at 400 per team. Row 2
below is the same GENRE — pinned-team piloting, one harness, paired draws — but it is **not that
schema**: it carries the zero-head-start TARGET (arm W) and the seed floor (W_b), and it carries
**no fixed cross-era REFERENCE**, which is the term `net` needs. Manufacturing one is a separate job
and is not smuggled in here.
---

## 6. THE READING — what these four rows say together

### 6.1 The fold did not fail; it was not resolved, and its SHAPE is not the one the era predicted

Written as one paragraph, with every hedge in place:

> *Against its frozen parent, `ai_v13_07_fold1` is **+5.1 pp on the untaught 8** at +6M (1.4× an
> imported 3.69 pp floor, CI clear of zero, 5 of 8 teams), **+0.068 on the DDTar team** and
> **−0.061 on the Big-5 team** (each CI clear of zero, each inside its own cell's seed floor),
> **+0.040 on the `SmallRL` away anchor** (inside 0.090), and has **no ladder at all**. By the
> registered two-clause rule nothing clears, so the verdict is **NOT DETECTED**. What the numbers
> add that the verdict word does not is that the row which moved is the **OFF-SLICE** one, and the
> **ON-SLICE** rows split in sign — which is the inverse of the gen-era fold's signature.*

🚨 **THE INVERSION IS THE FINDING.** §2.3's gen-era account is: a fold **TEACHES** (~+5 pp
on-slice against the frozen parent, all arms clearing zero individually) and **DIGS A HOLE**
off-slice (~3–4 pp by +1M, every teacher content). **This fold does neither.** It has no hole at
any depth, its off-slice level is its best row, and on-slice it is negative on one of its two
taught teams. ⚠️ **It is one arm, at n = 1, at a dose and a parent depth no gen-era cell occupied**
— so this is a **failure to reproduce a law in a new regime**, not a refutation of it, and §2.3
should not be edited from here.

### 6.2 The three descriptors that point the same way

None of these is a registered row and none carries a bar. They are listed together because they
agree, and because agreeing is the only thing three descriptors can do:

| descriptor | what it says |
|---|---|
| `win_rate_vs_pool` **0.488 → 0.446 → 0.412 → 0.372** against a FROZEN pool (§3.1) | the fold moves steadily away from what arm W's own recent snapshots beat |
| the fold **peaks at +3M and falls by +6M on BOTH taught teams** (0.4675 → 0.4313; 0.5938 → 0.5375) and is flat off-slice (51.56 → 51.31) | whatever the second half of the budget bought, it was not piloting |
| `collateral_kl_vs_parent` **0.277 → 0.365 → 0.381 → 0.402** and `on_slice_kl` **0.327 → 0.519 → 0.653 → 0.690**, both still rising at the last rollout | the policy is still moving, on and off the slice, at the moment the budget ended |

🚨 **Put beside `teacher_agreement_on_slice` **0.758 → 0.796 → 0.815 → 0.823**, still rising, these
say that AGREEMENT AND PILOTING CAME APART between +3M and +6M.** The fold agreed with its teachers
more and piloted their teams worse. **The registered stop rule's first conjunct — "agreement has
PLATEAUED" — is therefore not a proxy for "there is still something to learn", and on this arm it is
the conjunct that kept the stop signal at 0.0000 for the whole run.** Recorded as an instrument
observation; changing the rule is not this read's call.

### 6.3 What the continuation `ai_v13_08_fold1_cont` would decide — as registered, before the result

`PREDICTION.md` §4 fixed this in advance so the answer could not be fitted:

> *It is decisive for exactly one question — did the BUDGET or the RECIPE stop the fold? — and only
> in one direction: if the per-slice gain grows and the untaught row does not fall further, the +6M
> budget was the binding constraint; if the per-slice gain is flat at +12M while
> `teacher_agreement_on_slice` keeps rising, then agreement is not the quantity that predicts
> piloting and the stop rule is measuring the wrong thing.*

**The +3M → +6M leg has already run that experiment inside this arm and returned the second
answer**: agreement rose, both slices fell. The continuation (live since 2026-09-19 01:28 PT,
forked at 81,100,800, **pool seeded from the fold — i.e. the same 20 arm-W snapshots again**,
verified here) therefore tests whether that leg was a draw or a direction. Its first post-fork eval
cycle reads `win_rate_vs_pool` **0.372** at 82,000,032, continuing the decline.

**The cheap, decisive read on it when it lands** — and it is cheap, ~4 h CPU beside a GPU arm — is
exactly this read's rows 1 and 2 at +12M: if the Big-5 slice keeps falling and the untaught row
holds near +5 pp, the recipe is moving competence OFF the taught slice and the teacher choice is
the lever; if the Big-5 slice turns up, the +6M budget was the constraint after all and the
+3M → +6M dip was a draw.

⚠️ **What the continuation CANNOT decide:** whether any of this clears a floor. That needs a second
fold ARM (rule 22 — a lever may change the variance it is read against, and a control-replicate
floor does not bound a lever's), not more steps on this one.

### 6.4 The one thing this read would change about how the next fold is instrumented

**Not a recommendation about the recipe — an observation about what was measurable.** The fold's
only continuous competence signal during the run was `win_rate_vs_pool` against a pool that, because
nothing was promoted, never changed. It fell monotonically from the first cycle and no gate,
tripwire or stop rule reads it. The registered distill meters — agreement, gate rate, collateral KL
— all moved in the direction that reads as "working". **The row that would have said "the taught
slice is getting worse" is the per-slice piloting row, and it exists only offline, after the run.**
---

## 7. PROVENANCE — the five arms and the two teachers, side by side

| | **arm W** (PARENT/CONTROL) | **W_b** (the floor arm) | **`ai_v13_07_fold1`** | **t1 Big-5** | **t2 DDTar** |
|---|---|---|---|---|---|
| run | `ai_v13_02_flywheel_winprob` | `ai_v13_04_flywheel_winprob_b` | `ai_v13_07_fold1` | `ai_v13_05_exploit_big5starmie` | `ai_v13_06_exploit_ddtar_spikes` |
| `lineage.role` | `fresh` | `fresh` | **`fold`** | fork/exploiter | fork/exploiter |
| forked from | — | — | arm W @75,005,952 | arm W @75,005,952 | arm W @75,005,952 |
| steps | 75,005,952 | 75,005,952 | **81,100,800** (+6,094,848) | 83,066,880 (+8,060,928) | 83,066,880 (+8,060,928) |
| pin | `6eb9c776` | `6eb9c776` | `6eb9c776`, **one row** | `6eb9c776` | `6eb9c776` |
| config / arch | 119 / `gen3_critic_route_wave_v1` | 119 / same | 119 / same | 119 / same | 119 / same |
| `--seed` | 1001 | 1002 | 1001 (inherited) | 1001 | 1001 |
| ladder nodes | 20 | 20 | **0** | — | — |
| final `eval/win_rate_vs_bots` | 0.9100 @74.0M | 0.9212 | **0.9338 @80.0M** | 0.9712–0.9887 | 0.9638 |
| realized dose (`main.dose`) | 4.578e-08 | 5.035e-08 | **4.272e-09** (`[FROZEN; pinned 2.80e-05]`) | 3.815e-08 | 3.815e-08 |

*(arm W's and the fold's bot rows are the same instrument read from each run's own events; W_b's and
the two exploiters' are quoted from the floor read and the completion entries. The fold's, arm W's
and W_b's realized doses come from `python -m main.dose`, re-read here for the fold.)*

The fold's `model_config.json` differs from arm W's, W_b's and both teachers' **in exactly one
field** — `distill_target`, `"kl"` → `"action"` — verified here by diff. Everything
architecture-bearing is identical, which is the fact hazard **D-M** rests on.

### 7.1 The fold's own distill meters, REPRODUCED here from the events

The completion entry (`0158dff4`) banked these at +1M/+3M/+6M; this read recomputes them from the
same TensorBoard events, sliced post-fork, as a check that it is reading the run the ledger
described. It is, and the final post-fork points add two things the three-point
table could not show:

| row | +1M | +3M | +6M (banked) | **the FINAL post-fork points read here** |
|---|---|---|---|---|
| `teacher_agreement_on_slice` | 0.7581 | 0.7964 | 0.8153 | 0.8179 → 0.8153 → **0.8200 / 0.8232** |
| `on_slice_kl` | 0.3272 | 0.5185 | 0.6534 | **0.6903** |
| `kl` (overall) | 0.6621 | 0.5400 | 0.4896 | **0.4928** |
| `collateral_kl_vs_parent` | 0.2773 | 0.3647 | 0.3811 | 0.3770 → 0.3811 → **0.3855 / 0.4019** |
| `t1_gated_frac` (Big-5) | 0.2257 | 0.2918 | 0.3051 | 0.2749 |
| `t2_gated_frac` (DDTar) | 0.2101 | 0.1716 | 0.1222 | 0.1337 |
| `stop_signal` | 0.0000 | 0.0000 | 0.0000 | **0.0000 — never fired, at ANY of the 61 post-fork points** |
| `n_teachers_active` | 2.0 | 2.0 | 2.0 | **2.0 at every point** |
| `train/learning_rate` | 2.8e-05 | 2.8e-05 | 2.8e-05 | **2.8e-05, flat on all 59 post-fork points** |

✅ **Every banked value reproduces**, and the flat `2.8e-05` confirms the completion entry's warning
that the launcher's `▶️ Resuming at LR 2.50e-04` banner reports the CHECKPOINT's lr and not the
operating value.

⚠️ **TWO DISTINCT VALUES SIT AT STEP 81,100,800**, one from each of the run's last two child event
files (`tb/` holds three: an inherited placeholder plus one per child after the two restarts). They
are not a contradiction — they are the last rollout as logged by two writers — but "the last point"
is not a single number and is printed here as a pair rather than picked. **On the coarse
+1M/+3M/+6M grid agreement RISES monotonically (0.758 → 0.796 → 0.815); on the fine grid its last
four points WOBBLE (0.818, 0.815, 0.820/0.823).** "Still rising at the end" is a statement about the
coarse grid, and it is the grid the stop rule's plateau test reads.

🚨 **`collateral_kl_vs_parent` is rising at the end and so is `teacher_agreement_on_slice`** — which
is exactly the state in which the registered stop condition (agreement PLATEAUED *and* collateral
RISING) cannot fire, however large the collateral term gets. **The stop rule is conjunctive, and the
conjunct that never became true is the one about the TEACHER, not the one about the parent.** That
is a property of the rule's shape, recorded as an instrument observation rather than as a criticism
of the run.

---

## 8. WHAT WAS NOT DONE, AND WHY

* **The `--config auto` resolution was NOT re-run** (registration §3.1 asked for both). The
  registry config and each model's own config differ in `config_version` (101 vs 119) but carry the
  **same `arch_signature`** (`gen3_critic_route_wave_v1`), and `load_foreign_opponent` uses
  `config_path` only for the observation-family compatibility CHECK — the weights come from the zip.
  The floor read measured **three** 75M arms of this exact config under BOTH resolutions and got
  levels identical to 0.01 pp on every one (arm S 54.50/54.50, arm W 46.19/46.19, W_b 49.88/49.88).
  A second 8,000-battle invocation would return identical rows. **NOT RUN, on budget, and declared.**
* **No continuation control.** `--control` wants a plain +6M continuation of arm W with no teacher,
  no distillation term and no stable opponents; none exists and building one is ~6 GPU-h. Every
  delta here is therefore against a **FROZEN** parent (§6.1 prints the consequence beside it).
* **No second seed of the fold.** One arm; rules 19/22 make every verdict here a candidate.
* **No ladder refit** — there is nothing to fit (§3).
* **`main.exploitability` was not run** — it is bookkeeping over an artifact this arm does not have
  (§5).
* **No Foul Play cell, no Metamon HOME cell, no critic/calibration frame.** None is a registered
  row of this GO, and the offline critic frame in particular would need two full-capture
  `eval_traces` draws the fold's tree does not carry at a matched step.
* **No code under `src/` was changed. Nothing was written under `models/`.**
---

## 9. HAZARDS — every one a finding

| # | hazard | why it matters, and what was done |
|---|---|---|
| **D-A** | 🚨 **CLAUSE (b) IS NEAR-UNSATISFIABLE ON EVERY ROW IN THIS READ, AND THAT IS A PROPERTY OF THE BAR.** The untaught contrast's team-clustered CI has a half-width of ≈ 3.8–4.0 pp against a 3.69 pp floor, so clause (b) can only pass at \|Δ\| ≳ 7.5 pp; the per-slice Newcombe half-widths are ≈ 0.049 against cell floors of 0.0475 and 0.0850. **An effect the size of the floor cannot clear the rule at this n, whatever it is.** | The registered rule STANDS and every verdict here is WITHIN FLOOR by it. It is reported as a bar property rather than worked around, with the arithmetic printed (§1.5). The floor read logged clause (b) as near-VACUOUS on its entropy row; this is the opposite failure mode of the same fact — **clause (b) compares a SAMPLING interval to a RUN-LEVEL quantity.** The honest lever is a second fold ARM (rules 19/22), not more games. |
| **D-B** | 🚨 **THE TEACHER'S GATED SHARE ANTI-PREDICTS THE SLICE GAIN.** The share rose 0.226 → 0.305 on Big-5, where the fold ends **−0.061** against the parent; it fell 0.210 → 0.122 on DDTar, where the fold ends **+0.068**. The registered prediction was the opposite, at P = 0.60. | Reported as a DESCRIPTOR on two cells with both candidate readings stated and neither adopted (§2.3): the gate may fire where the student is WRONG rather than where it is learning, and the two teachers may simply not be equally worth copying. **No claim is made that gated share predicts anything, in either direction.** |
| **D-C** | 🚨 **AGREEMENT AND PILOTING CAME APART BETWEEN +3M AND +6M.** `teacher_agreement_on_slice` rose 0.796 → 0.815 → 0.823 while the fold's piloting fell on BOTH taught teams (Big-5 0.4675 → 0.4313, DDTar 0.5938 → 0.5375). The stop rule's first conjunct is "agreement has PLATEAUED", and it is the conjunct that held `stop_signal` at 0.0000 for the entire run. | Recorded as an instrument observation with the four-point series reproduced from the events (§7.1). **Changing the stop rule is not this read's call**, and the completion entry's reading — "it had not finished learning from these teachers" — is left standing as what the meter said, beside what the piloting row says. |
| **D-D** | 🚨 **THE LADDER ROW IS n = 0, NOT "TOO FEW".** The fold promoted zero snapshots in 6,094,848 steps; there is no `snapshot_ladder/` and the pool is arm W's 20 seeded files. | Disclosed in `PREDICTION.md` §0 **before any registered number existed**, precisely so the absence could not be presented afterwards as "the ladder did not separate them". Reported as a structural finding with the frozen-pool `win_rate_vs_pool` series as its descriptor (§3). |
| **D-E** | 🚨 **THE TWO EXPLOITERS' IDENTICAL 0.740 vs-TARGET HEADLINE DOES NOT SURVIVE A THIRD-PARTY OPPONENT.** On its own pinned team against `untaught_meter_opponent`, t1 Big-5 reads **+0.0838 [+0.0349, +0.1321]** over the parent and t2 DDTar **+0.0075 [−0.0413, +0.0563], NOT DETECTED**. | Reported in §2.4. A best-response edge against ONE target is not piloting skill; a fold distilling from such a teacher inherits whichever of the two the teacher actually is, and the teacher-admission question is not answered by a vs-target number. **Two teachers is n = 2 and this is a description.** |
| **D-F** | ⚠️ **A SPECIALIST IS BADLY OUT OF DISTRIBUTION OFF ITS OWN TEAM.** t2 pilots the Big-5 team at **0.2762** (−0.216 vs the parent); t1 pilots the DDTar team at **0.3725** (−0.098). Both are far below the generalist they were forked from. | A descriptor from the cells the design already required. It is the cost side of the narrowing whose benefit `exploiter_discrimination_2026-09-18` measured, on a different instrument, and it is recorded rather than developed. |
| **D-G** | ⚠️ **POOLING THE TWO SLICES PRODUCES AN EXACT NULL OUT OF A REAL SPLIT.** Pooled, fold +6M − arm W is **+0.003**; per team it is **−0.061** and **+0.068**, each with a CI clear of zero. | Rule 10 demonstrating itself on 1,600 games. The per-team rows are the result; the pooled table is printed as a footnote and labelled SECONDARY (§2.5). |
| **D-H** | ⚠️ **EVERY FLOOR IN THIS READ IS AN IMPORT FROM A 75M FRESH-ARM SEED PAIR ONTO A FOLD**, and a floor is a property of the DEPTH and the REGIME (§3.3). The meter's own FOLD floors are 1.19 / 1.66 / 2.46 / 4.27 / 1.00 pp, at ~1M depths on a 28M parent. | 3.69 pp sits between the frozen-dose 1.66 and controller-live 4.27 fold floors, which is why the GO chose it. It is labelled an IMPORT in `PREDICTION.md`, in §0 and in every row it bars. The per-slice floors are the exception — **they were MEASURED here, on the same cell, same opponent, same 800 games** (0.0475 and 0.0850). |
| **D-I** | ⚠️ **NO CONTINUATION CONTROL, so every delta is against a FROZEN parent** and credits the fold with whatever ordinary progress arm W would have made anyway (ledger 2026-09-06 cell 2: +3.45 pp on v8's parent). | §2.2 says the correction is NOT significant on OUR parents (three draws, two gen-era parents) — but that was measured at ~28M parent depth and ~1M fold depth, and this is 75M and +6M. It is an IMPORT and a standing limitation, printed beside every delta rather than appended. Building the control is ~6 GPU-h and is not proposed here. |
| **D-J** | ⚠️ **SENIORITY IS NOT SEPARABLE FROM EXTRACTION.** `fold − W = seniority + extraction`; the fold carries +6,094,848 steps arm W does not, and each teacher +8,060,928. | The matched-extraction standard's own remedy (a third arm at the fork's own starting point) is not available. The nearest bound in hand is the untaught row — the same 6M of ordinary progress read OFF the slice — and it is **+5.1 pp**, i.e. of the same order as the on-slice movements, which is exactly why the on-slice numbers cannot be read as extraction. Stated in §2 and in `PREDICTION.md` §6. |
| **D-K** | ⚠️ **ARM W IS THE LOWER OF THE TWO AVAILABLE PARENT SEEDS ON ALL THREE BATTLE INSTRUMENTS HERE** — untaught 46.19 vs W_b's 49.88; Big-5 0.4925 vs 0.5400; DDTar 0.4700 vs 0.5550. | So every "the fold beats its parent" number is measured against the weaker draw. Against W_b the fold's off-slice edge is **+1.44 pp**, not +5.13. Both contrasts are printed in §1.3 and the defensible statement is *"the fold is ahead of BOTH win-prob seeds off-slice, by 1.4 to 5.1 pp, against a 3.7 pp seed floor"* — which is the same shape, and the same caveat, the floor read attached to arm S's +8.31. |
| **D-L** | ⚠️ **A FORK'S TENSORBOARD DIRECTORY CARRIES ITS PARENT'S WHOLE HISTORY.** `models/ai_v13_07_fold1/tb` starts at 196,608 — arm W's first rollout. | Every series in §3.1 and §7.1 is SLICED at the fork step by `scripts/fold_run_rows.py`, and the slice is stated in the script's docstring. This is the same inheritance trap that made `g7_ladder` structurally wrong on a fork (ledger `36f8f7eb`), and it applies to every inherited row on any fork. |
| **D-M** | ⚠️ **THE `--config auto` RESOLUTION WAS NOT RE-RUN.** Registration §3.1 asked for both. | The registry config and each model's own differ in `config_version` (101 vs 119) but carry the **same `arch_signature`**, and `config_path` drives only the observation-family CHECK in `load_foreign_opponent` — the weights come from the zip. The floor read measured three 75M arms of this config under both resolutions and got levels identical to 0.01 pp on all three. A second 8,000-battle invocation would return identical rows. **NOT RUN on budget, and declared here rather than omitted.** |
| **D-N** | ⚠️ **A RESTARTED RUN's TENSORBOARD CARRIES TWO DISTINCT VALUES AT ITS FINAL STEP** — 81,100,800 appears twice in `teacher_agreement_on_slice` (0.8200 and 0.8232) and in every other per-rollout distill row, one from each of the last two child event files. | "The last point" is therefore not a single number. Every such row in §7.1 prints the PAIR rather than picking one, and the coarse +1M/+3M/+6M grid — the grid the banked entry and the stop rule's plateau test both read — is reported separately from the fine one, whose last four points wobble rather than rise. |
| **(context)** | The box carried the **live GPU arm `ai_v13_08_fold1_cont`** throughout, plus up to 14 concurrent CPU battle workers; load average ran **7 to 42**. | Everything here ran `CUDA_VISIBLE_DEVICES=""`, `nice 15`, `OMP_NUM_THREADS=1` for the anchor peer, from the MAIN checkout. **The GPU arm was never touched**; no process this read did not start was signalled; :8000 and :8001 were never touched and the only server started was `main.anchors`' own on :9450, which it stopped itself. **17,600 battles, 0 timeouts (0.0 %)** — no row is contention-sensitive by construction (none is a width meter, rule 23), and the meters are deterministic at seed 0 / concurrency 1, which the exact reproduction of both comparators (§1.1) demonstrates under exactly this load. |
---

## 10. WHAT IS IN THIS DIRECTORY

| path | what |
|---|---|
| `PREDICTION.md` | the pre-registration — bars, branches, priors and the two disclosed pre-look facts, committed before any registered number here existed |
| `README.md` | this note |
| `scripts/fold_untaught_delta.py` | row 1 — the paired team-clustered contrasts off the meter's own per-team rows at three depths, with the two reproduction checks. The paired procedure, the 20,000 draws and the seed are VERBATIM from the floor read's `floor_untaught_delta.py` |
| `scripts/fold_slice_read.py` | row 2 — the matched-extraction row: Wilson per cell, Newcombe on every difference, the seed floor measured on the SAME cell, the teacher's ceiling, the divergence, and the pooled row labelled SECONDARY |
| `scripts/fold_away_anchor.py` | row 4 — the `main.anchors` away cell joined to arm W and W_b with the SOP's own floors printed beside the measured one |
| `scripts/fold_run_rows.py` | row 3 — the fold's own TB rows POST-FORK ONLY (a fork inherits its parent's events), the structural ladder finding, and the `win_rate_vs_pool` descriptor |
| `scripts/harness/run_untaught.sh` | the untaught invocation as executed — all five refs in ONE call, 8 workers, seed 0 |
| `scripts/harness/run_slice.sh` | the per-slice invocation as executed — three ref-shards × 2 workers on the 2-team manifest, 800 games/cell |
| `scripts/harness/taught_slice_teams.json` | the TAUGHT-SLICE manifest: the two teams, in teacher order (the order is the seed offset — re-ordering it changes every game) |
| `scripts/harness/run_away_cell.sh` | the away cell as executed, on :9450 |
| `out/fold_untaught_delta.json` | every untaught level, per-team row, contrast and verdict |
| `out/fold_slice_read.json` | every per-slice cell, Wilson/Newcombe interval, floor and verdict |
| `out/fold_away_anchor.json` | the away cell with its regime instruments |
| `out/fold_run_rows.json` | the post-fork TB series, the distill-meter reproduction and the ladder structural record |
| `out/untaught/`, `out/slice/`, `out/anchors/` | the tools' own raw artifacts, unedited |
---

## 11. Ledger paragraph — ready to append (nothing in `ledger.md`, `UNDERSTANDING.md` or any design note was edited from here)

> ### 2026-09-19 · MEASUREMENT (MAJOR) · **THE ERA-1 FOLD's REGISTERED READ — branch (d) NOT DETECTED by the registered rule, and the SHAPE is the INVERSE of the gen-era fold**: the row that moved is the OFF-SLICE one (untaught **+5.4 / +5.1 pp** over the parent at +3M/+6M, 1.4× an imported 3.69 pp floor, CIs clear of zero) while the two TAUGHT slices split in SIGN (**−0.061** on Big-5, **+0.068** on DDTar) — 🚨 **no early off-slice HOLE at any depth**, 🚨 **the teacher's GATED SHARE anti-predicts the slice gain**, 🚨 **agreement and piloting came APART between +3M and +6M**, and 🚨 **the fold promoted ZERO snapshots, so the ladder row is n = 0 and `win_rate_vs_pool` fell 0.488 → 0.372 against a FROZEN pool**
>
> `designs/research_state/measurements/fold1_read_2026-09-19/`. Bars, branches and priors
> pre-registered in `PREDICTION.md` and committed BEFORE any registered number existed — including
> the two facts that were LOOKED AT while scoping and disclosed there rather than hidden (the
> missing ladder, and the `win_rate_vs_pool` series that explains it). **THE ARM.**
> `ai_v13_07_fold1`, a FORK of `ai_v13_02_flywheel_winprob` (**arm W**) at 75,005,952 distilling
> from the era's two win-prob exploiters — **t1 `ai_v13_05_exploit_big5starmie`** (BALANCE, pinned
> team `f6229d2c867e21d6`) and **t2 `ai_v13_06_exploit_ddtar_spikes`** (OFFENSE,
> `9eb3abdc52876a63`) — `--fork-lr 2.8e-5 --fork-lr-freeze`, `--distill-team-bias 0.4`, both
> teachers in the stable pool each piloting its own pin, +6,094,848 steps to **81,100,800**, pin
> `6eb9c776`, config v119, realized dose **4.272e-9 = 0.20× the v8 reference** (`main.dose`,
> re-read here). **THE RULE, fixed in advance and copied from the floor read so the two are
> commensurable: OUTSIDE THE FLOOR iff `|Δ| > floor` AND the Δ's CI excludes the floor POINT; else
> WITHIN FLOOR at n = 2 — one pair BOUNDS a floor and does not estimate one (rules 19/22), and
> WITHIN FLOOR is never "equivalent" (rule 6).** Sign convention `Δ = fold − arm W`. Everything
> CPU-only (`CUDA_VISIBLE_DEVICES=""`, `nice 15`), from the MAIN checkout, **nothing written under
> `models/`**, no file under `src/` changed; the only server was `main.anchors`' own on :9450,
> **:8000 and :8001 never touched**, and the live GPU arm `ai_v13_08_fold1_cont` never touched.
> **17,600 battles, 0 TIMEOUTS (0.0 %).** **ROW 1 — THE UNTAUGHT METER (and it IS the collateral
> read).** Registry opponent `untaught_meter_opponent` (= `ai_v9_29_rev1_0823@24,000,000`),
> untaught-8 in canonical order, 200 games/team, seed 0, concurrency 1, **all five refs in ONE
> invocation**. ✅ **arm W and W_b BOTH reproduce their banked 2026-09-18 levels EXACTLY — 46.19 pp
> (739/1600) and 49.88 pp (798/1600), Δ 0.00, and for the first time the check is on the PER-TEAM
> rows: all 8 identical on both arms** — a third independent confirmation that the meter is
> deterministic at seed 0 / concurrency 1, and the 3.69 pp floor re-measures at exactly −3.69 on the
> identical games. Fold levels **47.62 / 51.56 / 51.31 pp** at +1M / +3M / +6M ⇒ paired
> team-clustered (rule 10, 20,000 draws, one shared index set) **+1.44 [−0.44, +3.19] (5/8)**,
> **+5.38 [+1.63, +9.19] (7/8)**, **+5.13 [+1.06, +9.00] (5/8)**. Against the imported 3.69 pp
> floor: clause (a) passes at +3M and +6M at 1.4×, **clause (b) fails at all three ⇒ WITHIN FLOOR
> at n = 2 everywhere**, though both later CIs exclude ZERO. Against the OTHER seed the same
> contrasts are −2.25 / +1.69 / +1.44, so the defensible statement is *the fold is ahead of BOTH
> win-prob seeds off-slice by 1.4 to 5.1 pp against a 3.7 pp seed floor* — the same shape, and the
> same caveat, the floor read attached to arm S's +8.31. 🚨 **AND THE EARLY OFF-SLICE HOLE DID NOT
> HAPPEN.** §2.3's strongest fold law — every gen-era fold digs a ~3–4 pp hole by +1M regardless of
> teacher content, measured three times (−3.12 / −3.28 / −4.19, "indistinguishable") — puts the
> registered P(Δ at +1M negative) at 0.70. **It is +1.44, and the fold is above its parent at every
> depth measured.** Three differences from those cells could account for it and this read separates
> none: dose 0.20× and FROZEN, parent at 75M rather than ~28M and `--critic winprob`, and teachers
> that are 8M-step forks of the parent itself. **A failure to reproduce a law in a new regime at
> n = 1, NOT a refutation.** 🚨 **CLAUSE (b) IS NEAR-UNSATISFIABLE HERE AND THAT IS A BAR PROPERTY:**
> the team-clustered CI half-width is ≈ 3.8–4.0 pp against a 3.69 pp floor, so clause (b) needs
> `|Δ| ≳ 7.5 pp` — an effect the size of the floor cannot clear the rule at this n. The floor read
> logged clause (b) as near-VACUOUS on its entropy row; this is the opposite failure mode of the
> same fact, **clause (b) compares a SAMPLING interval to a RUN-LEVEL quantity**, and the lever is a
> second fold ARM (rules 19/22), never more games. **ROW 2 — PER-SLICE PILOTING, the
> matched-extraction row.** Same engine on a 2-team manifest, **800 games per cell, 12 cells, 9,600
> battles, 0 timeouts**, every ref piloting the SAME pinned team against the SAME fixed opponent
> drawing the SAME sequence under the SAME dice — the untaught meter's opponent and **NOT** arm W's
> pool sentinels, because a sentinel is the trainee's own snapshot and arm W's differ from W_b's
> (floor-read hazard F-G). Wilson per cell, **Newcombe on every difference, CONSERVATIVE under CRN**.
> 🚨 **THE TWO SLICES SPLIT IN SIGN.** **BIG-5**: t1 **0.5763**, W_b 0.5400, **arm W 0.4925**, fold
> +3M 0.4675, **fold +6M 0.4313**, t2 (off its pin) 0.2762 ⇒ **fold − W = −0.0612 [−0.1097, −0.0124]**
> against a **seed floor of 0.0475 measured on this very cell** — clause (a) ✅, (b) ❌, **WITHIN
> FLOOR, and NEGATIVE**; the teacher's ceiling is **+0.1450** above the fold and the gap the fold was
> asked to close (**t1 − arm W = +0.0838 [+0.0349, +0.1321]**) was not closed but crossed backwards.
> **DDTAR**: fold +3M **0.5938**, W_b 0.5550, **fold +6M 0.5375**, t2 **0.4775**, **arm W 0.4700**,
> t1 (off its pin) 0.3725 ⇒ **fold − W = +0.0675 [+0.0185, +0.1160]** against a **0.0850 seed floor
> on the same cell** — clause (a) ❌ ⇒ WITHIN FLOOR; at +3M it is +0.1238 [+0.0749, +0.1717], (a) ✅
> (b) ❌. 🚨 **AND THE FOLD BEATS ITS OWN TEACHER ON THAT SLICE (+0.0600 [+0.0110, +0.1086])** —
> because **t2's own edge over the parent on its own team is +0.0075 [−0.0413, +0.0563], NOT
> DETECTED**: there was almost nothing to teach. 🚨 **THE DIVERGENCE RUNS THE OPPOSITE WAY TO THE
> REGISTRATION.** The gated share rose 0.226 → 0.305 on Big-5 (where the fold ends −0.061) and fell
> 0.210 → 0.122 on DDTar (where it ends +0.068): **Big-5 − DDTar = −0.1287** against a registered
> P = 0.60 that Big-5 would gain more. **The gated share ANTI-PREDICTS the slice gain on the one
> pair of slices in hand** — a DESCRIPTOR on two cells, with both readings stated and neither
> adopted (the gate may fire where the student is WRONG rather than where it is learning; or the two
> teachers are simply not equally worth copying). 🚨 **THE 0.740 HEADLINE DOES NOT TRANSFER:** both
> exploiters reached 0.740 vs their own parent on their own pin, but against a THIRD-PARTY opponent
> on the same pins t1 is +0.0838 (detected) and t2 is +0.0075 (not) — **a best-response edge against
> ONE target is not piloting skill**, and a fold inherits whichever of the two its teacher actually
> is. ⚠️ **A specialist is badly OOD off its own team** (t2 on Big-5 0.2762, −0.216 vs the parent;
> t1 on DDTar 0.3725, −0.098), both far below the generalist they were forked from. ⚠️ **Pooled over
> both teams the fold reads +0.003 over its parent — an exact null manufactured out of a −0.061 /
> +0.068 split, each CI clear of zero.** Rule 10 demonstrating itself on 1,600 games; the pooled row
> is a footnote. **ROW 3 — THE LADDER IS n = 0.** 🚨 **The fold has NO `snapshot_ladder/` and
> promoted ZERO snapshots in 6,094,848 steps**; its `snapshots/` holds exactly the 20 files
> auto-seeded from arm W (26.0M → 72.0M, arm W's own mtimes). This is not "below the n ≥ 12 report
> floor" — there is no node to fit and no rating is quoted in either direction. **The reason is in
> the run's own eval rows: `win_rate_vs_pool` 0.488 → 0.446 → 0.412 (and 0.372 on the live
> continuation, whose pool is seeded from the fold, i.e. the SAME 20 files — verified), never near
> the 0.55 gate.** Because nothing was promoted, every point is against an IDENTICAL FROZEN opponent
> set, which makes the monotone fall unusually clean — 🚨 **and it dissociates from
> `win_rate_vs_bots`, which RISES 0.9238 → 0.9338 over the same span.** DESCRIPTOR, no bar attached,
> and not comparable across the 2026-09-07 regime boundary (every point is post-boundary and the
> regime is RECORDED). **ROW 4 — THE EXTERNAL ANCHOR.** `python -m main.anchors --opponent
> metamon:SmallRL --regime greedy --teamset away --games 100` on :9450 (its own server, started and
> stopped by the tool), `SmallRL` ckpt 40 / `VanillaAttention` / Metamon `@0a00a759`, Showdown pin
> `e0551883f`: the fold reads **0.540 [0.443, 0.634]** against arm W's **0.500** and W_b's 0.590 ⇒
> **+0.040 [−0.097, +0.175] against the 0.090 floor ⇒ WITHIN FLOOR at n = 2**, as registered in
> advance (P(clears) = 0.15). `status: OK`, **0 ties, 0 forfeits**, `regime_verified: true` with
> Metamon's `argmax_match_rate` **1.0000** over 2,732 decisions and our `stochastic` kwarg `[false]`,
> `distinct_our_teams` 19/20, role split 8 pp. **ROW 5 — COLLATERAL.** The untaught meter IS the
> off-slice read. 🚨 **`main.exploitability` is NOT APPLICABLE and was declared so BEFORE the first
> battle**: its own `--help` says *bookkeeping only* over `fleet_admission`-schema artifacts, of
> which this arm has produced none, and the committed ones are each the output of a separate
> 800-games-per-arm battery carrying a fixed cross-era REFERENCE (`rev1final`) that row 2 does not
> have. **THE READING.** 🚨 **The inversion is the finding: §2.3's gen-era account is that a fold
> TEACHES on-slice (~+5 pp, every arm clearing zero) and DIGS A HOLE off-slice (~3–4 pp by +1M).
> This fold does NEITHER — no hole at any depth, its off-slice level is its best row, and on-slice it
> is NEGATIVE on one of its two taught teams.** One arm, n = 1, at a dose and a parent depth no
> gen-era cell occupied ⇒ a failure to reproduce in a new regime, and §2.3 is NOT edited from here.
> 🚨 **AGREEMENT AND PILOTING CAME APART.** `teacher_agreement_on_slice` rose 0.758 → 0.796 → 0.815
> → **0.823** (still rising at the last rollout, reproduced here from the events) while the fold
> **peaked at +3M and fell by +6M on BOTH taught teams** and went flat off-slice (51.56 → 51.31);
> `collateral_kl_vs_parent` 0.277 → 0.402 and `on_slice_kl` 0.327 → 0.690 were both still rising at
> the end. **The stop rule is conjunctive and the conjunct that never became true is the one about
> the TEACHER ("agreement has PLATEAUED"), not the one about the parent** — so `stop_signal` read
> 0.0000 at every point, correctly by its own definition, on an arm whose piloting was falling.
> Recorded as an instrument observation; the completion entry's "it had not finished learning from
> these teachers" is left standing beside it, as what that meter said. **WHAT THE CONTINUATION WOULD
> DECIDE, registered in `PREDICTION.md` §4 before any result:** `ai_v13_08_fold1_cont` (live, forked
> at 81,100,800, +6M more, same teachers, same frozen dose) tests BUDGET vs RECIPE, and only in one
> direction — **and the +3M → +6M leg has already run that experiment inside this arm and returned
> the second answer**, so the continuation tests whether that leg was a draw or a direction. Rows 1
> and 2 re-run at +12M (~4 h CPU) are the cheap decisive read; **what it cannot decide is whether
> anything clears a floor — that needs a second fold ARM (rule 22), not more steps on this one.**
> **HAZARDS, each a finding.** (1) clause (b) near-UNSATISFIABLE at this n — a bar property, with
> the arithmetic printed; (2) the gated share ANTI-predicts the slice gain; (3) agreement vs
> piloting; (4) the ladder is n = 0, disclosed pre-registration; (5) the 0.740 headline does not
> transfer to a third-party opponent; (6) specialists are badly OOD off their pin; (7) pooling
> manufactures an exact null from a real split; (8) every floor here except the two per-slice ones is
> an IMPORT from a 75M FRESH-ARM pair onto a FOLD (the per-slice floors were MEASURED on the same
> cells); (9) NO continuation control — every delta is against a FROZEN parent and §2.2's "the
> correction does not bite on our parents" is itself an import from ~28M/+1M; (10) SENIORITY is not
> separable from extraction (+6,094,848 steps), and the nearest bound on it is the untaught row's
> own +5.1 pp, which is why the on-slice numbers cannot be read as extraction; (11) **arm W is the
> LOWER of the two parent seeds on all three battle instruments here**, so every "beats its parent"
> number is against the weaker draw; (12) a fork's TB carries its parent's whole history and every
> series here is SLICED at the fork step (the `g7_ladder` trap, `36f8f7eb`); (13) `--config auto`
> NOT re-run — same `arch_signature`, `config_path` drives only the compatibility check, and the
> floor read showed both resolutions identical to 0.01 pp on three arms of this config; declared,
> not omitted. **WHAT IS NOT CLAIMED:** that the fold paid, or hurt; that any row clears a floor;
> that WITHIN FLOOR means equivalent (rule 6); that a floor is known rather than bounded at n = 2;
> that the gated share predicts anything; that §2.3's hole law is refuted; that the recipe failed —
> the fold is unconverged by its own stop rule's own meter and one arm against an imported floor is
> the weakest evidential configuration this programme has. Tag: **MEASURED · THE ERA-1 FOLD's
> REGISTERED READ · branch (d) NOT DETECTED · the SHAPE INVERTED — off-slice moved, on-slice split
> in SIGN · NO early off-slice hole (§2.3's law not reproduced at 0.20× dose on a 75M parent) ·
> untaught +5.4/+5.1 pp vs a 3.69 imported floor, clause (b) near-unsatisfiable · Big-5 −0.061 /
> DDTar +0.068, the gated share ANTI-predicts · the fold BEATS t2 because t2 had +0.0075 to teach ·
> the 0.740 vs-target headline does NOT transfer · ladder n = 0, `win_rate_vs_pool` 0.488 → 0.372
> on a FROZEN pool while bots RISE · agreement and piloting came APART · anchor WITHIN FLOOR ·
> 17,600 battles, 0 timeouts · both comparators reproduce EXACTLY, per-team**.
