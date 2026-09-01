# M1 — THE AXIS SPLIT: fold quality on TAUGHT teams vs externality on UNTAUGHT teams

**Status: COMPLETE for the three-point comparison; rev-2's untaught row IN FLIGHT.** Predictions
were registered in the dispatch and are scored below without adjustment. **3,200 new battles**
(2 arms × 8 teams × 200), zero dropped. Scripts: `axis_split_taught_untaught.py` (assembly +
statistics, no battles) and `axis_split_untaught_arm.py` (the new battle arms). Every input is
banked under `axis_split_inputs/`, so this reproduces without the session-scoped job directory.

---

## 0. Headline

**THE MISSION'S CENTRAL HYPOTHESIS IS REFUTED, and the contradiction it was sent to dissolve
survives intact.** The taught and untaught axes **order identically** — Spearman **+1.00** on
three points, against *both* taught cuts:

| cut | ordering (best → worst) |
|---|---|
| TAUGHT-9 (fold quality) | rev-3 (6×2) → COMPFOLD (3×4) → rev-4 (3×8) |
| TAUGHT-COV3 (coverage) | rev-3 → COMPFOLD → rev-4 |
| **UNTAUGHT-8 (externality)** | **rev-3 → COMPFOLD → rev-4** |

**Every fold in this era ROBS the untaught set** — rev-3 −2.50pp, COMPFOLD −3.88pp, rev-4
−6.50pp, with 6-to-7 of 8 teams negative in each arm. **The v8 gift (+5.42pp) does not reproduce
at ANY shape we have measured, including at shapes broader than v8's.** rev-4 taught **24**
distinct teams — *more* than v8's 22 — and robbed hardest of all. So the v8 anomaly is not a
taught-axis metric being read as an untaught one; it is something outside fleet shape entirely.

**What DOES dissociate is which shape variable dominates, and it flips between axes:**

| contrast | TAUGHT-9 | UNTAUGHT-8 |
|---|---|---|
| teacher count (rev-3 − COMPFOLD) | **+2.89pp z = +2.15** | +1.38pp z = +0.79 |
| **distilled team count** (COMPFOLD − rev-4) | +1.19pp z = +0.88 | **+2.63pp z = +1.50** |

The taught axis is separated by teacher count and is null on team count — *reproducing the
ledger's −0.0289 / +0.0119 to the decimal*. On the untaught axis the ranks reverse. **Neither
untaught contrast clears z = 2, so this is SUGGESTIVE, not established** — but it is the one
place the two axes are not the same measurement, and the team-count contrast is the *clean*
one (§2).

---

## 1. Method — three cuts, ONE instrument

Every number below comes from one meter family: the arm **pilots a pinned team**, the opponent is
the **fixed** rev-1 @24M snapshot drawing from the validated 719-team pool, both sides stochastic,
in-process **rust** bridge, CPU. The three cuts differ only in *which teams are pinned*:

| cut | teams × games | what it is |
|---|---|---|
| **TAUGHT-9** | 9 × 300 | the standing 9-slice fold-quality meter — rev-2's own taught set |
| **TAUGHT-COV3** | 3 × 300 | the coverage teams rev-3's fleet ADDED |
| **UNTAUGHT-8** | 8 × 200 | 8 teams **no fold in this table ever pinned** |

The 9-slice and coverage rows for all five folds already existed (training session); the
**COMPFOLD and R3ACTION untaught rows are new here** and complete the 2×2 the comparison needs.
`axis_split_untaught_arm.py` is a **verbatim** re-use of the instrument that produced the existing
`untaught_R2ACTION` / `untaught_R4ACTION` rows — same 8 teams in the same order, same seed family
(`1000 + 9 + slice_index`), same fixed target, same `n = 200`, same stochastic/rust settings. A
re-implementation would have put the new arms on a second scale and made the cross-arm differences
uninterpretable.

**The cut membership is DERIVED and ASSERTED, never assumed.** The script walks each fold's
recorded `--distill-teacher` spec → each teacher run's recorded `--trainee-teams`, and then
*fails* unless: the 9-slice is exactly rev-2's 9 distilled teams and is taught by all five folds;
the coverage cut is exactly the 3 teams rev-3 added; rev-2 did **not** teach the coverage cut; and
the untaught-8 intersects **no** fold's distilled or trained team union. All assertions pass.

That last fact makes the coverage cut do double duty: **it is a TAUGHT reading for
rev-3/COMPFOLD/rev-4/REFOLD1 and an UNTAUGHT reading for rev-2** — which is why rev-2's row on it
is **−5.89pp**, reproducing the founding "treadmill" −5.9pp from a completely separate assembly.

### The arms, and why they are comparable

Four of the five folds **share one parent** (`ai_v9_59_R2ACTION_0827` final) and one target
(`R2ACTION`, from every teacher's recorded `--exploiter`). So `fold − parent` on a fixed team set
is a matched contrast in everything except fleet shape. rev-2's parent is rev-1 final.

### Statistics

Arms are **not battle-paired** (both sides act stochastically and the sim dice are free); they
share only the per-team opponent-team draw *sequence*. So each team's Δ is an unpaired
two-proportion contrast, the pooled Δ is the **equal-weight mean over teams**, its interval is a
**20,000-resample cluster bootstrap over TEAMS** (the unit the claim generalises over), and `z`
sums both arms' binomial variances. `n` was inherited from the existing instrument, not re-chosen
after seeing anything.

### What is NOT on this instrument

**v8's +5.42pp is probe P's** — a different harness (parent-vs-fold against a fixed *ancestor*,
CRN-paired, greedy, 16 different untaught teams). It is carried with that flag and never pooled.
Probe Q's rev-3 −0.75pp is a third harness again; §5 reconciles it.

---

## 2. Shape, derived — and two things the campaign's shorthand hides

| fold | run | teachers | teams/teacher (distilled·trained) | distinct taught | budget/team (M) |
|---|---|---|---|---|---|
| rev-2 | `ai_v9_59_R2ACTION` | 5 | 2·2 | 9 | 1.53 |
| rev-3 | `ai_v9_70_R3ACTION` | 6 | 2·2 | 12 | 2.53 |
| COMPFOLD | `ai_v9_91_COMPFOLD` | 3 | **4·8** | 12 | 1.26 |
| rev-4 | `ai_v9_76_R4ACTION` | 3 | 8·8 | 24 | 1.26 |
| REFOLD1 | `ai_v9_82_REFOLD1` | 3 | 8·8 | 24 | **1.76** |

1. **COMPFOLD is not a 3×4 fleet.** It reuses rev-4's *identical teacher checkpoints* (R4S3a/b/c,
   each trained on 8 teams at 1.26M/team) and simply **distils from 4 of each teacher's 8 teams**.
   Teacher identity, teacher count, teacher training and per-team budget are held exactly fixed
   against rev-4; only the distilled team list moves. That makes **COMPFOLD vs rev-4 the only
   single-variable contrast in the table** — distilled team count, 12 vs 24, nothing else.
2. 🚨 **The contrast the 40-team fleet's shape was chosen on is CONFOUNDED THREE WAYS.** COMPFOLD
   vs rev-3 changes teacher count (3 vs 6) **and** teacher training breadth (8 vs 2 teams)
   **and** per-team budget (1.26 vs 2.53M) simultaneously. Reading its −2.89pp as "distinct
   teacher count" is one of three available readings — and `exploitability_taught_untaught`
   already measured that same budget halving costing **−0.1217 `ordered`** on coverage cells.
   The fleet was scaled to 20 teachers on a number that is equally a budget number.

**REFOLD1 is the budget contrast at fixed shape:** its REVIVE teachers are rev-4's R4S3 teachers
forked a further 4M over the same 8 teams — 3×8 / 24 teams at **1.76M/team** against rev-4's
1.26M. On the taught cuts +40% budget bought **−0.59pp** (9-slice, z = −0.44) and **+1.22pp**
(cov3, z = +0.53): nothing. Its untaught row was **cut for budget** (§Cuts) and is the single most
valuable missing cell in this table.

---

## 3. THE TWO AXES

| fold | teachers | teams/teacher (distilled·trained) | distinct taught | budget/team (M) | TAUGHT-9 Δpp | TAUGHT-COV3 Δpp | UNTAUGHT-8 Δpp |
|---|---|---|---|---|---|---|---|
| rev-2 | 5 | 2·2 | 9 | 1.534 | **+1.33** [-1.81, +4.67] | **-5.89** [-12.33, +0.67] | *(in flight)* |
| rev-3 | 6 | 2·2 | 12 | 2.534 | **+0.41** [-2.37, +3.37] | **+6.00** [+2.00, +12.33] | **-2.50** [-4.62, +0.19] z=-1.44 |
| COMPFOLD | 3 | 4·8 | 12 | 1.258 | **-2.48** [-4.18, -0.85] | **+2.56** [-1.33, +9.00] | **-3.88** [-6.94, -1.31] z=-2.23 |
| rev-4 | 3 | 8·8 | 24 | 1.258 | **-3.67** [-5.93, -1.26] | **+1.78** [-3.67, +11.33] | **-6.50** [-9.75, -3.06] z=-3.73 |
| REFOLD1 | 3 | 8·8 | 24 | 1.758 | **-4.26** [-7.41, -1.07] | **+3.00** [-0.33, +9.33] | *(cut — see Cuts)* |
| **v8** *(probe P, other instrument)* | 3 | 7.33·7.33 | **22** | — | — | **+26.18** [+20.28, +32.85] *(6 taught)* | **+5.42** [+3.44, +7.42] *(16 teams)* |

All Δ are **fold − its own fork parent** on the same teams. Every row except rev-2 and v8 shares
the parent `R2ACTION`, so those four rows are directly comparable to one another as levels.

Per-team untaught Δpp (the sign counts are the load-bearing part — a pooled mean of eight teams is
one number, eight signs are eight):

| team | rev-3 | COMPFOLD | rev-4 |
|---|---|---|---|
| `U_61590463` | −4.0 | −5.5 | −0.5 |
| `U_92832108` | −2.0 | −1.0 | +0.5 |
| `U_ce35b736` | −7.0 | −6.0 | −12.0 |
| `U_9909f2e9` | −2.0 | +0.5 | −8.0 |
| `U_9d5f8458` | −1.5 | −13.0 | −11.0 |
| `U_f7ba5702` | −4.0 | −2.0 | −11.5 |
| `U_90b94599` | +5.5 | −4.0 | −8.0 |
| `U_dbf81d8e` | −5.0 | 0.0 | −1.5 |
| **negative** | **7/8** | **6/8** | **7/8** |

**The floor confound does not apply here, and that is the point of using this set.** The shared
parent's own win rate on all eight untaught teams is 0.51–0.66 (set mean **0.5825**) — there is
competence to remove on *every* team, in both directions. This is exactly the discriminator probe
Q's §3 asked for and could not supply.

---

## 4. Do the axes order the same way? — YES

**Spearman between cuts, over the three folds measured on both axes: +1.00 (TAUGHT-9 vs
UNTAUGHT-8) and +1.00 (TAUGHT-COV3 vs UNTAUGHT-8).** With three points ρ = +1 is the maximum
attainable and its permutation p is 1/6 ≈ 0.17, so the *coefficient* is not evidence; the
**ordering** is, because it reproduces across two independent taught cuts of different sizes and
different membership.

They are not the same measurement, though, and two differences are large:

- **LEVEL.** The taught cuts run from −4.3 to +6.0pp; the untaught cut is **uniformly negative**,
  −2.5 to −6.5. A fold that looks flat-to-positive on the meter it is scored by is removing
  2.5–6.5pp from teams nobody covered. On the two cuts where a comparison exists,
  taught-minus-untaught is +2.9pp (rev-3), +1.4pp (COMPFOLD) and +2.8pp (rev-4) on the 9-slice,
  and +8.5 / +6.4 / +8.3pp on the coverage cut.
- **WHICH VARIABLE SEPARATES** (§0's table). Teacher count carries the taught axis (z = 2.15) and
  vanishes on the untaught one (z = 0.79); distilled team count is null on the taught axis
  (z = 0.88, faithfully reproducing the ledger's `+0.0119 z=+0.87`) and is the larger effect on
  the untaught one (z = 1.50). **Suggestive only.**

So: **the dispatch's reconciliation candidate fails.** The two axes are not different functions of
shape in the sense that would have made both bodies of evidence true — they rank shapes the same
way, and the v8 gift is not recovered at v8-like breadth.

---

## 5. Reconciliation with probe Q — the treadmill is INTACT at rev-3, and probe Q's own §3 was right

Probe Q measured rev-3's untaught pull-down at **−0.75pp [−4.56, +3.00]**, declared a null, and
named its own leading alternative: *the well may already have been dry* — R2-ACTION sat at 0.4975
on probe Q's eight teams, i.e. with no per-team edge left to remove. It named the discriminator: a
fold whose untaught set is **not** already depressed.

This probe is that discriminator, arrived at from the other direction. On these eight teams the
same parent sits at **0.5825**, and rev-3's hop is **−2.50pp [−4.62, +0.19], z = −1.44** — same
sign, 1.75pp more negative, on a set with room to lose. The two intervals overlap heavily, so this
is a **consistency result, not a contradiction**; but it moves the reading from "rev-3 stopped
robbing" toward "**rev-3 robbed less because there was less to take**", and the registered
`narrow-fleet ⇒ robbery` link that probe Q recorded as BROKEN should be reinstated as
**unresolved-leaning-intact** rather than refuted. rev-3's own CI still includes zero: this is not
a positive claim that rev-3 robs, it is a removal of the reason to believe it does not.

---

## 6. Which shape variable predicts UNTAUGHT externality?

Rank correlations against the untaught Δ, over the three in-instrument folds (**n = 3 — read the
orderings, not the coefficients**):

| shape variable | values (rev-3 / COMPFOLD / rev-4) | ρ vs UNTAUGHT | ρ vs TAUGHT-9 |
|---|---|---|---|
| **teams per teacher (distilled)** | 2 / 4 / 8 | **−1.00** | **−1.00** |
| distinct taught teams | 12 / 12 / 24 | −0.87 *(tied pair)* | −0.87 |
| distinct teachers | 6 / 3 / 3 | +0.87 *(tied pair)* | +0.87 |
| budget per trained team | 2.53 / 1.26 / 1.26 | +0.87 *(tied pair)* | +0.87 |

**Teams per teacher is the only variable that separates all three points, and it orders both axes
perfectly.** Total distinct taught teams cannot be the answer: it ties rev-3 with COMPFOLD, which
differ by 1.4pp, and its largest value (rev-4's 24) is the *most negative* row in the table.

🚨 **The confound, stated plainly.** Across our arms, teams-per-teacher, total taught teams,
teacher count and per-team budget move together almost everywhere:

- rev-3 → COMPFOLD: teams/teacher 2→4 **and** teachers 6→3 **and** budget 2.53→1.26M.
- COMPFOLD → rev-4: teams/teacher 4→8 **and** total teams 12→24 (teachers and budget FIXED).

Only the second is clean, and it cannot separate *teams per teacher* from *total taught teams* —
they move together by construction whenever the teacher count is held fixed. **The cells that
break it:**

| cell | shape | separates |
|---|---|---|
| **3 teachers × 4 teams at 2.53M/team** | COMPFOLD's shape, rev-3's budget | teacher count vs BUDGET on both axes — the confound the 40-team decision rests on |
| **6 teachers × 4 teams (24 total)** | 12 teachers' worth of teams, rev-3's teams/teacher | total taught teams vs teams-per-teacher |
| **REFOLD1's untaught row** | 3×8 at 1.76M vs rev-4's 1.26M | budget at *identical* shape — already half-measured (taught side: null) |

The third is the cheapest by an order of magnitude: one 8-team arm, ~90 min on two niced cores,
and the model already exists.

---

## 7. Predictions, scored

| # | registered prediction | outcome |
|---|---|---|
| **1** | The two axes ORDER DIFFERENTLY — the taught ordering (fewer teams/teacher better) does not reproduce on the untaught axis | **REFUTED.** It reproduces exactly: rev-3 → COMPFOLD → rev-4 on all three cuts, ρ = +1.00 against both taught cuts. The axes differ in LEVEL (taught −4.3…+6.0 vs untaught uniformly −2.5…−6.5) and, suggestively, in which variable separates — but not in ordering, which is what the prediction claimed. |
| **2** | TOTAL DISTINCT TAUGHT TEAMS is the best single predictor of untaught externality among computable shape variables (v8's 22 the largest and the only positive) | **REFUTED, twice over.** Within the instrument it is beaten by teams-per-teacher (ρ −0.87 with a tie vs −1.00) because it cannot separate rev-3 from COMPFOLD. And its premise fails outright: **rev-4 taught 24 distinct teams — more than v8's 22 — and produced the most negative externality in the table (−6.50pp, z = −3.73).** Breadth in team count is not what made v8 gift. |

**The failure names its broken link, per the standing rule.** The link was
`BREADTH → generalizable content → externality flips ROBBERY→GIFT`. Breadth is now measured on
the untaught axis at 12, 12 and 24 distinct taught teams and at 2, 4 and 8 teams per teacher, and
**the externality is negative at every one of them, monotonically worse as breadth per teacher
rises.** Whatever produced v8's gift, it is not fleet breadth.

**What is left for the v8 anomaly** (none of it measured here, all of it outside fleet shape):
student maturity (v8's fold ran on a 277M-step parent; ours on ~30M), fork length, the era's
distillation target form, and one measurement-side asymmetry that must be stated — **v8's parent
sat at 0.383 on its untaught set while ours sits at 0.5825**, so v8's arm had more room above it
and ours more room below. Probe P argued the floor confound does not apply in reverse there
(0.383 is far from both bounds), and that argument stands; but a gift measured from 0.383 and a
robbery measured from 0.5825 are not equally protected from mean reversion, and this probe cannot
separate that from a real sign difference.

---

## 8. What the RUNNING 40-team fleet should be scored on

**Verified from the 20 recorded launch commands (banked as `axis_split_inputs/r5_fleet_teams.json`,
20 arms × 2 teams = 40 distinct): the rev-5 fleet's teams are DISJOINT from every existing meter.**
Zero overlap with the 9-slice, zero with the 3 coverage teams, zero with rev-4's extra 12, zero
with the untaught-8.

🚨 **Consequence, and it is a category error waiting to happen: the 9-slice meter is an UNTAUGHT
cut for rev-5.** For rev-2/3/4/COMPFOLD it measures taught teams and is the campaign's fold-quality
number (rev-3's 0.5793 etc.). For rev-5 it measures teams the fleet never pinned. **Comparing
rev-5's 9-slice number to rev-3's 0.5793 compares an externality reading to a fold-quality one** —
precisely the axis confusion this mission was sent to look for, arriving in the next measurement
rather than the last one. The same applies to the coverage cut.

What to do, in priority order:

1. **The load-bearing cut is a NEW taught meter on rev-5's own 40 teams** (a stratified sample of
   them — 9 or 12 to match the existing meters' power), measured `fold − parent` against the same
   fixed rev-1 @24M target with the same instrument. It does not exist and nothing substitutes for
   it. Sizing it now, while the fleet trains, is free.
2. **Score the externality on the UNTAUGHT-8** — unchanged, still untaught, four folds already on
   it. This is the cut that answers whether 20×2 breaks the treadmill, and §0 says it starts from
   a strong prior of *no*: at 2 teams/teacher rev-5 sits at the shape end that robs least
   (rev-3's), but at 40 distinct taught teams it is far past the breadth where rev-4 robbed −6.5.
   Those two readings point opposite ways, which is exactly what makes the cell worth running.
3. **Read the 9-slice and coverage cuts as SECOND AND THIRD untaught sets**, not as fold quality —
   they become a 20-team externality panel, which is more power on the externality question than
   this probe had.
4. **Do not carry the taught-axis shape verdict onto rev-5's externality.** The variable that
   separated the taught axis (teacher count) is the one that vanished on the untaught axis. The
   fleet was scaled to 20 teachers on a taught-axis number that is additionally confounded with
   budget (§2), and rev-5 is provisioned at 1.5M/team — below the 2.53M of the only arm in this
   table that did not lose ground on the taught cuts.

---

## Reproduce

```
export PYTHONPATH=$PYTHONPATH:src
# a new untaught arm (~90 min on a contended box, 2 cores, models/ read-only)
nice -n 15 python designs/research_state/measurements/axis_split_untaught_arm.py \
    models/ai_v9_91_COMPFOLD_0831/final_model.zip COMPFOLD \
    designs/research_state/measurements/axis_split_inputs/untaught_COMPFOLD.json 200 3
# the assembly (no battles, no models, ~3 s)
nice -n 15 python designs/research_state/measurements/axis_split_taught_untaught.py
```

Wall clock: 3,200 battles over ~2 h on two `nice -n 15` cores beside a 20-arm GPU fleet
(load 24–41; per-slice time drifted 450 s → 1370 s with the box's load and back).

## Cuts

- **REFOLD1's untaught arm was CUT for budget**, not for method — it is the clean
  budget-at-fixed-shape contrast and §6 names it as the cheapest of the three cells that would
  break the confound. Its taught rows are complete and null.
- **rev-2's untaught row is IN FLIGHT** at the time of writing. It is the fourth point and the one
  that would test whether 2 teams/teacher is protective *across* a parent change; its absence does
  not affect §4 or §7, both of which rest on the three folds that share a parent. rev-2's
  externality is separately readable here from its **coverage-cut −5.89pp**, which is on this
  instrument and reproduces the founding treadmill number.
- **v8 was not re-measured.** Probe P owns it; a second, worse instrument on the same question
  would have added a scale, not an answer.
- **`n` was not extended** on any arm after seeing a result (inherited: 300 taught / 200 untaught).
- **The teams-per-teacher vs total-teams separation is not attempted** — no existing arm varies
  one at fixed teacher count without the other (§6).

## Cross-references

- **Probe P** `v8_redistribution_pfsp_2026-08-30.md` — v8's +5.42pp untaught gift; §7 here bounds
  what can still explain it.
- **Probe Q** `rev3_untaught_pulldown_2026-08-30.md` — rev-3's −0.75pp null; §5 supplies the floor
  discriminator its §3 asked for and reinstates the treadmill link as unresolved.
- **Composition** `exploitability_taught_untaught_2026-08-31.md` — the same budget halving
  (2.50 → 1.25M/team) costing −0.1217 `ordered` on coverage cells; §2 here shows that halving is
  inside the teacher-count contrast the fleet shape was chosen on.
- **Ledger** — the composition test (`COMPFOLD vs rev-3 −0.0289 z=−2.14`, `vs rev-4 +0.0119
  z=+0.87`) is reproduced here to the decimal as the TAUGHT-9 rows, and extended to the untaught
  axis where the two contrasts swap ranks.
