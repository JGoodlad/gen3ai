# M1 — THE AXIS SPLIT: fold quality on TAUGHT teams vs externality on UNTAUGHT teams

**Status: IN FLIGHT.** Predictions were registered in the dispatch and are scored below without
adjustment. Scripts: `axis_split_taught_untaught.py` (assembly + statistics, no battles) and
`axis_split_untaught_arm.py` (the new battle arms). Every input is banked under
`axis_split_inputs/` so this reproduces without the session-scoped job directory.

---

## 0. The contradiction this probe was sent to dissolve

Our shape evidence orders **fold quality** 6×2 > 3×4 > 3×8 — monotone in *fewer teams per
teacher* — and the 40-team fleet (20×2) was provisioned on it. But **v8** was 3 teachers × ~7.3
teams, the *worst* shape by that metric, and produced +69 anchored ELO plus a measured untaught
**gift** of +5.42pp [+3.44, +7.42].

The hypothesis: **our shape metrics are TAUGHT-side and v8's surprise is UNTAUGHT, and nobody has
ever measured shape → untaught externality.** If the two axes order differently, both bodies of
evidence are true and one has been read as the other.

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

**The cut membership is DERIVED and ASSERTED, never assumed.** The script walks each fold's
recorded `--distill-teacher` spec → each teacher run's recorded `--trainee-teams`, and then
*fails* unless: the 9-slice is exactly rev-2's 9 distilled teams and is taught by all five folds;
the coverage cut is exactly the 3 teams rev-3 added; rev-2 did **not** teach the coverage cut; and
the untaught-8 intersects **no** fold's distilled or trained team union. All assertions pass.

That last fact is what makes the coverage cut do double duty: **it is a TAUGHT reading for
rev-3/COMPFOLD/rev-4/REFOLD1 and an UNTAUGHT reading for rev-2** — which is why rev-2's row on it
reproduces the founding "treadmill" number (below) rather than a fold-quality one.

### The arms, and why they are comparable

Four of the five folds **share one parent** — `ai_v9_59_R2ACTION_0827` final — and one target
(`R2ACTION`, per every teacher's recorded `--exploiter`). So `fold − parent` on a fixed team set
is a matched contrast in everything except fleet shape. rev-2's parent is rev-1 final; its row is
its own hop on the same meter.

### Statistics

Arms are **not battle-paired** (both sides act stochastically and the sim dice are free); they
share only the per-team opponent-team draw *sequence*. So each team's Δ is an unpaired
two-proportion contrast, the pooled Δ is the **equal-weight mean over teams**, its interval is a
**20,000-resample cluster bootstrap over TEAMS** (the unit the claim generalises over), and the
`z` sums both arms' binomial variances. `n` per cell is inherited from the existing instrument
(300 taught / 200 untaught) and was **not** re-chosen after seeing anything.

### What is NOT on this instrument

**v8's +5.42pp is probe P's**, a different harness (parent-vs-fold against a fixed *ancestor*,
CRN-paired, greedy, 16 different untaught teams). It is carried in the table with that flag and is
never pooled with the rest. Probe Q's rev-3 −0.75pp is a third harness again; §5 reconciles it.

---

## 2. Shape, derived

| fold | run | teachers | teams/teacher (distilled·trained) | distinct taught | budget/team (M) |
|---|---|---|---|---|---|
| rev-2 | `ai_v9_59_R2ACTION` | 5 | 2·2 | 9 | 1.53 |
| rev-3 | `ai_v9_70_R3ACTION` | 6 | 2·2 | 12 | 2.53 |
| COMPFOLD | `ai_v9_91_COMPFOLD` | 3 | **4·8** | 12 | 1.26 |
| rev-4 | `ai_v9_76_R4ACTION` | 3 | 8·8 | 24 | 1.26 |
| REFOLD1 | `ai_v9_82_REFOLD1` | 3 | 8·8 | 24 | **1.76** |

Two shape facts the campaign's shorthand hides, both derived here rather than quoted:

1. **COMPFOLD is not a 3×4 fleet.** It reuses rev-4's *identical teacher checkpoints* (R4S3a/b/c,
   each trained on 8 teams at 1.26M/team) and simply **distils from 4 of each teacher's 8 teams**.
   Teacher quality, teacher count and per-team budget are held exactly fixed against rev-4; only
   the distilled team list moves. That makes **COMPFOLD vs rev-4 the cleanest single-variable
   contrast in the whole table** — distilled team count, 12 vs 24, nothing else.
2. **The contrast the 40-team fleet's shape was chosen on is CONFOUNDED THREE WAYS.** COMPFOLD vs
   rev-3 changes teacher count (3 vs 6) *and* teacher training breadth (8 vs 2 teams) *and*
   per-team budget (1.26 vs 2.53M) simultaneously. Attributing its −2.9pp to "distinct teacher
   count" is one of three available readings, and the exploitability probe already measured that
   same budget halving costing −0.1217 `ordered` on coverage cells.

**REFOLD1 supplies the budget contrast at fixed shape:** its REVIVE teachers are rev-4's R4S3
teachers forked a further 4M over the same 8 teams, so it is 3×8 / 24 teams at **1.76M/team**
against rev-4's 1.26M — same shape, +40% budget.

---

## 3. THE TWO AXES

<!--TABLES-->

---

## 4. Do they order the same way?

<!--ORDERING-->

---

## 5. Reconciliation with probe Q, and the floor confound it could not resolve

<!--PROBEQ-->

---

## 6. Which shape variable predicts UNTAUGHT externality?

<!--PREDICTOR-->

---

## 7. Predictions, scored

<!--SCORED-->

---

## 8. What the RUNNING 40-team fleet should be scored on

<!--REV5-->

---

## Reproduce

```
export PYTHONPATH=$PYTHONPATH:src
# the three new battle arms (2 at a time, ~65 min each on a contended box)
nice -n 15 python designs/research_state/measurements/axis_split_untaught_arm.py \
    models/ai_v9_91_COMPFOLD_0831/final_model.zip COMPFOLD out.json 200 3
# the assembly (no battles, ~3 s)
nice -n 15 python designs/research_state/measurements/axis_split_taught_untaught.py \
    --out designs/research_state/measurements/axis_split_taught_untaught_2026-08-31
```

## Cuts

<!--CUTS-->
