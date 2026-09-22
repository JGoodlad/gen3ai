# `main.best_response_gap` — the population loop's meter, and its archive validation

**2026-09-22.** Built because the next-era **POPULATION loop** (AlphaStar-style: exploiters enter
the generalist's opponent POOL and the generalist absorbs them by its own gradient) had **no meter
in this repo**. Its meter is the **BEST-RESPONSE GAP**:

    gap(t) = P(a fresh exploiter trained against generalist G_t beats G_t) − 0.5

read at **matched exploiter budget** across rounds. **The loop is working iff the gap FALLS round
over round** (0.66 → toward 0.50).

Why this and not the admission gate: the exploiter flywheel's teachers turned out to be
**target-specific counterplay**. Ledger 2026-09-21 — *NONE of the three 5-team teachers is
admitted*: all three were LEVEL with the plateau parent on their own five teams against a fixed
third party (−1.00 / +0.58 / −1.12 pp, every CI straddling zero), while the offense teacher beat
that same parent 0.657 [0.610, 0.702] **head-to-head**. *The REDESIGN fails the same way*: a
teacher trained against a DISTRIBUTION of opponents is −1.37 pp [−2.78, +0.27]. An exploiter is
therefore read as an **opponent for the pool**, not as a teacher — and the gap is the quantity that
says whether the pool is doing its job.

```
export PYTHONPATH=$PYTHONPATH:src
python -m main.best_response_gap <exploiter run…> [--stat endpoint] [--allow-unmatched] [--play N]
```

Code: `src/agents/training/best_response_gap.py` (engine) + `src/main/best_response_gap.py` (CLI).
Tests: `src/agents/training/best_response_gap_test.py` (34, synthetic run dirs) and
`src/main/best_response_gap_integration_test.py` (23, `integration`-marked, SKIPS when
`utils.paths.main_models_dir()` is None). Design detail:
`designs/training/exploiter_and_distillation.md`.

---

## 1. THE REFUSAL — the cross-era comparison does not run

The six real exploiter runs are era-1 (`ai_v13_05/06/10`, target arm W
`ai_v13_02_flywheel_winprob` @75,005,952) and era-2 (`ai_v13_13/14/15`, target the plateau parent
`ai_v13_12_plateau` @95,158,272). Asked to compare them, the meter refuses:

```
$ python -m main.best_response_gap ai_v13_05_exploit_big5starmie ai_v13_06_exploit_ddtar_spikes \
      ai_v13_10_exploit_stall ai_v13_13_exploit5_offense ai_v13_14_exploit5_balance \
      ai_v13_15_exploit5_stall

[best_response_gap] REFUSAL (unmatched_dose) — REFUSING the comparison — the exploiters are not matched:
  DOSE MISMATCH — ai_v13_05_exploit_big5starmie vs ai_v13_13_exploit5_offense: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)
  DOSE MISMATCH — ai_v13_05_exploit_big5starmie vs ai_v13_14_exploit5_balance: 3.815e-08 vs 8.392e-09 (4.55x apart, tolerance 10%)
  … (9 pairs in all: every era-1 × era-2 pair)

  A best-response gap is a property of (the generalist, the exploiter's budget, its dose, its
  regime). Comparing two rounds that differ on any of those attributes to the GENERALIST what
  belongs to the exploiter — which is exactly how the 2026-09-21 era-2/era-1 read was confounded,
  by a 4.5x dose gap nobody registered (--fork-lr unset, the new parent's annealed rate inherited).
  Pass --allow-unmatched to print it anyway; the mismatch is then carried in the header and in the
  JSON, and belongs in every quote of the number.
```

Exit code **2**. The cause named is **dose**, not budget, because the BUDGET is genuinely matched:
both eras trained **8,060,928** post-fork steps. Within an era, `check_matched` returns `[]` — the
three arms of each round are matched on budget, dose and regime.

The three refusal classes: `UnmatchedBudgetError` (post-fork steps >2 % apart),
`UnmatchedDoseError` (`dose_rate` >10 % apart), `UnmatchedRegimeError` (`eval_sentinel_greedy`, the
exploiter opponent mix, the per-cycle sample size, or whether the target pilots its own pin or the
pool). `SeriesError` and `RunReadError` cover a run that is not an exploiter, one whose every
vs-target row is pre-fork, and one whose recorded target and evaluated external disagree.

---

## 2. THE ARCHIVE TABLE — printed under `--allow-unmatched`

Artifacts beside this file: `archive_read.json`, `archive_read.md`.

```
BEST-RESPONSE GAP — how much a fresh exploiter beats the generalist it trained on
  gap = win rate − 0.5; the population loop is working iff the gap FALLS round over round
  stat: pooled   regime: greedy-vs-greedy (EVAL regime, main.eval_worker FIXED branch)

🚨 UNMATCHED — this comparison was printed under --allow-unmatched. Every number below carries
   these confounds:  DOSE MISMATCH × 9 pairs, 3.815e-08 vs 8.392e-09 (4.55x apart)
  ⚠ CAVEAT × 3 (one per archetype): the TEAMSET SIZE changed (1/5 -> 5/5).

ROUND 1 · target ai_v13_02_flywheel_winprob @75,005,952 · budget 8,060,928 steps · dose 3.815e-08
  archetype  teams  run                             endpoint  pooled  n    gap pp   95% CI pp
  balance    1/5    ai_v13_05_exploit_big5starmie   0.7400    0.6975  400   +19.75  [+15.08, +24.05]
  offense    1/5    ai_v13_06_exploit_ddtar_spikes  0.7400    0.7375  400   +23.75  [+19.23, +27.82]
  stall      1/5    ai_v13_10_exploit_stall         0.7400    0.7225  400   +22.25  [+17.67, +26.41]

ROUND 2 · target ai_v13_12_plateau @95,158,272 · budget 8,060,928 steps · dose 8.392e-09
  archetype  teams  run                         endpoint  pooled  n    gap pp   95% CI pp
  balance    5/5    ai_v13_14_exploit5_balance  0.5300    0.5250  400    +2.50  [ -2.39,  +7.35]
  offense    5/5    ai_v13_13_exploit5_offense  0.7000    0.6575  400   +15.75  [+10.97, +20.23]
  stall      5/5    ai_v13_15_exploit5_stall    0.4500    0.4550  400    -4.50  [ -9.31,  +0.40]

ROUND-OVER-ROUND Δ (round 2 − round 1), paired on ARCHETYPE
  archetype  gap r1   gap r2   Δ pp     95% CI pp (Newcombe)
  balance     +19.75    +2.50   -17.25  [-23.76, -10.52]
  offense     +23.75   +15.75    -8.00  [-14.28,  -1.63]
  stall       +22.25    -4.50   -26.75  [-33.11, -20.04]
    MEAN Δ over 3 archetype(s):  -17.33 pp  [-26.67, -8.00]  (two-level bootstrap, 3 pairing units)
    VERDICT: THE GAP FELL — the generalist is absorbing its best responders
```

**The banked endpoints reproduce exactly** — era-1 **0.740 / 0.740 / 0.740**, era-2 **0.700 /
0.530 / 0.450** — and so do the pooled rates the ledger quotes for era-2 (**0.657 / 0.525 /
0.455**, i.e. 263 / 210 / 182 wins out of 400). `--stat endpoint` gives the gaps the ledger's own
numbers imply: era-1 flat at **+24.00 pp** on all three, era-2 **+20.00 / +3.00 / −5.00 pp**.

🚨 **THE VERDICT IS NOT EVIDENCE THAT THE LOOP WORKS.** Three things differ between these rounds
and the meter prints all three on the same page as the number:

1. **dose** — 3.815e-08 against 8.392e-09, **4.55×** (the refusal above);
2. **teamset size** — era-1 pins ONE team (the archetype's anchor), era-2 pins all five, so the
   two gaps are best responses inside **different subgame restrictions**;
3. **target identity** — the round-2 target is a DESCENDANT of the round-1 target
   (`ai_v13_12_plateau` ← `ai_v13_09_wcont` ← `ai_v13_02_flywheel_winprob`). In a population loop
   that is the design; here it is not, because no exploiter was ever folded back.

A clean read needs one round's generalist, its exploiters folded into the pool, the generalist
trained on, and a fresh exploiter at the SAME budget, dose, regime and teamset. That is the
experiment this meter exists to read; this table is its validation, not its result.

---

## 3. THE FINDING THAT CAME OUT OF BUILDING IT

🚨 **The training-time vs-target series is GREEDY-vs-GREEDY, not the training regime.**
`main.eval_worker`'s FIXED branch builds the cross-run opponent `stochastic=False, temperature=1.0`
("eval = greedy yardstick") against a greedy `EvalRLPlayer` trainee. **`eval_sentinel_greedy` does
not govern it** — that flag moves the `sentinel_*` branch, and an exploiter run has no sentinels at
all (`sentinels: []` in every row of all six files). Every vs-target number banked this month,
including 0.657 and 0.700, is an EVAL-regime number. `--play N` plays fresh head-to-head games and
defaults to the TRAINING regime (stochastic@1 both sides); `--play --greedy` reproduces the series'
own. The two are different populations, so the report prints a played rate BESIDE the gap table and
never folds it in.

Smoke (20 games, not a measurement): `ai_v13_13_exploit5_offense` vs `ai_v13_12_plateau`,
stochastic@1, rust bridge, seed 0, concurrency 1 → **11/20 = 0.5500 [0.3421, 0.7418], 0 timeouts**.
Consistent with the series' 0.6575 at this n; nothing more can be said from 20 games.

Also worth recording: the target in both eras is an **unpinned generalist**, so in eval it pilots
the shared random pool while the exploiter pilots its own pinned team(s). The vs-target number is
"my specialist team vs your random team", not a mirror match — the meter reads and prints that
(`target pilots the shared POOL`) rather than leaving a reader to assume.

---

## 4. Ready-to-append ledger paragraph

### 2026-09-22 · OPS · **THE POPULATION LOOP HAS A METER — `python -m main.best_response_gap`, and it REFUSES the era-2/era-1 comparison on the 4.5× dose gap that confounded last week's read. The banked endpoints reproduce from the runs' own `eval_results.jsonl` (0.740 ×3; 0.700 / 0.530 / 0.450), and 🚨 every vs-target number we have banked is a GREEDY-vs-GREEDY EVAL number, not a training-regime one.**

**WHAT IT IS.** `gap(t) = P(a fresh exploiter trained against generalist G_t beats G_t) − 0.5`, read at matched exploiter budget, per ROUND × ARCHETYPE. **The population loop is working iff the gap FALLS round over round.** It is OFFLINE (no training, no server, nothing written under `models/`) and reads only what each run wrote down: the target from the recorded `lineage.exploiter_target` block through `agents.training.lineage`, the vs-target series from `eval_results.jsonl`'s `externals["ext_<target>"]` (POST-FORK rows only), the budget from `num_timesteps − fork_step`, the dose from `main.dose`, the teams from `matchup_spec.read_recorded_trainee_teams`. Endpoint AND pooled rate, each with a Wilson interval; Newcombe intervals per archetype across rounds; a two-level bootstrap (archetypes resampled, then each cell's binomial redrawn) for the mean Δ.

**THE REFUSAL FIRED ON THE REAL ARCHIVE, AND NAMED DOSE.** Asked to compare `ai_v13_05/06/10` (target arm W @75,005,952) against `ai_v13_13/14/15` (target the plateau parent @95,158,272), it exits 2 on `UnmatchedDoseError` — **3.815e-08 against 8.392e-09, 4.55× apart** on all nine cross-era pairs — while the BUDGET is genuinely matched at **8,060,928 post-fork steps** on all six, which is why dose is the cause it names. Within each era the three arms are matched and the gate returns clean. `--allow-unmatched` prints the table with the confound in the header and in the JSON.

**THE TABLE, UNDER `--allow-unmatched`, AND THE BANKED NUMBERS IT REPRODUCES.** Round 1 gaps (pooled, n = 400): balance **+19.75 pp**, offense **+23.75**, stall **+22.25**. Round 2: balance **+2.50**, offense **+15.75**, stall **−4.50**. Paired on archetype: Δ **−17.25 / −8.00 / −26.75 pp**, **MEAN Δ −17.33 pp [−26.67, −8.00]**. The endpoints reproduce the banked values exactly (0.740 ×3; 0.700 / 0.530 / 0.450) and so do era-2's pooled rates (0.657 / 0.525 / 0.455). ⚠️ **The "GAP FELL" verdict is NOT evidence the loop works** — the rounds differ in dose (4.55×), in teamset size (1/5 → 5/5, a subgame-restriction confound the report raises as a CAVEAT) and in target identity, and no exploiter was ever folded back. The meter prints all three beside the number.

🚨 **THE FINDING: EVERY BANKED vs-TARGET NUMBER IS AN EVAL-REGIME NUMBER.** `main.eval_worker`'s FIXED branch builds the cross-run opponent `stochastic=False, temperature=1.0` ("eval = greedy yardstick") against a greedy `EvalRLPlayer` trainee, so the series is **greedy-vs-greedy** — and **`eval_sentinel_greedy` does not govern it** (that flag moves the `sentinel_*` branch, and an exploiter run has `sentinels: []` in every row). 0.657 and 0.700 are therefore not training-regime rates. `--play N` plays fresh head-to-head games in the TRAINING regime by default (`--greedy` for the eval one); a 20-game smoke put `ai_v13_13` at 11/20 = 0.5500 [0.3421, 0.7418], 0 timeouts, which is consistent with 0.6575 and says nothing more at that n. The two regimes are never folded together. Also recorded: both targets are unpinned generalists, so the vs-target number is "my specialist team vs your random team", not a mirror match.

Tag: **OPS · THE POPULATION LOOP HAS A METER · `main.best_response_gap`, OFFLINE, 57 tests · the era-2/era-1 comparison REFUSES on a 4.55× dose gap at a MATCHED 8,060,928-step budget · banked endpoints reproduce (0.740 ×3; 0.700/0.530/0.450) · mean Δ −17.33 pp [−26.67, −8.00] under `--allow-unmatched`, with dose, teamset size and target identity all named as confounds · 🚨 every banked vs-target number is GREEDY-vs-GREEDY EVAL, and `eval_sentinel_greedy` does not govern it**.
