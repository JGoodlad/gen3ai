# PROBE Q — rev-3's OWN untaught-team pull-down (P3's missing third point)

**Status: COMPLETE.** Registered 2026-08-30 (ledger `c06e386`); method, arms, team selection and
predictions were all frozen before the first battle. 4,800 paired battles, **0 dropped**.

## Headline

**H3 SELECTED: rev-3's own untaught pull-down is −0.75pp, CI [−4.56, +3.00], z = −0.39 — a NULL.**
The registered −4..−6pp band is excluded except at its most optimistic edge; the −5.9pp point is
excluded outright. **Both H1 (share-constant) and H2 (breadth-determined robbery) are refuted at
rev-3.**

**But the treadmill is not refuted — it is SHARPENED.** The same instrument re-measures rev-2's
own pull-down on these same never-taught teams at **−7.06pp, CI [−10.56, −3.50], z = −3.86** — an
independent replication of the registered −5.9pp on a different team set with a different harness.
And the **two-hop cumulative is −7.81pp, CI [−10.75, −4.38], z = −4.81**: rev-3 did not repair
rev-2's damage here, it merely stopped adding to it. *Each revolution repairs the last **only
where the next fleet's coverage arrives.*** On teams no fleet ever covers, the loss is permanent.

## The question

Revolution two's fold **redistributed**: R2-ACTION gained on the teams its fleet taught and LOST
**−5.9pp (z=2.5)** on the untaught coverage teams (ledger `ade78c1`, finding (c) — "the
treadmill"). Rev-3's fold then repaired that damage on those same three teams. Nobody has
measured whether **rev-3's own fold does the same thing to the teams IT never taught** — the third
point in P3's scaling row (rev-2 = 9 taught · rev-3 = 12 · v8 = 23).

## Registered predictions (scored, never adjusted)

| # | account | prediction |
|---|---|---|
| H1 | share-constant | R3-ACTION loses ≈ rev-2's −5.9pp analogue on never-taught teams |
| H2 | content-externality (breadth-determined) | rev-3's narrow 2-teams-per-teacher fleet ALSO robs, **−4 .. −6pp** |
| H3 | — | **≈0 or positive** refutes both and reopens the mechanism |

H1 and H2 point the same way at rev-3 (it is narrow AND at the same bias share), so this probe
CONFIRMS-or-REFUTES the treadmill; it does not separate H1 from H2. The separating point is v8
(broad), which is probe P's.

## Arms

| arm | run | checkpoint | relation |
|---|---|---|---|
| **A** | `ai_v9_70_R3ACTION_0828` | `final_model.zip` (32.62M) | the rev-3 fold |
| **B** | `ai_v9_59_R2ACTION_0827` | `final_model.zip` (28.07M) | **A's parent** — `--model` of A's argv |
| **C** | `ai_v9_29_rev1_0823` | `final_model.zip` (25M) | B's parent, the era base |

**A−B** = rev-3's one-hop redistribution. **A−C** = the two-hop cumulative. **B−C** = rev-2's own
untaught pull-down *re-measured on this team set with this instrument* — so the P3 row's rev-2
point does not have to be imported from another harness's scale.

## Meter convention

The standing per-team piloting meter: the arm pilots ONE pinned team; the opponent is the FIXED
reference model (**rev-1 final**) drawing from the 719-team pool; greedy (`stochastic=False`);
in-process **rust** bridge, no server; CPU, one thread, `nice -n 15`, two shards = two cores.

**Pairing.** Battle *i* of every (team, arm) cell uses the same opponent pool team and the same
gen-5 sim seed, drawn once from `random.Random(20260830)`. The three arms therefore walk an
identical draw sequence. A `FixedSequenceTeambuilder` wraps the validated 719-team pool so only
the ORDER is made deterministic, and a **pairing guard** refuses the cell if the opponent
teambuilder was not consulted exactly once per battle (an unpaired run would still report a
plausible win rate — the silent-GIGO shape this tree keeps re-learning).

`n = 200` battles per (team, arm) — 8 teams × 3 arms × 200 = **4,800 battles**. n was fixed from
the pilot's throughput BEFORE any team's data existed and is not extended: the rev-3 recap
adopted the no-optional-stopping call, and this probe inherits it.

**CIs.** Per-team rows: paired normal CI on the per-battle difference vector. Pooled: **bootstrap
over TEAMS** (20,000 reps) — the cluster is the team, not the battle, because within a team the
rows are one correlated sample of one matchup.

## Team selection (pre-registered, `rev3_untaught_pulldown_selection.json`)

Drawn from `data/teams/sample/` — the curated 32, which is the LEGAL exploiter universe
(`validate_exploiter_trainee_is_sample`) and therefore the space every taught team comes from.
The −5.9pp analogue was measured on sample teams; keeping the untaught set in the same universe
keeps both readings on one scale and makes the exclusions exact.

**Exclusions.** The taught union is read at RUN TIME from the recorded `--trainee-teams` of
F5a–e + F6a–f + F6-CURR — never hand-copied — and the probe hard-fails if any pick is a member.
It resolves to exactly **12** teams (the 9 meter + the 3 rev-3 coverage picks). Every pick is
additionally uncovered by the E1–E4 substrate exploiters, so "never taught" holds across eras.

**Two exclusions could not be executed as written, and both are stated rather than papered over:**
the ORIGINAL rejected coverage picks and the 2 held-out teams are **not recorded anywhere in the
tree**. The originals were rejected *by* the curated-32 constraint, i.e. they were not sample
teams, so drawing from `sample/` excludes them by construction. The held-outs are called "pool
teams" by the rev-2 capstone, presumptively non-sample; residual risk accepted, and a coincidence
would not corrupt anything — a held-out team is untaught either way; one row would merely double
as the narrowness instrument.

**Known structural gap:** hyper_offense CANNOT be represented — all four hyper_offense sample
teams are in the taught union. The untaught set is balance/stall-heavy while the fleet's taught
set is offense-heavy. Quoted as a caveat on the pooled number.

| # | team | archetype | pool WR |
|---|---|---|---|
| 1 | `a04c29cf769e9a11` Forretress Gengar Dug | balance | 0.537 |
| 2 | `9283210847f806ee` Special Wall-less TSS | balance | 0.736 |
| 3 | `e11829f0561ef5a9` MixMence Claydol TSS | balance | 0.859 |
| 4 | `d4e74946b54f1a4b` Zapdos + AeroBi Spikes Offense | offense | 0.740 |
| 5 | `9d5f845869e899ee` Yama BandMence | semi_stall | 0.877 |
| 6 | `b89e1e37caa40e6a` SkarmMag + Aerodactyl TSS | stall | 0.774 |
| 7 | `8cdc78b2d46f0515` Suicune + MiloDol | stall | 0.796 |
| 8 | `9909f2e98e981ccc` Zapdos MixMence Forre TSS | stall | 0.799 |

## Load-path acid test

Three checks, because a mis-resolved model path returns a plausible win rate rather than an error.

1. **Loads at the current architecture** — all three, `obs_dim 2501`.
2. **Distinct networks** — pairwise max|Δp| over a shared forward and pairwise parameter L2 are
   both non-zero.
3. **IN-SITU (the strongest, and free)** — greedy policies on identical (team, opponent-team,
   sim-seed) triples are DETERMINISTIC, so two arms that were secretly the same weights would
   produce byte-identical per-battle win vectors. The probe raises if any pair does.

Measured:

```
pairwise max|Δp|   R3|R2 0.00589   R3|REV1 0.01890   R2|REV1 0.02345
pairwise param L2  R3|R2 51.84     R3|REV1 47.11     R2|REV1 16.55
```

Function-space lineage order is as expected (the child sits nearer its parent). **Parameter-space
order is INVERTED and it is a finding, not a defect:** R3-ACTION is FARTHER from R2-ACTION (51.8)
than from rev-1 (47.1), while R2-ACTION sits only 16.5 from rev-1. The rev-3 fold moved a long
way from its parent and landed closer to the grandparent than the parent is far from it — i.e.
part of that motion was *back toward rev-1*. Consistent with the anchoring account (a fleet of
rev-1-descended teachers averages to "return to rev-1"); recorded here, not claimed as evidence.

## Results

Per-team win rate piloting the pinned team vs rev-1 final on the pool, n = 200 paired battles per
cell, no cell dropped a single battle.

| team | archetype | R3-ACTION | R2-ACTION | rev-1 | **A−B** | A−C | B−C |
|---|---|---|---|---|---|---|---|
| `a04c29cf769e9a11` Forretress Gengar Dug | balance | 0.425 | 0.405 | 0.515 | **+2.0** | −9.0 | −11.0 |
| `9283210847f806ee` Special Wall-less TSS | balance | 0.490 | 0.535 | 0.550 | **−4.5** | −6.0 | −1.5 |
| `e11829f0561ef5a9` MixMence Claydol TSS | balance | 0.410 | 0.490 | 0.545 | **−8.0** | −13.5 | −5.5 |
| `d4e74946b54f1a4b` Zapdos + AeroBi | offense | 0.460 | 0.440 | 0.555 | **+2.0** | −9.5 | −11.5 |
| `9d5f845869e899ee` Yama BandMence | semi_stall | 0.505 | 0.465 | 0.605 | **+4.0** | −10.0 | −14.0 |
| `b89e1e37caa40e6a` SkarmMag + Aero TSS | stall | 0.615 | 0.620 | 0.605 | **−0.5** | +1.0 | +1.5 |
| `8cdc78b2d46f0515` Suicune + MiloDol | stall | 0.545 | 0.470 | 0.575 | **+7.5** | −3.0 | −10.5 |
| `9909f2e98e981ccc` Zapdos MixMence Forre TSS | stall | 0.470 | 0.555 | 0.595 | **−8.5** | −12.5 | −4.0 |
| **set mean WR** | | **0.490** | **0.498** | **0.568** | | | |

Pooled, bootstrap over teams (20,000 reps, the cluster is the team):

| quantity | meaning | estimate | 95% CI | z |
|---|---|---|---|---|
| **A−B** | **rev-3's own one-hop pull-down** | **−0.75pp** | **[−4.56, +3.00]** | **−0.39** |
| A−C | two-hop cumulative | −7.81pp | [−10.75, −4.38] | −4.81 |
| B−C | rev-2's own one-hop, THIS instrument | −7.06pp | [−10.56, −3.50] | −3.86 |
| (A−B) − (B−C) | rev-3 robbed LESS than rev-2 by… | +6.31pp | [−0.50, +12.94] | +1.88 |

The last row is the direct comparison the mission asks for, and it is honest about missing: the
two revolutions' increments differ in the expected direction by +6.3pp, but **z = 1.88 does not
clear 2** — the difference is suggestive, not established.

### Reading

1. **The null is real and it is not a power failure.** The CI half-width is ±3.8pp, so a −5.9pp
   effect would have read at z ≈ −3. The instrument saw exactly that magnitude on the *other*
   arm of the same battles (B−C = −7.1pp at z = −3.9). This is a measured null, not an unresolved one.

2. **Large per-team churn under a zero mean.** The A−B rows span **+7.5 to −8.5pp** (sd 5.8pp)
   around −0.75. The rev-3 fold RESHUFFLES competence across untaught teams even where it does
   not net-remove it — so "no pull-down" is not "no effect", and a per-team reading is not
   substitutable by the pooled one.

3. ⚠️ **THE LEADING ALTERNATIVE, and it is not weak: the well may already have been dry.**
   R2-ACTION's mean WR on these 8 teams is **0.4975** — indistinguishable from 0.50, the level at
   which the pilot has no per-team edge left over the reference at all. rev-1 sat at 0.568 and
   rev-2 took ~7pp of that. A fold cannot redistribute away competence that is already gone, so
   "rev-3 stopped robbing" and "rev-3 had nothing left to rob **on this set**" predict the same
   number here. **This probe cannot separate them**, and the distinction matters for rev-4: under
   the first reading, breadth 12 was already enough; under the second, the treadmill is intact and
   rev-3 simply hit the floor. The discriminator is a fold whose untaught set is NOT already
   depressed — which is exactly what rev-4's coverage-class teams and the 40-team revolution
   supply, and it should be pre-registered there.

4. **The two-hop is the number that should drive the coverage argument.** −7.8pp cumulative on
   teams neither fleet taught, against a rev-1 baseline of 0.568 → 0.490. Rev-3's celebrated +6pp
   repair landed on the 3 coverage teams — *which rev-3 then taught*. Repair follows coverage; it
   does not radiate. That is a stronger argument for breadth than the original treadmill framing,
   because it removes the "the next revolution fixes it" consolation for anything outside the
   fleet's reach.

5. **Instrument validation, banked.** B−C = −7.06pp [−10.56, −3.50] on 8 sample teams with this
   harness independently reproduces the −5.9pp z=2.5 the training session measured on 3 coverage
   teams with its own. Two harnesses, two team sets, one number. The treadmill's founding
   measurement is now replicated.

6. **Pairing efficacy, stated rather than assumed:** mean per-battle A/B outcome correlation
   **0.142**. Greedy trajectories diverge after the first divergent decision, so the CRN pairing
   removes the team-draw and opening-dice terms and not much more. It is a real but modest
   variance reduction; the CIs above already reflect it.

## The P3 scaling row

| revolution | teachers × teams/teacher | distinct taught | bias | **share taken from untaught** | own untaught pull-down |
|---|---|---|---|---|---|
| **rev-2** `ai_v9_59` | 5 × 2 | 9 | 0.4 | **0.3950** | **−7.06pp** [−10.56, −3.50] *(this probe)* · −5.9pp *(registered, 3 coverage teams)* |
| **rev-3** `ai_v9_70` | 6 × 2 | 12 | 0.4 | **0.3933** | **−0.75pp** [−4.56, +3.00] *(this probe)* |
| **v8** `ai_v8_14` | 3 × 7.67 | 23 | 0.4 | **0.3872** | **+10.4pp** — probe P **INTERIM** (ledger `91d5125`); cited, never re-measured here |

🚨 **The registered x-axis is DEGENERATE, and that is the row's most important finding.** All
three folds ran `--distill-team-bias 0.4`, and `apply_distill_team_bias` splits that 0.4 across
the teacher-team list with the remainder as uniform pool rehearsal — so with K ≪ 719 the share
taken from untaught teams is `0.4·(1 − K/719)` = **0.3950 / 0.3933 / 0.3872**. A **0.8pp** spread
on x against an **~18pp** spread on y. P3's primary model ("≈ linear in share") is therefore
**UNIDENTIFIABLE on these three points, not merely unsupported** — and the owner's registered
1/√N alternative is not on this axis either. The row cannot be fit; it can only be re-specified.

**On the axis that does vary** — teams-per-teacher (2, 2, 7.67) and total breadth (9, 12, 23) —
the two NARROW points disagree with each other by +6.3pp, so breadth-per-teacher alone does not
order the three either. The v8 point remains the only positive sign and it differs on breadth
**and** lineage **and** ecology simultaneously.

**Scoring against the frozen prediction table (`91d5125`).** That table's row —
*"probe Q must show rev-3's own untaught pull-down ≈ −4..−6"* — **FAILS**. Under the standing
"any failure names its broken link" rule, the named link is *narrow-fleet ⇒ robbery*: rev-3 is as
narrow as rev-2 (2 teams/teacher) at an identical bias share and did **not** rob. The chain
`BREADTH → generalizable content → externality flips ROBBERY→GIFT` survives only if the floor
reading (§3) is what happened; the failure is real either way and should be litigated at rev-4
rather than absorbed.

## Reproduce

```
export PYTHONPATH=$PYTHONPATH:src
nice -n 15 python designs/research_state/measurements/rev3_untaught_pulldown.py \
  --n 200 --teams 8 --shard 0/2 \
  --teams-json designs/research_state/measurements/rev3_untaught_pulldown_selection.json \
  --out /tmp/probeq/s0 --resume /tmp/probeq/s0_cells.jsonl
# ... --shard 1/2 in a second process; then merge the two *_cells.jsonl and re-run with
# --shard 0/1 --resume <merged> to emit the report json.
```

Wall clock: **6,181 s** (shard 0) / 5,545 s (shard 1) on a contended box (load 19–32, nine agents
live), 1.5–4.3 s/battle. Two cores, `nice -n 15`, BLAS pinned to one thread, `models/` read-only.

## Cuts

- **Team count 8, not 10** — sized from the pilot's throughput against the 5h budget on a box at
  load ~31. The pooled CI (±3.8pp) is comfortably inside what the question needs.
- **`n = 200`, not extended.** Fixed from the pilot before any team's data existed, per the
  no-optional-stopping call adopted in the rev-3 recap.
- **Untaught set drawn from the curated 32, not the 719 pool** — see § Team selection. Cost: the
  set is balance/stall-heavy and hyper_offense is unrepresentable (all four are taught).
- **The v8 point is CITED, not measured.** Probe P owns it; re-measuring it here would have been
  a second, worse instrument on the same question.

