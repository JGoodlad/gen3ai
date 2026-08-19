# design — THE TICK-TOCK FLYWHEEL: the exploiter–generalist loop, decisions of record

> **[STATE 2026-08-18]** Owner design decisions recorded from live review; this is the
> OPERATIONAL design of the flywheel era. The science it runs on is
> [`design_exploiter_scaling.md`](design_exploiter_scaling.md) (§9's drift-vs-noise recipe, the
> battery); the gates it starts from are
> [`../research_state/substrate_exploiter_gates.md`](../research_state/substrate_exploiter_gates.md).
> Sequencing is unchanged: the era runs ON the mechanics-generation base (substrate-before-
> flywheel, README → Programme sequencing). NOTHING here is built yet; the automation driver is
> specced by this doc and built during gen-16's run.

## 0. The owner decisions (2026-08-18, verbatim intent)

| # | decision |
|---|---|
| D-A | **The exploiter-gate arms double as cycle-1 teachers.** The nine-gate battery's ON arms that PASS their gates fold into the first distill — the gates are not pure measurement. |
| D-B | **Coverage ambition: ~50 teams** (owner: "just tossing out a number" — a target, not a bar). |
| D-C | **Slice selection = SIMILARITY GROUPING, owner-in-the-loop**: the categorized sample teams are the archetype ANCHORS; pool teams are grouped to their nearest anchor; the owner curates the groups. |
| D-D | **Fold trigger: delegated** (owner: no strong opinion) → the default stands: task-arithmetic preview (grow the batch until averaged deltas interfere, then fold), K ≈ 3 as the starting point. |
| D-E | **Distill data: teachers play the FULL GENERALIST POOL**, not their target slice — generalization pressure at collection time. |
| D-F | **Distillation is ALWAYS full-distribution**: the aux loss targets the teacher's whole policy distribution (KL), never hard actions — "dark knowledge is very rich." |
| D-G | **TICK-TOCK cadence**: **tick** = generalist trains (RL self-play) AND distills the accumulated teachers; **tock** = exploiters fork from the tick's output and train on their slices. |

## 1. The loop, one full revolution

```
        ┌────────────────── TICK (generalist) ──────────────────┐
        │  RL self-play continues  +  KL-distill aux from the   │
        │  banked teachers' full distributions (D-F), collected │
        │  on full-pool games (D-E). Ends at the promotion      │
        │  gate: dense-ladder non-inferiority vs the previous   │
        │  base, else the fold is rejected and diagnosed.       │
        └──────────────┬────────────────────────────────────────┘
                       │ new base (the only artifact that crosses)
        ┌──────────────▼──────────── TOCK (exploiters) ─────────┐
        │  K exploiters fork the new base (warm), each on an    │
        │  owner-curated slice of ~5–10 similar teams (D-C,     │
        │  N≤10 per D4). Task-arithmetic preview decides the    │
        │  fold moment (D-D). Per-teacher behavioral readouts   │
        │  recorded (the gate pattern).                         │
        └───────────────────────────────────────────────────────┘
```

Q5 (what the base does between folds) is RESOLVED by D-G: the base is never idle and never
drifting under the forks — tick and tock alternate on the one GPU; exploiters always fork the
most recent tick's output.

## 2. Coverage arithmetic to ~50 (D-B)

The gates cover 8 distinct pilot teams (E1–E3). At ~7 similar teams per curated slice and K=3
exploiters per tock: gates(8) → tock-2 (+~21) → tock-3 (+~21) ≈ **50 teams by the third
revolution**, roughly 4–6 box-days of tock compute plus tick time. The number is a target;
T1.5's archetype-novelty regression (does gain track novelty?) is the standing check on whether
marginal slices still pay as coverage grows.

## 3. The grouping workflow (D-C — owner-in-the-loop)

1. A draft grouping is generated mechanically: each of the 719 pool teams scored against the
   ~33 sample-team anchors by pace-class match + style-tag overlap (`gen3_team_archetypes.json`)
   + species overlap; every pool team assigned to its nearest anchor; groups emitted as a
   readable worksheet (anchor → members, with the scores shown).
2. **The owner curates** — merges, splits, vetoes, renames. The curated grouping is committed as
   an artifact (the `pin_sha` convention) and becomes the slice registry the tocks draw from.
3. Slices are drawn to keep within-slice similarity HIGH (the drift-dominated regime needs
   coherent pressure) and cross-tock coverage growing toward D-B.

## 4. Distillation mechanics (D-E, D-F)

- Collection: each banked teacher plays the full generalist pool (bridge, no server); states +
  the teacher's FULL action distribution recorded. (The OPD machinery — `--opd-coef`, built
  2026-07 — is the delivery seam; coefficient and mixing per-teacher weights are driver knobs.)
- The tick's loss: PPO (self-play) + `Σ_teachers w_t · KL(π_teacher ‖ π_student)` on the
  teacher-collected states. Never hard actions (D-F). Teacher data is REPLAYABLE — the per-team
  restoring force is a data-mixing dial, which is why the fold has no 1/N wall (§9).
- Teachers are RETIRED after their fold banks (D2: ~76% retention without life support); the
  bank keeps the newest fold's teachers only, unless a slice's piloting decays below its floor
  (then its teacher returns for one revolution — the leaky-bucket refresh).

## 5. Gates and kill conditions (standing, from the prior reviews)

- **Per-fold promotion gate**: dense-ladder non-inferiority vs the previous base at matched
  snapshot count; a failed fold never becomes the base; every base snapshot retained.
- **Per-teacher attribution** at each fold (piloting delta on its slice); the K dial moves on it.
- **SUCCESS — the owner's definition (2026-08-18): TWO distill iterations with ELO
  improvement.** "I want to show the flywheel is working and the generalist is getting better."
  Measured form, fixed now so noise can't renegotiate it: each revolution's fold must be
  point-POSITIVE on the dense ladder vs the previous base (matched snapshot count, paired
  refit), and the TWO-revolution CUMULATIVE delta must be SEPARABLE (CI excluding 0). Per-node
  SE ≈ 10 makes single-revolution separability optional; the cumulative bar is the claim. The
  D1 precedent says the bar is realistic: one fold measured +69 with disjoint CIs.
- **The HEADROOM instrument (the owner's framing)**: exploiters run ~100% vs bots where the
  generalist runs ~90% — bots are not the goal, but the gap PROVES meaningful headroom exists.
  Formalized per fold as **headroom capture**: on each slice, headroom = teacher_wr −
  generalist_wr (equal-pilot, same opponents); after the fold, report the fraction captured
  (post-fold generalist wr − pre-fold, over the headroom). D1's precedent: 0.438 → 0.710
  against a teacher's 0.72 ≈ 93% captured. A fold that passes promotion but captures little
  headroom is a WARNING even when ELO drifts up — the flywheel's mechanism is headroom
  conversion, and this metric watches the mechanism, not just the outcome.
- Supporting metrics: anchored-ELO trajectory across bases; per-slice piloting; T1.2
  compounding (do later revolutions get cheaper?).
- **Kill**: two consecutive revolutions with flat ELO AND flat piloting ⇒ the flywheel is not
  the lever; the era ends and the ledger says so.

## 6. What the driver automates (built during gen-16)

Slice assignment from the curated registry · fork launches with pre-seeded pools (the
empty-pool guard's lesson) · the task-arithmetic preview · teacher collection runs · the tick's
distill-mix config · the promotion gate + computed per-revolution report into
`research_state/measurements/`. The launch-diff gate wraps every launch. Owner touchpoints per
revolution: the slice curation (§3) and the fold go/no-go if the preview is ambiguous.

## 7. Open items this doc does NOT decide

Tick length (steps per tick — likely the existing 25M-generation rhythm shortened; decide from
gate-arm evidence) · the distill coefficient schedule · whether the win-prob/belief aux family
needs re-weighting during distill-heavy ticks · the fingerprint aux (still deferred; its stage-1
offline form would make §3's similarity scoring strategy-aware instead of composition-only —
revisit after revolution 1).
