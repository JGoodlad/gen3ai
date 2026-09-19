# AMENDMENT 1 — registered 2026-09-19 ~09:00 UTC, after JOB 2's SMOKE and before any reported battery number

*The pre-registration in [`PREDICTION.md`](PREDICTION.md) stands and is scored as written. This
amendment records what a two-game smoke of the `playoff` arm MEASURED, and the design change it
forces. It was landed on main before the first battle of any reported cell. JOB 1 is UNCHANGED and
was already running when this was written.*

---

## 1. Three measured facts from the smoke (2 games, arm W, `--opponents self`, `--games-seed 7`)

**F1 — 🚨 `--impl rust` BREAKS THE PLAYOFF'S NESTED COUNTERFACTUAL ROLLOUTS. `--impl node` is clean
on the identical game.** One game under `--impl rust`: **250 failed rollouts, 63 of 75 decisions
lost their playoff** and were counted as `playoff_no_budget` fallbacks. The exception the arm
swallows (surfaced by wrapping `PlayoffRunner._live_rollout`, not by editing `src/`):

```
RuntimeError: local_sim_bridge error: unresolvable choice for p1: MoveName("crunch")
  — active metagross has moves ["Meteor Mash", "Hidden Power Fire", "Psychic", "Explosion"];
    bench ["tyranitar", "skarmory", "starmie", "jolteon", "salamence"]. Re-requesting would
    loop forever, so this fails loud.
```

The substituted token is a move of a BENCHED mon, so the replayed prefix has arrived at a different
board than the live decision was taken at — a dice divergence inside the scripted prefix, on the
same team. The SAME game under `--impl node` raised **0** exceptions and ran **64 playoffs**.

⚠️ **The failure is silent at the results-row level.** `fold_playoff` counts the decisions as
`n_playoff_no_budget` and `n_playoff_failed`, and `fallback_details` stays EMPTY because the
playoff's error string lives in `diag["playoff"]["error"]`, which no row field carries. A cell run
this way would report a clean-looking win rate for an arm that never adjudicated anything.

**Scope, stated honestly:** the identical rust path is clean on **48,096 offline rollouts** in
JOB 1 (0 errors), so it is not "rust cannot replay counterfactuals". The two differences between
the two callers are named as CANDIDATES, neither proven here: (a) JOB 2's record is a **LIVE,
partial** record built mid-battle while JOB 1's is a complete eval-trace record; (b) the live
record's seed is `sodium,<32 hex>` while the eval traces' are the `m,n,o,p` form. No root cause is
claimed. **The arm has been run as a cell for the first time today** (one commit, `000709ae`, no
prior measurement), so nothing was regressed — it has simply never been exercised on this path.

⚠️ **A second, separate documentation defect found on the way:** `run_local_battles`' docstring says
*"The Rust binary emits no `__RECON__`, so `start_extra`'s `resumeReseed` + the reconstruction join
degrade to no-ops under `rust`"*. That is **STALE** — `src/rust_sim/src/bin/sim_bridge.rs`
implements both (`gen3_bridge_resume_reseed_v1`), which is exactly why JOB 1's 48,096 reseeded
rollouts under `rust` produce real dice variation. A reader who believed the docstring would either
avoid a valid path or distrust a valid measurement. **No `src/` change is made in this record; both
are reported.**

**F2 — 🚨 AT R = 4 THE PLAYOFF IS A NO-OP. 64 of 64 run playoffs were INCONCLUSIVE and
`n_changed = 0`.** The gate is `|mean(d)| >= 2·SE(d)` over R paired differences in
{−1, −0.5, 0, +0.5, +1} with `MIN_PAIRS = 4` and an SE floored at `0.5/n`. At n = 4 that needs
roughly **three of four pairs to disagree decisively and in the same direction**; under common
random numbers most pairs return `d = 0` (both branches reach the same outcome), so it essentially
never fires. **Prediction B1 registered ">= 60 % inconclusive at R = 4" and the measured value is
100 %.**

**F3 — 🚨 THE COST IS ~1,200 s PER BATTLE.** One clean `--impl node` game: **1,199.9 s** for 75
decisions — 64 playoffs × 8 rollouts = **512 terminal rollouts in one battle** — at `nice 15` on a
box at load ~44 carrying a training arm and JOB 1. **400 pairs = 800 battles ≈ 267 CPU-hours for
ONE cell**, before either control. Rule 25's 400-pair bar was written for cells costing ~10–60 s a
battle; it is not affordable here and saying so is part of the finding.

## 2. What changes, and why

**A cell whose override rate is ZERO has an L2 that is the harness's own null BY CONSTRUCTION, and
measuring it at any n measures the harness.** The standing demonstration is the 2026-09-11 `defA`
cell: a literal no-op arm (`n_changed` exactly 0 across 37,909 decisions) read "0.51-ish" at
n = 150. Spending 267 CPU-hours to re-learn that would be the least informative purchase available.

So, registered:

1. **THE PRIMARY READ OF JOB 2 BECOMES THE MECHANISM ROW (BAR B3)** — `screen_decisive` /
   `played` / `inconclusive`, the overrule rate, the realized R, and the wall-clock per decision.
   BAR B1 (L2's lower bound > 0.50) is still reported, at whatever width the clock buys, and is
   **declared UNDERPOWERED against rule 25's 400-pair bar with F3 as the reason**.
2. **The battery runs on `--impl node`** (F1), at **R = 4** and, clock permitting, **R = 8**, over
   as many pairs as the window allows, with **both contemporaneous controls on the SAME game
   indices in the SAME window** (rule 23) — unchanged from §3.1 of the registration.
3. **A NEW, ZERO-COST INSTRUMENT: the playoff's own decision rule, applied offline to JOB 1's
   paired rollouts.** For re-roll k the three branches share ONE seed, so
   `d_k = score(top1,k) − score(top2,k)` is exactly the statistic `PlayoffRunner.adjudicate`
   forms. `playoff.paired_stats` / `is_conclusive` / `decide` / `SE_MULTIPLE` / `MIN_PAIRS` are
   IMPORTED, not re-implemented, and evaluated at R ∈ {1, 2, 4, 8, 16} on every fork.
   **Two declared differences from the live cell, cutting in opposite directions:** the offline
   rollouts are GREEDY against the banked SENTINEL (the live ones are the same network STOCHASTIC
   at temperature 1, which adds variance ⇒ the offline rate is an UPPER bound), and the offline
   pair is the POLICY's top-2 while the live pair is the SCREEN's top-2 (possibly further apart
   ⇒ easier to resolve, which cuts the other way). It is reported as a bound with a named
   direction on each side, never as a prediction of the live number. What it settles cleanly is
   the SHAPE — how the rate moves with R.
4. **JOB 1's fork count may be cut by the clock.** The measured rate under this box's load is
   ~3.8 min per fork per worker against the 1.85 min the banked K′ = 8 pass ran at. The loop is
   FORK-OUTER / K-INNER, so a stop yields **fewer forks at FULL K = 16**, never all forks at a
   ragged K; the realized fork and non-tied-pair counts are published beside every level.

## 3. The amendment's own predictions, registered before the numbers

| # | prediction |
|---|---|
| **A1** | the OFFLINE conclusive rate at R = 4 is **< 0.15**, rises monotonically with R, and reaches **0.25–0.55** at R = 16 |
| **A2** | the LIVE cell at R = 8 still overrules on **< 5 % of all decisions** |
| **A3** | L2 straddles 0.50 at both R — unchanged from B2, now for a mechanically stronger reason |
| **A4** | when the offline rule DOES conclude, it agrees with the K′ = 8 label on **0.60–0.80** of the pairs it resolves — better than the head's 0.5610 on that column, because it is conditioning on the cases the dice resolved |
| **A5** | the rust defect does NOT reproduce on an offline eval-trace record: JOB 1 finishes with `errors = 0` across its 48,096 rollouts under `--impl rust` |
