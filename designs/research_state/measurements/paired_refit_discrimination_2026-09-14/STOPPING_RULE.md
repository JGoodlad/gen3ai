# The wall-clock stopping rule for the rung-B cells — registered 2026-09-15 02:20 UTC

**Registered BEFORE any rung-B cell had a readable estimate.** At the moment this was written the
four Part A `defB` cells had played **6 / 6 / 12 / 10 orientation-games** (3–6 side-swap pairs) —
`report.py` on them prints 0.5000 at n = 3 and 0.6250 at n = 4, which are not estimates of
anything. The `grid` phase was already complete at its registered 100 pairs and is not affected by
this rule.

## Why a rule is needed

The box carries a live training arm (`ai_v13_01_flywheel_shaped`, arm S) and its eval workers;
measured load average through the `grid` phase was 23–40 on 16 cores. At that contention the
rung-B cell costs ~0.9 orientation-games per minute per shard, so the registered 100 pairs
(200 orientation-games) per head is ~3.7 h **per phase**, and Part C needs its own pair of cells in
its own window.

## The rule

1. **Part C is launched as soon as the refit head exists and runs CONCURRENTLY with Part A's
   `defB`.** Every within-part contrast stays contemporaneous and width-comparable, which is what
   rule 23 requires; no contrast is ever taken ACROSS the two parts' cells.
2. **All rung-B cells stop at 05:30 UTC**, or when a cell reaches its registered 100 pairs,
   whichever comes first.
3. **Every cell is reported at its REALIZED n, with its realized K**, and the reading is stated as
   *"not detected at ±<the realized half-width>"* — never as "no effect".
4. The rows are append-only, so a stopped cell is a smaller cell and never a corrupted one.

This is an OUTCOME-BLIND rule: it is a function of the clock and of the registered n, and of
nothing that has been measured.
