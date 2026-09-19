# AMENDMENT 2 — the battery is REDUCED, and the proxy becomes the row that scales

*Registered 2026-09-19 (coordinator instruction), before any reported battery number. Landed on
main before the read. [`PREDICTION.md`](PREDICTION.md) (`ca52dc7f`) and
[`AMENDMENT.md`](AMENDMENT.md) (`b477e74c`) stand and are scored as written.*

---

## 1. The arithmetic that closes the 400-pair cell

AMENDMENT 1 measured **~1,200 s per battle** at `--budget 20`. That figure was taken on a cell whose
`--playoff-rollouts 4` was **INERT**: the per-decision deadline bought exactly ONE pair
(realized R = 1.00), so the battle paid for 64 playoffs × **2** rollouts. **A cell that actually
realizes R = 4 pays for 64 × 8**, and the measured per-rollout cost on this box is **~10 s**, so a
decision costs `screen (~1.4 s) + 8 × 10 s ≈ 82 s` and a 75-decision battle **≈ 5,500 s**.

| cell | s / battle | 400 pairs = 800 battles | ≥ 60 pairs = 120 battles |
|---|---|---|---|
| `pf4` — `--budget 20`, realized R = 1.00 (a no-op) | ~1,340 | **298 CPU-h** | **44.7 CPU-h** |
| `pf4b` — `--budget 120`, realized R = 4 (the real cell) | ~5,500 | **1,222 CPU-h** | **183 CPU-h** |

**The hard stop leaves ~3–4 h of a heavily shared 16-core box** (a training arm at `nice 10` plus
two other agents; measured load 34–60 throughout). **≥ 60 pairs is not reachable for either cell,
by a factor of 10–50.** Registering the number rather than the wish is the point: the instruction's
floor is recorded as UNMET, with the arithmetic that makes it unmeetable here.

## 2. What is registered instead

1. **`pf4b` runs to whatever pair count fits the hard stop.** Its **L2 is reported as a
   DESCRIPTOR, explicitly labelled "UNDER-POWERED BY DESIGN"**, with its Wilson interval — and it
   is NOT used for any branch call. Rule 25's 400-pair bar is recorded as unmet.
2. **THE BATTERY-PROXY ROW IS THE OFFLINE INSTRUMENT** — the playoff's OWN decision rule
   (`playoff.paired_stats` / `is_conclusive` / `decide` / `SE_MULTIPLE` / `MIN_PAIRS`, **imported,
   never re-implemented**) applied to JOB 1's CRN-paired rollouts at R ∈ {1, 2, 4, 8, 16}:
   the **conclusive rate** and the **agreement with the K′ = 8 label when it concludes**, on every
   banked fork. That row costs nothing, scales with JOB 1, and is a thousand-fold better resolved
   than any live cell affordable today. Its two declared differences from the live cell and their
   directions are in AMENDMENT 1 §2.3 and are restated beside the table.
3. **THE BRANCH CALL RESTS ON THE K-CURVE + THE OFFLINE PROXY.** The live cell contributes the
   mechanism row (screen / played / inconclusive, realized R, wall per adjudicated decision) and
   the honest cost, and nothing else.
4. **No further battery cell is attempted.** The 400-pair playoff battery is closed as
   OUT OF BUDGET, with §1's arithmetic as the reason.

## 3. The two findings are filed, and one is guarded

Both AMENDMENT 1 findings are now rows in
[`designs/ops/TECH_DEBT_BACKLOG.md`](../../../ops/TECH_DEBT_BACKLOG.md) — the rust breakage (P1,
with the silence in `diag["playoff"]["error"]` named as a separable fix), the stale
`run_local_battles` docstring (P2), and the inert-`--playoff-rollouts` defect (P2).

**A THROWING GUARD LANDS for the second.** `playoff.short_r_refusal(row, requested)` returns a
refusal whenever a `playoff` cell's realized mean R falls below `0.9 ×` the requested R, and
`battery.run_cell` raises `SystemExit` on the **FIRST game** — after appending the row, so the
evidence for the refusal is on disk — unless `--playoff-allow-short-r` is passed. A cell with no
playoffs at all is not short (that is a decisive screen, a reported outcome) and is never judged.
Six tests, including the measured incident as a fixture and an assertion that the two cells which
differed only in a requested R the budget could not buy are separated by the guard and by nothing
else.

## 4. Predictions, registered before the numbers

| # | prediction |
|---|---|
| **C1** | `pf4b`'s realized mean R is **≥ 3.6** (the guard's floor), confirming that `--budget 120` buys the requested R where `--budget 20` did not |
| **C2** | `pf4b`'s `played` rate stays **below 5 % of all decisions** — the live analogue of A2 |
| **C3** | `pf4b`'s L2 lands anywhere in **0.3–0.7** at its pair count; the interval is ±0.2 or wider and the row is a descriptor, not evidence |
| **C4** | the guard, run against the banked `pf4`/`pf8` rows, REFUSES both and passes `pf4b` |
