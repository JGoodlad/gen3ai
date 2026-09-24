# wcont_b (seed 1002) — the registered completion read (2026-09-23)

**What this is.** The SECOND SEED of the win-prob-era continuation control. `ai_v13_21_wcont_b`
is arm W (`ai_v13_02_flywheel_winprob`) +12.09M, no teachers, seed 1002, pin `6eb9c776`, frozen
dose 4.272e-9 — seed 1001's `ai_v13_09_wcont` with only `--seed` and `--run-name` changed. Read
with the SAME recipe, the SAME floor and the SAME rule as seed 1001's +15.50 pp row
(`../wcont_control_read_2026-09-20/`), so the two seeds pair into one family claim.

## Verdict

> **REPLICATED — two seeds, one claim.** The plain win-prob continuation, with no teacher, moves
> the untaught-8 meter **+12.94 pp [+10.25, +15.44]** over its frozen parent at +12M (seed 1001:
> +15.50 [+12.00, +18.88]), **OUTSIDE the 3.69 pp floor by both clauses at all three depths, 8 of 8
> teams at every depth**. Seed 1002 lands **2.56 pp lower** than seed 1001 at +12M
> ([−5.06, −0.06], 3 of 8 teams favour it) — a seed spread smaller than the floor itself. The
> SmallRL greedy away anchor reads **0.710** [0.615, 0.790] — descriptive only: a 100-game cell is
> dominated by seed spread, and the continuation's EXTERNAL size is the n = 1,200 A/B's ~+0.03, not this.

Tag: **MEASUREMENT · untaught +12.94 pp @ +12M, OUTSIDE floor ×3 depths, 8/8 teams · seed pair
+12.94 / +15.50 · anchor 0.710 (descriptive, WITHIN FLOOR at n = 2)**.

## Row 1 — untaught 8 vs arm W (Δ = path − arm W, paired bootstrap over teams, seed 20260915)

| depth | step | level pp | Δ vs arm W | CI95 | teams ahead | (a) | (b) | verdict | seed 1001 Δ |
|---|---:|---:|---:|---|---:|---|---|---|---:|
| +3M | 78,006,048 | 59.44 | **+13.25** | [+8.94, +17.37] | 8/8 | ✅ | ✅ | **OUTSIDE THE FLOOR** | +12.31 |
| +6M | 80,912,688 | 58.75 | **+12.56** | [+9.62, +15.56] | 8/8 | ✅ | ✅ | **OUTSIDE THE FLOOR** | +14.25 |
| +12M | 87,097,344 | 59.13 | **+12.94** | [+10.25, +15.44] | 8/8 | ✅ | ✅ | **OUTSIDE THE FLOOR** | +15.50 |
| +0 | arm W | 46.19 | — | — | — | | | | — |

*Floor 3.69 pp (IMPORTED, |arm W − W_b|, the 75M fresh-arm seed pair) — the same floor and caveat
as seed 1001's read. OUTSIDE iff |Δ| > floor AND the Δ's CI excludes the floor point.*

⚠️ **+6M is not step-matched to seed 1001** — 80,912,688 here vs 81,143,280 there (230,592 steps;
the checkpoint grid differs by seed's restart timing). Declared, not corrected; +3M and +12M are the
same step numbers on both seeds.

**Contrasts:** seed 1002 − seed 1001 @ +12M **−2.56** [−5.06, −0.06], 3 of 8 teams. Seed 1002's
own +12M − +3M **−0.31** [−3.25, +2.94] — **all of seed 1002's gain is there by +3M**, the same
front-loaded shape seed 1001 showed (+12.31 of +15.50 by +3M), here even more so.

### Reproduction check (gates the comparison) — ✅ EXACT

| ref | banked | here | per-team rows identical |
|---|---|---|---|
| arm W | 46.19 pp (739/1600) | **46.19 pp (739/1600)** | ✅ 8 of 8 |
| seed 1001 `wcont_p12M` | 61.69 pp (987/1600) | **61.69 pp (987/1600)** | ✅ 8 of 8 |

Seed 1001's +12M was re-played in this read as a second check; it reproduced on every row, so the
paired seed-vs-seed contrast is taken on the same games.

## Row 3 — Metamon SmallRL, greedy vs greedy, AWAY team set, 100 games (`main.anchors`)

| arm | win rate | Wilson 95 % | half-cells |
|---|---|---|---|
| **`ai_v13_21_wcont_b` @ +12.09M (seed 1002)** | **0.710** | [0.615, 0.790] | 0.68 / 0.74 |
| `ai_v13_09_wcont` @ +12.09M (seed 1001, banked) | 0.640 | [0.542, 0.727] | 0.58 / 0.70 |
| arm W — the parent (banked) | 0.500 | [0.404, 0.596] | — |

| contrast | Δ | Newcombe 95 % | floor | (a) | (b) | verdict |
|---|---|---|---|---|---|---|
| seed 1002 − arm W | **+0.210** | [+0.075, +0.335] | 0.090 | ✅ | ❌ | WITHIN FLOOR at n = 2 |
| seed 1002 − seed 1001 | +0.070 | [−0.059, +0.196] | 0.090 | ❌ | ❌ | WITHIN FLOOR at n = 2 |

Registered DESCRIPTIVE, as for seed 1001: 100 games cannot clear a 0.090 run-level floor by clause
(b). Per-decision regime instruments: their `argmax_match_rate` 1.0000, 0 forfeits at the 250-turn
cap, 0 ties. 🚨 **Not a strength claim.** Ledger 2026-09-20 (*THE CONTINUATION'S +15.50 pp DOES NOT APPEAR
EXTERNALLY*) showed one arm's 100-game sub-cells ranging 0.34–0.59 across twelve seeds, and the
n = 1,200 A/B put the continuation at +0.039 [−0.000, +0.079] over arm W. This cell is recorded
because it was registered, and its direction agrees with row 1; its size is not quotable.

## How it was run

- **Code tree:** a worktree at `58389caa` (the tree seed 1001's read used), NOT today's main —
  main carries the `65f22334` training-input (obs) boundary, and the arm-W reproduction check is
  what licenses the choice. `POKESIM_SIM_BRIDGE_BIN` pointed at that worktree's own build.
- **Untaught meter:** `main.untaught_meter`, opponent registry `untaught_meter_opponent`,
  200 games/team, seed 0, concurrency 1, 8 workers, CPU, `nice 15` — **8,000 battles, 0 timeouts**.
- **Anchor:** `main.anchors --regime greedy --teamset away --games 100 --server rust`, own server on
  :9451, stopped by PID by the tool. The GPU was never touched; :8000 / :8001 were never touched.
- The box carried population-loop round-1 arm B (`ai_v13_22_popr1_loop`) throughout.

| file | holds |
|---|---|
| `scripts/harness/run_untaught.sh`, `run_away_cell.sh`, `run_all.sh` | the exact commands |
| `scripts/wcontb_delta.py` | the paired contrasts — imports seed 1001's `paired`/`verdict`/`newcombe` VERBATIM |
| `out/untaught/untaught_registry.{json,md}` | the meter's output (5 refs × 8 teams) |
| `out/anchors/away/wcontb_p12M_smallrl_greedy_away/` | the anchor cell: games, summary, both halves |
