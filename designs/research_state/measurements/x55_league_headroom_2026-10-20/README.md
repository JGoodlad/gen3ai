# X55 / LG-0: the league HEADROOM read (CPU, 2026-10-20)

**Status: READ.** Design: [`design_alphastar_league_path.md`](../../../endstate/design_alphastar_league_path.md) §6 row
LG-0 (backlog row AS-1 / X55). The predictions were registered and pushed FIRST, as their own commit (`798c4f2a`,
[`PREDICTIONS.md`](PREDICTIONS.md)), before any X35 production arm exists and before any game below was played (the
first game started at 10:21 PDT, the predictions were on `origin/main` at 08:20). No GPU; `models/` only read;
`data/` untouched.

**The question.** How much does an AlphaStar-style league have to work with at X26's plateau? Concretely: where the
archive's Nash weight sits, how much of it A1's evenly spaced archive (and R's A-block) holds, how P\* scores against
those nodes, and how much PFSP could reweight them.

## The answer, in one screen

1. **The archive's Nash sits inside C0's own pool.** The posterior Nash over X26's archive to 268M (66 ladder nodes)
   puts 0.90 of its mass on the 14 snapshots C0 is seeded with (164–268M) and 0.98 at ≥ 130M. Point support 6 nodes
   (194, 224, 244, 246, 254, 268M); posterior support 7.5, 90 % [5, 10]; effective number of nodes 9.4.
2. **A1's archive barely overlaps it.** Ladder nodes within ±5M of A1's eight nodes (30, 60, 100, 130, 160, 190, 230,
   260M: the frozen R file, not the league doc's "20–230M") hold 0.19 of the mass: 0.17 by the late four, 0.014 by
   the early four. R's A-block neighbourhoods hold 0.21.
3. **The top is flat over almost the WHOLE archive, and P\* has no detected hole in it.** P\*'s measured row (18
   cells, 5,000 games; table below): 0.61 against 30M, 0.57 against 60M, 0.56 against 100M, and 0.45–0.59 against
   every node from 130M to 270M (11 of those 14 cells have 0.5 inside their 95 % interval). No cell's 95 % interval lies below 0.5; the worst is the 244M snapshot, 0.450 [0.384, 0.516]. The
   transitive gain from X26's 30M node to P\* is about +11 pp of score.
4. **P\* is (within noise) a member of the archive's Nash cluster.** Its score against the posterior Nash mixture,
   on the six measured support snapshots (0.74 of the posterior mass), is **0.514 ± 0.014**. So the archive holds no
   strategy that beats P\* by more than a few pp: the in-archive "exploitability" of P\* is ≈ 0–5 pp.
5. **PFSP has almost nothing to tilt.** Stable PFSP (`max(0.05, 1 − p)`) over A1's eight nodes at the measured rates
   gives shares 0.85–1.10× uniform (max / min 1.30; 3.7 % of the archive slice moves). A2's factor over C0's pool
   (from the in-run sentinels) gives 0.97–1.05× (1.2 % moves). No A1 node is near the 0.80 mastery floor (max 0.625 at
   80M, an R node; 0.61 at 30M), so all eight stay live for A1's whole run.
6. **S(P\*) = 0.569 ± 0.009** (R's 7 members; the prediction was 0.567). Every X35 production arm starts there.
7. **The ladder's Bradley–Terry spine overstates long-range gaps.** Chained over 2M edges it puts the 268M node at
   0.733 against 30M and 0.64 against 60M; P\* measures 0.61 and 0.57. Long-range edges are flatter than the chain of
   short ones, so any Nash solved on spine-filled edges (the cycle monitor's, this read's part 1) under-weights old
   nodes. It does not change point 4: no old node beats P\*.

**What it means for the league (LG-0's two decisions).**
- **LG-2's "second arm" (re-pick the archive from the Nash support) is MOOT:** the support is already C0's recent
  pool, so re-picking it is C0 with a reweighted pool. And PFSP over A1's archive is near-uniform (max / min 1.3), so
  LG-2 ≈ A1 plus noise. Hardness weighting has no hardness to find here.
- **The prior for batch 1** is the registered [`PREDICTIONS.md`](PREDICTIONS.md): A1 Δ_pop +0.5 pp (80 % [−1.0,
  +2.0]), modal look 1 CONTINUE then FINAL UNDECIDED; A2 0.0 pp; the toy calibration NOT SHOWN. The read below does
  not move them (they are registered and stay as written); it says A1's early nodes are less "easy" than predicted
  (0.57–0.61, not 0.63–0.68), which if anything narrows the archive's diversity further.
- **Where a population lever can still move:** exploitability against a fresh best responder (R0 / R2; LG-1), which is
  not bounded by anything measured here, and team diversity (D1). Not the archive.

## Part 1 — existing games only (`headroom_existing.py` → `headroom_existing.json`)

Inputs (read-only): X26's in-run snapshot ladder (`models/rb_x26_s2601/snapshot_ladder/games.jsonl`, the 1,036
edges / 65,250 mirrored pairs among the 66 nodes ≤ 268M, regime `adfdbee824c9eb0a`), its in-run eval rows
(`eval_results.jsonl`: the trainee vs 5 greedy pool sentinels each 2M, 100 games each) and X35's frozen R file.

| quantity | archive to 268M (66 nodes) | last 80M (188–268M, 11 nodes) |
|---|---|---|
| point LP Nash support (> 0.01) | 6: 194, 224, 244, 246, 254, 268M | 6 (same) |
| posterior support, mean [5 %, 95 %] | 7.5 [5, 10] | 5.2 [3, 7] |
| posterior weight, effective n | 9.4 | 7.1 |
| mass on C0's seeded pool (164–268M) | **0.90** | 1.00 |
| mass at ≥ 130M | **0.98** | 1.00 |
| mass within ±5M of A1's nodes (all / early 4 / late 4) | **0.19** / 0.014 / 0.17 | 0.15 / 0 / 0.15 |
| mass within ±5M of R's A-block | 0.21 | 0.17 |

Posterior: 500 Dirichlet(c + ½) draws per edge, an LP Nash per draw (the monitor's `nash_lp`, unplayed edges from the
spine). Highest posterior mean weights: 268M 0.18 (P(in support) 0.91), 244M 0.14 (0.83), 254M 0.12 (0.75), 194M
0.12 (0.83), 246M 0.10 (0.62), 224M 0.07 (0.64). The MAX-ENTROPY Nash equals the LP point on the window (SLSQP, solver
ok: the Nash is unique there); on the 66-node archive SLSQP did not converge and the LP point is reported (F-X55-5).

**The P\*-neighbourhood proxy** (the trainee at 270–290M vs pool sentinels, 900 games each): 134M 0.538, 160M 0.546,
194M 0.491, 236M 0.509, 268M 0.510. **A2's tilt** at those rates: shares 0.97–1.05× C0's, total variation 1.2 %.

## Part 2 — P\*'s measured row (`play_pstar_row.sh`; read by `read_row.py` → `row.json`, `row_read.json`)

P\* = `rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip` (sha256 `814823c5…91ed`, checked by the script) against
X26 nodes, `main.h2h play-many`, greedy vs greedy, mirrored pairs, oracle reveal off, regime `adfdbee824c9eb0a` (the
ladder's and the R freeze's), CPU, into a scratch ledger (`~/.cache/gen3ai/x55_league_headroom/ledger`). Rule 5b:
`mem_cap.sh 8 timeout 2700 nice -n 19 taskset -c 5-7,13-15`, each chunk in a heavy slot, ≤ 45 min (five chunks,
10:21–14:30 PDT; 0.34–1.0 games/s on 3 threads beside the live X35 toy arm).

| X26 node | role | pairs | P\* score [95 %] | P1 prediction | inside ± 2 SE | spine (268M vs nearest ladder node) | in-run sentinel proxy |
|---|---|---|---|---|---|---|---|
| 30M | A1 N_1 | 200 | 0.608 [0.561, 0.654] | 0.68 | **no** | 0.733 |  |
| 60M | A1 N_2 | 200 | 0.573 [0.524, 0.621] | 0.63 | **no** | 0.640 |  |
| 80M | R A-block | 100 | 0.625 [0.562, 0.688] | 0.61 | yes | 0.593 |  |
| 100M | A1 N_3 | 200 | 0.564 [0.520, 0.607] | 0.60 | yes | 0.585 | 0.607 |
| 130M | A1 N_4 | 100 | 0.527 [0.466, 0.589] | 0.55 | yes | 0.572 |  |
| 140M | R A-block | 100 | 0.570 [0.506, 0.634] | 0.55 | yes | 0.563 |  |
| 160M | A1 N_5 | 100 | 0.490 [0.426, 0.554] | 0.55 | yes | 0.550 | 0.543 |
| 190M | A1 N_6 | 200 | 0.509 [0.462, 0.556] | 0.51 | yes | 0.526 |  |
| 194M snap | Nash snapshot | 100 | 0.593 [0.531, 0.654] | 0.49 | **no** | 0.526 | 0.496 |
| 210M | R A-block | 100 | 0.495 [0.426, 0.564] | 0.52 | yes | 0.543 |  |
| 224M snap | Nash snapshot | 100 | 0.502 [0.439, 0.566] | 0.53 | yes | 0.510 | 0.620 |
| 230M | A1 N_7 | 200 | 0.507 [0.458, 0.557] | 0.52 | yes | 0.509 |  |
| 244M snap | Nash snapshot | 100 | 0.450 [0.384, 0.516] | 0.54 | **no** | 0.522 | 0.545 |
| 246M snap | Nash snapshot | 100 | 0.575 [0.508, 0.642] | 0.51 | yes | 0.528 | 0.501 |
| 254M snap | Nash snapshot | 100 | 0.487 [0.422, 0.553] | 0.51 | yes | 0.527 | 0.509 |
| 260M | A1 N_8 | 200 | 0.510 [0.467, 0.553] | 0.52 | yes | 0.527 |  |
| 268M snap | Nash snapshot | 100 | 0.502 [0.442, 0.563] | 0.51 | yes | 0.500 | 0.510 |
| 270M | R A-block (P\* − 10M) | 200 | 0.506 [0.462, 0.551] | 0.51 | yes | 0.500 |  |

Pentanomial pair counts and digests are in `row.json`. The score's SE is `main.h2h`'s pair-clustered SE.

**P1, the registered row prediction: FAILED its pass mark** (14 of 18 cells inside the prediction ± 2 SE; the mark was ≥ 15). Misses: the two OLD nodes (30M, 60M), where
the prediction leaned on the steep spine and on X37b's home72 cell (0.688 against 25M), and two snapshot cells (194M
+10 pp, 244M −9 pp), where P\* differs from its 270–290M neighbours' sentinel rates by more than the cell's noise —
the per-edge cyclic component (≈ 3–5 pp of SD, `cycle_monitor_retro_2026-10-19/`) showing up on single edges. The
derived predictions held: S(P\*) 0.569 vs 0.567; no node mastered; none below 0.45 (the lowest is exactly 0.450).

## Part 3 — the predictions' simulator (`predictions_sim.py` → `predictions_sim.txt`)

R1's class probabilities under a predicted Δ and a σ prior (lognormal, median 1.0 pp, 10–90 % 0.70–1.43), the
registered two-look rule, cross-checked against `main.x35.readout.decide_r1` (0 / 2,000 mismatches). Reference rows:
a truly inert arm reads look 1 CONTINUE 0.45 / UNDECIDED 0.45 / NULL 0.10 and, after look 2, NULL 0.26; a true +3.0 pp
effect (the cyclic size) reads GAIN only 0.21 after look 2.

## Part 4 — F-AS-3, A1's collection cost (ESTIMATED; not measured on CPU)

Recorded in the X35 sheet as F-LS-25: +8 live opponents × (+1.20 ms of GPU wait per host step per snapshot, X26's own
T24 slope; or +1.55, the §1.6 bench) × ≈ 516 host steps per collect = +5.0 to +6.4 s on X26's 65.6 s iteration:
**+7.6 to +9.8 % wall time at equal steps**, on top of the declared ≈ 2 % for eval games, so ≈ 4.7–4.8 GPU-h per A1
seed instead of 4.4. A CPU measurement was not made because a CPU forward costs per row while the GPU per-slot path
costs per call, so a CPU wall would not transfer; A1 s3501's first 1M (`gpu_wait_ms_per_host_step` against C0's) is
the measurement.

## Files

| file | what |
|---|---|
| `PREDICTIONS.md` | the registered predictions (commit `798c4f2a`, before any game here) |
| `headroom_existing.py` / `.json` | part 1 (zero games) |
| `play_pstar_row.sh` | part 2's driver: the full 18-cell list at 200 pairs; with arguments, a subset (`PAIRS` sets the pairs) |
| `row.json` | `python -m main.h2h read <ledger> --json`: every cell's edge summary |
| `read_row.py` / `row_read.json` | part 2's read against P1 and part 1 |
| `predictions_sim.py` / `.txt` | part 3 |
| `chunks.log` | the five chunks' driver logs, concatenated |

```bash
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=""
python designs/research_state/measurements/x55_league_headroom_2026-10-20/headroom_existing.py > headroom_existing.json
CODE=$PWD scripts/ops/mem_cap.sh --name x55_row 8 timeout 2700 nice -n 19 taskset -c 5-7,13-15 \
    designs/research_state/measurements/x55_league_headroom_2026-10-20/play_pstar_row.sh   # resumable; re-run per chunk
python -m main.h2h read ~/.cache/gen3ai/x55_league_headroom/ledger --json > row.json
python designs/research_state/measurements/x55_league_headroom_2026-10-20/read_row.py row.json headroom_existing.json
python designs/research_state/measurements/x55_league_headroom_2026-10-20/predictions_sim.py
```

## FINDINGS (standing rule 7)

- **F-X55-1. The row was played SMALLER than registered.** The design and P1 named 18 cells at 200 pairs. At the box's
  0.34–1.0 games/s and a 15-deep heavy-slot queue (≈ 2 h per slot), the driver was cut down mid-read: 7 cells at 200
  pairs (30, 60, 100, 190, 230, 260, 270M) and 11 at 100 pairs (80, 130, 140, 160, 210M and the six Nash snapshots):
  5,000 games against the ≈ 7,200 registered. Every cell's SE is its own and P1's pass mark is judged per cell, so the
  read is valid but less precise than registered on those 11 cells (SE ≈ 3.1–3.5 pp instead of ≈ 2.2–2.5).
- **F-X55-2. P1 FAILED its pass mark.** The headroom model's per-node POINTS were too confident: the per-edge cyclic
  component (3–5 pp) is as large as the cell's 2 SE, and the old nodes are closer to P\* than both the spine and
  X37b's home72 cell (0.688 against 25M) suggested. The structural claims (flat top, no hole, S(P\*), no mastery)
  held. Treat point predictions of single edges at ± 2 SE as optimistic.
- **F-X55-3. The spine is steep at long range** (point 7). The cycle monitor's Nash and its "newest node in support"
  both fill unplayed long edges from it, so they understate old nodes. Measuring old edges (as here) is the fix where
  it matters.
- **F-X55-4. A3's ADOPT path is gated by a class R1 rarely reaches** (PREDICTIONS P4; recorded as F-LS-26 in the X35
  sheet): "R1 NULL and R2 FALLS" has P ≈ 0.07 even if exploiters work. And the toy calibration is predicted NOT SHOWN
  (P ≈ 0.85–0.92). Both are the orchestrator's to rule on; nothing here changes a rule.
- **F-X55-5. The max-entropy Nash was not solved on the 66-node archive** (SLSQP did not converge from the LP point;
  the LP point is reported, as the cycle monitor reports). On the 11-node window it converged and equals the LP point.
- **F-X55-6. The proxy and the measurement differ by node.** The 270–290M trainee's sentinel rates are not P\*'s:
  194M read 0.49 as a proxy and 0.59 for P\*; 244M 0.55 and 0.45. Single-edge rates near the plateau are dominated by
  which node meets which, so a PFSP weight computed from one cycle's 100 games (A2, LG-2) is mostly noise.
- **F-X55-7. Process.** (a) The heavy-slot queue held this unit for ≈ 2 h before its first game and ≈ 4.5 h of wall in
  all. (b) I edited `play_pstar_row.sh` while chunk 1 was executing it (bash reads a script as it runs); the edit was
  reverted to the original bytes within minutes and the parametrised version ran from a copy, and chunk 1 was ended by
  its own `timeout`, so nothing ran from the edited file. (c) Interim batch lines were visible in the driver logs while
  later chunks were queued; the predictions were already pushed, so nothing was adjusted after seeing them.
- **F-X55-8. Not done here:** the registered cycle-monitor rows (500 pairs per edge) for X26's late window; the
  zero-training S pre-read of X26's 290–380M nodes (F-RV-2; another agent's `x26_drift_preread` was running in a heavy
  slot during this unit); any GPU measurement of F-AS-3.
