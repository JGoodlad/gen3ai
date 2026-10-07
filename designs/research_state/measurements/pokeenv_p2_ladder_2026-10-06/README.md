# Poke-env retirement P2: the ladder on the Rust eval engine, and the paired shift (2026-10-06)

**Question.** The snapshot ladder (the ELO headline) moved from the PYTHON stack (two poke-env `RLPlayer`s over the
in-process bridge, the Python encoder) to the RUST eval engine (`agents/training/snapshot_ladder_play.py`, Rust rows,
seat-balanced mirrored pairs). How far do the same frozen edges move?

**Design.** `paired_shift.py` (durable, resumable: one fsynced row per (pair, stack) in `rows.jsonl`). Run
`ai_v14_01_base` (N0: 20 pool snapshots, config v121, 380 banked Python edges). 16 pairs, `a` = the NEWER node,
gaps 1 / 3 / 6 / 10 pool steps from starts 0 / 4 / 8 / 9. 100 games per pair per stack, CPU, both greedy:

- **Python**: the pre-P2 `_play_pair` verbatim (compile off, `impl rust`, concurrency 4), `a` on p1, a draw counts
  against `a`;
- **Rust**: `LadderEngine.play` (50 mirrored pairs, 25 each seat, the same team pairings both ways), `front ffi`,
  `profile selfcheck` (both stacks run the worktree's emission self-check Rust builds: release semantics plus checks).

Three contrasts per pair: (1) **Rust with `a` on p1, draws against `a`** − Python: the transport alone, same seat and
fold (mirroring changes the variance, not the expectation); (2) **the Rust edge as fitted** (seat-balanced, draws
out) − Python: everything a reader of a new ladder sees; (3) **Python now − Python banked** (the run's own
`games.jsonl`, played 2026-09-27/28): the SAME stack twice, the matched-noise floor. Intervals: per pair a binomial
variance (conservative for the mirrored side), the mean's 95 % interval three ways (the binomial model, an
empirical t over pairs, a 4,000-draw bootstrap over pairs, seed 20261006).

**Contamination.** The box was loaded (load1 ~8–11; a live X5 seed and several agents): every wall time in
`rows.jsonl` is contaminated and is NOT a cost measurement. Counts are unaffected.

## Result

All 16 pairs on both stacks (`rows.jsonl`; `summary.json` = `paired_shift.py read --json`). `a` win rate per pair:

| a | b | Python | Rust, a on p1 | Rust edge (fitted) | Python banked | Rust draws |
|---|---|---|---|---|---|---|
| 26.0M | 24.0M | 0.53 | 0.62 | 0.620 | — | 0 |
| 30.0M | 24.0M | 0.49 | 0.56 | 0.541 | 0.57 | 2 |
| 34.0M | 32.0M | 0.46 | 0.38 | 0.390 | — | 0 |
| 36.0M | 24.0M | 0.51 | 0.52 | 0.510 | 0.54 | 0 |
| 38.0M | 32.0M | 0.44 | 0.50 | 0.490 | 0.51 | 0 |
| 42.0M | 40.0M | 0.60 | 0.44 | 0.430 | — | 0 |
| 44.0M | 24.0M | 0.47 | 0.56 | 0.550 | 0.63 | 0 |
| 44.0M | 32.0M | 0.45 | 0.56 | 0.520 | — | 0 |
| 44.0M | 42.0M | 0.47 | 0.50 | 0.500 | — | 0 |
| 46.0M | 40.0M | 0.49 | 0.44 | 0.455 | 0.53 | 1 |
| 48.0M | 42.0M | 0.56 | 0.48 | 0.520 | 0.46 | 2 |
| 54.0M | 32.0M | 0.48 | 0.46 | 0.450 | 0.59 | 0 |
| 54.0M | 40.0M | 0.49 | 0.62 | 0.600 | — | 0 |
| 58.0M | 42.0M | 0.48 | 0.60 | 0.602 | — | 2 |
| 70.0M | 40.0M | 0.55 | 0.56 | 0.560 | 0.60 | 0 |
| 72.0M | 42.0M | 0.62 | 0.50 | 0.510 | 0.54 | 0 |

("—": the banked row for that pair is a v2 `eval_cycle` copy, which is not an edge.)

| contrast | k | mean Δ | 95 % CI (binomial model) | 95 % CI (bootstrap over pairs) | mean \|Δ\| |
|---|---|---|---|---|---|
| (1) Rust, a on p1, draws vs a − Python (the TRANSPORT alone) | 16 | **+1.31 pp** | [−2.90, +5.52] | [−3.12, +5.44] | 7.69 pp |
| (2) Rust edge as fitted − Python (all a reader sees) | 16 | **+0.99 pp** | [−2.46, +4.43] | [−3.12, +4.78] | 6.67 pp |
| (3) Python now − Python banked (the same stack, the floor) | 9 | −4.00 pp | [−8.58, +0.58] | [−8.89, +1.34] | 8.00 pp |

**Read.** NOT DETECTED: neither cross-stack mean excludes zero, and the per-pair dispersion across stacks (mean |Δ|
6.7–7.7 pp) is no larger than the SAME stack's against its own banked games (8.0 pp), so this sample shows no
pair-specific transport effect either. The mean edge may still move by up to about −3 / +5 pp (≈ ±25–35 Elo on one
edge near 50 %); with no pre-registered equivalence bar, this is "not detected", NOT "equivalent". **The seat
effect** (Rust, `a`'s rate on p1 minus on p2, /2): **u = +0.56 pp** (difference +1.13 pp [−0.51, +2.76], k = 16),
not detected; the Python ladder credited any such u to the newest node on every edge. Draws: 7 of 1,600 Rust games
(turn-limit draws), 3 of 1,600 Python ties.

**Decision it informs.** The transport is a REGIME BOUNDARY regardless (the protocol changed: seats, mirroring,
draws), so a fit or comparison across it is refused unless consented (`designs/training/eval_and_rating.md` "The
TRANSPORT boundary"). The measured shift says a consented mix would not be grossly wrong at this precision; it does
not license pooling.

**Untaught meter** (the sub-unit's paired read, `untaught/`): `rb_x5ab_blob_s1001` vs the v14 opponent, the untaught
8, 60 games/team, team-level pairing only: Δ new − old −4.38 pp [−8.75, +0.42], NOT DETECTED.
