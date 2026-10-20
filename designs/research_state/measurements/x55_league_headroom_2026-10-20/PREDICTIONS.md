# X55 / LG-0: PRE-REGISTERED PREDICTIONS for X35's population arms (2026-10-20)

**Registered BEFORE any X35 production arm exists** (production waits for FP32_ALT; only the toy block T1 is on the
GPU) **and before any game of this unit's own P\* row (part 2) was read.** The row's predictions (P1) were first written
at 08:16 PDT, before `play_pstar_row.sh` was queued (the file's sha256 then was `c4cf46b3…6487307`, kept in
`~/.cache/gen3ai/x55_league_headroom/predictions_p1_draft.sha256`; P1's numbers are unchanged here). This commit holds
ONLY this file; the read (README, scripts, the measured row) lands in the next commit. Nothing below may be edited
after an X35 read: a correction is a new dated section at the end.

**Design:** [`design_alphastar_league_path.md`](../../../endstate/design_alphastar_league_path.md) §6 row LG-0 (d).
**The rule the predictions are about:** R1 (`design_compute_efficiency.md` §5.6.4; `main.x35.readout.decide_r1`): Welch
over seed finals of the population score S, 97.5 % per look, ±2.0 pp bar; look 1 = 3 v 3 seeds, an UNDECIDED with
Δ̂ > 0 is CONTINUE, look 2 = 5 v 5.

## The basis: what existing games say about the league's headroom (zero new games)

From `headroom_existing.py` (next commit) on X26's in-run snapshot ladder (66 nodes ≤ 268M, 1,036 edges, 65,250 pairs)
and its in-run eval rows (the trainee vs 5 pool sentinels every 2M, 100 games each):

1. **The archive's Nash sits in C0's own pool.** The posterior Nash over the archive to 268M puts **0.90** of its mass
   on the 14 snapshots C0 is seeded with (164–268M) and **0.98** on nodes ≥ 130M; the point support is 6 nodes
   (194, 224, 244, 246, 254, 268M; posterior 7.5, 90 % [5, 10]).
2. **A1's archive barely touches it.** Ladder nodes within ±5M of A1's 8 nodes hold **0.19** of the mass (the late four,
   160–260M: 0.17; the early four, 30–130M: 0.014). A1's real nodes are 30, 60, 100, 130, 160, 190, 230, 260M
   (`reference_population_production.json`), not the "20–230M" of the league doc's F-AS-2.
3. **The top is FLAT, and P\* has no hole in it.** The trainee at 270–290M (P\* ± 10M) scores 0.49–0.55 against every
   pool sentinel from 134M to 268M (900 games each at 134, 160, 194, 236, 268M), and the trainee at 240–300M scores
   0.50–0.61 against every node from 100M up that has ≥ 300 games (two 100-game cells read 0.41 at 252M and 0.62 at
   224M). The worst is 194M at 0.491. **Nothing in the archive beats P\* by more
   than ≈ 1 pp**, so an archive lever has no measured weakness to repair; its value, if any, is generalisation.
4. **PFSP has almost nothing to tilt.** A2's factor recency × (1 + 2.5 (1 − p)) over C0's pool, at those rates, moves
   each snapshot's share by 0.97–1.05× (total variation **1.2 %** of the pool's games). Stable PFSP over A1's nodes
   (`max(0.05, 1 − p)`, provisional rates) gives 0.88–1.13× uniform. No A1 node is near the 0.80 mastery floor.

## P0. G1, the variance gate (AM-2)

- **Per-final SD of S: σ ≈ 1.0 pp** (prior median; 10–90 % 0.70–1.43). σ̂ over the six A1 / C0 finals: median 0.9,
  10–90 % [0.46, 1.59]. **P(G1 PASS) ≈ 0.74.**

## P1. P\*'s measured row (part 2, 200 pairs per cell) — written 08:16 PDT, before the games were queued

| node | predicted score of P\* | node | predicted |
|---|---|---|---|
| 30M (A1) | 0.68 | 230M (A1) | 0.52 |
| 60M (A1) | 0.63 | 260M (A1) | 0.52 |
| 100M (A1) | 0.60 | 270M (R) | 0.51 |
| 80M (R) | 0.61 | snapshot 268M | 0.51 |
| 130M (A1) | 0.55 | snapshot 244M | 0.54 |
| 140M (R) | 0.55 | snapshot 194M | 0.49 |
| 160M (A1) | 0.55 | snapshot 254M | 0.51 |
| 190M (A1) | 0.51 | snapshot 224M | 0.53 |
| 210M (R) | 0.52 | snapshot 246M | 0.51 |

Derived, also registered: **no node reaches 0.80** (none mastered at A1's start) and **no node falls below 0.45**;
**S(P\*) ≈ 0.567** (the mean over R's 7 members of {0.5 self, 0.51, 0.61, 0.55, 0.52, 0.6935, 0.5905}); stable PFSP's
weight ratio over A1's 8 nodes, max / min ≈ 1.5. **Pass mark for P1:** at least 15 of the 18 cells inside the
prediction ± 2 SE of the cell (pair SD ≈ 0.31, so ± ≈ 4.4 pp).

## P2. A1, the archive pool (P1 stage; look 2 = P3)

- **Δ_pop (A1 − C0) = +0.5 pp, 80 % [−1.0, +2.0].** Why small: (1)–(3) above say the archive adds no cluster coverage
  C0 lacks and holds no node that beats P\*. Why not zero: R's A-block (80, 140, 210M) sits next to A1's nodes (F20),
  and production (an X26-25M-strength policy by P\*'s score against it, 0.69 vs X37b's 0.688) may be easier to beat
  after training against early nodes; half the self-play games leave the cluster, which costs P\* / 270M a little.
- **Predicted class** (`predictions_sim.py`, Δ ~ N(0.5, 1.2²), σ from P0's prior, the registered two-look rule,
  cross-checked against `decide_r1`: 0 / 2,000 mismatches):

  | look 1 | GAIN 0.005 | NULL 0.055 | LOSS 0.001 | **CONTINUE 0.60** | UNDECIDED (Δ̂ ≤ 0) 0.34 |
  |---|---|---|---|---|---|
  | **after look 2** | GAIN 0.014 | NULL 0.14 | LOSS 0.001 | **UNDECIDED 0.84** (of which Δ̂ > 0: 0.48) | |

  **Modal: look 1 CONTINUE, then FINAL UNDECIDED.** The §7 branch it lands on is a coin toss: "UNDECIDED, Δ̂ > 0"
  0.48 against "Δ̂ ≤ 0 / NULL" 0.50.
- **Descriptive, also registered:** in A1's own `ext_x26_*` eval rows, no archive node except possibly 30M reaches the
  0.80 mastery floor by P\* + 20M (P(30M mastered) ≈ 0.2; every other node ≈ 0).

## P3. A2, PFSP over the recent pool (P4 stage)

- **Δ_pop (A2 − C0) = 0.0 pp, 80 % [−0.8, +0.8]**: the lever moves ≈ 1 % of the pool's games (basis 4); the 10
  sentinels' 100-game rates (SE ≈ 5 pp) make what tilt there is mostly noise.
- **Predicted class at look 1** (its only scheduled look): GAIN 0.000, NULL 0.08, LOSS 0.000, CONTINUE 0.46,
  UNDECIDED (Δ̂ ≤ 0) 0.46. With a look 2 it would end UNDECIDED 0.80, NULL 0.20.

## P4. A3, main exploiters folded in (LG-1; not in batch 1, registered here as the design asks)

- **R0, gap(P\*)** (a fresh 5M offense best responder of P\*): **+14 pp, 80 % [+6, +22]** (the old lineage read +14.0,
  +15.75 and +20.0 on its plateau parents); **P(gap(P\*) ≥ 0.15, A6's trigger) ≈ 0.45.**
- **R2, gap(ON) − gap(OFF) = −6 pp, 80 % [−13, +1]**; P("the gap FALLS": Newcombe upper bound < 0 at n ≈ 400 per
  cell) ≈ 0.45.
- **R1, Δ_pop = −0.3 pp, 80 % [−1.6, +1.0]**: exploiters take 40 % of the challenge games and are opponents P\*'s
  population never meets. Look 1: CONTINUE 0.38, UNDECIDED (Δ̂ ≤ 0) 0.55, NULL 0.06, GAIN 0.000. After look 2: NULL
  0.15, UNDECIDED 0.85.
- **ADOPT-CANDIDATE needs R1 NULL and R2 FALLS: P ≈ 0.07** even with look 2. The exploiter arm's co-primary is gated
  by a class R1 rarely reaches (FINDING F-X55-4).

## P5. A6, league exploiters (conditional)

- Triggered with P ≈ 0.45 (P4's R0). If run: **Δ_pop −0.3 pp, 80 % [−1.8, +1.2]; R2 −4 pp, 80 % [−11, +3]**. Class
  as A3's (look 1 CONTINUE 0.39, UNDECIDED 0.55, NULL 0.06).

## P6. The toy transfer calibration (T2, AM-3)

- T2 launches toy A1 only on a FINAL production A1 class GAIN / NULL / LOSS (P ≈ 0.16 with look 2) and toy A2 only on a
  FINAL A2 class (P ≈ 0.08 at its one scheduled look). **Predicted: the calibration reads NOT SHOWN for both arms
  (P ≈ 0.85 for A1, ≈ 0.92 for A2), and records no agreement (P(agreement recorded) < 0.05).**

## How these are scored

Each X35 read is set against its row here, in the read's own record: the class against the predicted class
probabilities (a log score, reported), Δ̂ against the 80 % interval, and, for P1, the pass mark above. A miss is a
FINDING about the headroom model (the flat top, the Nash in the pool), never a reason to re-read an arm.
