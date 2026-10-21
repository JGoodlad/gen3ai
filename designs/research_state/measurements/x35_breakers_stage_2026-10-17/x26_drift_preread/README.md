# X26 drift pre-read (X35 AM-2's zero-training variance read)

**Status: FINAL (2026-10-20 21:05 PDT).** All 10 nodes 290–380M are scored on the registered R. The interim read
(4 nodes, 09:10) shipped at `578bcb645`.

## What it is

`X35_BATCH1_REVIEW.md` F-RV-2 proposed reading, before any GPU is spent, how far X35's population score S wanders when
a plateaued run simply keeps training. X26 (`models/rb_x26_s2601`) plateaued at P\* = 280M and then trained on, plainly,
to 380M, which is what X35's control C0 does. Each of its 10M nodes 290…380M is scored here on the X35 readout's OWN
score:

- **S(node)** = `main.x35.readout.population_score`: the equal-weight mean, over the frozen reference population R's
  members, of the node's mirrored-pair score against that member; each cell 500 completed pairs (1,000 games).
- **Regime**: `main.h2h play-many`, greedy vs greedy, mirrored pairs, the eval team builder (no panel), reveal off, CPU
  eager, `torch_threads` 1, regime id `adfdbee824c9eb0a` (the same as the R freeze's band cells).
- **R**: the CURRENT registered R, `reference_population_production.json` (content `cef0960e…`, 7 members: X26 at
  280 / 270 / 80 / 140 / 210M, the registry baseline `production`, and T\* = X34 @ 290M). The pre-C5 R (6 members, no T\*)
  is reported beside it for continuity (see "Changes" below).
- **Per-node interval** (declared here; the readout defines none for one final): the cells are independent, so
  se(S) = sqrt(Σ se_m²) / n_members, se_m the pair-level standard error of each cell (`mirrored_pairs.pair_score_ci`'s
  variance); the interval is S ± 1.96 se.
- **Spread**: the sample SD of S over the scored nodes; the part beyond measurement noise, sqrt(SD² − mean se²); a
  χ² test of "measurement noise alone" (Σ (S − mean)² / se², df = n − 1); a linear trend and its residual SD.

It is DESCRIPTIVE only and never enters R1: one seed, X26's pin `9c033489` (no T62), its pool not reseeded, and every
node is a step of the SAME run, so successive nodes are not independent forks.

## Result

| node | S, registered R (7) | 95 % interval | se | S, pre-C5 R (6) | vs P\* (280M) | vs T\* |
|---|---|---|---|---|---|---|
| 290M | **56.73** | 55.64 – 57.81 | 0.55 | 56.45 | 52.90 | 58.40 |
| 300M | **56.97** | 55.88 – 58.06 | 0.55 | 56.28 | 51.15 | 61.10 |
| 310M | **56.18** | 55.09 – 57.27 | 0.56 | 55.60 | 51.10 | 59.65 |
| 320M | **56.51** | 55.44 – 57.58 | 0.55 | 55.86 | 49.05 | 60.40 |
| 330M | **56.43** | 55.34 – 57.51 | 0.55 | 55.75 | 51.15 | 60.50 |
| 340M | **56.79** | 55.70 – 57.88 | 0.56 | 56.20 | 49.40 | 60.30 |
| 350M | **57.90** | 56.84 – 58.96 | 0.54 | 57.79 | 51.85 | 58.55 |
| 360M | **57.81** | 56.73 – 58.88 | 0.55 | 57.34 | 53.95 | 60.60 |
| 370M | **58.06** | 56.99 – 59.13 | 0.55 | 57.51 | 54.35 | 61.35 |
| 380M | **58.23** | 57.15 – 59.31 | 0.55 | 57.85 | 54.10 | 60.50 |

Over the 10 nodes, in pp:

| statistic | registered R (7 members) | pre-C5 R (6) |
|---|---|---|
| mean S | 57.16 | 56.66 |
| **SD of S across nodes** | **0.76** | 0.87 |
| mean measurement se of one S | 0.55 | 0.59 |
| SD beyond measurement noise, sqrt(SD² − se²) | 0.52 | 0.64 |
| "measurement noise alone": χ² (df), p | 17.18 (9), 0.046 | 19.53 (9), 0.021 |
| linear trend, pp per 100M (± se) | **+2.05 ± 0.51** | +2.19 ± 0.66 |
| residual SD about the trend | 0.47 | 0.60 |

The shape is closer to a STEP than a ramp: 290–340M sit at 56.2–57.0, 350–380M at 57.8–58.2 (≈ +1.4 pp between 340M
and 350M). With 10 points, a ramp and a step are not separable; both say the same about σ (below).

Per member of R (the trend of the node's score against each member; a cell's se is ≈ 1.5 pp, so a slope's se is
≈ 1.6 pp per 100M):

| member of R | slope, pp per 100M | ± se |
|---|---|---|
| `production` | +1.84 | 1.48 |
| `tstar` | +1.19 | 1.62 |
| `x26_080m` | +4.86 | 1.61 |
| `x26_140m` | +1.89 | 1.63 |
| `x26_210m` | +5.11 | 1.63 |
| `x26_270m` | -3.83 | 1.63 |
| `x26_pstar` | +3.28 | 1.62 |

The climb is broad (6 of 7 members positive), largest against X26's own 80M and 210M, and NEGATIVE against its 270M
(≈ −2.4 se): the plain continuation gains on its older selves while slightly losing to its immediate predecessor, a
non-transitive signature consistent with the battery's MATERIAL cyclic component (B6). One seed; no per-member
slope is a finding on its own.

Every cell is OK at 500 completed pairs; `population_score` reports no INCONCLUSIVE reason on any node. Full
per-member scores and standard errors: [`results.json`](results.json).

## What it means for AM-2 and the power

- **The spread of plain continuation is at the LOW end of the review's assumed 0.8–1.6 pp.** S's SD across the 10
  nodes is 0.76 pp on the registered R (0.87 on the pre-C5 R), against a measurement se of 0.55 for one S (§1 of the
  review assumed 0.62 at 6 members; 7 members bring it to 0.55). About 0.52 pp is beyond measurement noise
  ("measurement noise alone" is rejected weakly: p = 0.046 on 9 df).
- **All of that excess is the CLIMB, none is wandering.** The trend is +2.05 ± 0.51 pp per 100M, and the residual SD
  about it (0.47) is at or below measurement noise (0.55): once the slow climb is removed, consecutive nodes differ by
  nothing the meter can see.
- **For AM-2's 1.2 pp gate:** if a 20M fork's run-to-run spread is like this within-run spread, σ̂ lands near
  0.55–0.8 pp and the gate PASSES (continue the registered order). This is a prior for the gate, not a read of it:
  G1's σ̂ over the six P1 finals stays the registered input.
- **For the power (§1's table):** σ ≈ 0.6–0.8 pp puts R1's 80 %-power effect at ≈ 4 pp with look 2 (§1: 4 pp at
  σ 0.8), and a truly inert arm reaches NULL ≈ 40 % of the time at σ 0.8 (§1's 0.416), more below it. The σ 1.6 and
  2.0–2.5 pp rows look unlikely from this read. NULL needs σ < 0.70 at 3 v 3 (§1): this read puts σ near that line,
  so look 1 may decide more often than §1's σ 1.2 rows say.
- **Why it is only a prior (UNVERIFIED that it transfers):** (1) the nodes are successive points of ONE run, sharing
  every random stream up to the node: within-run drift, not the between-seed spread of independent forks, which can
  be larger (seeds diverge from the fork's first step) or smaller (a fork is 20M; these span 10–90M); (2) X26's pin
  `9c033489` has no T62 and its pool was never reseeded, unlike every X35 arm; (3) 10 nodes give the SD 9 df: its
  90 % chi-square interval is ≈ 0.55–1.25 pp around the 0.76.

## The trend against the collapse battery's TREADMILL (`3e8a4124`)

The battery found steady ≈ 30 % greedy churn per 20M whose distance from 380M barely grows with lag (180M vs 380M
differ on 36 %), and called the plateau a treadmill: moving, getting nowhere. S says the policy IS getting somewhere,
slowly: +2 pp per 100M against a FIXED population. The two are consistent, because they measure different things:

- **Churn is a distance in ACTION space, equal-weighted over a fixed bank of positions** (7,028 decisions drawn from
  X26's 30 / 90 / 150M eval traces, mean entropy ≈ 1.06 nats). A flip between two near-equal moves counts as much as a
  game-losing blunder, and with that much entropy most flips are between near-equal moves: churn is dominated by
  value-neutral wander. It also reads positions from early play, not the states the plateau policy reaches.
- **S is an OUTCOME, weighted by consequence, on the states the policy actually reaches**, against opponents that do
  not move. Only the decisions that change who wins move it. A small, consistent shift in the few decisions that
  matter is invisible inside a 30 % wander of decisions that do not, and is exactly what a fixed-opponent outcome meter
  sees.
- **The battery's pool-relative edge (0.52–0.54, flat) is relative to a moving target**: the pool advances too, so
  flat is what slow progress looks like there. S's panel is frozen, so the same progress shows as a climb.
- So the right reading is "churn without much DISTANCE, plus a slow drift in the decisions that matter": a treadmill
  for the policy's bulk, not a dead stop for its strength. +2 pp per 100M is about +14 Elo per 100M (at ≈ 7 Elo per pp
  near 50–60 %), small next to the 9 → 3 fall in promotions per 50M the battery also measured. The per-member slopes
  add a caveat the battery shares: part of the climb is against X26's OLD selves (80M, 210M) and none of it against
  270M, so some of it may be cyclic advantage rather than absolute strength.

## What the trend implies

- **For X35's 20M forks:** the base climb over 20M is ≈ +0.4 pp (2.05 × 0.2). Every fork, b-ON and C0 alike, starts
  at P\* and runs 20M, so the climb is SHARED and the matched C0 absorbs it (R1's Δ is ON − OFF at the same step); it
  enters neither Δ̂ nor σ, provided every final is read at its own 20M node (the kit's fixed-step stop). A breaker
  still needs Δ ≥ +2 pp to clear §5.1's bar in 20M, ≈ 5× the base climb in that span.
- **For calling 280M a "plateau":** the measured climb is +2.05 ± 0.51 pp per 100M, ≈ +1.0 pp per 50M, which sits ON
  X26's plateau-rule FLAT threshold (≤ ≈ +7 Elo ≈ +1 pp per 50M; review §1). Its 95 % interval (t, 8 df: ≈ +0.9 to +3.2 pp
  per 100M) excludes zero. So 280M is a plateau to the plateau rule's resolution, not a stop: strength still rises
  at ≈ +1 pp per 50M (one seed, serially correlated nodes, so the interval is optimistic). "Plateau" in X35's
  documents should be read as "slow-climb regime below the plateau rule's threshold", not "no further gain".

## How it was run

- [`preread.py`](preread.py) `run` / `unit` / `read`. Every checkpoint is COPIED to `~/.cache/gen3ai/x26_drift_preread/ckpts/`
  with its sha256 checked against the source (and, for R's members, against the frozen R) and with its run's
  `model_config.json` beside it; `models/` is only read. The eval ledger is `~/.cache/gen3ai/x26_drift_preread/ledger`
  (family `x26_drift_preread_2026-10-19`, one request per node, `x26pre_<M>m_b100`), never `models/_ledger`.
- CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice -n 19 taskset -c 5-7,13-15`, under `mem_cap.sh` (12 GB cap; peak ≈ 2 GB).
- **One heavy slot per bounded UNIT, never per run** (orchestrator 2026-10-20): the driver holds no slot; each unit
  takes one through `mem_cap.sh --slot`, runs two engines (one cell each at a time, `--threads 2`) for at most 40 min,
  and releases it. Cells are recorded in 100-pair batches, so a unit's time box loses at most its in-flight batch.
- Throughput ≈ 0.55 games/s per engine, ≈ 1.1 games/s per unit: one node (7,000 games) ≈ 1.8 h of play; the slot
  queue roughly doubled the wall time.
- The code that played the games is the pinned tree `~/.cache/gen3ai/x26_drift_preread/code` (main `d825c6a1` + this
  driver; from 07:06 the driver and the C5 R file overlaid on the same h2h code, scratch commit `67636929d`). The h2h
  regime id is unchanged across the overlay.
- Re-read at any time: `PYTHONPATH=<code>/src python preread.py read --cache ~/.cache/gen3ai/x26_drift_preread --out results.json`.

## Changes after the start (each with its reason)

1. **2026-10-20 00:05** — the first launch held one heavy slot for the whole run; the orchestrator ruled a slot is held
   per bounded chunk (≤ 45 min) so other agents' jobs interleave. Stopped by PID (nothing recorded was lost) and
   relaunched with 40-min units; the per-node requests were renamed `…_b100` because the batch size (500 → 100,
   the resume grain) is part of a request's spec. The two killed requests (`x26pre_290m`, `x26pre_380m`) hold no row.
2. **2026-10-20 ~01:45** — the registered stop moved from 09:50 to **22:00 PDT** (orchestrator: X35's production arms
   are delayed about a day by the T62 verdict FP32_ALT, so AM-2's gate is more than a day away and more nodes are
   wanted). Recorded in the cache's `state.json` `stop_history`.
3. **2026-10-20 07:06** — R was RE-FROZEN under ruling C5 at 2026-10-19 23:51 PDT, one minute after this read first
   loaded R; the new R adds T\* as a 7th member. The driver now scores every node against the current R (the T\* cells
   of the nodes already played were added) and reports the pre-C5 6-member S beside it.

## Hazards and limits

Each is a FINDING, reported rather than footnoted:

1. **R moved under the read (C5).** R was re-frozen with T\* one minute after this read first loaded it. Handled (the
   T\* cells were added; both S are reported), but a read that began a minute later would have needed no change: any
   long X35-side read should re-load R at each unit (this driver loads it only at each start, and was restarted to pick up C5).
2. **The first launch held a heavy slot for the whole run** and, with stage 1's 28 h job in the other slot, held one of the box's two
   slots for 13 min, enough to stall every other agent's tests and ships, before the orchestrator stopped it. Corrected to 40-min units (orchestrator ruling, "Changes" 1).
3. **The slot queue doubles the wall time**: units waited 0–61 min for a slot (the first launch 1 h 55 min; 16 jobs queued at 08:00), so a node
   costs ≈ 3–4 h of wall clock for ≈ 1.8 h of play. All 10 nodes finished at 21:00, one hour before the stop.
4. **The batch size is 100 pairs, not the h2h default 500**: the schedule seeds each batch by its index
   (`play.cycle_seed`), so the team pairs are a different random sample of the same distribution. S's definition
   (500 completed mirrored pairs per cell, greedy vs greedy, the eval builder) is unchanged; its expectation is too.
5. **The code that played is not an X35 pin**: main `d825c6a1` (+ this driver; + the C5 R file from 07:06, scratch
   commit `67636929d` in the pinned tree). The h2h regime id `adfdbee824c9eb0a` is the R freeze's own, so the cells
   are comparable with the band cells; they are NOT comparable with a future X35 read at a pin whose regime differs.
6. **T\* was copied from a LIVE run directory** (`models/rb_toy_ent_s4002_deep`, X34), read-only, its sha256 checked
   against the frozen R (`cfa7f7be…`).
7. **The pinned job tree `~/.cache/gen3ai/x26_drift_preread/code` is a registered git worktree** of the main repo
   (detached). Removed with `git worktree remove` once the run ended; its scratch commit
   `67636929d` (the overlay) was never shipped and is recorded here only for provenance.
8. **Time-boxed engines lose their in-flight batch**: each lost batch's claim was voided by the next claim on that
   unit. The final ledger audits OK: 350 rows (10 nodes × 7 cells × 5 batches), 12 requests (10 `…_b100` + the 2
   empty first-launch ones), and 2 claims of dead writers that no later claim needed (inert). No row was
   double-counted (`population_score` would name a cell listed twice).
9. **The unit packer left one engine idle** when a node had a single cell left (it does not pull the next node's
   cells into the same unit), costing a few short units; throughput only, not the read.
