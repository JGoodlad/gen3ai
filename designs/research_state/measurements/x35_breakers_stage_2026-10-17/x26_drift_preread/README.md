# X26 drift pre-read (X35 AM-2's zero-training variance read)

**Status: INTERIM (2026-10-20 09:10 PDT).** 4 of the 10 nodes 290–380M are scored on the registered R; the run continues to its
registered stop (22:00 PDT 2026-10-20) and this page is updated with the final read.

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

## Result (interim)

| node | S, registered R (7) | 95 % interval | se | S, pre-C5 R (6) | vs P\* (280M) | vs T\* |
|---|---|---|---|---|---|---|
| 290M | **56.73** | 55.64 – 57.81 | 0.55 | 56.45 | 52.90 | 58.40 |
| 310M | **56.18** | 55.09 – 57.27 | 0.56 | 55.60 | 51.10 | 59.65 |
| 330M | **56.43** | 55.34 – 57.51 | 0.55 | 55.75 | 51.15 | 60.50 |
| 350M | not yet scored | | | | | |
| 380M | **58.23** | 57.15 – 59.31 | 0.55 | 57.85 | 54.10 | 60.50 |

Over the 4 scored nodes (290M, 310M, 330M, 380M), in pp:

| statistic | registered R (7 members) | pre-C5 R (6) |
|---|---|---|
| mean S | 56.89 | 56.41 |
| **SD of S across nodes** | **0.92** | 1.03 |
| mean measurement se of one S | 0.55 | 0.60 |
| SD beyond measurement noise, sqrt(SD² − se²) | 0.73 | 0.84 |
| "measurement noise alone": χ² (df), p | 8.31 (3), 0.040 | 8.89 (3), 0.031 |
| linear trend, pp per 100M (± se) | +1.96 ± 0.96 | +1.94 ± 1.29 |
| residual SD about the trend | 0.64 | 0.86 |

Every cell is OK at 500 completed pairs; `population_score` reports no INCONCLUSIVE reason on a scored node; the ledger
audits OK (140 rows, 4 nodes × 7 cells × 5 batches). Full per-member scores and standard errors: [`results.json`](results.json).

## What it means for AM-2 and the power

- **The spread of plain continuation is at the LOW end of the review's assumed 0.8–1.6 pp.** S's SD across the 4
  nodes is 0.92 pp on the registered R (1.03 on the pre-C5 R), against a measurement se of 0.55 for one S (the review's
  §1 assumed 0.62 at 6 members; 7 members bring it to 0.55). About 0.73 pp of it is beyond measurement noise, and
  "measurement noise alone" is rejected only weakly (p = 0.04, 3 df).
- **Most of that excess is a slow CLIMB, not wandering.** 380M sits ≈ 1.8 pp above 290–330M; the linear trend is
  ≈ +2.0 ± 1.0 pp per 100M, and the residual SD about it (0.64) is measurement noise. +2 pp per 100M is ≈ +1 pp per
  50M, which is X26's own plateau-rule FLAT threshold: the plateau is flat to the plateau rule's resolution, not to S's.
  Over a 20M fork the trend is worth ≈ 0.4 pp, shared by every fork (b-ON and C0 alike), so it does not enter σ.
- **For AM-2's 1.2 pp gate:** if a 20M fork's run-to-run SD were like this within-run spread, σ̂ would land near
  0.7–0.9 pp and the gate would PASS (continue the registered order). This is a prior for the gate, not a read of it:
  G1's σ̂ over the six P1 finals is still the registered input.
- **For the power (§1's table):** σ ≈ 0.8–0.9 puts R1's 80 %-power effect at ≈ 4 pp with look 2 (§1: 4 pp at σ 0.8,
  5 pp at σ 1.2), and NULL reachable for a truly inert arm about 40 % of the time at σ 0.8 (§1). The σ 1.6 and
  2.0–2.5 pp rows look unlikely from this read.
- **Why it is only a prior (UNVERIFIED that it transfers):** (1) the nodes are successive points of ONE run, so they
  share every random stream up to the node: their spread is within-run drift, not the between-seed spread of
  independent forks, which can be larger (seeds diverge from step 0 of the fork) or smaller (the fork is 20M, these
  span 10–90M); (2) X26's pin `9c033489` has no T62 and its pool was never reseeded, unlike every X35 arm; (3) four
  nodes give the SD only 3 df: its 90 % chi-square interval is ≈ 0.6–2.7 pp around the 0.92.

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
   long X35-side read should re-load R at each unit, as this driver now does.
2. **The first launch held a heavy slot for the whole run** and, with stage 1's 28 h job in the other slot, froze
   every other agent's tests and ships for ≈ 2 h. Corrected to 40-min units (orchestrator ruling, "Changes" 1).
3. **The slot queue doubles the wall time**: units waited 0–61 min for a slot (the first launch 1 h 55 min) (16 jobs queued at 08:00), so a node
   costs ≈ 3–4 h of wall clock for ≈ 1.8 h of play. The final read (22:00) will hold about 6–7 nodes, not 10.
4. **The batch size is 100 pairs, not the h2h default 500**: the schedule seeds each batch by its index
   (`play.cycle_seed`), so the team pairs are a different random sample of the same distribution. S's definition
   (500 completed mirrored pairs per cell, greedy vs greedy, the eval builder) is unchanged; its expectation is too.
5. **The code that played is not an X35 pin**: main `d825c6a1` (+ this driver; + the C5 R file from 07:06, scratch
   commit `67636929d` in the pinned tree). The h2h regime id `adfdbee824c9eb0a` is the R freeze's own, so the cells
   are comparable with the band cells; they are NOT comparable with a future X35 read at a pin whose regime differs.
6. **T\* was copied from a LIVE run directory** (`models/rb_toy_ent_s4002_deep`, X34), read-only, its sha256 checked
   against the frozen R (`cfa7f7be…`).
7. **The pinned job tree `~/.cache/gen3ai/x26_drift_preread/code` is a registered git worktree** of the main repo
   (detached). It must be removed (`git worktree remove`) once the run ends at 22:00; until then it carries an
   unshipped scratch commit.
8. **Time-boxed engines lose their in-flight batch**: each lost batch's claim was voided by the next claim (the
   ledger audits OK, 2 live claims belong to the running unit). No row was double-counted (`population_score` would
   name a cell listed twice).
