# The cycle monitor's first retro reads (X26, X34) — 2026-10-19

The cycle monitor (`python -m main.cycle_monitor`, `design_evaluation.md` §2.5 "As built") and the plateau's Tier 2
(§8.2 "As built") were built on 2026-10-19, after X26 had ended at 380M and while X34 (`rb_toy_ent_s4002_deep`) was
live. These reads are DESCRIPTIVE: no decision about either run changes because of them.

## X26 (`rb_x26_s2601`): the in-run snapshot ladder, read by the monitor's signals

X26 is pinned and carries none of this code, and a registered monitor row costs ≈ 1.5–2.5 h of rule 5b's CPU per
check (≈ 2 CPU-days to 380M, F-ED-25). So X26 is read through `main.cycle_monitor ladder`: the four signals on its
in-run snapshot ladder (`snapshot_ladder/games.jsonl`: 1,150 mirrored lines, 72 snapshots 2M–376M, one regime
`adfdbee824c9eb0a`, 50 pairs per pool line and 100 per promotion-baseline line), as of each 10M check. This is NOT the
registered instrument: the edges are the ones the pool chose (each node against its ~20-deep sliding window), at a
tenth to a fifth of the registered 500 pairs.

```bash
python -m main.cycle_monitor ladder rb_x26_s2601 --json                          > x26_ladder_accumulated.json
python -m main.cycle_monitor ladder rb_x26_s2601 --window-steps 80000000 --json  > x26_ladder_window80M.json
```

- **Accumulated matrix (the registered (a)'s scope): SUSPECTED at EVERY one of the 37 checks, 10M–370M.** The
  Hodge χ² sits at the bootstrap floor (p = 0.0005 of 2,000) throughout; the excess cyclic RMS (the w-weighted
  residual RMS above the null's mean, pp of score at ½) is 3.9–4.8 pp, 3.9–4.1 pp from 170M on. Triangles through the
  newest node are significant after Holm at 16 checks (last at 290M); the lag curve flags at 30M, 100M, 140M, 160M,
  360M and 370M. The Nash support of the point matrix is 4–9 nodes from 140M on, never above its null.
- **Last 80M only (`--window-steps 80000000`, a CURRENT-cycling view; not the registered scope):** SUSPECTED at 34 of
  37 checks, through 350M (Hodge p 0.0005–0.01 to 330M, excess RMS 2.7–4.2 pp from 170M); NONE at 290M and 360–370M,
  where the window holds 5–10 nodes (low power, not absence). The Nash signal is in its rounding band or flags at
  200–220M, 270–280M and 340–350M.
- **The plateau node (280M; Tier 1 FLAT at 270M and 280M, GAIN at 290M).** The ladder has no node between 268M and
  280.5M, so both checks read the 268M node: SUSPECTED (Hodge p 0.0005 accumulated, 0.0055 in the window; excess RMS
  ≈ 2.9 pp in the window; the Nash signal in its rounding band). **Under the two-tier rule X26 at 280M reads CONTINUE
  (monitor SUSPECTED), not a plateau**: a cyclic component of a few pp sits beside a flat Tier 1. It is never
  CONFIRMED here (a ladder read cannot top up), so it is not a TREADMILL call either.
- **The null is calibrated on this graph** (`calibration` below): 30 synthetic data sets on the SAME 66-node,
  1,036-edge graph and pair counts, under a TRANSITIVE truth with a per-pair team effect (logit shift N(0, 0.5),
  swapped in the mirror): Hodge flagged 1 / 30 (one more in its band), triangles 1 / 30. So the flags above are
  real non-transitivity in the ladder's games, not a miscalibrated null.
- **Consistent with what was known:** `design_collapse_and_counterfactuals.md` §3.4 read a real, small cyclic width
  on the same ladder at 162M (`main.elo`'s Hodge: excess 32 Elo, p = 0.005 at 200 bootstraps, twenty significant
  3-cycles). The monitor reads the same structure, and finds it persists to 370M.

**What the pending candidate "the collapse battery on X26 180–380M" is, relative to this.** The battery
(`design_collapse_and_counterfactuals.md` §4) has ten items over five collapse types; the cycle monitor IS its type-(c)
STRATEGIC item B6 (Hodge width vs its null, plateau Tier 1, sentinel win rates by age), done properly: the four
signals, per check, with calibrated nulls. It is NOT B1–B5 or B7–B10 (entropy, starvation, coverage, opponent model,
plasticity, exploitability), which stay a separate read.

## X34 (`rb_toy_ent_s4002_deep`, from `rb_toy_ent_s4002`): the REGISTERED instrument, offline

X34's toy plays ≈ 15–20 games/s on rule 5b's CPU, so its registered monitor rows were played in full: every check
20M–230M, 500 mirrored pairs per edge at the hybrid lags, `--lineage-run rb_toy_ent_s4002`, into a ledger root
OUTSIDE `models/` (`~/gen3ai_archive/cycle_monitor_retro_2026-10-19/smoke_ledger`; the unit was barred from writing
under `models/`), one `cycle_flag` decision per check.

**PENDING** at this commit (the tick is resumable; the result lands in `x34_status.txt` with the next commit).

## Files

- `x26_ladder_accumulated.json`, `x26_ladder_window80M.json` — every check's verdict, signals, summary, the newest
  node's lag curve and the significant triangles.
- `calibration_x26_graph.txt` — the null-calibration run on X26's own graph.
- `x34_status.txt` (PENDING) — `python -m main.cycle_monitor status rb_toy_ent_s4002_deep --lineage-run
  rb_toy_ent_s4002 --out <the retro root>` after the tick, with Tier 1 read from the archive ledger.
