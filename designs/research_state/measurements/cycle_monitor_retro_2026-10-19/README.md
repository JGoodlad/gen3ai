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

X34's toy plays ≈ 15–20 games/s on rule 5b's CPU when it has a core to itself (≈ 1.3 min per 1,000-game edge;
≈ 2.5 min beside the gate), so its registered rows were started in full: 500 mirrored pairs per edge at the hybrid
lags, `--lineage-run rb_toy_ent_s4002`, into a ledger root OUTSIDE `models/`
(`~/gen3ai_archive/cycle_monitor_retro_2026-10-19/smoke_ledger`; the unit was barred from writing under `models/`).

**PARTIAL: rows 20M–60M complete (14 edges, 14,000 games), 70M partial; the tick was STOPPED** because the box's two
heavy slots were held (one by a 28-hour job) with eleven jobs queued, and the full retro (≈ 154 edges, ≈ 3 h of a
slot) would have blocked them. What the rows read (`x34_status.txt`; the 40M–60M verdicts are the rule's on the rows,
not yet recorded as decisions):

| check | verdict | signals | excess cyclic RMS | Nash support |
|---|---|---|---|---|
| 20M | NOT_APPLICABLE (one edge) | — | — | — |
| 30M | **NONE** (decided, `verify` re-derives it) | — | — | — |
| 40M | NONE | all clear | 0.0 pp | 1 |
| 50M | SUSPECTED (top-up pending) | Hodge (p 0.0015) + triangles (min p 3.1e-5) | 2.0 pp | 1 |
| 60M | SUSPECTED (top-up pending) | Hodge (p 0.0005) + triangles (min p 0.003) | 2.1 pp | 1 |

So the toy's early archive (rb_toy_ent_s4002, 10M–60M) already carries a detectable cyclic residual of ≈ 2 pp while
its Nash frontier stays on the newest node: the same shape as X26's, smaller. Its plateau region (X34 is past 230M,
Tier 1 FLAT once at 200M) is NOT read.

**To finish it** (NOT re-run on 2026-10-19 by the orchestrator's ruling: heavy slots were scarce; resumable at the
batch, the rows already banked are never replayed, and the status reader applies the size bar below by itself):

```bash
# from a worktree at or after the size-bar commit, CPU, rule 5b, when a heavy slot is free (≈ 3 h):
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=""
scripts/ops/mem_cap.sh 10 nice -n 19 taskset -c 5-7,13-15 python -m main.cycle_monitor tick rb_toy_ent_s4002_deep \
    --lineage-run rb_toy_ent_s4002 --out ~/gen3ai_archive/cycle_monitor_retro_2026-10-19/smoke_ledger \
    --threads 3 --torch-threads 3 $(for g in $(seq 20 10 230); do printf -- '--grid %d000000 ' "$g"; done)
# add --panel outside_v1 to play the registered panel's cells too (+ ≈ 1,000 games per check)
python -m main.cycle_monitor status rb_toy_ent_s4002_deep --lineage-run rb_toy_ent_s4002 \
    --out ~/gen3ai_archive/cycle_monitor_retro_2026-10-19/smoke_ledger      # reads only; prints sizes + counted
# or in a GPU lease window (the toy's rows take minutes there): the same with --device cuda under the lease token
```

## Re-read under the SIZE BAR (2026-10-19, the ORCHESTRATOR's ruling on F-ED-27; DESCRIPTIVE)

The amended Tier-2 rule (`design_evaluation.md` §8.2 "The size bar"): a SUSPECTED / CONFIRMED check counts for the
plateau only when its non-transitive SIZE on the recent window (the last 8 checks' nodes; for the ladder, its nodes
in the last 80M of grid) is MATERIAL — the one-sided 95 % lower bound of the per-edge cyclic SD above 1.7 pp; a
NEGLIGIBLE one (upper bound below 1.7 pp) counts as NONE; the bar inside the interval is INCONCLUSIVE.

```bash
python -m main.cycle_monitor ladder rb_x26_s2601 --tier1 --json > x26_ladder_sizebar.json   # ≈ 10 min, 1 core
python -m main.cycle_monitor status rb_toy_ent_s4002_deep --lineage-run rb_toy_ent_s4002 \
    --out ~/gen3ai_archive/cycle_monitor_retro_2026-10-19/smoke_ledger > x34_status_sizebar.txt
python size_bar_oc.py 40 > size_bar_oc.txt                                                   # the OC simulation
```

- **X26 (`x26_ladder_sizebar.json`): the size is MATERIAL at 28 of 37 checks and INCONCLUSIVE at 9; NEVER
  NEGLIGIBLE.** 3.2–5.2 pp through 270M (the 270M window: 3.26 pp [1.86, 4.55], MATERIAL); 280M 2.64 [0.50, 4.15],
  290M 2.21 [0.00, 3.80], 300M 3.00 [1.55, 4.29] INCONCLUSIVE; 310M 3.49 [1.81, 5.06] MATERIAL; 320M–370M
  INCONCLUSIVE (the ladder thins: 4–8 nodes in the window, intervals 0–8 pp wide).
- **X26's two-tier status under the new rule: CONTINUE at every check** (as of the newest check with both tiers,
  320M: Tier 1 GAIN, the counted monitor INCONCLUSIVE). At the Tier-1 plateau pair 270M / 280M it reads CONTINUE
  (270M MATERIAL, 280M INCONCLUSIVE); at 310M (Tier 1 FLAT) MATERIAL. No check reads TRANSITIVE_IMPROVEMENT either:
  the cyclic component is ≈ 3–4 pp of per-edge SD, twice the bar, wherever the window can resolve it. So the
  amendment does not change X26's read: its cycling is MATERIAL, not merely persistent. (Caveat: the ladder is
  50–100 pairs per line on the pool's edges, not the registered 500-pair rows; and 270M / 280M read the same 268M
  node, so their windows overlap.)
- **X34 (`x34_status_sizebar.txt`, the registered rows):** 30M / 40M NONE (size 0.00 pp [0, 3.71] and [0, 2.78]);
  50M 2.63 pp [1.48, 3.87] and 60M 2.54 pp [1.64, 3.49] — INCONCLUSIVE, and still SUSPECTED pending their top-ups.
- **The estimator's operating characteristics (`size_bar_oc.txt`, SIMULATED):** unbiased on the target's scale;
  NEGLIGIBLE / INCONCLUSIVE / MATERIAL = 34/6/0 at τ = 0, 17/23/0 at 1 pp, 1/35/4 at 1.7 pp, 0/13/27 at 2.5 pp,
  0/0/40 at 4 pp (40 runs a cell). One check resolves sizes below ≈ 1 pp or above ≈ 2.5 pp (F-ED-28).

## Files

- `x26_ladder_sizebar.json` — the X26 ladder re-read with each check's windowed size, its state, the counted
  verdict, Tier 1 and the DESCRIPTIVE two-tier signature.
- `x34_status_sizebar.txt` — X34's status under the size bar.
- `size_bar_oc.py` / `size_bar_oc.txt` — the size estimate's simulated operating characteristics.

- `x26_ladder_accumulated.json`, `x26_ladder_window80M.json` — every check's verdict, signals, summary, the newest
  node's lag curve and the significant triangles.
- `calibration_x26_graph.txt` — the null-calibration run on X26's own graph.
- `x34_status.txt` — `python -m main.cycle_monitor status rb_toy_ent_s4002_deep --lineage-run rb_toy_ent_s4002
  --out <the retro root>` after the stopped tick, with Tier 1 read from the archive ledger.
