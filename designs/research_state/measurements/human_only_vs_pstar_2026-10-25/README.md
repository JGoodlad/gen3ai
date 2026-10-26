# P\* against Metamon's HUMAN-ONLY models: the record (FR-1 read; FR-3 playing), 2026-10-25

The rules are in [`PREREG.md`](PREREG.md), committed at `57fd657b` before any counted game. Rows:
[`rows.jsonl`](rows.jsonl); per-cell stats and every unit's counting verdict: [`cells.json`](cells.json) (both written
by [`scripts/aggregate.py`](scripts/aggregate.py)); raw units in `/home/goodlad/dev/gen3ai-reads/hr_reads_2026-10-25/`
(not committed). Abbreviations as in the PREREG.

## FR-1: P\* vs `metamon:SmallRL`, **OURS-BEATS**

1,600 games, **80 / 80 units counted** (regime `f0828cc22aa53b41`; one regime stamp per cell), every game a twin of
X51's P\* vs Kakuna game (the same two teams; 400 / 400 twins matched per panel).

| panel | n | P\* win rate [Wilson 95 %] | vs 0.50 | vs Kakuna on the same twins (X51) | Δ-K (Newcombe) | paired twins: Δ, seed bootstrap; P\*-only wins vs SmallRL / vs Kakuna |
|---|---:|---|---|---|---|---|
| home72 | 400 | **0.785 [0.742, 0.822]** | BETTER | 0.598 [0.549, 0.644] | +0.188 [+0.124, +0.249] | +0.188 [+0.135, +0.240]; 118 / 43 |
| H (`heldout72_hpfix`) | 400 | 0.678 [0.630, 0.721] | BETTER | 0.497 | +0.180 [+0.112, +0.246] | +0.180 [+0.110, +0.248]; 125 / 53 |
| F (`forum72`) | 400 | 0.663 [0.615, 0.707] | BETTER | 0.530 | +0.132 [+0.065, +0.199] | +0.133 [+0.078, +0.188]; 110 / 57 |
| P72 | 400 | 0.658 [0.610, 0.702] | BETTER | 0.433 | +0.225 [+0.157, +0.290] | +0.225 [+0.155, +0.298]; 130 / 40 |
| **U (H, F, P72 equal weight; Kish n_eff 1,200)** | 1,200 | **0.666 [0.639, 0.692]** | **OURS-BEATS** | 0.487 [0.459, 0.515] | **+0.179 [+0.140, +0.218]: SmallRL WEAKER THAN KAKUNA** | — |

- **The verdict:** W-U's LB 0.639 > 0.50 ⇒ **OURS-BEATS**. The human-only 14M model does not outperform P\*; it is
  ≈ 18 pp easier than Kakuna on every panel.
- **Against the prediction** (GUESS W-U 0.72, home72 0.80): a little lower on U, as predicted on home72. The predicted
  panel order (F ≥ H > P72) did NOT hold: the three panels are level (0.658–0.678), so P72's −10 pp deficit vs Kakuna
  does not appear against SmallRL. The P72 gap is something Kakuna exploits, not a general weakness of P\* there.
- Ties 3, forfeit-limit games 8 of 1,600; the halves agree within 4 pp in every cell; median 29–36 turns.
- **Caveats (PREREG §6):** SmallRL differs from P\* in architecture and scale as well as data, so this does not show
  that human data loses to ours inside our network (X59-S is that read). H18 (Baton Pass misread) favours P\*.

## FR-3: P\* vs `LargeRL` / `LargeIL` (HELD, then resumed)

20 LargeRL and 18 LargeIL units were played (the last ones finishing after the hold) before the orchestrator's priority order (2026-10-25: FR-1 first, the CPU is the bottleneck)
put FR-3 on hold (`hold_fr3` in the run dir; the queue skips FR-3 while it exists and never re-runs a unit whose child
is still alive). It resumes after this record ships. No FR-3 number is read before its 1,600 games per model finish.

## FINDINGS

1. **Two units finished `status OK`, n = 20, but did not count** (`peer_clean = False`): `sr_hpfix_10` (FR-1) and
   `lrl_home72_02` (FR-3). In each, one acceptor peer hit upstream's "Battle is already finished, call reset" and
   Metamon's `metamon_to_amago.py:323` handler retried by RECURSION: 985 "Force resetting due to long-tail error"
   lines, i.e. until Python's recursion limit, in both (hence the identical count). An upstream peer bug, not ours,
   unfixed (patching the peer would change the anchor). Each is re-run once with the same seed (the registered rule):
   `sr_hpfix_10`'s re-run is clean and counts; `lrl_home72_02` is moved aside and re-runs when FR-3 resumes. The first
   tries are kept as `cells/<unit>.unclean.*`.
2. Throughput: CPU contention (a live training run and up to 9 queued heavy jobs) cut the queue from 27 units/h to
   ≈ 8 units/h; the orchestrator declined a BLOCKING mark (FR-1 does not block the GPU) and held FR-3 instead.
3. The queue (`scripts/hrrq.py`) was restarted mid-read to add the hold: both supervisors stopped by PID while their
   unit children ran on; the restarted lane skips any unit whose child is alive. No unit was played twice.
4. The PREREG's registration findings (the play tree `d32ffcee`; `HF_HUB_OFFLINE`) stand.
