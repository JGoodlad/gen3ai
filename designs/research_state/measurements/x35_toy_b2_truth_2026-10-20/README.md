# The toy side's B2 truth (AM-X47b-2, 2026-10-20): READ A's turns re-branched under T\*

The orchestrator ruled on F-AMX-3 (AM-X47b-2, [`BATCH1_LAUNCH_SHEET.md`](../x35_breakers_stage_2026-10-17/BATCH1_LAUNCH_SHEET.md)
§0) that the TOY side's G-C B2 reads starvation against what the TOY calls near-best. This directory holds that truth.
**Production keeps its production truth.**

| side | B2 truth | rows | continuation stamp (`main.x35.gc.TRUTH_BY_SIDE`) |
|---|---|---|---|
| production (A1 / A2 / D1 vs C0) | READ A, X26@150M greedy on both sides | [`../collapse_pilot_2026-10-17/rows_X26at150M_S32.compact.jsonl.gz`](../collapse_pilot_2026-10-17/rows_X26at150M_S32.compact.jsonl.gz) | `X26@150M greedy (both sides)` |
| toy (X47, toy A1 / A2 vs toy C0) | **this record**, T\* greedy on both sides | [`rows_toyTstar290M_S32.compact.jsonl.gz`](rows_toyTstar290M_S32.compact.jsonl.gz) | `toyTstar@290M greedy (both sides)` |

`python -m main.x35.gc pool` refuses truth rows whose stamp is not the side's, and `readout decide`, `decide-d1` and
`x47 decide` refuse a G-C record whose truth is not its seeds' side's.

## What was run

READ A's recipe ([`../collapse_pilot_2026-10-17/README.md`](../collapse_pilot_2026-10-17/README.md)) with ONE input
changed, the continuation:
- **the same bank** `bank_x26` (content sha `46233db2…`) and **the same 306-turn subset** `subset_x26.json`;
- every legal action × **32** shared dice seeds (CRN), played to the end in the TRUE world;
- **the continuation is T\*** = `rb_toy_ent_s4002_deep/checkpoints/checkpoint_290000087_steps.zip`, sha256
  `cfa7f7bee7b7a43959f496ce6d5a34556bae914de9c80f90f9342a05a22965e7` (R's `toy_pstar`, re-hashed from a read-only
  copy). It plays greedy on both sides.

The command is [`run_truth.sh`](run_truth.sh) `<archive dir>`, then [`summarize.py`](summarize.py) `<rows.jsonl>`.
- **Where:** CPU only (`CUDA_VISIBLE_DEVICES=""`, `nice -n 19 taskset -c 5-7,13-15`, one OpenMP thread, 3 workers) at
  worktree commit `72c045b8`.
- **Heavy slot:** under `scripts/ops/mem_cap.sh` (8 GB cap), one heavy slot per ≤ 45-min chunk, resumable.
- **Time:** ONE chunk: 306 / 306 turns in **22.4 min** wall (≈ 1.1 CPU-h against the ≈ 3.7 CPU-h estimate; the toy
  forward is cheap). The job waited 6,877 s for the slot first (a training run is live).
- **Memory:** peak 1.50 GB, exit 0.
- **Outputs:** the full rows are in `~/gen3ai_archive/x35_toy_b2_truth_2026-10-20/rows_toyTstar290M_S32.jsonl`
  (sha256 `0ed510b3…79b7`). The committed compact form (`truth.compact`) has sha256 `02cf2415…705e`.

## What it shows ([`summary.json`](summary.json))

- **Completeness:** 306 rows, 306 ok, 0 refused, 0 missing, 32 seeds each, and the stamp is the registered one.
- **Decisive turns** (some action separably worse than the best by more than ε = 0.1):
  - toy truth: **178**;
  - production truth: **179**;
  - both: **150**.
- **Near-best sets:** the action-wise set is identical on only **77 of 306** turns.
- **So the two continuations disagree on what is near-best on most turns.** A toy's B2 read against production's
  truth would have measured the gap between the two continuations as much as the arm. That is the reason for the
  refresh.

**Plumbing check (not a read).** `gc pool --side toy` ran end to end on real reads of `bank_x26` (78 s, 1.9 GB RSS).
- **Pairs:** T\* against X47 s3601's newest checkpoint (341M, from the LIVE run), entered as three identical pairs.
- **Result:** B1 −0.020 [−0.034, −0.006] nats; B2 +0.056 [−0.011, +0.122] on the toy truth.
- **Refusal:** the same call on production's truth was refused, exit 2, naming AM-X47b-2.
- **Why this is no read:** the pairs are neither finals nor a control, and the reads are kept in the archive only.
