# V-RANK record

| file | what |
|---|---|
| `stage0.json` | `v_rank stage0` on the first 5 blocks (PREREG §5), outcome-free: s̄ 0.0528, so 200 final blocks, the cap does not bind. Its forum72 exclusion (2 of 10) tripped §5's hold; the orchestrator's ruling is in `operator_notes.md` |
| `v_rank_look1.json` | look 1's `main.reads record` (100 blocks): verdict CONTINUE, β_U +0.390 [+0.181, +0.575], Δβ +0.036 [−0.232, +0.277]; COMPLETE, no VOID |
| `v_rank_look2.json` | look 2's `main.reads record` (the final, 200 blocks, both looks' positions): verdict RANK-HOLDS-ON-U, β_U +0.315 [+0.153, +0.449], β_home +0.333 [+0.210, +0.447], Δβ −0.018 [−0.215, +0.167]; 1,124 U / 1,114 home positions; COMPLETE, no VOID |
| `operator_notes.md` | the operators' state notes for both looks: concurrency 3, the stage-0 hold ruling, timings, the dropped deals, the cost |
| `branch_driver.py` | the scheduling-only driver that ran look 1's 4 branch shards 3 at a time |
| `branch_driver2.py` | the same driver for look 2 (the plan root `branches2` and the log name changed, nothing else) |

Each record carries every position's rows in `inputs` (10 MB for look 1, 20 MB for look 2, which holds both looks' rows); the ledger
entries of 2026-11-19 and 2026-11-20 are their readings.
