# V-RANK record

| file | what |
|---|---|
| `stage0.json` | `v_rank stage0` on the first 5 blocks (PREREG §5), outcome-free: s̄ 0.0528, so 200 final blocks, the cap does not bind. Its forum72 exclusion (2 of 10) tripped §5's hold; the orchestrator's ruling is in `operator_notes.md` |
| `v_rank_look1.json` | look 1's `main.reads record` (100 blocks): verdict CONTINUE, β_U +0.390 [+0.181, +0.575], Δβ +0.036 [−0.232, +0.277]; COMPLETE, no VOID |
| `operator_notes.md` | the operator's state notes: concurrency 3, the stage-0 hold ruling, timings |
| `branch_driver.py` | the scheduling-only driver that ran the 4 branch shards 3 at a time |

The 10 MB `v_rank_look1.json` carries every position's rows in `inputs`; the ledger entry of 2026-11-19 is its reading.
