# V-RANK operator notes (state side; operational choices, none moves a statistic)
- 2026-11-18: operator = a sonnet worker dispatched by the orchestrator. Play worktree /home/goodlad/dev/gen3ai-worktrees/vrank_play at c933afd5a.
- CONCURRENCY 3, not the PREREG's 4: raw lane `--concurrency 3`; at most 3 branch shards running at once (plan --shards 4 as registered; the 4th shard
  waits until a supervisor ends). The orchestrator's operational choice to protect the live B1 training run (GPU, no lease). Moves no statistic.
- Every process: nice -n 19, taskset agent-cpus (5-7,13-15), OMP_NUM_THREADS=1, CUDA_VISIBLE_DEVICES="".
- Timings: raw lane 18:15-~18:44 (30 units, K=3). Stage-0 score (57 states, 1 scorer, mem_cap 8) 1757 s wall, peak 1.07 GB. Branch plan 327 units over 4 shards
  (119/119/38/51); s0,s1 and a chain s2->s3 ran (3 at once); ~60 min wall; 327/327 counted, 0 dropped, 0 peer drops, classify: 6 identical (nulls).
- Stage 0 (outcome-free): s_bar 0.0528, cap_binds false, final_blocks 200; exclusion forum72 2/10 (0.20), p72 1/9, home72 1/28, H 0/10.
  >15 % on forum72 => PREREG 5 hold: stopped here for the orchestrator.

## Ruling on the stage-0 forum72 exclusion hold (2026-11-18 20:15 PST; the orchestrator's, relayed to the operator)
Stage 0: forum72 exclusion 2/10 (0.20) > 15 % (PREREG 5 hold). Ruling: PROCEED to the full score (reads no outcome). HOLD AGAIN before look 1's branch
lanes are filled if the exclusion rate over ALL scored look-1 states is still > 15 % on any panel. Why: 2 of 10 is too small to rule on (95 % interval
roughly 3-56 %); exclusions are outcome-free and 5's hold binds only before the lanes are filled; a high forum72 rate would matter because the read would keep
only states where the honest belief fits, biasing beta_U. Rule: every panel <= 15 % => continue straight through (plan, lanes <= 3 at once, classify, decide,
record, ship); any panel > 15 % => HOLD and report.
- Full score done (Nov 19 ~00:55): 1168/1168 scored, 0 missing, 0 duplicates (part_0of1 57 + part_0of2 561... + part_1of2 550). Exclusion by panel (searched=false): H 1/194 (0.5 %), F 10/197 (5.1 %), P72 10/193 (5.2 %), home72 30/584 (5.1 %). All <= 15 %: continue per the ruling.
- Look-1 plan (Nov 19 ~00:58, --shards 4 as registered; the shard count of an existing plan root is fixed): 6,615 units (1798/1800/1493/1524); 6,288 new.
  To keep <= 3 supervisors at once with 4 shards, a scheduling-only driver ($A/branch_driver.py; pid in branch_driver.log) time-slices: 3 workers, each runs
  `main.x35.kakuna run --state sK --stop-at-epoch now+3000` (a clean stop between units, resumable) on the shard with most units left, not running. No unit,
  seed or plan byte changes. Stop it with the file $A/STOP_DRIVER (it ends after the slices in flight).
- Look 1 finished (Nov 19 20:58 PST): the branch lanes ran 01:00 to 20:58 (6,288 new units; 16 not counted = p72 unit 2002 game 1's two positions, all attempts G-BRANCH
  failures classed peer_first by `v_rank classify`: 2 P72 positions dropped). decide: CONTINUE (beta_U +0.390 [+0.181, +0.575]); record written with `main.reads record --out <file>`
  (the PREREG's `--out $M/record` names a directory but the verb writes ONE FILE: v_rank_look1.json, 10 MB, into record/). No VOID.
