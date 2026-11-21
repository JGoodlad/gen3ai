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

## Look 2 (operator: a worker dispatched by the orchestrator, 2026-11-19; spec `read_spec_look2.json` shipped at ec17bd0d8 before any look-2 game)
- Same play worktree (c933afd5a), same rule-5b shape (nice 19, agent cores, OMP 1, no GPU), concurrency 3 (the orchestrator's cap: at most 3 lanes). Env: $A/env2.sh.
- Raw lane started Nov 19 21:07 PST: `main.reads run --state $A/look2 --lane anchor --hours 3 --concurrency 3` (30 fresh units: u2 2005-2009 x 3 panels, home2 2115-2129).
- Raw lane 21:07-21:40 PST (30 units, K=3): 30/30 counted, unplayed 0, one engine, regimes equal to look 1's; u1 / home1 imported from look 1's state.
- Draw (once): selected2.jsonl 1,172 states (H 197, F 197, P72 193, home72 585).
- Full score started 21:42 PST: two scorers --part 0/2 and 1/2 into scored2 (mem_cap 8 each), as PREREG 8 step 4.
- Full score done Nov 20 02:40 PST: 1,172/1,172 scored, 0 missing, 0 duplicates (part_0of2 582 + part_1of2 590; 18,020 s and 16,686 s wall, peak 1.2 GB each).
  Exclusion by panel (searched=false), outcome-free: H 4/197 (2.0 %), F 10/197 (5.1 %), P72 8/193 (4.1 %), home72 25/585 (4.3 %). All <= 15 %.
- Look-2 plan (Nov 20 02:42, --shards 4 as registered, fresh root branches2): 6,790 units (1825/1809/1605/1551), 6,760 branches + 30 nulls.
- Branch lanes started Nov 20 02:43: $A/branch_driver2.py (look 1's branch_driver.py with the plan root branches2 and the log name branch2_; scheduling only,
  3 workers, slices of 3000 s via --stop-at-epoch; pid in $A/branch_driver2.pid). Stop it with the file $A/STOP_DRIVER.
- Look 2 finished (Nov 20 22:47 PST): the branch lanes ran 02:42 to 22:47 (20.1 h; 6,790 units, 6,781 counted; 9 not counted = p72 unit 2007 game 4's two
  positions, 8 branch units and that deal's null, every attempt (36 a unit) a G-BRANCH failure classed peer_first by `v_rank classify`: 2 P72 positions dropped,
  the 29 other nulls classed identical). decide (frozen rule, look 2, both looks' positions): RANK-HOLDS-ON-U, beta_U +0.315 [+0.153, +0.449],
  beta_home +0.333 [+0.210, +0.447], delta -0.018 [-0.215, +0.167]; positions U 1,124 / home 1,114; no boundary flag; COMPLETE, no VOID.
  Record: `main.reads record --state $A/look2 --out <record>/v_rank_look2.json` (20 MB). Outputs: $A/classify2.out, decide2.out, record2.out.
- Look-2 cost: branches 225,806 CPU-s (62.7 CPU-h; 8,632 of it the 324 failed attempts), scoring 34,318 CPU-s (9.5 CPU-h), about 72.3 CPU-h in all
  (raw lane not metered, 33 min wall at 3); 25.7 h wall from the raw lane's start to the last branch.
