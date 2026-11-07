# S′-HONEST stage 1b (2026-11-06): the incremental builder `r0s4_honest_inc` on whole-game decision sequences

The incremental leak-honest builder (`design_sprime_honest.md` §9) measured OUTCOME-BLIND (no referee, no game result,
no outcome field is read) against `r0s4_honest` at the same states, its cost projected on the replication designer's
own cost model, and the identity of `r0s4_honest` and `r0s4_rolls` at this build's engine stamp. No PREREG; nothing
registered.

All runs: this build's checkout, CPU only, `nice -n 19 taskset -c 5-7,13-15`, one thread a process, under
`mem_cap.sh` in ≤ 45-min chunks, a training run live on the other cores (rule 5b). P\* = `rb_x26_s2601` @280,000,170
(sha256 `814823c5…`), greedy.

| file | what it is |
|---|---|
| [`seq_probe.py`](seq_probe.py) | the measurement: per battle, every searched decision from turn 10 to its latest target on ONE live searcher; the live searcher's own `_search` (P\*, playouts) at each target, the builder step alone elsewhere; `summarize` (CPU by panel and turn bucket, the projected CPU a searched game on the cost model of `sprime_honest_play_2026-11-06/inputs/cost_model.json`, exclusions, the agreement with `r0s4_honest` and the noise floors) |
| [`run.sh`](run.sh) | one probe run in bounded chunks (it resumes mid-battle from the carry the last written decision returned) |
| [`proofs.sh`](proofs.sh) | the identity proofs: the stage-1 tooth (`r0s4_honest`, 24 states) and S′-LIVE's 600-decision `r0s4_rolls` identity re-score |
| [`smt_probe.sh`](smt_probe.sh), [`smt_workload.py`](smt_workload.py) | the SMT probe (one workload alone, on two physical cores, on two SMT siblings) and the builder-only workload (also the late-turn profile) |
| [`run2.sh`](run2.sh) | the same with two workers inside one capped slot (each peaks at ≈ 1 GB of the scope's 8 GB) |
| [`tables.py`](tables.py) | `summary.json` as the tables below |
| [`floor_keys.txt`](floor_keys.txt) | the replicates' subset: stage 1's first 100 states (its own noise floor's) and the 90 late-probe states up to turn 63 |
| [`summary.json`](summary.json) | the summary (`seq_probe.py summarize --honest-rep --inc-rep`) |
| `inc_rows.jsonl.gz`, `inc_rep_rows.jsonl.gz`, `honest_rep_rows.jsonl.gz` | every decision's row: the main run (9,758 decisions, 444 targets); the incremental replicate under another key tag (162 targets); `r0s4_honest` re-run under another key tag at the subset's 190 targets |
| [`tooth.json`](tooth.json), `tooth_rows.jsonl.gz`, [`rescore.json`](rescore.json), `rescore_rows.jsonl.gz` | the identity proofs (`proofs.sh`) |
| [`smt.jsonl`](smt.jsonl) | the SMT probe's runs (wall and CPU seconds of each copy) |

**The runs, in order.** The final code's stage-1b run is the fourth: three earlier ones were stopped as they exposed the
builder's design faults, each recorded in the design doc (§9.2 C17, C24, C27–C29): retiring carried worlds before a
fresh one could replace them emptied the carry; an exclusion that dropped the carry collapsed a battle into a cold
rebuild at every later decision; a stuck battle's full attempt every 4th decision cost ≈ 1,600 CPU-s over one
stretch; a failing warm chain cost ≈ 40 CPU-s every decision. A harness fault too: a battle longer than one 45-min
chunk restarted from scratch at every chunk (the probe now writes every decision and resumes mid-battle from the carry
the last one returned). The rows here are all of the final code (engine stamp
`port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130`).

**Per panel and turn bucket** (world CPU-s over EVERY decision of the sequences, exclusions included; "other" = the forward and playouts of the searched targets; exclusions over every decision; at the targets, both builders' exclusions and full-decision CPU-s):

| panel / bucket | decisions | world CPU-s mean (median, p90, max) | other | excl. (all decisions) | targets | excl. at targets inc / honest | CPU-s at targets inc / honest | same pick inc vs honest |
|---|---|---|---|---|---|---|---|---|
| F/10-19 | 1242 | 4.31 (0.56, 3.83, 314.18) | 2.32 | 1.8 % | 62 | 1 / 3 | 7.82 / 19.79 | 36 / 59 (61 %) |
| F/20-29 | 650 | 3.66 (0.57, 2.88, 297.04) | 2.21 | 2.9 % | 45 | 2 / 3 | 8.89 / 24.52 | 27 / 42 (64 %) |
| F/30-63 | 752 | 3.37 (0.94, 3.34, 232.61) | 2.88 | 5.3 % | 23 | 1 / 2 | 4.57 / 27.8 | 15 / 21 (71 %) |
| F/64+ | 692 | 7.45 (2.21, 4.85, 391.94) | 2.9 | 15.0 % | 18 | 3 / 10 | 4.46 / 116.83 | 5 / 8 (62 %) |
| H/10-19 | 1240 | 1.19 (0.45, 1.94, 80.25) | 2.58 | 0.2 % | 56 | 0 / 0 | 3.32 / 4.15 | 32 / 56 (57 %) |
| H/20-29 | 629 | 1.25 (0.54, 1.93, 51.31) | 2.6 | 0.2 % | 52 | 0 / 3 | 4.69 / 16.42 | 30 / 49 (61 %) |
| H/30-63 | 764 | 1.7 (0.93, 2.77, 47.47) | 2.14 | 0.4 % | 22 | 0 / 1 | 3.23 / 30.2 | 14 / 21 (67 %) |
| H/64+ | 696 | 5.05 (2.01, 7.88, 270.63) | 2.0 | 0.7 % | 18 | 0 / 2 | 5.97 / 77.75 | 10 / 16 (62 %) |
| P72/10-19 | 1216 | 1.25 (0.46, 1.94, 307.75) | 2.4 | 0.3 % | 58 | 0 / 0 | 3.22 / 5.82 | 35 / 58 (60 %) |
| P72/20-29 | 635 | 1.41 (0.53, 2.02, 115.16) | 2.32 | 0.8 % | 48 | 0 / 0 | 3.39 / 8.02 | 29 / 48 (60 %) |
| P72/30-63 | 637 | 2.49 (0.84, 4.26, 56.17) | 2.82 | 5.2 % | 24 | 0 / 0 | 3.8 / 10.5 | 18 / 24 (75 %) |
| P72/64+ | 605 | 7.85 (2.37, 9.95, 354.91) | 2.07 | 20.5 % | 18 | 4 / 9 | 4.2 / 113.71 | 7 / 9 (78 %) |

**Projected per searched game** (S′-LIVE's search-path decisions a game by turn bucket × the bucket's mean decision CPU, plus S′-LIVE's per-game overhead; K = 6 with its κ by bucket: the replication designer's cost model, `sprime_honest_play_2026-11-06/inputs/cost_model.json`):

| panel | CPU-min K = 2 inc / honest | CPU-min K = 6 inc / honest | projected exclusions inc / honest |
|---|---|---|---|
| F | 5.5 / 29.6 | 8.7 / 48.1 | 5.8 % / 14.8 % |
| H | 2.9 / 14.6 | 4.6 / 23.4 | 0.3 % / 3.4 % |
| P72 | 3.3 / 11.8 | 5.3 / 19.2 | 4.0 % / 4.9 % |
| **U** | 3.9 / 18.7 | 6.2 / 30.3 | 3.4 % / 7.7 % |

**The read** (3,840 searched games and their raw twins):

| builder | CPU-h (K = 6) | wall days, agent cores K = 6 | wall days, whole box K = 14 |
|---|---|---|---|
| inc | 434 | 3.1 | 1.3 |
| honest | 1976 | 14.2 | 6.1 |

**Agreement at the targets** (both searched; the floors are each builder against its own replicate under another key tag, on the replicate subset):

| panel | inc vs honest | honest vs honest′ | inc vs honest on that subset | inc vs inc′ | departs from π: inc / honest |
|---|---|---|---|---|---|
| H | 86 / 142 (61 %) | 33 / 63 (52 %) | 39 / 64 (61 %) | 34 / 56 (61 %) | 41.9 % / 37.3 % |
| F | 83 / 130 (64 %) | 38 / 61 (62 %) | 42 / 64 (66 %) | 31 / 52 (60 %) | 37.6 % / 34.6 % |
| P72 | 89 / 139 (64 %) | 43 / 57 (75 %) | 38 / 57 (67 %) | 32 / 49 (65 %) | 36.1 % / 38.9 % |
| U | 258 / 411 (63 %) | 114 / 181 (63 %) | 119 / 185 (64 %) | 97 / 157 (62 %) | 38.6 % / 37.0 % |

**The correlation bound** (the design's C17): 8.3 % of the 9,395 searched decisions held a world older than 3
decisions (548 at age 4, 98 at 5, a tail to 36); the warm chain fit in 7,787, refit in 1,122, failed in 18, waited in
62 (FINDING S′H-15 of the design doc).

**The identity proofs** at the stamp: the stage-1 tooth PASS, 24 / 24 (`r0s4_honest`); S′-LIVE's identity re-score of
`r0s4_rolls` PASS, 600 / 600, 0 mismatches (4,644 CPU-s).

**The SMT probe** (`smt_probe.sh` on battle `0494d3ea…`'s first 26 decisions, builder only, twice each): alone 24.6 /
27.8 s; two copies on two physical cores 26.6–28.7 s; two on one core's siblings 31.4–33.1 s. A core with both siblings
busy gives ≈ 1.63× one worker's throughput (each sibling 1.23× slower than alone).

**Limits (FINDINGs).**
- The playouts at the targets ran with P\*; the builder step elsewhere ran without (the record is the same: the carry
  does not depend on the playouts). The projection adds the targets' measured forward-and-playout CPU by bucket.
- `r0s4_honest`'s records at the targets are stage 1's (B3, its post-fix run) and the late probe's, both at the
  previous stamp; the tooth shows its records unchanged at this one (24 / 24), not each of the 444.
- The noise-floor replicates skip turns 64+ (≈ 5.5 CPU-h more) and the incremental replicate stopped at 162 of its 190
  targets; the pick agreement at 64+ has no floor.
- The runs shared the two heavy slots with other agents and a training run (the measured CPU is each process's, the
  wall time is not a rate); the SMT probe ran beside other agent-core work.
- The projection takes S′-LIVE's per-game decision counts; dlive's stage-1 states are from raw games.
