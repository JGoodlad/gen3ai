# The live trainer's CPU profile, and what one Kakuna unit costs (2026-11-09)

The raw record behind [`designs/ops/core_split.md`](../../../ops/core_split.md). Read-only: every figure comes from
`/proc`, the run's own log, the heavy-slot event log and the user journal. Nothing was pinned, reniced or signalled.

**The run.** `rb_x60t_s3507` (launcher 2157804, trainer 2158269), pinned to CPUs 0-4,8-12 at nice 10, 256 envs on 8
env-worker threads per env process. The agent cores (5-7,13-15) were loaded by other sessions' work throughout.

| file | what it is |
|---|---|
| `profile_rows.jsonl.gz` | 1,199 one-second intervals (1,200.8 s, about 17 PPO cycles, one eval inside): CPU seconds per thread group, run-queue wait, every logical CPU's busy share. `python -m main.ops.trainer_cpu_profile sample` |
| `profile_summary.json` | `python -m main.ops.trainer_cpu_profile summarize profile_rows.jsonl --json` |
| `burst_rows.jsonl.gz` | 53,678 readings at 4 ms over 215 s (3 rollouts): how many env workers, the learner and the autograd thread are RUNNABLE at each instant. `python -m main.ops.trainer_cpu_profile burst` |
| `burst_summary.json` | `summarize` of the burst file |
| `kakuna_unit_scopes.tsv.gz` | one row per finished Kakuna unit scope (`gen3ai-capped-x35kakuna-*`), 2026-11-04 to 11-09 18:10: systemd's CPU time, wall time and memory peak, from the user journal (13,498 units) |
| `analyze.py` | the figures the two summaries do not hold; its output is `supplement.json` |
| `supplement.json` | the agent cores' occupancy and the phase sequence, the log's per-iteration spread, the anchor units' slot wait and run time, the Kakuna units' CPU time by day |

Unzip a row file before `summarize` (the tool reads plain JSON lines).

## What it measured

| phase | seconds | busy threads, mean / p95 / max | env workers | learner | autograd | launcher | run-queue wait, s per s | busiest physical core, p95 (of 2.0) |
|---|---|---|---|---|---|---|---|---|
| rollout | 424.7 | 2.82 / 3.07 / 3.42 | 1.98 | 0.72 | 0.00 | 0.12 | 0.013 | 1.27 |
| update | 751.1 | 1.17 / 1.35 / 1.41 | 0.00 | 0.57 | 0.42 | 0.11 | 0.000 | 1.07 |
| transition (excluded band) | 25.0 | 1.63 / 1.91 / 2.16 | 0.62 | 0.90 | 0.00 | 0.11 | 0.004 | 1.11 |
| all | 1,200.8 | 1.76 / 2.96 / 3.42 | 0.71 | 0.63 | 0.27 | 0.11 | 0.005 | 1.26 |

- The compile-worker pool, the CUDA threads and the env processes' main threads used 0.00 of a thread in every phase.
- The trainer's 10 CPUs were 1.93 busy on average with every tenant counted (p95 3.18); the 6 agent CPUs were 5.57 busy
  (median 5.99) and at or over 5.7 in 59.6 % of the seconds.
- The run's log over the same window: 1,670,941 steps in 1,221.0 s = **1,368 steps/s**. Over its 129 logged iterations:
  `collect_ms` median 24,900; the 123 iterations without an eval average 24,953 (sd 824, 3.3 %), the 6 with one 58,833;
  `update_wall_s` averages 44.17 (sd 4.7 %).
- **The rollout is a strict alternation** (burst file): 17.2 times a second all 8 workers of the training env process
  become runnable for 16.0 ms, then the learner thread runs alone. Workers and the learner were runnable together in
  0.0 % of 17,949 readings. Share of the rollout with k workers runnable: 0: 72.6 %, 8: 19.0 %, 6: 4.3 %, 7: 1.5 %,
  1 to 5: 2.7 % together. In the update phase exactly one thread is runnable (mean 1.00, peak 2).
- **One Kakuna unit** (20 games, greedy, home teams) on 11-09, beside the live run: CPU median 179 s, mean 231 s; wall
  median 150 s; 1.10 threads busy while it runs; memory peak at most 2.7 GiB. Its WAIT for a heavy slot was a median
  145 s in the morning and 1,477 s after 16:00, against a run time of 147 to 179 s.

## Limits

- One run, one recipe, one 20-minute window; the agent cores were loaded by whatever the box was doing.
- The phase rule (rollout at ≥ 1.0 env-worker threads busy, update at ≤ 0.10, the band between excluded) is the tool's.
- The burst probe reads run states at 4 ms, so a burst's length is known to about ±4 ms and its MEAN to well under 1 ms
  (over 400 bursts per rollout); the three rollouts' means were 15.2, 16.3 and 16.5 ms.
- A journal line names a scope by its mem_cap label, so `x35kakuna` covers every lane built on `main.x35.kakuna`;
  11-04, 11-05, 11-07 and 11-08 mix unit sizes and are not 20-game units throughout.
