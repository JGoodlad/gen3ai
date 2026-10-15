# T33: the first real Tier-1 plateau ticks, the eval cost, and the trainer's idle budget (2026-10-15)

This is the evidence directory for `designs/endstate/design_x26_readiness.md` §2.6 and its F13–F16. The task is unit T33
(`designs/ops/TASK_BACKLOG.md`). No GPU was used. All CPU work ran under rule 5b: `taskset -c 5-7,13-15`, `nice -n 19`,
`scripts/ops/mem_cap.sh`, and one heavy job at a time.

## What was run

- `tick_loop.sh` (detached) runs `python -m main.plateau tick rb_toy_ent_s4001 --delta-steps 5000000 --lag 4
  --threads 4 --torch-threads 2 --n-envs 64 --max-checks 1` once per due check.
  - Each check is timed with `/usr/bin/time` (wall, user and system CPU, max RSS), and its status is saved afterwards.
  - The ledger is the archive's `models/_ledger`. **The meter writes there by design**, as `plateau_t1` requests,
    their rows and one decision per check. Nothing else under `models/` was written.
  - `py.sh` pins this worktree's `src/` together with the rule 5b CPU set and priority.
- `prod_check.sh` ran one production-network Tier-1 check: `rb_bf_eb_s2001` at 15M vs 5M (Δ 5M, lag 2). It used a
  SCRATCH ledger (`/tmp`), so it is not a registered check.
- `prod_bigbatch.sh` ran one 160-pair `main.h2h play` batch (`rb_bf_eb_s2001` at 15M vs 10M, purpose `audit`,
  scratch ledger), to measure the tail effect of the meter's 40-pair batches.
- `phase_stats.py` gives the per-update phase medians (`collect_ms`, `train_ms`, `*_ms_per_host_step`) over the last
  ¾ of each run's `launcher_child.log`.
- `sampler.py` and `phase_util.py` are a read-only 0.5 s sampler of `/proc/stat` and the trainer's process
  jiffies. They are joined with `nvidia-smi --query-gpu=timestamp,utilization.gpu -lms 500`, and each interval is
  classified as collect or update by the env core's CPU rate. They were run for 175 s on the live toy.
  - **Production procedure (≤ 10 min, no lease, no GPU work):** run the same pair for 300 s beside any live
    production run, then run `phase_util.py`. The trainer's PIDs come from `ps`: the `train_rl_agent.py` child and
    its `rust_env_proc` children. Edit the CPU sets in `phase_util.py` if the trainer's affinity differs from
    0–4 and 8–12.

## Results

### Tier-1 ticks (`rb_toy_ent_s4001`)

`ticks.tsv` holds the raw timings (start and end epoch, wall s, user s, system s, max RSS in kB, rc, load1).
`status_final.txt` is the meter's own status report at hand-off. The table is in the readiness doc §2.6.

- **10 checks, 25M–70M, all GAIN; status CLIMBING; no CONFLICT and no INCONCLUSIVE.**
- The newest node's score against the node 20M back falls 91.9 → 77.5 → 69.2 → 66.2 → 59.5 → 57.3 → 59.0 → 56.5 →
  56.1 → 60.8 %.
- The 10 ticks took 26 min of wall and 34 CPU-min in total.
- The run died at ≈ 72M on a `CompileCanary` FATAL (08:24). The 65M and 70M nodes trained after the last passing
  canary (62.83M), so those two verdicts are void pending the training session (readiness doc F17).
- The operator stopped the loop at 08:39, because nothing further was due.
- `tick_stderr.log` holds every batch line. `status_final.json` is the machine-readable status.

### Seed 4002 (`s4002/`)

The same procedure, run retrospectively on the finished `rb_toy_ent_s4002`: 16 checks, 25M–100M, all GAIN, status CLIMBING at 100M, so a `CAP` stop and no plateau. See `s4002/README.md` and the readiness doc §3.8.

### Cost of a check

| network | games | wall | CPU-s | games/s |
|---|---|---|---|---|
| toy, 40-pair batches | 160–640 per check | 66–299 s | 98–323 | 2.1–8.3 per batch, 4.4 pooled over the first 6 checks |
| production, 40-pair batches (GAIN at 120 pairs) | 240 | 410 s | 650 | 0.57–0.67 |
| production, one 160-pair batch | 320 | 296 s | 510 | 1.16 |

Each check's engine start-up takes 10–22 s. Peak RSS is 0.83 GB for the toy and 1.07 GB for production.

### Phase medians from the run logs

| run | collect_ms | train_ms | inference wait per host step × host steps | fps |
|---|---|---|---|---|
| rb_toy_ent_s4001 (39.9–45.4M) | 9,900 | 9,210 | 5.0 ms × 418 = 2.1 s | 5,250 |
| rb_bf_eb_s2001 (12.0–15.1M) | 19,650 | 42,100 | 18.2 ms × 418 = 7.6 s | 1,573 |
| rb_es_end2_s2001 (12.1–15.0M) | 13,900 | 54,400 | 18.7 ms × 419 = 7.8 s | 1,427 |
| rb_hib_ctrl_s3103 (15.9–18.1M) | 16,500 | 41,900 | 21.0 ms × 419 = 8.8 s | 1,679 |

### The toy, sampled for 175 s

| phase | trainer CPUs busy (of 10) | Python trainer | env core | GPU util |
|---|---|---|---|---|
| collect | 44 % | 0.48 cores | 3.70 cores | 29.5 % |
| update | 16 % | 1.08 cores | 0.00 cores | 81.5 % |

The agent CPUs (5–7, 13–15) were 100 % busy throughout.

## Caveats

- The rates are contention-dependent: the agent cores ran at load 11–15, beside two other agents' jobs. A quiet box
  would be faster, and nothing here is a benchmark.
- The production check is ONE edge (15M vs 5M), early in a run. Near-plateau games could be longer or shorter.
- The (c) GPU rate is the trainer's own model-vs-bot eval cycle. An h2h runs two networks, so §2.6 halves it for
  the pessimistic end.
