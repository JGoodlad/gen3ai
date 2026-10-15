# T33 on seed 4002: the Tier-1 plateau ticks, run retrospectively on `rb_toy_ent_s4002` (2026-10-15)

This is the second toy seed through T33's procedure (the parent `../README.md`). It feeds
`designs/endstate/design_x26_readiness.md` §3.8 and its FINDINGS F32–F35. No GPU was used.

## The run

`rb_toy_ent_s4002` is T-ENT (§3.2) at seed 4002, pinned at `3899277a` for its whole life (`metadata.json`
`pin_history`). It is the same recipe as s4001. It reached the 100M CAP cleanly (final step 100,081,428), with a
checkpoint every 1M. All 11 of its `CompileCanary` checks PASS (updates 10, 100, …, 1000; the last at 98.31M), so no
node is suspect. Its 100M node trained 1.7M steps after the last canary, which is the ordinary gap between scheduled
canaries, and the run ended with no FATAL.

## What was run

- `tick_loop.sh` is T33's script with two changes: the scratch dir is `/tmp/t33_s4002`, and the `mem_cap` unit name is
  `t33s4002tick`. It runs `python -m main.plateau tick rb_toy_ent_s4002 --delta-steps 5000000 --lag 4 --threads 4
  --torch-threads 2 --n-envs 64 --max-checks 1` once per due check, timed by `/usr/bin/time`.
  - Rule 5b: `../py.sh` pins `taskset -c 5-7,13-15` and `nice -n 19`, each check runs under `scripts/ops/mem_cap.sh`
    (12 GB), and only this one heavy job ran.
  - The meter wrote its 16 `plateau_t1` requests, rows and decisions to the archive's `models/_ledger`, by design.
    Nothing else under `models/` was written.
- The loop ran from 14:47 to 15:17 PDT (16 checks, 29.5 min wall, 58 CPU-min; load1 7.3–11.6 at each check's start).
- `scores.py` gives each check's newest-node score: (W + D/2) / games, pooled over the check's `[h2h] batch` lines.
  Run on T33's s4001 log, it reproduces the parent's published scores exactly (91.9 … 60.8 %).

## Files

| file | holds |
|---|---|
| `ticks.tsv` | per check: start and end epoch, wall s, user s, system s, max RSS kB, rc, load1 |
| `tick_stderr.log` | every engine-up, batch (W/L/D, pair counts) and decision line |
| `status_final.txt` / `status_final.json` | the meter's own status after the 100M check |
| `scores.tsv` | `scores.py`'s output: check, pairs, W, L, D, score % |

## Result

**16 checks, 25M–100M, all GAIN; no FLAT, no CONFLICT, no INCONCLUSIVE; Tier-1 status CLIMBING at the 100M check.**
Under the §3.4 rule the seed did not plateau before its cap: it is a `CAP` stop at 100M.

| check | pairs | newest's score vs W-back | LLR |
|---|---|---|---|
| 25M | 80 | 88.4 % | +3.036 |
| 30M | 120 | 81.7 % | +4.139 |
| 35M | 160 | 65.0 % | +3.536 |
| 40M | 160 | 63.6 % | +3.381 |
| 45M | 200 | 61.1 % | +3.463 |
| 50M | 320 | 58.5 % | +3.723 |
| 55M | 400 | 55.0 % | +3.159 |
| 60M | 280 | 56.9 % | +2.952 |
| 65M | 680 | 53.6 % | +3.071 |
| 70M | 400 | 55.2 % | +3.068 |
| 75M | 440 | 55.3 % | +3.146 |
| 80M | 880 | 52.9 % | +2.967 |
| 85M | 720 | 53.6 % | +3.175 |
| 90M | 1,200 | 52.6 % | +3.420 |
| 95M | 760 | 53.4 % | +3.286 |
| 100M | 600 | 53.8 % | +3.015 |

From 80M on, the newest node scores 52.6–53.8 % against the node 20M back, which is about +18 to +27 Elo per 20M
steps. The GSPRT's H1 is 0.52, so a FLAT needs the true score near 0.51 or below; these are above H1, and so they read
GAIN. The detail, the comparison with s4001 and the §3.5 implication are in the readiness doc §3.8.

## Caveats

- These rates are contention-dependent (agent cores at load 7–12) and are not a benchmark.
- The score is a point estimate over a sequential test's games; a GSPRT stopped at its bound overstates the effect a
  little (it stops on a favourable run). The verdicts are the registered quantity, the scores are descriptive.
