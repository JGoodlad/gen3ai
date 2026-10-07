# P2: the untaught meter on the old and new stack (2026-10-06)

This is a small paired read taken when `main.untaught_meter` moved its games off poke-env and onto the Rust eval core
(poke-env retirement P2). It is not a verdict. It checks how big the transport boundary is at this resolution.

- **Ref:** `rb_x5ab_blob_s1001`, its last snapshot (15,047,954 steps).
- **Opponent:** `untaught_meter_opponent_v14` (`ai_v14_01_base`, snapshot 24M).
- **Teams:** the untaught 8.
- **Settings:** seed 0, TRAINING regime (both sides sample at T = 1.0), CPU.
- **OLD stack:** HEAD `11d27574`'s `untaught_meter.play_cells`, snapshotted out of git. Two poke-env `RLPlayer`s over the in-process bridge, the Python encoder, one process per team. The bridge was the emission self-check build (the only one in the worktree).
- **NEW stack:** `agents.training.untaught_rust` (the Rust eval core with T2 eager, 32 envs, release build), one engine for all 8 teams.

The two stacks draw different schedules: different RNG, different opponent-team draws. They are paired at the TEAM level only, so the bootstrap is over teams. Each stack's 60-game cells begin with its 20-game cells (checked on `opp_teams`). Timings are contaminated: a training arm launched mid-read. The counts are fine.

| games/team | NEW `rust_eval` | OLD `python_bridge` | Δ NEW − OLD (cluster CI95) | verdict, no floor |
|---|---|---|---|---|
| 20 | 50.62pp [40.00, 61.25] | 50.62pp [43.13, 58.13] | −0.00 [−15.00, +15.62] | NOT DETECTED |
| 60 | 50.21pp [44.58, 55.42] | 54.58pp [51.46, 57.92] | −4.38 [−8.75, +0.42] | NOT DETECTED |

- **Timeouts and draws:** the old stack had 0 timeouts at both depths. The new stack had 1 game (20/team) and 2 games (60/team) end at the turn limit, all scored as draws.
- **What it says:** no shift is detected. The 60-game interval still allows a few pp either way, so the boundary stamp (`_meta.transport`) and the readers' refusal to mix transports are warranted. Never difference a `rust_eval` level against a `python_bridge` level.

- **Files:**
  - `new_rust_eval*.json`, `old_python_bridge*_t<team>.json`: per-team cells.
  - `summary.json`: the read.
- **Scripts:**
  - `scripts/compare.py`: the driver. Its OLD mode imports `/tmp/old_um/untaught_meter_old.py` = `git show 11d27574:src/agents/training/untaught_meter.py`.
  - `scripts/both60.sh`: the 60-game run.
  - `scripts/read.py`: the aggregation, through the meter's own `aggregate(..., allow_transport_mix=True)`.
