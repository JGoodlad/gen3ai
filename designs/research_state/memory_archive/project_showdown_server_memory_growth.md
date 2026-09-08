---
name: project_showdown_server_memory_growth
description: "The :8001 Showdown node server is the dominant \"RAM-over-time\" consumer; launcher restarts never recycle it"
metadata: 
  node_type: memory
  type: project
  originSessionId: efe52a87-8b27-4306-a5ee-edd8dbc6796d
---

Memory investigation 2026-06-03, training at **64 envs** on the 89.7 GB / 8-core (Ryzen 9800X3D) box. Measured by **PSS** (smaps_rollup), not RSS — forked env workers share COW pages so summed RSS over-counts by ~14 GB.

Real (PSS) footprint:
- **env workers ~30 GB** — ~470 MB × 64. Scales with `--n-envs`. NOT a true leak: each launcher restart (every `--restart-interval-hours`, ~3h) respawns them, resetting their Python heaps. Sawtooth, bounded.
- **node Showdown server ~19 GB and the real "over time" growth**: main dispatcher ~5 GB **approaching its 6 GB `--max-old-space-size=6144` cap**, ~16 `room-battle.js` sim-pool workers ~660 MB each (~11 GB), 4 `sockets.js` ~440 MB each.
- trainer_main ~2 GB (rollout buffer + model), eval workers ~1–3 GB (transient per cycle).

**Root cause of the creep:** the launcher's periodic restarts recycle only the **Python training child**. The Showdown node server is started **manually** (`npm run showdown -- 8001`) and **persists across the whole session** — uptime was ~47.5 h here — so its V8 heaps accumulate (retained battle/room/log objects; V8 rarely returns RSS to the OS) with nothing ever reclaiming it. That, plus going 48→64 envs (more concurrent battles → bigger room-battle pool), is what drives RAM up and into swap over a multi-day run.

Mitigation ideas (none applied yet): coordinate a server recycle at a launcher-restart boundary; cap Showdown replay/log/room retention in its config; or fewer concurrent battles. **Do NOT bounce :8001 mid-run** — see [[feedback_dont_kill_training_server]]. Related: [[project_local_sim_bridge]] (server-free path used for fuzz, not training).

Baseline htop screenshot (t=0): Mem 50.5/89.7 G, Swap 5.42/8 G, all 16 cores ~80%, load ~30.
