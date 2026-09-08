---
name: project_throughput_profile
description: Live py-spy profile of self-play training — what bounds FPS and which optimizations actually pay
metadata: 
  node_type: memory
  type: project
  originSessionId: efe52a87-8b27-4306-a5ee-edd8dbc6796d
---

> **Archived 2026-09-08** — ai_v5-era py-spy profile; superseded by project_throughput_compile and the rust bridge. Preserved verbatim; nothing below is current.

Profiled the live self-play run with **py-spy** (attach by PID, `ptrace_scope=0` — see [[feedback_ptrace_debug]]) 2026-06-03. Findings (RTX 3080 Ti 12 GB, Ryzen 9800X3D 8c/16t, 89 GB RAM):

- **Wall-clock: ~86% rollout collection, ~14% PPO update.** GPU idle ~86% of the time (2–4% util during rollout, 100% only in the update).
- **Per env worker: ~37% busy / ~63% blocked** (~47% at the SB3 per-step barrier `step_wait`, ~16% on the JS-sim round-trip). Of the busy CPU, **~70% is the self-play OPPONENT's NN forward run on CPU** (`SingleAgentWrapper.step → opponent.choose_move → features_extractor.forward`). obs build is only ~2–4% of wall — NOT the bottleneck in self-play (contradicts the obs-benchmark gate, which uses a random action + no opponent net).
- **Barrier semantics (confirmed from SB3 source):** `step_wait` does `[recv() for all remotes]` — every rollout step runs at the speed of the slowest worker. A long *game* does NOT stall others (workers auto-reset in-step); the cost is per-step tail latency (resets, CPU descheduling, swap faults).
- **Workload is LATENCY-bound, not CPU-bound** (~24% CPU idle at 48 envs). So oversubscription HELPS hide sim latency → **n_envs is not the FPS lever** (48 ≈ 64 ≈ flat; 32 didn't help). Treat n_envs as "as many as RAM allows without swapping."
- Opponent pool: each worker loads ONE snapshot per `pool_generation` (LRU-cached); up to ~14 distinct active weight sets across the workers.

**Real levers, ranked:** (1) **distill the opponent to a small CPU net** — kills the dominant per-tick critical-path cost, no IPC/VRAM/correctness baggage; (2) **async vec-env** (act on ready cohorts, over-provision envs) — removes the straggler tax, stays on-policy. GPU-batching the opponent is marginal at this scale (tiny batches split ≤14 ways, IPC + a new barrier, only ~860 MB free VRAM). Overlapping the PPO update caps at ~14% and costs on-policy purity — low ROI. Memory side: [[project_showdown_server_memory_growth]].

**UPDATE 2026-06-05 — async vec-env is BUILT (`--async-rollout`).** `AsyncSubprocVecEnv` + on-policy `collect_rollouts_async` (`src/agents/training/async_vec_env.py`), dispatched by `InstrumentedMaskablePPO`; flag-guarded (default off = stock `SubprocVecEnv`); masks ride in the Dict obs; drain-safe `env_method`. Design: `designs/ai_v5/design_async_rollout.md`. **Measured (bridge, GPU forward, steady FPS): +20% at n_envs=16, +14% at production n_envs=64 (1489→1695); async@32 ≈ sync@64 with half the envs.** CAVEAT: benchmarked with HEURISTIC opponents (not `--self-play`), where more envs kept helping (sync 16/32/64 = 1024/1365/1489) — contrary to the self-play "n_envs not the lever / 24% idle CPU" finding above. The self-play regime is the latency-bound/idle-CPU one async targets, so its gain is plausibly ≥ this, but UNMEASURED — re-bench under `--self-play` (seeded pool) before quoting a production self-play number. Not committed (awaiting `/gen3ai-ship`).
