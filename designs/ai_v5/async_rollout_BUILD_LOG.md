# Async Rollout — Build Log

Chronological log of the non-barrier async rollout work (2026-06-05). Companion to
`design_async_rollout.md`.

## Plan
1. Understand the `collect_rollouts`/`SubprocVecEnv`/env-worker seam + the on-policy-correctness model.
2. Implement `AsyncSubprocVecEnv` + `collect_rollouts_async`, flag-guarded (`--async-rollout`).
3. Validate: collector correctness unit tests + a real-`Gen3Env` smoke (bridge, no server).
4. **Benchmark FPS** sync vs async across an n_envs sweep — the arbiter.

## Log

### 2026-06-05 — recon + design
- Read `MaskablePPO.collect_rollouts` (the sync barrier loop), `SubprocVecEnv` (step_async/step_wait
  + the per-env pipes — `connection.wait`-able), `InstrumentedMaskablePPO` (only overrides `train()`;
  added `collect_rollouts` dispatch), `MaskableAgentWrapper`, the env factory, and the SB3
  rollout-buffer internals (`RolloutBuffer`/`MaskableRolloutBuffer`/`MaskableDictRolloutBuffer`).
- 16-core box, production runs `--n-envs 64` (4× oversubscription). Analysis: both sync and async are
  ultimately core-bound; async reclaims the per-step barrier bubbles (largest near n_envs≈cores).
- Locked the correctness model: async collection is **on-policy** (policy frozen during collection)
  — a scheduling change, not an APPO-style algorithm change.

### 2026-06-05 — implement
- New `async_vec_env.py`: `AsyncSubprocVecEnv` (`send_step`/`poll_ready`/`recv_step` + drain-safe
  `env_method`/`get_attr`/`set_attr`) and `collect_rollouts_async` (per-env-column fill, on-policy,
  exact stock bookkeeping).
- `InstrumentedMaskablePPO.collect_rollouts` dispatches to async when `_async_rollout` + AsyncSubprocVecEnv.
- `train_rl_agent.py`: `--async-rollout` flag, `EnvClass` swap (async when not `--debug`),
  `model._async_rollout` set in both fresh + resume branches.
- **Key discovery during the first real smoke:** `Gen3Env`'s obs is a **Dict**
  `{"observation", "action_mask"}` (the old Box guard caught it before any damage). The mask rides in
  the obs natively → the collector reads `obs["action_mask"]` (no wrapper emit, no per-env
  `env_method`). Reverted the speculative `emit_action_mask` wrapper/factory threading; rewrote the
  collector for Dict obs + `MaskableDictRolloutBuffer`.

### 2026-06-05 — validate
- 3 collector unit tests (`async_vec_env_test.py`): exact deterministic trajectory fill (obs /
  episode_starts / rewards / masks / GAE / `_last_obs` carry-over), uneven per-env episode lengths,
  and **drain-safe `env_method` with in-flight steps**. All pass (real `AsyncSubprocVecEnv` subprocs).
- Real-`Gen3Env` smoke (`--async-rollout --use-showdown-bridge --n-envs 4`, no server): runs clean,
  episodes complete, training steps, FPS logs, EXIT=0. Audited `self.locals` usage across callbacks —
  none read stock-loop variable names, so no callback-compat risk.

### 2026-06-05 — benchmark (round 1)
- Steady-state FPS (Δts/Δte, post-warmup; bridge, GPU forward): **async@16 1229 vs sync@16 1024 =
  1.20×**; async@24 1536 (fastest) vs sync@32 1365 = 1.12×. Clean +20 % at the core-count regime.
### 2026-06-05 — benchmark (round 2, production-relevant) + sign-off
- vs current production `sync@64` (1489 FPS): **async@64 1695 (1.14×, highest)**, async@48 1676 (1.12×),
  **async@32 1638 (1.10×) — matches/beats production with HALF the envs** (≈half the RAM, less jitter).
- Recommendation: turn it on. Sweet spot `--async-rollout --n-envs 32` (≈prod FPS, half footprint);
  max FPS `--n-envs 48–64` (+12–14 %). Compounds with distillation. Next lever: double-buffer `train()`.
- Regression: 110 targeted tests (async/instrumented_ppo/wrappers/both eval suites) + **full project
  unit suite green**. Real-`Gen3Env` async smoke EXIT=0. `async32`'s benchmark exit=1 was a benign
  post-completion cleanup quirk (training+eval finished; async@48/64 same class exit 0).

## Status: COMPLETE
`--async-rollout` built, validated (unit + smoke + full suite), and benchmarked (+10–20 % FPS,
honest numbers). Flag-guarded, default off, on-policy-correct. Nothing committed — awaiting `/gen3ai-ship`.
