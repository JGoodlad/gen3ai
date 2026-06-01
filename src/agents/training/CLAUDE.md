# CLAUDE.md — Training (`src/agents/training/`)

Callbacks, reward manager, episode/turn tracking, stall detection, and the bot-eval pipeline.
**How to launch training** (commands, flags) lives in the root `CLAUDE.md` → Training /
Launcher; this file documents the subsystems' internal design. The `TurnDelta` fold and the
LiveView/TurnView/LegalActions read-models it consumes are documented in
`src/agents/battle/CLAUDE.md`. The obs-build performance gate is in
`src/agents/observation/CLAUDE.md`.

## Bot evaluation (subprocess, non-blocking)

`PerOpponentEvalCallback` (non-self-play path) does **not** eval in-process. On each
scheduled step it snapshots the live weights (`model.save`) and spawns `--eval-workers`
(default 3) `main.eval_worker` subprocesses that **work-steal** opponents from a shared
pool (atomic `O_EXCL` claim files — a worker that finishes a cheap opponent grabs the
next, so uneven per-opponent cost self-balances), load the **frozen** snapshot, and play
against the shared Showdown server **without pausing training**. Each opponent writes
`result__<opponent>.json`; when all workers finish the parent merges them → TensorBoard +
TUI + best-model (the winning snapshot is promoted by copy, not re-saved). Forensic traces
land under `<run_dir>/eval_traces/step_<N>/<opponent>/`. The eval summary itself is
written to `metadata.json` as a **top-level `latest_eval`** block (step-labeled, NOT
nested under a checkpoint) — robust to the async timing (an eval can finish after a
newer checkpoint, or before any checkpoint exists); `save_model_snapshot` carries it
forward so a later checkpoint never erases it.

The frozen snapshot makes parallel eval correct (a worker can't read mutating in-memory
weights), and the fresh process returns all eval memory to the OS on exit (no fragmentation
in the trainer). Behaviors:
- A trigger that fires while the previous cycle still runs is **skipped** (logged) — on CPU
  an eval can outlast its interval; cadence just goes sparser.
- A worker crash is **logged-and-continued**, never fatal (its opponents are just missing
  for that cycle).
- **Graceful shutdown waits for eval to finish**: a scheduled restart is self-initiated by
  `GracefulRestartCallback` at a rollout boundary and the launcher won't force-kill until the
  child overruns the deadline by `--restart-grace-minutes` (20 min), so the drain budget is a
  full `_ABORT_EVAL_DRAIN_SEC` (10 min) AFTER the checkpoint is saved — long enough for a CPU
  eval to complete. Even the pathological forced-SIGTERM case (already overran → ~90s SIGKILL)
  is safe: the checkpoint is saved first, only the in-flight eval can be lost.
- **On resume the last eval is re-published to the TUI** from the resumed checkpoint's
  `metadata.json`, so the eval panel isn't blank until the next cycle.

| Flag | Default | Notes |
|------|---------|-------|
| `--eval-workers` | `3` | Eval subprocesses per cycle; work-steal opponents from a shared pool. Capped at the opponent count. |
| `--eval-device` | `cpu` | Device for eval-worker inference. `cpu` decouples eval from the training GPU. |

Eval concurrency in the worker is `_EVAL_SUBPROCESS_CONCURRENCY` (5/opponent) — low so
the shared server isn't flooded while training also uses it.

## Showdown port threading (the `server_config` seam)

`train_rl_agent.py --showdown-port <port>` builds **one** `ServerConfiguration` in `main()`
via the single constructor `localhost_server_configuration(port)` (in
`poke_env.ps_client.server_configuration`) and threads it to **every** Showdown client —
the training-env players (carried into the `SubprocVecEnv` spawn workers via the env-factory
closures), eval, and self-play. Every player-creating callback takes a `server_config` param
(defaulting to port 8000 for standalone use) and builds its players from it — **never** from a
bare `LocalhostServerConfiguration` constant. `server_port_threading_test.py` is the
regression guard: it fails if any of these callbacks hardcodes the default port instead of
threading the configured one (the original bug had the now-retired replay recorder connecting
to :8000 while training ran on :8001; eval forensic traces inherit the same guard).
There is no environment variable; `train_rl_agent.py`'s own default is 8000, but the **launcher**
overrides it to 8001 before forwarding (see `src/main/launcher/CLAUDE.md`). The launcher
forwards `--showdown-port` verbatim (it strips only launcher-owned flags).
