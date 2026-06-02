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
land under `<run_dir>/eval_traces/step_<N>/<opponent>/`, alongside a per-cycle
**`eval_manifest.json`** (`write_eval_manifest`) recording exactly which model produced them
— `num_timesteps`, `git_hash` + `arch_signature` (read from the run's `metadata.json` /
`model_config.json`), and a `snapshot` pointer. The eval snapshot is normally ephemeral
(`model.save` → workers load → deleted in `_cleanup`) and the eval `step` rarely lines up with
a persisted `checkpoint_<N>_steps.zip`, so the prober can't reload the *exact* weights unless
they're retained: `--keep-eval-snapshots N` copies the snapshot into
`eval_traces/step_<N>/snapshot.zip` (keeping the N most-recent) and points the manifest at it.
The prober consumes the manifest to load the exact model, falling back to the nearest
checkpoint. **The trainer grooms the traces it writes**: after each cycle
`_prune_eval_traces` keeps only the `--keep-eval-trace-steps` (default 20) most-recent eval
step dirs, and `_prune_eval_snapshots` keeps the `--keep-eval-snapshots` (default 10)
most-recent snapshots — so `eval_traces/` stays bounded without any external task
(`python -m main.prober.groom` is the manual fallback). The eval summary itself is
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
| `--keep-eval-snapshots` | `10` | Retain the N most-recent eval weight snapshots in `eval_traces/step_<N>/snapshot.zip` (~27MB each; default ≈270MB) for bit-exact prober replay. `0` writes the identity manifest only; the prober then loads the nearest persisted checkpoint. The trainer auto-prunes to this cap each cycle. |
| `--keep-eval-trace-steps` | `20` | The trainer keeps only the N most-recent eval **step dirs** under `eval_traces/` after each cycle (`0` = keep all), so forensic data stays bounded. `python -m main.prober.groom` is the manual fallback. |

Eval concurrency in the worker is `_EVAL_SUBPROCESS_CONCURRENCY` (5/opponent) — low so
the shared server isn't flooded while training also uses it.

## Self-play opponents (`--self-play`, gated behind pathology hunting)

When `--self-play` is set, `SelfPlayCallback` replaces `PerOpponentEvalCallback` and the
training opponents become frozen snapshots of the agent itself, drawn from a directory-backed
`SnapshotPool` (`snapshot_pool.py`; state reconstructed from `<run_dir>/snapshots/` on every
restart — no manifest). Design lives in `designs/ai_v5/`. Key behaviors:

- **Opponents sample, they don't argmax.** Training opponents are built with `stochastic=True`
  (now the `RLPlayer` default) so the learner trains against the policy's full action
  distribution — a richer, less-exploitable signal than the greedy move. Temperature is
  `--self-play-temp` (default `1.0` = the policy's own distribution; >1 flatter). **Eval and
  measurement players stay greedy**: the measured trainee, the pool sentinels, and the bot-eval
  players pass `stochastic=False` explicitly, so `win_rate_vs_bots` (curriculum) and
  `win_rate_vs_pool` (promotion) remain stable, comparable control signals.
- **Opponent snapshots are version-checked.** They load via `load_model_snapshot` (not a raw
  `MaskablePPO.load`), and `SnapshotPool` writes a shared `model_config.json` next to its
  snapshots, so an arch-mismatched snapshot fails with a clean `ModelVersionError` instead of
  loading mismatched weights.
- **The opponent matches the trainee's strictness — a stale decision CRASHES** (crash-over-corruption).
  `SingleAgentWrapper` polls the opponent's `choose_move` on the *training* thread while POKE_LOOP
  mutates its battle, so the opponent can read a request whose captured snapshot (`ctx.legal`)
  diverges from the live battle by serialize time — e.g. a faint flips the battle to a force-switch
  (`available_moves` → `[]`), or a switch target leaves `available_switches`. That raises a
  `StaleDecisionError` (from `assert_decision_current` / `action_to_order`), which **propagates and
  crashes the worker**, exactly like the trainee (`gen3_env.py` asserts with no fallback → the
  launcher restarts from the last checkpoint). It is **not** caught and deferred to
  `choose_default_move()`: in self-play the opponent IS the trainee's training signal, so a garbage
  default move would be gigo. (An all-zero mask still legitimately returns `idx=None` → a default;
  that is "no legal action", not staleness.) The underlying race — the opponent's main-thread read
  is **not** protected by POKE_LOOP's per-battle `asyncio` lock (which only serialises POKE_LOOP
  coroutines, not the training thread) — is **prevented at the source** by
  `SingleAgentWrapper._settle_opponent_battle`, which drains POKE_LOOP's in-flight `parse_request`
  for the opponent's battle (yields until its decision signature is stable) **before** the poll;
  the strict crash above is the backstop, and the comprehensive `assert_decision_current` (every
  action axis: moves+disabled, switches+species, force_switch/trapped/maybe_trapped/wait/struggle)
  is the detector. **Full context — mechanism, why it was hard, and the three-tier verification
  (deterministic guard → single-env `--widen` fuzz → faithful `GEN3_FORCE_SELFPLAY` stress) — is
  in `RACE_FUZZ_README.md`.** (`StaleDecisionError` lives in `agents/action/mapper.py`. The settle
  is **unconditional** in production — no disable switch; the deterministic
  `single_agent_wrapper_test.py` guards that `step` still calls it. `GEN3_FORCE_SELFPLAY` forces
  100% self-play for the stress.)
- **Self-play engages in the first process, not only after a restart.** The env is built before
  the model exists (the model needs the env's spaces), so on the first self-play process
  `_maybe_engage_self_play` seeds the pool from the loaded weights and rebuilds the env with
  pool opponents (then `set_env`). The worker watchdog is started *after* this, just before
  `learn()`. Later restarts find the pool already populated and skip the rebuild.
- **`--debug --self-play` exercises the real path** (seed → pool eval → promotion) on a fast
  eval cadence, so a CPU smoke against a `9XXX` server validates the wiring without disrupting
  the `:8001` training server. `selfplay_opponent_fuzz_test.py` covers the opponent load + legal
  play (both modes) + version check in-process via the local bridge (no server).

## Process liveness guards (`watchdog.py`)

Two daemon-thread watchdogs keep a hung/abandoned run from lingering:

- **`start_subprocess_watchdog`** — for the `SubprocVecEnv` path. A crashed worker leaves the
  parent blocked on a pipe `recv` forever; this thread polls `processes` and `os._exit(1)`s the
  moment a worker dies with a nonzero exitcode. Started *after* env construction (and, in
  self-play, after `_maybe_engage_self_play` rebuilds the env), right before `learn()`. It is a
  **no-op on the `--debug` DummyVecEnv path** (no worker processes to watch).
- **`start_orphan_watchdog`** — for the `--debug` smoke path, which has no worker watchdog. A
  smoke run is a child of the launching shell/agent; if that parent dies the run is orphaned
  (PPID changes) and a hung smoke (e.g. a vanished `9XXX` server) would otherwise sit as a
  multi-GB zombie indefinitely. This thread captures the launching PPID up front and `os._exit`s
  when `os.getppid()` *changes* (by-change, not `== 1`, so PID-namespace subreapers count).
  Started early in `main()` inside the `if args.debug:` block — before team/env/server setup —
  so a startup hang is covered too. **Real launcher-managed runs keep a live parent and never
  arm it.** Regression test: `watchdog_test.py` (subprocess-driven orphan + no-false-fire).

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
