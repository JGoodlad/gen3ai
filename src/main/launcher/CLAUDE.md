# CLAUDE.md — Training Launcher (`src/main/launcher/`)

The launcher wraps `train_rl_agent.py` for long, unattended runs. **Invocation commands
(start fresh / resume) live in the root `CLAUDE.md` → Launcher section** — this file documents
how it works internally. Modules: `checkpoint.py`, `worktree.py`, `child.py`, `input.py`,
`run.py`, `state.py`, `ui.py` (+ `__init__.py` `main()`).

## What it provides

- **Periodic restarts** — kills and relaunches the child every N hours to reclaim pymalloc
  fragmentation; the child saves a checkpoint on SIGTERM and the launcher picks it up
  automatically.
- **Crash auto-restart** — when the child *self-crashes* (unhandled exception → any
  non-`INTERRUPTED` exit), the launcher snapshots its output to a per-crash
  `<run_dir>/crashes/restart_err_<token>.txt` (a timestamp + random hex so back-to-back crashes
  never collide; never overwritten, unlike the reused `launcher_child.log`; folded under a
  `crashes/` subfolder so they don't clutter the checkpoint listing) and relaunches from the last
  checkpoint. A **circuit-breaker** (`--max-crash-restarts`, default 3) stops the run after that
  many *consecutive rapid* crashes (each within `_FAST_CRASH_SECONDS` = 600 s / 10 min of launch)
  so a deterministic startup crash can't spin forever; a crash after sustained progress resets the
  counter. The window is deliberately well past the 3+ min it takes to bring up the SubprocVecEnv
  workers + Showdown connections, so a startup-time crash is still counted as "rapid" rather than
  misread as progress. If a crash has no checkpoint to resume from, it's fatal — the launcher
  propagates the child's exit code rather than masking it. The dashboard shows a
  `↻ N restarts (M crash)` badge and the exit summary reports the crash count.
- **Worktree isolation** — at startup, creates a detached git worktree pinned to the current
  HEAD (or to the commit recorded in the checkpoint's `metadata.json` when resuming). Agent
  pushes to `main` never affect a running session.
- **Rich TUI** — live dashboard showing metrics, FPS, restart countdown; `l` for logs, `r` to
  restart now, `c` for forced checkpoint, `q` to quit cleanly.
- **Crash reporting** — child stdout/stderr is streamed live to `<run_dir>/launcher_child.log`
  (complete even if the child hard-`os._exit`s, bypassing Python cleanup) and held in a
  5000-line in-memory scrollback. On a non-zero exit the last 100 lines are dumped to the
  terminal after the TUI closes; on *every* exit (crash, complete, quit) the full log path is
  printed and the file is finalized (the in-memory buffer is flushed to it as a fallback if
  streaming never started).

## Exit codes (`src/main/exit_codes.py`)

| Code | `TrainExitCode` | Meaning |
|------|----------------|---------|
| 0 | `COMPLETE` | All steps done — launcher stops |
| 15 | `INTERRUPTED` | SIGTERM received, checkpoint saved — launcher restarts |
| 1 | `CRASH` | Unhandled exception — launcher saves `crashes/restart_err_<token>.txt` and auto-restarts from the last checkpoint (up to `--max-crash-restarts` consecutive rapid crashes, then gives up; any non-enum exit code is treated the same way). A crash with no checkpoint to resume from is fatal: the child's exit code is propagated and the crash log printed. |

## Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--restart-interval-hours` | `3.0` | Set to `0` for a single run with no restart |
| `--max-crash-restarts` | `3` | Consecutive rapid self-crashes (each < 10 min after launch) to auto-restart through before giving up. `0` = unlimited. A crash after sustained progress resets the counter (see **Crash auto-restart**). |
| `--restart-grace-minutes` | `20.0` | Force-kill window after a scheduled restart's deadline (child overran its rollout boundary). A child that ignores the SIGTERM is SIGKILL'd after a 90 s grace, and a launcher-forced kill restarts from the last checkpoint (not treated as a fatal crash). The dashboard shows `⚠ no child output for Nm` once the child has been silent > 2 min, so a stall is *visible* — but there is no auto-restart on stall. A connection failure crashes loudly rather than hanging: a connect-time failure via the `Gen3Player` connect guard, and a **mid-battle** websocket drop via the `_AsyncQueue` disconnect guard (`PSClient.listen()` sets `_disconnected` on an abnormal close; the env's blocking `step`/`reset` get races against it and raises `ShowdownException` instead of waiting forever for a message the dead listen task can't deliver). Both deterministic — no timeout guess. Other hangs are surfaced for investigation rather than guessed at with a timeout. |
| `--no-pin` | off | Skip worktree creation; run from the current source tree |
| `--sync-to-main` | off | When resuming from a checkpoint, pin the isolated worktree to the current HEAD instead of the checkpoint's original git hash. Use this to pick up UI or tooling fixes on `main` without discarding the checkpoint. |

All other flags are forwarded verbatim to `train_rl_agent.py` (the launcher strips only
launcher-owned flags).

## Resume contract

The checkpoint must have a `metadata.json` with a `git_hash` field (written automatically by
`save_model_snapshot()`). The launcher pins the worktree to that exact commit so the resumed
run uses the same code as the original — unless `--sync-to-main` is passed.

## Showdown port default

The launcher **defaults `--showdown-port` to 8001** (`DEFAULT_TRAINING_SHOWDOWN_PORT` in
`launcher/checkpoint.py`, injected in `launcher/__init__.main()` via
`_apply_default_showdown_port`) so a long session never rides on the shared dev server (8000),
where a routine dev `npm run stop` would drop every worker's connection at once and the
connection guard would crash the run. An explicit `--showdown-port` (any spelling) always wins;
the resolved port shows in the TUI events panel (`🔌 Showdown server :8001`). This default lives
**only** here — `train_rl_agent.py` run directly still defaults to 8000. See the root
`CLAUDE.md` → Showdown Server, and the port-threading detail in
`src/agents/training/CLAUDE.md`.
