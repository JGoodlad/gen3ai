# Implementation: Launcher Infrastructure Rewrite

The launcher (`src/main/launcher/`) was refactored from a two-file monolithic script
into a modular 8-module package, then progressively extended with a Rich TUI, a
process-restart loop, git worktree isolation, child→launcher IPC for live metrics,
and LR persistence across restarts.

Primary themes: reliable long-run process management, code isolation between training
sessions, live observability without polling, and clean separation of concerns across
modules.

---

## Background

Before this work the launcher was two files:

- `launcher.py` (~613 LOC) — restart loop, subprocess management, terminal UI, argument
  parsing, worktree creation, checkpoint discovery, and keyboard handling all in one
- `launcher_ui.py` — Rich rendering, but tightly coupled to launcher state

After: a package with 8 focused modules totalling the same functionality but with
clear ownership, testable boundaries, and significantly more capability.

---

## Package Structure

```
src/main/launcher/
  __init__.py      — public API re-export + argparse entry point
  run.py           — main restart loop
  child.py         — subprocess spawn + IPC reader threads
  ipc.py           — child→launcher pipe writer (used by train_rl_agent.py)
  state.py         — immutable snapshot + mutable thread-safe launcher state
  ui.py            — pure Rich rendering
  input.py         — raw terminal keyboard handling
  checkpoint.py    — checkpoint discovery + CLI argument patching
  worktree.py      — git worktree creation, pruning, LR/hash reading
```

### `__init__.py`

Public API re-export and argparse entry point. Defines launcher-specific flags; all
remaining args are forwarded verbatim to `train_rl_agent.py`:

| Flag | Default | Effect |
|---|---|---|
| `--restart-interval-hours` | `3.0` | Hours between forced restarts; `0` = single run |
| `--no-pin` | off | Skip worktree creation; run from current source tree |
| `--sync-to-main` | off | On resume, pin worktree to current HEAD instead of checkpoint's git hash |

### `run.py`

Main restart loop. Execution flow:

1. **Setup phase**: prune stale worktrees, create isolated worktree, determine
   `run_dir`, start keyboard input thread
2. **Outer restart loop**: spawn child (via `child.py`), enter inner poll loop
3. **Inner poll loop** (2 Hz): render TUI, check restart deadline, dispatch keyboard
   commands, wait on child with 500 ms timeout
4. **Exit check**: on child exit, inspect code:
   - `COMPLETE` (0) → stop launcher
   - `INTERRUPTED` (15) → find latest checkpoint, patch CLI args, restart
   - Any other code → crash path, dump child output, stop

### `child.py`

Subprocess spawn and IPC plumbing.

- `_build_child_env()`: copies current env, sets `PYTHONUNBUFFERED=1`
- `_launch_child()`: spawns `train_rl_agent.py` with `LAUNCHER_METRICS_FD` set to the
  write end of a pipe; captures stdout/stderr; starts two reader threads
- `_read_metrics_pipe()`: reads JSON-line stream from the pipe; routes payloads:
  - `{"_event": "..."}` → `state.add_event()`
  - flat metric dict → `state.update_metrics()`
- `_read_child_stdout()`: captures child stdout lines to `state.add_log()`

### `ipc.py`

Thread-safe pipe writer for child→launcher communication. Used by `train_rl_agent.py`
and training callbacks; safe to import in any context.

Key behaviors:
- `init()`: eagerly opens the write end of the pipe; no-op if `LAUNCHER_METRICS_FD`
  is absent or negative (i.e., when running outside the launcher)
- `send_event(msg)`: posts `{"_event": msg, "ts": ...}` as a JSON line
- `send_metrics(payload)`: sends a flat dict of metric names as a JSON line
- Global lock protects all file operations against concurrent writes from callback
  threads
- `close()`: flushes and closes the pipe
- `_reset_for_testing()`: resets module state for test isolation

### `state.py`

Two classes with a clean mutable/immutable split:

**`LauncherSnapshot`** — immutable dataclass, one per TUI render:

| Field | Type | Notes |
|---|---|---|
| `pid` | int | Child PID |
| `run_start` | datetime | When this restart began |
| `deadline` | datetime | Scheduled restart time |
| `restart_count` | int | Number of restarts so far |
| `interval_hours` | float | Configured restart interval |
| `view_mode` | str | `"dashboard"`, `"logs"`, or `"confirm_quit"` |
| `metrics` | dict | Latest metric values from IPC |
| `metrics_step` | int | Training step of last metrics update |
| `metrics_ts` | float | Timestamp of last metrics update |
| `log_lines` | deque | Last 500 lines of child stdout (max) |
| `events` | list | Last 30 events from IPC (max) |
| `initial_git_hash` | str | Short hash of the worktree commit |
| `run_dir` | str | Path to the active run folder |

**`LauncherState`** — mutable, thread-safe:

- `add_log(line)`, `add_event(msg)`, `update_metrics(payload)` — called from reader
  threads
- `snapshot()` — atomic read under `threading.Lock`; returns a `LauncherSnapshot`

### `ui.py`

Pure Rich rendering; no subprocess or I/O. `LauncherUI.render()` dispatches based on
`view_mode`:

- **Dashboard** (default view):
  - Row 1: PID, restart count, elapsed time, restart countdown
  - Row 2: git hash badge, model folder badge, highlights (steps / FPS / reward)
  - Metrics table: two-column layout — left column for rollout/eval metrics, right
    column for train/time metrics
  - Recent output: last 6 lines of child stdout
  - Events panel: last 5 events, timestamped
  - Footer: keybinding reference
- **Logs view** (`l` key): full scrollable child output, height dynamic to terminal
- **Confirm-quit** (`q` key, then confirm): simple confirmation prompt

`_fmt_val()` formats metric values to 4 significant figures.

### `input.py`

Raw terminal keyboard handling via a state machine:

- `_setup_raw_input()`: switches stdin to cbreak mode via `termios`
- `_restore_tty()`: registered with `atexit`; restores terminal on exit
- `_read_keys()`: runs in a thread, reads single chars
- `_dispatch_command()` — key mappings:

| Key | Action |
|---|---|
| `r` | Send SIGTERM to child (triggers checkpoint + restart) |
| `c` | Send SIGUSR1 to child (forced checkpoint without restart) |
| `q` | Enter confirm-quit view (dashboard only) |
| `l` | Switch to logs view |
| `d` | Switch back to dashboard |
| `s` | Send status/stats event via IPC |

`_PollFlags` is a small state machine tracking pending quit/restart/sigterm actions
between key read and main loop dispatch.

### `checkpoint.py`

Checkpoint discovery and CLI argument patching for restarts:

- `find_latest_checkpoint(run_dir)`: finds the most recent `.zip` in `run_dir`
- `_find_model_arg()`: locates `--model` in the current argv
- `_insert_or_replace_model_arg()`: patches `--model <path>` in args list
- `_insert_or_replace_run_dir_arg()`: patches `--run-dir <path>` in args list
- `_strip_launcher_args()`: removes launcher-specific flags before forwarding to
  `train_rl_agent.py`

### `worktree.py`

Git worktree lifecycle and checkpoint metadata reading:

- `_git_hash()`: reads current HEAD hash
- `_read_checkpoint_git_hash()`: reads `git_hash` from `metadata.json` alongside a
  checkpoint
- `_read_checkpoint_lr()`: reads `current_lr` from `metadata.json` for LR persistence
- `_prune_stale_launcher_worktrees()`: removes worktree directories left by crashed
  sessions
- `_create_run_worktree(hash)`: runs `git worktree add --detach <path> <hash>`;
  symlinks `deps/pokemon-showdown/dist` and `deps/pokemon-showdown/node_modules` from
  the main repo; returns `(train_script_path, src_dir_path, cleanup_fn)`

---

## Exit Codes (`src/main/exit_codes.py`)

New module. `train_rl_agent.py` exits with one of these; `run.py` reads the code to
decide restart vs. stop:

| Code | `TrainExitCode` | Launcher action |
|---|---|---|
| 0 | `COMPLETE` | All steps done — stop |
| 15 | `INTERRUPTED` | SIGTERM caught, checkpoint saved — find checkpoint, restart |
| 1 | `CRASH` | Unhandled exception — dump child output, stop |

Using a named exit code (15 = `SIGTERM`'s signal number) rather than a custom value
makes the intent legible and avoids collisions with OS conventions.

---

## Key Feature: IPC (Child→Launcher Metrics Pipe)

The IPC design avoids polling the child for metrics: the child pushes data whenever
it has something to report.

**Mechanism:**

1. `run.py` creates a pipe pair `(read_fd, write_fd)` before spawning the child
2. `write_fd` is passed via `LAUNCHER_METRICS_FD` environment variable
3. `ipc.init()` in the child process opens `write_fd` as a file object
4. `child._read_metrics_pipe()` in the parent reads from `read_fd` in a dedicated
   thread at whatever rate the child writes

**Payload format** (JSON lines, one per flush):

```json
{"_event": "▶️  Resuming at LR 3.00e-04"}
{"rollout/ep_rew_mean": 0.42, "rollout/ep_len_mean": 187.3, ...}
```

**Routing** in `_read_metrics_pipe()`:
- Lines with `"_event"` key → `state.add_event()`
- All other lines → `state.update_metrics()` (merged into the running metrics dict)

The global lock in `ipc.py` protects concurrent writes from multiple callback threads
(metrics exporter and signal handler run on different threads).

`ipc` is safe to import in `train_rl_agent.py` regardless of whether the launcher is
present — all calls are no-ops when `LAUNCHER_METRICS_FD` is absent or invalid.

---

## Key Feature: Worktree Isolation

Each launcher session runs the training child in a detached git worktree pinned to a
specific commit. This means pushes to `main` during a training run never affect the
running session.

**On fresh start:**
```
run.py → _create_run_worktree(current HEAD hash)
       → git worktree add --detach /tmp/launcher-<hash> <hash>
       → symlink dist/ and node_modules/ from main repo
       → child PYTHONPATH = worktree/src/
```

**On resume from checkpoint:**
1. `_read_checkpoint_git_hash()` reads `git_hash` from `metadata.json`
2. Unless `--sync-to-main` is set, the worktree is pinned to that hash
3. With `--sync-to-main`, the worktree is pinned to current HEAD instead

`--sync-to-main` is the escape hatch for picking up UI or tooling fixes from `main`
without discarding the checkpoint.

**Stale worktree pruning:**
`_prune_stale_launcher_worktrees()` runs at startup, removing `git worktree`
directories from crashed sessions. Without this, repeated crashes accumulate stale
worktrees that `git worktree list` and VS Code's git integration must process on every
operation.

The pokemon-showdown submodule is symlinked (not re-initialized) in each worktree
because `dist/` and `node_modules/` are already built in the main repo. Full
re-initialization would require re-running `npm install` and the Showdown build for
every restart.

---

## Key Feature: LR Persistence Across Restarts

PPO's learning rate is managed by `AdaptiveLRCallback` and drifts over the course of
a run. Without persistence, each restart resets LR to the `--lr` CLI argument,
discarding schedule progress.

**On SIGTERM (in `train_rl_agent.py`):**

1. Signal handler reads current LR from `AdaptiveLRCallback`
2. Calls `save_model_snapshot(..., current_lr=lr)`, which writes `current_lr` to
   `metadata.json` alongside the checkpoint `.zip`
3. Exits with `TrainExitCode.INTERRUPTED` (15)

**On resume (in `worktree.py` / `run.py`):**

1. `_read_checkpoint_lr()` reads `current_lr` from `metadata.json`
2. Value is injected into the child's `AdaptiveLRCallback` at init, so LR continues
   from where it left off
3. An IPC event is sent: `"▶️  Resuming at LR {lr:.2e}"` — visible in the dashboard
   events panel

**Fallback:** if `metadata.json` is absent or `current_lr` is missing (checkpoint
saved before this feature), the child falls back to the `--lr` CLI argument silently.

---

## Run Directory Continuity

A single `models/run_<timestamp>/` directory is used for the entire launcher session,
including all restarts. Checkpoints from all restarts accumulate in the same folder.

**On fresh start:** `run_dir = models/run_<new timestamp>/`

**On resume from checkpoint:** `run_dir = os.path.dirname(os.path.abspath(checkpoint))`

The run folder name (not full path) is displayed in the TUI dashboard as a badge.
This allows the user to match what they see on screen to the directory on disk.

---

## Integration Points

### `src/main/train_rl_agent.py`

- Imports `ipc.send_event` for key lifecycle messages
- On SIGTERM: writes `current_lr` to `metadata.json`, exits `INTERRUPTED`
- `MetricsExporterCallback` delegates metric pushes to `ipc.send_metrics()`

### `src/agents/training/metrics_exporter_callback.py`

- Calls `ipc.send_metrics(payload)` each training step
- Payload is a flat dict of SB3 metric names and float values

### `src/agents/model/snapshot.py`

- `save_model_snapshot()` accepts optional `current_lr` parameter
- Writes `current_lr` into `metadata.json` when provided

### `src/utils/git.py` (new)

- `get_git_hash(short=False)` — reads current HEAD hash via `git rev-parse`
- `get_repo_root()` — returns the repo root path via `git rev-parse --show-toplevel`
- Used by `worktree.py` (hash for worktree creation) and `snapshot.py` (hash written
  to `metadata.json`)

---

## Test Coverage

| File | What it validates |
|---|---|
| `src/main/launcher_test.py` | Integration tests: all 3 `TrainExitCode` exit paths (COMPLETE, INTERRUPTED, CRASH), restart loop behavior, checkpoint discovery |
| `src/main/launcher_ui_test.py` | UI rendering tests: dashboard layout, metrics table, events panel, log view, confirm-quit view |
| `src/main/launcher/ipc_test.py` | Comprehensive IPC pipe tests: no-op when `LAUNCHER_METRICS_FD` absent or FD negative, event and metrics JSON serialization, thread safety under concurrent writes, pipe closure and cleanup |

The IPC tests use `_reset_for_testing()` to isolate module state between test cases,
since `ipc.py` is a module-level singleton (one pipe per process).

---

## Files Changed

| Path | Change |
|---|---|
| `src/main/launcher/` | Replaced 2-file monolith with 8-module package |
| `src/main/exit_codes.py` | New: `TrainExitCode` enum |
| `src/main/train_rl_agent.py` | IPC integration; LR persistence on SIGTERM |
| `src/agents/training/metrics_exporter_callback.py` | Delegates to `ipc.send_metrics()` |
| `src/agents/model/snapshot.py` | `current_lr` written to `metadata.json` |
| `src/utils/git.py` | New: `get_git_hash()`, `get_repo_root()` |
| `src/main/launcher_test.py` | Integration tests |
| `src/main/launcher_ui_test.py` | UI rendering tests |
| `src/main/launcher/ipc_test.py` | IPC pipe tests |
