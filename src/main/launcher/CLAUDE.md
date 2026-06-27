# CLAUDE.md — Training Launcher (`src/main/launcher/`)

The launcher wraps `train_rl_agent.py` for long, unattended runs. **Invocation commands
(start fresh / resume) live in the root `CLAUDE.md` → Launcher section** — this file documents
how it works internally. The UI is **Textual**, built on the shared `src/main/tui/` base.
Modules: framework-agnostic core `checkpoint.py`, `worktree.py`, `child.py`, `input.py`,
`state.py`, `ipc.py` + pure formatters `format.py`; the UI `app.py` + `launcher.tcss`; the run
loop / supervisor `run.py`; entry points `__init__.main()` (`python -m main.launcher`) and the
`tui.py` back-compat alias (`python -m main.launcher.tui`).

> **History:** the launcher used to have a second **Rich** frontend (`ui.py` + a Rich `run()`
> loop). It was removed once the Textual UI proved out — Textual is now the only UI. If you're
> reading old commits/docs that mention "two frontends", that's why.

## How the UI reconciles with Textual's event loop

`run()` sets up the session (worktree pin, run dir, at-exit handlers) on the main thread, then
drives a `LauncherApp` whose `@work(thread=True)` worker runs the supervisor loop **beside** the
render loop. `LauncherState` (a lock-protected snapshot) is the bridge.

- `_prepare_session()` (worktree pin + run-dir + initial events + at-exit handlers) runs on the
  main thread **before** the screen opens — a pin failure `sys.exit`s with a clean message.
  **Run-dir resolution** (`checkpoint.resolve_launch_run_dir`, three cases): a **fresh** run (no
  `--model`) honours `--run-dir` verbatim, then `--run-name <name>` (→ `models/<name>`,
  basename-sanitized — a memorable name without the full path), else a timestamped `models/run_<ts>`.
  A **plain resume** (`--model`, no fork signal) takes the checkpoint's own folder (continue it). A
  **fork** — a `--model` resume WITH an explicit `--run-name`, or with `--exploiter` — instead writes
  to a fresh `--run-name`/timestamped dir: the `--model` is only the INIT (an exploiter trained vs a
  frozen target, or a named experiment forked off a still-running run), so its own checkpoints must
  NOT land in the source checkpoint's dir (which may be a live run / the exploiter's target); forking
  onto an existing run (one with a `metadata.json`) is refused (`ValueError` → launcher FATAL). The
  chosen folder (the one the run writes into) shows in the TUI 🗂 badge.
- `LauncherApp` (a `Gen3App` subclass) renders from `state.snapshot()` on a `set_interval(0.5)`
  timer. Input is split by latency sensitivity: **view navigation** (`l`/`e`/`d`, the `q` confirm
  overlay, `n`/`y`, ctrl-c) is handled **app-locally** via the `view_mode` reactive — switching is
  instant. **Child-control** keys (`r`/`c`/`p`/`s`, plus the confirmed force-eval `f`) and the
  confirmed-quit sentinel `"__quit__"` go to the supervisor's `cmd_q`, where `_supervise` handles
  them via `input._dispatch_command` (latency there is irrelevant — they aren't view changes).
- **Force eval (`f`):** like `q`, the keypress is app-local — it opens a `confirm_force_eval`
  overlay rather than acting immediately. `y` then routes the `f` control char to `cmd_q` →
  `_dispatch_command` sends the child **SIGUSR2**; `train_rl_agent`'s handler flags a
  `request_forced_eval()` that the active eval callback consumes on its next `_on_step` to launch
  an off-cadence eval cycle. **The accept-vs-reject decision is the child's** (it owns the
  authoritative "eval already running" state, `_pending`): a request that lands mid-cycle is
  REJECTED and reported back to the Events panel, mirroring the normal cadence's skip-while-running
  rule. See `src/agents/training/CLAUDE.md` → Bot evaluation.
- `_supervise()` runs in a `@work(thread=True)` worker, drives `LauncherState`, and **returns**
  an exit code; it then asks the app to exit via `call_from_thread`. A render fault in the timer
  is swallowed (surfaced once as an event) so a cosmetic bug can never crash the app — which, via
  the child reap below, would otherwise kill the run.
- **Quit / Ctrl-C:** `q` (or ctrl-c) opens a confirm overlay; `y` (or a second ctrl-c) pushes
  `"__quit__"` → the supervisor SIGTERMs the child (which checkpoints on SIGTERM) and waits, then
  the app exits. The on-screen "waiting for child to save…" event covers the wait; `_reap`
  (`run()`'s `finally`) narrates it on stderr if the screen is already down.
- **SIGHUP / SIGTERM (closed terminal / external kill):** the child stays in the launcher's
  session (`child._launch_child`, no `start_new_session`), so a closed tmux/SSH terminal SIGHUPs
  the whole group — and `train_rl_agent` now handles SIGHUP itself (checkpoints, like SIGTERM),
  so it saves before exiting (see **What it provides** below). The app *also* installs asyncio
  SIGHUP+SIGTERM handlers that route to the same clean `"__quit__"` save-and-exit path, so the
  **launcher** tears down cleanly too rather than dying abruptly. Two complementary backstops →
  a closed terminal never costs a checkpoint or orphans the run.
- **No orphan child:** on any exit `run()`'s `finally` sets a `shutdown` Event and `_reap`s the
  tracked child (SIGTERM → 10s grace → SIGKILL), narrating progress on stderr.
- **Stdout discipline:** the child's stdout only reaches `state.add_log` + `launcher_child.log`
  (never the terminal), and the launcher's own stderr prints fire via `atexit` after the screen
  closes — so a stray `print()` never corrupts the Textual screen.

Tests: `src/main/launcher_app_test.py` (Pilot render/keys/view/confirm/ctrl-c/signal + a
deterministic `_supervise` exit-code/crash-restart/`_reap` suite), plus `launcher_test.py`
(checkpoint/strip/dispatch/crash-log helpers) and `launcher/state_test.py`.

## What it provides

- **Periodic restarts** — kills and relaunches the child every N hours to reclaim pymalloc
  fragmentation; the child saves a checkpoint on SIGTERM and the launcher picks it up
  automatically. The child also checkpoints on **SIGHUP** (`_setup_signal_handlers` routes it
  to the same graceful path) — the child shares the launcher's session (`child.py` spawns it
  without `start_new_session`), so closing the controlling terminal/tmux window SIGHUPs the
  whole group; without that handler the child died mid-iteration with no checkpoint. Running
  the launcher under `nohup` prevents the SIGHUP entirely; the handler is the in-code backstop.
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
  propagates the child's exit code rather than masking it. "Checkpoint" here means a *real* run
  checkpoint at `<run>/checkpoints/checkpoint_*_steps.zip` / `…/checkpoint_forced_*` (current
  layout) or `<run>/*_steps.zip` / `forced_*` (legacy, at the root): `find_latest_checkpoint`
  deliberately skips `*.zip` artifacts nested under `snapshots/` (the self-play pool — whose
  step-0 seed is written at startup, *before* any rollout), `best_model/`, and `eval_traces/` —
  but NOT `checkpoints/`, which IS resumable. The caller derives `run_dir` via
  **`run_dir_for_checkpoint`** (a plain `dirname`, then strip a trailing `checkpoints/`), so a
  checkpoint in the subdir still resolves to the run root for the `--run-dir` arg + the TUI 🗂
  badge. Counting one of the ARTIFACT dirs instead would mis-derive `run_dir` to the artifact
  subdir (`…/snapshots`, the wrong dir shown in the badge) and let a startup crash silently
  "resume" from the freshly-initialised seed instead of failing loudly. The dashboard shows a
  `↻ N restarts (M crash)` badge and the exit summary reports the crash count.
- **Non-recoverable config errors don't loop** — a checkpoint arch-family mismatch (or a resume
  `vf_coef`/reward-config drift) fails the *same* way on every retry, so auto-restarting just burns
  the circuit-breaker and hides the cause behind the logs. `train_rl_agent.py` exits these with a
  dedicated `FATAL_CONFIG` (3) code (raised for any `ModelVersionError`); `_supervise` classifies
  them via `_fatal_config_reason(rc, log_lines)` — the exit code is the primary signal, plus a
  defensive scan for a `[ModelVersion] FATAL` line in the captured output (catches a FATAL that
  escaped as a generic exit 1). On a match it saves the crash log, prints the FATAL reason straight
  into the **Events panel** (`🛑 Fatal config error — will NOT restart`, then the reason lines), and
  returns immediately — no restart, no checkpoint discovery — so the fix is on-screen, not buried in
  `crashes/restart_err_*.txt`.
- **Worktree isolation** — at startup, creates a detached git worktree pinned to the current
  HEAD (or to the commit recorded in the checkpoint's `metadata.json` when resuming). Agent
  pushes to `main` never affect a running session.
- **Textual TUI** — live dashboard showing metrics, FPS, restart countdown; `l` logs · `e`
  events · `d` dashboard · `r` restart · `c` forced checkpoint · `p` plots · `s` status ·
  `f` force eval → confirm → `y`/`n` (off-cadence eval cycle; child rejects if one is already
  running) · `q`/ctrl-c → confirm → `y`/`n` quit · `v` copy mode (inherited from `Gen3App`) freezes the
  2 Hz refresh + hands the mouse back to the terminal for native select-and-copy, same key
  resumes — the **portable** copy path (works on Terminal.app); `super+c` (⌘C) also copies the
  Textual selection on terminals that forward ⌘C + honour OSC 52. See `src/main/tui/CLAUDE.md`
  → Copying text. Built on the shared `src/main/tui/` base — see **How the
  UI reconciles with Textual's event loop** above. **Skill rating (ELO)** surfaces as a
  badge-row headline `🏅 ELO 1532 ±40` (`app.py::_elo_badge`, cyan) AND inside the eval panel: the
  table has a dedicated **`elo` column** — the model's own rating (±CI) on the `all` row, and each
  opponent's anchored ELO on its row (bots = their fixed anchor; sentinels = their rating this
  cycle, from `eval/elo_vs_<opp>` recorded by `eval_callback._record_opponent_elos`). This is the
  at-a-glance "is it going well?" number during self-play pool play — anchored Bradley-Terry over
  the fixed bots, so it rises with strength even while `win_rate_vs_pool` sits pinned near 50% (see
  `src/agents/training/CLAUDE.md` → ELO / skill rating). *(The per-sentinel ELO is a noisy
  single-cycle estimate — `python -m main.elo … --source tb` is the well-anchored canonical fit.)*
  **Metrics layout** — the dashboard's metrics row is **three side-by-side tables** so a metric-rich
  run stays readable instead of one over-long column: a **left misc column** (rollout / time, then the
  `grad/*` / `popart/*` diagnostics), a **dedicated `train/*` column** (by far the
  biggest section — all the PPO losses, `return_*`, `value_pred_std`, `grad_norm`, the opponent-mix
  `*_fraction` telemetry, then the `belief/*` aux diagnostics rendered directly **below** train when a
  belief aux is on), and the **eval column**. Non-eval metrics are split across the first two
  **by whole top-level section — a section is never split across columns**
  (`app.py::_fill_metric_sections`); the two narrow metric columns hug their content (`width: auto`)
  so the wider eval column (`width: 1fr`) gets the horizontal slack.
  **Gradient-balance + value-scale diagnostics** (always on) ride that layout: the `grad/*` block
  (`policy share` + `value share` — the two RL heads' slices of ONE common-denominator pie; `aux share (all)`
  = the total non-RL draw; `log val/pol grad` = the aux-independent non-saturating `log10(‖g_v‖/‖g_p‖)`
  ratio; `policy-value cos`, policy/value grad-norms; plus, when an aux is on, its OWN share broken out —
  `species blf` / `move blf` / `latent` / `move-lat` / `winprob` / `valdist` — so any single scaffold
  crowding out the rest is visible) sits in the left column, while
  `train/return_*`, `train/value_pred_std`, and `train/grad_norm` join the train column — together the
  direct shared-trunk pressure gauge for tuning `vf_coef` / preparing PopArt (computed
  in `agents/training/grad_balance.py`; see `src/agents/training/CLAUDE.md`). They need no new launcher
  wiring: they ride the same generic `MetricsExporterCallback` scalar path and auto-route by their
  `grad/` / `train/` section prefix; only their display order + short labels are declared in
  `format.py`. Under **`--use-popart`** a `popart/*` block also appears (`value mu`, `value sigma`,
  `value head |W|` — the value-target normalizer state; same generic path), and `grad/value_policy_logratio`
  should be seen falling toward ~0.
- **Crash reporting** — child stdout/stderr is streamed live to `<run_dir>/launcher_child.log`
  (complete even if the child hard-`os._exit`s, bypassing Python cleanup) and held in a
  5000-line in-memory scrollback. The on-disk log is a **disk ring buffer**
  (`child._CappedChildLog`, `_CHILD_LOG_MAX_BYTES` ≈ 1 MiB): it streams every line
  line-buffered, but once the file passes the cap it's rewritten keeping only the recent
  tail, and a pre-existing oversized file (e.g. a legacy multi-GB log) is trimmed on open —
  so a long multi-restart run can't grow it without bound. On a non-zero exit the last 100
  lines are dumped to the terminal after the TUI closes; on *every* exit (crash, complete,
  quit) the full log path is printed and the file is finalized (the in-memory buffer is
  flushed to it as a fallback if streaming never started).

## Exit codes (`src/main/exit_codes.py`)

| Code | `TrainExitCode` | Meaning |
|------|----------------|---------|
| 0 | `COMPLETE` | All steps done — launcher stops |
| 15 | `INTERRUPTED` | SIGTERM received, checkpoint saved — launcher restarts |
| 1 | `CRASH` | Unhandled exception — launcher saves `crashes/restart_err_<token>.txt` and auto-restarts from the last checkpoint (up to `--max-crash-restarts` consecutive rapid crashes, then gives up; any non-enum exit code is treated the same way). A crash with no checkpoint to resume from is fatal: the child's exit code is propagated and the crash log printed. |
| 3 | `FATAL_CONFIG` | **Non-recoverable** config/architecture error — `train_rl_agent.py` raises it for a `ModelVersionError` (checkpoint arch-family mismatch, or a resume `vf_coef`/reward-config drift). Restarting would hit the *identical* error every time, so the launcher does **not** restart: it saves the crash log, surfaces the reason on-screen, and gives up immediately (returning this code) instead of looping until the crash circuit-breaker trips. See **Crash auto-restart**. |

## Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--restart-interval-hours` | `3.0` | Set to `0` for a single run with no restart |
| `--max-crash-restarts` | `3` | Consecutive rapid self-crashes (each < 10 min after launch) to auto-restart through before giving up. `0` = unlimited. A crash after sustained progress resets the counter (see **Crash auto-restart**). |
| `--restart-grace-minutes` | `20.0` | Force-kill window after a scheduled restart's deadline (child overran its rollout boundary). A child that ignores the SIGTERM is SIGKILL'd after a 90 s grace, and a launcher-forced kill restarts from the last checkpoint (not treated as a fatal crash). The dashboard shows `⚠ no child output for Nm` once the child has been silent > 2 min, so a stall is *visible* — but there is no auto-restart on stall. A connection failure crashes loudly rather than hanging: a connect-time failure via the `Gen3Player` connect guard, and a **mid-battle** websocket drop via the `_AsyncQueue` disconnect guard (`PSClient.listen()` sets `_disconnected` on *any* close it did not initiate — an abnormal drop OR a clean peer/server-initiated close such as the server stopping; a close the client requests via `stop_listening` sets `_closing` and exits clean, since terminating the connection on purpose is not an error. The env's blocking `step`/`reset` get races against `_disconnected` and raises `ShowdownException` instead of waiting forever for a message the dead listen task can't deliver). Both deterministic — no timeout guess. A **silent** stall (the server sends no next message *and* does not close) can't be caught deterministically, so `_AsyncQueue.race_get` bounds it with a generous watchdog (`_RACE_GET_TIMEOUT_S`=120 s, ~100× a normal step; override `GEN3_RACE_GET_TIMEOUT_S`) that **crashes** (raises `ShowdownException` → worker dies → restart from checkpoint) rather than recovers — recovering would feed PPO a fabricated transition. With `GEN3_RACE_TRACE=1` the wedged battle's cross-thread interleaving is dumped into the crash log. The one known cause — `race_get` stranding a queued force-switch on a stale `_trying_again` event (an upstream poke-env bug) — is now fixed (see `src/agents/training/CLAUDE.md`), so this watchdog is a should-never-fire backstop. |
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
**only** here — `train_rl_agent.py` run directly still defaults to 8000.

**Bridge mode is port-free.** When `--use-showdown-bridge` is in the child args,
`_apply_default_showdown_port` injects **no** default port and the events panel shows
`🌉 Transport: in-process bridge (no Showdown server)` instead of a port — the bridge connects to
no server at all (training AND eval run in-process), so any `--showdown-port` passed alongside it
is inert (built into `server_config` but never connected to, so it can't even disturb the live
:8001 server). Guarded by `default_port_test.py::test_bridge_mode_*`. See the root
`CLAUDE.md` → Showdown Server, and the port-threading detail in
`src/agents/training/CLAUDE.md`.
