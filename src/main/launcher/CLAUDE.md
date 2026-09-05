# CLAUDE.md — Training Launcher (`src/main/launcher/`)

The launcher wraps `train_rl_agent.py` for long, unattended runs. **Invocation commands
(start fresh / resume) live in the root `CLAUDE.md` → Launcher section** — this file documents
how it works internally. The UI is **Textual**, built on the shared `src/main/tui/` base.
Modules: framework-agnostic core `checkpoint.py`, `worktree.py`, `child.py`, `input.py`,
`state.py`, `ipc.py` + pure formatters `format.py`; `pinned_argv.py` + `pinned_argv_probe.py`
(validate the child argv against the PINNED commit's parser, not this tree's); the UI `app.py` +
`launcher.tcss`; the run loop / supervisor `run.py`; the resolve-and-print `dry_run.py`; entry
points `__init__.main()` (`python -m main.launcher`) and the `tui.py` back-compat alias
(`python -m main.launcher.tui`).

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
  NOT land in the source checkpoint's dir (which may be a live run / the exploiter's target). The
  chosen folder (the one the run writes into) shows in the TUI 🗂 badge.
  **A fork is IDEMPOTENT ("copy once from the source, resume in place after").** The FIRST launch
  copies the source `--model` into the new dir; a *re-launch* of the same fork command (launcher
  process death → reboot / re-running the launch script) detects that the fork dir already holds its
  OWN resumable checkpoint and RESUMES it from that (`checkpoint.resolve_fork_resume_model`, swapped
  in `run._prepare_session`) instead of re-copying the source (which would silently discard the
  fork's progress). So a fork command is safe to re-run unattended. The clobber guard now FATALs only
  when the fork target exists but has **no** resumable checkpoint — a genuine run-name collision or a
  fork that crashed before its first save. (The launcher's OWN 6h/crash restart loop was already
  idempotent — it finds the run dir's latest checkpoint and replaces `--model`; this extends the same
  guarantee to a full launcher-process restart.) Tests: `launcher_test.py::TestResolveLaunchRunDir`
  (`test_idempotent_fork_with_checkpoint_resumes_not_raises` / `test_fork_first_launch_keeps_source_model`
  / `test_fork_onto_existing_run_without_checkpoint_raises`).
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
- **🚨 Headless when stdin is not a TTY (`run.headless_mode()`).** A detached launch
  (`nohup … < /dev/null &`, systemd, cron, an agent's background shell) leaves stdin on
  /dev/null, and Textual's input thread then **busy-loops a whole core forever**: an fd at EOF
  is *permanently* readable, so `selector.select(0.1)` returns instantly, `os.read` yields
  `b""`, and the `if not unicode_data: break` inside `linux_driver.run_input_thread` breaks
  only the inner `for` — the outer `while` spins at full speed. Measured on a live 15 h run
  (2026-08-14): **96% of a core** (83% user), **13 h 34 m** of CPU burned by one thread, plus a
  **982 MB** launcher log of full-screen ANSI repaints growing at **17 KB/s**, because the
  "screen" was a redirected file. A standalone A/B of a two-line Textual app isolated the cause
  to stdin alone — /dev/null **98%** of a core, a real pty **0%**, headless **0%** and 0 bytes
  of stdout.
  So `run()` passes `app.run(headless=headless_mode())`: `HeadlessDriver` starts no input
  thread and writes nothing. **Nothing is lost** — with stdin on /dev/null there is no keyboard
  to serve, and the repaints were going somewhere no one could read as a screen; the supervisor
  worker, restarts, events, metrics and checkpointing are all driver-independent. A TTY on
  stdin keeps the full interactive TUI (the normal foreground case).
  Because headless has no screen, `LauncherState.event_sink` echoes every event as a plain
  `[HH:MM:SS] …` line so a detached run stays followable by `tail -f` — wired **before**
  `_prepare_session` so the setup events (worktree pin, run dir, transport) are captured too,
  and bound to the **real** `sys.stdout` captured before `app.run()`, since Textual replaces
  `sys.stdout` for the duration of the run and a plain `print()` from the supervisor thread
  would be swallowed by its capture. The sink is called *outside* the state lock (a slow write
  must never stall a reader thread) and its exceptions are swallowed.
  Measured after: **1.8% of a core**, and a **1.5 KB** log with **0** escape sequences for a
  full run. Gate: `headless_test.py` — including an end-to-end CPU assertion that fails at
  96.7% if the fix is reverted (bound: <30%).
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
- **Worktree isolation** — at startup, creates a detached git worktree pinned to the commit
  `worktree.resolve_pin` chooses (`--pin-commit` > the checkpoint's recorded `metadata.json`
  hash on a resume > HEAD). Agent pushes to `main` never affect a running session.

  🚨 **THE STARTUP PRUNE REMOVES ONLY A DEAD LAUNCHER'S WORKTREE, and that is a 2026-09-05
  fix paid for with a run.** `_prune_stale_launcher_worktrees` used to force-remove EVERY
  `launcher-*` worktree it found, on the assumption that such a directory can only be a
  crashed session's debris. It is not: it is also every LIVE run's isolated checkout. A
  one-second validation command — `python -m main.launcher --pin-commit deadbeef --steps 1`,
  which exits `FATAL_CONFIG` before creating a worktree of its own — deleted a live
  production run's. The run kept going on its already-open file descriptors and looked
  healthy for hours; it died at its next 3 h periodic restart, when the launcher re-exec'd
  the child out of a directory that no longer existed (**exit 2, no `final_model`**). The
  resume then surfaced a second defect (see *Which commit a checkpoint records*).

  **The ownership record.** `_create_run_worktree` now writes a claim naming this process:
  `<worktree>.owner.json` — pid, its `/proc/<pid>/stat` **start time** (the pid-reuse guard;
  a pid alone is not an identity, `(pid, starttime)` is), the run dir, and `sys.argv`. It
  lives **BESIDE** the worktree, not inside it, so it cannot appear in that worktree's
  `git status`: the pinned tree is a real checkout of a real commit, and an ignore rule added
  today does not exist in a checkout of a commit from last month, while the child itself runs
  `git` in there. Being a `tempfile.mkdtemp` sibling, it is collected by the same /tmp
  cleanup, and `cleanup()` removes both.

  **The prune rule**, per `launcher-*` worktree — and every ambiguity resolves to KEEP,
  because a stale directory in /tmp costs disk and a deleted live one costs a run:

  | state | verdict |
  |---|---|
  | owner file present, pid alive (`os.kill(pid, 0)`) **and** its starttime matches | **KEEP** — a live run |
  | owner alive but starttime unrecorded / `/proc` unreadable | **KEEP** — cannot verify reuse |
  | owner pid gone, or alive with a DIFFERENT starttime (pid reuse) | remove |
  | no owner file (pre-fix worktree), mtime < 24 h | **KEEP** — may be a live pre-fix run |
  | no owner file, mtime > 24 h (`_LEGACY_ORPHAN_MAX_AGE_S`) | remove — abandoned |
  | the directory no longer exists | remove — the case git's own prune handles |

  It **reports every decision** through a `report` callable (`state.add_event` from the
  launcher, `print` standalone), naming the owning pid on each skip — a startup that leaves
  debris behind must say so rather than look like a no-op. Gate:
  `worktree_prune_test.py`, over a real temp git repo: a current-pid worktree survives, a
  dead-pid one is removed, a reused pid is not mistaken for the owner, an unverifiable live
  owner is kept, legacy fresh/old split correctly, a non-`launcher-*` worktree is untouched,
  and the claim provably does not change the worktree's `git status`.
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
| `--nice` | `10` | Scheduling niceness for the launcher **and everything it spawns** — `0` disables. Applied in `run.main()` before any child exists (`run._apply_nice`, default `run.DEFAULT_NICE`); niceness is inherited across fork/exec, so the training child, its SubprocVecEnv workers and every eval worker are covered without per-spawn wiring — including the processes created by later periodic and crash restarts. It only ever **raises** niceness: a negative target needs `CAP_SYS_NICE`, so it no-ops rather than failing. **On an idle box this changes nothing** — niceness only arbitrates under contention. Why it defaults on: a run holds ~940 processes, and at nice 0 it competes on equal terms with interactive work sharing the box (measured 2026-08-13 at load 17–25 on 16 cores: an interactive client in the *same cgroup* as the run waited 2.1 s in the run queue per 1 s of CPU it received, and every training process sat at nice 0). Note the limit of the mechanism: nice arbitrates **within** a cgroup, so a client in its own systemd scope is already protected by cgroup `cpu.weight` and gains little — the flag's value is for whatever shares the run's own scope. Gate: `nice_test.py` (including the inheritance test — without it the workers silently revert to nice 0 and nothing else would notice). |
| `--no-pin` | off | Skip worktree creation; run from the current source tree |
| `--sync-to-main` | off | When resuming from a checkpoint, pin the isolated worktree to the current HEAD instead of the checkpoint's original git hash. Use this to pick up UI or tooling fixes on `main` without discarding the checkpoint. |
| `--pin-commit COMMIT` | unset | **Pin the isolated worktree to a NAMED commit** (full sha or unambiguous prefix — resolved with `git rev-parse --verify <spec>^{commit}` and announced at startup as the full sha plus its subject line). Spelled `--pin-to-hash` before 2026-09-05; both spellings still parse, `--pin-commit` is the name. Beats the checkpoint's recorded `git_hash` on a genuine FORK and HEAD on a fresh run; **refused** beside `--sync-to-main` (argparse — they name two different sources of truth) and beside `--no-pin`; **refused** on a same-run RESTART whose checkpoint records a different hash (see the resume contract). An unresolvable commit exits `FATAL_CONFIG` naming it — never a silent fall-back to HEAD, which is the whole failure it exists to prevent. |

| `--dry-run` | off | **Resolve this launch and PRINT it, then exit — creating nothing.** Role (FRESH / FORK of <parent> / RESTART of <run>), the run dir the argv would write into, the pin (sha + subject + source), `--steps` beside the checkpoint's recorded `num_timesteps` so `+X steps` is visible, the effective config a `--model` inherits (per-flag `INHERITED` vs `from the argv`), the pool as recorded, and a `(child-only: …)` line for everything that needs torch. Exits `0`, or `FATAL_CONFIG` (3) on any refusal the real path makes. See **Validating a launch without launching** below. |

All other flags are forwarded verbatim to `train_rl_agent.py` (the launcher strips only
launcher-owned flags).

**The launcher owns NO compile default.** `--compile-opponents` / `--compile-opponents-preload` /
`--compile-trainer` all default ON in `train_rl_agent`'s own parser (2026-08-17), and the launcher's
only job is to be transparent to them and to their `--no-` opt-outs. Two ways it could stop being:
`_strip_launcher_args` could grow an entry that eats one, or argparse could abbreviation-match an
unknown token against a launcher flag (it parses with `parse_known_args`, and `--no-pin` lives right
next to `--no-compile-*`). `compile_flag_forwarding_test.py` pins both against the REAL parser —
`build_launcher_parser()` was extracted from `main()` for exactly that, so the test interrogates the
parser rather than a hand-copied twin. Same failure class as `default_port_test.py`, mirrored: that
one catches a launcher-injected default drifting from the trainer's, this one catches the launcher
silently swallowing a child flag.

## Validating a launch without launching — `--dry-run`

🚨 **A "dry launch" of the real command is safe on a FORK and DESTRUCTIVE on a same-run RESTART.
That asymmetry cost a run's provenance on 2026-09-05.** To check that a restart with a larger
`--steps` would still launch, a session launched the real command and killed it a few seconds
later, after the startup lines. A fork writes a NEW directory, so that habit had always been
harmless; a RESTART operates on the REAL run directory, and those seconds were enough to write
`final_model_interrupted.zip`/`.json`, repoint `latest.txt` at that phantom artifact, overwrite
`metadata.json` (whose `steps` became a target that never ran) and `model_config.json`, and leave
`.compile_quorum` files behind. **"Dry" was a property of forks, never of the launcher.**

`--dry-run` makes it a property of the launcher. It performs *everything the launcher resolves
before a child exists* — argv parse, the fork-vs-restart classification (`fork_lr.
is_same_run_checkpoint`, IMPORTED), the idempotent-fork `--model` swap, the run dir
(`resolve_launch_run_dir`, **without** the `makedirs` that follows it in `_prepare_session`), the
pin (`resolve_pin`), and the effective config a `--model` inherits — prints one startup-shaped
block, and exits.

**What it prints**, in order: role · run dir (flagged `EXISTS — a real launch WRITES INTO IT` when
it does) · `--model` · pin sha + subject + source · `--steps` beside the checkpoint's recorded
`num_timesteps` and the `+X steps` delta · interpreter · transport · restart/grace/nice · the
effective config with each reported flag marked `INHERITED` or `from the argv` (`distill_teacher`,
`distill_target`, `distill_coef`, `distill_topk`, `grad_accum_steps`, `fork_lr`, `fork_lr_freeze` —
`dry_run.REPORTED_DESTS`) · the pool as recorded (`N snapshot(s)` + `win_rate_vs_bots`, so pool
drift is visible BEFORE launch) · then one `(child-only: …)` line per fact it structurally cannot
compute.

**It refuses what the real path refuses**, with the same exit code: an unresolvable `--pin-commit`
and a same-run RESTART whose `--pin-commit` differs from the checkpoint's recorded hash both leave
`FATAL_CONFIG` (3); `--pin-commit` + `--sync-to-main` and `--pin-commit` + `--no-pin` still fail at
parse time (a dry run is not a way around a refusal the parser owns); a stale flag or a refused
combination is `FATAL_CONFIG` too. A run-dir resolution failure exits `1`, as `_prepare_session`
does.

**What it never does, and how that is enforced.** No run dir created or modified, no worktree, no
startup prune, no child, no `metadata.json` / `latest.txt` / `model_config.json` write, no
environment export — and not even the `--nice` change, because `main()` returns into `dry_run`
*before* `_apply_nice`. That ordering is the guarantee: `dry_run.py` imports only pure resolvers and
never reaches `_create_run_worktree` / `_prune_stale_launcher_worktrees` / `_launch_child`. It is
PROVEN rather than asserted by `dry_run_test.py`, which sha256s (+ mtime) every file in a fake run
dir before and after a same-run-restart dry run and requires byte-identity, checks `git worktree
list` is unchanged, and booby-traps all four effectful entry points so a future edit that reaches
one FAILS the suite.

**What it cannot know.** The architecture-compatibility verdict, the `ModelVersion` round-trip, the
resolved compile flags, the pool SEEDING and the obs dim all need torch and a built model in the
child. Each is printed as `(child-only: …)` rather than guessed at — a dry run that invented them
would be worse than one that names the gap.

**It is the EXECUTING complement to `python -m main.checkargs`** (root `CLAUDE.md` → *Will this
command still launch?*): `checkargs` answers "do these flags still parse and cohere?" from an argv
anywhere; `--dry-run` answers "what would THIS command do, on THIS box, right now?" — and calls
`checkargs.check` for the flag half rather than re-implementing it, so the two cannot drift.

```bash
python -m main.launcher --dry-run --model models/<run>/checkpoints/checkpoint_N_steps.zip \
  --steps 30000000 --device cuda
```

## An argv is validated by the parser of the tree that will RUN it — `pinned_argv.py`

🚨 **`--pin-commit` refused the exact command it exists for, and the reason is a class of drift no
presence check can see.** On 2026-09-05::

    python -m main.launcher --pin-commit b13b30b2 <the argv that run recorded> --dry-run
    error: argument --hp-type-belief-coef: invalid float value: 'learned'

At `b13b30b2` **`--hp-type-belief` TOOK A VALUE** (`learned`). Today that flag is deleted, so
argparse — which abbreviation-matches by default — resolved the token onto the surviving
`--hp-type-belief-coef` and handed it the value. Every argv check the launcher performed (its own
`--dry-run`, and `main.checkargs`) read the **CURRENT** tree's `build_parser()`, while the child
runs the **PINNED** tree's. **A same-named flag whose ARITY or TYPE changed is invisible to a "does
the parser still know this flag?" test**, because the current parser thinks it does — and the
result was that re-running an old recipe on its own commit, the one thing `--pin-commit` is for,
could not be validated at all.

**The rule now: when the resolved pin names a commit other than the HEAD of the checkout the
launcher is running from, every argv validation runs against the PINNED tree's parser.** Pin ==
HEAD is unchanged in every respect (no subprocess, no archive, the current parser) — pinned to your
own tree, the current parser IS the right one.

**How the pinned parser is obtained** (`pinned_argv.pinned_parser_check`, one subprocess):
`git archive <sha> -- src/main src/agents src/utils src/poke_env data` into a temp dir, copy
`pinned_argv_probe.py` beside it, run it with a **clean environment** (the caller's `PYTHONPATH`
names the *current* `src` in every worktree shell on this box, so inheriting it would silently
validate against the parser we are trying not to use). **Measured on this repo (2026-09-05, box
under a live run): `b13b30b2` 3.25 s cold / 2.89 s warm (`ast_scan`, 171 options); `HEAD~30`
3.33 s cold / 2.84 s warm (`build_parser`, 579 options).** The archive+extract is only ~0.4 s of
that — the rest is the probe subprocess importing the pinned tree — so the per-sha cache saves
little and the whole check is ~3 s either way. `data/` is 18 MB of the archive and `src/` 20 MB;
`src/rust_sim`'s 66 MB is excluded, since nothing on the parser path imports it. Time-boxed at
60 s, and a timeout is UNAVAILABLE, never a pass.

**`git archive`, never `git worktree add`.** A worktree is a durable, registered, prunable object,
and a one-second validation command that touched the worktree list has already cost this program a
live production run (see the prune incident above). An archive is a read of the object database
and leaves nothing registered anywhere. `pinned_argv_test.py` asserts `git worktree list` is
unchanged.

**Three outcomes, and only one of them is a verdict:**

| mode | what it is | a failure means |
|---|---|---|
| `build_parser` | the pinned tree's own `build_parser()` — the parser the child constructs | **`FATAL_CONFIG` (3)**, naming the offending token. The child would die on it ~40 s later, with a run dir already on disk |
| `ast_scan` | a STATIC read of every `…add_argument(…)` call in the pinned `train_rl_agent.py` (+ `main/train/parser/*.py`), replayed into a synthetic parser carrying only each option's SPELLING and ARITY | a **WARNING**. `build_parser()` landed 2026-08-16 (`26b28509`); every commit before it — `b13b30b2` included — builds its parser inline inside `main()`, which cannot be called without starting a training job, so this is the only cheap way to ask them anything. A reconstruction can be incomplete, so it may not refuse |
| `unavailable` | `parser_unavailable_at_pin` — git failed, the pinned tree will not import here, nothing was statically readable, or the probe timed out | a **WARNING** naming the reason, and the launch proceeds with the argv marked UNVALIDATED. **Never a silent pass** — but also never a refusal on a check we could not run |

`data/` is in the archive and that is not optional: `utils.paths.repo_root()` is
`__file__`-relative, so a pinned tree looks for its data beside itself and the `gen3_data` facade
raises `FileNotFoundError` at import — which would silently demote every recent pin from
`build_parser` to the static scan.

**Only the PARSER is pinned, and the other checks say so.** The extractor dependency graph
(`agents.model.flag_registry`) and the value-conditional refusals (`main.train.combination_checks`)
are still read from the current tree, so whenever a pinned check ran their findings print as
`ℹ️ ADVISORY — the CURRENT tree, not the pinned parser` and do **not** fail the dry run. That
inversion is the fix: judging a pinned argv by today's flag set is precisely what made
`--pin-commit` unusable.

**Where it is wired.** `run._prepare_session` (the real launch — checked **before**
`_create_run_worktree`, so a refusal creates nothing, and against `child_args` **with `--run-dir`
injected**, because that is the argv the child receives), `launcher/dry_run.py`, and
`main.checkargs --pin <sha>`. `checkargs` defaults the pin to the git_hash **recorded by the argv's
`--model` checkpoint** — the commit `worktree.resolve_pin` would pin a resume to — and always
prints which parser it used.

```bash
python -m main.checkargs --pin b13b30b2 --argv "--steps 1000 --hp-type-belief learned"
python -m main.checkargs models/<run>          # pins itself to that checkpoint's git_hash
```

Gate: `pinned_argv_test.py`, over a real 4-commit temp repo whose `--flag` **changes arity** across
commits (a deleted flag would test the easy half): the same argv validates clean at commit 1 and is
refused at commit 2 with the token named, `--dry-run` exits 0 and 3 respectively, a pin naming HEAD
spawns no probe at all, a commit with no `build_parser` degrades to the static scan and one with no
readable parser reports `parser_unavailable_at_pin` and still launches.

## Which interpreter the child runs

`child.resolve_child_python()`, in precedence order:

| # | Source | Notes |
|---|---|---|
| 1 | **`$GEN3AI_PYTHON`** | Explicit override. Set it only to run the child under a *different* interpreter than the launcher — a blank/whitespace value falls through rather than becoming `argv[0]` |
| 2 | **`sys.executable`** | The default: the launcher's OWN interpreter |

**`sys.executable` is the correct default, not a guess.** The launcher is already running under the
environment the run wants, so the child inherits it on any machine under any env name — there is no
conda prefix, env name or absolute path to keep in sync, and a fresh clone needs no source edit.
The resolved value is announced in the events panel at startup (`🐍 Interpreter: …`, marked
`(pinned by $GEN3AI_PYTHON)` when the override is live), because if the launcher was started from
the wrong environment then *every* child inherits that, and this line is where it shows.

It is resolved at **spawn** time, not import time: a launcher process outlives a dozen children
across periodic and crash restarts, so an import-time constant would pin the first value.

> **History.** This was a hardcoded `/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3` with
> no flag and no override until 2026-08-22 — a fresh clone died with `FileNotFoundError` on its
> first launcher run and the only fix was editing the source. On this box the change is
> behaviour-identical (the launcher *is* started with that interpreter, so `sys.executable` resolves
> to it). Gate: `interpreter_test.py`, whose durable half fails if **any** launcher module
> re-introduces a machine-specific path — not just the old line.

**Recorded commands are unaffected.** `run.py` records `LAUNCHER_COMMAND = " ".join(sys.argv)`, and
`sys.argv[0]` is the launcher's `__main__.py`, never the interpreter — verified over all 104
archived `models/*/metadata.json` (0 embed a python or conda path). The launcher constructs the
child argv itself, so an old run's recorded command relaunches unchanged.

## Resume contract

🚨 **A PERIODIC RESTART is a resume of the SAME run, and one flag has to tell the two apart.**
`--fork-lr` pins the LR of a checkpoint being FORKED (`--lr` is inert on any resume — the optimizer's
saved rate wins), and the restart loop re-invokes the same argv into the same run dir every
`--restart-interval-hours`. So the trainer keys the pin on WHERE the resumed checkpoint lives
(`main/train/fork_lr.py::is_same_run_checkpoint`): outside the run dir ⇒ a FORK, pin applies; a
checkpoint this run wrote (`<run>/checkpoints/*.zip`, or `<run>/*.zip` for the legacy layout) ⇒ a
RESTART, the pin is NOT re-applied and the KL controller keeps its adapted rate. That is the same
predicate this package's own `checkpoint.resolve_fork_resume_model` uses to decide whether a restart
re-inits from the source or continues in place — and because that function SWAPS `--model` to the
fork's own checkpoint once the fork has progress, restart #2 of a fork reads RESTART for the same
reason a plain resume does. `--fork-lr-freeze` is the exception: it is a property of the RUN, so it
persists across every restart, re-read from `metadata.json`'s `dose.fork_lr_pin`.

🚨 **THE SAME SPLIT GOVERNS THE SELF-PLAY POOL, and it is why the restart loop is safe here.** A
FORK begins in a new run dir whose `snapshots/` is empty, and an empty pool does not disable
`--self-play` — it silently falls back to the BOT pool. `agents.training.pool_seed` therefore
auto-seeds a genuine fork's pool from its parent (the zips AND `summary.json` /
`win_rate_vs_bots.txt` / `model_config.json`, since the starting `self_play_fraction` comes from the
metadata) and REFUSES a fork whose pool is still empty with `FATAL_CONFIG`. It keys on the SAME
imported `is_same_run_checkpoint`, so a periodic restart never re-seeds — which matters more here
than for `--fork-lr`: re-seeding on every restart would overwrite the run's own grown pool with the
parent's stale one every few hours. The two flags (`--no-fork-pool-seed`, `--allow-empty-pool`) are
trainer-owned and forwarded verbatim; the launcher must never acquire a default for either
(`pool_seed_flag_forwarding_test.py`, same shape as `compile_flag_forwarding_test.py`).

The checkpoint must have a `metadata.json` with a `git_hash` field (written automatically by
`save_model_snapshot()`). The launcher pins the worktree to that exact commit so the resumed
run uses the same code as the original — unless `--sync-to-main` or `--pin-commit` is passed.

### Which commit a checkpoint records

🚨 **The pin only works if the recorded hash IS the code that ran, and until 2026-09-05 it was
not.** The run whose worktree the prune deleted (above) then failed to resume correctly,
because its checkpoint **sidecar** recorded `fff95a16` — the ambient HEAD of the main checkout
— while the run-level `metadata.json` recorded the actual pin `eb5261ff`. `resolve_pin` reads
the sidecar first, so the resume pinned the wrong commit. Two independent causes:

1. `agents.model.snapshot.record_checkpoint` resolved `git_hash or get_git_hash()`, never
   consulting `$LAUNCHER_GIT_HASH`. Worse, the truthy value it produced then **won** the
   `git_hash or env or …` chain inside `_build_snapshot_entry`, so that function's env
   fallback was dead code for the whole checkpoint path.
2. `utils.git.get_git_hash()` ran `git rev-parse HEAD` **in the process cwd**. The launcher
   puts the pinned worktree on the child's `PYTHONPATH` but spawns it with **no `cwd=`** (see
   the `PYTHONPATH` note above — that split is deliberate, so `models/` lands in the main
   checkout), so the child *imports* the pin while *standing in* un-pinned `main`.

Fixed at the root: `get_git_hash()` is anchored at `utils.paths.repo_root()`, the checkout the
code was **imported from** — in a detached launcher worktree that is the pin, in the main
checkout nothing changes. And **one resolver**, `snapshot.resolve_git_hash`, now serves the
run-level metadata and every sidecar: explicit argument → `$LAUNCHER_GIT_HASH` → the imported
checkout's HEAD, **raising `GitHashMismatchError` when the launcher's pin and the imported
tree name different commits** (a producer-side GIGO throw — a warning in a training child's
stdout is a line in a 1 MiB ring buffer nobody reads). Gate:
`src/agents/model/snapshot_git_hash_test.py`.

**`pin_history` — the scalar `git_hash` is "current", not "the code that ran".** It is
rewritten on every save, so on a run that restarts every 3 h it names the LAST code to touch
the run (observed on `ai_v9_171`: `eb5261ff`, then `fff95a16` after one resume). `metadata.json`
therefore also carries an **append-only** `pin_history` — `{git_hash, pin_source, first_step,
last_step}` per contiguous commit span, written by the same save path, with the same
immutability contract as `lineage` (an existing entry is never rewritten except to advance its
`last_step`). A legacy run with no history is seeded with one `derived: true` span from its
scalar hash, so *absent* never reads as *one commit*. Every checkpoint sidecar stamps the
history as of its write.

**`python -m main.sidecar_audit <models_dir_or_run>…`** is the offline reader (JSON only, no
torch, no `.zip` opened): per run it prints the run-level pin, the `pin_history` spans, and
every sidecar's hash, flags a run with >1 span as **PIN-SPLIT**, and separates a sidecar whose
hash *is* a recorded span (explained — a restart) from one that appears nowhere (misattributed
— the shape this defect leaves). `--json`, `-v`, and `--strict` (exit 1 on any unexplained
hash).

🚨 **THE PIN HAS FOUR SOURCES AND ONE DECISION FUNCTION** (`worktree.resolve_pin`, returning a
`PinDecision(sha, source, subject)`; every refusal is a `PinRefused` carrying the exit code to
leave with). In precedence order: an explicit **`--pin-commit`**, the resumed checkpoint's
recorded **`git_hash`**, **HEAD** under `--sync-to-main`, **HEAD** on a fresh run. The chosen
source is exported to the child as `LAUNCHER_PIN_SOURCE` and recorded as `metadata.json`'s
top-level **`pin_source`** (`"pin_commit"` / `"checkpoint"` / `"sync_to_main"` / `"head"`) beside
the `git_hash` it chose — so a finished run can say whether its commit was NAMED or inherited.

**Why `--pin-commit` exists, measured.** A batch of arms launched sequentially under
`--sync-to-main` each pins to HEAD *at its own launch*, so a commit landing mid-batch splits the
batch across two commits and nothing in any run's output says so — that happened on 2026-09-04
(arm 1 on `0c76e2ee`, arms 2-4 on `52ab5914`). Naming the commit on every arm removes HEAD from
the decision. It is the launcher half of the fix; the chain script's half is to record the pin
once and refuse to launch when HEAD has moved off it.

**A RESTART may never MOVE the pin, and that is the one case `--pin-commit` does not win.** The
launcher re-invokes the identical argv into the identical run dir every
`--restart-interval-hours`, so a `--pin-commit` that differs from the resumed checkpoint's
recorded `git_hash` would silently walk a live run onto other code every few hours. That is
`FATAL_CONFIG`, naming both commits and the three ways out. Fork-vs-restart is
`main.train.fork_lr.is_same_run_checkpoint`, **IMPORTED** — the same predicate `--fork-lr`, the
pool seeding and `resolve_fork_resume_model` key on. The **fork swap runs first** on purpose:
once an idempotent fork has its own progress, re-running its launch command is a RESTART of that
fork, so the guard is checked against the fork's own checkpoint rather than the source it was
originally forged from. Gate: `pin_commit_test.py`.

**The child's `PYTHONPATH` is what makes that pin real, and it must never be "cleaned up."** The
spawn passes no `cwd=`, so `PYTHONPATH=<worktree>/src` is the only thing making a resumed run
*import* the code its checkpoint was saved on. Measured (2026-08-22 scope survey, Finding B): with
an editable install present and no `PYTHONPATH`, a pinned old-commit child imports `agents` from the
**main checkout** — an old checkpoint silently resuming on current HEAD, the arch-drift disaster
class. `PYTHONPATH` entries land in `sys.path` *before* a `.pth`'s, so the pin and an editable
install coexist correctly exactly as long as that line stays. Note the deliberate split it creates:
the child **imports from the worktree** while writing `models/` **relative to the launcher's cwd**
(the main checkout).

That second half is why `models/` exists **only in the main checkout** and never in a worktree —
so anything else that needs the run archive must reach across rather than look beside itself.
`utils.paths.main_models_dir()` is that reach (via git's shared `--git-common-dir`, the same fact
`utils.git.get_main_repo_root()` reads); see the root `CLAUDE.md` § *Path discovery*. Four tests
used to encode this box's absolute path instead and therefore skipped forever everywhere else.

## Showdown port default

⚠️ **The port default is now MOSTLY UNREACHABLE, because the transport default inverted.**
`--use-bridge` defaults to `rust`, so a launcher run with no transport flag is a BRIDGE run and
gets no port at all. The port logic below applies only to an explicit `--use-bridge off`.

**Bridge mode is port-free, and it is the default.** `child_uses_bridge` treats an ABSENT
`--use-bridge` as a bridge run (matching `train_rl_agent`'s own default — a drift between the two
is what `default_port_test.py` now exists to catch), so `_apply_default_showdown_port` injects
**no** default port and the events panel shows `🌉 Transport: in-process bridge [rust] (no Showdown
server)` instead of a port. The bridge connects to no server at all (training AND eval run
in-process), so any `--showdown-port` passed alongside it is inert — built into `server_config` but
never connected to, so it cannot even disturb the live :8001 server.

**When `--use-bridge off` IS passed**, the launcher **defaults `--showdown-port` to 8001**
(`DEFAULT_TRAINING_SHOWDOWN_PORT` in `launcher/checkpoint.py`, injected in
`launcher/__init__.main()` via `_apply_default_showdown_port`) so a long websocket session never
rides on the shared dev server (8000), where a routine dev `npm run stop` would drop every worker's
connection at once and the connection guard would crash the run. An explicit `--showdown-port` (any
spelling) always wins; the resolved port shows in the TUI events panel (`🔌 Showdown server :8001`).
This default lives **only** here — `train_rl_agent.py` run directly still defaults to 8000.

Guarded by `default_port_test.py` (both directions: absent flag ⇒ bridge ⇒ no port; explicit `off`
⇒ the 8001 injection). See the root `CLAUDE.md` → In-process bridge transport, and the
port-threading detail in `src/agents/training/CLAUDE.md`.
