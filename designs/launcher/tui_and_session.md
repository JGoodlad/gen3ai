# The launcher's TUI and session lifecycle

> Lifted out of `src/main/launcher/CLAUDE.md` (2026-10-10). Always-current: update it with the code.
> The UI is **Textual** — the only UI (a second Rich frontend, `ui.py`, was removed once Textual proved
> out) — built on the shared `src/main/tui/` base. Run-dir resolution lives in
> [`restarts_and_resume.md`](restarts_and_resume.md).

## How the UI reconciles with Textual's event loop

`run()` sets up the session (worktree pin, run dir, at-exit handlers) on the main thread, then
drives a `LauncherApp` whose `@work(thread=True)` worker runs the supervisor loop **beside** the
render loop. `LauncherState` (a lock-protected snapshot) is the bridge.

- `_prepare_session()` (worktree pin + run-dir + initial events + at-exit handlers) runs on the
  main thread **before** the screen opens — a pin failure `sys.exit`s with a clean message.
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
  so it saves before exiting (see [`restarts_and_resume.md`](restarts_and_resume.md) → Periodic restarts). The app *also* installs asyncio
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

## The dashboard, keys and metrics layout

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
  cycle, from `eval/elo_vs_<opp>` recorded by `eval_record._record_opponent_elos`). This is the
  at-a-glance "is it going well?" number during self-play pool play — anchored Bradley-Terry over
  the fixed bots, so it rises with strength even while `win_rate_vs_pool` sits pinned near 50% (see
  `src/agents/training/CLAUDE.md` → ELO / skill rating). *(The per-sentinel ELO is a noisy
  single-cycle estimate — `python -m main.elo … --source tb` is the well-anchored canonical fit.)*
  **Metrics layout** — the dashboard's metrics row is **three side-by-side tables** so a metric-rich
  run stays readable instead of one over-long column: a **left misc column** (rollout / time, then the
  `grad/*` diagnostics), a **dedicated `train/*` column** (by far the
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
  `species blf` / `move blf` / `latent` / `move-lat` / `winprob` — so any single scaffold
  crowding out the rest is visible) sits in the left column, while
  `train/return_*`, `train/value_pred_std`, and `train/grad_norm` join the train column — together the
  direct shared-trunk pressure gauge for tuning `vf_coef` (computed
  in `agents/training/grad_balance.py`; see `src/agents/training/CLAUDE.md`). They need no new launcher
  wiring: they ride the same generic `MetricsExporterCallback` scalar path and auto-route by their
  `grad/` / `train/` section prefix; only their display order + short labels are declared in
  `format.py`. `grad/value_policy_logratio` should be seen falling toward ~0. (The `popart/*` block that once appeared under `--use-popart` left with PopArt, L1.)
