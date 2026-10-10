# Crash restarts, crash logs and exit codes

> Lifted out of `src/main/launcher/CLAUDE.md` (2026-10-10). Always-current: update it with the code.
> The leaf keeps the exit-code summary; this doc holds the full table.

## Crash auto-restart

- **Crash auto-restart** — when the child *self-crashes* (unhandled exception → any
  non-`INTERRUPTED` exit), the launcher snapshots its output to a per-crash
  `<run_dir>/crashes/restart_err_<token>.txt` (a timestamp + random hex so back-to-back crashes
  never collide; never overwritten, unlike the reused `launcher_child.log`; folded under a
  `crashes/` subfolder so they don't clutter the checkpoint listing) and relaunches from the last
  checkpoint. A **circuit-breaker** (`--max-crash-restarts`, default 3) stops the run after that
  many *consecutive rapid* crashes (each within `_FAST_CRASH_SECONDS` = 600 s / 10 min of launch)
  so a deterministic startup crash can't spin forever; a crash after sustained progress resets the
  counter. The window is deliberately well past the 3+ min it takes to bring up the env core
  and its inference service, so a startup-time crash is still counted as "rapid" rather than
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

## Crash reporting — the child logs

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

  🚨 **THE RING TRIMS SILENTLY, so a ROTATING full copy rides beside it**:
  **`<run_dir>/launcher_child.full.log`**, with older generations at `.1` … `.7`
  (`child._RotatingChildLog`, **64 MiB × (1 live + 7 backups) = a hard 512 MiB ceiling per
  run**). Both sinks are fed by the same reader thread through `child._ChildLogFanout`, and
  the **ring is written FIRST and is unchanged in every respect** — same path, same cap,
  same trim marker, same tail — because it is what the TUI reads and what the crash dump
  tails; the full copy is strictly additive and a failure in it is swallowed. `log.path` is
  still the RING's path, so every "log written to …" line is unchanged; the exit summary
  additionally names the rotating copy and how many generations exist.

  **Why rotate rather than keep or drop.** 2026-09-06: a per-worker compile count taken
  across a restart became unrecoverable the moment the ring wrapped, and had to be settled
  from source instead — *a read that cannot be redone is a read that cannot be checked*. But
  the ring exists because of a **982 MB repaint log**, so unbounded is not an option: the
  volume this writes to also holds `models/`. 64 MiB × 8 is chosen against that incident —
  the storm fills the rotation and **stops** at roughly half its size, a normal
  restart cycle never engages the rotation at all, and 64 MiB is a file a reader can
  actually `grep` and copy off the box. Gate: `launcher_test.py::TestRotatingChildLog`
  (rotation at the cap, the file-count bound, the ring untouched by the fan-out, and a
  planted repaint storm proving the on-disk total is bounded).

## Exit codes (`src/main/exit_codes.py`)

| Code | `TrainExitCode` | Meaning |
|------|----------------|---------|
| 0 | `COMPLETE` | All steps done — launcher stops |
| 15 | `INTERRUPTED` | SIGTERM received, checkpoint saved at the next safe point (or, past the 120 s safe-point deadline, NO save — the last periodic checkpoint stands) — launcher restarts |
| 1 | `CRASH` | Unhandled exception — launcher saves `crashes/restart_err_<token>.txt` and auto-restarts from the last checkpoint (up to `--max-crash-restarts` consecutive rapid crashes, then gives up; any non-enum exit code is treated the same way). A crash with no checkpoint to resume from is fatal: the child's exit code is propagated and the crash log printed. |
| 3 | `FATAL_CONFIG` | **Non-recoverable** config/architecture error — `train_rl_agent.py` raises it for a `ModelVersionError` (checkpoint arch-family mismatch, or a resume `vf_coef`/reward-config drift), exits with it on the T2 inference service's `ParityFailure` / `VacuousParity` (mapped by NAME; the verdict is deterministic in code + weights + fixture and a restart resumes the same weights; `parity_refusal_exit_test.py`), and for any `main.exit_codes.FatalConfigError` raised anywhere (mapped by NAME through `exit_code_for`, like codes 4/5; `gen3_supply_guard_v2`): a `--bot-weights` typo (it used to exit 1, i.e. be RESTARTED; the consensus warm start, whose failure did so without bound, was deleted by P11), a mis-wired fork arm (`LeverConfigError`). Restarting would hit the *identical* error every time, so the launcher does **not** restart: it saves the crash log, surfaces the reason on-screen, and gives up immediately (returning this code) instead of looping until the crash circuit-breaker trips. See **Crash auto-restart** above. |
| 4 | `FATAL_NONFINITE` | **The learner went non-finite** — a NaN / Inf loss or gradient (`main.exit_codes.NonFiniteLearnerError`, Lane K's K9 fail-closed guard; a guard's own class must subclass it — the NAME is matched along the MRO and the `__cause__` chain), or T2 refused NaN / Inf WEIGHTS (`NonFiniteWeights`, matched by name). The trainer's two fail-fast handlers — and, since 2026-09-30, `build_and_train`'s FRESH-path `learn()` handler, which used to `os._exit(1)` — exit `exit_codes.exit_code_for(exc)`, so this error is 4 and every other uncaught exception stays `CRASH`. Same class as `FATAL_CONFIG`: a restart would resume the checkpoint that produced it and replay the same update, so the launcher saves the crash log, shows `🛑 Non-finite learner — will NOT restart` + the error line, and returns 4. Pinned by `nonfinite_exit_test.py`. |
| 6 | `FATAL_CUDA_LEAK` | **K6's CUDA memory trend STOPPED the child** — `agents.training.learner_lifecycle.CudaMemoryLeakError` (a SUSTAINED growth of live CUDA memory projected an OOM inside the declared horizon; `designs/training/learner_lifecycle.md` "The memory half"). The trainer checkpointed (`final_model_exception.zip`) before exiting, and a fresh process clears a leak, so the launcher **RESTARTS** from that checkpoint with a loud `⚠️ CUDA memory leak STOP #n` event — at most `exit_codes.CUDA_LEAK_RESTART_CAP` (2) times per launcher session, independent of `--max-crash-restarts`; the next one (`🛑 … over the cap … will NOT restart`) ends the session with code 6: a reproducible leak wants a human. Pinned by `cuda_leak_exit_test.py`. |
| 5 | `FATAL_SUPPLY` | **A live lever's supply starved in flight** — `main.exit_codes.SupplyStarvedError` (`gen3_supply_guard_v2`: `agents.training.lever_supply.LeverStarvedError` when a live lever — the self-play pool, PFSP (`--pfsp-scale`), the fork arm — delivers nothing for its declared floor, `designs/training/supply_guards.md`; the cf label supply's guard, `CfLabelSupplyError`, was deleted with the cf training half). A restart would train the same run on the same missing supply, so the launcher saves the crash log, shows `🛑 Starved supply — will NOT restart` + the `[SUPPLY]` lines, and returns 5. Pinned by `agents/training/lever_supply_test.py`. |
| 7 | `FATAL_LIVE_PARSE` | **A LIVE websocket session could not read its input** (T28, owner 2026-10-07) — `main.live.halt.LiveParseHalt` (mapped by NAME): an unparseable or unclassified protocol line, an encoder raise, or a choice it could not send. The live entry point (`main.play`, or `main.anchors` when its live client halts — P6) wrote the durable HALT marker (`python -m main.live.halt status`) before exiting, and every live entry point refuses to start while it exists — exiting 7 itself. The launcher never runs live play, but a child exiting 7 is never restarted (`🛑 Live parse panic — will NOT restart`). Cleared only by `python -m main.live.halt clear --fixed-by <commit>` (an ancestor of HEAD that touches a test file). Pinned by `src/main/live/halt_test.py`. |
| 8 | `FATAL_DISK` | **The disk guard stopped the run** (`utils/disk_guard.py`, 2026-10-09): at a checkpoint save the free space on the run archive's filesystem fell below ONE more checkpoint, or the save itself failed with ENOSPC (the torn file is removed; `latest.txt` still names the previous checkpoint). The checkpoint just written stands, the process exits cleanly through `DeferredAbort.disk_stop` (no new save), and a restart would meet the same full disk, so the launcher saves the crash log, shows `🛑 Disk full (guard) — will NOT restart` + the `[DiskGuard]` line, and returns 8. Free space, then resume with `--model`. Pinned by `main/launcher/disk_exit_test.py`, `main/train/disk_guard_wiring_test.py`. |
