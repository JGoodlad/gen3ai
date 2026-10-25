# designs/launcher/ — the launcher's topic docs

Detail lifted out of `src/main/launcher/CLAUDE.md` on 2026-10-10. **Always-current**: each doc OWNS
its topic and is updated in the same pass as the launcher code, exactly like the leaf. The leaf keeps
only the rules for changing the code; these hold the mechanism, the evidence, the gates, the flag and
exit-code tables and the module map (below). The operator's chapter (how to launch, resume, fork) is
`designs/ops/training_runbook.md` → *Launcher*; the frozen pre-cleanup leaf is
`designs/research_state/claude_md_archive/src_main_launcher_CLAUDE_2026-10-10.md` (history).

| doc | owns |
|---|---|
| [`restarts_and_resume.md`](restarts_and_resume.md) | run-dir resolution (archive, fork, idempotent fork, the fresh-into-progress refusal), periodic restarts and the deferred safe-point save, the RESUME role a restart re-launches in, recipe inheritance on restart, a Rust-core run under the launcher, the resume contract (`--fork-lr`, the self-play pool) |
| [`crashes_and_exit_codes.md`](crashes_and_exit_codes.md) | crash auto-restart and its circuit breaker, the non-recoverable classes, the child logs (the ring and the rotating full copy), the full exit-code table |
| [`flags.md`](flags.md) | the full flag table with every note, and why the launcher owns no trainer default |
| [`pinning_and_worktrees.md`](pinning_and_worktrees.md) | worktree isolation, the startup prune and its ownership record, the PINNED-parser argv check (`pinned_argv.py`), which commit a checkpoint records, `pin_history`, the pin's four sources, the child's `PYTHONPATH` |
| [`launch_guards.md`](launch_guards.md) | `--dry-run`, the ARCH-SURFACE and RECIPE-SURFACE guards, the desktop-GPU, disk-space, Showdown-submodule and CUDA-visibility preflights |
| [`interpreter.md`](interpreter.md) | which interpreter (and torch) the child runs |
| [`gpu_window.md`](gpu_window.md) | the GPU WINDOW (`--gpu-window`, T46): pause a live run at a safe point for a guest GPU job by file handoff, its files and phases, the HOLD rule, the CLI, the tests |
| [`tui_and_session.md`](tui_and_session.md) | how the supervisor rides beside Textual's event loop, quit / SIGHUP / SIGTERM teardown, HEADLESS mode, stdout discipline, the dashboard, keys and metrics layout |

## Module map (`src/main/launcher/`)

| file | does |
|---|---|
| `__init__.py` / `__main__.py` | entry point `main()` (`python -m main.launcher`); `tui.py` is the back-compat alias (`python -m main.launcher.tui`) |
| `run.py` | the run loop / supervisor (`_prepare_session`, `_supervise`, `_reap`), `build_launcher_parser()`, the module constants (`DEFAULT_RESTART_INTERVAL_HOURS`, `DEFAULT_NICE`, `KILL_GRACE_SECONDS`, `_FAST_CRASH_SECONDS`), `headless_mode()`, `_arch_surface_gate` |
| `checkpoint.py` | run-dir resolution, `find_latest_checkpoint`, `run_dir_for_checkpoint`, the RESUME-role argv (`resume_child_args`), the idempotent-fork swap, the fresh-into-progress refusal, `_strip_launcher_args` |
| `worktree.py` | the pin (`resolve_pin` → `PinDecision` / `PinRefused`), `_create_run_worktree` + its ownership record, the startup prune, the Showdown-submodule link |
| `child.py` | spawning the child (`_launch_child`, `resolve_child_python`), the log sinks (ring + rotating full copy), checkpoint events |
| `torch_runtime.py` | which interpreter / torch a resume or fork runs (`resolve_for_launch`, `KNOWN_ENVS`) |
| `pinned_argv.py` + `pinned_argv_probe.py` | validate the child argv against the PINNED commit's parser, not this tree's |
| `dry_run.py` | `--dry-run`: resolve and print the launch, create nothing |
| `gpu_window/` | the GPU window: `protocol.py` (the files), `controller.py` (the launcher's half, driven by `_supervise`), `cli.py` (`python -m main.launcher.gpu_window arm / guest / status / resume / preempt / cancel`) |
| `disk_gate.py` / `submodule_gate.py` / `cuda_visibility_gate.py` | the launcher halves of the disk-space, Showdown-submodule and CUDA-visibility preflights |
| `state.py`, `ipc.py`, `input.py`, `format.py` | the lock-protected state snapshot, child IPC, key dispatch, pure formatters |
| `app.py` + `launcher.tcss` | the Textual UI (a `Gen3App` on the shared `src/main/tui/` base) |
