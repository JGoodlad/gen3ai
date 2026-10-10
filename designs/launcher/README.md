# designs/launcher/ — the launcher's topic docs

Detail lifted out of `src/main/launcher/CLAUDE.md` on 2026-10-10. **Always-current**: each doc OWNS
its topic and is updated in the same pass as the launcher code, exactly like the leaf. The leaf keeps
the rules, the module map, the hazards and one line per flag; these hold the mechanism, the evidence
and the gates. The operator's chapter (how to launch, resume, fork) is
`designs/ops/training_runbook.md` → *Launcher*; the frozen pre-cleanup leaf is
`designs/research_state/claude_md_archive/src_main_launcher_CLAUDE_2026-10-10.md` (history).

| doc | owns |
|---|---|
| [`restarts_and_resume.md`](restarts_and_resume.md) | run-dir resolution (archive, fork, idempotent fork, the fresh-into-progress refusal), periodic restarts and the deferred safe-point save, the RESUME role a restart re-launches in, recipe inheritance on restart, a Rust-core run under the launcher, the resume contract (`--fork-lr`, the self-play pool) |
| [`crashes_and_exit_codes.md`](crashes_and_exit_codes.md) | crash auto-restart and its circuit breaker, the non-recoverable classes, the child logs (the ring and the rotating full copy), the full exit-code table |
| [`flags.md`](flags.md) | the full flag table with every note, and why the launcher owns no trainer default |
| [`pinning_and_worktrees.md`](pinning_and_worktrees.md) | worktree isolation, the startup prune and its ownership record, the PINNED-parser argv check (`pinned_argv.py`), which commit a checkpoint records, `pin_history`, the pin's four sources, the child's `PYTHONPATH` |
| [`launch_guards.md`](launch_guards.md) | `--dry-run`, the ARCH-SURFACE and RECIPE-SURFACE guards, the desktop-GPU, disk-space and Showdown-submodule preflights |
| [`interpreter.md`](interpreter.md) | which interpreter (and torch) the child runs |
| [`tui_and_session.md`](tui_and_session.md) | how the supervisor rides beside Textual's event loop, quit / SIGHUP / SIGTERM teardown, HEADLESS mode, stdout discipline, the dashboard, keys and metrics layout |
