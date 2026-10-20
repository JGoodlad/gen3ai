# CLAUDE.md — the training launcher (`src/main/launcher/`)

The launcher wraps the trainer for long unattended runs: periodic and crash restarts, a git-worktree
pin, a Textual TUI (headless when detached) and the child's logs. Operating it (launch, resume, fork)
is `designs/ops/training_runbook.md`. This file holds the rules for changing its code; the mechanism,
evidence and tables are in `designs/launcher/` (`README.md` indexes them: flags in `flags.md`, exit
codes in `crashes_and_exit_codes.md`, the child's interpreter in `interpreter.md`).

## Rules for changing the launcher

- 🚨 **`--dry-run` reaches only pure resolvers.** `dry_run_test.py` booby-traps every effectful entry
  point; keep it that way, because "launch and kill" to validate overwrote a real run's records on a
  same-run restart.
- **Every same-run restart (interval, crash, the `r` key) re-launches in the RESUME role**
  (`checkpoint.resume_child_args`): strip `main.train.combination_checks.fresh_only_flags()`, add
  `--model <latest checkpoint>` and `--run-dir`. Never re-pass the original argv.
- 🚨 **A `--model` launch inherits its recipe from the run that wrote the checkpoint**
  (`recipe_surface.inherit_recorded_recipe`; a fork: the parent's), and `_prepare_session` passes those
  knobs to the child TYPED (`materialize_recorded_recipe`), because a child pinned to an older commit
  resolves an untyped knob to its own parser default. A missing value refuses by name, never guesses.
- **Fork vs restart is one predicate, imported, never re-derived**:
  `main.train.fork_lr.is_same_run_checkpoint`. `--fork-lr`, the fork's pool seeding, the idempotent-fork
  `--model` swap (`checkpoint.resolve_fork_resume_model`, run first), the restart pin guard and
  `--diagnostic-fork-consent` all key on it.
- **Run dirs resolve through `utils.paths.run_archive_dir()`**, and the child gets an ABSOLUTE
  `--run-dir`.
- **The startup prune removes only a DEAD launcher's worktree**: every ambiguity resolves to KEEP, and
  a tree holding run data is kept (`utils.worktree_guard`). A validation command never touches the
  worktree list (`pinned_argv.py` uses `git archive`).
- **A pinned argv is judged by the PINNED commit's parser** (`pinned_argv.pinned_parser_check`). Only
  `build_parser` / `parse_args_hook` verdicts may refuse; `ast_scan` and `unavailable` warn.
- 🚨 **Never "clean up" the child's `PYTHONPATH=<worktree>/src`, and pass no `cwd=` to the spawn**:
  that line is the only thing that makes a pinned child import its pin. The child stands in the launching checkout,
  so it reads its pin's `data/` only because every runtime reader resolves `data/` through `utils.paths.repo_path`,
  never the CWD (`pinned_data_test.py`).
- **The recorded `git_hash` has one resolver**, `agents.model.snapshot.resolve_git_hash`; never derive
  it elsewhere. **A restart never moves the pin**: `--pin-commit` differing from the resumed
  checkpoint's hash is `FATAL_CONFIG`.
- **The arch and recipe surface guards are one function each** (`main.train.arch_surface.report`,
  `main.train.recipe_surface.report`), read by the launcher, `--dry-run`, `main.checkargs` and
  `resolve_config`. Never add a second copy of the comparison, or a hand list of keys (they derive from
  the flag registry).
- **A `--model` argv's architecture is judged before launch by the trainer's own code**
  (`main.train.fork_arch`: `resume_version_build` + `check_compatible`, via `checkargs.check`). A growth
  option opens a field only through `fork_arch.DECLARED_GROWTH`, honoured by the trainer too.
- **The launcher owns no trainer default**: everything not launcher-owned is forwarded verbatim, and
  both parsers are `allow_abbrev=False` (the forwarding tests pin this against the real
  `build_launcher_parser()`).
- 🚨 **A detached launch runs HEADLESS** (`run.headless_mode()`, stdin not a TTY): Textual's input
  thread busy-loops a core on a `/dev/null` stdin (`headless_test.py`).
- **The child's save is deferred to a safe point** (`src/main/train/deferred_abort.py`);
  `KILL_GRACE_SECONDS` must stay above the safe-point deadline plus the save budget
  (`src/main/train/deferred_abort_test.py`). **The save is atomic and `latest.txt` moves last**
  (`designs/training/checkpoint_writes.md`): a failed or killed save leaves the previous file and its old mtime, so the
  freshness checks here (the GPU window's, `find_latest_checkpoint`'s) refuse it instead of loading a torn zip.
- **The preflights refuse `FATAL_CONFIG` before the pin, worktree or run dir exist** (the desktop on
  the GPU, low disk, an unusable `deps/pokemon-showdown`, a `--device cuda` launch under `CUDA_VISIBLE_DEVICES` hiding every GPU; `launch_guards.md`), and `--dry-run` prints
  each verdict. The launcher's ask is the only one a pinned older trainer gets.
- **A new non-recoverable error class goes through `exit_codes.exit_code_for`**, never a bare
  `sys.exit`.
- **A GPU window pauses through the SAME stop and restart paths** (`--gpu-window`, `gpu_window/`): SIGTERM to
  the safe-point save, then the ordinary same-run RESTART (same pin, same command); the launcher stays alive so
  the lease stays with its holder. Every step that cannot finish cleanly HOLDS (never auto-resumes over it), and
  a request is honoured only by the launcher it names (`designs/launcher/gpu_window.md`).
- **The child's interpreter is decided once per launcher session** (`torch_runtime.resolve_for_launch`;
  a resume or fork gets the torch its run recorded) and held in the child's `$GEN3AI_PYTHON`, so every
  restart spawns the same one; never a machine path (`interpreter_test.py`).
