# CLAUDE.md — Training Launcher (`src/main/launcher/`)

The launcher wraps `train_rl_agent.py` for long, unattended runs: periodic restarts, crash
auto-restart, git-worktree isolation (a pin), a Textual TUI (headless when detached) and live child
logs. **How to launch / resume / fork is the operator chapter `designs/ops/training_runbook.md` →
*Launcher*; this file is the rules and the map for working ON the launcher. The detail — mechanism,
evidence, gates — is the topic docs in [`designs/launcher/`](../../../designs/launcher/README.md)**
(lifted out 2026-10-10; the old leaf is frozen at
`designs/research_state/claude_md_archive/src_main_launcher_CLAUDE_2026-10-10.md`, history).

## Module map

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
| `disk_gate.py` / `submodule_gate.py` | the launcher halves of the disk-space and Showdown-submodule preflights |
| `state.py`, `ipc.py`, `input.py`, `format.py` | the lock-protected state snapshot, child IPC, key dispatch, pure formatters |
| `app.py` + `launcher.tcss` | the Textual UI (a `Gen3App` on the shared `src/main/tui/` base) |

## Where the detail is

| I am about to touch… | Read |
|---|---|
| restarts, run dirs, forks, the resume contract, a Rust-core run under the launcher | [`designs/launcher/restarts_and_resume.md`](../../../designs/launcher/restarts_and_resume.md) |
| crash restart, the child logs, an exit code | [`designs/launcher/crashes_and_exit_codes.md`](../../../designs/launcher/crashes_and_exit_codes.md) |
| a launcher flag | [`designs/launcher/flags.md`](../../../designs/launcher/flags.md) (+ `designs/ops/flag_census.md`) |
| the pin, the worktree prune, the pinned-parser check, a recorded `git_hash` | [`designs/launcher/pinning_and_worktrees.md`](../../../designs/launcher/pinning_and_worktrees.md) |
| `--dry-run`, the arch / recipe guards, a startup preflight | [`designs/launcher/launch_guards.md`](../../../designs/launcher/launch_guards.md) |
| the child's interpreter / torch | [`designs/launcher/interpreter.md`](../../../designs/launcher/interpreter.md) |
| the TUI, quit / signal teardown, headless mode | [`designs/launcher/tui_and_session.md`](../../../designs/launcher/tui_and_session.md) |

## Hazards and rules (each has cost a run or a launch)

- 🚨 **Never "launch the real command and kill it" to validate — use `--dry-run`.** Harmless on a
  fork, DESTRUCTIVE on a same-run restart: a few seconds wrote `final_model_interrupted.zip`,
  repointed `latest.txt` and overwrote `metadata.json` / `model_config.json` of the real run
  (2026-09-05). `--dry-run` reaches only pure resolvers; `dry_run_test.py` booby-traps every effectful
  entry point and byte-checks a fake run dir — keep it that way.
- 🚨 **Every same-run restart (interval, crash, forced `r`) re-launches in the RESUME role**
  (`checkpoint.resume_child_args`): strip `main.train.combination_checks.fresh_only_flags()` (today
  `--arch`), add `--model <latest checkpoint>` + `--run-dir`. Never re-pass the original argv
  (`ai_v14_01_base` crash-looped out at its first restart, 2026-09-26). A FRESH launch into a dir
  holding a checkpoint or `model_config.json` is REFUSED (`FreshRunDirHasProgress`, `FATAL_CONFIG`).
  On restart the CHILD restores an `--arch production` run's untyped recipe knobs
  (`recipe_surface.inherit_on_restart`); a missing value refuses, never guesses.
- 🚨 **Fork vs restart is ONE predicate, imported, never re-derived:**
  `main.train.fork_lr.is_same_run_checkpoint` (where the resumed checkpoint lives). `--fork-lr`, the
  fork's pool seeding, the idempotent-fork `--model` swap (`checkpoint.resolve_fork_resume_model`, run
  FIRST) and the restart pin guard all key on it. A fork command is safe to re-run: it resumes its
  own progress instead of re-copying the source.
- 🚨 **Run dirs resolve through `utils.paths.run_archive_dir()`** (`$GEN3AI_MODELS_DIR`, else the MAIN
  checkout's `models/`) and the child gets an ABSOLUTE `--run-dir`; a run dir inside a worktree's own
  `models/` is a `RunArchiveError` → `FATAL_CONFIG` (eight runs died with their worktree, 2026-09-23).
- 🚨 **The startup prune removes only a DEAD launcher's worktree.** Each `launcher-*` worktree has a
  `<worktree>.owner.json` claim (pid + `/proc` start time) BESIDE it; every ambiguity resolves to
  KEEP, and a tree holding RUN DATA (`utils.worktree_guard`, the same guard `scripts/land.sh` runs) is
  kept. The old prune-everything rule deleted a live production run's checkout (2026-09-05).
  Corollary: **a validation command must never touch the worktree list** — `pinned_argv.py` uses
  `git archive`, never `git worktree add`.
- 🚨 **A pinned argv is judged by the PINNED commit's parser** whenever the pin is not this
  checkout's HEAD (`pinned_argv.pinned_parser_check`): a flag whose ARITY changed, or one ADDED since
  the pin (`✗ NOT IN PINNED TREE`), is invisible to a presence check against this tree. Only
  `build_parser` / `parse_args_hook` verdicts may refuse; `ast_scan` and `unavailable` WARN, and the
  mode is printed on every run. The arch / recipe / combination checks are ADVISORY under a non-HEAD pin.
- 🚨 **The child's `PYTHONPATH=<worktree>/src` must never be "cleaned up", and the spawn passes no
  `cwd=`.** That line is the only thing making a pinned child IMPORT its pin (an editable install's
  `.pth` sits after it); the child stands in main and writes the absolute `--run-dir`.
- 🚨 **The recorded `git_hash` has ONE resolver** — `agents.model.snapshot.resolve_git_hash`
  (explicit → `$LAUNCHER_GIT_HASH` → the IMPORTED checkout's HEAD, raising `GitHashMismatchError` when
  they disagree). A sidecar once recorded main's ambient HEAD instead of the pin. The scalar `git_hash`
  is "current"; the code that ran is the append-only `pin_history` (`python -m main.sidecar_audit`).
- 🚨 **A RESTART may never MOVE the pin**: `--pin-commit` differing from the resumed checkpoint's
  recorded hash on a same-run restart is `FATAL_CONFIG`. Pin precedence: `--pin-commit` > the
  checkpoint's `git_hash` > HEAD (`--sync-to-main` or fresh); the source is recorded as `pin_source`.
- 🚨 **The ARCH and RECIPE surface guards are ONE function each, four readers**
  (`main.train.arch_surface.report`, `main.train.recipe_surface.report` — the launcher, `--dry-run`,
  `main.checkargs`, `resolve_config`). A FRESH argv that differs from `designs/production_config.json`
  is REFUSED (`--allow-nonproduction-arch` / `--allow-nonproduction-recipe` consent); never add a
  second copy of the comparison, and never a hand list of keys (it is derived from the flag registry).
- 🚨 **The launcher owns NO trainer default.** Everything not launcher-owned is forwarded verbatim;
  `compile_flag_forwarding_test.py` and `pool_seed_flag_forwarding_test.py` pin that against the REAL
  `build_launcher_parser()`, and both parsers are `allow_abbrev=False`.
- 🚨 **A detached launch runs HEADLESS** (`run.headless_mode()`: stdin not a TTY). Textual's input
  thread busy-loops a core on a /dev/null stdin (96% of a core for 13 h plus a 982 MB repaint log,
  2026-08-14); `headless_test.py` fails at 96.7% CPU if reverted. Events echo as plain `[HH:MM:SS]`
  lines to the REAL `sys.stdout`, outside the state lock.
- 🚨 **The child's save is DEFERRED to a safe point** (`main/train/deferred_abort.py`): a SIGTERM /
  SIGHUP / SIGINT, and the `c` key's SIGUSR1, only record the request. A 120 s watchdog
  (`SAFE_POINT_DEADLINE_SEC`) exits 15 WITHOUT a save; `KILL_GRACE_SECONDS` (150 s) must stay above
  that plus the save budget (`deferred_abort_test`). The child shares the launcher's session (no
  `start_new_session`), so a closed terminal SIGHUPs both — both handle it.
- 🚨 **A resume or fork runs under the torch its run RECORDED** (see *Which interpreter the child
  runs* below).
- 🚨 **Three preflights refuse `FATAL_CONFIG` BEFORE the pin / worktree / run dir exist**, and
  `--dry-run` prints each verdict: the desktop holding the GPU (`utils/desktop_gpu.py`,
  `--allow-desktop-gpu`), too little disk for the run (`disk_gate.py` → `utils/disk_guard.py`,
  `--allow-low-disk`), and an unusable `deps/pokemon-showdown` (`submodule_gate.py` →
  `utils/showdown_deps.py`, no opt-out). The launcher's ask is the only one a PINNED older trainer gets.
- The launcher has **no Showdown port**: training and eval run in-process on the Rust core, and
  `--showdown-port` is deleted (`designs/deleted_flags.md`). A PINNED resume of a pre-Rust run runs its
  own commit's trainer with that trainer's own default.

## Exit codes (`src/main/exit_codes.py`) — the launcher's reaction

Full table with causes and pins: [`designs/launcher/crashes_and_exit_codes.md`](../../../designs/launcher/crashes_and_exit_codes.md).
Errors are mapped by class NAME through `exit_codes.exit_code_for`; a new non-recoverable class goes
through that mapping, never a bare `sys.exit`.

| code | `TrainExitCode` | launcher |
|---|---|---|
| 0 | `COMPLETE` | stops |
| 15 | `INTERRUPTED` | restarts from the latest checkpoint |
| 1 | `CRASH` (and any non-enum code) | saves `crashes/restart_err_<token>.txt`, restarts; `--max-crash-restarts` consecutive crashes each < 10 min after launch stop it; no checkpoint ⇒ fatal |
| 3 | `FATAL_CONFIG` | **no restart** (a `ModelVersionError`, any `FatalConfigError`, a T2 `ParityFailure` / `VacuousParity`); reason shown in the Events panel. The launcher's own refusals (preflights, pin, surfaces) also exit 3 |
| 4 | `FATAL_NONFINITE` | **no restart** — a NaN / Inf loss, gradient or weights |
| 5 | `FATAL_SUPPLY` | **no restart** — a live lever's supply starved |
| 6 | `FATAL_CUDA_LEAK` | restarts at most `CUDA_LEAK_RESTART_CAP` (2) times per session, then stops |
| 7 | `FATAL_LIVE_PARSE` | **no restart** — live play's T28 halt (the launcher never runs live play) |
| 8 | `FATAL_DISK` | **no restart** — the in-flight disk guard stopped cleanly after a save; free space, then resume (`--allow-low-disk` keeps the warning, drops the stop) |

## Flags

Full notes: [`designs/launcher/flags.md`](../../../designs/launcher/flags.md). Launcher-owned flags
are stripped; everything else is forwarded verbatim to `train_rl_agent.py`.

| Flag | Default | Notes |
|------|---------|-------|
| `--restart-interval-hours` | `6.0` | `0` = a single run, no restart. `run.DEFAULT_RESTART_INTERVAL_HOURS` (owner rule 2026-10-04, in code 2026-10-10; it was `3.0`). Pinned by `restart_interval_default_test.py` |
| `--max-crash-restarts` | `3` | consecutive rapid (< 10 min) self-crashes before giving up; `0` = unlimited; sustained progress resets it |
| `--restart-grace-minutes` | `20.0` | force-kill window past a scheduled restart's deadline; SIGKILL after `KILL_GRACE_SECONDS`. A silent child shows `⚠ no child output for Nm` — no auto-restart on stall |
| `--nice` | `10` | niceness for the launcher and everything it spawns (inherited across fork/exec); only ever raises; `0` disables. `nice_test.py` |
| `--no-pin` | off | no worktree; run from the current tree |
| `--sync-to-main` | off | pin a resume to HEAD instead of the checkpoint's `git_hash` |
| `--pin-commit COMMIT` | unset | pin to a NAMED commit (legacy spelling `--pin-to-hash`); refused beside `--sync-to-main` / `--no-pin`, and on a restart that would move the pin; unresolvable ⇒ `FATAL_CONFIG` |
| `--allow-torch-switch` | off | consent to resume under a torch other than the recorded one (launcher-owned) |
| `--dry-run` | off | resolve and print the launch, create nothing; exits 0 or the refusal's code |
| `--arch production` | unset | *(forwarded)* the production ARCH surface + TRAINING RECIPE as if typed; FRESH-only, stripped on restart |
| `--allow-nonproduction-arch` / `--allow-nonproduction-recipe` | off | *(forwarded)* consent to a FRESH run off the production mirror |

## Which interpreter the child runs

`torch_runtime.resolve_for_launch` decides ONCE per launcher session and holds the choice in
`child_env[$GEN3AI_PYTHON]`, so every restart spawns the same interpreter. FRESH:
`child.resolve_child_python()` — `$GEN3AI_PYTHON`, else `sys.executable` (never a machine path;
`interpreter_test.py` fails on one). RESUME / FORK: the interpreter carrying the run's RECORDED torch
(`metadata.json` `torch_version`; none = 2.5.1), else the sibling env `KNOWN_ENVS` names (selected and
announced), else `FATAL_CONFIG` unless `--allow-torch-switch`. HEAD's code runs torch >= 2.8 only
(`utils/torch_floor.py`), so a 2.5.1 run resumes only PINNED — `--no-pin` / `--sync-to-main` make it
refuse at startup. A bare `train_rl_agent.py --model …` is not checked by this. Detail:
[`designs/launcher/interpreter.md`](../../../designs/launcher/interpreter.md).

## Tests

`src/main/launcher_app_test.py` (Pilot: render, keys, confirm overlays, signals; `_supervise`
exit codes, crash restart, `_reap`) · `src/main/launcher_test.py` (checkpoint / strip / dispatch /
crash-log / run-dir / log sinks) · in this directory: `state_test.py`, `headless_test.py`,
`dry_run_test.py`, `restart_resume_role_test.py`, `worktree_prune_test.py`, `pinned_argv_test.py`,
`pin_commit_test.py`, `torch_runtime_test.py`, `interpreter_test.py`, `nice_test.py`,
`compile_flag_forwarding_test.py`, `pool_seed_flag_forwarding_test.py`, `reap_grace_test.py`,
`restart_interval_default_test.py`, `submodule_gate_test.py`, `checkpoint_event_test.py`,
`ipc_test.py`, and one per non-restart exit code (`parity_refusal_exit_test.py`,
`nonfinite_exit_test.py`, `cuda_leak_exit_test.py`, `disk_exit_test.py`).
