# Restarts, run dirs and the resume contract

> Lifted out of `src/main/launcher/CLAUDE.md` (2026-10-10). Always-current: update it with the code.
> Crash restarts and exit codes: [`crashes_and_exit_codes.md`](crashes_and_exit_codes.md). Which
> commit a resume pins: [`pinning_and_worktrees.md`](pinning_and_worktrees.md).

## Run-dir resolution (in `_prepare_session`, before the screen opens)

- **Run-dir resolution** (`checkpoint.resolve_launch_run_dir`, three cases): a **fresh** run (no
  `--model`) honours `--run-dir` (made absolute), then `--run-name <name>` (→ `<archive>/<name>`,
  basename-sanitized — a memorable name without the full path), else a timestamped `<archive>/rb_run_<ts>` (era-prefixed, `utils/era.py`; an explicit name is accepted as typed).
  🚨 **`<archive>` is `utils.paths.run_archive_dir()` — `$GEN3AI_MODELS_DIR`, else the MAIN checkout's
  `models/` — never a cwd-relative `models/`** (from a worktree that directory is deleted silently with
  it; 2026-09-23). Every resolved dir is ABSOLUTE and passes `checked_run_dir`: an explicit `--run-dir`
  (or a resumed checkpoint's own dir) inside a linked worktree's own `models/`, or no archive at all, is a
  typed `RunArchiveError` → `FATAL_CONFIG` (3, `sys.exit` in `_prepare_session`; `--dry-run` prints
  `REFUSED (run dir)` and returns 3). `archive_anchored_args` first re-points a `--model models/<run>/…`
  the cwd does not hold at the archive (a fallback — a main-checkout launch is byte-identical).
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
  🚨 **A FRESH launch into a run dir that already holds a run is REFUSED** (`FreshRunDirHasProgress`,
  exit `FATAL_CONFIG`, `--dry-run` too): a resumable checkpoint (`find_latest_checkpoint`, run-scoped —
  the step-0 `snapshots/` seed does not count) or a `model_config.json` (`checkpoint.run_dir_progress`).
  The message names both fixes — `--model <that run's latest checkpoint>` to continue, or a new
  `--run-name`/`--run-dir`. Before 2026-09-26 there was no fresh-side guard at all (the fork guard
  above never covered a no-`--model` argv), so `ai_v14_01_base`'s argv with `--arch` dropped and no
  `--model` resolved as a step-0 FRESH run INTO the live run dir. A dir with only `metadata.json`
  (a fresh run that crashed before its first save) is still accepted.

## Periodic restarts and the resume role

- **Periodic restarts** — kills and relaunches the child every N hours to reclaim pymalloc
  fragmentation; the child saves a checkpoint on SIGTERM and the launcher picks it up
  automatically. 🚨 **The child's save is DEFERRED to a SAFE POINT** (`main/train/deferred_abort.py`,
  `gen3_deferred_abort_v1`, P10 review F1): a SIGINT / SIGTERM / SIGHUP only records the request,
  and the abort (pending-scalar dump, `final_model_interrupted.zip`, exit 15) runs on the main thread
  at the next loop event (a collector step, a rollout start / end, training start / end) or at
  every HOST STEP of the in-process Rust eval cycle (P10-A2), never inside an update or a logger
  dump. A signal mid-update therefore waits for that update (`train_ms` median ~41 s, max 50.6 s at
  the production recipe, five `sizing_*` runs 2026-10-02/03); one mid-eval waits one host step. If no
  safe point comes within `SAFE_POINT_DEADLINE_SEC` (120 s: ~2x the worst measured stretch without
  one, ~58.5 s = that max + the ~8 s of loop around an update) a watchdog exits 15
  WITHOUT a save (a save then could tear the checkpoint) and the launcher resumes from the last
  periodic checkpoint; 120 s + a 15 s save budget fit inside `KILL_GRACE_SECONDS` (150 s, module
  level in `run.py`, pinned by `deferred_abort_test`) — the grace of BOTH the supervisor's SIGKILL
  escalation and `_reap` (an abnormal app exit; it was 10 s, which SIGKILLed a child mid-update).
  The abort line names the longest stretch between safe points so far. **SIGUSR1 (the `c` key's
  forced checkpoint) is deferred the same way** (`gen3_deferred_checkpoint_v1`): recorded, saved at
  the next safe point, training continues — expect the `💾 [CHECKPOINT] Forced save` line up to an
  update after the key (the events panel shows `Checkpoint requested` at once and `Checkpoint saved →
  <file>` only at the save, `child.checkpoint_event`). The old in-handler abort deadlocked on TensorBoard's non-reentrant writer
  lock when the signal landed inside a dump, and could save mid-update; the old SIGUSR1 handler
  could save mid-update too. The child also checkpoints on **SIGHUP** (`_setup_signal_handlers` routes it
  to the same graceful path) — the child shares the launcher's session (`child.py` spawns it
  without `start_new_session`), so closing the controlling terminal/tmux window SIGHUPs the
  whole group; without that handler the child died mid-iteration with no checkpoint. Running
  the launcher under `nohup` prevents the SIGHUP entirely; the handler is the in-code backstop.
- 🚨 **Every same-run restart (interval, crash, forced `r`) re-launches from the RESUME role** —
  `checkpoint.resume_child_args`: strip every FRESH-only flag the trainer refuses beside `--model`
  (`main.train.combination_checks.fresh_only_flags()` — the trainer's OWN list, declared per check as
  `fresh_only=`; today `--arch`), then `--model <latest checkpoint>` + `--run-dir`. The stripped
  flags are announced (`✂️  Resume argv: dropped FRESH-only --arch`); nothing is lost, the run's
  `model_config.json` holds what `--arch production` expanded to (see the v125 note below). Before 2026-09-26 the loop
  re-passed the original argv, so a fresh `--arch production` run (`ai_v14_01_base`) died at its
  first 3 h restart: exit 2 ×3, circuit-breaker, ~40 GPU-min. `--dry-run` prints the restart argv as
  an `on restart :` line. That refusal now also exits `FATAL_CONFIG` (not argparse's 2), so if a new
  fresh-only flag ever slips the list the launcher stops at once instead of crash-looping. Tests:
  `restart_resume_role_test.py`, `dry_run_test.py::test_h_*`,
  `combination_checks_test.py::test_every_check_refusing_a_resume_declares_what_to_strip`.
  🚨 **"Nothing is lost" holds only for what `model_config.json` RECORDS.** Until config v125 it
  recorded the `opp_intent` BOOL but not `opp_intent_coef`, the dose `--arch production` writes to
  turn it on. So the first restart of a fresh `--arch production` run that did not also type
  `--opp-intent-coef` died with `[ModelVersion] FATAL: opp_intent mismatch: saved=True,
  current=False` (2026-09-30, `~/gen3ai_archive/cutover_prep/fresh2`, on the Rust core). `opp_intent_coef` is now a `ModelVersion` field. A pre-v125 checkpoint migrates
  the dose from its run's `metadata.json:cli_args`, announced as `[Resume] MIGRATION`. With neither
  source the resume is REFUSED (`FATAL_CONFIG`, naming the flag); the dose is never guessed. Tests:
  `main/train/derived_toggle_resume_test.py`. (a) Every key the umbrella's ARCH half writes is a
  recorded field. (b) Fresh → save → this function's argv → resolve returns every surface value unchanged.
  🚨 **The umbrella's RECIPE half (K10(a)) writes knobs `model_config.json` does NOT record**
  (`n_envs`, `n_epochs`, `ent_coef`, `self_play`, …) plus value-CHECKED recorded fields with concrete
  parser defaults (the reward values, `vf_coef`) that `_resolve` cannot inherit. For a run whose
  `original_command` carried `--arch production`, the CHILD resolves each untyped one on a same-run
  restart (`main.train.recipe_surface.inherit_on_restart`): `--lr` / `--batch-size` / `--n-steps`
  untouched (INERT — SB3 restores them); recorded tri-state fields left to `_resolve` (the
  mechanism above — `opp_intent_coef` included); value-checked fields from `model_config.json`; the rest
  from `metadata.json:cli_args` — each announced `[Recipe] … from <source>`, and a MISSING value
  REFUSED (`FATAL_CONFIG`, naming the flag). The launcher's restart argv is unchanged. Test:
  `recipe_surface_test.py::test_a_launcher_restart_of_a_fresh_arch_production_run_keeps_the_whole_recipe`.

## A Rust-core run under the launcher (M5; F-LG-6, closed 2026-09-30)

Exercised end to end on CPU: a fresh launch, an interval restart, a crash restart (the child
SIGKILLed by PID) and a SIGTERM stop, all pinned. Run dirs: `~/gen3ai_archive/cutover_prep/fresh{1..4}`.
What that established:

- 🚨 **The pinned worktree has no Rust env build**, because `git worktree add` gives no `target/`. The
  env core loads from THIS checkout's `src/rust_env/target/<profile>/`
  (`utils.rust_env.proc.default_path` / `ffi.default_path`), so every Rust-core launch through
  the launcher died about 10 s in with `ProcLoadError: …/rust_env_proc does not exist`. The trainer
  now builds its own checkout's core at startup, before the model exists
  (`utils.rust_env.build.ensure_built`, from `rust_env_setup.build_rust_vec_env`). It is an
  incremental `cargo build` into the crate's own `target/`: about 7 s cold in a fresh pin and 0.0 s
  on a restart, printed as `🦀 [ENV CORE BUILD]`. The stamp check still refuses a foreign build. ⚠️ A
  checkpoint recorded at a commit BEFORE this fix still cannot be resumed through the launcher on the
  Rust core: its pin has no build step. Use `--sync-to-main`.
- **A restart keeps the run's env core** (`gen3_env_core_switch_v1`): there is one core and no flag to
  type for it (`--env-core` was deleted, P11b), so every restart re-declares the same core, T2 slots and
  eval core (the SIZES come from `recipe.sizing` / the run's `cli_args`). The events to look for are
  `🦀 [RUST ENV] T2 up …`, `🦀 [RUST EVAL] eval core up …` and `🦀 [ENV CORE] rust — …`. `--dry-run`
  prints `env core : rust [the only core]` beside what the checkpoint recorded, and the trainer emits
  `🔀 [ENV CORE] CORE SWITCH …` for a python-era winprob checkpoint (`rust_env_setup.env_core_switch_line`,
  keyed on the recorded `env_core`). That switch is not refused, because moving a winprob checkpoint
  onto the only core is the M5 carry-over; a SHAPED-critic checkpoint IS refused (D4,
  `refuse_python_era_checkpoint`).
- **A pin that predates the Rust env core flags (before `ac67fa6c`) is refused before anything exists.** The
  pinned parser check names `--rust-eval-envs` (and the other collector flags the argv types) as `NOT IN PINNED TREE`
  (`FATAL_CONFIG`). No flag here is FRESH-only, so none is stripped on a restart. (A recorded argv that
  types a flag HEAD deleted — `--env-core`, `--use-bridge`, `--critic`, `--gamma`, `--victory-value`, `--draw-penalty`, `--terminal-indicator`, … — is the opposite case: the PINNED
  parser knows it, so it is ADVISORY, and `checkargs` still builds the effective config from the rest.)
- `metadata.json` and every sidecar record `env_core` (the core's stamp, T2, trigger) beside
  `git_hash` / `pin_history`. The `*_after_freeze` counters are ENFORCED after every update and eval
  cycle (`LifecycleViolation`), not logged. A clean restart is the absence of that error.
- **A T2 parity refusal STOPS the launcher (2026-09-30).** Before, the per-slot gate refused a
  COLLAPSED win-prob critic (`VacuousParity`: V near-constant even on the seeded perturbation) at a
  trainee load, and exited `CRASH`. The restart resumed `final_model_exception.zip` and T2's
  startup refused the same weights again (`fresh3`: three crashes, then the circuit breaker).
  - Now the gate walks a declared perturbation LADDER and judges the first informative rung
    (`gen3_parity_perturb_ladder_v1`, `designs/research_state/measurements/m5_t2/PROGRESS.md`
    "Flat weights"), so a collapsed critic like fresh3's is judged, not refused. A critic
    saturated beyond what the ladder's capped scale can move is still refused.
  - A refusal that remains exits `FATAL_CONFIG` (see the exit-code table in [`crashes_and_exit_codes.md`](crashes_and_exit_codes.md)), and the launcher shows
    the service's exception line.
  - A fresh `--arch production` model passes T2 startup on the first rung.

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

🚨 **AND A FORK OF A FROZEN RUN MUST NAME ITS OWN DOSE.** The freeze is a property of the parent's
run, not of its weights — a fork inherits the pinned NUMBER through SB3's optimizer state and
leaves the freeze behind, so a live KL controller starts annealing away from a rate that was chosen
precisely because it should not move. That combination is now a startup `[ForkLR] FATAL`
(`gen3_fork_lr_inherit_guard_v1`, `FATAL_CONFIG`/exit 3, refused before the run dir is created);
`--allow-inherited-fork-lr` is the deliberate opt-in, and `python -m main.checkargs` prints the
same verdict offline. The three era-2 exploiters are what it stands for: the plateau parent's
frozen 2.80e-05 → 8.36e-05, median 5.5e-05, 0.39× the v8 reference against era-1's 1.78×, on argvs
that `checkargs` and `--dry-run` had both passed. Detail:
[`designs/training/step_size_and_batch.md`](../training/step_size_and_batch.md).

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
