# Launch guards — `--dry-run`, the arch / recipe surfaces, the startup preflights

> Lifted out of `src/main/launcher/CLAUDE.md` (2026-10-10). Always-current: update it with the code.
> The pinned-parser check (the other pre-launch refusal) is in [`pinning_and_worktrees.md`](pinning_and_worktrees.md).

## Validating a launch without launching — `--dry-run`

🚨 **A "dry launch" of the real command is safe on a FORK and DESTRUCTIVE on a same-run RESTART.
That asymmetry cost a run's provenance on 2026-09-05.** To check that a restart with a larger
`--steps` would still launch, a session launched the real command and killed it a few seconds
later, after the startup lines. A fork writes a NEW directory, so that habit had always been
harmless; a RESTART operates on the REAL run directory, and those seconds were enough to write
`final_model_interrupted.zip`/`.json`, repoint `latest.txt` at that phantom artifact, overwrite
`metadata.json` (whose `steps` became a target that never ran) and `model_config.json`, and leave
`.compile_quorum` files behind (the compile quorum was deleted in U3; the incident record stands). **"Dry" was a property of forks, never of the launcher.**

`--dry-run` makes it a property of the launcher. It performs *everything the launcher resolves
before a child exists* — argv parse, the fork-vs-restart classification (`fork_lr.
is_same_run_checkpoint`, IMPORTED), the idempotent-fork `--model` swap, the run dir
(`resolve_launch_run_dir`, **without** the `makedirs` that follows it in `_prepare_session`), the
pin (`resolve_pin`), and the effective config a `--model` inherits — prints one startup-shaped
block, and exits.

**What it prints**, in order: role · run dir (flagged `EXISTS — a real launch WRITES INTO IT` when
it does) · `--model` · pin sha + subject + source · `--steps` beside the checkpoint's recorded
`num_timesteps` and the `+X steps` delta · interpreter · transport · restart/grace/nice · the
effective config with each reported flag marked `INHERITED`, `RESTORED at restart from <source>` or `from the argv` (`grad_accum_steps`, `fork_lr`,
`fork_lr_freeze` — `dry_run.REPORTED_DESTS`; a same-run restart of an `--arch production` run restores its untyped recipe rows from `metadata.json:cli_args`, even when HEAD's `ModelVersion` cannot read a PINNED run's `model_config.json`, in which case a `could NOT be read under THIS tree` line says so — `dry_run_test.py::test_i_*`) · the pool as recorded (`N snapshot(s)` + `win_rate_vs_bots`, so pool
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

**It is the EXECUTING complement to `python -m main.checkargs`** (`designs/ops/training_runbook.md` → *Will this
command still launch?*): `checkargs` answers "do these flags still parse and cohere?" from an argv
anywhere; `--dry-run` answers "what would THIS command do, on THIS box, right now?" — and calls
`checkargs.check` for the flag half rather than re-implementing it, so the two cannot drift.

```bash
python -m main.launcher --dry-run --model models/<run>/checkpoints/checkpoint_N_steps.zip \
  --steps 30000000 --device cuda
```

## Is this the ARCHITECTURE you meant? — the ARCH-SURFACE guard (`gen3_arch_surface_guard_v1`)

> **"it launches" and "it is the experiment" are INDEPENDENT checks, and only the RESOLVED-CONFIG
> DIFF tests the second.**

**2026-09-06, ~7 GPU-hours** (ledger `2026-09-06 · INCIDENT`). The first win-prob-critic arm was
launched from a design document's 38-token command block: the critic flags and the PPO knobs, none
of the production feature flags, so every architecture flag silently took its OFF default. Three
gates ran and all three passed — `python -m main.checkargs` exit 0, `resolve_config` accepted,
`--dry-run` clean. All three were RIGHT. The run trained a near-bare network for **25,131 s / 24.4M
steps / 6 checkpoints** and was still holding the GPU when it was found a second time; **31 keys of
its `model_config.json` differ from `designs/production_config.json`** (every edge family off, zero
entity seats, no belief slots, no event window, no intent heads). Every validator here answered
*does this launch*; none answered *is this the architecture you meant*, and only a launch-time
answer arrives before the GPU-hours do.

🚨 **THIS IS A DIFFERENT FAILURE FROM A REFUSED FLAG COMBINATION, AND THE TWO SHARE NO MESSAGE, NO
SUMMARY LINE AND NO REFUSAL PATH.** Rebuilding that arm from an older generation's recorded
`original_command` also fails — on nine flags the win-prob critic (the only critic) SUBSUMES. That failure is **LOUD
and PRE-launch**: `checkargs` names it, nothing starts, it is fixed in a minute. Arch drift is
**SILENT and POST-launch**. A guard that catches the first is no protection against the second.

`_prepare_session` now runs that comparison. It is the LAST thing before anything exists on disk —
immediately before `_create_run_worktree` on the pinned path, before the `makedirs` under
`--no-pin` — and deliberately AFTER the pin decision and the pinned-parser check, because "this
command cannot launch at all" must reach the reader before a question about its intent.

| the argv | what the guard does |
|---|---|
| **FRESH**, un-pinned or pinned to HEAD | prints the diff and **REFUSES** (`FATAL_CONFIG`), naming every differing key with both values |
| FRESH + `--allow-nonproduction-arch` | prints the diff, launches, and stamps `arch_source` in `model_config.json` |
| FRESH + `--arch production` | applies the whole surface first, so there is usually nothing to print |
| **FORK / RESTART** | prints the diff as **INFO** — a resume INHERITS its parent's surface through `config.inherit_saved_flag`, so its silence is the parent's architecture. A same-run RESTART also keeps the run's `arch_source` / `recipe_source` (`arch_surface.inherit_arch_source_on_restart`, `recipe_surface.inherit_on_restart`) — the stripped `--arch` no longer nulls them at the first restart |
| **PINNED to a non-HEAD commit** | prints the diff as **ADVISORY** — see below |

**The ADVISORY rung is `gen3_pinned_argv_parser_v1`'s lesson applied a second time.**
`designs/production_config.json` is THIS tree's mirror; a child pinned to another commit is built by
that commit's registry, its flags and its own mirror — and `--arch` does not exist before
2026-09-06, so a pinned older argv could not even take the remedy the message offers. Refusing there
would be the same false POSITIVE that made `--pin-commit` unusable. The diff is still computed and
printed; only the gate is dropped.

**ONE function, four readers.** `main.train.arch_surface.report` serves this, `--dry-run`,
`python -m main.checkargs` and `resolve_config`'s (report-only) print. Three copies of a guard is
three things to keep in step, and this tree has paid for that shape twice already. The key set is
DERIVED from `flag_registry.arch_surface_flags()` — the `structural` × `family=arch` rows — never a
hand list, which would go stale the first time a toggle landed and then silently under-report.

**The compared-key count is RECONCILED, not merely smaller.** Every block prints
`<arch> arch + <critic> critic + <n> non-structural = <total> registry rows` (`arch_surface.surface_partition()`;
57 + 6 + 2 = 65 at HEAD on 2026-10-10, 39 + 7 + 3 = 49 when it landed 2026-09-06 — read it from the code), because a guard that compares fewer keys than a reader's own count leaves them unable
to tell an excluded row from a forgotten one. `family=critic` is excluded because the win-prob critic
IMPLIES one of those readouts and REFUSES two others, so gating them would refuse every critic arm.

**`--arch production` is the remedy**, and what it does NOT set it NAMES on every run: the critic
readouts the win-prob critic implies, and `--belief-grad-mode`. The win-prob critic (a constant now), its three reward
values and the SUPERVISION DOSES (`--move-belief-coef` and siblings) it used to only name are now
APPLIED by its recipe half (next section). Silence that reads as coverage is the same failure one
layer down. Measured against the incident's own config: of its 31 differing keys, 26 are refused on
the surface, the doses are applied and compared by the RECIPE surface, and the last is the enable
coefficient of a refused surface row — none can pass unmentioned.

## Is this the RECIPE you meant? — the RECIPE-SURFACE guard (K10(a))

The ARCH guard's twin, for the TRAINING RECIPE: five parser defaults (`--n-envs`, `--batch-size`,
`--n-epochs`, `--ent-coef`, `--clip-range-vf`) and more (`--grad-accum-steps` 1 vs 32, `--self-play`,
the critic, the reward values, the doses) differed from the live recipe, and `--arch production`
applied none of them. `main.train.recipe_surface` holds the declared rows; the values live in
`designs/production_config.json`'s `recipe` block: `recipe.fresh` (N0's measured fresh recipe) and
`recipe.fork` (E5). Spec, values and their sources:
`designs/endstate/design_learner_recipe.md` §3.22.

| the argv | what the guard does |
|---|---|
| **FRESH**, a differing knob the argv did NOT type | prints the RECIPE SURFACE block and **REFUSES** (`FATAL_CONFIG`), naming every untyped knob |
| FRESH, every differing knob TYPED | INFO — a typed value is the arm's lever |
| FRESH + `--allow-nonproduction-recipe` | prints the block and launches |
| FRESH + `--arch production` | applies every untyped `recipe.fresh` knob first (N0's measured recipe) |
| **FORK** | INFO, compared with `recipe.fork` (E5: 5 epochs at a frozen 5.6e-5) — a fork's recipe is its ARGV plus what it inherits from its parent's config |
| **same-run RESTART** of an `--arch production` run | the child resolves each untyped knob by ONE route (INERT / `model_config.json` / `metadata.json:cli_args`), announced; a MISSING value REFUSES by name (`checkargs` / `--dry-run` report it) |
| **PINNED to a non-HEAD commit** | ADVISORY, for the ARCH guard's reason |

The gate runs inside `_arch_surface_gate` (first, same last stop), and in `--dry-run` and
`python -m main.checkargs`, all from `main.checkargs.check`'s one `recipe_surface.report`. The doc
and the block are held together by `src/recipe_doc_gate_test.py`.

## The startup preflights — desktop GPU, disk space, the Showdown submodule

🚨 **THE DESKTOP-GPU REFUSAL (T23, 2026-10-05).** A CUDA run does not start while a display process (gnome-shell,
Xorg, Xwayland, a display manager, … — the declared `utils.desktop_gpu.DISPLAY_PROCESS_NAMES`) holds the GPU: the
trainer exits `FATAL_CONFIG` (the launcher does not restart it) naming the process, pid, VRAM and the fix
(`sudo systemctl stop gdm.service`). `--dry-run` calls the SAME `check_for_run` on the resolved `--device` /
`--debug` / `--allow-desktop-gpu` and prints a `desktop GPU :` line (✓ / exempt / tolerated / ✗ REFUSED; the refusal
fails the dry run, FATAL_CONFIG, and is ADVISORY when the child runs a pinned other commit, whose trainer may not
carry the check). `--debug` (CPU) is exempt; NVML unreadable is a refusal for a CUDA run. `--allow-desktop-gpu`
rides to the child verbatim (restarts keep it) and is recorded in `metadata.json` (`cli_args`). Gate:
`utils/desktop_gpu_test.py`, `dry_run_test.py` (h).

🚨 **THE DISK-SPACE GUARD (2026-10-09).** A launch whose run-archive filesystem has less free space than REQUIRED is
refused `FATAL_CONFIG` (exit 3) naming the shortfall and printing the arithmetic (`utils/disk_guard.py`: the checkpoints
still to be written x the checkpoint size, eval traces, the compile cache not yet on disk, log allowances, x 1.25 + a 4 GiB
reserve; the checkpoint size is read from the run's own newest checkpoint, the fork source, or the newest same-architecture
run). `main/launcher/disk_gate.py` asks in `_prepare_session` BEFORE the pin / worktree / run dir exist — the only guard a
PINNED child gets, its trainer predating this check — and `--dry-run` prints the same verdict (`disk space :` line); the
trainer asks again at its own startup (`main/train/disk_preflight.py`, before `os.makedirs`). `--allow-low-disk` is the
recorded opt-out (a trainer flag, forwarded verbatim; `metadata.json` `cli_args._disk_guard`); when the PINNED trainer has
no such flag the launcher CONSUMES it (`disk_gate.consume_if_absent_at_pin`) instead of failing the pinned-parser check.
In flight, `run_io._TrackingCheckpointCallback` reads `shutil.disk_usage` once per save: free < 2 x the next save warns
(`[DiskGuard] LOW DISK`), free < 1 x stops cleanly with `FATAL_DISK` (8, [`crashes_and_exit_codes.md`](crashes_and_exit_codes.md)); the opt-out keeps the
warning and drops the stop. `--debug` is exempt from the preflight. Gate: `utils/disk_guard_test.py`,
`main/train/disk_guard_wiring_test.py`, `main/launcher/disk_exit_test.py`, `dry_run_test.py` (j).

🚨 **THE SUBMODULE PREFLIGHT (2026-10-10, F-GE-4 of `designs/research_state/measurements/gpu_checks_endstate_2026-10-09/`).**
`_create_run_worktree` replaces the pinned worktree's empty `deps/pokemon-showdown` placeholder with a LINK to the
LAUNCHING checkout's submodule (`worktree.showdown_link_source(repo_root)`), and an un-pinned child reads the launching
tree's own. A launch from a checkout whose submodule was never initialised or built therefore gave the child an empty
directory, and it died minutes later in `Teambuilder`'s team validation (`utils/bridge/validate_team.js` does
`require('<repo>/deps/pokemon-showdown')`; `team_validator` swallows `Cannot find module` into `{"valid": False}`, so the
visible symptom is "No valid teams found" after the model and the T2 service are up). `main/launcher/submodule_gate.py`
asks BEFORE the pin / worktree / run dir exist, in `_prepare_session` right after the disk question, and `--dry-run` prints
the same verdict (`showdown deps :` line, fails the dry run): `utils/showdown_deps.py` `NEEDS` — the closed list of files
`validate_teams_locally` reads (sentinels of the three things that can be wrong: the submodule `checkout`, its `build`
(`dist/`), its `modules` (`node_modules/`); `os.path.exists` follows links, so a dangling `dist` link reads as missing) —
must be present and `node` on PATH, else `FATAL_CONFIG` (3) naming the missing files, the checkout and the fix:
`git submodule update --init` (only when `package.json` is missing), then `./scripts/bootstrap.sh --skip-env` (links the
main checkout's built `dist/` and `node_modules/` into a worktree; leaves the shared conda env alone); or `node build` /
`npm ci` in the submodule. Never advisory (a pinned trainer predates any check of its own) and NO opt-out flag. Pinned by
`submodule_gate_test.py` (the launcher refuses before a worktree or run dir exists, for a placeholder and an unbuilt
checkout; the healthy path; `--no-pin` asks about its own tree; `--dry-run` fails/passes; the checked directory IS the
one a real pinned worktree is linked to) and `utils/showdown_deps_test.py` (the verdict, the fix text, and a REAL node
trace of one validation proving every listed file is read). A test that fakes the launcher's repo root with a throwaway
repository writes a stub checkout (`showdown_deps.write_stub_checkout`) after its commits.
