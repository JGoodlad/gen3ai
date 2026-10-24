# The launcher's flags

> Lifted out of `src/main/launcher/CLAUDE.md` (2026-10-10). Always-current: update it with the code.
> This is the launcher's one flag table (the leaf keeps none); `restart_interval_default_test.py`
> reads its `--restart-interval-hours` row. `designs/ops/flag_census.md` names each flag's live user.
> The second table (below) classes EVERY flag, the launcher's and the trainer's, for a `--model` launch;
> its source is `src/main/train/launch_flags.py`, pinned flag-by-flag by `launch_flags_test.py`.

## The table

| Flag | Default | Notes |
|------|---------|-------|
| `--restart-interval-hours` | `6.0` | Set to `0` for a single run with no restart. Default `run.DEFAULT_RESTART_INTERVAL_HOURS` = 6 h (owner rule 2026-10-04, in code 2026-10-10; it was `3.0`). A launch that TYPES the flag is unaffected; one that omits it now restarts every 6 h. Pinned by `restart_interval_default_test.py`. |
| `--max-crash-restarts` | `3` | Consecutive rapid self-crashes (each < 10 min after launch) to auto-restart through before giving up. `0` = unlimited. A crash after sustained progress resets the counter (see [`crashes_and_exit_codes.md`](crashes_and_exit_codes.md)). |
| `--restart-grace-minutes` | `20.0` | Force-kill window after a scheduled restart's deadline (child overran its rollout boundary). A child that ignores the SIGTERM is SIGKILL'd after a 150 s grace (`KILL_GRACE_SECONDS`), and a launcher-forced kill restarts from the last checkpoint (not treated as a fatal crash). The dashboard shows `⚠ no child output for Nm` once the child has been silent > 2 min, so a stall is *visible* — but there is no auto-restart on stall. (The websocket disconnect / `race_get` stall guards this row used to describe lived in the poke-env env and were deleted with it, T27 P6; training is in-process on the Rust core and opens no connection.) |
| `--nice` | `10` | Scheduling niceness for the launcher **and everything it spawns** — `0` disables. Applied in `run.main()` before any child exists (`run._apply_nice`, default `run.DEFAULT_NICE`); niceness is inherited across fork/exec, so the training child and every process it spawns are covered without per-spawn wiring — including the processes created by later periodic and crash restarts. It only ever **raises** niceness: a negative target needs `CAP_SYS_NICE`, so it no-ops rather than failing. **On an idle box this changes nothing** — niceness only arbitrates under contention. Why it defaults on: at nice 0 a run it competes on equal terms with interactive work sharing the box (measured 2026-08-13, on the since-deleted Python env core's ~940-process runs, at load 17–25 on 16 cores: an interactive client in the *same cgroup* as the run waited 2.1 s in the run queue per 1 s of CPU it received, and every training process sat at nice 0). Note the limit of the mechanism: nice arbitrates **within** a cgroup, so a client in its own systemd scope is already protected by cgroup `cpu.weight` and gains little — the flag's value is for whatever shares the run's own scope. Gate: `nice_test.py` (including the inheritance test — without it the workers silently revert to nice 0 and nothing else would notice). |
| `--no-pin` | off | Skip worktree creation; run from the current source tree |
| `--sync-to-main` | off | When resuming from a checkpoint, pin the isolated worktree to the current HEAD instead of the checkpoint's original git hash. Use this to pick up UI or tooling fixes on `main` without discarding the checkpoint. |
| `--pin-commit COMMIT` | unset | **Pin the isolated worktree to a NAMED commit** (full sha or unambiguous prefix — resolved with `git rev-parse --verify <spec>^{commit}` and announced at startup as the full sha plus its subject line). Spelled `--pin-to-hash` before 2026-09-05; both spellings still parse, `--pin-commit` is the name. Beats the checkpoint's recorded `git_hash` on a genuine FORK and HEAD on a fresh run; **refused** beside `--sync-to-main` (argparse — they name two different sources of truth) and beside `--no-pin`; **refused** on a same-run RESTART whose checkpoint records a different hash (see [`pinning_and_worktrees.md`](pinning_and_worktrees.md)). An unresolvable commit exits `FATAL_CONFIG` naming it — never a silent fall-back to HEAD, which is the whole failure it exists to prevent. |
| `--allow-torch-switch` | off | **Consent** to resume/fork a run under a torch OTHER than the one its run recorded (`metadata.json` `torch_version`; unrecorded = 2.5.1). Without it the launcher selects the env carrying the recorded torch, or refuses `FATAL_CONFIG` — see [`interpreter.md`](interpreter.md). Launcher-owned (stripped). |
| `--gpu-window` | off | **Honour GPU-WINDOW requests for this run** (T46): `python -m main.launcher.gpu_window arm <run_dir> --precheck …` pauses the run at its next safe point at an absolute epoch, a guest job takes the GPU by file handoff, and the run re-launches (same pin, same command, RESUME role) on the guest's DONE; any mid-window failure HOLDS. Off unless typed: a launch without it never reads a request. Launcher-owned (stripped). See [`gpu_window.md`](gpu_window.md). |
| `--arch production` | unset | *(forwarded)* Apply the whole ARCH surface AND the TRAINING RECIPE from `designs/production_config.json` as if typed — see [`launch_guards.md`](launch_guards.md) → the ARCH-SURFACE and RECIPE-SURFACE guards. Refused on a resume (`FATAL_CONFIG`); the launcher's own restarts STRIP it (resume role). |
| `--allow-nonproduction-arch` | off | *(forwarded)* Consent to a FRESH run whose architecture differs from the production mirror; without it that launch is REFUSED. |
| `--allow-nonproduction-recipe` | off | *(forwarded)* Consent to a FRESH run whose training recipe differs from the mirror's `recipe` block on a knob the argv did NOT type; without it that launch is REFUSED. A TYPED differing value never needs it. |
| `--dry-run` | off | **Resolve this launch and PRINT it, then exit — creating nothing.** Role (FRESH / FORK of <parent> / RESTART of <run>), the run dir the argv would write into, the pin (sha + subject + source), `--steps` beside the checkpoint's recorded `num_timesteps` so `+X steps` is visible, the effective config a `--model` inherits (per flag: `TYPED`, `RESTORED from <record>`, `INHERITED` or `default`; the K9(b) mode on its own `K9(b) check :` line), the pool as recorded, a `--model` argv's architecture verdict against the checkpoint's `model_config.json` (`main.train.fork_arch`, the trainer's own `check_compatible`), and a `(child-only: …)` line for everything that needs a built model. Exits `0`, or `FATAL_CONFIG` (3) on any refusal the real path makes. See [`launch_guards.md`](launch_guards.md) → `--dry-run`. |

All other flags are forwarded verbatim to `train_rl_agent.py` (the launcher strips only
launcher-owned flags).

## What a `--model` launch carries over: every flag, once

**An argv is not a config.** With `--model`, every flag the argv does not TYPE is carried over from the run that wrote
the checkpoint: the run itself on a same-run restart, the PARENT on a fork. Each flag below is in exactly ONE class
(`launch_flags_test.py` fails on a flag in none or two), and a typed flag always wins, even one equal to the default.
The resolver is `recipe_surface.inherit_recorded_recipe` (checkargs, `--dry-run`, `resolve_config` and the launcher
all call it); `_prepare_session` writes what it restored into the child argv TYPED (`materialize_recorded_recipe`),
because a child pinned to an older commit resolves an untyped flag to that commit's own default. A bool is written
BARE (`--flag` / `--no-flag`, omitted when it is the parser's own value): `--flag true` is refused by a pin whose parser
spelled the flag `store_true` or `BooleanOptionalAction` (found on the real dry run of an X35 arm, pin `440b61c3`).

| Class | Restart | Fork | Flags |
|---|---|---|---|
| **recipe row** (`recipe_surface.ROWS`) | yes | yes | `--n-envs --n-epochs --grad-accum-steps --ent-coef --vf-coef --clip-range(-vf) --min-lr --max-lr --anneal-lr-start-steps --weight-decay --self-play --policy-gae-lambda --trunk-precision --overlap-collect --fork-fraction` and the belief / intent doses, the T2 and collector sizes; `--lr --batch-size --n-steps` are INERT (SB3 restores them). A missing value REFUSES by name |
| **recorded field** (a `ModelVersion` tri-state flag) | yes | yes | every structural / numerics / diagnostic flag `config.resolve_config` resolves by a `_resolve` line, plus `--promote-threshold` and `--eval-sentinel-greedy` (`resolve_eval_sentinel_regime`): from the checkpoint's `model_config.json` |
| **RECORDED row** | yes | yes | safety: `--behaviour-check --supply-starve-cycles --rust-env-refusal-budget --rust-env-respawn-budget`; cadence: `--checkpoint-every-steps --eval-freq --eval-games --eval-shard-games --n-sentinels --snapshot-ladder-games`; retention: `--keep-eval-snapshots --keep-eval-trace-steps --keep-stalls --keep-crashes --forensic-{win,loss,draw}-quota`; telemetry: `--value-sidecar(-fraction,-seed) --team-wr-tracking`; learner: `--anneal-min-lr --adaptive-batch(-target,-band,-min-accum,-max-accum,-every) --opp-belief-moves-weight`; self-play schedule: `--self-play-{temp,start-wr,full-wr} --heuristic-floor --pfsp-scale --pool-spread --team-block-episodes`. A key the record lacks (the run predates the flag) or holds as None stays untyped, never an error |
| **RESTART-ONLY row** | yes | **no** | `--seed`, `--trainee-team(s)`, `--bot-weights`, `--stable-opponents` and its four knobs. A fork is a new experiment that declares its own seed and matchup; the report names what it did not carry. `--team-mix(-sides)` follows the same rule through `matchup_setup.inherit_team_mix` |
| **PER-LAUNCH** (`launch_flags.PER_LAUNCH`, a reason each) | no | no | identity and dose: `--model --run-dir --run-name --steps`; process: `--debug --debug-eval --device --log-level`; fresh-only sugar: `--arch --unified-moves --unified-damage --damage-matrices`; consents: every `--allow-*` and `--diagnostic-fork-consent`; fork acts: `--fork-lr(-freeze) --grow --fork-pool-seed --tb-inherit --exploiter(-keep-bots,-bot-fraction)`; host-bound performance: `--compile-trainer --learner-cudagraphs --device-batch --grad-checkpointing --t2-* --rust-env-{front,profile,threads} --rust-eval-envs`; the launcher's own session: every flag in the first table above |

The dry run prints each of `--grad-accum-steps --checkpoint-every-steps --eval-freq --eval-games --seed` with its source,
and the K9(b) mode with its own (`typed` / `inherited from <record>` / `the default`). Gap closed 2026-10-24: only the
recipe rows were restored, so a manual resume of an arm recorded `--behaviour-check warn --checkpoint-every-steps
1000000` ran K9(b) FATAL with saves every 2.4M. The launcher's own restarts were never affected: they re-launch the
session's child argv, which kept every flag the original launch typed.

## The launcher owns NO compile default, and forwards everything else

**The launcher owns NO compile default.** `--compile-trainer` defaults ON in `train_rl_agent`'s own parser (2026-08-17; the compile-opponents flags were deleted in U3), and the launcher's
only job is to be transparent to it and to its `--no-` opt-out. Two ways it could stop being:
`_strip_launcher_args` could grow an entry that eats one, or argparse could abbreviation-match an
unknown token against a launcher flag (it parses with `parse_known_args`, and `--no-pin` lives right
next to `--no-compile-*` — the launcher parser and the trainer's are `allow_abbrev=False` since deletion pass
P11, `main/train/parser_abbrev_test.py`, so an abbreviation is refused instead of matched).
`compile_flag_forwarding_test.py` pins both against the REAL parser —
`build_launcher_parser()` was extracted from `main()` for exactly that, so the test interrogates the
parser rather than a hand-copied twin. It catches the launcher silently swallowing a child flag
(its sibling `default_port_test.py`, which caught a launcher-injected default drifting from the
trainer's, went with the injection in deletion pass P11).
