# The launcher's flags

> Lifted out of `src/main/launcher/CLAUDE.md` (2026-10-10). Always-current: update it with the code.
> The leaf keeps a one-line-per-flag table (and the `--restart-interval-hours` row that
> `restart_interval_default_test.py` reads); this is the full table. `designs/ops/flag_census.md` names each flag's live user.

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
| `--arch production` | unset | *(forwarded)* Apply the whole ARCH surface AND the TRAINING RECIPE from `designs/production_config.json` as if typed — see [`launch_guards.md`](launch_guards.md) → the ARCH-SURFACE and RECIPE-SURFACE guards. Refused on a resume (`FATAL_CONFIG`); the launcher's own restarts STRIP it (resume role). |
| `--allow-nonproduction-arch` | off | *(forwarded)* Consent to a FRESH run whose architecture differs from the production mirror; without it that launch is REFUSED. |
| `--allow-nonproduction-recipe` | off | *(forwarded)* Consent to a FRESH run whose training recipe differs from the mirror's `recipe` block on a knob the argv did NOT type; without it that launch is REFUSED. A TYPED differing value never needs it. |
| `--dry-run` | off | **Resolve this launch and PRINT it, then exit — creating nothing.** Role (FRESH / FORK of <parent> / RESTART of <run>), the run dir the argv would write into, the pin (sha + subject + source), `--steps` beside the checkpoint's recorded `num_timesteps` so `+X steps` is visible, the effective config a `--model` inherits (per-flag `INHERITED` vs `from the argv`), the pool as recorded, and a `(child-only: …)` line for everything that needs torch. Exits `0`, or `FATAL_CONFIG` (3) on any refusal the real path makes. See [`launch_guards.md`](launch_guards.md) → `--dry-run`. |

All other flags are forwarded verbatim to `train_rl_agent.py` (the launcher strips only
launcher-owned flags).

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
