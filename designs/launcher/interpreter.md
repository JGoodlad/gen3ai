# Which interpreter the child runs

> Lifted out of `src/main/launcher/CLAUDE.md` (2026-10-10). Always-current: update it with the code.

## The recorded torch decides

🚨 **A resume or fork runs under the TORCH its run RECORDED** (`torch_runtime.py`, owner 2026-09-30:
torch 2.8 / `gen3ai_torch28` is the default, torch 2.5.1 / `gen3ai_stable` is LEGACY, kept so old
pinned runs resume on their own torch). Every save writes `metadata.json`'s `torch_version` (and a
`torch` key on each `pin_history` span); a run with NO record predates it and **reads as 2.5.1**.
`resolve_for_launch` decides ONCE per launcher session, after the fork swap and before anything is
created, and the choice rides in `child_env[$GEN3AI_PYTHON]` so every restart spawns the same one:

| launch | child interpreter |
|---|---|
| FRESH | `resolve_child_python()` (below) — its torch becomes the run's record at the first save |
| RESUME / FORK, the default interpreter carries the recorded torch | that interpreter |
| … it does not, no `$GEN3AI_PYTHON` | the sibling conda env `KNOWN_ENVS` names for that torch (`2.5.1` → `gen3ai_stable`, `2.8.0` → `gen3ai_torch28`; found beside the launcher's own env, never a machine path), **SELECTED and announced** |
| … nothing carries it, or `$GEN3AI_PYTHON` names the wrong torch | **`FATAL_CONFIG` refusal**, unless `--allow-torch-switch` (launcher-owned consent; the new torch is then recorded at the child's first save) |

Versions compare by RELEASE (`2.5.1+cu121` ≡ `2.5.1`). The startup event and `--dry-run` print the
interpreter, its torch and the run's recorded torch with its source. The probe reads the env's torch
METADATA in a subprocess (~50 ms; torch is not imported). Gate: `torch_runtime_test.py` (fake
interpreters, so it reads the same on any box; reverting the refusal or the `child_env` hold fails
it). ⚠️ The guard is the LAUNCHER's: a bare `train_rl_agent.py --model …` is not checked by it.
🚨 **HEAD's CODE runs torch >= 2.8 only** (deletion pass K1, 2026-10-02): the trainer's `main()` calls
`utils/torch_floor.py` first and exits `FATAL_CONFIG` on an older torch (HEAD has no 2.5.1 path left).
So a 2.5.1 run resumes only PINNED (its own commit's `src/`, the selected `gen3ai_stable`): with
`--no-pin` / `--sync-to-main` the child is HEAD code on 2.5.1 and refuses at startup — drop the flag,
or switch the run to 2.8 with `--allow-torch-switch`.


Underneath, `child.resolve_child_python()` — the FRESH default — in precedence order:

| # | Source | Notes |
|---|---|---|
| 1 | **`$GEN3AI_PYTHON`** | Explicit override. Set it only to run the child under a *different* interpreter than the launcher — a blank/whitespace value falls through rather than becoming `argv[0]` |
| 2 | **`sys.executable`** | The default: the launcher's OWN interpreter |

**`sys.executable` is the correct default, not a guess.** The launcher is already running under the
environment the run wants, so the child inherits it on any machine under any env name — there is no
conda prefix, env name or absolute path to keep in sync, and a fresh clone needs no source edit.
The resolved value is announced in the events panel at startup (`🐍 Interpreter: …`, marked
`(pinned by $GEN3AI_PYTHON)` when the override is live), because if the launcher was started from
the wrong environment then *every* child inherits that, and this line is where it shows.

`_launch_child` takes argv[0] from the session's `child_env[$GEN3AI_PYTHON]` (set by
`resolve_for_launch` above), falling back to `resolve_child_python()` at spawn time for a caller that
set none (tests), so a launcher that outlives a dozen children across restarts spawns one torch.

> **History.** This was a hardcoded `/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3` with
> no flag and no override until 2026-08-22 — a fresh clone died with `FileNotFoundError` on its
> first launcher run and the only fix was editing the source. On this box the change is
> behaviour-identical (the launcher *is* started with that interpreter, so `sys.executable` resolves
> to it). Gate: `interpreter_test.py`, whose durable half fails if **any** launcher module
> re-introduces a machine-specific path — not just the old line.

**Recorded commands are unaffected.** `run.py` records `LAUNCHER_COMMAND = " ".join(sys.argv)`, and
`sys.argv[0]` is the launcher's `__main__.py`, never the interpreter — verified over all 104
archived `models/*/metadata.json` (0 embed a python or conda path). The launcher constructs the
child argv itself, so an old run's recorded command relaunches unchanged.
