# CLAUDE.md — Gen3AI Project Guide

## Development Stage

**Rapid iteration — checkpoint compatibility is not a concern.** Breaking changes to the observation space, network architecture, or action space are fine. Do not add backwards-compatibility shims or hesitate to change dims, layer sizes, or layouts.

Architecture constants (embedding dims, layer sizes, etc.) are defined as module-level constants in `src/agents/model/features_extractor.py` — that is the single source of truth. When you change one, change it there and nowhere else. See [Model Versioning](#model-versioning).

## Documentation Maintenance

Keep docs in sync **automatically, as part of the same change** — no need to be asked:

- **Every `CLAUDE.md`** (root, `src/agents/model/`, `designs/`, anywhere): always current. If a change makes one stale, fix it in the same pass.
- **Every `README.md`** (including `designs/ai_v3/README.md`, the architecture digraph + dimension reference): always current. When you change dims, layout, obs/architecture, or anything a README documents, update it without being prompted.

**Do NOT auto-update other docs under `designs/`** — `impl_step*.md`, `design_*.md`, `todo.md`, etc. are explicit-only. Touch them only when the user asks (directly or via `/gen3ai-update-design-docs`). The lone exception is `CLAUDE.md` files inside `designs/`, which follow the always-current rule above.

## Git Workflow

This is a personal project — no pull requests needed. Work is pushed directly to `main`, but **all edits and commits must happen in a worktree or branch, never on the main checkout itself.**

Main must never be in a dirty state. The `/gen3ai-ship` skill is the only mechanism that lands code on main — it commits in the worktree branch and pushes:

```bash
git push origin <worktree-branch>:main
```

Never `git add` or `git commit` from `/home/goodlad/dev/gen3ai` directly.

**NEVER run `git add`, `git commit`, or `git push` unless the user's current message explicitly contains `/gen3ai-ship`.** Completing a task, writing tests, or any other finishing signal is NOT permission to commit. This applies even when the task feels "done". Only `/gen3ai-ship` authorises a commit.

---

## Python Environment

The project uses a dedicated conda environment, **not** `deps/venv`. Always prefix commands with the correct interpreter and `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 <script>
```

The conda env is `gen3ai_stable`. `deps/venv` exists but is outdated — ignore it.

---

## Git Worktree Setup

When opening a new git worktree (e.g. via Claude Code), the `deps/pokemon-showdown` submodule directory is created but left empty. Two steps are required before training or running tests:

**Step 1 — initialize the submodule** (gets source files, fixes VS Code git integration):
```bash
git submodule update --init
```

**Step 2 — symlink the build artifacts** (the submodule checkout has no compiled `dist/` or installed `node_modules/`, but the main repo already has them):
```bash
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/dist \
      deps/pokemon-showdown/dist
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/node_modules \
      deps/pokemon-showdown/node_modules
```

Both `dist/` and `node_modules/` are gitignored in pokemon-showdown, so these symlinks don't affect `git status`. Without step 2, training fails with `Cannot find module '.../dist/sim/index.js'`.

Do **not** symlink the entire `deps/pokemon-showdown` directory — git treats the submodule path as a symlink rather than a real checkout, which breaks `git status` and VS Code's git integration.

---

## Running Tests

### Test file naming conventions

| Pattern | Requires | Marker |
|---|---|---|
| `*_test.py` | Nothing — pure unit tests with mocks | — |
| `*_integration_test.py` | `deps/pokemon-showdown` Node bridge (no battles, no live server) | `@pytest.mark.integration` |
| `*_fuzz_test.py` | `deps/pokemon-showdown` — runs **real battles in-process via the local BattleStream bridge** (`utils/bridge/local_battle_runner.py`); **no live server**. The default for fuzzing. | none — run directly as scripts (no `test_*` funcs, so `pytest` imports but collects nothing) |
| `*_fuzz_e2e_test.py` | A **live Showdown server** — fuzz whose checks need real async-server timing (e.g. `effectiveness_fuzz_e2e_test`, whose TurnDelta-vs-BattleContext effectiveness window is decision-timing-sensitive) | run directly as scripts |
| `*_e2e_test.py` | A **live Showdown server** on localhost:8000 | `@pytest.mark.e2e` (scripts only, run directly) |
| `*_benchmark.py` | `deps/pokemon-showdown` bridge (no live server) — **performance profiling, not pass/fail**: plays a real battle in-process, then `cProfile`s a hot path | none — run directly as scripts (no `test_*` funcs → `pytest` collects nothing). Place in a dir with no stdlib-shadowing names (e.g. `training/`, not `observation/`) |

### Unit tests only (default)
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not integration and not e2e" -q
```

### Unit + integration (requires symlinked deps/pokemon-showdown)
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -q
```

### Fuzz tests (`*_fuzz_test.py`, run directly as scripts)
Run battles **in-process via the local BattleStream bridge — no `npm run showdown`
needed** (`utils/bridge/local_battle_runner.py`):
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/fuzz_test.py [n_battles]
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/transition_fuzz_test.py [n_battles]
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/battle/event_log_fuzz_test.py [n_battles]
# also bridge-backed (no server): poke_env_gaps/{abilities,item_consumption,move_outcome}_fuzz_test.py
#                                  and training/hidden_power_tracker_fuzz_test.py
```

### E2E tests (`*_e2e_test.py` / `*_fuzz_e2e_test.py`, require a live server)
```bash
# Start server first: npm run showdown
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/telemetry_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/effectiveness_fuzz_e2e_test.py [n_battles]
```

### Benchmarks (`*_benchmark.py`, run directly as scripts)
Profile a hot path on a real bridge battle (no server). `obs_build_benchmark.py` plays until a
representative late-game decision, then reports a component wall-clock breakdown
(`state_encoder.encode` vs deque-cached turn-history vs `live_view()`) plus a `cProfile`
`tottime` ranking — use it to catch obs-pipeline regressions and confirm an optimization moved
the bottleneck:
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/obs_build_benchmark.py [--turn 25] [--reps 400] [--top 22] [--battles 200] [--seed 0]
```
Absolute ms scale with machine load; the component **ratios** and the cProfile ranking are the
load-stable signal — run on an otherwise-idle box for a clean baseline.

**Every change under `src/agents/observation/` must run this benchmark before/after and
confirm no meaningful regression** — that gate, the canonical baseline, and the
load-stable regression criteria live in `src/agents/observation/CLAUDE.md`.

For the **top-down** view — where a whole trainer turn's CPU goes (parse + obs + reward +
mask + map + tracker), GPU-excluded and server-free — use `trainer_turn_benchmark.py`. It
walks a real bridge battle, times every per-decision CPU stage `Gen3Env` runs (a random legal
action stands in for the policy forward), and prints a stacked breakdown. Baseline: obs build
≈ 88% of our CPU (`state_encoder.encode` ≈ 80%), parse ≈ 7%, reward ≈ 4%, everything else <1%.
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/trainer_turn_benchmark.py [--decisions 150] [--warmup 3] [--seed 0]
```

### What "fuzz test" means in this project

**Fuzz tests run real battles — by default in-process via the local BattleStream bridge (no server), or against a live server — and validate observations or behaviour against the actual protocol stream.** They are NOT deterministic scenario tests with fixed inputs.

The canonical pattern (see `src/agents/training/poke_env_gaps/`):

1. Subclass `Player` and override `_handle_battle_message` to intercept raw Showdown protocol lines mid-battle.
2. Archive per-turn snapshots of the state you care about (e.g. which items were consumed, which moves were used).
3. In `choose_move()`, validate that the encoded observation vector matches what the archived protocol events say should be there.
4. Run N random battles; any validation failure raises immediately with a detailed error.

This catches poke-env parsing bugs and encoder gaps that unit tests with mocks cannot — the test exercises the Showdown sim → poke-env → encoder pipeline end to end (the bridge feeds the identical protocol stream the live server would). When asked to write a fuzz test, always follow this pattern rather than writing parametrized unit tests with hand-crafted mock objects.

---

## Smoke Test

Before a full training run, verify the full pipeline (env, reward, replay callback, stall detection, evaluation) with a quick debug run (~1 min):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug --steps 10000
```

What to look for:
- `[ModelVersion] Round-trip smoke test PASSED` — serialization and reload are healthy (printed early, before training begins)
- `🏁 Episode Finished` lines appearing throughout — episodes completing and resetting
- `[STALL LOGGED]` may appear if a 250-turn game occurs — should be followed by another `🏁 Episode Finished`, not a hang
- `Win rate vs Random` and `Win rate vs Heuristic` printed at the end — evaluation ran

A hang after `[STALL LOGGED]` or a crash before "Training complete" indicates a regression in the env/stall/forfeit pipeline. A `[ModelVersion] FATAL` error at startup means the checkpoint was saved with a different architecture than the current code.

---

## Launcher (preferred for long runs)

`src/main/launcher/` wraps `train_rl_agent.py` with:
- **Periodic restarts** — kills and relaunches the child every N hours to reclaim pymalloc fragmentation; the child saves a checkpoint on SIGTERM and the launcher picks it up automatically
- **Worktree isolation** — at startup, creates a detached git worktree pinned to the current HEAD (or to the commit recorded in the checkpoint's `metadata.json` when resuming). Agent pushes to `main` never affect a running session
- **Rich TUI** — live dashboard showing metrics, FPS, restart countdown; `l` for logs, `r` to restart now, `c` for forced checkpoint, `q` to quit cleanly
- **Crash reporting** — child stdout/stderr is streamed live to `<run_dir>/launcher_child.log` (complete even if the child hard-`os._exit`s, bypassing Python cleanup) and held in a 5000-line in-memory scrollback. On a non-zero exit the last 100 lines are dumped to the terminal after the TUI closes; on *every* exit (crash, complete, quit) the full log path is printed and the file is finalized (the in-memory buffer is flushed to it as a fallback if streaming never started)

### Exit codes (`src/main/exit_codes.py`)

| Code | `TrainExitCode` | Meaning |
|------|----------------|---------|
| 0 | `COMPLETE` | All steps done — launcher stops |
| 15 | `INTERRUPTED` | SIGTERM received, checkpoint saved — launcher restarts |
| 1 | `CRASH` | Unhandled exception — launcher stops, crash log printed |

### Starting a fresh run via launcher
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --steps 15000000 \
  --n-envs 64 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

### Resuming from a checkpoint
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --model models/<run>/checkpoint_NNNN_steps.zip \
  --steps 15000000 \
  --device cuda
```

The checkpoint must have a `metadata.json` with a `git_hash` field (written automatically by `save_model_snapshot()`). The launcher pins the worktree to that exact commit so the resumed run uses the same code as the original.

### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--restart-interval-hours` | `3.0` | Set to `0` for a single run with no restart |
| `--restart-grace-minutes` | `20.0` | Force-kill window after a scheduled restart's deadline (child overran its rollout boundary). A child that ignores the SIGTERM is SIGKILL'd after a 90 s grace, and a launcher-forced kill restarts from the last checkpoint (not treated as a fatal crash). The dashboard shows `⚠ no child output for Nm` once the child has been silent > 2 min, so a stall is *visible* — but there is no auto-restart on stall. A connection failure crashes loudly rather than hanging: a connect-time failure via the `Gen3Player` connect guard, and a **mid-battle** websocket drop via the `_AsyncQueue` disconnect guard (`PSClient.listen()` sets `_disconnected` on an abnormal close; the env's blocking `step`/`reset` get races against it and raises `ShowdownException` instead of waiting forever for a message the dead listen task can't deliver). Both deterministic — no timeout guess. Other hangs are surfaced for investigation rather than guessed at with a timeout. |
| `--no-pin` | off | Skip worktree creation; run from the current source tree |
| `--sync-to-main` | off | When resuming from a checkpoint, pin the isolated worktree to the current HEAD instead of the checkpoint's original git hash. Use this to pick up UI or tooling fixes on `main` without discarding the checkpoint. |

All other flags are forwarded verbatim to `train_rl_agent.py`.

---

## Training

Run directly (no restart loop, no worktree isolation):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --model <path/to/checkpoint.zip> \
  --steps 15000000 \
  --n-envs 64 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

Omit `--model` to start a fresh run. Use `--debug` for a single env (DummyVecEnv). Use `--device cpu` on machines without a GPU.

Checkpoints are saved automatically. Models land in `models/run_<timestamp>/`.

### Bot evaluation (subprocess, non-blocking)

`PerOpponentEvalCallback` (non-self-play path) does **not** eval in-process. On each
scheduled step it snapshots the live weights (`model.save`) and spawns `--eval-workers`
(default 3) `main.eval_worker` subprocesses that **work-steal** opponents from a shared
pool (atomic `O_EXCL` claim files — a worker that finishes a cheap opponent grabs the
next, so uneven per-opponent cost self-balances), load the **frozen** snapshot, and play
against the shared Showdown server **without pausing training**. Each opponent writes
`result__<opponent>.json`; when all workers finish the parent merges them → TensorBoard +
TUI + best-model (the winning snapshot is promoted by copy, not re-saved). Forensic traces
land under `<run_dir>/eval_traces/step_<N>/<opponent>/`. The eval summary itself is
written to `metadata.json` as a **top-level `latest_eval`** block (step-labeled, NOT
nested under a checkpoint) — robust to the async timing (an eval can finish after a
newer checkpoint, or before any checkpoint exists); `save_model_snapshot` carries it
forward so a later checkpoint never erases it.

The frozen snapshot makes parallel eval correct (a worker can't read mutating in-memory
weights), and the fresh process returns all eval memory to the OS on exit (no fragmentation
in the trainer). Behaviors:
- A trigger that fires while the previous cycle still runs is **skipped** (logged) — on CPU
  an eval can outlast its interval; cadence just goes sparser.
- A worker crash is **logged-and-continued**, never fatal (its opponents are just missing
  for that cycle).
- **Graceful shutdown waits for eval to finish**: a scheduled restart is self-initiated by
  `GracefulRestartCallback` at a rollout boundary and the launcher won't force-kill until the
  child overruns the deadline by `--restart-grace-minutes` (20 min), so the drain budget is a
  full `_ABORT_EVAL_DRAIN_SEC` (10 min) AFTER the checkpoint is saved — long enough for a CPU
  eval to complete. Even the pathological forced-SIGTERM case (already overran → ~90s SIGKILL)
  is safe: the checkpoint is saved first, only the in-flight eval can be lost.
- **On resume the last eval is re-published to the TUI** from the resumed checkpoint's
  `metadata.json`, so the eval panel isn't blank until the next cycle.

| Flag | Default | Notes |
|------|---------|-------|
| `--eval-workers` | `3` | Eval subprocesses per cycle; work-steal opponents from a shared pool. Capped at the opponent count. |
| `--eval-device` | `cpu` | Device for eval-worker inference. `cpu` decouples eval from the training GPU. |

Eval concurrency in the worker is `_EVAL_SUBPROCESS_CONCURRENCY` (5/opponent) — low so
the shared server isn't flooded while training also uses it.

---

## Playing / Evaluation

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/play.py
```

Requires the Showdown server to be running (see below).

---

## Showdown Server

> **⚠️ Never stop or restart the training Showdown server on port 8001.** A server started
> manually from a bash shell on **port 8001** (`npm run showdown -- 8001`) is the dedicated
> **training** server consumed by `main.launcher --showdown-port=8001`. Claude must NOT stop,
> restart, SIGTERM/SIGKILL, or `npm run stop -- 8001` it — and must not bounce it as a side
> effect of any other task — **unless the user explicitly asks in their current message.**
> Killing it mid-run drops every poke-env websocket at once (`ConnectionClosedError: no close
> frame received or sent` in `ps_client.listen`) and crashes training.
>
> **If Claude needs its own server, bind it to a unique port in the `9XXX` range** (e.g. 9001)
> that no other agent/session uses — never 8000 (dev) or 8001 (training). Pass it through
> (`npm run showdown -- 9001`, `--showdown-port=9001`). **Only ever kill the process on the
> port you started** — never `npm run stop` (no arg → kills :8000), never `npm run stop --
> 8001`, never a blanket node/showdown kill. Prefer the in-process bridge (no server) for
> throwaway work entirely.

Start the local Pokémon Showdown instance:

```bash
npm run showdown            # port 8000 (default)
npm run showdown -- 8001    # explicit port
```

Stop it:

```bash
npm run stop                # stops the :8000 instance
npm run stop -- 8001        # stops the :8001 instance
```

The server must be running for any battles (training or play). It binds to port 8000 by
default. Pokémon Showdown has **no `--port` flag** — the port is a positional argument
(`pokemon-showdown start [PORT]`); `npm run showdown -- 8001` forwards it (npm appends
`-- <args>` to the script).

### Separate training port

To run training without clashing with a development server on 8000, start the server on a
separate port and point the trainer at it with `--showdown-port`:

```bash
npm run showdown -- 8001                            # server on 8001 (dev stays on 8000)
... -m main.launcher ...                            # launcher DEFAULTS to :8001 (see below)
... -m main.launcher --showdown-port 8123 ...       # explicit port still wins
npm run stop -- 8001                                # kills the 8001 instance
```

**The launcher defaults `--showdown-port` to 8001** (`DEFAULT_TRAINING_SHOWDOWN_PORT` in
`launcher/checkpoint.py`, injected in `launcher/__init__.main()` via
`_apply_default_showdown_port`) — a long launcher session must not ride on the shared dev
server (8000), where a routine dev restart / `npm run stop` drops *every* worker's
connection at once and the connection guard then crashes the whole run. An explicit
`--showdown-port` (any spelling) always wins, and the resolved port is shown in the TUI
events panel (`🔌 Showdown server :8001`). Note this default lives **only** in the launcher;
`train_rl_agent.py` run directly still defaults to 8000 (ad-hoc runs against the dev server).

`train_rl_agent.py --showdown-port <port>` builds **one** `ServerConfiguration` in `main()`
via the single constructor `localhost_server_configuration(port)` (in
`poke_env.ps_client.server_configuration`) and threads it to **every** Showdown client —
the training-env players (carried into the `SubprocVecEnv` spawn workers via the env-factory
closures), eval, and self-play. Every player-creating callback takes a `server_config` param
(defaulting to port 8000 for standalone use) and builds its players from it — never from a
bare `LocalhostServerConfiguration` constant. `server_port_threading_test.py` is the
regression guard: it fails if any of these callbacks hardcodes the default port instead of
threading the configured one (the original bug had the now-retired replay recorder connecting
to :8000 while training ran on :8001; eval forensic traces inherit the same guard).
There is no environment variable; `train_rl_agent.py`'s own default is 8000, but the
launcher overrides it to 8001 (above) before forwarding. The launcher forwards
`--showdown-port` verbatim (it strips only launcher-owned flags).

---

## Repository Structure

```
src/
  agents/
    model/           # Gen3FeaturesExtractor (PyTorch feature extractor)
    observation/     # Observation encoders (state_encoder, pokemon, moves, etc.)
    action/          # Action mask + mapping via LegalActions: pure action_to_choice →
                     #   Choice → serialize.choice_to_order (the one poke-env order touch)
    training/        # Callbacks and reward manager
    battle/          # Event-sourced battle layer (Gen3Battle, BattleEvent log, TurnView,
                     #   LiveView/LegalActions read-models, StrictBattleView boundary)
  main/
    launcher/          # Restart loop + Rich TUI (preferred for long runs)
                     #   checkpoint.py, worktree.py, child.py, input.py,
                     #   run.py, state.py, ui.py
    exit_codes.py      # TrainExitCode enum (COMPLETE=0, INTERRUPTED=15, CRASH=1)
    train_rl_agent.py  # Training entry point (also callable directly)
    eval_worker.py     # Subprocess bot-eval worker (frozen snapshot, CPU)
    play.py            # Battle / evaluation entry point
  poke_env/          # Forked poke-env library
  utils/
    git.py           # get_git_hash(), get_repo_root()
    (other utils)    # Hidden Power, teambuilder, team loader, logging
data/
  pokemon/           # JSON mappings: gen3_species, gen3_moves, gen3_items, gen3_abilities
  teams/             # Downloaded sample teams (gen3ou pool)
models/              # Saved PPO checkpoints (run_<timestamp>/ subdirs)
deps/
  pokemon-showdown/  # Git submodule — local Showdown server
```

---

## Observation Vector

The full observation is a **3321-dim float32 vector** (`Gen3ObservationEncoder.dimension`):

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 107) | 642 | 0 |
| Opp team (6 × 107) | 642 | 642 |
| Active context ×2 (boosts + full volatiles, `VOLATILE_DIM`=44) | 116 | 1284 |
| Global env | 18 | 1400 |
| Reactive + matchups | 302 | 1418 |
| Prev-turn action mask | 11 | 1720 |
| Turn history (`N_HISTORY_TURNS` × 159) | 1590 | 1731 |
| **Total** | **3321** | |

Per-Pokémon slot (107 dims): species ID + 6 base stats, item ID + known + consumed, 2 type IDs, ability ID + known, 7-dim condition (status one-hot), 4 × 11-dim move slots, HP fraction, species_known flag, sleep_counter_norm, toxic_counter_norm, **spread block (18 dims)**, **HP-candidate block (17 dims)**, active flag. The item block is 3 dims: `[item_id, known, consumed]` — `consumed=1` when the item was spent this battle (Berry activated, Knock Off, Trick, etc.) and `item_id` retains the identity of the consumed item so the model knows what was lost. `species_known = 1.0` for all populated slots (own team and revealed opponent mons), `0.0` for unseen opponent slots. Sleep counter: `min(turns_slept, 4) / 4` (Gen 3 max 4 turns); toxic counter: `min(turns_poisoned, 8) / 8` (practical max before fainting with Leftovers).

Move slot (11 dims, layout in `src/agents/observation/moves.py`): move ID, base power (/200), has_secondary, has_recoil, type ID, category (0=status, 1=physical, 2=special), known flag, current PP (/MAX_PP), max PP (/MAX_PP), accuracy (raw% / 100), never_miss bit. Accuracy is split into a continuous scalar plus a categorical bit: never-miss moves carry accuracy=100 in the mapping → encode as `[1.0, 1]`, while a genuine 100%-accuracy move is `[1.0, 0]` — same scalar, distinguished only by the bit. A 100%-accuracy move can still miss into evasion (Double Team) or after Sand-Attack; a never-miss move (Swift, Aerial Ace, all status/self moves) bypasses the accuracy/evasion check entirely.

Spread block (18 dims, appended at offset 71 within each slot): IVs ×6 each/31 + EVs ×6 each/252 + spread_known (1.0 own, 0.0 opp) + nature modifiers ×5 [atk, def, spa, spd, spe] as raw floats (0.9/1.0/1.1). Opponent slots have all 18 dims as zeros; `spread_known=0` distinguishes "unknown opponent" from "own Pokémon with 0 EVs". Own-team `mon.ivs/evs/nature` are populated by the poke-env fork's **`backfill_teambuilder_spread`** (`Battle.parse_request`): gen3ou has no team preview, so `apply_teambuilder_team` never attaches the spread — the backfill matches the declared teambuilder team to the request-built team by species and fills in IVs/EVs/nature (spread only, never re-running `_update_from_teambuilder`). Without it this block emitted a constant fallback (all-31 IVs, 0 EVs, neutral nature) for every own mon.

Global env (18 dims, layout in `src/agents/observation/global_env.py`): weather block (7: one-hot + cause-aware permanence + turns-remaining), spikes ×2 (2), log-turn (1), per-side screens (8: Reflect / Light Screen / Safeguard / Mist × both sides).

Reactive block (302 dims, layout in `src/agents/observation/reactive.py`): 14 scalar dims then the two 144-dim matchup matrices. Scalars: active-move power ×4 (/200) + active-move multiplier ×4 (/4), fainted counts ×2, active-status flag (1), `forced_struggle` (1), and the two **gen3_trapping_signals_v1** bits — `trapped` (1) and `maybe_trapped` (1) — sourced from the per-decision `LegalActions` snapshot (`legal.trapped` / `legal.maybe_trapped`), the same server-authoritative surface the mask is built from. They sit BEFORE the matchups so the feature extractor picks them up in `non_matchup_rest` automatically (it reads the matchup offset from the layout). `trapped` is redundant with the mask (switch bits already zeroed) but explicit; `maybe_trapped` is the high-value one — switches stay legal there, so it is the only way the model sees the trap risk before attempting a blind pivot.

Each TurnDelta slot (159 dims, layout in `src/agents/observation/turn_delta_encoder.py`; all offsets computed from named `OFFSET_*` / `*_DIM` constants — never hardcode indices). TurnDelta is now **folded from the event log** (`Gen3Battle.events_since(cursor)` per decision window) rather than diff-heuristics.

- **Base block (53 dims, indices 0–52)** — our/opp move features (5 each: raw move_id int, power_norm, has_secondary, has_recoil, raw type_id int), switched/failed flags, cant onehots (`CANT_DIM` = 12 ea, from `gen3_effects.CANT_REASONS`: slp/frz/par/flinch/recharge/attract/disable/taunt/imprison/focuspunch/nopp/truant — source-derived, crash-don't-drop enforced in the encoder), summed HP deltas, faint flags, opp_move_known, effectiveness onehots (4 ea), move-order (2).
- **Extended block (106 dims, indices 53–158)** — our/opp boost deltas (7 each); `phase_is_forced_switch` (1); our/opp `target_hp_delta` (1 each); per-side HP-level vectors (6 each); our/opp target_status onehots (7 each, at move-fire time); our/opp move-outcome onehots (3 each: `[hit, miss, fail]`); our/opp move-crit (1 each); **gen3_turn_delta_v2 additions**: `our_faint_causes`/`opp_faint_causes` (8 each, multi-hot over `attack/hazard/weather/status/recoil/selfko/leechseed/other`); **status-transition onehots** (4 × 7): `our/opp_status_applied` (status GAINED this window) + `our/opp_status_cured` (status LOST this window); **item-used bits** (2): `our/opp_item_used` — a single bit per side marking an item was consumed/removed this window (Berry/Knock Off/Trick). These (status transitions + item-used) are the per-turn *events*; the cause-**identity** (which item, which ability) lives in the per-mon item/ability block — the history carries the event, the block carries the what (parity with the collapsed `ability_activated` volatile). `our_attempted_move_id` (1, raw int embedded — the move we PRESSED, even if it never fired); **species block** (6 raw ints, embedded): `our_actor`/`opp_actor`/`our_target`/`opp_target`/`our_switch_to`/`opp_switch_to`; **gen3_trapping_signals_v1 additions**: `attempted_switch_rejected` (1 bit — the server REFUSED a switch we chose this window, |error|[Unavailable choice], i.e. we tried to pivot and got trapped) + `our_attempted_switch_to` (1 raw int embedded — the mon we PRESSED a switch to).

`our_faint_causes` / `opp_faint_causes`: multi-hot — the rare 2-on-one-side window (Pursuit/Future-Sight into a low-HP mon on hazards) can set 2 bits; the common Explosion double-KO is one bit each side. Cause derived from the DAMAGE event's `[from]` clause immediately preceding the FAINT in the event log. All-zeros when no faint. `our_attempted_move_id`: decoded from the pressed action index at build time, preserved even when the move never fired (cant / frozen / KO-before-acting). `attempted_switch_rejected` / `our_attempted_switch_to` (gen3_trapping_signals_v1): the rejected-pivot history — folded from the out-of-band `CHOICE_REJECTED` event (`TurnView.attempted_rejected`). On a rejected pivot `our_switch_to` is the unknown sentinel (the switch never happened) while `attempted_switch_to` names the mon we tried to bring in; both are zero on every turn with no rejection. (This RESTORES `attempted_switch_to`, dropped earlier on the "switches always execute" assumption — false exactly in the trap-reveal case.) Opp attempted action is not observable. Faint *counts* are kept on the `TurnDelta` dataclass (for reward) but not encoded — redundant with the faint flags + cause popcount.

**Embedded-ID manifest (the layout-driven contract):** which slot positions carry raw embedding IDs and which table each routes to is declared once in `TURN_DELTA_EMBEDDED_IDS` (in `turn_delta_encoder.py`). Both the encoder's layout and `Embeddings.embed_delta_slot` read it — there are **no hardcoded positions in the extractor** and the embed-width formula derives from the manifest, so a raw id can never silently leak through as a scalar. Adding an embedded ID is a one-line manifest change. Current manifest: 3 move IDs (our/opp/attempted) → 3×16, 2 type IDs → 2×16, 7 species IDs (our/opp actor + our/opp target + our/opp switch_to + attempted_switch_to) → 7×32 = 48 + 32 + 224 = 304 embedded dims + 147 pass-through scalars = 451-dim per slot. Positional encodings are added, one self-attention pass runs, and the last (most-recent) slot's output flows into the projection block. All zeros on the first turn of each episode.

Actor species resolution prefers `damaging_event.user_species` (protocol-truth) and falls back to `prev_active` for switches and non-damaging moves; target species comes from the OTHER side's `damaging_event.target_species`. Species ID 0 is the unknown sentinel.

---

## Feature Extractor Architecture

`Gen3FeaturesExtractor` in `src/agents/model/features_extractor.py` is decomposed into
named phase `nn.Module`s, each owning its layers, chained by a thin `forward_internal`
orchestrator. Data flows:

**`ObsUnpack` → `PokemonEncoder` → `TeamTransformer` → `CLSPool` → `ProjectionAssembler`**, then **two** root projection heads (policy + value), each `pre_proj_norm` → `projection` → `ReLU`. `forward` returns a `(pi_features, vf_features)` tuple — the transformer body is shared, but the actor and critic read it through independent CLS pools and projection heads (the **value-dedicated CLS readout**, H4 / Option C). This extractor is paired with `Gen3DualHeadMaskablePolicy` (`src/agents/model/policy.py`), which unpacks the tuple and routes each half to its own `mlp_extractor` branch; stock SB3 policies assume a single-tensor extractor and won't work.

The embedding tables live in a shared `Embeddings` module and are passed as a forward
argument to the phases that need them (Pokémon encoding and turn-history embedding), so
they register exactly once. An immutable `ExtractorContext` dataclass produced by
`ObsUnpack` carries the ~30 unpacked tensors to the downstream phases, keeping each
phase's forward signature narrow. See `src/agents/model/CLAUDE.md` for the phase contract.

1. **`Embeddings`** — shared tables: species (32), move (16), item (16), ability (16), type (16, shared for Pokémon types, move types, and TurnDelta move/type IDs). Owns the Hidden Power soft-type blend (`hp_soft_type`) and the per-slot TurnDelta embedder (`embed_delta_slot`).
2. **`ObsUnpack`** (stateless) — peels the flat 3321-dim observation into the named tensors of `ExtractorContext`: per-Pokémon block + categorical IDs, the global/reactive feature slices, the matchup matrices, and (hoisted here) the active-slot indices + fainted key-masks used downstream.
3. **`PokemonEncoder`** — embeds + stitches the enriched per-Pokémon vector; runs the **shared move processor** (Linear→ReLU→Linear, `MOVE_NET_HIDDEN`) over every move slot (input: move/type embeddings, remnants, known flag, battle context, per-move matchup ×6 + matchup-validity ×6, HP-candidate distribution, and prev-turn move validity), a **within-Pokémon move self-attention** (MHA 32-dim, 2 heads, + LayerNorm residual), then the **role encoder** (Linear→ReLU→Linear, `ROLE_ENCODER_HIDDEN`) → 12 × 128 role tokens.
4. **`TeamTransformer`** — builds a 23-token sequence (6 our-team + 6 their-team role tokens + `N_HISTORY_TURNS`=10 history tokens + 1 global token), adds token-type and history-positional embeddings, and runs a `TRANSFORMER_N_LAYERS`-deep `nn.TransformerEncoderLayer` stack (d_model 128, `TRANSFORMER_N_HEADS` heads, FFN `TRANSFORMER_FFN_DIM`, post-LN) under a key-padding mask that masks fainted team slots and empty history slots. History tokens come from `embed_delta_slot`; the global token from the two active-contexts + non-matchup scalars. Returns the two refined team-token blocks.
5. **`CLSPool`** — one learned CLS query per side cross-attends over its 6 post-transformer team tokens (fainted slots key-masked) → a 128-dim pooled team token per side (+ LayerNorm). Also extracts `our_active_refined` = the transformer output of our active slot. A **third learned query, `value_cls`**, cross-attends over **all 12 team tokens** (both sides, fainted key-masked) → a 128-dim global `value_pooled` summary — a whole-board "who's winning" read for the critic, a different aggregation than the policy's our-active-centric pools.
6. **`ProjectionAssembler`** — emits a `(pi_combined, vf_combined)` pair. Policy: `our_pool(128) + their_pool(128) + our_active_refined(128) + active_ctx_enc(32) + opp_ctx_enc(32) + non_matchup_rest`. Value: `value_pooled(128) + active_ctx_enc(32) + opp_ctx_enc(32) + non_matchup_rest`. `active_ctx_encoder` (Linear→ReLU→Linear, `ACTIVE_CTX_HIDDEN`) is shared by both heads — it encodes inputs, not the contested body representation.
7. **Root heads** — two parallel `pre_proj_norm` (LayerNorm) → `projection` (Linear) → `ReLU` heads, one per `*_combined`, both emitting `PROJECTION_DIM`. SB3 sizes the shared `mlp_extractor` from `features_dim = PROJECTION_DIM`, then `Gen3DualHeadMaskablePolicy` feeds the policy half to `forward_actor` and the value half to `forward_critic`.

Both projection input dimensions are discovered automatically via a dummy forward pass in
`__init__` (run through the assembled phases), so they stay correct when the architecture
changes without any manual update.

---

## Model Versioning

Every model save writes two files alongside the `.zip`:

- `model_config.json` — all weight-shape-relevant architecture params (embedding dims, layer sizes, obs dim, etc.)
- `metadata.json` — git hash, timestamp, SB3/Python versions

`load_model_snapshot()` in `src/agents/model/snapshot.py` checks these before calling `MaskablePPO.load()`. A mismatch causes a hard `[ModelVersion] FATAL` error at startup, not a silent wrong-output bug later.

### Key files

| File | Purpose |
|------|---------|
| `src/agents/model/features_extractor.py` | Module-level constants `ROLE_TOKEN_SIZE`, `PROJECTION_DIM`, `MOVE_NET_HIDDEN`, `ROLE_ENCODER_HIDDEN`, `ACTIVE_CTX_HIDDEN` — **single source of truth** for architecture dims |
| `src/agents/model/model_version.py` | `ModelVersion` dataclass, `MODEL_CONFIG_VERSION`, `ARCH_SIGNATURE`, `_migrate_config()` |
| `src/agents/model/snapshot.py` | `save_model_snapshot()` / `load_model_snapshot()` |
| `src/agents/model/snapshot_test.py` | Unit tests including a full save→load→forward-pass round-trip |

### When you change the architecture

**Changing a dim** (e.g. `ROLE_TOKEN_SIZE = 128 → 256`):
1. Update the constant in `features_extractor.py`
2. `check_compatible()` catches it automatically — old models can't load, which is correct

**Adding an optional feature** (e.g. dropout):
1. Add the field with a default to `ModelVersion` in `model_version.py`
2. Bump `MODEL_CONFIG_VERSION` (e.g. 1 → 2)
3. Add a migration in `_migrate_config()`: `if version < 2: data.setdefault("dropout_rate", 0.0); data["config_version"] = 2`
4. Decide if the field is weight-relevant (add to `_WEIGHT_FIELDS` in `check_compatible`) or advisory (skip)

**Structural change** (e.g. adding LSTM — different forward pass):
1. Change `ARCH_SIGNATURE = "gen3_lstm_v1"` in `model_version.py`
2. Old models fail with a clear arch-family error; no code needed to support them

### Startup smoke test

`_run_roundtrip_test()` in `train_rl_agent.py` runs automatically before `model.learn()` on every training start. It saves to a temp dir, reloads, and runs a zero forward pass. If serialization is broken it crashes in seconds rather than hours.

---

## Data Dependencies

Training requires JSON mapping files in `data/pokemon/`:
- `gen3_species.json` — species ID → `{num, baseStats}`
- `gen3_moves.json` — move ID → `{num, basePower, type, hasSecondary, hasRecoil}`
- `gen3_items.json` — item ID → `{num}`
- `gen3_abilities.json` — ability ID → `{num}`

These are loaded at startup and will raise `FileNotFoundError` / `ValueError` if missing or empty.

---

## Event-Sourced Battle Layer (`src/agents/battle/`, ai_v4)

poke-env is a **state tracker** — each `|...|` protocol line overwrites "current board"
fields. RL/reward/replay need the opposite: *what happened, in order*. The event-sourced
layer captures that without reimplementing poke-env (as-built record:
`designs/ai_v4/impl_step8_strict_battle_api_and_turndelta_fold.md`). **Status: the event log + read models are
live and CONSUMED in training.** The per-decision `TurnDelta` history block folds entirely
from the log (`build_from_events`, see "Per-decision history fold (Phase 5)" below) and feeds
the obs turn-history; the action masker/mapper and training display read the LiveView /
LegalActions / TurnView surfaces. (The full reward-manager rewrite onto TurnView/LiveView is
the separate, in-progress Step 5.)

- **`Gen3Battle(Battle)`** (`gen3_battle.py`) — subclasses poke-env's singles `Battle`.
  Its `parse_message` override classifies each line, calls `super().parse_message`
  (state tracking is **verbatim** poke-env), then appends a `BattleEvent` with
  attribution resolved **before** the line mutates state. State-equivalence with the
  classic `Battle` is structural (every line still flows through `super()`).
- **`BattleEvent` / `EventKind` / `MESSAGE_POLICY`** (`battle_event.py`) — the immutable,
  ordered schema and the completeness registry. Every protocol keyword poke-env can emit
  is classified `EVENT` / `STATE_ONLY` / `CONTROL` / `COSMETIC` / `UNSUPPORTED`; an
  unclassified or non-gen3 keyword **raises** (a deliberate tripwire). The conservation
  invariant (`Gen3Battle.assert_conservation()`) proves no line is silently dropped.
- **`TurnView`** (`turn_view.py`) — the **history** read surface ("what happened, in
  order"). Folds one turn's events into per-side intent (`move_id`, `switched`,
  `cant_reason`/`cant_move`, `crit`/`missed`/`failed`, `effectiveness`, `damaging_move`,
  `status_applied`/`status_cured`, `item_lost`/`item_gained`) + turn-level facts
  (`move_order`, `we_moved_first`, `both_attacked`, `someone_fainted`,
  `damage_on(species, side=…)`). `TurnDelta.build_from_events` (`training/turn_delta.py`)
  folds this on every production path — the diff-based detective is retired (see below).
- **`LiveView` / `LiveSide` / `LivePokemon` / `LiveMove`** (`live_view.py`) — the
  **current-board** read surface ("what is true now"), built via `battle.live_view()`. An
  immutable snapshot of HP, status, boosts, revealed moves/item/ability, volatiles, hazards,
  weather, team sizes/reveal counts — holding **only primitives, no past-turn state** and no
  reference back to poke-env's `Pokemon`. A consumer literally cannot reach `last_move`
  through it, so current-state and history come from two disjoint, separately-fuzzed sources
  that can't drift. Opponent fields are reveal-gated (unknown item → `None`, only revealed
  moves listed; `ability` is `None` unless disclosed or uniquely inferable from species).
  `LivePokemon` also carries the **spread block** (`base_stats` — public, both sides;
  `ivs`/`evs`/`nature` — own side only, gated by `spread_known` mirroring the obs encoder),
  `consumed_item` (id-form) and `status_counter`; `moves` is a tuple of `LiveMove(id,
  current_pp, max_pp)` with a `move_ids` accessor for id-only call-sites. `LiveView` carries
  the meta `turn`/`battle_tag`/`finished`/`won`/`lost`.
- **`LegalActions` / `LegalMove` / `LegalSwitch`** (`live_view.py`) — the
  **server-authoritative** legality surface, built via `LegalActions.from_battle(battle)`
  (or `strict_view().legal`): per-slot `LegalMove(id, current_pp, max_pp, disabled, target)`,
  `LegalSwitch(species, slot)` (slot = the team index the 11-dim action space uses),
  `force_switch`/`trapped`/`maybe_trapped`/`wait`/`struggle`, and a read-only
  (`MappingProxyType`) mirror of `last_request`. **Hybrid sourcing (kept deliberately):**
  `move_slots` is **wire-truth** (straight from `last_request['active'][0]['moves']`), while
  the legality *flags* (`switches`/`force_switch`/`trapped`/`maybe_trapped`/`wait`/`struggle`)
  are poke-env's **derived** interpretation (`available_switches`/`force_switch`/
  `available_moves`/…) — kept byte-identical and flagged as the second poke-env-interpreted
  seam (alongside Choice→BattleOrder serialization) a future fully-owned `Player` would
  re-derive. **Struggle is single-sourced:** `from_battle` filters the lone `struggle` entry
  OUT of `move_slots` and surfaces it only as the `struggle` flag — so it can never set both a
  move slot and bit 10 (the historical "struggle double-enabling" class). **The action
  masker/mapper read through this:** `Gen3ActionMasker.mask_from_legal(legal)` builds the
  mask, `Gen3ActionMapper.action_to_choice(action, legal)` is a pure decode to a poke-env-free
  `Choice`, and `serialize.choice_to_order(choice, battle)` is the lone serialization touch.
  The immutable snapshot captured at observation time (carried on the `BattleContext`) **is**
  the per-decision source, retiring the old `battle._gen3_decision_context` stash;
  `assert_decision_current` fail-loud-guards a mid-decision request shift.
- **`StrictBattleView`** (`strict_view.py`) — the **strict boundary** our non-`battle/` code
  is meant to read through, built via `battle.strict_view()`. Exposes **only** `.live`
  (`LiveView`), `.turn_view(n)`/`.history` (`TurnView`), `.legal` (`LegalActions`),
  `.events_since(cursor)`/`.event_cursor`, and scalar meta
  (`turn`/`battle_tag`/`finished`/
  `won`/`lost`). `turn` is the current-turn **int** — same name/meaning as `battle.turn` and
  `LiveView.turn` so a consumer migrating off `battle.turn` is a drop-in; the per-turn
  `TurnView` accessor is `turn_view(n)` (mirrors `.live`→`LiveView`, `.history`→all
  `TurnView`s) to avoid the method-vs-property name clash. `__getattr__` raises a helpful
  error naming the right accessor; the raw `Gen3Battle` is held privately and never
  returned. **The `action/` cluster is migrated** (Phase 3, partial): the masker/mapper read
  through `LegalActions` (built via `battle.strict_view().legal` / `from_battle`) and the
  `_gen3_decision_context` stash is gone. **The `training/` display / replay / control-flow
  cluster is also migrated:** `gen3_env.py`, `inference/player.py`, and `training/stall.py`
  read meta (`finished`/`turn`/`battle_tag`) through `battle.strict_view()` (stall keeps the
  `battle.save_replay(...)` seam — a poke-env method, not state), and `battle_recorder.py`
  builds the strict view once per record/finalize/summary call and reads current-board state
  (active/team/`species`/`item`/`hp`/`fainted`/`status`/boosts, `won`/`lost`/`turn`) only
  through its `LiveView` — the stale-`last_move` recorder fallback is dropped (history now
  comes from the event-log `TurnDelta`). Phases 1–2 are additive and obs-neutral; the action
  + training-display migrations are pure re-sourcing (no `ARCH_SIGNATURE` bump — these are
  display/replay/control-flow only, behaviour-preserving).
- **Per-decision history fold (Phase 5, DONE).** `TurnDelta` now folds **entirely** from the
  event log + LiveView and lives in its own module **`training/turn_delta.py`** (relocated out
  of the now-deleted `battle_context.py`). `build_from_events(prev, curr, action, events)` is
  the **sole production builder** — wired into the training env, `episode_tracker.py`,
  `reward_tracker.py`, and the forensic `battle_recorder.py`. What it folds from the event
  stream: moves/switches/cant/effectiveness/faints+causes/status-transitions/item-lost/
  outcome/crit/move-order, the **per-slot HP delta** (from each `DAMAGE/HEAL/SETHP`
  `hp_after` + `FAINT`→0 — bit-identical to `curr_hp − prev_hp`, no float-sum noise; a self-KO
  Explosion/Selfdestruct emits NO `-damage` on the user, so its HP→0 comes from the FAINT),
  and the **target-HP** attribution. **Findings — what the event log canNOT fold
  value-identically** (so these stay sourced from the LiveView-projected decision snapshot,
  which is current-board, not poke-env-raw and not a heuristic reconstruction): the per-slot
  **HP-after** (intrinsically current-board) and the **boost-stage delta** (`SETBOOST`/
  `clearboost`/`invert`/`copy`/`swap` carry no realized stage amount in the event payload —
  only an `op`). The per-decision snapshot itself relocated to **`training/battle_snapshot.py`**
  (`BattleContext`, still the reward `record_action` / mapper / `prev_mask` source; it is the
  documented seam exempted in `strict_api_lock_test`, inheriting `battle_context.py`'s status).
  The legacy diff detective `TurnDelta.build` is **retired from every production path** and
  marked legacy/test-only (retained for the poke-env-gap fuzz harnesses). Value-identity is
  pinned by the 15-min `turn_delta_fold_equivalence_fuzz_test.py` (event-fold vs a frozen
  snapshot-diff reference: 0 field diffs, 0 reward diffs over 150k+ decisions, all corner
  paths covered). The remaining Phase-5 work is the `LiveView` **event-fold** itself
  (side-condition counters / ability-item inference / type-change), tracked in
  `designs/ai_v4/todo_live_battle.md`.
- **Injection seam (wired into training):** `poke_env.player.Player.__init__` takes
  `battle_class=Battle` (default, with a `None`-guard since `PokeEnv` threads a `None`
  default to its `_EnvPlayer` agents). `Gen3Player` defaults `battle_class=Gen3Battle`
  (so RL / eval / replay / stat-tracking players inherit it), and `Gen3Env` defaults it
  too (both env agents track the log; the trainee `battle1` is what obs/reward/replay
  read). These + the (unchanged) parser are the only edits to the poke-env core.
- **Per-decision event window:** `Gen3Battle.event_cursor` + `events_since(cursor)` slice
  the log by "since the agent was last asked to act" — the granularity `TurnDelta` needs,
  which is NOT a protocol `|turn|N` boundary (a forced switch splits a turn into two
  decision windows; a faint window spans a turn boundary). `events_for_turn(N)` remains
  for protocol-turn slicing. Package re-exports are lazy (PEP 562 `__getattr__`) so
  importing one submodule doesn't force-load the others (avoids an import cycle).

**Verification:** unit tests in `src/agents/battle/*_test.py` (schema, registry audit,
scripted parse + state-equivalence, TurnView fold, `live_view_test.py` for the current-board
surface + widened fields, `strict_view_test.py` for the strict boundary + `LegalActions`
extraction + the forbidden-access guard). The spine is `event_log_fuzz_test.py` — real
`gen3ou` battles where both players run `Gen3Battle`; it independently re-derives each turn
from the intercepted raw protocol and asserts the event log matches, plus conservation +
event-kind coverage, the widened LiveView fields (spread/PP/consumed/counter/meta) per mon,
and the `LegalActions` legality surface against the live server request at every decision.

---

## Current State (End of Step 1)

Step 1 is complete. The pipeline is stable and all known correctness issues are resolved.

**What was hardened in Step 1:**

- `BattleContext` / `TurnDelta` / `SlotRegistry` — clean per-turn state, no battle object mutation
- `EpisodeTracker` — owns all per-episode mutable state; extension point for turn history
- `StallConfig` / `StallLogger` — stall detection extracted, shared by env and inference
- poke-env `env.py` — three lifecycle bugs fixed: stale battle queue, forfeit popup, stale `_choose_move()` hang
- Action mask — struggle double-enabling fixed; mask correctness audited and tested. **Now
  single-sourced through `LegalActions`** (ai_v4): struggle is the `legal.struggle` flag and
  never a move slot (not inferable from a slot), which structurally prevents the
  double-enabling class; `LegalActions` is hybrid (`move_slots` = wire-truth, the legality
  flags = poke-env-derived, the latter a future owned-`Player` re-derivation seam)
- `RLPlayer` / `StatTrackingRLPlayer` — exception-safe `choose_move()`, stall detection on inference

**Ready for Step 2: Turn History + Action Mask as Features**

- `EpisodeTracker._history` holds the full episode; cap to `deque(maxlen=N)` for N-frame history
- Each `BattleContext` already carries `obs` and `mask` — no new fields needed
- Extend `Gen3ObservationEncoder` to accept `history: list[BattleContext]` alongside the current frame
