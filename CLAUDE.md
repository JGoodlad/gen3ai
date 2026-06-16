# CLAUDE.md — Gen3AI Project Guide

## Development Stage

**Rapid iteration — checkpoint compatibility is not a concern.** Breaking changes to the observation space, network architecture, or action space are fine. Do not add backwards-compatibility shims or hesitate to change dims, layer sizes, or layouts.

Architecture constants (embedding dims, layer sizes, etc.) are defined as module-level constants in `src/agents/model/features_extractor.py` — that is the single source of truth. When you change one, change it there and nowhere else. See [Model Versioning](#model-versioning).

## Documentation Maintenance

Keep docs in sync **automatically, as part of the same change** — no need to be asked:

- **Every `CLAUDE.md`** (root, and the directory leaves — `src/agents/model/`, `src/agents/observation/`, `src/agents/battle/`, `src/agents/training/`, `src/main/launcher/`, `designs/`, anywhere): always current. If a change makes one stale, fix it in the same pass.
- **Every `README.md`**: always current. When you change dims, layout, obs/architecture, or anything a README documents, update it without being prompted. **Exception:** `designs/ai_v3/README.md` is a **frozen ai_v3 historical** digraph — do NOT update it for current-arch changes; the live architecture is this file's [Feature Extractor Architecture](#feature-extractor-architecture) section + `src/agents/model/CLAUDE.md`.

**Do NOT auto-update other docs under `designs/`** — `impl_step*.md`, `design_*.md`, `todo.md`, etc. are explicit-only. Touch them only when the user asks (directly or via `/gen3ai-update-design-docs`). The lone exception is `CLAUDE.md` files inside `designs/`, which follow the always-current rule above.

**Leaf CLAUDE.md map** — when working in one of these areas, read its leaf for the detail this root only summarises:

| Directory | Leaf covers |
|---|---|
| `src/agents/model/` | Feature-extractor phase contract, dual-head policy, architecture-constant rules, model versioning |
| `src/agents/gen3_data/` | The data facade: concept modules over `data/`, the acquisition-vs-access split, how it threads into the encoders |
| `src/agents/observation/` | Obs-build performance gate (mandatory benchmark) + the full per-block obs-vector layout |
| `src/agents/battle/` | Event-sourced battle layer (Gen3Battle, BattleEvent log, LiveView/TurnView/LegalActions, StrictBattleView, TurnDelta fold) |
| `src/agents/training/` | Bot-eval subprocess architecture + Showdown-port (`server_config`) threading |
| `src/main/launcher/` | Launcher internals: restarts, crash reporting, exit codes, flags, port default |
| `src/main/prober/` | Forensic-replay inspector (Textual TUI) + the pure probe engine `probe_replay.py` shares; trace discovery; worker-thread model |
| `src/main/tui/` | Thin shared Textual base (`Gen3App`, theme, `gradient_color`) — shared by the prober + launcher UIs |
| `designs/` | Which `ai_vN` folder is relevant; version map |

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
# also bridge-backed (no server): poke_env_gaps/{abilities,item_consumption,move_outcome,snatch,incoming_damage}_fuzz_test.py,
#                                  poke_env_gaps/move_alignment_fuzz_test.py (per-move obs features ↔ legal.move_slots[k] ↔ action 6+k, forces Choice-lock/Disable),
#                                  poke_env_gaps/belief_labels_fuzz_test.py (hidden-opp belief labels == actual opp team + no-leak),
#                                  poke_env_gaps/damage_op_probe_fuzz_test.py (AUTHORITATIVE DamageOperator physics gate — CONSTRUCTED single-turn
#                                      scenarios via the OMNISCIENT BattleStream `utils/bridge/damage_probe.js`: exact both-side HP + the sim's OWN
#                                      stats, zero measurement confounds; one modifier per scenario [type/STAB/SE/resist/4×/immunity/Thick Fat/
#                                      Choice Band/item/boosts/burn/screens/weather]) + poke_env_gaps/damage_op_fuzz_test.py (looser random-game net),
#                                  training/hidden_power_tracker_fuzz_test.py,
#                                  utils/bridge/reconstruction_fuzz_test.py (battle replay/re-roll invariants),
#                                  and training/obs_roundtrip_fuzz_test.py (offline obs == live obs, bit-for-bit)
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

Before a full training run, verify the core pipeline (env, reward, replay callback, stall
detection) with a quick debug run (~1 min):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug --steps 10000
```

`--debug` defaults to **CPU** (so a smoke never contends with a live GPU training run — an
explicit `--device cuda` still wins) and **skips all eval** by default — both the periodic eval
callback and the final win-rate eval. So the plain smoke above needs **no eval opponents / eval
server connection**; add `--use-showdown-bridge` to make it fully serverless (the in-process
sim runs the training battles too, no Showdown server at all). To also exercise the eval
pipeline (final win-rate eval, and the self-play seed → pool eval → promotion path under
`--self-play`), add `--debug-eval` — that path needs a server (default `:8000`) or
`--use-showdown-bridge`.

What to look for:
- `[ModelVersion] Round-trip smoke test PASSED` — serialization and reload are healthy (printed early, before training begins)
- `🏁 Episode Finished` lines appearing throughout — episodes completing and resetting
- `[STALL LOGGED]` may appear if a 250-turn game occurs — should be followed by another `🏁 Episode Finished`, not a hang
- `Win rate vs Random` / `Win rate vs Heuristic` at the end — **only with `--debug-eval`**; the default smoke ends at `Training complete` with no eval

A hang after `[STALL LOGGED]` or a crash before "Training complete" indicates a regression in the env/stall/forfeit pipeline. A `[ModelVersion] FATAL` error at startup means the checkpoint was saved with a different architecture than the current code.

---

## Launcher (preferred for long runs)

`src/main/launcher/` wraps `train_rl_agent.py` with **periodic restarts** (reclaim pymalloc
fragmentation; child saves a checkpoint on SIGTERM, launcher relaunches), **crash auto-restart**
(a self-crash relaunches from the last checkpoint after dumping a per-crash
`<run_dir>/crashes/restart_err_<token>.txt`, with a `--max-crash-restarts` circuit-breaker), **git-worktree
isolation** (agent pushes to `main` never affect a running session), a **Textual TUI** (metrics,
FPS, restart countdown, a `↻ N restarts (M crash)` badge; `l` logs, `e` events, `d` dashboard,
`r` restart, `c` checkpoint, `p` plots, `s` status, `f` force eval (confirm-gated; rejected if an
eval cycle is already running), `q`/ctrl-c quit, `v` copy mode
[freeze + native terminal select-and-copy — the portable copy path, works on Terminal.app]),
and **live crash-log
streaming** to `<run_dir>/launcher_child.log`.

The UI is **Textual** (built on the shared `src/main/tui/` base), launched with
`python -m main.launcher …` (or the back-compat alias `python -m main.launcher.tui …`). A closed
terminal (SIGHUP) or external `kill` (SIGTERM) is caught and turned into a clean,
checkpoint-saving shutdown rather than a lost checkpoint.

**Internals — how the UI reconciles the restart loop with Textual's event loop, the
quit/ctrl-c/SIGHUP teardown, crash reporting + auto-restart, exit codes
(`COMPLETE`/`INTERRUPTED`/`CRASH`/`FATAL_CONFIG` — the last gives up without restarting on an
arch/config mismatch instead of looping), the full flag table (`--restart-interval-hours`,
`--restart-grace-minutes`, `--max-crash-restarts`, `--no-pin`, `--sync-to-main`), the resume
contract, and the `:8001` Showdown-port default — live in `src/main/launcher/CLAUDE.md`.**

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
  --model models/<run>/checkpoints/checkpoint_NNNN_steps.zip \
  --steps 15000000 \
  --device cuda
```

Periodic + forced checkpoints live in `models/<run>/checkpoints/` (each `.zip` beside its
`.json` sidecar); legacy runs kept them at the run root and still resume. The checkpoint must
carry a `metadata.json` with a `git_hash`; the launcher pins the isolated
worktree to that commit so the resumed run uses the original code (override with
`--sync-to-main`). All non-launcher flags are forwarded verbatim to `train_rl_agent.py`.
`python -m main.launcher.tui …` is an alias for the same command.

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

**GPU OOM lever — `--grad-accum-steps K`:** keep a large effective batch when the full minibatch
won't fit. It runs K `--batch-size` micro-batches and steps the optimizer once per group of K,
giving the **exact** gradient of a `batch_size·K` batch at the activation-memory cost of one
micro-batch (stock SB3 steps per-minibatch, so `--batch-size` alone couples effective batch to the
memory peak). E.g. `--batch-size 4096 --grad-accum-steps 4` trains like `--batch-size 16384` at ~¼
the peak. `K=1` (default) is byte-identical to stock; it's a train-loop knob (not version-locked) —
forward it on every resume like `--batch-size`. With `K>=2` it also emits a **`train/noise_scale`**
diagnostic (McCandlish critical batch size) that tells you, as a number, whether your effective batch
is too small / about right / bigger than needed. Details: `src/agents/training/CLAUDE.md` → Gradient
accumulation.

Checkpoints are saved automatically into `models/run_<timestamp>/checkpoints/` (each `.zip`
beside its per-checkpoint `.json` sidecar); the run-level `model_config.json` / `metadata.json`
/ `latest.txt` and the `final_model*.zip` / `best_model/` stay at the run root.

### In-process bridge transport (`--use-showdown-bridge`, opt-in)

`--use-showdown-bridge` (default off) swaps **both training and eval** from a websocket
Showdown server to an in-process `BattleStream` subprocess — no server, no port, no
`/challenge` connection storm, deterministic delivery (poke-env issue #907). **A run needs no
Showdown server at all.** It reuses the *entire* obs/reward/mask/wrapper stack unchanged:

- **Training** — `attach_bridge_transport` (`src/utils/bridge/bridge_session.py`) swaps the two
  `_EnvPlayer` agents' transport for a background-pumped bridge subprocess per env. The child is
  **persistent by default** — one long-lived Node process reused across every episode (a fresh
  `START` rebuilds a clean `BattleStream`), which kills the per-episode Node-spawn cost. A
  single-env latency A/B (`bridge_vs_websocket_latency_benchmark.py`) measured ~13 ms/step
  websocket → ~6 ms/step persistent bridge (~2.1×); spawn-per-battle was only ~11 ms/step, so the
  reuse is the win. **But the 2.1× is single-env transport-only — at production `n_envs=64` the
  measured end-to-end training-FPS gain is just ~5%** (bridge 1192 vs websocket 1140 fps, vs-bots,
  CUDA), because oversubscription hides the per-step transport latency behind the SubprocVecEnv
  barrier (the box is CPU-saturated; n_envs is not the FPS lever — see
  `src/agents/training/CLAUDE.md` throughput notes). The bridge also started ~17% faster (no
  `/challenge` connection storm) and ran steadier. **The case for the bridge is operational, not
  FPS:** no server at all → no RAM-growth leak, no connection storm, no port tuning, deterministic.
- **Eval / self-play eval / final eval** — the eval players (built `start_listening=False`) play
  in-process via `run_local_battles` (the synchronous driver) instead of `battle_against`. Threaded
  as a `use_showdown_bridge` config key through `PerOpponentEvalCallback` / `SelfPlayCallback` →
  `eval_worker`. Each eval worker plays its opponents **one game at a time by default**
  (`--eval-concurrency-per-worker`, default `1`; threaded to `run_local_battles(concurrency=…)` /
  `max_concurrent_battles`). Overlapping battles within an opponent is single-thread asyncio
  latency-hiding, **not multi-core** — it nets negative under training contention (the old default's
  "measured slower"), but ~2× decisions/sec on spare cores (idle box / cycle tail); see
  `src/agents/training/CLAUDE.md` → intra-worker concurrency. Cross-opponent parallelism comes from the
  `--eval-workers` (5) subprocesses work-stealing the pool; this takes all eval load off the server.

**Persistent-child lifecycle (measured + optimized):** a child's RSS is **flat** — ~189 MB fresh
→ one-time ~+36 MB V8 warmup → ~229 MB with ~0 growth over thousands of battles
(`bridge_heap_growth_benchmark.py`). A child plays only ~2150 battles in the launcher's 3h restart
window, so the bridge does **not** reintroduce the server's RAM-creep and needs **no recycle within
3h** — the 3h restart owns the lifecycle. `recycle_every` (default 5000) is a backstop that never
fires under the launcher; it only caps marathon / no-launcher runs. A child that **dies** mid-run
**crashes** the env (no in-place recovery → launcher restart; resuming risks a corrupted PPO
transition).

Default stays websocket (opt-in): the end-to-end training-FPS gain at scale is only ~5% (the win
is operational — no server — not throughput). See `src/utils/bridge/README.md` and
`designs/ai_v5/design_local_sim_bridge_transport.md`.

### Non-barrier async rollout (`--async-rollout`, opt-in)

Stock `SubprocVecEnv.step()` is a **barrier** — each step waits for the *slowest* of N env workers,
so the latency-bound rollout (py-spy: ~86% wall, GPU ~86% idle) is straggler-gated and the policy
forward never overlaps env stepping. `--async-rollout` swaps in `AsyncSubprocVecEnv` + an on-policy
`collect_rollouts_async` (`src/agents/training/async_vec_env.py`) that keeps every worker
continuously in-flight and forwards whichever envs are **ready**, filling each env's own buffer
column. It stays **exactly on-policy** (PPO freezes the policy during collection — a scheduling
change, not an APPO-style algorithm change). Masks ride natively in the Dict obs (`obs["action_mask"]`);
`env_method` is drain-safe so the eval callback's mid-collection pushes don't desync.
**Measured (bridge, GPU forward, steady-state FPS): +20% at n_envs=16 (=logical cores); +14% at the
production `--n-envs 64` (1489→1695); `--async-rollout --n-envs 32` matches production `sync@64` FPS
with half the envs** (≈half the RAM). Off by default (= stock `SubprocVecEnv`), ignored under
`--debug`. Full design: `designs/ai_v5/design_async_rollout.md`.

### Bot evaluation

Bot eval runs in **frozen-snapshot subprocesses** (`--eval-workers`, default 5) that
**work-steal at battle granularity** from a shared pool and play the live server (or the bridge)
**without pausing training**; results merge into TensorBoard + TUI + best-model and land in
`metadata.json` as a top-level `latest_eval` block. Each opponent's `EVAL_GAMES` are split into
**shard units** (`--eval-shard-games`, default 25 → 4 shards/opponent) so any idle worker drains a
straggler's remaining games instead of one worker grinding a whole opponent — the long eval tail
collapses to one shard. The mechanism lives in the well-encapsulated **`eval_sharding/` package**
(deep `ShardedEvalPool` interface; aggregation is **exact** — Σwon/Σfinished etc., raw δ pooled then
one CVaR), with a documented **`rating.py` seam** (`MatchRecord` / `RatingModel` / `BradleyTerryRating`)
ready for a future Glicko-2/TrueSkill without touching the live ELO path. **`--self-play` eval shares
this exact non-blocking pipeline** (with the worker pool doubled to 10, since sentinel matchups infer
for both players) — the workers additionally work-steal the pool sentinels' shards, and a winning
cycle promotes its frozen snapshot into the pool by file-copy (`SnapshotPool.add_from_path`). The full
design (battle-level work-stealing, exact aggregation, graceful-shutdown drain, resume re-publish,
sentinels + promotion, `--eval-workers` / `--eval-shard-games` / `--eval-device`) is in
`src/agents/training/CLAUDE.md`.

### ELO / skill rating

Under self-play pool play, `win_rate_vs_pool` is pinned near 50% by the promotion gate (a
sliding window of recent selves) and `win_rate_vs_bots` saturates — so neither tracks real
progress. The **ELO subsystem** fits an **anchored Bradley-Terry** rating over the win-records
every eval cycle already produces (trainee vs the 9 fixed bots + pool sentinels — no new
battles), giving one absolute number that rises with skill. Each cycle appends a row to an
append-only `<run>/eval_results.jsonl` and records a live `eval/elo` (+CI) to TensorBoard + a
`🏅 ELO` TUI badge. The fixed bots are the anchor; `python -m agents.training.bot_elo_calibration`
plays a one-time bot-vs-bot round-robin (bridge, no server) → `data/gen3_bot_elo_anchors.json`,
making snapshot ELOs **comparable across runs**. Offline: `python -m main.elo <run_dir>` prints a
ladder and plots an Elo-vs-step curve (and can backfill a running run from TensorBoard with
`--source tb`). Full design: `src/agents/training/CLAUDE.md` → ELO / skill rating.

### Opponent distillation (`--distill-opponents`)

On a `--self-play` run, distil the frozen pool opponents into a **cheaper network** for faster
rollouts (the opponent forward is ~70% of env-worker CPU → ~+15–25% rollout throughput at near-zero
quality cost). Off by default. It is **all-or-nothing** — the per-step barrier means one full-opponent
worker straggles and gates the batch, so the pool is only ever 100% distilled or 100% full. An
idempotent reconcile loop (`SelfPlayCallback` → `DistilledOpponentManager`) **backfills the whole pool
on enable** (incl. sentinels), then **atomically switches** full↔distilled; each new promotion is
distilled before it's sampled. A fail-closed gate (fidelity + head-to-head) + capacity escalation +
drift auto-revert keep it safe. Distilling runs in a non-blocking subprocess on `--eval-device`;
artifacts + per-snapshot gate manifests live in `models/<run>/distilled/` (auto-cleaned with the pool
window — the manifest is the source of truth, `summary.json` holds only a re-publish block). Full
design: `designs/ai_v5/distill_integration.md` (§8 all-or-nothing, §7 restart resilience); module map:
`src/agents/training/distill/CLAUDE.md`.

---

## Playing / Evaluation

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/play.py
```

Requires the Showdown server to be running (see below).

---

## Prober (forensic-replay inspector)

An interactive Textual TUI that browses the `eval_traces` a run writes and
analyzes each saved decision point (faithfulness, type matchups, an intervention
sweep, gradient saliency). No server needed — it reads saved traces and a
checkpoint. Point it at a run dir; it auto-discovers the trace tree and resolves
the checkpoint (best_model → latest; override with `--ckpt`):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.prober models/run_<timestamp>
```

The same analysis is available headless for one invocation via the
`probe_replay.py` CLI (`python -m main.probe_replay <ckpt> <summary.json>
<states.npz> <inv>`); both share the pure engine in `src/main/prober/engine.py`.

**For agents/scripts**, a JSON API + CLI (`ProbeSession` / `python -m
main.prober.query summary|list|scan|overview|find|analyze|falsify|falsify-scan|calibration`) exposes
the same probing infrastructure programmatically — list/filter battles, **`scan` the worst turn in
every loss across an opponent (model-free, ranked)**, digest one battle, find
decisions the model disagrees with, deeply analyze one decision, **`falsify`
a battle's worst decisions: luck-vs-mistake dice attribution by RE-ROLLING the
real turns** (fix-both luck percentile + paired alternative-action sweep via the
battle-reconstruction layer; bridge-eval traces with a `*_reconstruction.json`
sibling only), or **`falsify-scan` a whole run: aggregate that split
across every loss into a CRATER-FRACTION BRACKET** (|δ|-weighted — `aleatoric`=LUCK /
`unattributed`=NEUTRAL residual / proven `policy_reducible`=MISTAKE; an input to the
distributional-critic decision where `critic_headroom_upper_bound` = LUCK+NEUTRAL is an
explicit UPPER BOUND with `caveats`, not a measurement), or **`calibration` to split that
`unattributed` bucket** into `critic_overvalued` vs `lost_position` via recorded V(s) vs realized
return G(s) — a selection-aware reliability curve that self-diagnoses the eval-quota confound
(`bias_on_wins`/`bias_on_losses`). Internals
— engine/app split, the model-resolution ladder, Outcome panel, flags, and the
agent API — are in `src/main/prober/CLAUDE.md`.

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
... -m main.launcher ...                            # launcher DEFAULTS to :8001
... -m main.launcher --showdown-port 8123 ...       # explicit port still wins
npm run stop -- 8001                                # kills the 8001 instance
```

**The launcher defaults `--showdown-port` to 8001** (see `src/main/launcher/CLAUDE.md`) so a long
session never rides on the shared dev server; an explicit `--showdown-port` always wins.
`train_rl_agent.py` run directly still defaults to 8000. How a single `ServerConfiguration` is
built once and threaded to **every** Showdown client (env spawn-workers, eval, self-play) — and
the `server_port_threading_test.py` regression guard — is documented in
`src/agents/training/CLAUDE.md`.

---

## Repository Structure

```
src/
  agents/
    model/           # Gen3FeaturesExtractor (PyTorch feature extractor) — has CLAUDE.md
    gen3_data/       # The data facade: concept modules (moves/species/items/abilities/natures/
                     #   type_chart/priors) over data/ — single interface, poke-env-free — has CLAUDE.md
    observation/     # Observation encoders (state_encoder, pokemon, moves, etc.) — has CLAUDE.md
    action/          # Action mask + mapping via LegalActions: pure action_to_choice →
                     #   Choice → serialize.choice_to_order (the one poke-env order touch)
    training/        # Callbacks, reward manager, eval pipeline — has CLAUDE.md
                     #   elo.py (Bradley-Terry skill rating), bot_elo_calibration.py (anchor round-robin)
                     #   eval_sharding/ (battle-level work-stealing pkg), rating.py (Glicko-ready seam)
    battle/          # Event-sourced battle layer (Gen3Battle, BattleEvent log, TurnView,
                     #   LiveView/LegalActions read-models, StrictBattleView) — has CLAUDE.md
  main/
    launcher/          # Restart loop + Textual TUI (preferred for long runs) — has CLAUDE.md
                     #   core: checkpoint.py, worktree.py, child.py, input.py, state.py, ipc.py
                     #   UI: app.py + launcher.tcss · run loop: run.py · format.py · tui.py (alias)
    prober/            # Forensic-replay inspector (Textual TUI) — has CLAUDE.md
                     #   engine.py (pure analysis), model.py, discovery.py, app.py
    tui/               # Shared Textual base (Gen3App, theme, colors) — has CLAUDE.md
    exit_codes.py      # TrainExitCode enum (COMPLETE=0, INTERRUPTED=15, CRASH=1, FATAL_CONFIG=3)
    train_rl_agent.py  # Training entry point (also callable directly)
    eval_worker.py     # Subprocess eval worker (frozen snapshot, CPU) — work-steals shard units
    probe_replay.py    # Forensic-replay CLI (thin wrapper over main.prober.engine)
    elo.py             # Offline ELO analyzer CLI (ladder + Elo-vs-step curve)
    play.py            # Battle / evaluation entry point
  poke_env/          # Forked poke-env library
  utils/
    git.py           # get_git_hash(), get_repo_root()
    (other utils)    # Hidden Power, teambuilder, team loader, logging
data/                # Source of truth — derived by tools/, read via agents.gen3_data
  pokemon/           # species/moves/items/abilities/type_chart/natures + smogon stats & priors
  teams/             # Downloaded sample teams (gen3ou pool)
  gen3_bot_elo_anchors.json  # Fixed bot ELO anchors (bot_elo_calibration.py round-robin); optional
models/              # Saved PPO checkpoints (run_<timestamp>/ subdirs)
                     #   <run>/checkpoints/  — periodic + forced checkpoints (.zip + .json sidecars)
                     #   <run>/best_model/   — best-by-eval export; <run>/snapshots/ — self-play pool
                     #   <run>/model_config.json, metadata.json, latest.txt, final_model*.zip — run root
                     #   latest.txt holds a run-RELATIVE path (e.g. checkpoints/checkpoint_123_steps.zip)
deps/
  pokemon-showdown/  # Git submodule — local Showdown server
tools/               # Acquisition layer (knows the 3 upstreams) — has CLAUDE.md
  pokemon_data_extractor/  # pokedex/Showdown/GenData -> data/pokemon/ reference files
  smogon_stats_downloader/ # Smogon usage stats -> data/pokemon/ priors
  sample_team_downloader/, others_team_downloader/  # -> data/teams/
```

---

## Observation Vector

The full observation is a **3457-dim float32 vector** (`Gen3ObservationEncoder.dimension`):

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 110) | 660 | 0 |
| Opp team (6 × 110) | 660 | 660 |
| Active context ×2 (boosts + full volatiles, `VOLATILE_DIM`=44) | 116 | 1320 |
| Global env | 18 | 1436 |
| Reactive + move-effects + **incoming-damage** + **turns_since_progress** + **protect-odds** + matchups | 402 | 1454 |
| Prev-turn action mask | 11 | 1856 |
| Turn history (`N_HISTORY_TURNS` × 159) | 1590 | 1867 |
| **Total** | **3457** | |

**The full per-block layout** — the 110-dim per-Pokémon slot (incl. a 3-dim
`gen3_sleep_wake_belief_v1` block: `sleep_is_deterministic` [Rest], computed `p_wake`, and
`sleep_counter_reliable` — zeros unless the mon is asleep), the 11-dim move slot, the 18-dim
spread block, global env, the 402-dim reactive block (**19 scalars** — the 14 prior + the
log-saturated **`turns_since_progress`** no-progress clock at `vec[14]`, `gen3_markovian_progress_v1`
(the no-progress reward keys on the SAME EpisodeTracker-owned counter) + the **2 protect-odds scalars**
at `vec[15]`/`vec[16]`, `gen3_protect_odds_v1` — P(a Protect/Detect/Endure succeeds NOW) for our /
the opp active mon, the gen3 floored-doubling stall odds (100/50/25/12.5, floor 1/8) from each mon's
`LivePokemon.protect_counter` (the only obs view of the stall counter; public both sides, no leak) +
the **2 `gen3_wish_wired_v1` `wish_floating` scalars** at `vec[17]`/`vec[18]` (our/opp side — the
pending-Wish "floating heal": `WISH_HEAL_FRACTION` ≈0.5 of the slot mon's max HP when a Wish cast last
turn resolves at the end of this turn, else 0; gen3 Wish heals the RECIPIENT's maxhp/2 so the fraction
is a constant ≈0.5 — GIGO-proof — slot-keyed so it survives faint/Roar-phaze/switch; reconstructed from
the event log since poke-env doesn't track it; fuzz-validated vs the real sim's resolve heals) — +
the 44-dim action-aligned
move-effect block, 4 slots × 11 feats (incl. the `gen3_status_cure_moves_v1` **cures_self_status** /
**cures_team_status** bits — Refresh self-cure, Heal Bell / Aromatherapy team-cure) + the **51-dim incoming-damage / OHKO belief block**
[`gen3_incoming_crit_split_v1`: per our mon, phys/spec expected-damage + the modal **no-crit** P(KO) +
the **crit-risk DELTA** per channel (crit-inclusive − no-crit ∈ [0, _CRIT_P] — a decorrelated "crit
tax" feature, so the model prices the modal line without over-weighting the coinflip) + P(outspeed) +
a **threat-provenance** scalar (1.0 = a revealed move, <1 = a usage-prior guess; 0.0 = no KO threat —
the "how much are we guessing" signal), then 3 opp recovery scalars] +
288 matchup), and the 159-dim
TurnDelta slot (incl. the embedded-ID manifest) — lives in **`src/agents/observation/CLAUDE.md`**.
Every offset is computed
from named constants; never hardcode indices.

---

## Feature Extractor Architecture

`Gen3FeaturesExtractor` in `src/agents/model/features_extractor.py` is decomposed into named
phase `nn.Module`s chained by a thin orchestrator:

**`ObsUnpack` → `PokemonEncoder` → `[BeliefSlots?]` → `TeamTransformer` → `[BeliefHead?]` →
`[MoveBelief?]` → `CLSPool` → `[DamageOperator?]` → `ProjectionAssembler`**, then **two** root projection
heads (policy + value), each `pre_proj_norm` → `projection` → `ReLU`. The bracketed phases are
flag-gated (with all off the chain is the byte-for-byte baseline): `BeliefSlots`/`BeliefHead` under the
hidden-opponent **belief aux** (`--opp-belief-aux-coef>0`) — `BeliefSlots` fills the un-revealed
opponent team slots with distinct learned unknown-mon tokens (refined in-lineup by the transformer so
both heads attend over the imagined mons), and `BeliefHead` aux-supervises those refined tokens to
predict each hidden mon's species + moves (privileged labels, training-only, never in the forward).
Under `--opp-belief-latent-coef>0` `BeliefHead` also carries an asymmetric SimSiam **latent** predictor:
each believed slot's refined token is regressed (cosine) toward the stop-grad `pokemon_encoder` role-token
of the TRUE hidden mon — graded identity supervision the hard species CE can't give (target from a
training-only `belief_target_slots` obs key, stashed for the loss only, never in pi/vf — leak-safe).
`MoveBelief` (`--move-belief-mode`) predicts + reinjects each opp slot's moveset into its token (and under
`--move-prior-fusion` fuses the Smogon move-frequency **prior** into that prediction as a log-odds residual
+ pins revealed moves certain, so the belief is a unified posterior — *known certain, unknown prior⊕learned*);
and `DamageOperator` (`--damage-op`) consumes that move belief's predicted moves to compute the believed-move
incoming damage to each of our mons (a differentiable gen3 calc), appended to **both** projection heads —
so the gradient sharpens the move belief toward real KO threats (`designs/ai_v6/design_differentiable_damage_op.md`);
its effect block carries per-status SECONDARY probabilities (incoming opp threat + per-OUR-move outgoing,
accuracy-folded, ×Serene Grace / Shield Dust — `gen3_unified_move_system_v1`). Inside `PokemonEncoder`, the
flag-gated `MoveLatentEncoder` (`--move-latent`) concatenates a context-free mechanics-grounded per-move
latent into the move network; its latent table is the Stage-3 similarity-grading target
(`--move-belief-latent-coef`, so Rock Slide ≈ Hidden Power Rock — `designs/ai_v6/design_unified_move_system.md`).
A separate optional `WinProbHead` (`--win-prob-mode none|read_only|shaping`) reads `value_pooled` and emits a
calibrated **P(win)** logit — a SIDE readout (stashed for the aux loss + the prober, **never** in pi/vf, so
projection dims are unchanged), supervised by the Monte-Carlo episode outcome (win=1/loss=0); `read_only`
stop-grads its input (a risk-free diagnostic), `shaping` lets the win objective shape the trunk.
`forward` returns a `(pi_features, vf_features)` tuple — the transformer body is shared, but the
actor and critic read it through independent CLS pools and projection heads (the
**value-dedicated CLS readout**, H4 / Option C). It must be paired with
`Gen3DualHeadMaskablePolicy` (`src/agents/model/policy.py`), which unpacks the tuple and routes
each half to its own `mlp_extractor` branch; stock SB3 policies assume a single-tensor extractor
and won't work. Both projection input dims are auto-discovered via a dummy forward pass in
`__init__`, so they stay correct as the architecture changes.

**The phase-by-phase data flow (the 7-phase contract, dims, and the `ExtractorContext` /
`Embeddings` ownership rules) is documented in `src/agents/model/CLAUDE.md`.**

---

## Model Versioning

Every model save writes two **run-level** files at the run root:

- `model_config.json` — all weight-shape-relevant architecture params (embedding dims, layer sizes, obs dim, etc.)
- `metadata.json` — git hash, timestamp, SB3/Python versions (+ `snapshot_history`, `latest_eval`)

These are run-level (one per run), NOT per-checkpoint: a periodic checkpoint `.zip` lives one
level down in `checkpoints/` beside its own per-checkpoint `.json` sidecar, so
`load_model_snapshot()` searches the zip's dir **and its parent** (the run root) for
`model_config.json`. `load_model_snapshot()` in `src/agents/model/snapshot.py` checks it before
calling `MaskablePPO.load()`. A mismatch causes a hard `[ModelVersion] FATAL` error at startup,
not a silent wrong-output bug later. `model_config.json` additionally records `vf_coef` (`--vf-coef`,
the PPO value-loss coefficient): it is **fixed for a run's lifetime**, so resuming with a
different value is a FATAL error — enforced resume-only (frozen eval/pool/distill opponents are
exempt, since vf_coef doesn't affect a forward pass). See `src/agents/model/CLAUDE.md` →
resume-immutable training hparams. `_run_roundtrip_test()` in `train_rl_agent.py` runs automatically
before every `model.learn()` (save → reload → zero forward pass), so serialization breakage
crashes in seconds rather than hours.

The architecture-constant single source of truth is the module-level constants
(`ROLE_TOKEN_SIZE`, `PROJECTION_DIM`, `MOVE_NET_HIDDEN`, `ROLE_ENCODER_HIDDEN`,
`ACTIVE_CTX_HIDDEN`) at the top of `features_extractor.py`; `ARCH_SIGNATURE` /
`MODEL_CONFIG_VERSION` live in `model_version.py` (current `ARCH_SIGNATURE`:
`gen3_wish_wired_v1` — WIRES two reactive scalars (`vec[17]` our side, `vec[18]` opp side) with the
pending-Wish "floating heal" signal. gen3 Wish (gen4-inherited) heals the RECIPIENT's `maxhp/2` at the
END of the turn after cast, slot-keyed (survives faint / Roar-phaze / switch / self-KO), duration 2,
double-Wish fails. Because the heal is the recipient's own maxhp/2, the heal fraction is ALWAYS ≈0.5, so
each scalar is a flat `WISH_HEAL_FRACTION` (0.5) when a wish cast last turn resolves this turn, else 0 —
no max-HP read, GIGO-proof. poke-env tracks none of it → reconstructed from our event log
(`wish_belief.py`); fuzz-validated vs the real sim (every actual resolve was flagged pending the turn
before). It first reserved the dims (`gen3_wish_reserve_v1`, `REACTIVE_SCALAR_DIM` 17 → 19,
obs dim 3455 → 3457) — wiring them is a VALUES-only change (same dim). It stacks on three prior obs
changes: `gen3_protect_odds_v1` (2 reactive
protect-success scalars, obs 3409 → 3411); `gen3_status_cure_moves_v1` — two static per-move bits
**cures_self_status** (Refresh) + **cures_team_status** (Heal Bell / Aromatherapy), so the head
connects a status-cure move to the per-mon status one-hots (prober-verified gap: the head routed its
own status onto Recover/switch but never the cure move), `MOVE_EFFECT_FEATURES` 9 → 11 (3411 → 3419);
and `gen3_sleep_wake_belief_v1` — a 3-dim per-mon SLEEP WAKE belief block [`sleep_is_deterministic`
(Rest), a COMPUTED `p_wake` from the verified gen3 sleep-RNG tables (opp time∈{2,3,4,5}, Rest time=3,
Early Bird halves; opp Early-Bird prior marginalised; Rest source from the event log's `[from]` clause;
fuzz-calibrated vs the real sim RNG), `sleep_counter_reliable`], `POKEMON_VECTOR_DIM` 106 → 109
(3419 → 3455). All four are retrain-class; current
`MODEL_CONFIG_VERSION`: **28** — v16 added the in-place
hidden-opponent belief-aux toggle `opp_belief_slots` + its coef `opp_belief_aux_coef`, v17 the
move-belief reinjection toggle `move_belief_mode` + `move_belief_coef`, v18 the latent-belief toggle
`opp_belief_latent` + `opp_belief_latent_coef`, v19 the differentiable damage-operator toggle
`damage_op`, v20 the unified-move-belief prior-fusion toggle `move_prior_fusion`, v21 the
unified-architecture ablation toggle `mask_incoming_damage_obs`, v22 the tri-state win-probability head
`win_prob_mode` (none/read_only/shaping) + its coef `win_prob_coef`, v23 the **unified damage system** —
the OUTGOING per-move direction `damage_outgoing` (our active → opp active, action-aligned — the
equal-effectiveness move tie-break) + the LEGALITY-only move-prior gate `move_candidate_floor` (>0 drives
moves a species can't learn to ~0 while legal moves keep their true usage — rare-but-liftable, never pruned,
so surprise-move anticipation survives), both reachable via the one `--unified-damage {off,incoming,both}` knob (which
desugars into `move_belief_mode`/`damage_op`/`move_prior_fusion`/`damage_outgoing`); the op's per-mon
feature is now the **3-roll + P(KO) + accuracy** representation `[low,high,crit,pko,accuracy]×{phys,spec}`
(`pko=acc·P(KO|hit)` — the operator does the multiplicative physics so the ReLU head stays additive); none
bump `ARCH_SIGNATURE` since each OFF is byte-identical and the directions are GPU-operator outputs (obs dim
unchanged at 3457). **v24 the unified MOVE system** (`gen3_unified_move_system_v1`) — the structural
`move_latent` toggle (a context-free `MoveLatentEncoder`: a mechanics-grounded per-move latent —
move/type embeddings ⊕ a structured `MOVE_ATTR` of BP/category/accuracy/priority/drain/per-status
secondary chances — concatenated into the move network, **and** the similarity-grading target so Rock
Slide ≈ Hidden Power Rock) + its training-only grading coef `move_belief_latent_coef` (cosine of the
predicted move distribution's expected latent → the true moveset's mean latent + VICReg). v24 ALSO
enriches the `DamageOperator`'s effect block with per-status SECONDARY probabilities — incoming (the opp
active's damaging-move para/flinch/freeze, accuracy-folded, ×Serene Grace) + per-OUR-move outgoing ("what
status can this move cause, with what probability", ×our Serene Grace, ×opp Shield Dust) — **intrinsic to
`--damage-op`** (no separate flag; the secondary data is newly extracted into `gen3_moves.json`). The one
umbrella knob is `--unified-moves {off,incoming,both}` (sets `--unified-damage` + `--move-latent` +
`--move-belief-latent-coef 0.05`). `move_latent` OFF stays byte-identical (NO `ARCH_SIGNATURE` bump); a v23
`--damage-op` checkpoint won't load into v24 (the op's output dim grew). **v25 the SPREAD belief +
disable-redundant master flag** (`gen3_unified_spread_belief_v1`) — `--spread-belief` (the THIRD belief
leg: predicts the opp's hidden SPREAD = 5 derived stats per slot from a usage prior ⊕ a learned head,
reinjected into the opp token, so the `DamageOperator` consumes BELIEVED opp stats instead of its
hand-coded de-timid/neutral constants) + its training-only `--spread-belief-coef` (speed supervision from
observed move order — flag wired, loss staged); and `--unified-obs`, ONE master switch that zeros the
now-GPU-subsumed CPU obs regions from the model's view (incoming-damage + active-move scalars + move-effect
block; granular `--mask-*-obs` underneath, reward/PBRS untouched). Pure-unified run = `--unified-moves both
--spread-belief --unified-obs`. OFF byte-identical. **v26 op-physics parity** (`gen3_unified_op_physics_v1`,
intrinsic to `--damage-op`, values-only) — the op now folds stat-stage boosts/burn/weather/paralysis +
fixed-damage moves (validated by the constructed Showdown probe `damage_op_probe_fuzz_test.py`, 19/19).
**v27 op status-landing** (`gen3_unified_status_landing_v1`, intrinsic to `--damage-outgoing`) — the op's
OUTGOING direction gains a per-OUR-move STATUS-LANDING block (8 dims: P(a dedicated status move lands vs THIS
opponent — Toxic/Will-O-Wisp/Thunder Wave/Spore/**Leech Seed**) + a `known` bit), the GPU home for the masked
move-effect `status_will_land`. Folds accuracy × per-MOVE type immunity (incl. the v26-deferred **Leech Seed
→Grass**) × ability immunity (revealed→exact, else the Smogon prior) × already-statused × **Sleep Clause** (a
2nd inflicted sleep fails; a Rest self-sleep does NOT consume our cap, reusing `sleep_is_deterministic`) ×
**Substitute** (a Sub blocks every status move incl. Leech Seed, read from the public volatile). gen3 rules
imported from `gen3_mechanics` (one source); Shield Dust is N/A (it only scales SECONDARY effects). A v26
`--damage-outgoing` checkpoint won't load (SB3 `load_state_dict` projection in_features mismatch — the dim is
runtime-discovered, not a `check_compatible` field). `--mask-move-effects-obs` now requires `--move-latent`
AND `--damage-outgoing`. **v28 op Choice Band** (`gen3_unified_choice_band_v1`, intrinsic to `--damage-op`) —
the op prices CB (×1.5 physical Atk): OUTGOING applies our own (known) CB ×1.5 deterministically; INCOMING
exposes a per-our-mon CB-CONDITIONAL physical tail (`phys_high_cb` + `P(OHKO|CB)`) + a shared `p_cb`
(P(opp holds CB) — a species usage prior collapsing to 0/1 on item reveal), **decorrelated** so the head
weights them (OHKO is a nonlinear threshold a mean-field blend would blur). Move-lock + the ChoiceBandTracker
disproof are a follow-up. **v29 the distributional VALUE head** (`gen3` interpretability side readout,
`value_dist_mode` none/read_only/shaping + `value_dist_bins`) — `ValueDistHead` reads `value_pooled` and
emits per-atom return-distribution logits (softmax = the critic's predicted return distribution; sharp =
confident, wide = uncertain, bimodal = coinflip), a SIDE readout stashed for the prober + a future aux
loss, **never in pi/vf** (projection dims unchanged → OFF byte-identical, no `ARCH_SIGNATURE` bump); mode +
bins gated in `check_compatible`, the support (vmin/vmax) resume-only. Phase-A foundation (head +
versioning); the distributional aux loss + capture/prober/launcher are follow-ons. Current
`MODEL_CONFIG_VERSION` = **29**. Full design:
`designs/ai_v6/design_distributional_value_critic.md` (and `design_unified_move_system.md` for v24,
`design_unified_damage_system.md` for v23).
**The full versioning playbook — what to do when you change a dim vs add an optional feature vs
make a structural change — is in `src/agents/model/CLAUDE.md`.**

---

## Data Dependencies

**`data/` is the single source of truth — the runtime reads only `data/`, never live from
poke-env.** The split is **acquisition vs. access**: `tools/` (the only layer that knows the three
upstreams — poke-env static data, the Showdown source tree, Smogon usage stats) *derives* and
normalizes each file into `data/pokemon/`; the runtime reaches all of it through the
**`agents.gen3_data` facade**, blind to provenance (see `src/agents/gen3_data/CLAUDE.md`).

Reference data (deterministic) under `data/pokemon/`, all regenerable via
`tools/pokemon_data_extractor/sync.py`:
- `gen3_species.json` — species id → `{num, baseStats, name}`
- `gen3_moves.json` — move id → `{num, basePower, type, accuracy, never_miss, hasSecondary, hasRecoil,
  priority, secondaryEffects {col: percent}, drainFraction, recoilFraction, …}` (the structured
  secondary/priority/drain fields are `gen3_unified_move_system_v1` — GPU-side only, NOT in the obs vector)
- `gen3_items.json` — item id → `{num, name}` (`num` is the item-dex number; cross-gen aliases share one num)
- `gen3_abilities.json` — ability id → `{num, name}`
- `gen3_type_chart.json` — `{DEF: {ATT: multiplier}}` effectiveness chart (was live `GenData`)
- `gen3_natures.json` — nature → `{num, stat multipliers}` (was live `poke_env/.../natures.json`)
- `gen3_learnset.json` — species id → `[move_id, ...]` gen3 legal movepool (the hard legality gate the
  move-belief prior uses to prune impossible candidate moves; via `gen3_data.learnset`)

Smogon-derived priors (probabilistic), via `tools/smogon_stats_downloader/`:
- `gen3_smogon_stats.json` (raw aggregated stats) → `gen3_ability_priors.json`, `gen3_hidden_power_priors.json`

All are loaded once (lazy singletons) and raise `FileNotFoundError` / `ValueError` if missing or
empty. The data layer is poke-env-free; the only poke-env touches left in the battle layer are a
parser sentinel (`GenData.UNKNOWN_ITEM`) and the `to_id_str` string util — neither is static data.

---

## Event-Sourced Battle Layer (`src/agents/battle/`, ai_v4)

poke-env is a **state tracker** (each `|...|` line overwrites "current board"); RL/reward/replay
need *what happened, in order*. The event-sourced layer (`Gen3Battle` + `BattleEvent` log) gives
that without reimplementing poke-env, and our non-`battle/` code reads battle state **only**
through the read-models it exposes — `LiveView` (current board), `TurnView` (history fold),
`LegalActions` (server-authoritative legality) — via the `StrictBattleView` boundary
(`battle.strict_view()`). The boundary is enforced by `src/agents/strict_api_lock_test.py` (the
lock) + the `src/agents/enums.py` seam. The per-decision `TurnDelta` history block folds entirely
from the event log (`build_from_events`) and feeds the obs turn-history.

**This is live and consumed across training; ai_v4 is closed out.** The full design — every
read-model, the injection seam, the per-decision event window, the TurnDelta fold's
value-identity findings, and the verification harnesses — is documented in
**`src/agents/battle/CLAUDE.md`** (as-built record: `designs/ai_v4/impl_step8_strict_battle_api_and_turndelta_fold.md`).
