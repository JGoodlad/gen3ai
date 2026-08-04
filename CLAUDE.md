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
#                                  poke_env_gaps/team_signature_fuzz_test.py (the --zarch-lut team signature is
#                                      CONSTANT within a battle AND matches the offline table — a drifting signature would
#                                      re-condition the policy mid-game; a mismatched one silently makes the LUT a no-op),
#                                  poke_env_gaps/damage_op_probe_fuzz_test.py (AUTHORITATIVE DamageOperator physics gate — CONSTRUCTED single-turn
#                                      scenarios via the OMNISCIENT BattleStream `utils/bridge/damage_probe.js`: exact both-side HP + the sim's OWN
#                                      stats, zero measurement confounds; one modifier per scenario [type/STAB/SE/resist/4×/immunity/Thick Fat/
#                                      Choice Band/item/boosts/burn/screens/weather]) + poke_env_gaps/damage_op_fuzz_test.py (looser random-game net),
#                                  training/hidden_power_tracker_fuzz_test.py,
#                                  utils/bridge/reconstruction_fuzz_test.py (battle replay/re-roll invariants),
#                                  utils/bridge/reroll_many_parity_fuzz_test.py (batched reroll_many == per-call reroll_turn, bit-for-bit obs),
#                                  utils/bridge/search_clone_parity_fuzz_test.py (serializeBattle clone == reroll_many, bit-for-bit obs + value_crn anchor + depth-2),
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
server connection**; add `--use-bridge=node` (or `--use-bridge=rust`) to make it fully serverless
(the in-process sim runs the training battles too, no Showdown server at all). To also exercise the
eval pipeline (final win-rate eval, and the self-play seed → pool eval → promotion path under
`--self-play`), add `--debug-eval` — that path needs a server (default `:8000`) or
`--use-bridge={node,rust}`.

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

### In-process bridge transport (`--use-bridge {off,node,rust}`, opt-in)

`--use-bridge` (default `off` = websocket) swaps **both training and eval** from a websocket
Showdown server to an in-process `BattleStream` subprocess — no server, no port, no
`/challenge` connection storm, deterministic delivery (poke-env issue #907). **A run needs no
Showdown server at all.** It reuses the *entire* obs/reward/mask/wrapper stack unchanged.

**Two bridge impls:** `--use-bridge=node` is the Node `local_sim_bridge.js` (the original bridge
behavior). `--use-bridge=rust` swaps the child binary for the std-only pokesim
`src/rust_sim/src/bin/sim_bridge.rs` — a byte-for-byte protocol-compatible drop-in (validated by
`src/rust_sim/harness/gen_sim_bridge_diff.js`), so nothing above the transport changes. The
`--use-showdown-bridge` boolean flag is a **DEPRECATED back-compat alias for `--use-bridge=node`**
(kept because the launcher + existing scripts pass it); both resolve into one internal
`bridge_enabled: bool` + `bridge_impl: "node"|"rust"`, and if both are passed they must agree.
`bridge_impl` is threaded to `attach_bridge_transport(env, …, impl=…)` (training),
`run_local_battles(…, impl=…)` (eval driver), and the eval-worker shard config
(`bridge_impl` alongside `use_showdown_bridge`). The binary is resolved by
`src/utils/bridge/sim_bridge_bin.py::resolve_sim_bridge_bin()`: `$POKESIM_SIM_BRIDGE_BIN`
(absolute override) first, else `cargo build --release --bin sim_bridge` in `src/rust_sim`
(cached; a clear error, never a silent node fall-back, if cargo/crate/binary is missing).

**`rust` deferrals + coverage limit (honest, warned at startup; re-audited 2026-08-03).** The old
"the Rust bridge emits **no `__RECON__`**" claim is **STALE** — `2b826d4` shipped both
`gen3_bridge_recon_record_v1` and `gen3_bridge_resume_reseed_v1`, and a **seeded** rust battle
passes the whole forensic stack: `reconstruction_fuzz_test` (replay reproduced the rust-recorded
winner + turn), `reroll_many_parity_fuzz_test`, and `search_clone_parity_fuzz_test` (the clone's
successor obs equals the rust-recorded `states.npz` next obs **bit-for-bit**) all PASS when the
record is fed to the Node `replay_driver.js` / `search_driver.js` — a strong cross-impl parity
result, since the offline replay/clone layer is Node either way. What is **actually** still broken:

- **The SEEDLESS path — the one every production caller uses — produces nothing.**
  `sim_bridge.rs::emit_recon` early-returns on an empty seed, and a `None` seed builds the battle
  from the FIXED `DEFAULT_CONSTRUCT_SEED = "0,0,0,0"` (`state.rs`) instead of minting a random one
  the way Showdown does. `bridge_session.py` (training) and `eval_worker` → `run_local_battles`
  (eval) both pass **no seed**, so under `--use-bridge=rust`: (a) eval traces get **no
  `*_reconstruction.json` sibling** → prober `falsify` / `better-line` / `replay-counterfactual`
  are unavailable, and (b) **every training episode replays one dice stream**. Every existing gate
  (`sim_bridge_bin_test`, `gen_sim_bridge_diff.js`) is inherently SEEDED, which is why this shipped.
- **A STRING `seed` is SILENTLY IGNORED** — `handle_start` parses only the `[a,b,c,d]` array form,
  so `"1,2,3,4"` / `"sodium,<hex>"` fall through to `0,0,0,0`. Node honors both identically. A rust
  re-run of a **node**-recorded record (whose resolved seed is `"sodium,…"`) therefore silently
  replays a *different battle* — the GIGO class, not a loud failure.
- **`resumeReseed` accepts ONLY the array form**, but its one production producer
  (`prober.falsifier.fresh_seeds`) emits the `"a,b,c,d"` STRING that Node's `new PRNG()` requires —
  so the counterfactual Monte-Carlo **hard-errors** on rust today (`START: resumeReseed needs both
  turn and seed`) despite the startup warning calling it supported.
- **The clone-and-branch SEARCH server has no rust path** (`Battle::serialize`/`deserialize` are
  `todo!()`) and `teacher/generate.py` calls `run_local_battles` with no `impl`, so
  `train_rl_agent`'s error on `--search-teacher`/`--teacher-persistent` + `rust` is **still the
  right verdict** — for those reasons, not the stale "no `__RECON__`" one.

**Coverage (MEASURED, not asserted).** The port fail-louds (`__ERR__` → `RuntimeError` → env crash
→ launcher restart; the child survives via `catch_unwind`) rather than desync. On the **training
pool it is a non-issue**: 719/719 `data/teams/` teams construct, and 1500 random-play rust battles
hit **zero** coverage errors (the only 4 failures were the 1000-turn runaway cap). On
`gen3randombattle` the remaining construction failures are down to **Forecast/Castform alone
(~2.6% of teams → ~5% of battles, measured over 3000 generated teams)**: the Deoxys/Unown forme
DATA gap is fixed (`gen3_species_formes_v1`, ROUND 38), `transform` is modeled
(`gen3_transform_v1`, ROUND 33) and the wrap family too (`gen3_partial_trap_v1`, ROUND 32).
Arbitrary ladder gen3ou is ~5% of battles (INFERRED from Smogon usage weights). The old inverse
hazard — unmodeled items/moves running as SILENT no-ops / generic hits — is **CLOSED by the
ROUND 39/40 silent-no-op audits**: the 5 genuinely-effectful unmodeled items
(`gen3_unmodeled_item_failloud_v1`) and the 16 silent-desync moves (`fakeout`, `rollout`, the
lock-in family, `eruption`, … — `gen3_unmodeled_move_failloud_v2`; full-universe measurement:
369 gen3-legal moves → 278 modeled, 74 runtime fail-louds, 16 ran unguarded) now FAIL LOUD at
construction, and every gen3 ability is modeled, verified-no-op, or fail-loud (Forecast).
Measured exposure of the guarded sets is ZERO on both surfaces (0 pool carriers; 0 in the entire
curated randbats movepool) — latent-hazard guards, not active-bug fixes. (The former
**seeded speed-tied-lead** / unspecified-gender divergence is FIXED —
`gen3_turn0_construction_v1` models the turn-0 construction window, so a seeded rust battle is
byte-for-byte with node.) See `src/utils/bridge/README.md`. Transport parity (poke-env sends
move-ids/species names, e.g. `move hiddenpowerice`) is guarded by
`src/utils/bridge/bridge_impl_parity_test.py`.

**`rust` is FASTER than node, and its child is ~25× smaller** (`bridge_impl_throughput_benchmark.py`,
a same-invocation A/B of N parallel env workers on an idle 16-core box): at 8 workers **1.18×**
(1852 → 2182 steps/s), at the production `--n-envs 48` **1.41×** (1942 → 2729 steps/s), with the
bridge child at **9 MB RSS vs node's ~224 MB** — so the children cost ~0.4 GB under rust vs ~10.7 GB
under node at `n_envs=48`. (An older note recording node 798 vs rust 427 fps at 8 envs was measured
on a CPU-saturated box and is superseded.) **`gen3_bridge_forfeit_win_v1` (2026-07-31) was the
blocker that made rust unusable for training**: `FORCELOSE` emitted a bare `__END__` with no `|win|`
line, so poke-env never marked the battle finished and the next `reset()` hung forever. Since the
training seam forfeits whenever `reset()` lands mid-battle, every episode boundary could wedge —
the multi-env soaks logged finished episodes yet completed ZERO PPO iterations. Fixed in
`BridgeSession::forfeit`; the durable gate is `bridge_session_fuzz_test.py --impl rust` (its
every-9th-episode forfeit-reset is the reproducer, and `--impl` exists because that fuzz previously
only ever tested node — the coverage hole that let this ship).

It reuses the *entire* obs/reward/mask/wrapper stack unchanged:

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

### Compiled CPU opponents (`--compile-extractor`, opt-in)

`torch.compile`s each frozen self-play OPPONENT's feature extractor in the env workers — the measured
**68% of rollout-worker time**, run on CPU at B=1 where the graph is dispatch-bound. A **runtime perf
knob**: never versioned, never in `check_compatible`, NOT inherited on resume — re-pass it each launch
like `--grad-checkpointing`.

**Measured: B=1 CPU forward 6.371 → 0.976 ms (6.53×)** on the literal production arch (1 graph, 0
graph breaks, max|Δ| vs eager 5.07e-07), and **+33.3% marginal training FPS at `--n-envs 48`**
(406.5 → 541.8, disjoint ranges, 48/48 workers compiled) — the first throughput lever here the
`SubprocVecEnv` barrier does NOT absorb. But the per-forward win has **saturated**: doubling it
(3.6× → 6.53×) moved end-to-end only ~31% → ~33%, so the opponent forward is no longer the rollout
bottleneck and further compiler work on this path is spent effort.

Startup: `agents.model.compile_prewarm` warms the shared on-disk Inductor cache in the trainer before
any worker exists, halving worker startup (**59.6 s -> 30.1 s** wall for 16 workers). Going further —
a `set_forkserver_preload` that compiles ONCE and lets workers inherit it (0.12 s each) — **was built
and HUNG a 48-env run**; forking is only safe from a single-threaded process and the extractor import
starts poke-env's global asyncio loop thread. See the training leaf before retrying it.
Failure is loud on stderr + the launcher event stream, and `--compile-extractor-strict` promotes it to
a hard error (falling back to eager is an invisible ~6.5× regression). "The model still compiles" is a
**default-on test** (`species_posterior_compiles_test.py`; `GEN3AI_SKIP_COMPILE_TESTS=1` opts out).

Full detail — the four guards, the Inductor crash root-caused to one op, and the startup-cost table —
is in `src/agents/training/CLAUDE.md` → Compiled CPU opponents.

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
**shard units** (`--eval-shard-games`, default 25 → 4 shards/opponent; per-opponent game count
overridable with `--eval-games`) so any idle worker drains a
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
main.prober.query summary|list|scan|overview|find|analyze|lookahead|better-line|replay-counterfactual|falsify|falsify-scan|calibration`)
exposes the same probing infrastructure programmatically — list/filter battles, **`scan` the worst turn in
every loss across an opponent (model-free, ranked)**, digest one battle, find
decisions the model disagrees with, deeply analyze one decision, **`lookahead`
a decision: one-ply VALUE-DELTA — re-roll each legal action one turn (opp plays its recorded move),
materialize the successor, and read the critic's V(s′) per action** (the model-scored counterpart to
`falsify`), **`better-line` a decision: SEARCH for a better trajectory — a shallow CRN-anchored BEAM
over the critic that branches a tree by CLONING mid-battle states (the warm `serializeBattle`
search-server, `utils/bridge/search_session.py`), returning ONE contrastive line ("at turn T do X
instead" + per-ply ΔV/ΔP(win)); opponent RECORDED at the divergence ply, reloaded policy at interior
plies; `--depth N`, `--confirm-rollouts N` for a rollout-to-end win-% confirmation; the depth-≥2
generalization of `lookahead`, bridge-eval traces only**, **`replay-counterfactual` a decision: substitute a move and play the rest LIVE vs the
RELOADED real opponent to a win/loss — "could it have won if it hadn't choked this turn?"** (a scripted
prefix over `run_local_battles`; `--rollouts N` resamples the post-divergence dice for a Monte-Carlo
win-prob ± Wilson CI; bridge-eval traces only), **`falsify`
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
                     #   arch_constants.py — the weight-shape dims (single source of truth)
                     #   damage_op.py      — DamageOperator + decode_damage_block (split out 2026-08-01;
                     #                       re-exported by features_extractor, so old imports still work)
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

The full observation is a **2889-dim float32 vector** (`Gen3ObservationEncoder.dimension`):

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 110) | 660 | 0 |
| Opp team (6 × 110) | 660 | 660 |
| Active context ×2 (boosts + full volatiles, `VOLATILE_DIM`=44) | 116 | 1320 |
| Global env | 18 | 1436 |
| Reactive scalars (11) + matchups (288) + **active-req-moves** (12) | 311 | 1454 |
| Prev-turn action mask | 11 | 1765 |
| Turn history (`N_HISTORY_TURNS` × 159) | 1113 | 1776 |
| **Total** | **2889** | |

**`gen3_cpu_damage_deleted_v1` (v48) — the `--unified-obs` DELETE step.** Three CPU obs regions that
`--unified-obs` previously only **masked** from the model are now removed from the encoder entirely
(reactive 414 → 311, obs 2992 → 2889): the **51-dim incoming-damage / OHKO belief**, the **44-dim
action-aligned move-effect block**, and the **8 active-move scalars** (base-power ×4 + type-multiplier
×4). All three had live GPU homes (the `DamageOperator`'s incoming/outgoing blocks from the LEARNED
move belief, the `MoveLatentEncoder` move latent, the v27/v37 status-landing), so the masks existed
only to A/B the replacement — that A/B is settled and the producers are gone. The three
`--mask-*-obs` flags and `--unified-obs` are deleted with them. **This is a pure CPU refund on the
dominant rollout cost centre:** obs build was ~73% of per-decision controllable CPU, and the measured
benchmark moved **7,396 → 6,444 calls/encode (−12.9%)** with `state_encoder.encode` 0.456 → 0.363 ms
(−20%). `agents.observation.incoming_damage` (the math core) **stays** — the reward PBRS and the
prober both import it; only the obs WRITE is removed. Retrain-class: the obs-dim weight-field check
auto-rejects every pre-v48 checkpoint (no `ARCH_SIGNATURE` bump needed).

**The full per-block layout** — the 110-dim per-Pokémon slot (incl. a 3-dim
`gen3_sleep_wake_belief_v1` block: `sleep_is_deterministic` [Rest], computed `p_wake`, and
`sleep_counter_reliable` — zeros unless the mon is asleep), the 11-dim move slot, the 18-dim
spread block, global env, the 414-dim reactive block (**19 scalars** — the 14 prior + the
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
288 matchup + the **12-dim active-req-moves block** (`gen3_op_move_align_v1`: OUR active mon's 4 moves in
**REQUEST order** — `[move_num ×4, resolved_type_id ×4, legal_now ×4]` — the DamageOperator's OUTGOING
per-move blocks read THIS so their per-move output aligns with action logit 6+k, instead of the per-mon
block's sorted-by-id order; sits after the matchups, consumed only by the op via ObsUnpack, never the
raw-scalar path)), and the 159-dim
TurnDelta slot (incl. the embedded-ID manifest) — lives in **`src/agents/observation/CLAUDE.md`**.
Every offset is computed
from named constants; never hardcode indices.

---

## Feature Extractor Architecture

`Gen3FeaturesExtractor` in `src/agents/model/features_extractor.py` is decomposed into named
phase `nn.Module`s chained by a thin orchestrator:

**`ObsUnpack` → `PokemonEncoder` → `[BeliefSlots?]` → `TeamTransformer` → `[BeliefHead?]` →
`[MoveBelief?]` → `CLSPool` → `[DamageOperator?]` → `ProjectionAssembler`**, then **two** root projection
heads (policy + value), each `pre_proj_norm` → `projection` → `ReLU`. Under `--damage-op-prefuse` (v50)
the `[MoveBelief?] → [SpreadBelief?] → [HPTypeBelief?] → [DamageOperator?]` group moves **before**
`TeamTransformer` and runs exactly once, injecting its per-our-mon incoming rows onto our role tokens
via a zero-init `prefuse_proj`; the same block still feeds both heads. The bracketed phases are
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
+ pins revealed moves certain, so the belief is a unified posterior — *known certain, unknown prior⊕learned*;
`--move-belief-prefuse` moves this reinjection BEFORE the transformer so the believed moves co-refine through
attention instead of being grafted on after);
and `DamageOperator` (`--damage-op`) consumes that move belief's predicted moves to compute the believed-move
incoming damage to each of our mons (a differentiable gen3 calc), appended to **both** projection heads —
so the gradient sharpens the move belief toward real KO threats (`designs/ai_v6/design_differentiable_damage_op.md`);
its effect block carries per-status SECONDARY probabilities (incoming opp threat + per-OUR-move outgoing,
accuracy-folded, ×Serene Grace / Shield Dust — `gen3_unified_move_system_v1`). Under `--damage-topk K` it
ALSO emits a **DISCRETE top-K incoming block** (`gen3_unified_topk_incoming_v1`): the opp active's K
most-believed moves INDIVIDUALLY, each with its move LATENT identity (gathered from `MoveLatentEncoder`,
typed-HP-aware) + belief + per-OUR-mon `[high, pko, status_lands]` — so the policy reasons in the discrete
move space (anticipate the move, pick the damage-/status-immune safe pivot, e.g. Thunder-Wave→Ground=0)
instead of only the collapsed worst-case. Inside `PokemonEncoder`, the
flag-gated `MoveLatentEncoder` (`--move-latent`) concatenates a context-free mechanics-grounded per-move
latent into the move network; its latent table is the Stage-3 similarity-grading target
(`--move-belief-latent-coef`, so Rock Slide ≈ Hidden Power Rock — `designs/ai_v6/design_unified_move_system.md`).
A separate optional `WinProbHead` (`--win-prob-mode none|read_only|shaping`) reads `value_pooled` and emits a
calibrated **P(win)** logit — a SIDE readout (stashed for the aux loss + the prober, **never** in pi/vf, so
projection dims are unchanged), supervised by the Monte-Carlo episode outcome (win=1/loss=0); `read_only`
stop-grads its input (a risk-free diagnostic), `shaping` lets the win objective shape the trunk. A sibling
`PubValHead` (`--pubval-mode`, v43) applies the same pattern with an EXOGENOUS target — the frozen
human-replay-calibrated public value V_pub (`data/gen3_pubval.json`), a dense per-step credit-assignment
signal from outside the self-play bootstrap (the v43 note below).
`forward` returns a `(pi_features, vf_features)` tuple — the transformer body is shared, but the
actor and critic read it through independent CLS pools and projection heads (the
**value-dedicated CLS readout**, H4 / Option C). It must be paired with
`Gen3DualHeadMaskablePolicy` (`src/agents/model/policy.py`), which unpacks the tuple and routes
each half to its own `mlp_extractor` branch; stock SB3 policies assume a single-tensor extractor
and won't work. **The action head is POINTER-NATIVE (v51, `gen3_pointer_native_v1`, no flag):**
the policy's `_build` deletes SB3's flat `action_net` (a raising stub takes its slot) and the
`PointerNativeActionHead` scores each action from the token of the entity it selects — move logit
k ← the REQUEST-slot-k move token ⊕ its op cells, switch logit j ← our-team token j ⊕ its
incoming/OAX cells, struggle ← the context — with `latent_pi` (the policy tower's output, so the
op block / beliefs / FiLM all condition it) as the shared context. Position-equivariant by
construction; zero-init scorers ⇒ uniform-over-legal cold start. Both projection input dims are
auto-discovered via a dummy forward pass in `__init__`, so they stay correct as the architecture
changes.

**The phase-by-phase data flow (the 7-phase contract, dims, and the `ExtractorContext` /
`Embeddings` ownership rules) is documented in `src/agents/model/CLAUDE.md`.**

---

## Model Versioning

Every model save writes two **run-level** files at the run root:

- `model_config.json` — all weight-shape-relevant architecture params (embedding dims, layer sizes, obs dim, etc.)
- `metadata.json` — git hash, timestamp, SB3/Python versions (+ `snapshot_history`, `latest_eval`,
  and run provenance: `cli_args` [the latest process's full argparse namespace], `launcher_command`,
  and `original_command` — the **immutable** original invocation that created the model, written once
  at creation and preserved verbatim across every restart/checkpoint, unlike `cli_args` which is
  overwritten by the resuming process. Provenance lives ONLY here, never in `model_config.json`, which
  is the weight-shape/arch record used for `check_compatible`)

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
**`gen3_typed_hp_belief_v1`** — the v52 discrete typed-HP belief, stacking directly on
`gen3_pointer_native_v1` (the v51 pointer-native action head, the fresh-generation cross-era break —
the flat positional `action_net` is deleted and every action is scored from the token of the entity it
selects; see the v51 entry below). Under it the model **only ever reasons over DISCRETE typed Hidden Power**. The
presence×type composition `P(HP_t) = presence · P(type=t)` happens ONCE, in `HPTypeBelief.compose_typed_hp`,
right beside the move-belief head; from that point on the posterior carries HP at its 16 real typed move-nums
**355-370** and the bare typeless **237** is driven hard-off (a finite `-30` logit, not `-inf`, so the BCE sees
~0 loss and no NaN). 237 survives only as the belief's internal PRESENCE channel, read immediately before it is
masked. It supersedes `gen3_opp_hp_typed_candidates_v1`, which had made only the DamageOperator typed while the
belief, its labels, its prior, the token reinjection and the latent grading still spoke in 237.
**The invariant this buys is structural**: `Σ_t P(HP_t) == presence`, and presence is reveal-pinned, so once
the opponent has been SEEN using Hidden Power the belief can be unsure WHICH type it is but can never conclude
there is none — no penalty term, no coefficient. Two certain facts eliminate candidates first: **moveset
exhaustion** (4 moves revealed, none of them HP ⇒ presence 0, derived from `opp_move_ids` alone) and
**effectiveness narrowing** (the `HiddenPowerTracker`'s hard zeros in the obs `hp_probs` are certain physics, so
the type belief is restricted to the survivors and renormalised — with a uniform-over-survivors fallback so an
off-meta HP can never be renormalised back to "immune"). The **`--hp-type-belief` mode flag is DELETED**: its
`off` state was a correctness bug behind a flag (a typeless BP-0 candidate, and a REVEALED HP priced as
nonexistent because the obs `hp_probs` it sourced the type from is empty until HP actually fires), so the head is
unconditional whenever there is a move belief — and it no longer requires `--damage-op`, since the composition
lives in the belief. The op is now a plain consumer (no `hp_type_fix`, no `SPECIES_HP_PRIOR`, no
`hp_type_belief` argument), which also closes a real divergence: `forward` used to get the learned posterior
while `refine_candidates` did not, so the between-layers refine kernels priced HP off a different belief than the
head block. The **move-belief LABELS are now the TRUE TYPED num** (`gen3_env._move_num` no longer folds to 237) —
they used to supervise a dead channel while leaving the 16 typed ones as BCE negatives, i.e. actively training
"this opponent has no Hidden Power of any type". Leak-safety is unchanged: the labels are training-only Dict keys
(the same privileged fact `hp_type_label` already carried) and the OBSERVATION still shows the opponent's HP bare,
so the model must still guess the type. **The one deliberate hold-out is the TURN-HISTORY opp-move slot**, which
keeps num 237 — the history records what was OBSERVED, and the type genuinely was not. OUR-side HP carries its
distinct num + real type in the obs/history throughout (`gen3_typed_hidden_power_ids_v1`). A data-derived `HP_TYPED_NUMS` + a throwing GIGO
guard pin the 355-370 ↔ `HP_TYPE_ORDER` alignment. The prober decodes the op's typed-HP candidates via the
NORMAL move-name path (`hiddenpower(ice)`) — no HP-special collapse. Design:
`designs/ai_v6/design_typed_hidden_power_ids.md` + the model leaf's v38 note. It supersedes
`gen3_own_hp_typed_history_v1` (the hp_probs one-hot workaround is reverted) and stacks on
`gen3_op_move_align_v1` (the request-ordered active-req-moves block — `REACTIVE_DIM` 402 → 414, obs dim
3457 → 3469) and the prior `gen3_rest_loop_stall_v1` rest-loop clock re-meaning, back through
`gen3_wish_wired_v1` — which WIRES two reactive scalars (`vec[17]` our side, `vec[18]` opp side) with the
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
`MODEL_CONFIG_VERSION`: **38** (the v33–v38 additions are the bolded entries below) — v16 added the in-place
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
versioning); the distributional aux loss + capture/prober/launcher are follow-ons.
**v30 the DISCRETE top-K incoming move-space** (`gen3_unified_topk_incoming_v1`, `damage_topk_k` /
`--damage-topk`) — the `DamageOperator`'s incoming block collapses the opp active's whole moveset into the
worst phys/spec hit per defender (`_chan_max`), hiding WHICH move it is + the per-pivot consequences. This
adds a discrete block: for the opp active's **K most-believed CANDIDATE moves** (default K=5, auto-on under
`--unified-moves`; a mon runs 4 moves so the 5th is the surprise candidate) it surfaces — per move — its
move **LATENT** identity (gathered from the `MoveLatentEncoder`, incl. **typed-HP** rows so HP-Rock ≠
HP-Ice; differentiable → sharpens the latent) + belief weight (differentiable → sharpens the move belief)
+ accuracy + is_phys, then **per OUR mon** `[high, pko, status_lands]` — so the policy can anticipate the
discrete move AND pick the immune/safe pivot (damage-immune pivot = 0 from the chart; status-immune pivot,
e.g. **Thunder Wave → a Ground mon**, = 0 via `_incoming_status_lands`). Decorrelated physics (the belief
gradient rides the `w` feature, not the damage); the 5th slot is zeroed once all 4 opp moves are revealed;
added ALONGSIDE the worst-case `_chan_max` summary (the §4.3 hybrid). STRUCTURAL int (scales `out_dim` by
`K·53` → both projections; gated in `check_compatible` like `opp_belief_cls_k`; OFF=0 byte-identical, no
`ARCH_SIGNATURE` bump); requires `--damage-op` + `--move-latent`; threaded through `arch_toggles`; the
prober decodes exact move names from the stashed `last_topk_idx`.
**v31 the DAMAGE RE-ATTEND** (`gen3_damage_reattend_v1`, `damage_reattend` / `--damage-reattend`) — lets
attention reason OVER the computed physics (today the `DamageOperator` block is a POST-pool concat no
attention sees). When on, after the op computes the damage, its per-OUR-mon INCOMING rows are projected
(small-init, identity-at-init) onto the 6 our-team tokens, ONE more `TransformerEncoderLayer` re-attends the
12 team tokens (our↔opp), and the CLS pools are derived ONCE on the re-attended tokens — so the pi/vf pools
are **damage-AWARE board summaries** instead of damage-blind ones. It is a BOARD-level enrichment (the
"needs a per-bench pointer head" follow-up it originally deferred landed at v51 — the pointer head reads
the re-attended `our_team_out` per token, so a bench token now flows straight into its own switch
logit). STRUCTURAL like `opp_belief_slots` (adds 3 modules; re-pooling keeps
the pooled shapes ⇒ projection widths UNCHANGED; gated in `check_compatible`, OFF byte-identical, NO
`ARCH_SIGNATURE` bump); requires `--damage-op`; threaded through `arch_toggles`; PopArt strongly recommended
(soft-warns without it).
**v32 the MOVE-BELIEF PRE-FUSE** (`gen3_move_prefuse_v1`, `move_belief_prefuse` / `--move-belief-prefuse`) —
moves the `MoveBelief` reinjection from POST-transformer (the default — believed moves grafted onto the
already-refined opp tokens) to PRE-transformer (reinjected into the opp ROLE tokens before the body), so the
predicted moves **co-refine** with the species/team belief through the 2 attention layers. Same `MoveBelief`
module/params (one shared `_apply_move_belief` helper, only the input tensor + timing differ; the stashed
`last_move_belief_logits` is identical, so the damage op + BCE aux still read it) → state_dict identical,
projection widths unchanged. FORWARD-BEHAVIOR toggle like `move_prior_fusion` (gated in `check_compatible`,
OFF byte-identical, NO `ARCH_SIGNATURE` bump); requires `--move-belief-mode != off`; threaded through
`arch_toggles`.
**v33 ITERATIVE damage refinement** (`gen3_iterative_damage_v1`, `damage_refine_rounds` /
`--damage-refine-rounds N`) — the `DamageOperator` runs ONCE post-transformer (a one-shot read of the FINAL
belief). This recomputes a LEAN per-our-mon incoming-damage summary BETWEEN transformer layers — as the opp
token (hence the move belief) is enriched by attention — and injects it back onto our-mon tokens, so each
layer attends over physics from the FRESHEST belief (physics-in-the-loop), and the per-round read sharpens
the move-belief head. `TeamTransformer.forward` gains a `between_layers` callback (before each of the first
N layers); per round it re-reads the belief (`MoveBelief.move_logits`, the posterior — factored out of
`forward`), computes a LEAN `DamageOperator.discrete_incoming → [B,6,4]` `[phys_high, spec_high, phys_pko,
spec_pko]` (top-`_DMG_REFINE_K`=8 candidates, reusing the validated `_rolls` physics — ~50× cheaper than the
full ~416 sweep, so the per-round recompute is cheap), and injects via a **zero-init `refine_proj`** Linear
(true identity-at-init, gradient still flows; weight-tied across rounds → N-independent shape). STRUCTURAL int
gated in `check_compatible` (0↔N a state_dict change, N↔M a forward change; OFF=0 byte-identical, no
`ARCH_SIGNATURE` bump); requires `--damage-op` only (NOT `--move-latent`); NOT auto-set by `--unified-moves`
(an explicit A/B lever); threaded through `arch_toggles` + both extractor-kwargs sites.
**v34 the OUTGOING per-move DAMAGE MATRIX** (`gen3_per_move_matrices_v1`, `damage_matrices_outgoing` /
`--damage-matrices outgoing`) — the legacy outgoing block prices our active's 4 moves vs the opp ACTIVE
only; this adds `DamageOperator._outgoing_matrix`: our 4 moves × the opp's **6 mons** (active + REVEALED
bench), per (move, opp mon) `[low,high,crit,pko,type_mult]` + a per-opp-mon `revealed` bit — so the policy
prices a KO on a **switch-in** (the equal-effectiveness tie-break extended to bench targets). REVEALED-gated
(unrevealed opp slots zeroed — Gen3 has no team preview; belief-driven outgoing-vs-unrevealed is a TODO);
reuses the validated `_outgoing_block` physics broadcast over 6 defenders (the active column is byte-for-byte
the single-active block). STRUCTURAL bool toggle gated in `check_compatible` like `damage_op`; OFF
byte-identical (no `ARCH_SIGNATURE` bump); requires `--damage-op`; threaded through `arch_toggles` + both
extractor-kwargs sites.
**v35 the INCOMING per-move DAMAGE MATRIX** (`gen3_per_move_matrices_v1`, `damage_matrices_incoming` /
`--damage-matrices incoming`) — the ENRICHED evolution of the v30 top-K block (`_incoming_matrix`,
REUSES `--damage-topk K` as its K — one knob, try 4/5/6 — and replaces the lean top-K block at that K).
Per opp-active top-K move: a richer header `[latent, belief, acc,
is_phys, EXPLICIT effect bits(6: recovery/status/phaze/boost/hazard/protect), EXPLICIT secondary chances(10)]`
+ a richer per-(OUR mon, move) cell `[low,high,crit,pko,type_mult,status_lands]`. The effect/secondary bits
are **gathered PER MOVE** (un-collapsed — the mid-ladder "this move phazes / flinches" nuance the worst-case
`p_effect`/`p_sec` maxes collapsed; those are kept-but-superseded, deletion deferred to an A/B). Reuses the
validated `_damage_rolls` tensors + the candidate latent table; STRUCTURAL bool gated in `check_compatible`
like `damage_op`; OFF byte-identical; requires `--damage-op` + `--move-latent`. The two matrices compose
under `--damage-matrices {off,incoming,outgoing,both}`.
**v36 the BIDIRECTIONAL in-trunk THREAT field** (`gen3_bidir_threat_trunk_v1`) — makes the model's threat,
BOTH directions, dynamic (known⊕believed) and INFUSED INTO THE TRUNK so attention reasons over it. Three
toggles: **`--threat-refine-outgoing`** (#1) the SYMMETRIC mirror of the incoming refine — a new lean
`DamageOperator.discrete_outgoing` (our active's 4 known moves → each opp mon → `[phys_high,spec_high,
phys_pko,spec_pko]`) injected onto the OPP token slice via a **zero-init `outgoing_proj`** riding the SAME
between-layers `--damage-refine-rounds` loop (STRUCTURAL — a saved weight; requires `--damage-op` +
`--damage-refine-rounds>0`); **`--threat-unrevealed-outgoing`** (#2) the EXPECTED-LATENT defender — keep an
UNREVEALED opp mon LATENT and marginalize the move-belief's `P(species)` (read per-round from the factored
`BeliefHead.species_logits`) through `SPECIES_EXP_MULT[n_species,19]` (type chart × the per-species expected
ability immunity — Levitate/Water&Volt Absorb/Flash Fire/Thick Fat, folded from `gen3_ability_priors`) +
`SPECIES_SPREAD_PRIOR` (E[bulk]/E[maxhp]), with **P(KO) NULLED** (a full-HP switch-in is ~never OHKO'd —
owner decision, drops the Jensen-threshold complexity; forward toggle, no new params; requires
`--threat-refine-outgoing` + `--opp-belief-aux-coef>0`); **`--threat-prob-outspeed`** (#3) UNCERTAINTY-AWARE
`P(outspeed)` — divide the speed gap by the believed speed STD (`SPECIES_SPREAD_PRIOR`; sigmoid≈normal-CDF)
not a fixed scale (forward toggle, no new params). Needs a NEW data fact — **species→types** (added to the
extractor → `gen3_species.json` → `SpeciesData.types`; the obs still reads revealed types live). All three
OFF byte-identical (NO `ARCH_SIGNATURE` bump), version-gated, threaded through `arch_toggles` + both
extractor sites.
**v37 STATUS-LANDING into the trunk** (`gen3_status_trunk_v1`, `threat_status_refine` /
`--threat-status-refine`) — the LAST CPU-obs deprecation gap. The move-effect block's board-conditional
`status_will_land` was heads-only (v27 `_status_landing`); status immunity (type × ability ×
already-statused × Sleep-Clause × Substitute) is a computed MECHANICS fact (the class of type
effectiveness), and LEARNING it would force attention to correlate non-local info (the move's status intent
on one token, the defender's types+ability on another). So COMPUTE it and inject into the trunk, BOTH
directions: **INCOMING** `discrete_incoming_status` (opp active's top-K believed status moves → per OUR mon,
onto OUR tokens — "will I be statused") + **OUTGOING** `discrete_outgoing_status` (our active's status moves
→ per opp mon, revealed-gated, onto OPP tokens — the in-trunk home for the masked `status_will_land`), each
a per-defender `[P(major), P(immobilize=para/frz/slp)]` reusing the v27 status-landing physics + buffers via
two zero-init residuals on the refine loop. The major-vs-immobilize split makes the trunk signal
SELF-CONTAINED (no cross-move correlation). STRUCTURAL bool (adds two Linears); OFF byte-identical (NO
`ARCH_SIGNATURE` bump); requires `--damage-op` + `--damage-refine-rounds>0`; threaded through `arch_toggles`
+ both extractor sites. **Completes the FULL `--unified-obs` deprecation** (verified by a deprecation-gap
audit: every CPU-obs signal has a GPU home — damage→trunk/refine, status→trunk/v37, effects→move latent, PP
→per-mon slot, provenance/p_outspeed/crit→explicit op channels, per-move status_will_land+known→v27 heads;
honest residuals = opp-recovery heads-only + Rest-cure coarsening). The dedicated `pbrs_roar`
phaze-out-boosts PBRS is folded INTO `--all-shaping-pbrs` (no new flag/version, no `ARCH_SIGNATURE` bump).
Current `MODEL_CONFIG_VERSION` = **39** (the v38/v39 additions are the bolded entries below). Full design:
`designs/ai_v6/design_bidirectional_threat_trunk.md` (+ `gen3ai/tmp/{model_v36_full,stacking_levels}.png`)
(and `design_per_move_damage_matrices.md` for v34/v35, `design_iterative_damage_refinement.md` for v33,
`design_topk_incoming_moves.md` for v30, `design_distributional_value_critic.md` for v29,
`design_unified_move_system.md` for v24, `design_unified_damage_system.md` for v23).
**v38 UNIFIED typed-HP candidates + the opponent HIDDEN-POWER-TYPE belief** (SUPERSEDED by v52
`gen3_typed_hp_belief_v1` — the `--hp-type-belief` mode flag and the op-side scatter described here are GONE;
kept for the history of how the "immune" GIGO was first attacked) (`hp_type_belief_mode` /
`--hp-type-belief {off,prior,learned}`) — fixed the
DamageOperator rendering the opponent's Hidden Power as 0-damage/**"immune"** (a prober-surfaced GIGO) by
making HP **16 ordinary typed moves end-to-end**, eliminating the HP special-casing that bred the prober
ambiguity. Builds on main's `gen3_typed_hidden_power_ids_v1` (typed move-nums **355-370** with real BP 70 +
type; bare 237 = BP 0): the op now treats the opp's HP as those **real typed candidates** — the candidate axis
is `C = n_moves` (the synthetic appended-16 expansion, the old 237-collision workaround, is REMOVED), the bare
237 (BP 0) is the masked **presence token**, and the per-type HP belief is scattered onto 355-370. A shared
`DamageOperator._opp_candidate_weights` (the single source for all 3 candidate sites) masks 237 + the raw
355-370 (`HP_CAND_MASK`) and `index_add`s `P(HP present)·P(HP type)` onto `HP_TYPED_NUMS`. Type source —
**`off`**: the obs `hp_probs` (effectiveness-narrowed, the A/B baseline); **`prior`**: the Smogon
`SPECIES_HP_PRIOR` floor (`build_hp_type_prior`); **`learned`**: the `HPTypeBelief` head's posterior
`softmax(head_delta + log prior[species])` (zero-init → cold-start == prior), which the op consumes (its damage
gradient sharpens it) AND which **reinjects** the presence-gated expected typed-HP embedding into the opp token
(attention reasons over the believed type), supervised by a training-only CE
(`instrumented_ppo._hp_type_belief_loss`, `--hp-type-belief-coef`, metrics `belief/hptype_*`) against the
privileged true HP type from agent2's typed move-id (`Gen3Env._hp_type_labels`; the obs keeps the opp HP
typeless 237 → no leak). All on/off the belief NARROWS by the obs `hp_probs` (its effectiveness hard-zeros are
CERTAIN; an off-meta-survivor fallback spreads uniform so it never re-immunes). Multiple un-ruled-out types
stay live (a distribution, not argmax) → the top-K surfaces **hp-ice + hp-grass distinctly at their real nums
(365/363)** with real per-mon damage — the "force the model to guess which HP, simulate each" read. A
data-derived `HP_TYPED_NUMS` + a throwing GIGO guard pin the 355-370 ↔ `HP_TYPE_ORDER` alignment
(`MOVE_TYPE_IDX[355+j]==HP_TYPE_IDX[j]`, `MOVE_BP[237]==0`). The prober decodes the op's typed-HP candidates via
the NORMAL move-name path (`hiddenpower(ice)`) — no HP-special index→type collapse (the old ambiguity is gone).
The op's forward-math changed (out_dim + projection widths UNCHANGED — C is internal — so NOT shape-caught) →
the `ARCH_SIGNATURE` **bump** forces a clean reload of any pre-unification `damage_op` checkpoint;
`hp_type_belief_mode` is STRING-gated in `check_compatible`, the obs VECTOR dim is unchanged (the label is a
separate Dict key), `hp_type_belief_coef` is training-only. Requires `--damage-op`; threaded through
`current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites.
**v39 the TRANSPOSED outgoing DAMAGE MATRIX — switch-in offense** (`gen3_per_move_matrices_v1`;
`damage_matrices_outgoing_all` / `--damage-matrices-outgoing-all`) — the TRANSPOSE of v34's
`damage_matrices_outgoing`. v34 prices our ACTIVE's 4 moves × the opp's 6 mons (broadening the DEFENDER axis);
v39 broadens the ATTACKER axis: `DamageOperator._outgoing_attacker_matrix` prices OUR **6 MONS'** 4 moves → the
opp **ACTIVE** only. Fixes a confirmed high-impact error: today the op's outgoing block prices ONLY the current
active attacker, so on a **forced switch** (active fainted → the single-active block zeroes) the policy picks
switch-ins **BLIND to offense** — this surfaces what every candidate switch-in would DO to the opp active. Per
(attacker mon, move) cell `[low,high,crit,pko]` + a per-attacker `p_outspeed` + an `alive` bit (`_DMG_OAX` =
6·16 + 6 + 6 = **108**). **PARITY (the hard requirement):** the OUR-ACTIVE mon's row reproduces `_outgoing_block`
**byte-for-byte** (its boosts/CB/burn + request-ordered moves + the same opp-active defender + the same `_rolls`
kernel); bench rows reuse the SAME validated physics with **NEUTRAL boosts** (gen3 resets boosts on switch) +
the per-mon sorted-by-id moves (the active slot is overwritten with the request slice so it ties out). STRUCTURAL
bool toggle gated in `check_compatible` like `damage_op` (widens both projections via the op out_dim); OFF
byte-identical (NO `ARCH_SIGNATURE` bump); requires `--damage-op`. Appended LAST (all prior op offsets
untouched); `decode_damage_block(..., matrices_outgoing_all=True)` mirrors it (`outgoing_matrix_all`). Threaded
through `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs`
sites. Design: `designs/ai_v6/design_per_move_damage_matrices.md`.
**v40 the NATURE/EV GENERATIVE spread belief + op nature-marginalization** (`gen3_nature_ev_belief_v1`;
`spread_belief_nature` / `--spread-belief-nature` + `spread_belief_nature_marginalize` /
`--spread-belief-nature-marginalize`) — fixes the `SpreadBelief` head's "over-estimates the largest EV"
order-statistic bias (`belief/spread_largest_bias`). The additive head predicts the DERIVED stat directly (a
point estimate sitting BETWEEN the nature ×1.1/×0.9 modes); **`--spread-belief-nature`** swaps it for a
GENERATIVE head — predict a NATURE categorical ⊕ Smogon log-prior + per-stat EVs ⊕ prior (the move/HP-type
prior-fusion pattern), IV 31, and **COMPUTE** `believed = (2·base + 31 + E[EV]/4 + 5)·E[nature_mult]`. The nature
coupling (one stat ×1.1, one ×0.9) + the EV budget become STRUCTURAL, so the head can't inflate every stat. Same
`believed [B,6,5]` op interface (projection widths UNCHANGED); supervised by nature CE + EV smooth_l1
(`_nature_ev_belief_loss`, folded at `spread_belief_coef`, metrics `belief/natureev_*`) against the TRUE
(nature, EVs) **deterministically INVERTED** from agent2's known `mon.stats` (`invert_nature_evs`, GIGO-guarded;
training-only `belief_nature`/`belief_ev` Dict keys — gen3 hides the opp nature/EVs so no leak).
**`--spread-belief-nature-marginalize`** then makes the op MARGINALISE the nonlinear P(KO) over the believed
nature distribution (3-point quadrature on each candidate's one offensive stat — EXACT — restoring the
asymmetry the mean-field `ko` at E[mult] blurs). `spread_belief_nature` STRUCTURAL (requires `--spread-belief`);
`marginalize` FORWARD-BEHAVIOR (requires it + `--damage-op`); both version-checked, OFF byte-identical (NO
`ARCH_SIGNATURE` bump). Smoke: `nature_acc` rises + `largest_bias` trends to 0.
**v41 the BELIEF TRUNK-GRADIENT MODE** (`gen3_belief_grad_mode_v1`; `belief_grad_mode` /
`--belief-grad-mode {shaping, detached}`) — a knob on whether the four STATE-prediction belief heads
(move / spread / hp-type / the species-moves-latent aux) reshape the shared trunk. `shaping` (default) =
they READ the live trunk so their gradient reshapes it (current behavior); `detached` = they READ a
STOP-GRAD trunk (`opp_tokens.detach()` at the logit-read; reinject WRITE keeps the live identity term) so
NO belief gradient reshapes the trunk — the belief stays computed / reinjected / consumed by the op (fully
"in the system"), it just can't drag the trunk toward predicting hidden state at the policy's expense
(kills the belief↔policy gradient interference). `detach()` is value-preserving → the FORWARD (eval /
frozen pool / distill opponent) is BIT-IDENTICAL; only the TRAINING gradient differs. So it is a
**RESUME-IMMUTABLE training hparam** (the `vf_coef` class): recorded on `ModelVersion`, enforced
resume-only via `check_belief_grad_mode` (intentional migration: `--allow-belief-grad-mode-change`),
EXCLUDED from `check_compatible` (a frozen opponent's forward is
unaffected, so gating it would break self-play). NO `ARCH_SIGNATURE` bump; `shaping` is byte-for-byte the
v40 forward+backward. The win-aligned heads (`--win-prob-mode` / `--value-dist-mode`) keep their own
`read_only`/`shaping`. Design rationale: a representation-rank probe found the 128-dim trunk runs in ~3–5
effective dims, so capacity isn't the constraint — the risk this isolates is gradient interference.
**v42 the TURN-HISTORY DEPTH cut** (`N_HISTORY_TURNS` 10 → 7) — a retrain-class obs-dim change: the
observation drops from 10 to 7 consecutive TurnDelta slots (159 dims each), so the turn-history block is
1113 dims (was 1590) and the total obs is **2992** (was 3469). `n_history_turns`/`total_dim` are already in
`_WEIGHT_FIELDS`, so `check_compatible` auto-rejects any pre-v42 checkpoint on the obs-dim weight-field check
(NO `ARCH_SIGNATURE` bump — the obs-dim weight-field check already catches it).
**v43 the PUBLIC-VALUE aux head** (`gen3_pubval_aux_v1`; `pubval_mode` / `--pubval-mode
{none,read_only,shaping}` + the training-only `--pubval-coef`, default 0.1) — `PubValHead` (the WinProbHead
pattern, a named subclass) reads `value_pooled` and is regressed toward the FROZEN human-replay-calibrated
public value **V_pub = P(win | PUBLIC board)** (`agents.training.pubval` + `data/gen3_pubval.json`: a
17-feature logistic over material/hazards/status/boosts/turn/weather aggregates, fit by `python -m
agents.training.pubval_calibration` on the 170k-game rated gen3ou replay corpus — held-out-by-game AUC 0.734,
turn-1 AUC 0.500 leakage-clean, calibrated). The value-INDEPENDENT exogenous signal (human outcomes, not the
self-play bootstrap) as a DENSE per-step shared-trunk target — the trunk sees WHEN the game swung (the
credit-assignment lever aimed at the measured defensive/positional value blindness). The target rides a
training-only `pubval_target` obs Dict key computed env-side per decision from the LiveView (PUBLIC state
only — leak-free; live↔corpus-parser parity is structural via ONE shared feature definition, guarded
end-to-end by `poke_env_gaps/pubval_parity_fuzz_test.py`). SIDE readout — never in pi/vf, NEVER in GAE
(V^human ≠ V^π). `read_only` = a stop-grad learnability probe ("can the trunk carry V_pub?"); `shaping` = the
human positional prior shapes the trunk (the experiment). STRUCTURAL + resume-immutable string gate (like
`win_prob_mode`); OFF byte-identical (NO `ARCH_SIGNATURE` bump). Metrics `pubval/*` (watch `mae`→0, not the
entropy-floored `bce`) + `grad/pubval_share`; the acceptance gate = the critic's defensive-AUC-by-style
transfer. Design: `designs/ai_v8/design_public_info_value.md`.
**v44 the TEAM-ARCHETYPE latent + head FiLM** (`gen3_zarch_film_v1`; `zarch_film` / `--zarch-film
{off,heads}` + `zarch_dim` / `--zarch-dim` [default 32 = `ZARCH_DIM`] + the training-only
`--zarch-recon-coef` [1.0] / `--zarch-vicreg-coef` [0.1]) — the amortization-gap **STORAGE** fix
(`designs/learning/amortization_gap_and_conditioning.md`: per-team distillation was shown to fix
greedy-local play on distilled teams but NOT generalize to neighbors AND to interfere with the rest —
the literal signature of conflicting per-team strategies cancelling in one shared head). `ZArchEncoder`
builds a **TEAM-STATIC, permutation-invariant DeepSets latent z_arch** over OUR team's **INVARIANT**
facts only — species ⊕ item ⊕ ability ⊕ moves (mean move-emb) ⊕ the 18-dim spread block, per-mon atom
MLP → mean over 6 → LayerNorm — DETERMINISTIC (no VIB sampling in v1: per-forward sampling would break
team-static, PPO's epoch ratio recompute, and eval determinism; LUT-first is the chosen operating
point) with **DETACHED embedding reads** (zero trunk gradient interference, the `belief_grad_mode`
philosophy). Two **zero-init FiLM generators** (one per root head) then modulate the post-projection
pre-ReLU head features `h·(1+Δγ(z)) + Δβ(z)` — identity-at-init, so ON starts byte-identical; per-team
gradients land in different rank-`zarch_dim` subspaces instead of cancelling. Anti-collapse = the
species multi-hot **reconstruction BCE** (a constant z can't reconstruct different teams; Species
Clause ⇒ lossless) + a **VICReg per-dim variance floor** (`zarch/std` is the collapse monitor;
`film/{pi,vf}_{gamma,beta}_norm` the deviation-from-identity read). Coefs auto-zeroed on a single-team
(pinned `--trainee-team`) run (z is constant there → degenerate variance floor; FiLM stays on as a
learned per-team bias). STRUCTURAL: `zarch_film` string + `zarch_dim` int gated in `check_compatible`
(the `value_dist_mode`/`bins` pattern); OFF byte-identical (NO `ARCH_SIGNATURE` bump); requires
nothing (independent of the belief/damage stack); threaded through `current_model_version` /
`arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites.
**v45 the DISTRIBUTIONAL VALUE CRITIC — Phase B** (`gen3_dist_critic_v1`; `value_from_dist` /
`--value-from-dist` + the migration hatch `--allow-value-from-dist-change`) — promotes the v29
`ValueDistHead` from a SIDE readout to the actual CRITIC. When on: GAE / bootstrap / deployment read
**E[Z]** (the distribution's mean, `policy._critic_value` → `head.mean(logits)` → `_denorm`, same
PopArt peg as the scalar), the **HL-Gauss CE becomes the PRIMARY value loss** (weighted by `vf_coef`,
not the aux `value_dist_coef`), and the scalar `value_net` FREEZES as a fallback + the E[Z]-vs-V
monitor (its MSE term dropped from the loss; PopArt still POPs it harmlessly + keeps the μ/σ peg
alive for the CE's normalized targets). The "Stop Regressing" recipe (Farebrother) — a categorical
critic resists the crystallization the scalar MSE breeds. **WARM-STARTABLE** on a `--value-dist-mode
shaping` lineage (the offline probe confirmed E[Z]≈V at pearson 0.988): no state_dict change (both
heads always exist) and the frozen forward's ACTION selection is unchanged, so it is RESUME-IMMUTABLE
(the `belief_grad_mode`/`vf_coef` class) — recorded on `ModelVersion`, enforced resume-only via
`check_value_from_dist`, EXCLUDED from `check_compatible` (gating a frozen opponent would false-reject
self-play). NO `ARCH_SIGNATURE` bump; requires `--value-dist-mode shaping` (the head must be a live
trunk-shaping critic). Threaded as a POLICY kwarg (`value_from_dist`, like `use_popart`) through both
`policy_kwargs` sites + the resume enforce; `value_share` (grad-balance) now points at the CE term.
**v46 the PER-TEAM LUT** (`gen3_zarch_lut_v1`; `zarch_lut` / `--zarch-lut {off,add,only}`) — a FREE,
unconstrained conditioning code per pinned `--trainee-teams` team, layered on the v44 z_arch. It tests
ONE thing: the multi-team exploiter ceiling (N=1 0.84 / N=3 0.835 / N=10 0.825 all distil cleanly, but
**N=20 stalls ~0.66**). The FiLM diagnosis is SNR/ill-conditioning, not capacity — the DeepSets z is
COMPOSITIONAL, so z-similar teams sit at `z̄ + ε_i` with tiny ε and the generator's gradient is
proportional to that residual; a **random-init** LUT makes the codes large and ~orthogonal from step 0,
which is exactly the intervention that story predicts. If N=20 still stalls with a free code, the
ceiling is NOT conditioning signal. `add` = `LN(z_deepsets + code)` (keeps composition — an UNMATCHED
team hits the ZERO-init row 0 ⇒ z is exactly the DeepSets z); `only` = `LN(code)` (the sharpest
ablation). The team is identified **from the OBSERVATION** (`agents.model.team_signature`: sorted
species(6) ⊕ moves(24)) so **no env / eval / prober / frozen-opponent plumbing changes**; species alone
is NOT enough (5 of the def-20 cluster's 20 teams share a roster — that would make the "per-team" code a
per-PAIR code), and `build_roster_table` THROWS on a duplicate signature or a move-set mutator
(Mimic/Transform/Sketch). The GIGO canary is **`zarch/lut_hit_frac`** (must be ~1.0 — a missed lookup
falls through to row 0 and silently makes the experiment a no-op) + `zarch/lut_code_dist`. STRUCTURAL
string + int (`zarch_lut_teams`, the Embedding height) gated in `check_compatible`; OFF byte-identical
(NO `ARCH_SIGNATURE` bump); requires `--zarch-film heads` + `--trainee-teams`.
**v47 the FROZEN pre-attention move belief** (`gen3_belief_single_compute_v1`;
`move_belief_single_compute` / `--move-belief-single-compute`) — compute the move belief **exactly
once** per forward and freeze it. Under `--move-belief-prefuse` the belief is predicted + reinjected
BEFORE the transformer, but the between-layers refine callback then **re-read** `move_logits` off the
(reinjected, then attention-enriched) opp tokens — so the belief was computed **3× in the production
config** (prefuse + one per `--damage-refine-rounds` round) and the physics consumed a *different*
posterior than the one attention was handed. ON, the refine kernels reuse the stashed pre-transformer
logits, giving the intended pipeline: **belief ONCE (pre-attention) → physics ONCE → N attention
layers that CANNOT revise it.** With `--damage-refine-rounds 1` the callback fires only before layer 0
(on pre-attention role tokens), so both transformer layers then reason over frozen physics — the
`next_run_plan` item-3 "prefuse-style, no between-layer recompute" arm, and the shape the owner
specified. The stash is **live, not detached**, so the op's damage gradient still reaches the same
belief computation the reinjection used (one posterior, one gradient path). Also strictly cheaper: one
fewer move-belief head pass per forward. **Cold-start inert by construction** (pinned by a test): under
`--move-prior-fusion` `MoveBelief.move_head` is ZERO-init (posterior == the Smogon prior ⇒
token-independent) AND `refine_proj` is ZERO-init (injection ×0), so frozen and per-round are
byte-identical at step 0 and can only diverge as those paths learn. FORWARD-BEHAVIOR toggle like
`move_belief_prefuse` (same `MoveBelief` params → state_dict identical, projection widths unchanged);
gated in `check_compatible` (bool); OFF byte-identical (NO `ARCH_SIGNATURE` bump); requires
`--move-belief-prefuse` (without it the only belief is POST-transformer, so there is nothing to reuse —
enforced at the CLI and the extractor). Threaded through `current_model_version` /
`arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites. Tests:
`belief_single_compute_test.py`.
**v48 the CPU-DAMAGE DELETION** (`gen3_cpu_damage_deleted_v1`) — the delete step of the
`--unified-obs` deprecation playbook: the 51-dim incoming-damage block, the 44-dim move-effect block
and the 8 active-move scalars are removed from the OBSERVATION (reactive 414 → 311, obs 2992 → 2889),
along with the three `mask_*_obs` `ModelVersion` fields and the `--unified-obs` / `--mask-*-obs` CLI
flags. See the observation-vector section above for the rationale and the measured CPU refund.
`_migrate_config` **POPs** the three dead keys (`from_json_file` does `cls(**data)`, so a stale key
would raise `TypeError` rather than the clear arch error). Retrain-class, caught by the existing
obs-dim weight-field check — NO `ARCH_SIGNATURE` bump.
**v49 the CANDIDATE-AXIS CAP + the POINTER ACTION HEAD.**
**`damage_candidate_k` / `--damage-candidate-k K`** (`gen3_topk_candidates_v1`) — the op priced ALL
~400 move-nums per defender even though the opponent runs four moves; the belief already says which
candidates matter. This caps the INCOMING candidate axis at the K most-believed (per batch row,
selection DETACHED, gathered weights still differentiable so the belief gradient rides the survivors),
with **NO tail bound** — the truncated mass is DROPPED, so a rare-but-lethal candidate below rank K is
simply not priced (the on-policy probe measured top-16 owning 94.2% of channels with misses BIMODAL,
which is why the plan called the tail mandatory; this flag is the explicit trade). `_damage_rolls`'
per-candidate args became `[B,C]` (one call site). Measured: **+11.4% forward / +63.5% op at B=256**
(learner) but only **+0.3% at B=1** — the CPU/PFSP opponent is DISPATCH-bound (~14.3k aten calls at
~0.44 µs), not tensor-size bound, so this is a learner lever, not an eval-latency one.
FORWARD-BEHAVIOR (no new params), unconditional int gate; 0 byte-identical; requires `--damage-op`.
**`pointer_head` / `--pointer-head`** (`gen3_pointer_head_v1`) — score each action FROM THE TOKEN OF
THE ENTITY IT SELECTS: move logit *k* from the move at **REQUEST slot k**, switch logit *j* from
our-team token *j*. Fixes two measured defects structurally rather than by guard: **F2** (switch
logits are read from a permutation-INVARIANT pool, so a bench mon's token never reaches its own logit)
and the **ordering bug class** (`agents/action/ordering_integrity.py` exists solely because the
extractor reads moves SORTED-BY-ID while actions use REQUEST order — the head permutes by move-num
IDENTITY, making a misaligned logit unrepresentable). The per-move tokens already existed inside
`PokemonEncoder` (post move-self-attention, `[B,12,4,32]`) and were merely flattened away; the head
stashes them instead. It ADDS a **zero-init delta** to the flat head's logits (identity-at-init,
warm-starts from any checkpoint, clean A/B) — a guarantee that only actually holds because of the
**M1** fix below. The policy adds it in `_get_action_dist_from_latent`, the single point all three
logit sites funnel through. STRUCTURAL bool gate; OFF byte-identical. NO `ARCH_SIGNATURE` bump (both
OFF reproduce v48 exactly). **SUPERSEDED at v51** — the delta head and its `pointer_head` flag/field
are deleted; the pointer head became THE action head (see v51 below).
**v50 the PRE-ATTENTION UNIFIED DAMAGE OPERATOR** (`gen3_damage_op_prefuse_v1`; `damage_op_prefuse` /
`--damage-op-prefuse`) — ONE damage computation per forward instead of two. Today the op runs **twice**:
a LEAN `discrete_*` recompute inside the between-layers refine loop (×`--damage-refine-rounds`, 2 in the
production config) **plus** the FULL 835-dim block after the transformer. ON, the spread + HP-type
beliefs and the FULL op all run on the **PRE-transformer role tokens**, the per-OUR-mon incoming rows are
injected onto our tokens through a **zero-init `prefuse_proj`** (the `refine_proj` convention →
identity-at-init), and the **SAME full block is concatenated to both heads** — so the ledger-P1 head
dependency is preserved at full width, only its inputs move from refined to un-refined. **The
justification is CPU cost, not architecture:** at B=1 on CPU — the PFSP frozen-opponent regime, which
sits on the rollout critical path — the forward is 6.45 ms / 14,337 aten calls (~0.44 µs each, so
DISPATCH-bound), the op is ~75% of it (2.454 ms post-transformer + ~2.4 ms refine loop) and the attention
layers are 0.27 ms (4%); measured `--damage-refine-rounds` 2→1 = +14.0%, 2→0 = +28.2%. The
"attention now reasons over full-fidelity physics" story is **secondary and unsupported** — physics-into-
the-trunk measured NULL 3-for-3 (**K9/K10**) and the lean kernel was already a 91.8%-agreement proxy for
the full op (**K10a**). Two properties bound the risk, both test-pinned: the op **requires
`--move-belief-prefuse`**, so the move belief — its dominant input — is **bit-identical** in both shapes
and only the spread + HP-type posteriors are re-sourced; and at **cold start the block is bit-identical**
(every belief head is zero-init ⇒ token-independent posteriors), so the divergence is created by
TRAINING, not by the reordering. STRUCTURAL (adds `prefuse_proj`) → bool compare in `check_compatible`;
OFF byte-identical (NO `ARCH_SIGNATURE` bump); **mutually exclusive with `--damage-refine-rounds > 0`**
(the loop is what it replaces — and that also drops the v36/v37 outgoing/status trunk residuals, which
ride that loop and are NOT reproduced). Threaded through `current_model_version` /
`arch_toggles_from_model` / `_run_arch_toggles` + both `extractor_kwargs` sites. **Measured:**
`tmp/pfsp_opponent_sweep.py` B=1 **6.452 → 4.617 ms (+28.2%, −4,126 aten calls)** — the same as
`--damage-refine-rounds 0` alone (4.620 ms), so the CPU win IS the loop deletion and the prefuse rides
along ~free; `tmp/damage_prefuse_kl.py` on 3000 real states puts the head-block re-sourcing at block
cosine **0.988** and masked KL **3.0% of the zero-block ceiling** (2.9% argmax flips) on a short-trained
snapshot — a floor, not a verdict.
**M1 — SB3 was destroying every zero-init in the extractor (FIXED 2026-08-01).** `ActorCriticPolicy._build()`
orthogonally re-inits EVERY `nn.Linear` in the feature extractor (`ortho_init` defaults True, nothing
overrode it), so **13** Linears documented as zero-init were random from step 0 in every real run —
`refine_proj`/`outgoing_proj`/`status_*_proj`/`film_pi`/`film_vf`, plus the belief heads whose
zero-init is what makes the **cold-start posterior equal the Smogon prior**. Guarded by
`Gen3FeaturesExtractor.restore_identity_init()`. **This puts a standing caveat on the K10 and D4
result families** — see `designs/research_state/ledger.md` → M1 and the model leaf.
**v51 the POINTER-NATIVE ACTION HEAD** (`gen3_pointer_native_v1`, NO flag — the fresh-generation
reset, `designs/ai_v9/design_pointer_action_head.md` §0) — the flat positional `action_net` is
DELETED (`Gen3DualHeadMaskablePolicy._build` swaps in a raising stub and rebuilds the optimizer) and
the `PointerNativeActionHead` is THE action head: move logit *k* ← the REQUEST-slot-k move token ⊕
its op cells `[low,high,crit,pko,p_land,known,sec×10]`, switch logit *j* ← our-team token *j* ⊕ its
incoming row + CB tail (+ OAX attacker row under `--damage-matrices-outgoing-all`), struggle ← the
context — with **`latent_pi`** as the shared context, so the op block / beliefs / FiLM condition
every score. Position-EQUIVARIANT (one shared scorer per entity; the `ordering_integrity.py`
sorted-vs-request bug class is unrepresentable at the logits). The op owns the cell layout
(`DamageOperator.pointer_cells`, offsets pinned against `decode_damage_block`); widths are 0 when a
source block is off (the Linear NARROWS, never zero-pads). Zero-init scorers built AFTER SB3's
ortho-init ⇒ cold-start policy is uniform-over-legal. The v49 `pointer_head` field is REMOVED
(`_migrate_config` POPs it); no gate exists because there is no off state — the cross-era break
rides the **`ARCH_SIGNATURE` bump**, so every pre-v51 checkpoint fails loud (owner decision
2026-08-03: no resume/warm-fork across the boundary; pools/opponents re-seed from the new lineage).
**v52 the DISCRETE typed HIDDEN POWER, end to end** (`ARCH_SIGNATURE` `gen3_typed_hp_belief_v1`) — the model
never reasons over a typeless Hidden Power again. See the `ARCH_SIGNATURE` paragraph above for the full
description: the presence×type composition moves into `HPTypeBelief.compose_typed_hp` next to the move head, so
the posterior EVERY consumer reads (damage op, top-K, move BCE, latent grading, token reinjection, prober)
carries HP at the 16 real typed nums 355-370 with the bare 237 hard-off; `Σ_t P(HP_t) = presence` makes "a
revealed HP must exist as some type" structural; moveset exhaustion and effectiveness narrowing eliminate
impossible types; the `--hp-type-belief` mode flag is deleted (its `off` state was a correctness bug) and the
head is unconditional under a move belief, no longer requiring `--damage-op`; the belief LABELS use the true
typed num (they previously trained the typed channels toward zero); and the learnset gate stops marking all 16
typed HPs unlearnable. `_migrate_config` **POPs** the dead `hp_type_belief_mode` key. Retrain-class — the
forward math changed while the projection widths did not, so the `ARCH_SIGNATURE` bump is what catches it.
Tests: `hp_type_belief_test.py` + the extended `poke_env_gaps/belief_labels_fuzz_test.py`.
**v53 the HP-BELIEF FACTORISATION ABLATION** (`gen3_hp_belief_ablation_v1`; `hp_belief_mode` /
`--hp-belief-mode {composed,flat}`) — measures what the v52 presence×type factorisation is actually WORTH.
BOTH arms reason over the DISCRETE typed HP nums 355-370 and drive the bare BP-0 num 237 hard-off via the
shared `mask_typeless_hp` helper — the typeless candidate is the "opp HP reads immune" bug, NOT the variable.
`composed` (DEFAULT) is byte-for-byte v52: the `HPTypeBelief` head, the structural `Σ_t P(HP_t) = presence`
(reveal-pinned) + moveset exhaustion + effectiveness narrowing. `flat` is the ABLATION: NO `HPTypeBelief`
head — the multi-label move head predicts the 16 typed channels INDEPENDENTLY, each off its own real
per-typed Smogon usage prior (the prior table already writes the typed cells' own rates beside the 237
presence sum), i.e. Hidden Power is treated exactly like any other move — no factorisation, no reveal
constraint, no narrowing. STRUCTURAL: `flat` drops a module (a state_dict change as well as a forward one),
STRING-gated in `check_compatible` (the `win_prob_mode` pattern), fresh-only; `_migrate_config` defaults
pre-v53 configs to `composed`. `--hp-belief-mode flat` AUTO-ZEROES `--hp-type-belief-coef` with a loud note
(the ablation builds no head, so there is no posterior for the CE to supervise; the zarch single-team
auto-zero precedent — the coef defaults to 0.05, so erroring would make the ablation flag fail out of the
box). Default byte-identical → NO `ARCH_SIGNATURE` bump. Tests: `hp_type_belief_test.py` (both arms
237-masked, the version gate, the invalid-mode raise, the migration default).
Current `MODEL_CONFIG_VERSION` = **53**, `ARCH_SIGNATURE` = **`gen3_typed_hp_belief_v1`**.
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
- `gen3_species.json` — species id → `{num, baseStats, name, types}` (`types` UPPERCASED to the TypeEncoder
  axis — `gen3_bidir_threat_trunk_v1`, for the op's expected-latent read; the obs still reads revealed types live).
  **419 rows** = the 386 base forms + the 33 gen-3 ALTERNATE/COSMETIC FORMES (`gen3_species_formes_v1`:
  Deoxys-Attack/-Defense/-Speed with their own base stats, the 27 Unown letters, Castform's 3 weather
  formes), each carrying `baseSpecies`. Formes were missing before and cost the `src/rust_sim` port
  **6.6% of gen3 random-battle teams / ~14% of battles** at construction. A forme SHARES its base's
  `num`, and the obs species channel + every `table[species.num]` model buffer are num-keyed — so
  num-indexed consumers MUST iterate `gen3_data.species.base_form_ids()` (see
  `src/agents/gen3_data/CLAUDE.md`)
- `gen3_moves.json` — move id → `{num, basePower, type, accuracy, never_miss, hasSecondary, hasRecoil,
  priority, secondaryEffects {col: percent}, drainFraction, recoilFraction, …}` (the structured
  secondary/priority/drain fields are `gen3_unified_move_system_v1` — GPU-side only, NOT in the obs vector).
  **Typed Hidden Power has distinct nums** (`gen3_typed_hidden_power_ids_v1`): bare `hiddenpower`=237,
  the 16 typed variants=355-370 (Showdown ships them all at 237; the extractor tool overrides — see
  `tools/CLAUDE.md`). OUR known HP uses the distinct num; the opponent's unrevealed HP is the bare 237.
- `gen3_items.json` — item id → `{num, name}` (`num` is the item-dex number; cross-gen aliases share one num)
- `gen3_abilities.json` — ability id → `{num, name}`
- `gen3_type_chart.json` — `{DEF: {ATT: multiplier}}` effectiveness chart (was live `GenData`)
- `gen3_natures.json` — nature → `{num, stat multipliers}` (was live `poke_env/.../natures.json`)
- `gen3_learnset.json` — species id → `[move_id, ...]` gen3 legal movepool (the hard legality gate the
  move-belief prior uses to prune impossible candidate moves; via `gen3_data.learnset`)
- `gen3_move_aliases.json` — `{alias_id: canonical_move_id}` from Showdown's `aliases.ts`
  (`wisp`→`willowisp`, `sd`→`swordsdance`, …). **Consumed ONLY by the `src/rust_sim` port** (its dex
  resolves a packed-team move alias like the RL runtime never touches); the `agents.gen3_data` facade
  does NOT load it, so it is obs-neutral. `gen3_move_alias_resolution_v1`.

Smogon-derived priors (probabilistic), via `tools/smogon_stats_downloader/`:
- `gen3_smogon_stats.json` (raw aggregated stats) → `gen3_ability_priors.json`, `gen3_hidden_power_priors.json`

Pool-derived (a committed calibration artifact, same pattern):
- `data/teams/gen3_team_archetypes.json` — every pool team labeled by PACE class
  (hyper_offense/offense/balance/semi_stall/stall via a transparent composition rubric) + style
  TAGS (sand/spikes/spin/spinblock/phaze/**trap**/**trap_core**/wish/boom/choice/…), keyed by
  `sha1(team_str)[:10]` (the MatchupSpec `pin_sha` convention, so labels join every provenance
  record). Derived by `python -m agents.training.team_archetypes` (a k-means cross-tab prints as
  the unsupervised sanity check); consumed by league targeting (the `trap_core` exploiter
  shortlist) and future archetype-aware team sampling. Loader:
  `agents.training.team_archetypes.load_team_archetypes`.

Human-replay-derived (a committed calibration artifact, like `gen3_bot_elo_anchors.json`):
- `gen3_pubval.json` — the frozen public-value logistic (V_pub, `gen3_pubval_aux_v1`): 17 public-board
  features → P(win), fit on the rated gen3ou replay corpus (`replays/showdown/gen3ou/`, local-only, NOT in
  the repo) by `python -m agents.training.pubval_calibration`; provenance (n_games, AUC, git hash) in `meta`.
  Consumed by `Gen3Env` when `--pubval-mode != none` (via `agents.training.pubval.PubValModel`).

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
