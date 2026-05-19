# CLAUDE.md — Gen3AI Project Guide

## Development Stage

**Rapid iteration — checkpoint compatibility is not a concern.** Breaking changes to the observation space, network architecture, or action space are fine. Do not add backwards-compatibility shims or hesitate to change dims, layer sizes, or layouts.

Architecture constants (embedding dims, layer sizes, etc.) are defined as module-level constants in `src/agents/model/features_extractor.py` — that is the single source of truth. When you change one, change it there and nowhere else. See [Model Versioning](#model-versioning).

## Git Workflow

This is a personal project — no pull requests needed. Merge completed work directly to `main` and push:

```bash
git push origin <worktree-branch>:main
```

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
| `*_integration_test.py` | `deps/pokemon-showdown` Node bridge (no live server) | `@pytest.mark.integration` |
| `*_e2e_test.py` | Live Showdown server on localhost:8000 | `@pytest.mark.e2e` (scripts only, run directly) |

### Unit tests only (default)
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not integration and not e2e" -q
```

### Unit + integration (requires symlinked deps/pokemon-showdown)
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -q
```

### E2E tests (run directly as scripts, require a running server)
```bash
# Start server first: npm run showdown
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/fuzz_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/telemetry_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py [n_battles]
```

---

## Smoke Test

Before a full training run, verify the full pipeline (env, reward, replay callback, stall detection, evaluation) with a quick debug run (~1 min):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug --steps 10000
```

What to look for:
- `[ModelVersion] Round-trip smoke test PASSED` — serialization and reload are healthy (printed early, before training begins)
- `🏁 Episode Finished` lines appearing throughout — episodes completing and resetting
- `🎥 [REPLAY]` fires once early (step 1), then training continues — replay callback works
- `[STALL LOGGED]` may appear if a 250-turn game occurs — should be followed by another `🏁 Episode Finished`, not a hang
- `Win rate vs Random` and `Win rate vs Heuristic` printed at the end — evaluation ran

A hang after `[STALL LOGGED]` or a crash before "Training complete" indicates a regression in the env/stall/forfeit pipeline. A `[ModelVersion] FATAL` error at startup means the checkpoint was saved with a different architecture than the current code.

---

## Launcher (preferred for long runs)

`src/main/launcher.py` wraps `train_rl_agent.py` with:
- **Periodic restarts** — kills and relaunches the child every N hours to reclaim pymalloc fragmentation; the child saves a checkpoint on SIGTERM and the launcher picks it up automatically
- **Worktree isolation** — at startup, creates a detached git worktree pinned to the current HEAD (or to the commit recorded in the checkpoint's `metadata.json` when resuming). Agent pushes to `main` never affect a running session
- **Rich TUI** — live dashboard showing metrics, FPS, restart countdown; `l` for logs, `r` to restart now, `c` for forced checkpoint, `q` to quit cleanly
- **Crash reporting** — child stdout/stderr is captured; on a non-zero exit the last 100 lines are dumped to the terminal after the TUI closes

### Exit codes (`src/main/exit_codes.py`)

| Code | `TrainExitCode` | Meaning |
|------|----------------|---------|
| 0 | `COMPLETE` | All steps done — launcher stops |
| 15 | `INTERRUPTED` | SIGTERM received, checkpoint saved — launcher restarts |
| 1 | `CRASH` | Unhandled exception — launcher stops, crash log printed |

### Starting a fresh run via launcher
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/launcher.py \
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
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/launcher.py \
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
| `--no-pin` | off | Skip worktree creation; run from the current source tree |

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

---

## Playing / Evaluation

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/play.py
```

Requires the Showdown server to be running (see below).

---

## Showdown Server

Start the local Pokémon Showdown instance:

```bash
npm run showdown
```

Stop it:

```bash
npm run stop
```

The server must be running for any battles (training or play). It binds to port 8000 by default.

---

## Repository Structure

```
src/
  agents/
    model/           # Gen3FeaturesExtractor (PyTorch feature extractor)
    observation/     # Observation encoders (state_encoder, pokemon, moves, etc.)
    action/          # Action masking and mapping
    training/        # Callbacks and reward manager
  main/
    launcher.py        # Restart loop + Rich TUI (preferred for long runs)
    launcher_ui.py     # TUI state and rendering (LauncherState, LauncherUI)
    exit_codes.py      # TrainExitCode enum (COMPLETE=0, INTERRUPTED=15, CRASH=1)
    train_rl_agent.py  # Training entry point (also callable directly)
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

The full observation is a **1141-dim float32 vector** (`Gen3ObservationEncoder.dimension`):

| Block | Dims | Offset | Notes |
|---|---|---|---|
| Our team (6 × 61) | 366 | 0 | base encoder |
| Opp team (6 × 61) | 366 | 366 | base encoder |
| Active context ×2 | 46 | 732 | base encoder |
| Global env | 13 | 778 | base encoder |
| Reactive + matchups | 300 | 791 | base encoder |
| Prev-turn action mask | 11 | 1091 | appended by `gen3_env.embed_battle()` |
| TurnDelta block | 39 | 1102 | appended by `gen3_env.embed_battle()` |

Per-Pokémon slot (61 dims): species ID + 6 base stats, item ID + known, 2 type IDs, ability ID + known, 7-dim condition (status one-hot), 4 × 9-dim move slots, HP fraction, species_known flag, active flag, sleep_counter_norm, toxic_counter_norm. `species_known = 1.0` for all populated slots (own team and revealed opponent mons), `0.0` for unseen opponent slots. Sleep counter: `min(turns_slept, 4) / 4` (Gen 3 max 4 turns); toxic counter: `min(turns_poisoned, 8) / 8` (practical max before fainting with Leftovers).

Global env (13 dims): weather one-hot (6), spikes ×2 (2), log-turn (1), our reflect (1), our light screen (1), opp reflect (1), opp light screen (1).

TurnDelta block (39 dims): our_move_id (raw int), our_power_norm, our_has_secondary, our_has_recoil, our_type_id (raw int), opp_move_id (raw int), opp_power_norm, opp_has_secondary, opp_has_recoil, opp_type_id (raw int), our_switched, opp_switched, our_failed_to_move, opp_failed_to_move, our_cant_onehot (5), opp_cant_onehot (5), our_hp_delta, opp_hp_delta, we_fainted, opp_fainted, opp_move_known, our_effectiveness_onehot (4: immune/resisted/normal/SE), opp_effectiveness_onehot (4), move_order (2: we_first/opp_first, all-zero=na). The extractor embeds the 4 raw IDs through shared move/type embedding tables, producing a 89-dim block for the projection. All zeros on the first turn of each episode. See `src/agents/observation/turn_delta_encoder.py`.

---

## Feature Extractor Architecture

`Gen3FeaturesExtractor` in `src/agents/model/features_extractor.py`:

1. **Embedding lookups** — species (32), move (16), item (16), ability (16), type (16, shared for both Pokémon and move types, and for TurnDelta move/type IDs)
2. **Shared move processor** — Linear(58→64)→ReLU→Linear(64→32) per move slot; input includes move/type embeddings, power/secondary/recoil/category remnants, known flag, battle context, and per-move type matchup against all 6 opponents
3. **Within-Pokémon move self-attention** — MHA(32, 2 heads) + LayerNorm residual across the 4 move slots of each Pokémon; lets the role encoder see "this mon has two physical attackers"
4. **Role encoder** — Linear(262→256)→ReLU→Linear(256→128) per Pokémon; input is the full enriched Pokémon vector (244 dims, including `species_known` flag + 2 status counters) + broadcasted global context + validity bits
5. **Team-wide attention** — five `MultiheadAttention` paths with residuals (fainted slots masked/zeroed throughout):
   - *Pressure*: our active ← their team (what threatens us right now)
   - *Safety*: our team ← their active (what can switch in safely)
   - *Synergy*: our team ← our team (internal team cohesion)
   - *Threat*: their team ← our active (which of their bench counters us most)
   - *Opp Synergy*: their team ← their team (opponent team cohesion)
6. **Attention pool** — one learned query per side attends over the 6 role tokens (fainted key-masked) producing a single 128-dim pooled team token per side; replaces the slot-order flatten for permutation equivariance
7. **Pre-projection LayerNorm** — normalises the concatenated projection input to equalise per-block scales (embeddings, 0/1 validity bits, HP fractions, ±1 TurnDelta deltas)
8. **Projection** — Linear(562→512)→ReLU; input is: our_pool(128) + their_pool(128) + our_active_refined(128) + active_ctx_enc(32×2) + global+scalars(29) + turn_delta_embedded(89)

The projection input dimension is discovered automatically via a dummy forward pass in `__init__`, so no magic constant needs updating when the architecture changes.

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

## Current State (End of Step 1)

Step 1 is complete. The pipeline is stable and all known correctness issues are resolved.

**What was hardened in Step 1:**

- `BattleContext` / `TurnDelta` / `SlotRegistry` — clean per-turn state, no battle object mutation
- `EpisodeTracker` — owns all per-episode mutable state; extension point for turn history
- `StallConfig` / `StallLogger` — stall detection extracted, shared by env and inference
- poke-env `env.py` — three lifecycle bugs fixed: stale battle queue, forfeit popup, stale `_choose_move()` hang
- Action mask — struggle double-enabling fixed; mask correctness audited and tested
- `RLPlayer` / `StatTrackingRLPlayer` — exception-safe `choose_move()`, stall detection on inference

**Ready for Step 2: Turn History + Action Mask as Features**

- `EpisodeTracker._history` holds the full episode; cap to `deque(maxlen=N)` for N-frame history
- Each `BattleContext` already carries `obs` and `mask` — no new fields needed
- Extend `Gen3ObservationEncoder` to accept `history: list[BattleContext]` alongside the current frame
