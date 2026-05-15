# CLAUDE.md — Gen3AI Project Guide

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
```

---

## Smoke Test

Before a full training run, verify the full pipeline (env, reward, replay callback, stall detection, evaluation) with a quick debug run (~1 min):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug --steps 10000
```

What to look for:
- `🏁 Episode Finished` lines appearing throughout — episodes completing and resetting
- `🎥 [REPLAY]` fires once early (step 1), then training continues — replay callback works
- `[STALL LOGGED]` may appear if a 250-turn game occurs — should be followed by another `🏁 Episode Finished`, not a hang
- `Win rate vs Random` and `Win rate vs Heuristic` printed at the end — evaluation ran

A hang after `[STALL LOGGED]` or a crash before "Training complete" indicates a regression in the env/stall/forfeit pipeline.

---

## Training

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

Omit `--model` to start a fresh run. Use `--debug` to run with a single env (DummyVecEnv) for debugging. Use `--device cpu` on machines without a GPU.

Checkpoints are saved automatically during training. Models land in `models/`.

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
    train_rl_agent.py  # Training entry point
    play.py            # Battle / evaluation entry point
  poke_env/          # Forked poke-env library
  utils/             # Gen 3 utilities (Hidden Power, teambuilder, team loader)
data/
  pokemon/           # JSON mappings: gen3_species, gen3_moves, gen3_items, gen3_abilities
  teams/             # Downloaded sample teams (gen3ou pool)
models/              # Saved PPO checkpoints
deps/
  pokemon-showdown/  # Git submodule — local Showdown server
```

---

## Observation Vector

The encoded battle state is a **1021-dim float32 vector**:

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 55) | 330 | 0 |
| Opp team (6 × 55) | 330 | 330 |
| Active context ×2 | 44 | 660 |
| Global env | 13 | 704 |
| Reactive + matchups | 304 | 717 |

Per-Pokémon slot (55 dims): species ID + 6 base stats, item ID + known, 2 type IDs, ability ID + known, 8-dim condition (status one-hot), 4 × 8-dim move slots, HP fraction, active flag.

Global env (13 dims): weather one-hot (6), spikes ×2 (2), log-turn (1), our reflect (1), our light screen (1), opp reflect (1), opp light screen (1).

---

## Feature Extractor Architecture

`Gen3FeaturesExtractor` in `src/agents/model/features_extractor.py`:

1. **Embedding lookups** — species (32), move (16), item (16), ability (16), type (16, shared for both Pokémon and move types)
2. **Shared move processor** — Linear(55→64)→ReLU→Linear(64→32) per move slot; input includes move/type embeddings, power/secondary/recoil/category remnants, known flag, battle context, and per-move type matchup against all 6 opponents
3. **Role encoder** — Linear(237→256)→ReLU→Linear(256→128) per Pokémon; input is the full enriched Pokémon vector + broadcasted global context
4. **Team-wide attention** — three `MultiheadAttention` paths with residuals:
   - *Pressure*: our active ← their team (what threatens us right now)
   - *Safety*: our team ← their active (what can switch in safely)
   - *Synergy*: our team ← our team (internal team cohesion)
5. **Projection** — Linear(N→512)→ReLU; input concatenates our refined team (6×128), opponent refined team (6×128), our refined active token (128), and raw remaining context

The projection input dimension `N` is discovered automatically via a dummy forward pass in `__init__`, so no magic constant needs updating when the architecture changes.

---

## Data Dependencies

Training requires JSON mapping files in `data/pokemon/`:
- `gen3_species.json` — species ID → `{num, baseStats}`
- `gen3_moves.json` — move ID → `{num, basePower, type, hasSecondary, hasRecoil}`
- `gen3_items.json` — item ID → `{num}`
- `gen3_abilities.json` — ability ID → `{num}`

These are loaded at startup and will raise `FileNotFoundError` / `ValueError` if missing or empty.
