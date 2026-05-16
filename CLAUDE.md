# CLAUDE.md — Gen3AI Project Guide

## Development Stage

**Rapid iteration — checkpoint compatibility is not a concern.** We have no trained checkpoints worth preserving. Breaking changes to the observation space, network architecture, or action space are fine. Do not add backwards-compatibility shims or hesitate to change dims, layer sizes, or layouts.

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

The full observation is a **1107-dim float32 vector** (`Gen3ObservationEncoder.dimension`):

| Block | Dims | Offset | Notes |
|---|---|---|---|
| Our team (6 × 59) | 354 | 0 | base encoder |
| Opp team (6 × 59) | 354 | 354 | base encoder |
| Active context ×2 | 46 | 708 | base encoder |
| Global env | 13 | 754 | base encoder |
| Reactive + matchups | 300 | 767 | base encoder |
| Prev-turn action mask | 11 | 1067 | appended by `gen3_env.embed_battle()` |
| TurnDelta block | 29 | 1078 | appended by `gen3_env.embed_battle()` |

Per-Pokémon slot (59 dims): species ID + 6 base stats, item ID + known, 2 type IDs, ability ID + known, 7-dim condition (status one-hot), 4 × 9-dim move slots, HP fraction, species_known flag, active flag. `species_known = 1.0` for all populated slots (own team and revealed opponent mons), `0.0` for unseen opponent slots.

Global env (13 dims): weather one-hot (6), spikes ×2 (2), log-turn (1), our reflect (1), our light screen (1), opp reflect (1), opp light screen (1).

TurnDelta block (29 dims): our_move_id (raw int), our_power_norm, our_has_secondary, our_has_recoil, our_type_id (raw int), opp_move_id (raw int), opp_power_norm, opp_has_secondary, opp_has_recoil, opp_type_id (raw int), our_switched, opp_switched, our_failed_to_move, opp_failed_to_move, our_cant_onehot (5), opp_cant_onehot (5), our_hp_delta, opp_hp_delta, we_fainted, opp_fainted, opp_move_known. The extractor embeds the 4 raw IDs through shared move/type embedding tables, producing a 89-dim block for the projection. All zeros on the first turn of each episode. See `src/agents/observation/turn_delta_encoder.py`.

---

## Feature Extractor Architecture

`Gen3FeaturesExtractor` in `src/agents/model/features_extractor.py`:

1. **Embedding lookups** — species (32), move (16), item (16), ability (16), type (16, shared for both Pokémon and move types, and for TurnDelta move/type IDs)
2. **Shared move processor** — Linear(58→64)→ReLU→Linear(64→32) per move slot; input includes move/type embeddings, power/secondary/recoil/category remnants, known flag, battle context, and per-move type matchup against all 6 opponents
3. **Within-Pokémon move self-attention** — MHA(32, 2 heads) + LayerNorm residual across the 4 move slots of each Pokémon; lets the role encoder see "this mon has two physical attackers"
4. **Role encoder** — Linear(260→256)→ReLU→Linear(256→128) per Pokémon; input is the full enriched Pokémon vector (242 dims, including `species_known` flag) + broadcasted global context + validity bits
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
