# Gen3AI: Advanced Pokémon AI for Gen 3 OU

Reinforcement learning agent for Generation 3 Overused Pokémon battles, built on `poke-env` and a local Pokémon Showdown server.

## Project Goals

- Learn strategic play specific to ADV Gen 3: no physical/special split, Sandstream weather, Spikes/Rapid Spin, and high-stakes switching
- Train via PPO against a diverse opponent pool (random, heuristic, staller, aggressive, setup sweeper)
- Evaluate against progressively stronger opponents

---

## Environment Setup

Uses the **`gen3ai_stable` conda environment** — not `deps/venv` (outdated, ignore it).

Always prefix Python commands with:
```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 <script>
```

### Git Worktrees
When opening a new worktree, the `deps/pokemon-showdown` submodule is empty. Symlink it from the main repo:
```bash
rmdir deps/pokemon-showdown
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown deps/pokemon-showdown
```

---

## Showdown Server

```bash
# Start (with performance flags)
NODE_ENV=production node --turbo-fast-api-calls --max-old-space-size=2048 deps/pokemon-showdown/pokemon-showdown start --no-security

# Stop cleanly (Ctrl+C orphans subprocesses — use this instead)
npm run stop
```

The server runs on port 8000. Key config at `deps/pokemon-showdown/config/config.js` — subprocess counts (`simulator`, `network`) require a full restart; most other settings reload live.

---

## Testing

Three tiers of tests — run from the repo root:

| Pattern | Requires | Command |
|---|---|---|
| `*_test.py` | Nothing (pure unit tests) | See below |
| `*_integration_test.py` | Symlinked `deps/pokemon-showdown` Node bridge | See below |
| `*_e2e_test.py` | Live Showdown server on `localhost:8000` | Run directly as scripts |

### Unit tests only
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not integration and not e2e" -q
```

### Unit + integration
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -q
```

### E2E tests (requires running server)
```bash
# Start the server first, then:
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/fuzz_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/telemetry_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/gen3_env_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py
```

---

## Training

### New run
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --steps 50000000 \
  --n-envs 96 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

### Continue from checkpoint
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --model models/<run-name>/final_model \
  --steps 50000000 \
  --n-envs 96 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

### Debug mode
Single environment with full trace logging — no 96-env overhead:
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug
```

Checkpoints save to `models/` automatically. TensorBoard logs always write to `./tensorboard/` in the repo root, regardless of which worktree training is launched from.

### TensorBoard
```bash
cd ~/dev/gen3ai && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/tensorboard --logdir ./tensorboard/ --host 0.0.0.0 --port 6006
```

---

## Play / Evaluate

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/play.py
```

Requires the Showdown server to be running.

---

## Repository Structure

```
src/
  agents/
    action/          # Action masking, mapping, and fuzz tests
    inference/       # RLPlayer — loads a model checkpoint and battles
    model/           # Gen3FeaturesExtractor (PyTorch)
    observation/     # Observation encoders: species, moves, items, abilities,
                     #   active context, global env, reactive/matchups
    opponents/       # Scripted opponents: staller, aggressive, setup sweeper
    training/        # Gen3Env, reward manager, battle context, wrappers,
                     #   stall detection, replay recorder
  main/
    train_rl_agent.py  # Training entry point
    play.py            # Battle / evaluation entry point
  poke_env/          # Forked poke-env library
  utils/             # Gen 3 utilities, team loader, teambuilder, logging
data/
  pokemon/           # JSON mappings: gen3_species, gen3_moves, gen3_items, gen3_abilities
  teams/             # ADV OU sample teams pool
models/              # Saved PPO checkpoints
tensorboard/         # Training logs (always written here from any worktree)
deps/
  pokemon-showdown/  # Git submodule — local Showdown server
designs/             # Architecture design docs
```

---

## Observation Vector (1021-dim float32)

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 55) | 330 | 0 |
| Opp team (6 × 55) | 330 | 330 |
| Active context ×2 | 44 | 660 |
| Global env | 13 | 704 |
| Reactive + matchups | 304 | 717 |

Per-Pokémon slot (55 dims): species ID + 6 base stats, item ID + known flag, 2 type IDs, ability ID + known flag, 8-dim status one-hot, 4 × 8-dim move slots, HP fraction, active flag.

Global env (13 dims): weather one-hot (6), spikes ×2 (2), log-turn (1), our reflect (1), our light screen (1), opp reflect (1), opp light screen (1).

---

## Model Architecture (`Gen3FeaturesExtractor`)

1. **Embedding lookups** — species (32-dim), move (16), item (16), ability (16), type (16, shared)
2. **Shared move processor** — Linear(55→64)→ReLU→Linear(64→32) per move slot, including per-move type matchup against all 6 opponents
3. **Role encoder** — Linear(237→256)→ReLU→Linear(256→128) per Pokémon, with broadcasted global context
4. **Team-wide attention** (three `MultiheadAttention` paths with residuals):
   - *Pressure*: our active ← their team
   - *Safety*: our team ← their active
   - *Synergy*: our team ← our team
5. **Projection** — Linear(N→512)→ReLU over concatenated team tokens + remaining context

The projection input dimension `N` is computed via a dummy forward pass at init — no magic constants.

---

## Data Dependencies

Training requires JSON files in `data/pokemon/`:
- `gen3_species.json` — `{num, baseStats}`
- `gen3_moves.json` — `{num, basePower, type, hasSecondary, hasRecoil}`
- `gen3_items.json` — `{num}`
- `gen3_abilities.json` — `{num}`

---

*Built with love for the ADV community.*
