# Gen3AI: Advanced Pokémon AI for Gen 3 OU

Reinforcement learning agent for Generation 3 Overused Pokémon battles, built on `poke-env` and a local Pokémon Showdown server.

## Project Goals

- Learn strategic play specific to ADV Gen 3: no physical/special split, Sandstream weather, Spikes/Rapid Spin, and high-stakes switching
- Train via PPO against a diverse opponent pool (random, heuristic, staller, aggressive, setup sweeper)
- Evaluate against progressively stronger opponents

---

## Environment Setup

Uses the **`gen3ai_stable` conda environment**. To create it from scratch:
```bash
conda env create -f environment.yml
```

To update an existing env after `environment.yml` changes:
```bash
conda env update -f environment.yml
```

Always prefix Python commands with:
```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 <script>
```

### Git Worktrees
When opening a new worktree, the `deps/pokemon-showdown` submodule is empty. Run:
```bash
git submodule update --init
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/dist deps/pokemon-showdown/dist
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/node_modules deps/pokemon-showdown/node_modules
```
Do **not** symlink the entire `deps/pokemon-showdown` directory — it breaks `git status` and VS Code git integration.

---

## Showdown Server

```bash
# Start (with performance flags)
NODE_ENV=production node --turbo-fast-api-calls --max-old-space-size=6144 deps/pokemon-showdown/pokemon-showdown start --no-security

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

### Via launcher (recommended for long runs)

`launcher.py` wraps the training script with periodic restarts to reclaim memory fragmentation, a Rich TUI dashboard, and **git worktree isolation** — it pins the child process to the exact commit at launch so agent pushes to `main` can't affect a running session.

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/launcher.py \
  --restart-interval-hours 3 \
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

Resume from a checkpoint (launcher reads the saved `git_hash` and pins to that commit):
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/launcher.py \
  --restart-interval-hours 3 \
  --model models/<run>/checkpoint_NNNN_steps.zip \
  --steps 50000000 \
  --device cuda
```

Key launcher flags: `--restart-interval-hours` (default 3, set 0 for one-shot), `--no-pin` (skip worktree isolation). All other flags pass through to `train_rl_agent.py`.

**TUI keys:** `r` restart now · `c` force checkpoint · `q` quit cleanly · `l` logs · `d` dashboard

### Direct (no restart loop)

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

### Debug mode
Single environment with full trace logging — no 96-env overhead:
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug
```

Checkpoints save to `models/run_<timestamp>/` automatically. TensorBoard logs always write to `./tensorboard/` in the repo root.

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
    launcher.py        # Restart loop + Rich TUI (preferred entry point)
    launcher_ui.py     # TUI state and rendering
    exit_codes.py      # TrainExitCode enum (COMPLETE/INTERRUPTED/CRASH)
    train_rl_agent.py  # Training script (also callable directly)
    play.py            # Battle / evaluation entry point
  poke_env/          # Forked poke-env library
  utils/             # Gen 3 utilities, team loader, teambuilder, logging
data/
  pokemon/           # JSON mappings: gen3_species, gen3_moves, gen3_items, gen3_abilities
  teams/             # ADV OU sample teams pool
models/              # Saved PPO checkpoints (run_<timestamp>/ subdirs)
tensorboard/         # Training logs (always written here from any worktree)
deps/
  pokemon-showdown/  # Git submodule — local Showdown server
designs/             # Architecture design docs
tools/               # Data generation and team sync utilities
```

---

## Observation Vector (1107-dim float32)

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 59) | 354 | 0 |
| Opp team (6 × 59) | 354 | 354 |
| Active context ×2 | 46 | 708 |
| Global env | 13 | 754 |
| Reactive + matchups | 300 | 767 |
| Prev-turn action mask | 11 | 1067 |
| TurnDelta block | 29 | 1078 |

Per-Pokémon slot (59 dims): species ID + 6 base stats, item ID + known, 2 type IDs, ability ID + known, 7-dim status one-hot, 4 × 9-dim move slots, HP fraction, species_known flag, active flag.

Global env (13 dims): weather one-hot (6), spikes ×2 (2), log-turn (1), our reflect (1), our light screen (1), opp reflect (1), opp light screen (1).

TurnDelta block (29 dims): move/type IDs and metadata for both sides last turn, switch flags, fail flags, cant one-hots, HP deltas, faint flags, opp_move_known. All zeros on the first turn of each episode.

---

## Model Architecture (`Gen3FeaturesExtractor`)

1. **Embedding lookups** — species (32-dim), move (16), item (16), ability (16), type (16, shared across Pokémon types, move types, and TurnDelta IDs)
2. **Shared move processor** — Linear(58→64)→ReLU→Linear(64→32) per move slot; includes per-move type matchup against all 6 opponents
3. **Within-Pokémon move self-attention** — MHA(32, 2 heads) + LayerNorm residual across the 4 move slots of each Pokémon
4. **Role encoder** — Linear(260→256)→ReLU→Linear(256→128) per Pokémon, with broadcasted global context and validity bits
5. **Team-wide attention** (five `MultiheadAttention` paths with residuals, fainted slots masked):
   - *Pressure*: our active ← their team
   - *Safety*: our team ← their active
   - *Synergy*: our team ← our team
   - *Threat*: their team ← our active
   - *Opp Synergy*: their team ← their team
6. **Attention pool** — one learned query per side attends over 6 role tokens → single 128-dim pooled team token per side
7. **Pre-projection LayerNorm** — normalises concatenated inputs to equalise per-block scales
8. **Projection** — Linear(562→512)→ReLU; input: our_pool + their_pool + our_active_refined + active_ctx_enc + global/scalars + turn_delta_embedded

The projection input dimension is discovered via a dummy forward pass at init — no magic constants.

---

## Data Dependencies

Training requires JSON files in `data/pokemon/`:
- `gen3_species.json` — `{num, baseStats}`
- `gen3_moves.json` — `{num, basePower, type, hasSecondary, hasRecoil}`
- `gen3_items.json` — `{num}`
- `gen3_abilities.json` — `{num}`

---

*Built with love for the ADV community.*
