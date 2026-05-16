# Gen3AI: Advanced Pokémon AI for Gen 3 OU

This repository is dedicated to building a high-performance, custom-modeled AI agent for playing Generation 3 Overused (OU) Pokémon battles. It utilizes `poke-env` for the reinforcement learning environment and a local Pokémon Showdown server for training and evaluation.

## 🚀 Project Overview

The goal is to move beyond simple heuristic-based bots and develop a model that understands the unique mechanics of the ADV (Gen 3) metagame, including:
- **No Physical/Special Split**: Damage types are determined by the move's type.
- **Sandstream Dominance**: Managing weather effects (Tyranitar).
- **Spikes & Rapid Spin**: The importance of entry hazards and hazard removal.
- **Precision Switching**: Predicting opponent moves in a high-stakes environment.

## 📂 Repository Structure

- `deps/`: Third-party dependencies and environments.
  - `pokemon-showdown/`: Local instance of the Showdown server (Git Submodule).
  - `venv/`: Python virtual environment.
- `src/`: 1st-party core project logic.
  - `main/`: Entry points for the application (e.g., `play.py`).
  - `agents/`: Custom AI agent implementations.
  - `utils/`: Shared utilities (e.g., Gen 3 Hidden Power logic).
- `tools/`: 1st-party developer tools and scripts.

## 🛠 Installation & Setup

### Cloning the Repository
Since this project uses a Git Submodule for the Showdown server, clone it using:
```bash
git clone --recursive <repo-url>
```
*If you've already cloned it without `--recursive`, run:*
```bash
git submodule update --init --recursive
```

### Initial Setup
Run the following to install all Python and Node.js dependencies:
```bash
npm run setup
```

## 🎮 Running the Project

### Start the Showdown Server
```bash
npm run showdown
```

To stop it cleanly (use this instead of Ctrl+C, which orphans subprocesses):
```bash
npm run stop
```

The server runs on port 8000. Config is at `deps/pokemon-showdown/config/config.js` — subprocess counts require a full restart to take effect, but most other settings are picked up live.

### Play / Evaluate
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/play.py
```

### Sync Sample Teams (Smogon)
Automatically downloads and indexes the latest ADV OU sample teams.
```bash
npm run sync-teams
```

### TensorBoard
```bash
cd ~/dev/gen3ai && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/tensorboard --logdir ./tensorboard/ --host 0.0.0.0 --port 6006
```

## 🧪 Testing Suite

We maintain two distinct test suites to ensure both logic and infrastructure integrity:

### 1. Unit Tests (Fast, No Server Required)
Used for verifying local logic like Gen 3 IV fixes and teambuilder parsing.
```bash
npm test
```

### 2. Integration Tests (Local Validator Bridge)
Validates that all downloaded teams are legal and valid using a local Node.js bridge to the Showdown library. This is fast and does not require a running server.
```bash
npm run test-all
```

*(Note: These scripts point to `deps/` and `tools/` so you never have to `cd` manually.)*

## 🧠 Training & Reinforcement Learning

The project uses Stable Baselines3 (PPO) to train a Gen 3 OU agent. The system features a modular observation space and an entity-based embedding layer for Pokémon species.

### Key Components
- **Observation Space (1021 dims)**: Encodes team state, active context, global env, and type matchups.
- **Species Embedding**: Learned 32-dimensional latent vector per species; moves (16), items (16), abilities (16), types (16) also embedded.
- **Fail-Fast Mappings**: Training validates all metadata JSONs in `data/pokemon/` before starting.

### Start a New Training Run
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

### Continue from a Checkpoint
Add `--model <path>` to resume from a saved checkpoint:
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

Checkpoints are saved to `models/` automatically. TensorBoard logs always write to `./tensorboard/` in the repo root regardless of which worktree training is launched from.

### Debug Mode
Use `--debug` to run with a single environment (DummyVecEnv) and full trace logging — useful for inspecting observations, rewards, and action masks without spinning up 96 envs:
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug
```

---

## 📈 Long-term Goals
- [ ] Implement a custom neural network architecture tailored for Gen 3 state representation.
- [ ] Train via Self-Play Reinforcement Learning.
- [ ] Evaluate against high-ladder ADV OU players.
- [ ] Integrate with the `poketeam` logic engine for team-building optimization.

---

*Built with love for the ADV community.*
