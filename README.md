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

You can run everything from the root directory using `npm` scripts:

### 1. Start the Showdown Server
```bash
npm run showdown
```

### 2. Run your AI Agents
```bash
npm run battle
```

### 3. Sync Sample Teams (Smogon)
Automatically downloads and indexes the latest ADV OU sample teams.
```bash
npm run sync-teams
```

## 🧪 Testing Suite

We maintain two distinct test suites to ensure both logic and infrastructure integrity:

### 1. Unit Tests (Fast, No Server Required)
Used for verifying local logic like Gen 3 IV fixes and teambuilder parsing.
```bash
npm test
```

### 2. Integration Tests (Requires Showdown Server)
Validates that all downloaded teams are actually legal and valid within the current Pokémon Showdown environment.
```bash
# 1. Start the server in one terminal
npm run showdown

# 2. Run the integration suite in another
PYTHONPATH=src deps/venv/bin/pytest -m integration
```

*(Note: These scripts point to `deps/` and `tools/` so you never have to `cd` manually.)*

## 📈 Long-term Goals
- [ ] Implement a custom neural network architecture tailored for Gen 3 state representation.
- [ ] Train via Self-Play Reinforcement Learning.
- [ ] Evaluate against high-ladder ADV OU players.
- [ ] Integrate with the `poketeam` logic engine for team-building optimization.

---

*Built with love for the ADV community.*
