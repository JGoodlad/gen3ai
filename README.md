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
  - `agents/`: Custom AI agent implementations.
  - `utils/`: Shared utilities (e.g., Gen 3 Hidden Power logic).
- `tools/`: 1st-party scripts and developer tools.
  - `run_battle.py`: High-level script for running/testing battles.

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

*(Note: These scripts point to `deps/` and `tools/` so you never have to `cd` manually.)*

## 📈 Long-term Goals
- [ ] Implement a custom neural network architecture tailored for Gen 3 state representation.
- [ ] Train via Self-Play Reinforcement Learning.
- [ ] Evaluate against high-ladder ADV OU players.
- [ ] Integrate with the `poketeam` logic engine for team-building optimization.

---

*Built with love for the ADV community.*
