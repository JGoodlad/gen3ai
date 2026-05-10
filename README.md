# Gen3AI: Advanced Pokémon AI for Gen 3 OU

This repository is dedicated to building a high-performance, custom-modeled AI agent for playing Generation 3 Overused (OU) Pokémon battles. It utilizes `poke-env` for the reinforcement learning environment and a local Pokémon Showdown server for training and evaluation.

## 🚀 Project Overview

The goal is to move beyond simple heuristic-based bots and develop a model that understands the unique mechanics of the ADV (Gen 3) metagame, including:
- **No Physical/Special Split**: Damage types are determined by the move's type.
- **Sandstream Dominance**: Managing weather effects (Tyranitar).
- **Spikes & Rapid Spin**: The importance of entry hazards and hazard removal.
- **Precision Switching**: Predicting opponent moves in a high-stakes environment.

## 📂 Repository Structure

- `tools/`
  - `pokemon-showdown/`: A local, high-performance instance of the Pokémon Showdown server. Configured with `--no-security` for faster AI training.
  - `poke-env/`: The Python development environment.
    - `venv/`: Virtual environment containing `poke-env` and its dependencies.
    - `first_battle.py`: A quick-start script to verify the connection between the AI and the server.

## 🛠 Getting Started

You can run everything from the root directory using `npm` scripts:

### 1. Start the Showdown Server
```bash
npm run showdown
```

### 2. Run your AI Agents
```bash
npm run battle
```

*(Note: These scripts point directly to the `tools/` directory so you don't have to `cd` around.)*

## 📈 Long-term Goals
- [ ] Implement a custom neural network architecture tailored for Gen 3 state representation.
- [ ] Train via Self-Play Reinforcement Learning.
- [ ] Evaluate against high-ladder ADV OU players.
- [ ] Integrate with the `poketeam` logic engine for team-building optimization.

---

*Built with love for the ADV community.*
