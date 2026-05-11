# 🧬 Gen3 RL Architecture & Training Guide

This guide covers how to train, debug, and optimize the Gen3 Reinforcement Learning pipeline.

## 🚀 Quick Start (Production Training)
To start a high-performance training run using the full capacity of your RTX 3080 Ti:

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/bin/python3 src/main/train_rl_agent.py --steps 25000000 --n-envs 32 --batch-size 4096
```

## 🔍 Debugging & Deep Traces
If you want to see exactly what the model is "thinking" and verify the state encoding, use the `--debug` flag. This disables parallel workers to allow deep diagnostic traces to print to your console.

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/bin/python3 src/main/train_rl_agent.py --steps 1000 --debug
```

### What to look for in Traces:
- **Team Summaries**: Real-time HP, Status, Items, and Abilities for both teams.
- **Momentum Block**: Type effectiveness matchups for all 4 moves against the current opponent.
- **Integrity Checks**: Automated warnings if the state vector becomes desynchronized.

## 🛠️ Requirements & Troubleshooting
- **Tensorboard (MANDATORY)**: Professional logging is required. The script will fail-fast if `tensorboard` is not installed.
- **Python Path**: Always include `src` in your `PYTHONPATH`.
- **Conda Environment**: Use the absolute path `/home/goodlad/miniconda3/bin/python3` to ensure you are using the environment with `torch` and `stable-baselines3` installed.

## 🧠 Strategic Tuning
The pipeline now includes several "Hardened" features you can tune:

### 1. The Switching Subsidy
To prevent the model from getting stuck in an "attack-only" local optima, we reward the first 5 switches in every battle.
- **Current Reward**: `+0.1` points per switch (first 5).
- **Entropy**: We use `ent_coef: 0.01` to encourage variety in actions.
- **Adjustment**: If the model switches too much, lower the reward in `train_rl_agent.py`.

### 2. Shared Latent Spaces
The model uses shared 16-dim embeddings for:
- **Types**: Pokémon and Move types share the same latent concepts.
- **Species/Moves/Items/Abilities**: All categorical data is embedded before reaching the policy network.

## 🏎️ Performance Optimization

### 1. Parallel Environments (`--n-envs`)
- **Current Sweet Spot**: 32 envs.
- **Max Effort**: 48-64 envs (monitor CPU context switching).

### 2. Batch Size
- **Default**: 4096.
- **Note**: The script will automatically cap the batch size in `--debug` mode to match the smaller rollout buffer.

### 3. Monitoring
Run these to check for bottlenecks:
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Watch CPU utilization
htop
```
