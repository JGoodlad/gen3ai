# Gen3 AI Training Instructions

This document explains how to run and monitor the Gen3 Reinforcement Learning training on the remote Ubuntu workstation.

## 🚀 1. Start Training (The Marathon)

The training runs in a `tmux` session named `gen3_training`. 

To start a new training run:
1. SSH into the desktop: `ssh goodlad@goodlad-desktop.local`
2. Attach to the tmux session: `tmux a -t gen3_training` (or create it: `tmux new -s gen3_training`)
3. In Window 0 (Training), run:
```bash
cd ~/dev/gen3ai
PYTHONPATH=src:. /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --steps 10000000 --n-envs 32 --device cuda
```

## 🎮 2. Showdown Server (Backend)

The training requires a local Pokemon Showdown server running with `--no-security`.
- **Location**: tmux Window 1
- **Command**: `npm run showdown`
- **Access UI**: `http://goodlad-desktop.local:8000/`

## 📊 3. Monitor Progress (TensorBoard)

Track reward curves and learning metrics in real-time.
- **Location**: tmux Window 2
- **Command**: `/home/goodlad/miniconda3/envs/gen3ai_stable/bin/tensorboard --logdir ./tensorboard/ --host 0.0.0.0 --port 6006`
- **Access UI**: `http://goodlad-desktop.local:6006/`

## 🛠️ Environment Reference

- **Conda Env**: `gen3ai_stable` (Python 3.11.15)
- **Conda Path**: `/home/goodlad/miniconda3/bin/conda`
- **Python Path**: `/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3`
- **Device**: `cuda` (NVIDIA RTX 3080 Ti)

- **Username Errors**: Avoid underscores in usernames; Showdown strips them and breaks the `poke-env` handshake.

## 🏎️ Tuning for Performance

To get the most out of your 16-core CPU and RTX 3080 Ti:

### 1. Parallel Environments (`--n-envs`)
- **Current Sweet Spot**: 32 envs (~50% CPU, ~50% GPU).
- **Max Effort**: 48-64 envs. 
- *Note*: Going above 64 on a 16-core machine may cause diminishing returns due to context switching.

### 2. Showdown Workers
If you increase `--n-envs`, you must also increase Showdown workers to prevent a bottleneck.
- **Location**: `deps/pokemon-showdown/config/config.js`
- **Settings**:
  - `simulator: 8` (The primary bottleneck; handle battle logic)
  - `network: 4` (Handles WebSocket traffic)
  - `validator: 4` (Handles team validation)

### 3. Monitoring the "Gauges"
Run this command to see if you are bottlenecked:
```bash
# Check CPU usage (Look for %id - higher is more idle)
top -bn1 | head -n 20

# Check GPU usage (Look for GPU-Util and Pwr:Usage)
nvidia-smi
```
- **Ideal State**: CPU idle at 20-30%, GPU util at 50-70%.
