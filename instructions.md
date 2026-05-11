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

## 💡 Troubleshooting

- **"Connection Reset"**: The script uses a 0.1s stagger. If it still crashes, increase the stagger in `src/main/train_rl_agent.py`.
- **"Broken Pipe"**: Ensure `multiprocessing.set_start_method('spawn')` is called at the very top of `train_rl_agent.py`.
- **Username Errors**: Avoid underscores in usernames; Showdown strips them and breaks the `poke-env` handshake.
