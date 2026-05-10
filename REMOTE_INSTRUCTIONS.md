# 🌍 Remote Training Instructions (Ubuntu)

This guide explains how to migrate the Gen3AI training pipeline to a remote Ubuntu server for high-performance training.

## 1. Prerequisites (Remote Server)
Run the following commands on your Ubuntu machine to install the core dependencies:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv nodejs npm tmux git rsync
```

## 2. Code Deployment
### Option A: Via GitHub (Recommended)
1. Push your local changes to GitHub.
2. On the remote server:
   ```bash
   git clone --recursive <your-repo-url>
   cd gen3ai
   ```

### Option B: Via Rsync (Direct Transfer)
From your local Mac terminal:
```bash
rsync -avz --exclude 'venv' --exclude 'tensorboard' --exclude 'models' . user@remote-ip:~/gen3ai
```

## 3. Environment Setup
Initialize the submodules and install dependencies:
```bash
npm run setup
```

## 4. Persistent Training Session (`tmux`)
Use `tmux` to ensure training continues if you disconnect from SSH.

1. **Start a new session**:
   ```bash
   tmux new -s pokemon_ai
   ```
2. **Launch Training**:
   ```bash
   PYTHONPATH=src ./deps/venv/bin/python3 src/main/train_rl_agent.py --steps 4000000 --n-envs 16
   ```
   *(Note: You can increase `--n-envs` if the remote server has more CPU cores.)*
3. **Detach**: Press `Ctrl + B`, then `D`.
4. **Reattach later**: `tmux attach -t pokemon_ai`.

## 5. Monitoring via SSH Tunnel
To view TensorBoard on your local Mac:
1. **Start TensorBoard on Remote**:
   ```bash
   ./deps/venv/bin/tensorboard --logdir ./tensorboard/ --port 6006
   ```
2. **Tunnel from Local Mac**:
   ```bash
   ssh -L 6006:localhost:6006 user@remote-ip
   ```
3. **Open Browser**: Go to `http://localhost:6006`.

## 6. GPU Acceleration (NVIDIA Only)
If your Ubuntu machine has an NVIDIA GPU:
1. Ensure CUDA drivers are installed.
2. Reinstall torch with CUDA support:
   ```bash
   ./deps/venv/bin/pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
   ```
3. The script will automatically detect the GPU or you can force it:
   ```bash
   python3 src/main/train_rl_agent.py ... --device cuda
   ```
