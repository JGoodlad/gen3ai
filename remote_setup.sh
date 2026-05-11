#!/bin/bash
set -e

echo "🚀 Starting Gen3AI Remote Setup..."

# 1. Ensure the dev directory exists
mkdir -p ~/dev
cd ~/dev

# 2. Install Git and core tools
echo "🛠 Installing Git..."
sudo apt update
sudo apt install -y git

# 3. Clone or Update the Repository
if [ ! -d "gen3ai" ]; then
    echo "📦 Cloning repository..."
    git clone --recursive https://github.com/JGoodlad/gen3ai.git
    cd gen3ai
else
    echo "🔄 Repository already exists, pulling latest changes..."
    cd gen3ai
    git pull origin main
    git submodule update --init --recursive
fi

# 3. Install System Dependencies
echo "🛠 Installing system dependencies (Node.js, NPM, Python venv, Tmux)..."
sudo apt update
sudo apt install -y python3-pip python3-venv nodejs npm tmux

# 4. Project Setup
echo "🏗 Running project setup (npm install, pip install, build showdown)..."
npm run setup

# 5. GPU Verification
echo "🔍 Verifying GPU status..."
./deps/venv/bin/python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "✅ Setup Complete!"
echo "To start training, run: tmux new -s gen3_training"
echo "Inside tmux: PYTHONPATH=src:. ./deps/venv/bin/python3 src/main/train_rl_agent.py --steps 10000000 --n-envs 16 --device cuda"
