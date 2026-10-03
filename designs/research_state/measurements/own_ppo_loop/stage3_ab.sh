#!/bin/bash
# usage: stage3_ab.sh <worktree> <run-name>   — an E3 real-run arm (design_own_ppo_loop.md §4): the root smoke, seed 42, Rust core
WT=$1; NAME=$2
export PYTHONPATH=$WT/src
export GEN3AI_MODELS_DIR=${GEN3AI_MODELS_DIR:?set a scratch run archive}
cd $WT
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
T=train_rl_agent
exec /home/goodlad/dev/gen3ai/scripts/ops/mem_cap.sh 16 timeout -s KILL 560 $PY $WT/src/main/$T.py --debug --seed 42 --run-name $NAME --steps 10000
