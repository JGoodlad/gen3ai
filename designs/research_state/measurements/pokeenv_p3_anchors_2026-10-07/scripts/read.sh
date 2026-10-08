#!/usr/bin/env bash
# usage: read.sh <blocked:0|1> <anchors args...>
WT=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a92f44ba7fef947d3
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH=$WT/src CUDA_VISIBLE_DEVICES="" POKESIM_SIM_BRIDGE_BIN=$WT/src/rust_sim/target/release/sim_bridge
cd "$WT" || exit 9
blocked=$1; shift
if [ "$blocked" = 1 ]; then
  exec "$PY" -m utils.poke_env_blocker main.anchors "$@"
else
  exec "$PY" -m main.anchors "$@"
fi
