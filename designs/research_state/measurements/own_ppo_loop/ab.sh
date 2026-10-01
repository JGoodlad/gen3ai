#!/bin/bash
# E3 real-run A/B: upstream loop (seam) vs owned loop, matched seed; Rust core (identity bar) then Python core (bounded)
export PYTHONPATH=/home/goodlad/dev/gen3ai/.claude/worktrees/own-ppo-loop-scoping/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
WT=/home/goodlad/dev/gen3ai/.claude/worktrees/own-ppo-loop-scoping
cd $WT
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
run() { # name mode extra...
  local name=$1 mode=$2; shift 2
  GEN3AI_PPO_LOOP=$mode timeout 1200 /home/goodlad/dev/gen3ai/scripts/ops/mem_cap.sh 16 $PY $WT/src/main/train_rl_agent.py \
    --debug --seed 42 --run-name $name "$@" > ~/.cache/gen3ai/tmp/ppo_det/ab_$name.txt 2>&1
  echo "$name exit $?" >> ~/.cache/gen3ai/tmp/ppo_det/ab_status.txt
}
R="--steps 4000 --arch production --env-core rust --n-envs 4 --n-steps 256 --batch-size 256"
run abr_ref sb3_reference $R
run abr_own owned $R
P="--steps 6000"
run abp_ref sb3_reference $P
run abp_own owned $P
echo done >> ~/.cache/gen3ai/tmp/ppo_det/ab_status.txt
