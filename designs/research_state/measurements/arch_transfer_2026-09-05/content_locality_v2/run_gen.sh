#!/usr/bin/env bash
# Gen-era v2 arm. Run from the worktree root.
#   ./run_gen.sh <n_battles_per_team> <out.json> <out.log>
set -euo pipefail
WT=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a16403d60f0c9a879
HERE="$WT/designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2"
export PYTHONPATH="$WT/src"
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export GEN3AI_TIMEOUT_SCALE=12
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$WT"
exec nice -n 10 /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
  "$HERE/gen_era_locality_v2.py" "$HERE/$2" "$1" 2>&1 | tee "$HERE/$3"
