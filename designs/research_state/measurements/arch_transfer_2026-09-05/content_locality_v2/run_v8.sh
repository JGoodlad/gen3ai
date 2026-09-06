#!/usr/bin/env bash
# v8-era v2 arm, run FROM THE ERA CHECKOUT (read-only; nothing is written there).
#   ./run_v8.sh <n_battles_per_team> <out.json> <out.log>
set -euo pipefail
WT=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a16403d60f0c9a879
HERE="$WT/designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2"
export PYTHONPATH=/tmp/v8rep_era/src
export PYTHONDONTWRITEBYTECODE=1
export ERA_ROOT=/tmp/v8rep_era
export GEN3AI_TIMEOUT_SCALE=12
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /tmp/v8rep_era
exec nice -n 10 /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
  "$HERE/v8_era_locality_v2.py" "$HERE/$2" "$1" 2>&1 | tee "$HERE/$3"
