#!/usr/bin/env bash
# gpu_run.sh <GB> <timeout_s> <log> <python args...> — one harness under the lease, the GPU lock and mem_cap.
set -u
W=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a30fcd2b9887c1fda
GB=$1; TO=$2; LOG=$3; shift 3
unset GEN3AI_MODELS_DIR
export PYTHONPATH=$W/src
export GEN3AI_GPU_LEASE_TOKEN=$(cat /home/goodlad/.cache/gen3ai/x5pn/lease_token)
cd "$W"
scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh "$GB" timeout "$TO" \
  /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 "$@" > "$LOG" 2>&1
echo "exit $?" >> "$LOG"
