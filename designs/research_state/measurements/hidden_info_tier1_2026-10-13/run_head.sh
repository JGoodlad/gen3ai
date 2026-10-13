#!/usr/bin/env bash
# Run a Python script against THIS STUDY's HEAD worktree (the information-lever builder), CPU only, at SCHED_IDLE +
# nice 19, pinned to the agent cores, under a memory cap and a timeout.
#   run_head.sh <GB> <timeout> <script.py> [args...]
set -euo pipefail
GB="$1"; TO="$2"; SCRIPT="$3"; shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$SCRIPT" in /*) ;; *) SCRIPT="$HERE/$SCRIPT" ;; esac
WT=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a70fa77a612561867
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH="$WT/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${HIB_THREADS:-1}" MKL_NUM_THREADS="${HIB_THREADS:-1}"
CPUS="${HIB_CPUS:-5-7,13-15}"
cd "$WT"
exec chrt --idle 0 nice -n 19 "$WT/scripts/ops/mem_cap.sh" --name "hib_head" "$GB" timeout "$TO" taskset -c "$CPUS" "$PY" "$SCRIPT" "$@"
