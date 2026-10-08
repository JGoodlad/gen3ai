#!/usr/bin/env bash
# Run read_look1.py with the estimator imported from the P_st checkout (args pass through, e.g. --progress).
set -u
PIN_DIR=${PIN_DIR:-/home/goodlad/dev/gen3ai/.claude/worktrees/st-look1-pin-6c6d2e09}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$PIN_DIR" || exit 5
PYTHONPATH=$PIN_DIR/src exec "$PY" "$HERE/read_look1.py" "$@"
