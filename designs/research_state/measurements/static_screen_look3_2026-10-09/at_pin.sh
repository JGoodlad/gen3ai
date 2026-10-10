#!/usr/bin/env bash
# Run one of this folder's Python scripts with the code imported from the P_st checkout (args pass through):
#   bash at_pin.sh read_look3.py [--progress]      bash at_pin.sh plan_look3.py --out cells.json
set -u
PIN_DIR=${PIN_DIR:-/home/goodlad/dev/gen3ai/.claude/worktrees/st-look1-pin-6c6d2e09}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
script=$1; shift
cd "$PIN_DIR" || exit 5
PYTHONPATH=$PIN_DIR/src exec "$PY" "$HERE/$script" "$@"
