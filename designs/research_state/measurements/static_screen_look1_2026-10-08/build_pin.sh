#!/usr/bin/env bash
# Build the pin checkout's release Rust env core (the eval core main.h2h plays on), so the play starts warm.
set -eu
PIN_DIR=${PIN_DIR:-/home/goodlad/dev/gen3ai/.claude/worktrees/st-look1-pin-6c6d2e09}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
cd "$PIN_DIR"
echo "pin HEAD $(git rev-parse HEAD)"
PYTHONPATH=$PIN_DIR/src "$PY" -c "from utils.rust_env import build; print(build.ensure_built('release', emit=print))"
