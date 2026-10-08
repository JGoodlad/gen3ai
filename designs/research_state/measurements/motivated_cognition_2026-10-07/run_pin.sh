#!/usr/bin/env bash
# Run a python script AT THE 706fa536 TRAINING PIN: cwd and PYTHONPATH are the pin checkout (data/ is read
# from the cwd, and HEAD's data/ differs: gen3_item_priors.json / gen3_ability_priors.json), CUDA hidden.
# REFUSES (exit 5) unless the pin checkout's HEAD file names 706fa536 and its src/ + data/ match the commit
# (checked by the caller once; see README "The pin").
set -u
PIN=706fa536ef53d9a680d2461f16613e0d1430d176
PIN_DIR=${PIN_DIR:-/home/goodlad/dev/gen3ai/.claude/worktrees/x5-look3-pin-706fa536}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
HEADF=$(sed -n 's/^gitdir: //p' "$PIN_DIR/.git")/HEAD
[ "$(cat "$HEADF")" = "$PIN" ] || { echo "REFUSED: $PIN_DIR is not at $PIN"; exit 5; }
cd "$PIN_DIR" || exit 5
export PYTHONPATH=$PIN_DIR/src CUDA_VISIBLE_DEVICES=
exec "$PY" "$@"
