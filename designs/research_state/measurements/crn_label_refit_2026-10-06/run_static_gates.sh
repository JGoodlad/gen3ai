#!/usr/bin/env bash
# The ship's targeted gate for this docs + measurement unit: the STATIC tier (no src/ change).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../../.." && pwd)"
cd "$WT"
export PYTHONPATH="$WT/src"
export CUDA_VISIBLE_DEVICES=
export GEN3AI_SKIP_DEPS_GUARD=1   # the static tier needs no Showdown build; this worktree has none
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
exec "$WT/scripts/ops/gate_lock.sh" "$PY" -m pytest src/ -m static -q -n 4 -p no:cacheprovider
