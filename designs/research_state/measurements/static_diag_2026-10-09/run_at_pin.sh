#!/usr/bin/env bash
# Run one of this folder's scripts against the P_st pin's code (6c6d2e09), CPU only, gentle:
# nice 19, 4 intra-op threads, under a 24 GB mem cap. Never touches the GPU (CUDA hidden).
set -euo pipefail
PIN=/home/goodlad/dev/gen3ai/.claude/worktrees/st-look1-pin-6c6d2e09
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
script="$1"; shift
export PYTHONPATH="$PIN/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$PIN"
exec nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name stdiag 24 timeout 3h "$PY" "$HERE/$script" "$@"
