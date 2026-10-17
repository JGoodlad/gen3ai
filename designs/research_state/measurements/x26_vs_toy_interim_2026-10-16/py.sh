#!/usr/bin/env bash
# This worktree's python under rule 5b (agent cores, low priority, one thread, no GPU). Usage: py.sh <python args...>
# Set CHECKOUT to run at another checkout's src (default: this worktree).
HERE=$(cd "$(dirname "$0")" && pwd)
CHECKOUT=${CHECKOUT:-$(git -C "$HERE" rev-parse --show-toplevel)}
export PYTHONPATH="$CHECKOUT/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$CHECKOUT" || exit 2
exec nice -n 19 taskset -c 5-7,13-15 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 "$@"
