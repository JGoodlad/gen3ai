#!/bin/bash
# Run this worktree's python (PYTHONPATH pinned to the worktree src) on the agent cores at low priority.
WT=${WT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd "$WT" || exit 9
export PYTHONPATH="$WT/src"
exec taskset -c 5-7,13-15 nice -n 19 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 "$@"
