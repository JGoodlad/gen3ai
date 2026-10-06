#!/usr/bin/env bash
# compact_rows.py with this worktree's src/ on the path.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="$WT/src"
export CUDA_VISIBLE_DEVICES=
exec /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 "$HERE/compact_rows.py"
