#!/usr/bin/env bash
# The ship's push guard (python -m utils.push_guard) from this worktree.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../../.." && pwd)"
cd "$WT"
export PYTHONPATH="$WT/src"
exec /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m utils.push_guard
