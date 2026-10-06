#!/usr/bin/env bash
# The fit and the score (README §2-§4), CPU, capped, nice 15.  run_fit.sh [--smoke]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../../.." && pwd)"
cd "$WT"
export PYTHONPATH="$WT/src"
export CUDA_VISIBLE_DEVICES=   # CPU only: torch cannot create a CUDA context (no GPU lease)
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
nice -n 15 "$WT/scripts/ops/mem_cap.sh" --name crnfit 12 timeout 2h "$PY" "$HERE/refit.py" --threads 3 "$@"
nice -n 15 "$PY" "$HERE/score.py" "$@"
