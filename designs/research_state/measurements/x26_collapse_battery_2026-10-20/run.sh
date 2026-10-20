#!/usr/bin/env bash
# Run one battery step at THIS checkout under root CLAUDE.md rule 5b (a live run holds the GPU):
# nice 19, the 3-core agent pin (5-7 + their SMT siblings 13-15), one OpenMP thread, no GPU.
#     designs/research_state/measurements/x26_collapse_battery_2026-10-20/run.sh <python args...>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}"
cd "$ROOT"
exec nice -n 19 taskset -c 5-7,13-15 "$PY" "$@"
