#!/usr/bin/env bash
# H7 z-swap, v8 era. Run from anywhere; it cds into the READ-ONLY era checkout (TeamLoader
# resolves data/teams relative to CWD) and writes every artifact back into this directory.
#   ./run_era.sh <per_team> <out_basename> [best_model|final_interrupted]
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-3}"
OUT="${2:-zswap_n${N}}"
MODE="${3:-best_model}"
ERA=/tmp/v8rep_era
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
cd "$ERA"
export PYTHONPATH="$ERA/src"
export PYTHONDONTWRITEBYTECODE=1
export ERA_ROOT="$ERA"
export GEN3AI_TIMEOUT_SCALE=12
export ZSWAP_TEACHER_FILE="$MODE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec nice -n 10 "$PY" "$D/era_zswap.py" "$D/$OUT.json" "$N" 400
