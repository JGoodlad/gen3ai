#!/usr/bin/env bash
# The five fits + the two init controls, and the held-out read on the contested set.
set -uo pipefail
MAIN=/home/goodlad/dev/gen3ai
HERE=$MAIN/designs/research_state/measurements/offline_leaf_fit_2026-09-18
T=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit}
OUT=${2:-$T/fit}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$MAIN/src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$MAIN" || exit 1
nice -n 15 "$PY" "$HERE/fit_heads.py" \
  --train-forks "$T/forks_branch" --eval-forks "$T/forks_eval" \
  --snapshot "$T/src_run/ai_v13_02_flywheel_winprob/eval_traces/step_75005952/snapshot.zip" \
  --guard-tree "$T/tree_guard/eval_traces/step_74000016" \
  --rank-coef 0.3 --cond-n 80 --threads 4 --out "$OUT"
