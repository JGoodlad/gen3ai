#!/usr/bin/env bash
# RULE 25's indexing clause, EXECUTED on the contested read set:
# `exploiter_discrimination_2026-09-18/verify_indexing.py`, invoked BY PATH and UNMODIFIED,
# re-derives the ORIGINAL head's pairwise accuracy from the raw artifacts with its own reader,
# its own pair enumeration and its own agreement rule, and prints ten pairs in full.
#   usage: run_verify.sh <expected pairwise acc for `original`> [tmp] [out]
set -uo pipefail
MAIN=/home/goodlad/dev/gen3ai
V=$MAIN/designs/research_state/measurements/exploiter_discrimination_2026-09-18/verify_indexing.py
EXPECT=${1:?the `original` pairwise accuracy from fit_report.json}
T=${2:-/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit}
OUT=${3:-$T/verify_indexing.json}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$MAIN/src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$MAIN" || exit 1
nice -n 15 "$PY" "$V" --forks "$T/forks_eval" \
  --snapshot "$T/src_run/ai_v13_02_flywheel_winprob/eval_traces/step_75005952/snapshot.zip" \
  --expect "$EXPECT" --threads 2 --out "$OUT"
