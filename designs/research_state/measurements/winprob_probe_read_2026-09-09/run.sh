#!/usr/bin/env bash
# Reproduce the probe read end to end. Read-only over models/; CPU only, no GPU, no server.
# Fixed seeds throughout; every number in README.md comes out of these calls.
#
# ~7 min of forwards + ~35 min of decoding on two niced cores. Nothing is written under models/.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(git rev-parse --show-toplevel)/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
NICE="nice -n 10"
TMP="${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/probe}"
M=/home/goodlad/dev/gen3ai/models
mkdir -p "$TMP"

A=$M/ai_v12_02_winprob_critic
C=$M/ai_v12_11_ladder_ctrl10M

# 1. the strength axis — an ALL-STEPS bot-anchored refit per run (the committed ladder.json
#    rates only the survivors of pool grooming, and would leave sentinel cells unrated).
$NICE $PY refit_ladder.py --run "$A" --out "$TMP/ladder_A.json"
$NICE $PY refit_ladder.py --run "$C" --out "$TMP/ladder_CTRL.json"

# 2. the per-state table + the frozen model's features (one checkpoint forwards every state)
$NICE $PY extract.py --run "$A" --snapshot "$A/eval_traces/step_74000016/snapshot.zip" \
    --cycles step_50000016,step_60000000,step_70000032,step_74000016 \
    --ladder "$TMP/ladder_A.json" --out-dir "$TMP/A"
$NICE $PY extract.py --run "$C" --snapshot "$C/eval_traces/step_10000032/snapshot.zip" \
    --cycles step_2000016,step_4000032,step_6000000,step_8000016,step_10000032 \
    --ladder "$TMP/ladder_CTRL.json" --out-dir "$TMP/CTRL"

# 3. the linear probe read
$NICE $PY decode.py --dir "$TMP/A"    --out "$TMP/decode_A.json"    --perm 40 --boot 2000
$NICE $PY decode.py --dir "$TMP/CTRL" --out "$TMP/decode_CTRL.json" --perm 40 --boot 2000

# 4. the non-linearity check on the headline bucket
$NICE $PY mlp_probe.py --dir "$TMP/A"    --out "$TMP/mlp_A.json"    --bucket t1_3
$NICE $PY mlp_probe.py --dir "$TMP/CTRL" --out "$TMP/mlp_CTRL.json" --bucket t1_3

# 5. the committed summary (small JSON only — no feature arrays)
$NICE $PY summarize.py --tmp "$TMP" --out-dir .
