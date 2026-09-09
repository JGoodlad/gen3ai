#!/usr/bin/env bash
# Reproduce the mixture diagnostic end to end. Read-only over models/; no GPU, no server.
# Fixed seeds throughout; every number in README.md comes out of these four calls.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(git rev-parse --show-toplevel)/src"
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
TMP="${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/mixture}"
mkdir -p "$TMP"
$PY refit_ladder.py --out "$TMP/ladder_refit_all_steps.json"
$PY extract.py      --out "$TMP/states.npy" --ladder "$TMP/ladder_refit_all_steps.json"
$PY analyze.py      --states "$TMP/states.npy" --out "$TMP/mixture_stats.json" \
                    --boot 10000 --boot-head 10000
$PY sensitivity.py  --states "$TMP/states.npy" --out "$TMP/sensitivity.json" --boot 4000
