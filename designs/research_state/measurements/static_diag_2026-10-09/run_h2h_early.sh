#!/usr/bin/env bash
# H1's early diagonal: `main.h2h play-many` at the P_st pin, CPU only (CUDA hidden), nice 19, 2 core threads +
# 3 torch threads, under a 24 GB cap, into THIS study's own ledger root (never models/). Resumable (same args).
#   run_h2h_early.sh <cells.json> <pairs>
set -euo pipefail
PIN=/home/goodlad/dev/gen3ai/.claude/worktrees/st-look1-pin-6c6d2e09
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
A=/home/goodlad/gen3ai_archive/static_diag_2026-10-09
export PYTHONPATH="$PIN/src" CUDA_VISIBLE_DEVICES=""
cd "$PIN"
exec nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name stdiag-h2h 24 timeout 3h "$PY" -m main.h2h play-many \
  --cells "$1" --pairs "$2" --out "$A/ledger" --label static_diag_early --purpose audit --request-kind adhoc \
  --device cpu --threads 2 --torch-threads 3 --n-envs 64 --batch-pairs 200
