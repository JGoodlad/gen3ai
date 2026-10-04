#!/usr/bin/env bash
# The two-architecture proof (CPU only), under a memory cap, from the repo ROOT (`main.h2h` refuses any other cwd:
# the team pool is read cwd-relative). Outputs land in this directory. Export the tree's PYTHONPATH first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
cd "$ROOT"
scripts/ops/mem_cap.sh 24 "$PY" "$HERE/cross_proof.py" play --out "$HERE/rows.json" 2>&1 | tee "$HERE/run.log"
"$PY" "$HERE/cross_proof.py" compare "$HERE/rows.json" > "$HERE/compare.json"
gzip -f "$HERE/rows.json"
