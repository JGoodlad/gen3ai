#!/usr/bin/env bash
# The multi-cell proof (CPU only), under a memory cap. The team pool is read cwd-relative (data/teams), so the
# scripts run from the repo ROOT; outputs land in this directory. Export the tree's PYTHONPATH first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
cd "$ROOT"
scripts/ops/mem_cap.sh 24 "$PY" "$HERE/multicell_proof.py" games --out "$HERE/games.json" 2>&1 | tee "$HERE/games.log"
scripts/ops/mem_cap.sh 24 "$PY" "$HERE/multicell_proof.py" rows --out "$HERE/rows.json" 2>&1 | tee "$HERE/rows.log"
"$PY" "$HERE/multicell_proof.py" compare "$HERE/games.json" "$HERE/rows.json" > "$HERE/compare.json"
gzip -f "$HERE/games.json" "$HERE/rows.json"
