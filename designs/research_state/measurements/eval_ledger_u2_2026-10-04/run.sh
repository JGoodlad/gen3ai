#!/usr/bin/env bash
# The U2 storage-only proof (README.md). CPU only, under mem_cap. BEFORE = a detached worktree at e5f393cb (its own
# Rust self-check build, via ./scripts/bootstrap.sh --skip-env); AFTER = the U2 tree. Same script for both.
set -eu
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
BEFORE_TREE=${BEFORE_TREE:?a worktree at e5f393cb}
AFTER_TREE=${AFTER_TREE:?the U2 tree}
(cd "$BEFORE_TREE" && PYTHONPATH="$BEFORE_TREE/src" "$AFTER_TREE/scripts/ops/mem_cap.sh" --name u2before 16 \
  timeout 2400 "$PY" "$HERE/digest_proof.py" play "$HERE/before.json")
(cd "$AFTER_TREE" && PYTHONPATH="$AFTER_TREE/src" scripts/ops/mem_cap.sh --name u2after 16 \
  timeout 2400 "$PY" "$HERE/digest_proof.py" play "$HERE/after.json")
"$PY" "$HERE/digest_proof.py" compare "$HERE/before.json" "$HERE/after.json" | tee "$HERE/compare.txt"
