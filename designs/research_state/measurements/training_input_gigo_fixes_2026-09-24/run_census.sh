#!/usr/bin/env bash
# The census: capture the same deterministic battles from the BASE tree and the FIX tree (each with
# its own src/ and its own core_events build), then diff. Resumable: a capture skips battles already
# in its JSONL. Usage: BASE_WT=<base worktree> FIX_WT=<fix worktree> OUT=<dir> run_census.sh
set -euo pipefail
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3}
D=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/tmp/gigo/census}; mkdir -p "$OUT"
cap() {  # cap <tree> <tag> <mode> <args…>
  local wt=$1 tag=$2; shift 2
  (cd "$wt" && PYTHONPATH="$wt/src" GEN3AI_SKIP_DEPS_GUARD=1 nice -n 10 "$PY" "$D/census.py" "$@")
}
for side in base fix; do
  wt=$([ $side = base ] && echo "$BASE_WT" || echo "$FIX_WT")
  [ -s "$OUT/golden_$side.jsonl" ] || cap "$wt" $side golden "$OUT/golden_$side.jsonl"
  cap "$wt" $side corpus "$OUT/corpus_$side.jsonl" commit
  cap "$wt" $side corpus "$OUT/corpus_$side.jsonl" random 0:120
  cap "$wt" $side corpus "$OUT/corpus_$side.jsonl" policy 100:110
  cap "$wt" $side corpus "$OUT/corpus_$side.jsonl" ladder 0:60
done
(cd "$FIX_WT" && PYTHONPATH="$FIX_WT/src" "$PY" "$D/census.py" diff "$OUT/golden_base.jsonl" "$OUT/golden_fix.jsonl") | tee "$D/golden_census.txt"
(cd "$FIX_WT" && PYTHONPATH="$FIX_WT/src" "$PY" "$D/census.py" diff "$OUT/corpus_base.jsonl" "$OUT/corpus_fix.jsonl") | tee "$D/corpus_census.txt"
