#!/usr/bin/env bash
# The two cells, in sequence, for one teacher-file mode:
#   n=3 (reproduces content_locality's canonical STATE batch exactly) then n=9 (POWER EXTENSION).
# Each ACID-checks itself against content_locality afterwards.
#   ./run_both.sh [best_model|final_interrupted]
# HEADLINE is best_model: v8_14's argv names each teacher as a run DIR, and the fold's own
# resolver takes best_model/best_model.zip first. final_interrupted is the labelled SECONDARY
# (the file content_locality used, which is not a rung of that resolver at all).
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
MODE="${1:-best_model}"
SUF=""
[ "$MODE" = "final_interrupted" ] && SUF="_finalint"
for N in 3 9; do
  BASE="zswap${SUF}_n${N}"
  "$D/run_era.sh" "$N" "$BASE" "$MODE" > "$D/${BASE}.log" 2>&1
  echo "${BASE} measurement exit=$?"
  "$PY" "$D/acid_vs_content_locality.py" "$D/${BASE}.json" \
        "$D/../content_locality/v8_era_n${N}.json" "$MODE" >> "$D/${BASE}.log" 2>&1
  echo "${BASE} acid exit=$?"
  "$PY" "$D/analyze.py" "$D/${BASE}.json" "$D/analysis${SUF}_n${N}.json" \
        > "$D/analysis${SUF}_n${N}.log" 2>&1
  echo "${BASE} analyze exit=$?"
done
echo "BOTH CELLS DONE (mode=$MODE)"
