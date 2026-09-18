#!/usr/bin/env bash
# W_b vs Metamon SmallRL — the HOME greedy cell, arm S's and arm W's own harness reused
# VERBATIM (run_cell.sh, unchanged) with the SAME team seeds, so all three arms draw the
# identical team pairs.
#
#   greedy : BOTH sides greedy, our 719-team HOME pool
#            arm S comparator 0.630 [0.532, 0.718]; arm W comparator 0.650 [0.553, 0.736]
#
# Role is balanced INSIDE the cell (Showdown makes the challenger p1, and p1/p2 is not a priori
# neutral). The MIXED cell is NOT run here: rule 1 of the anchors SOP — the recurring read is
# greedy-vs-greedy, and a t=1.0 opponent is not a fixed yardstick across policies.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/metamon
PORT=${PORT:-9450}
PER_CELL=${PER_CELL:-100}
AGENT=${AGENT:-SmallRL}
HALF=$(( PER_CELL / 2 ))
export GEN3AI_MODEL=models/ai_v13_04_flywheel_winprob_b/final_model.zip
BASE_OURS=20260914      # identical to arm S's and arm W's — the SAME team pairs
BASE_THEIRS=20270914
k=0
for REGIME in greedy; do
  for WHO in ours metamon; do
    TAG="B${k}"
    OUT="$ROOT/cells/${AGENT}_${REGIME}_home_${WHO}"
    if [ -f "$OUT/cell.json" ]; then echo "[armWb] SKIP $OUT"; k=$(( k + 1 )); continue; fi
    echo "[armWb] === half-cell $k: $AGENT regime=$REGIME teams=home challenger=$WHO n=$HALF"
    bash "$ROOT/scripts/run_cell.sh" "$AGENT" "$REGIME" home "$WHO" "$HALF" "$PORT" "$OUT" \
      $(( BASE_OURS + 100 * k )) $(( BASE_THEIRS + 100 * k )) "$TAG"
    k=$(( k + 1 ))
  done
done
echo "[armWb] done $(date -Is)"
