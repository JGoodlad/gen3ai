#!/usr/bin/env bash
# ARM W vs Metamon SmallRL — the SAME two registered cells arm S was read on, same harness,
# SAME team seeds, so the two arms draw the identical team pairs.
#
#   greedy : BOTH sides greedy, our 719-team HOME pool
#            arm S comparator: 0.630 [0.532, 0.718] (flywheel_armS_reads_2026-09-14)
#   mixed  : OUR side greedy, Metamon at its OWN eval default (temperature 1.0), HOME pool
#            arm S comparator: 0.840 [0.756, 0.899]
#
# Role is balanced INSIDE every cell (Showdown makes the challenger p1).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PORT=${PORT:-9450}
PER_CELL=${PER_CELL:-100}
AGENT=${AGENT:-SmallRL}
HALF=$(( PER_CELL / 2 ))
export GEN3AI_MODEL=models/ai_v13_02_flywheel_winprob/final_model.zip
BASE_OURS=20260914      # identical to arm S's — the SAME team pairs
BASE_THEIRS=20270914
k=0
for REGIME in greedy mixed; do
  for WHO in ours metamon; do
    TAG="W${k}"
    OUT="$ROOT/cells/${AGENT}_${REGIME}_home_${WHO}"
    if [ -f "$OUT/cell.json" ]; then echo "[armW] SKIP $OUT"; k=$(( k + 1 )); continue; fi
    echo "[armW] === half-cell $k: $AGENT regime=$REGIME teams=home challenger=$WHO n=$HALF"
    bash "$HERE/run_cell.sh" "$AGENT" "$REGIME" home "$WHO" "$HALF" "$PORT" "$OUT" \
      $(( BASE_OURS + 100 * k )) $(( BASE_THEIRS + 100 * k )) "$TAG"
    k=$(( k + 1 ))
  done
done
echo "[armW] done $(date -Is)"
