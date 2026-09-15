#!/usr/bin/env bash
# ARM S vs Metamon SmallRL — the two registered regimes, each as two role-balanced half-cells.
#
#   greedy : BOTH sides greedy, our 719-team HOME pool
#            comparator: ai_v12_02_winprob_critic@75M = 0.520 [0.423, 0.615] (matched-regime 2x2)
#   mixed  : OUR side greedy, Metamon at its OWN eval default (temperature 1.0), HOME pool
#            comparator: ai_v12_02_winprob_critic@75M = 0.742 [0.657, 0.812] (the de-risk)
#
# Role is balanced INSIDE every cell (Showdown makes the challenger p1).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PORT=${PORT:-9450}
PER_CELL=${PER_CELL:-100}
AGENT=${AGENT:-SmallRL}
HALF=$(( PER_CELL / 2 ))
export GEN3AI_MODEL=models/ai_v13_01_flywheel_shaped/final_model.zip
BASE_OURS=20260914
BASE_THEIRS=20270914
k=0
for REGIME in greedy mixed; do
  for WHO in ours metamon; do
    TAG="A${k}"
    OUT="$ROOT/cells/${AGENT}_${REGIME}_home_${WHO}"
    if [ -f "$OUT/cell.json" ]; then echo "[armS] SKIP $OUT"; k=$(( k + 1 )); continue; fi
    echo "[armS] === half-cell $k: $AGENT regime=$REGIME teams=home challenger=$WHO n=$HALF"
    bash "$HERE/run_cell.sh" "$AGENT" "$REGIME" home "$WHO" "$HALF" "$PORT" "$OUT" \
      $(( BASE_OURS + 100 * k )) $(( BASE_THEIRS + 100 * k )) "$TAG"
    k=$(( k + 1 ))
  done
done
echo "[armS] done $(date -Is)"
