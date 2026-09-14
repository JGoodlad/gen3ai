#!/usr/bin/env bash
# The whole matched-regime 2x2, for one Metamon model: 4 cells x 2 role half-cells.
#
#   bash run_all.sh <agent> <games_per_cell> <port> <root_out_dir> <prefix>
#
# A CELL is (sampling regime x team set) with BOTH sides at the same setting. Each cell is played
# as TWO half-cells that differ only in who sends the challenge, because Showdown makes the
# challenger p1 and p1/p2 is not a priori neutral. Role is therefore balanced INSIDE every cell
# rather than confounded with one.
#
# Seeds are derived from a single base and written into each cell.json, so a half-cell is
# reproducible from its own record without consulting this script.
set -uo pipefail

AGENT="${1:?agent}"; PER_CELL="${2:?games per cell}"; PORT="${3:?port}"
ROOT="${4:?root out dir}"; PREFIX="${5:?username prefix char, e.g. S}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OURS=20260914
BASE_THEIRS=20270914
HALF=$(( PER_CELL / 2 ))

mkdir -p "$ROOT"
k=0
for REGIME in greedy t1; do
  for TEAMS in home away; do
    for WHO in ours metamon; do
      TAG="${PREFIX}${k}"
      OUT="$ROOT/${AGENT}_${REGIME}_${TEAMS}_${WHO}"
      if [ -f "$OUT/cell.json" ]; then
        echo "[all] SKIP $OUT (already has cell.json)"
        k=$(( k + 1 )); continue
      fi
      echo "[all] === half-cell $k: $AGENT regime=$REGIME teams=$TEAMS challenger=$WHO n=$HALF"
      bash "$HERE/run_cell.sh" "$AGENT" "$REGIME" "$TEAMS" "$WHO" "$HALF" "$PORT" "$OUT" \
        $(( BASE_OURS + 100 * k )) $(( BASE_THEIRS + 100 * k )) "$TAG"
      k=$(( k + 1 ))
    done
  done
done
echo "[all] done: $AGENT"
