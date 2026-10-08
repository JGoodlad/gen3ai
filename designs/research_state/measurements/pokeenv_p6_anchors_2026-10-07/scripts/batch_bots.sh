#!/usr/bin/env bash
# Gate (a) of P6: every one of the NINE roster bots as our side vs metamon:SmallRL (greedy), 12 seeded games each
# (6 per half), on BOTH paths (bot_cell.sh). Team set alternates home / away by bot index; each bot its own
# --bot-seed and --seed-base. Two lanes at a time. Plus one TEETH cell: the old path seeded with a DIFFERENT
# bot seed than the new one (the comparison must report it).
here=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/tmp/p6s2/work/gate_a}
mkdir -p "$OUT"
BOTS=(random heuristic heuristic2 staller staller_v2 aggressive aggressive_v2 setup_sweep setup_sweep_v2)
run() {  # index
  local i=$1 bot=${BOTS[$1]} ts=home
  [ $((i % 2)) = 1 ] && ts=away
  "$here/bot_cell.sh" "$OUT" "$(printf '%02d' "$i")_${bot}_${ts}" "$bot" --bot-seed $((101 + i)) \
      --opponent metamon:SmallRL --regime greedy --teamset "$ts" --games 12 --seed-base $((914007 + 1000 * i))
}
for i in 0 2 4 6 8; do
  run "$i" &
  a=$!
  if [ "$i" -lt 8 ]; then run $((i + 1)); fi
  wait "$a"
done
echo DONE >> "$OUT/status.txt"
