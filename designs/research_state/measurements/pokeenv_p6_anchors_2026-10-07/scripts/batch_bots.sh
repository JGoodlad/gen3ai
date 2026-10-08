#!/usr/bin/env bash
# Gate (a) of P6: every one of the NINE roster bots as our side vs metamon:SmallRL (greedy), 12 seeded games each
# (6 per half), on BOTH paths (bot_cell.sh). Team set alternates home / away by bot index; each bot its own
# --bot-seed and --seed-base. Two lanes at a time. Plus one TEETH cell: the old path seeded with a DIFFERENT
# bot seed than the new one (the comparison must report it).
here=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/tmp/p6s2/work/gate_a}
mkdir -p "$OUT"
BOTS=(random heuristic heuristic2 staller staller_v2 aggressive aggressive_v2 setup_sweep setup_sweep_v2)
ONLY=${ONLY:-}   # e.g. ONLY="0 2 4": re-run just those cells
run() {  # index
  local i=$1 bot=${BOTS[$1]} ts=home
  [ $((i % 2)) = 1 ] && ts=away
  if [ -n "$ONLY" ] && ! [[ " $ONLY " == *" $i "* ]]; then return 0; fi
  PORT=$((9520 + 2 * i)) "$here/bot_cell.sh" "$OUT" "$(printf '%02d' "$i")_${bot}_${ts}" "$bot" --bot-seed $((101 + i)) \
      --opponent metamon:SmallRL --regime greedy --teamset "$ts" --games 12 --seed-base $((914007 + 1000 * i))
}
for i in 0 2 4 6 8; do
  run "$i" &
  a=$!
  if [ "$i" -lt 8 ]; then run $((i + 1)); fi
  wait "$a"
done
# TEETH: cell 00's NEW read (bot seed 101) against an OLD read whose Python bot is seeded with 102 — the identical
# protocol otherwise. `random` draws at every decision, so the comparison must report differing battles.
if [ -z "$ONLY" ] || [[ " $ONLY " == *" teeth "* ]]; then
  T="$OUT/teeth_random_seed_mismatch"
  rm -rf "$T"; mkdir -p "$T"
  cp -r "$OUT/00_random_home/new" "$T/new"
  timeout 3000 "$here/read.sh" legacybot --our-transport poke-env --our-side bot:random --port 9540 \
      --bot-seed 102 --opponent metamon:SmallRL --regime greedy --teamset home --games 12 --seed-base 914007 \
      --capture-dir "$T/old/cap" --out "$T/old" > "$T.old.log" 2>&1
  echo "teeth old rc=$?" >> "$OUT/status.txt"
fi
echo DONE >> "$OUT/status.txt"
