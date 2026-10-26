#!/usr/bin/env bash
# Crawl E1-0's play in bounded SHORT-lane chunks (<= 560 s of play each, root rule 5b): the bot mode first, then the self
# mode (G-REPRO). Each chunk caps new work; a chunk killed at its bound loses at most the state in flight (rows are
# fsync'd per state). Stops when a mode reports "0 to go".  Usage: loop_e10.sh [max_chunks]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX="${1:-60}"
n=0
for mode in bot self; do
  while [ "$n" -lt "$MAX" ]; do
    n=$((n + 1))
    out="$("$HERE/run_e10.sh" 560 --max-s 590 -- --mode "$mode" --limit 40 2>&1 | grep -v 'heavy_slots\] waiting')"
    echo "$out" | grep -E '^\[play_e10\] (bot|self):|REFUSED|Error|mem_cap\] done' | tail -4
    echo "$out" | grep -q ' 0 to go' && break
  done
done
echo "[loop_e10] done after $n chunks"
