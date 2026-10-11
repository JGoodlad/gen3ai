#!/usr/bin/env bash
# W playout shards of one checkpoint in ONE capped job (run through run_at_pin.sh):
#   run_at_pin.sh <cp> <threads per shard> <GB> <timeout> shards.sh <W> [--det] [driver args...]
# Phase 1's driver (run_playouts.py) by default; --det runs Phase 2's (run_det_playouts.py).
# Each shard is resumable (its own fsync'd JSONL); logs go to the archive's <cp>/logs/.
set -uo pipefail
W="$1"; shift
DRIVER=run_playouts.py; TAG=""
if [ "${1:-}" = "--det" ]; then DRIVER=run_det_playouts.py; TAG="det_"; shift; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$HOME/gen3ai_archive/value_disc_pilot/$VDP_CP/logs"
mkdir -p "$LOGS"
pids=()
for ((i = 0; i < W; i++)); do
  "$VDP_PY" "$HERE/$DRIVER" --shard "$i" --of "$W" "$@" >> "$LOGS/${TAG}shard_${i}_of_${W}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
exit "$rc"
