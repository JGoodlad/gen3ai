#!/usr/bin/env bash
# Run build_chain.sh <cp> once the process <pid> (another checkpoint's chain) has exited: ONE world builder at a time,
# so the CPU allowance (3 workers beside a live GPU run) goes 1 builder + 2 playout shards.
#   build_after.sh <pid> <cp>
set -uo pipefail
PID="$1"; CP="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
bash "$HERE/build_chain.sh" "$CP"
