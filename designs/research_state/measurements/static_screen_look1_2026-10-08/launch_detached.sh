#!/usr/bin/env bash
# Start play_look1.sh DETACHED (its own session, stdin closed) with the GPU lease token file, logging to run.log.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export GEN3AI_GPU_LEASE_TOKEN_FILE=${GEN3AI_GPU_LEASE_TOKEN_FILE:-$HOME/.cache/gen3ai/st_look1/lease.token}
nohup setsid bash "$HERE/play_look1.sh" >> "$HERE/run.log" 2>&1 < /dev/null &
echo "launched pid $!"
