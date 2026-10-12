#!/usr/bin/env bash
# Wait (bounded, 8 h) for process <pid> to exit, then run run_chain.sh with the remaining arguments.
#   run_after.sh <pid> <cp> <shard> <nshards> <cpus> <arm> [<arm> ...]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid="$1"; shift
timeout 28800 bash -c "while kill -0 $pid 2>/dev/null; do sleep 30; done"
exec "$HERE/run_chain.sh" "$@"
