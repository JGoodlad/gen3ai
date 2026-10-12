#!/usr/bin/env bash
# Wait (bounded, 8 h) for <pid> to exit, then run seq_remaining.sh with the remaining arguments.
#   after_then_seq.sh <pid> <tag> <cpus> <cp>:<arm>[:interim] ...
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid="$1"; shift
timeout 28800 bash -c "while kill -0 $pid 2>/dev/null; do sleep 20; done"
exec "$HERE/seq_remaining.sh" "$@"
