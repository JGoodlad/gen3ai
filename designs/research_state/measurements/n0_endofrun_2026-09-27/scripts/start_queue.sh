#!/usr/bin/env bash
# Start the N0 end-of-run CPU work queue DETACHED (it survives the session that starts it).
# Run from anywhere; the queue runs from the PINNED tree this script lives in.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/../../../../.." && pwd)"
STATE="${N0Q_STATE:-/home/goodlad/dev/gen3ai-reads/n0_endofrun_2026-09-27}"
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
mkdir -p "$STATE"
if [ -f "$STATE/supervisor.pid" ] && kill -0 "$(cat "$STATE/supervisor.pid")" 2>/dev/null; then
  echo "already running: pid $(cat "$STATE/supervisor.pid")"; exit 0
fi
cd "$TREE"
PYTHONPATH="$TREE/src" setsid nohup nice -n 15 "$PY" "$HERE/queue.py" run \
  >> "$STATE/supervisor.log" 2>&1 < /dev/null &
sleep 2
echo "started: pid $(cat "$STATE/supervisor.pid")  state $STATE  tree $TREE"
