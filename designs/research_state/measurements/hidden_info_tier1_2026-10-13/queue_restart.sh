#!/usr/bin/env bash
# Restart the playout queue without duplicating work: stop the queue process (its in-flight unit keeps running), wait
# for that unit's play processes to exit, then start the queue again with the given arguments.
#   queue_restart.sh <queue pid> "<child pid> <child pid> ..." [queue_play.py args...]
set -uo pipefail
QPID="$1"; CHILDREN="$2"; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kill "$QPID" 2>/dev/null || true
for p in $CHILDREN; do
  while kill -0 "$p" 2>/dev/null; do sleep 20; done
done
cd "$HERE"
exec nice -n 19 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -I queue_play.py "$@" \
  >> /home/goodlad/gen3ai_archive/hidden_info_tier1/logs/queue_play.log 2>&1
