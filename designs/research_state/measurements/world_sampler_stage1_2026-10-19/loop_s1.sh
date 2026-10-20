#!/usr/bin/env bash
# Stage 1's runner under the orchestrator's heavy-slot policy (2026-10-20): ONE bounded chunk per heavy slot. Each
# chunk is `queue_s1.py --one-chunk` inside its own `mem_cap.sh --slot` (its builds and plays take no second slot);
# the slot is released between chunks. Stops on QUEUE_DONE (exit 3) or the registered stop (exit 4); any other exit
# retries after a minute, at most 5 times in a row. Launched detached (setsid nohup) with the env of launch_s1.sh plus
# WS1_RUN_S1 = the BUILDER checkout's run_s1.sh (so every row keeps the launch commit's drivers).
#   loop_s1.sh <chunk size>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
CHUNK="${1:-4}"
ARCH="${WS1_ARCHIVE:?}"
bad=0
echo "$(date '+%F %T') loop up: pid $$, runner $(git -C "$REPO" rev-parse --short HEAD), chunk $CHUNK, run_s1 ${WS1_RUN_S1:?}" >> "$ARCH/logs/queue.log"
while true; do
  "$REPO/scripts/ops/mem_cap.sh" --name ws1_chunk --slot 2 timeout 3h \
    /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -I "$HERE/queue_s1.py" --one-chunk --chunk "$CHUNK"
  rc=$?
  case "$rc" in
    0) bad=0 ;;
    3|4) echo "$(date '+%F %T') loop ends (rc $rc)" >> "$ARCH/logs/queue.log"; exit 0 ;;
    *) bad=$((bad + 1)); echo "$(date '+%F %T') chunk call rc $rc ($bad in a row)" >> "$ARCH/logs/queue.log"
       [ "$bad" -ge 5 ] && { echo "$(date '+%F %T') loop gives up" >> "$ARCH/logs/queue.log"; exit 1; }
       sleep 60 ;;
  esac
done
