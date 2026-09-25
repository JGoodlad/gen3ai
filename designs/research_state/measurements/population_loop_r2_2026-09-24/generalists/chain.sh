#!/usr/bin/env bash
# popr2 chain — TRAINING_RUN_SOP §2 LAYER 2. Round-2 generalists IN ORDER, adjacent: B2 -> C2.
# A stage starts only when the previous launcher EXITED with "Training complete"; anything else is
# BLOCKED (no auto-relaunch). `touch $J/HOLD` pauses before the next launch.
set -u
REPO=/home/goodlad/dev/gen3ai
J=/home/goodlad/.claude/jobs/popr2_2026-09-24
STATUS=$J/chain_status.txt
log() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 60; done; }
completed() { grep -aq "Training complete" "$REPO/models/$1/launcher_child.full.log" 2>/dev/null; }
log "CHAIN START (pid $$)"
for arm in B2 C2; do
  if [ -f "$J/HOLD" ]; then log "HOLD present before $arm — chain PAUSED"; exit 0; fi
  run=$(grep -oP -- '--run-name \K\S+' "$J/argv_$arm.txt")
  out=$(cd "$REPO" && bash "$J/launch_$arm.sh" 2>&1); rc=$?
  printf '%s\n' "$out" > "$J/launch_$arm.out"
  pid=$(grep -oP 'launcher pid \K[0-9]+' <<<"$out" | tail -1)
  if [ "$rc" -ne 0 ] || [ -z "$pid" ]; then log "BLOCKED: launch_$arm.sh ($run) rc=$rc — see $J/launch_$arm.out"; exit 1; fi
  echo "$pid" > "$J/$arm.pid"; log "LAUNCHED $arm $run launcher pid $pid"
  sleep 90
  (cd "$REPO" && setsid nohup bash scripts/ops/watch_run.sh "$run" --pid-file "$J/$arm.pid" \
      --launcher-log "$J/launch/${arm}_${run}.log" --status-file "$J/watch_$run.txt" \
      > "$J/watch_$arm.out" 2>&1 < /dev/null &)
  wait_pid "$pid"
  if ! completed "$run"; then log "BLOCKED: $run launcher exited WITHOUT 'Training complete'"; exit 1; fi
  log "DONE $arm $run"
done
log "CHAIN COMPLETE — both round-2 generalists finished"
