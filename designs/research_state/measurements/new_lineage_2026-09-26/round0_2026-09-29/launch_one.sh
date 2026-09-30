#!/usr/bin/env bash
# Launch one round-0 arm after (a) a not-before time (the 10-min heads-up), (b) the GPU has no compute app,
# and (c) the precise tenant check is clean. Then dry-run, launch, and start the stock watcher.
#   bash launch_one.sh <run-name> <not-before HH:MM>      Touch $J/HOLD to stop it.
set -u
REPO=/home/goodlad/dev/gen3ai
J=/home/goodlad/.claude/jobs/popr0_2026-09-29
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
RUN=$1; NB=$2
STATUS=$J/chain_status.txt
log() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
log "LAUNCH-ONE START (pid $$) $RUN not-before $NB"
nb=$(date -d "$NB" +%s)
while [ "$(date +%s)" -lt "$nb" ]; do sleep 20; done
waited=0
while :; do
  [ -f "$J/HOLD" ] && { log "HOLD present — $RUN NOT launched"; exit 0; }
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -c .)
  if pgrep -f "^[^ ]*python[0-9.]* [^ ]*train_rl_agent\.py" >/dev/null; then tr=1; else tr=0; fi
  [ "$apps" = 0 ] && [ "$tr" = 0 ] && break
  [ "$waited" = 0 ] && log "waiting: gpu_apps=$apps trainer=$tr ($(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' '))"
  [ "$waited" -ge 5400 ] && { log "BLOCKED: GPU/trainer still busy after 90 min (gpu_apps=$apps trainer=$tr)"; exit 1; }
  sleep 30; waited=$((waited+30))
done
log "GPU empty, no trainer (waited ${waited}s)"
cd "$REPO" || exit 1
[ -e "models/$RUN" ] && { log "BLOCKED: models/$RUN exists"; exit 1; }
git fetch -q origin 2>/dev/null
PIN=$(tr ' ' '\n' < "$J/argv_$RUN.txt" | grep -A1 -- '--pin-commit' | tail -1)
d=$(git diff --stat "$PIN" origin/main -- data/ | tail -1)
[ -z "$d" ] || { log "BLOCKED: data/ differs $PIN..origin/main: $d"; exit 1; }
export PYTHONPATH="${PYTHONPATH:-}:src"
# shellcheck disable=SC2046
nice -n 15 "$P" -m main.launcher --dry-run $(cat "$J/argv_$RUN.txt") > "$J/dryrun_${RUN}_prelaunch.txt" 2>&1; rc=$?
[ "$rc" = 0 ] || { log "BLOCKED: dry-run rc=$rc"; exit 1; }
grep -aq "FORK of .*ai_v14_07_g0p_k2" "$J/dryrun_${RUN}_prelaunch.txt" || { log "BLOCKED: dry-run not a FORK of ai_v14_07_g0p_k2"; exit 1; }
# shellcheck disable=SC2046
nohup "$P" -m main.launcher $(cat "$J/argv_$RUN.txt") > "$J/launch/$RUN.log" 2>&1 < /dev/null &
pid=$!; echo "$pid" > "$J/$RUN.pid"
log "LAUNCHED $RUN launcher pid $pid"
n=0; until [ -d "models/$RUN" ]; do kill -0 "$pid" 2>/dev/null || { log "EXITED before run dir"; exit 1; }; sleep 10; n=$((n+1)); [ $n -ge 60 ] && break; done
setsid nohup bash scripts/ops/watch_run.sh "$RUN" --pid-file "$J/$RUN.pid" --launcher-log "$J/launch/$RUN.log" \
  --status-file "$J/watch_$RUN.txt" > "$J/watch_$RUN.out" 2>&1 < /dev/null &
log "watcher started for $RUN"
