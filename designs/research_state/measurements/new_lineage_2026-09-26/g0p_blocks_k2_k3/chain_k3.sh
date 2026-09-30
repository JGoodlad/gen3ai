#!/usr/bin/env bash
# G0' block 3 chain: wait for K2's launcher to exit, verify K2 completed, pre-check, dry-run, launch K3,
# start its watcher. Orchestrator-approved 2026-09-28 ~21:00 PT. Touch $J/HOLD to stop it launching.
set -u
REPO=/home/goodlad/dev/gen3ai
J=/home/goodlad/.claude/jobs/g0p_k2_2026-09-28
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
PIN=a5b2fdbabaa412056ee1f80f3ffbb553f0dda18d
STATUS=$J/chain_status.txt
RUN=ai_v14_08_g0p_k3
log() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
K2PID=$(cat "$J/K2.pid")
log "K3 CHAIN START (pid $$) — waiting on K2 launcher pid $K2PID"
while kill -0 "$K2PID" 2>/dev/null; do sleep 30; done
cd "$REPO" || { log "BLOCKED: cannot cd $REPO"; exit 1; }
grep -aq "Training complete" models/ai_v14_07_g0p_k2/launcher_child.full.log 2>/dev/null \
  || { log "BLOCKED: K2 launcher gone without 'Training complete'"; exit 1; }
[ -f models/ai_v14_07_g0p_k2/final_model.zip ] || { log "BLOCKED: K2 final_model.zip missing"; exit 1; }
log "K2 complete"
[ -f "$J/HOLD" ] && { log "HOLD present — K3 NOT launched"; exit 0; }
[ -e "models/$RUN" ] && { log "BLOCKED: models/$RUN already exists"; exit 1; }
[ "$(grep -c __DG__ "$J/argv_K3.txt")" = "0" ] || { log "BLOCKED: __DG__ placeholder in argv_K3"; exit 1; }
git fetch -q origin 2>/dev/null
d=$(git diff --stat "$PIN" origin/main -- data/ | tail -1)
[ -z "$d" ] || { log "BLOCKED: data/ differs $PIN..origin/main: $d"; exit 1; }
# precise tenant check: only a PYTHON train_rl_agent.py process counts; wait up to 90 min for a test to finish
waited=0
while pgrep -f "^[^ ]*python[0-9.]* [^ ]*train_rl_agent\.py" >/dev/null; do
  [ "$waited" -ge 5400 ] && { log "BLOCKED: a trainer still live after 90 min: $(pgrep -af '^[^ ]*python[0-9.]* [^ ]*train_rl_agent\.py' | cut -c1-160 | head -2)"; exit 1; }
  [ "$waited" = 0 ] && log "tenant check: trainer live — waiting ($(pgrep -af '^[^ ]*python[0-9.]* [^ ]*train_rl_agent\.py' | cut -c1-120 | head -1))"
  sleep 60; waited=$((waited+60))
done
log "tenant check: no trainer live (waited ${waited}s)"
export PYTHONPATH="${PYTHONPATH:-}:src"
# shellcheck disable=SC2046
nice -n 15 "$P" -m main.launcher --dry-run $(cat "$J/argv_K3.txt") > "$J/dryrun_K3.txt" 2>&1; rc=$?
[ "$rc" = 0 ] || { log "BLOCKED: dry-run rc=$rc — see $J/dryrun_K3.txt"; exit 1; }
grep -aq "FORK of .*ai_v14_07_g0p_k2" "$J/dryrun_K3.txt" || { log "BLOCKED: dry-run did not resolve a FORK of ai_v14_07_g0p_k2"; exit 1; }
grep -aq "+8,000,000 steps" "$J/dryrun_K3.txt" || { log "BLOCKED: dry-run step delta is not +8,000,000"; exit 1; }
log "dry-run rc 0: FORK of ai_v14_07_g0p_k2, +8,000,000"
# shellcheck disable=SC2046
nohup "$P" -m main.launcher $(cat "$J/argv_K3.txt") > "$J/launch/K3_$RUN.log" 2>&1 < /dev/null &
pid=$!; echo "$pid" > "$J/K3.pid"
log "LAUNCHED K3 $RUN launcher pid $pid"
n=0; until [ -d "models/$RUN" ]; do kill -0 "$pid" 2>/dev/null || { log "EXITED before run dir existed — see $J/launch/K3_$RUN.log"; exit 1; }; sleep 10; n=$((n+1)); [ $n -ge 60 ] && break; done
setsid nohup bash scripts/ops/watch_run.sh "$RUN" --pid-file "$J/K3.pid" --launcher-log "$J/launch/K3_$RUN.log" \
  --status-file "$J/watch_K3.txt" > "$J/watch_K3.out" 2>&1 < /dev/null &
log "watcher started for $RUN"
