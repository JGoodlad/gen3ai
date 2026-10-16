#!/usr/bin/env bash
# W5 (10-16 ~03:15): the chain v4 HELD at 03:07 on an E7 INCONCLUSIVE that was MY read bug (min_pairs 1,000 vs the registered
# 500); the corrected read (read_w4.py, min_pairs per cross) = E7 PASS; E6 WITHIN (ruling); E10 OK (03:07:34). This script does
# W5 alone: waits for x26/GO (touched after the E11 data/ freeze announcement), checks an idle GPU, launches rb_x26_s2601 at
# P_X26 from the MAIN checkout with the validated argv (x26_argv.txt), and starts its watcher.
set -u
REPO=/home/goodlad/dev/gen3ai; J=/home/goodlad/.claude/jobs/x26; ST=$J/status.txt
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3; OPS=$REPO/scripts/ops; P=$(cat "$J/pin.txt"); RUN=rb_x26_s2601
log() { echo "[$(date -Is)] $1" >> "$ST"; }
echo $$ > "$J/chain.pid"
log "X26 READY (corrected W4 read: E7 PASS; E6 WITHIN; E10 OK) — w5_launch.sh (pid $$) waits for $J/GO"
until [ -f "$J/GO" ]; do sleep 15; done
for _ in $(seq 40); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 15; done
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { log "HOLD: GPU busy before X26"; exit 2; }
[ -d "$REPO/models/$RUN" ] && { log "HOLD: models/$RUN exists"; exit 3; }
log "START $RUN pin=$P (main HEAD $(git -C "$REPO" rev-parse --short=8 HEAD)) — data/ FROZEN from now"
( cd "$REPO" && export PYTHONPATH="$REPO/src"; exec taskset -c 0-4,8-12 "$PY" -m main.launcher --restart-interval-hours 6 --pin-commit "$P" $(cat "$J/x26_argv.txt") ) > "$J/$RUN.launcher.log" 2>&1 < /dev/null &
LP=$!; echo "$LP" > "$J/$RUN.pid"
( for _ in $(seq 60); do [ -d "$REPO/models/$RUN" ] && break; sleep 30; done
  "$OPS/watch_run.sh" "$REPO/models/$RUN" --pid-file "$J/$RUN.pid" --launcher-log "$J/$RUN.launcher.log" --status-file "$J/$RUN.repo.watch.txt" ) > "$J/$RUN.repo.watch.out" 2>&1 < /dev/null &
wait "$LP"; RC=$?; log "END $RUN launcher_rc=$RC"
