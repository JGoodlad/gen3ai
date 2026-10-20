#!/usr/bin/env bash
# STAGE 1c's runner under the heavy-slot policy: ONE bounded chunk per heavy slot (PREREGISTRATION §7). Each chunk is
# `run_1c.py --one-chunk` (no new state after --budget-min; the whole call capped by `timeout`) inside its own
# `mem_cap.sh --slot`, released between chunks. PHASE A ends on A_DONE (exit 3) or the registered stop (exit 4); then the
# determinism recheck (G3) runs in one more slot. PHASE B (`loop_1c.sh play`) runs the identity tooth, then the playout
# chunks to B_DONE / B_STOPPED. Any other exit retries after a minute, at most 5 times in a row. Launched detached by
# `launch_1c.sh`; CPU only, SCHED_IDLE, nice 19, the agent cores (root rule 5b).
#   loop_1c.sh [a|play] <budget-min> <timeout>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
PHASE="${1:-a}"; BUDGET="${2:-30}"; TO="${3:-45m}"
ARCH="${S1C_ARCHIVE:-$HOME/gen3ai_archive/search_1c}"
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 S1C_ARCHIVE="$ARCH"
mkdir -p "$ARCH/logs"
log() { echo "$(date '+%F %T') $*" >> "$ARCH/logs/queue.log"; }
slot() {  # slot <name> <timeout> <cmd...>
  local name="$1" to="$2"; shift 2
  chrt --idle 0 nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name "$name" --slot 16 timeout "$to" \
    taskset -c "${S1C_CPUS:-5-7,13-15}" "$@"
}
log "loop up ($PHASE): pid $$, checkout $(git -C "$REPO" rev-parse --short HEAD), budget $BUDGET min, timeout $TO"
bad=0
if [ "$PHASE" = a ]; then
  while true; do
    slot s1c_chunk "$TO" "$PY" -I "$HERE/run_1c.py" --one-chunk --budget-min "$BUDGET"; rc=$?
    case "$rc" in
      0) bad=0 ;;
      3|4) log "phase A ends (rc $rc)"; break ;;
      *) bad=$((bad + 1)); log "chunk rc $rc ($bad in a row)"
         [ "$bad" -ge 5 ] && { log "loop gives up"; exit 1; }; sleep 60 ;;
    esac
  done
  slot s1c_recheck "$TO" "$PY" -I "$HERE/run_1c.py" --recheck; log "recheck rc $?"
  log "PHASE_A_LOOP_DONE"
else
  slot s1c_identity "$TO" "$HERE/run_play.sh" play_1c.py --identity; rc=$?
  log "identity rc $rc"
  [ "$rc" -eq 0 ] || { log "THE IDENTITY TOOTH did not pass: phase B stops"; exit 1; }
  while true; do
    slot s1c_play "$TO" "$HERE/run_play.sh" play_1c.py --one-chunk --budget-min "$BUDGET"; rc=$?
    case "$rc" in
      0) bad=0 ;;
      3|4) log "phase B ends (rc $rc)"; break ;;
      *) bad=$((bad + 1)); log "play chunk rc $rc ($bad in a row)"
         [ "$bad" -ge 5 ] && { log "loop gives up"; exit 1; }; sleep 60 ;;
    esac
  done
  log "PHASE_B_LOOP_DONE"
fi
