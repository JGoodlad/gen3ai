#!/usr/bin/env bash
# THE RUNNER for the toy crosses (PREREG.md): registers the two families (once), then plays R-T5 then R-T6 on CPU under
# rule 5b (py.sh: taskset 5-7,13-15, nice 19, OMP 1, no GPU) + mem_cap, 250 mirrored pairs per cell in 50-pair batches.
# INCREMENTAL + RESUMABLE: play-many records each batch in models/_ledger and a re-run skips what is recorded.
# Status lines -> $HERE/status.txt; engine output -> /tmp/toyx_run.log. Start detached:
#   setsid nohup <this> > /dev/null 2>&1 < /dev/null &
set -u
HERE=$(cd "$(dirname "$0")" && pwd); TOP=$(git -C "$HERE" rev-parse --show-toplevel)
ST=$HERE/status.txt; LOG=/tmp/toyx_run.log; PLAN=$HERE/plan
log() { echo "[$(date -Is)] $1" >> "$ST"; }
echo $$ > /tmp/toyx_run.pid
C=$(git -C "$TOP" rev-parse HEAD)
[ -z "$(git -C "$TOP" status --porcelain --untracked-files=no -- src)" ] || { log "HOLD: $TOP/src is dirty"; exit 5; }
log "RUNNER start pid $$ at $C (src clean)"
reg() {
  "$HERE/py.sh" -m main.eval_ledger family-register --id "$1" --decision-kind ab_verdict --rule "$2" \
    --protocol gen3_eval_protocol_v1_h2h --commit "$C" >> "$LOG" 2>&1 || log "family-register $1 rc!=0 (already registered on a resume? see $LOG)"
}
if [ ! -f "$HERE/.families_registered" ]; then
  reg toyx_rt5_eqcompute "toy_crosses_2026-10-16/PREREG.md R-T5: T-ENT s4001/s4002/s4003 at the equal-compute checkpoint x rb_bf_eb s2001-s2008 15M finals, 250 pairs/cell, DESCRIPTIVE"
  reg toyx_rt6_cap "toy_crosses_2026-10-16/PREREG.md R-T6: T-ENT s4001b/s4002/s4003 100M final x rb_bf_eb s2001-s2008 15M finals, 250 pairs/cell, DESCRIPTIVE"
  touch "$HERE/.families_registered"
fi
for X in "rt5 toyx_rt5_eqcompute toyx_rt5" "rt6 toyx_rt6_cap toyx_rt6"; do
  set -- $X
  log "PLAY $1 (24 cells x 250 pairs)"
  "$TOP/scripts/ops/mem_cap.sh" 16 "$HERE/py.sh" -m main.h2h play-many \
    --cells "$PLAN/cells_$1.json" --pairs 250 --batch-pairs 50 --device cpu --torch-threads 1 --threads 2 \
    --purpose ab --family "$2" --request "$3" --label toy_crosses --oracle-reveal-mode off >> "$LOG" 2>&1 \
    || { log "HOLD: play-many $1 failed (rc $?, $LOG)"; exit 10; }
  log "PLAYED $1"
done
log "RUNNER done — read: py.sh read_crosses.py"
