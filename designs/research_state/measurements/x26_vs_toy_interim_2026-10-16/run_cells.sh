#!/usr/bin/env bash
# THE RUNNER for the X26-vs-toy INTERIM head-to-head: registers the family (once), then plays the 3 cells on CPU under
# rule 5b (py.sh: taskset 5-7,13-15, nice 19, OMP 1, no GPU) + mem_cap, 500 mirrored pairs per cell in 50-pair batches
# (the toy-crosses invocation: toy_crosses_2026-10-16/run_crosses.sh). INCREMENTAL + RESUMABLE: play-many records each
# batch in models/_ledger and a re-run skips what is recorded. Status -> $HERE/status.txt; engine log -> /tmp/xvt_run.log.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); TOP=$(git -C "$HERE" rev-parse --show-toplevel)
ST=$HERE/status.txt; LOG=/tmp/xvt_run.log
log() { echo "[$(date -Is)] $1" >> "$ST"; }
echo $$ > /tmp/xvt_run.pid
C=$(git -C "$TOP" rev-parse HEAD)
[ -z "$(git -C "$TOP" status --porcelain --untracked-files=no -- src)" ] || { log "HOLD: $TOP/src is dirty"; exit 5; }
log "RUNNER start pid $$ at $C (src clean)"
if [ ! -f "$HERE/.family_registered" ]; then
  "$HERE/py.sh" -m main.eval_ledger family-register --id xvt_interim --decision-kind ab_verdict \
    --rule "x26_vs_toy_interim_2026-10-16 RESULTS.md: X26 70M checkpoint x toy s4001b/s4002/s4003 100M finals, 500 pairs/cell, DESCRIPTIVE and INTERIM, NOT R-T2" \
    --protocol gen3_eval_protocol_v1_h2h --commit "$C" >> "$LOG" 2>&1 || log "family-register rc!=0 (see $LOG)"
  touch "$HERE/.family_registered"
fi
log "PLAY (3 cells x 500 pairs)"
"$TOP/scripts/ops/mem_cap.sh" 16 "$HERE/py.sh" -m main.h2h play-many \
  --cells "$HERE/plan/cells.json" --pairs 500 --batch-pairs 50 --device cpu --torch-threads 1 --threads 2 \
  --purpose ab --family xvt_interim --request xvt_interim --label x26_vs_toy_interim --oracle-reveal-mode off >> "$LOG" 2>&1 \
  || { log "HOLD: play-many failed (rc $?, $LOG)"; exit 10; }
log "PLAYED"
