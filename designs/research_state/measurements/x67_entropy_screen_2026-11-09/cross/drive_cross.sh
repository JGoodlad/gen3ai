#!/usr/bin/env bash
# X67, the entropy screen: the registered CROSS (PREREG §5), run ONCE after the third X67 fork's arm-end checks.
#   3 x c cells (rows rb_x35_x67ent_s6701..s6703, columns the C0 finals the plan admits, 5 <= c <= 8), each
#   final_model.zip, 1,000 mirrored pairs a cell (batches of 500), both sides greedy, schedule seed 0, the players'
#   eval team builder (no panel), --oracle-reveal-mode off, purpose ab, family x67_entropy_screen_h2h, request
#   x67_entropy_screen, label x67_entropy_screen, the archive's ledger.
# PLAYED AT THE PRODUCTION SIDE PIN f4b25238 (a detached checkout PIN_DIR, verified by rev-parse + clean src/data),
# never at HEAD: HEAD's game and observation are not the ones these finals trained on (PREREG §4).
#   git worktree add --detach "$PIN_DIR" f4b25238def57900c0305bb2a3a5a8f9edaef285
#   (cd "$PIN_DIR" && ./scripts/bootstrap.sh --skip-env --with-rust)     # once, under mem_cap, rule 5b
# usage: drive_cross.sh <PIN_DIR> <WORK_DIR> cpu|cuda
#   cuda: the caller holds the GPU lease (GEN3AI_GPU_LEASE_TOKEN or _FILE in the environment); ~20 min at c = 8.
#   cpu : rule 5b's shape (nice 19, the agent cores, CUDA hidden); ~3 h at c = 5, ~5 h at c = 8. No outer timeout:
#         the job is slot-queued by mem_cap.
# Resumable: the ledger's resume skips recorded batches. Exit 0 = played and read; the verdict is WORK_DIR/result.
set -u
PIN_DIR=${1:?PIN_DIR}; W=${2:?WORK_DIR}; DEV=${3:?cpu|cuda}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
PIN=f4b25238def57900c0305bb2a3a5a8f9edaef285
MAIN=/home/goodlad/dev/gen3ai; OPS=$MAIN/scripts/ops
HERE=$MAIN/designs/research_state/measurements/x67_entropy_screen_2026-11-09
FAMILY=x67_entropy_screen_h2h; REQUEST=x67_entropy_screen
mkdir -p "$W"; LOG=$W/run.log
log() { echo "[$(date -Is)] $1" | tee -a "$LOG"; }
cd "$PIN_DIR" || { log "HOLD: no pin checkout at $PIN_DIR"; exit 5; }
HEAD=$(git rev-parse HEAD)
if [ "$HEAD" != "$PIN" ] || [ -n "$(git status --porcelain --untracked-files=no -- src data)" ]; then
  log "HOLD: play checkout HEAD=$HEAD (need $PIN) or src/data dirty"; exit 5; fi
export PYTHONPATH=$PIN_DIR/src GEN3AI_SHARED_COMPILE_CACHE=0
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=""
"$PY" -c "import main.h2h; print('main.h2h from', main.h2h.__file__)" >> "$LOG" 2>&1
n=$("$PY" "$HERE/cross/plan_cross.py" --out "$W/cells_plan.json" --shas "$W/shas.json" 2>>"$LOG") \
  || { log "HOLD: plan REFUSED (see $LOG)"; exit 3; }
log "plan: $n cells; registering the family and playing (1000 pairs a cell, batches of 500, $DEV)"
RULE="x67_entropy_screen_2026-11-09/PREREG.md §5 (pin f4b25238): X67 forks (P*+20M, ent_coef 0.0125) x C0 finals (ent_coef 0.05), 3 x c cross, D=mean(h)-50, V=s2R/3+s2C/c with binomial floors, Welch df, 90% interval; PASS iff lower end > 0, WORSE iff upper end < 0, else NOT-SHOWN; one look (screen_rule.py)"
"$PY" -m main.eval_ledger family-register --id "$FAMILY" --decision-kind ab_verdict --rule "$RULE" \
  --protocol gen3_eval_protocol_v1_h2h --commit "$PIN" >> "$LOG" 2>&1 || { log "HOLD: family-register failed (see $LOG)"; exit 3; }
PLAY=("$PY" -m main.h2h play-many --cells "$W/cells_plan.json" --pairs 1000 --batch-pairs 500 --seed 0 --device "$DEV"
      --purpose ab --family "$FAMILY" --request "$REQUEST" --label x67_entropy_screen --oracle-reveal-mode off)
if [ "$DEV" = cuda ]; then
  "$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 "${PLAY[@]}" >> "$LOG" 2>&1 || { log "HOLD: play-many failed (see $LOG)"; exit 4; }
else
  nice -n 19 taskset -c 5-7,13-15 "$OPS/mem_cap.sh" 24 "${PLAY[@]}" --threads 3 --torch-threads 3 >> "$LOG" 2>&1 \
    || { log "HOLD: play-many failed (see $LOG)"; exit 4; }
fi
log "played; reading and applying the coded rule"
V=$("$PY" "$HERE/cross/read_cross.py" "$W/shas.json" "$W/result" "$HERE/screen_rule.py" 2>>"$LOG") \
  || { log "HOLD: read failed (see $LOG)"; exit 6; }
log "DONE: $(echo "$V" | cut -c1-400)"
