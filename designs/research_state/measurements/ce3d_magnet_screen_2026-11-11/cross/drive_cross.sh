#!/usr/bin/env bash
# CE-3d, the trailing-magnet screen, Amendment 1: the registered CROSS (PREREG §5), run ONCE after the third swap
# fork's arm-end checks. X67's cross (../../x67_entropy_screen_2026-11-09/cross/drive_cross.sh) with the swap forks
# as its rows.
#   3 x c cells (rows rb_x35_ce3dswap_s7321..s7323, columns the C0 finals the plan admits, 5 <= c <= 8), each
#   final_model.zip, 1,000 mirrored pairs a cell (batches of 500), both sides greedy, schedule seed 0, the players'
#   eval team builder (no panel), --oracle-reveal-mode off, purpose ab, family ce3d_swap_vs_c0_h2h, request
#   ce3d_swap_vs_c0, label ce3d_magnet_screen, the archive's ledger.
# PLAYED AT THE CE-3d PIN (a detached checkout PIN_DIR, verified by rev-parse + clean src/data), never at HEAD and
# never at the plain production pin: f4b25238 refuses a config that carries the magnet's two fields, and HEAD's game
# and observation are not the ones these finals trained on (PREREG §4). The CE-3d pin loads C0's v161 finals as
# f4b25238 does (its only non-magnet change is the T2 near-tie judge, which no play path calls). X67's cross was
# played at f4b25238: the play pin differs between the two screens and PREREG §4.3 says so.
#   git worktree add --detach "$PIN_DIR" 333a73ddda7829b9ccdf9e17ef04624eb40edeba
#   (cd "$PIN_DIR" && ./scripts/bootstrap.sh --skip-env --with-rust)     # once, under mem_cap, rule 5b
# usage: drive_cross.sh <PIN_DIR> <WORK_DIR> cpu|cuda
#   cuda: the caller holds the GPU lease (GEN3AI_GPU_LEASE_TOKEN or _FILE in the environment); ~20 min at c = 8.
#   cpu : rule 5b's shape (nice 19, the agent cores from utils.core_split, CUDA hidden). No outer timeout: the job
#         is slot-queued by mem_cap. Measured rate (X67's cross, three threads): 48,000 games in 27,677 s, so
#         ~ 4.8 h at c = 5 and ~ 7.7 h at c = 8.
# Resumable: the ledger's resume skips recorded batches. Exit 0 = played and read; the verdict is WORK_DIR/result.
set -u
PIN_DIR=${1:?PIN_DIR}; W=${2:?WORK_DIR}; DEV=${3:?cpu|cuda}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
PIN=333a73ddda7829b9ccdf9e17ef04624eb40edeba
MAIN=/home/goodlad/dev/gen3ai; OPS=$MAIN/scripts/ops
HERE=$MAIN/designs/research_state/measurements/ce3d_magnet_screen_2026-11-11
FAMILY=ce3d_swap_vs_c0_h2h; REQUEST=ce3d_swap_vs_c0
mkdir -p "$W"; LOG=$W/run.log
log() { echo "[$(date -Is)] $1" | tee -a "$LOG"; }
cd "$PIN_DIR" || { log "HOLD: no pin checkout at $PIN_DIR"; exit 5; }
HEAD=$(git rev-parse HEAD)
if [ "$HEAD" != "$PIN" ] || [ -n "$(git status --porcelain --untracked-files=no -- src data)" ]; then
  log "HOLD: play checkout HEAD=$HEAD (need $PIN) or src/data dirty"; exit 5; fi
CPUS=$(PYTHONPATH=$MAIN/src "$PY" -m utils.core_split agent-cpus) || { log "HOLD: utils.core_split failed"; exit 5; }
export PYTHONPATH=$PIN_DIR/src GEN3AI_SHARED_COMPILE_CACHE=0
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=""
"$PY" -c "import main.h2h; print('main.h2h from', main.h2h.__file__)" >> "$LOG" 2>&1
n=$("$PY" "$HERE/cross/plan_cross.py" --out "$W/cells_plan.json" --shas "$W/shas.json" 2>>"$LOG") \
  || { log "HOLD: plan REFUSED (see $LOG)"; exit 3; }
log "plan: $n cells; registering the family and playing (1000 pairs a cell, batches of 500, $DEV)"
RULE="ce3d_magnet_screen_2026-11-11/PREREG.md §5, Amendment 1 (play pin ${PIN:0:8}): swap forks (P*+20M, --ent-coef 0 --magnet-kl-coef 0.05, pin ${PIN:0:8}) x C0 finals (ent_coef 0.05, pin f4b25238), 3 x c cross, D=mean(h)-50, V=s2R/3+s2C/c with binomial floors, Welch df, 90% interval; PASS iff lower end > 0, WORSE iff upper end < 0, else NOT-SHOWN; one look (screen_rule.py, X67's rule)"
"$PY" -m main.eval_ledger family-register --id "$FAMILY" --decision-kind ab_verdict --rule "$RULE" \
  --protocol gen3_eval_protocol_v1_h2h --commit "$PIN" >> "$LOG" 2>&1 || { log "HOLD: family-register failed (see $LOG)"; exit 3; }
PLAY=("$PY" -m main.h2h play-many --cells "$W/cells_plan.json" --pairs 1000 --batch-pairs 500 --seed 0 --device "$DEV"
      --purpose ab --family "$FAMILY" --request "$REQUEST" --label ce3d_magnet_screen --oracle-reveal-mode off)
if [ "$DEV" = cuda ]; then
  "$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 "${PLAY[@]}" >> "$LOG" 2>&1 || { log "HOLD: play-many failed (see $LOG)"; exit 4; }
else
  nice -n 19 taskset -c "$CPUS" "$OPS/mem_cap.sh" 24 "${PLAY[@]}" --threads 3 --torch-threads 3 >> "$LOG" 2>&1 \
    || { log "HOLD: play-many failed (see $LOG)"; exit 4; }
fi
log "played; reading and applying the coded rule"
V=$("$PY" "$HERE/cross/read_cross.py" "$W/shas.json" "$W/result" "$HERE/screen_rule.py" 2>>"$LOG") \
  || { log "HOLD: read failed (see $LOG)"; exit 6; }
log "DONE: $(echo "$V" | cut -c1-400)"
