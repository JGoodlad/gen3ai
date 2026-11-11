#!/usr/bin/env bash
# CE-3d, the trailing-magnet screen: the registered CROSSES (PREREG §5), run ONCE after the sixth fork's arm-end checks.
#   deciding   : 3 x 3, the MAGNET forks rb_x35_ce3dmag_s7301..s7303 against the CONTROL forks
#                rb_x35_ce3dent_s7311..s7313, 2,000 mirrored pairs a cell; family ce3d_magnet_vs_control_h2h
#   mag_vs_c0  : 3 x c, the magnet forks against the C0 finals the plan admits (5 <= c <= 8), 1,000 pairs a cell;
#                family ce3d_magnet_vs_c0_h2h                    (REPORTED)
#   ent_vs_c0  : 3 x c, the control forks against the same C0 finals, 1,000 pairs a cell;
#                family ce3d_control_vs_c0_h2h                   (REPORTED)
#   Each final_model.zip, batches of 500 pairs, both sides greedy, schedule seed 0, the players' eval team builder
#   (no panel), --oracle-reveal-mode off, purpose ab, label ce3d_magnet_screen, the archive's ledger. ONE family and
#   ONE request a cross (a family's requests are its LOOKS, and each cross has one). The deciding cross is played
#   FIRST. A failed play HOLDs (resumable); the reader names an incomplete reported cross as unavailable.
# PLAYED AT THE CE-3d PIN (a detached checkout PIN_DIR, verified by rev-parse + clean src/data), never at HEAD and
# never at the plain production pin: f4b25238 refuses a config that carries the magnet's two fields, and HEAD's game
# and observation are not the ones these finals trained on (PREREG §4). The CE-3d pin loads C0's v161 finals as
# f4b25238 does (its only non-magnet change is the T2 near-tie judge, which no play path calls).
#   git worktree add --detach "$PIN_DIR" 333a73ddda7829b9ccdf9e17ef04624eb40edeba
#   (cd "$PIN_DIR" && ./scripts/bootstrap.sh --skip-env --with-rust)     # once, under mem_cap, rule 5b
# usage: drive_cross.sh <PIN_DIR> <WORK_DIR> cpu|cuda
#   cuda: the caller holds the GPU lease (GEN3AI_GPU_LEASE_TOKEN or _FILE in the environment).
#   cpu : rule 5b's shape (nice 19, the agent cores from utils.core_split, CUDA hidden). No outer timeout: each
#         job is slot-queued by mem_cap. Measured rate (X67's cross, three threads): 48,000 games in 27,677 s, so
#         deciding (36,000 games) ~ 5.8 h and each reported cross (48,000 at c = 8) ~ 7.7 h: ~ 21 h in all.
# Resumable: the ledger's resume skips recorded batches. Exit 0 = played and read; the verdict is WORK_DIR/result.
set -u
PIN_DIR=${1:?PIN_DIR}; W=${2:?WORK_DIR}; DEV=${3:?cpu|cuda}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
PIN=333a73ddda7829b9ccdf9e17ef04624eb40edeba
MAIN=/home/goodlad/dev/gen3ai; OPS=$MAIN/scripts/ops
HERE=$MAIN/designs/research_state/measurements/ce3d_magnet_screen_2026-11-11
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
n=$("$PY" "$HERE/cross/plan_cross.py" --out-dir "$W/plan" 2>>"$LOG") || { log "HOLD: plan REFUSED (see $LOG)"; exit 3; }
log "plan: $n cells; registering the three families and playing ($DEV)"
SRC="ce3d_magnet_screen_2026-11-11/PREREG.md §5 (pin ${PIN:0:8})"
RULE_DECIDING="$SRC, the DECIDING cross: magnet forks (P*+20M, --magnet-kl-coef 0.05, ent_coef 0.05) x entropy-matched control forks (ent_coef 0.1), 3 x 3 at 2,000 pairs a cell, D=mean(h)-50, V=s2R/3+s2C/3 with binomial floors, Welch df, 90% interval; PASS iff lower end > 0, WORSE iff upper end < 0, else NOT-SHOWN; one look (screen_rule.decide)"
RULE_MAG="$SRC, REPORTED, never deciding: magnet forks x C0 finals (ent_coef 0.05, pin f4b25238), 3 x c at 1,000 pairs a cell, X67's statistic (screen_rule.report); no class"
RULE_ENT="$SRC, REPORTED, never deciding: control forks (ent_coef 0.1) x C0 finals (ent_coef 0.05, pin f4b25238), 3 x c at 1,000 pairs a cell, X67's statistic (screen_rule.report); no class"
play() {  # <cross> <request; its family is <request>_h2h> <pairs> <the family's rule>
  "$PY" -m main.eval_ledger family-register --id "$2_h2h" --decision-kind ab_verdict --rule "$4" \
    --protocol gen3_eval_protocol_v1_h2h --commit "$PIN" >> "$LOG" 2>&1 || { log "family-register failed for $2_h2h"; return 3; }
  local PLAY=("$PY" -m main.h2h play-many --cells "$W/plan/cells_$1.json" --pairs "$3" --batch-pairs 500 --seed 0
              --device "$DEV" --purpose ab --family "$2_h2h" --request "$2" --label ce3d_magnet_screen
              --oracle-reveal-mode off)
  log "playing $1 (family $2_h2h, request $2, $3 pairs a cell)"
  if [ "$DEV" = cuda ]; then
    "$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 "${PLAY[@]}" >> "$LOG" 2>&1
  else
    nice -n 19 taskset -c "$CPUS" "$OPS/mem_cap.sh" 24 "${PLAY[@]}" --threads 3 --torch-threads 3 >> "$LOG" 2>&1
  fi
}
play deciding ce3d_magnet_vs_control 2000 "$RULE_DECIDING" || { log "HOLD: play-many failed on the DECIDING cross (see $LOG)"; exit 4; }
play mag_vs_c0 ce3d_magnet_vs_c0 1000 "$RULE_MAG" || { log "HOLD: play-many failed on magnet vs C0 (see $LOG)"; exit 4; }
play ent_vs_c0 ce3d_control_vs_c0 1000 "$RULE_ENT" || { log "HOLD: play-many failed on control vs C0 (see $LOG)"; exit 4; }
log "played; reading and applying the coded rule"
V=$("$PY" "$HERE/cross/read_cross.py" "$W/plan/shas.json" "$W/result" "$HERE/screen_rule.py" 2>>"$LOG") \
  || { log "HOLD: read failed (see $LOG)"; exit 6; }
log "DONE: $(echo "$V" | cut -c1-400)"
