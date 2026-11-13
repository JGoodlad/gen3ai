#!/usr/bin/env bash
# X70, the ent-0 control of the trailing-magnet screen: the CROSSES (PREREG §5), each run ONCE after the fourth X70
# fork's arm-end checks. CE-3d's driver (../../ce3d_magnet_screen_2026-11-11/cross/drive_cross.sh) with this
# screen's three crosses. The fourth argument picks the cross:
#   anchor (the default) THE REGISTERED CROSS, which decides: 3 x 4 cells (rows rb_x35_ce3dswap_s7321..s7323, columns
#          rb_x35_x70ent0_s7331..s7334), 4,000 mirrored pairs a cell, family x70_swap_vs_ent0_h2h, request
#          x70_swap_vs_ent0; read by screen_rule.decide into WORK_DIR/result/verdict.json.
#   c0     REPORTED, decides NOTHING (PREREG §5.4): 4 x 8 cells (rows the X70 forks, columns rb_x35_c0_s3501..s3508),
#          1,000 pairs a cell, family x70_ent0_vs_c0_h2h, request x70_ent0_vs_c0; read by screen_rule.report into
#          WORK_DIR/result/report_c0.json (no class, never a verdict.json).
#   x67    REPORTED, decides NOTHING: 4 x 3 cells (columns rb_x35_x67ent_s6701..s6703), 1,000 pairs a cell, family
#          x70_ent0_vs_x67_h2h, request x70_ent0_vs_x67; WORK_DIR/result/report_x67.json.
#   All: each final_model.zip, batches of 500 pairs, both sides greedy, schedule seed 0, the players' eval team
#   builder (no panel), --oracle-reveal-mode off, purpose ab, label x70_ent0_control, the archive's ledger. No cross
#   waits on another and none reads another's cells; give each its OWN WORK_DIR (the driver HOLDs on a WORK_DIR that
#   another cross has used). PLAY THE DECIDING CROSS FIRST.
# PLAYED AT THE FIRST CE-3d PIN 333a73dd (a detached checkout PIN_DIR, verified by rev-parse + clean src/data), the
# pin CE-3d's two crosses were played at, never at HEAD and never at the plain production pin: f4b25238 refuses a
# config that carries the magnet's two fields (a swap fork's AND an X70 fork's: the training pin writes
# magnet_kl_coef 0.0 into every config), and HEAD's game and observation are not the ones these finals trained on.
# The forks' TRAINING pin is 9e97cad5 (TRAIN_PIN below = 333a73dd + the startup fit check's fix, three paths no play
# code imports).
#   git worktree add --detach "$PIN_DIR" 333a73ddda7829b9ccdf9e17ef04624eb40edeba
#   (cd "$PIN_DIR" && ./scripts/bootstrap.sh --skip-env --with-rust)     # once, under mem_cap, rule 5b
#   (CE-3d's play checkout /home/goodlad/dev/gen3ai-reads/ce3d_play_333a73dd is that checkout, if it still stands.)
# usage: drive_cross.sh <PIN_DIR> <WORK_DIR> cpu|cuda [anchor|c0|x67]
#   cuda: the caller holds the GPU lease (GEN3AI_GPU_LEASE_TOKEN or _FILE in the environment). ESTIMATES from the
#         figure X67's and CE-3d's drivers carry (48,000 games in about 20 min; UNVERIFIED, neither cross was played on
#         the card): anchor ~40 min, c0 ~27 min, x67 ~10 min.
#   cpu : rule 5b's shape (nice 19, the agent cores from utils.core_split, CUDA hidden). No outer timeout: the job
#         is slot-queued by mem_cap. Measured rate (CE-3d's deciding cross, three threads): 48,000 games in
#         16,579 s, so ~ 9.2 h for anchor (96,000 games), ~ 6.1 h for c0 (64,000), ~ 2.3 h for x67 (24,000).
# Resumable: the ledger's resume skips recorded batches. Exit 0 = played and read; the result is WORK_DIR/result.
set -u
PIN_DIR=${1:?PIN_DIR}; W=${2:?WORK_DIR}; DEV=${3:?cpu|cuda}; CROSS=${4:-anchor}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
PIN=333a73ddda7829b9ccdf9e17ef04624eb40edeba          # the PLAY pin (the first CE-3d pin)
TRAIN_PIN=9e97cad5709da21a1f9d4d8846096c87eb17b1d0    # the swap and X70 forks' TRAINING pin; named in RULE only
MAIN=/home/goodlad/dev/gen3ai; OPS=$MAIN/scripts/ops
HERE=$MAIN/designs/research_state/measurements/x70_ent0_control_2026-11-13
COMMON="D=mean(h)-50, V=s2R/r+s2C/c with binomial floors, Welch df, 90% interval"
X70="X70 forks (P*+20M, --ent-coef 0, no magnet, pin ${TRAIN_PIN:0:8})"
case "$CROSS" in
  anchor) FAMILY=x70_swap_vs_ent0_h2h; REQUEST=x70_swap_vs_ent0; PAIRS=4000
       RULE="x70_ent0_control_2026-11-13/PREREG.md §5 (play pin ${PIN:0:8}): swap forks (P*+20M, --ent-coef 0 --magnet-kl-coef 0.05, pin ${TRAIN_PIN:0:8}) x $X70, 3 x 4 cross, 4000 pairs a cell, $COMMON; ANCHOR-ADDS iff lower end > 0, ANCHOR-COSTS iff upper end < 0, else NOT-SEPARATED; one look (screen_rule.decide)" ;;
  c0)  FAMILY=x70_ent0_vs_c0_h2h; REQUEST=x70_ent0_vs_c0; PAIRS=1000
       RULE="x70_ent0_control_2026-11-13/PREREG.md §5.4 (play pin ${PIN:0:8}): REPORTED, DECIDES NOTHING, NO CLASS. $X70 x C0 finals (ent_coef 0.05, pin f4b25238), 4 x 8 cross, 1000 pairs a cell, $COMMON; a reading only; one look (screen_rule.report)" ;;
  x67) FAMILY=x70_ent0_vs_x67_h2h; REQUEST=x70_ent0_vs_x67; PAIRS=1000
       RULE="x70_ent0_control_2026-11-13/PREREG.md §5.4 (play pin ${PIN:0:8}): REPORTED, DECIDES NOTHING, NO CLASS. $X70 x X67 finals (ent_coef 0.0125, pin f4b25238), 4 x 3 cross, 1000 pairs a cell, $COMMON; a reading only; one look (screen_rule.report)" ;;
  *)   echo "usage: drive_cross.sh <PIN_DIR> <WORK_DIR> cpu|cuda [anchor|c0|x67]" >&2; exit 2 ;;
esac
mkdir -p "$W"; LOG=$W/run.log
log() { echo "[$(date -Is)] $1" | tee -a "$LOG"; }
if [ -e "$W/cross" ] && [ "$(cat "$W/cross")" != "$CROSS" ]; then
  log "HOLD: WORK_DIR $W belongs to the '$(cat "$W/cross")' cross; give '$CROSS' its own"; exit 5; fi
echo "$CROSS" > "$W/cross"
cd "$PIN_DIR" || { log "HOLD: no pin checkout at $PIN_DIR"; exit 5; }
HEAD=$(git rev-parse HEAD)
if [ "$HEAD" != "$PIN" ] || [ -n "$(git status --porcelain --untracked-files=no -- src data)" ]; then
  log "HOLD: play checkout HEAD=$HEAD (need $PIN) or src/data dirty"; exit 5; fi
CPUS=$(PYTHONPATH=$MAIN/src "$PY" -m utils.core_split agent-cpus) || { log "HOLD: utils.core_split failed"; exit 5; }
export PYTHONPATH=$PIN_DIR/src GEN3AI_SHARED_COMPILE_CACHE=0
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=""
"$PY" -c "import main.h2h; print('main.h2h from', main.h2h.__file__)" >> "$LOG" 2>&1
n=$("$PY" "$HERE/cross/plan_cross.py" --cross "$CROSS" --out "$W/cells_plan.json" --shas "$W/shas.json" 2>>"$LOG") \
  || { log "HOLD: plan REFUSED (see $LOG)"; exit 3; }
log "plan ($CROSS): $n cells; registering family $FAMILY and playing ($PAIRS pairs a cell, batches of 500, $DEV)"
"$PY" -m main.eval_ledger family-register --id "$FAMILY" --decision-kind ab_verdict --rule "$RULE" \
  --protocol gen3_eval_protocol_v1_h2h --commit "$PIN" >> "$LOG" 2>&1 || { log "HOLD: family-register failed (see $LOG)"; exit 3; }
PLAY=("$PY" -m main.h2h play-many --cells "$W/cells_plan.json" --pairs "$PAIRS" --batch-pairs 500 --seed 0 --device "$DEV"
      --purpose ab --family "$FAMILY" --request "$REQUEST" --label x70_ent0_control --oracle-reveal-mode off)
if [ "$DEV" = cuda ]; then
  "$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 "${PLAY[@]}" >> "$LOG" 2>&1 || { log "HOLD: play-many failed (see $LOG)"; exit 4; }
else
  nice -n 19 taskset -c "$CPUS" "$OPS/mem_cap.sh" 24 "${PLAY[@]}" --threads 3 --torch-threads 3 >> "$LOG" 2>&1 \
    || { log "HOLD: play-many failed (see $LOG)"; exit 4; }
fi
log "played; reading and applying the coded $([ "$CROSS" = anchor ] && echo rule || echo report)"
V=$("$PY" "$HERE/cross/read_cross.py" "$W/shas.json" "$W/result" "$HERE/screen_rule.py" 2>>"$LOG") \
  || { log "HOLD: read failed (see $LOG)"; exit 6; }
log "DONE ($CROSS): $(echo "$V" | cut -c1-400)"
