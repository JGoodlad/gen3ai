#!/usr/bin/env bash
# CE-3d, the trailing-magnet screen, Amendments 1 to 3: the CROSSES (PREREG §5), each run ONCE after the third swap
# fork's arm-end checks. X67's cross (../../x67_entropy_screen_2026-11-09/cross/drive_cross.sh) with the swap forks
# as its rows. The fourth argument picks the cross:
#   c0  (the default) THE REGISTERED CROSS, which decides: 3 x c cells (rows rb_x35_ce3dswap_s7321..s7323, columns
#       the C0 finals the plan admits, 5 <= c <= 8), family ce3d_swap_vs_c0_h2h, request ce3d_swap_vs_c0; read by
#       screen_rule.decide into WORK_DIR/result/verdict.json.
#   x67 Amendment 2's REPORTED cross, which decides NOTHING (PREREG §5.4): 3 x 3 cells against X67's finals
#       rb_x35_x67ent_s6701..s6703, family ce3d_swap_vs_x67_h2h, request ce3d_swap_vs_x67; read by
#       screen_rule.report_x67 into WORK_DIR/result/report_x67.json (no class, never a verdict.json).
#   Both: each final_model.zip, 1,000 mirrored pairs a cell (batches of 500), both sides greedy, schedule seed 0, the
#   players' eval team builder (no panel), --oracle-reveal-mode off, purpose ab, label ce3d_magnet_screen, the
#   archive's ledger. Neither waits on the other and neither reads the other's cells; give each its OWN WORK_DIR
#   (the driver HOLDs on a WORK_DIR that another cross has used).
# PLAYED AT THE FIRST CE-3d PIN 333a73dd (a detached checkout PIN_DIR, verified by rev-parse + clean src/data), never
# at HEAD and never at the plain production pin: f4b25238 refuses a config that carries the magnet's two fields, and
# HEAD's game and observation are not the ones these finals trained on (PREREG §4). That pin loads the v161 finals
# trained at f4b25238 (C0's and X67's: no magnet key in either) as f4b25238 does; its only non-magnet change is the
# T2 near-tie judge (PREREG §4.3). X67's own cross was played at f4b25238: the play pin differs between the two
# screens and PREREG §4.3 says so. Amendment 3 moved the swap forks' TRAINING pin to 9e97cad5 (SWAP_PIN below =
# 333a73dd + the startup fit check's fix, three paths no play code imports); the PLAY pin did not move.
#   git worktree add --detach "$PIN_DIR" 333a73ddda7829b9ccdf9e17ef04624eb40edeba
#   (cd "$PIN_DIR" && ./scripts/bootstrap.sh --skip-env --with-rust)     # once, under mem_cap, rule 5b
# usage: drive_cross.sh <PIN_DIR> <WORK_DIR> cpu|cuda [c0|x67]
#   cuda: the caller holds the GPU lease (GEN3AI_GPU_LEASE_TOKEN or _FILE in the environment); ~20 min at c = 8,
#         ~8 min for x67 (both ESTIMATES scaled from X67's).
#   cpu : rule 5b's shape (nice 19, the agent cores from utils.core_split, CUDA hidden). No outer timeout: the job
#         is slot-queued by mem_cap. Measured rate (X67's cross, three threads): 48,000 games in 27,677 s, so
#         ~ 4.8 h at c = 5, ~ 7.7 h at c = 8, and ~ 2.9 h for x67 (18,000 games).
# Resumable: the ledger's resume skips recorded batches. Exit 0 = played and read; the result is WORK_DIR/result.
set -u
PIN_DIR=${1:?PIN_DIR}; W=${2:?WORK_DIR}; DEV=${3:?cpu|cuda}; CROSS=${4:-c0}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
PIN=333a73ddda7829b9ccdf9e17ef04624eb40edeba          # the PLAY pin (the first CE-3d pin)
SWAP_PIN=9e97cad5709da21a1f9d4d8846096c87eb17b1d0     # the swap forks' TRAINING pin (Amendment 3); named in RULE only
MAIN=/home/goodlad/dev/gen3ai; OPS=$MAIN/scripts/ops
HERE=$MAIN/designs/research_state/measurements/ce3d_magnet_screen_2026-11-11
COMMON="D=mean(h)-50, V=s2R/3+s2C/c with binomial floors, Welch df, 90% interval"
case "$CROSS" in
  c0)  FAMILY=ce3d_swap_vs_c0_h2h; REQUEST=ce3d_swap_vs_c0
       RULE="ce3d_magnet_screen_2026-11-11/PREREG.md §5, Amendments 1 and 3 (play pin ${PIN:0:8}): swap forks (P*+20M, --ent-coef 0 --magnet-kl-coef 0.05, pin ${SWAP_PIN:0:8}) x C0 finals (ent_coef 0.05, pin f4b25238), 3 x c cross, $COMMON; PASS iff lower end > 0, WORSE iff upper end < 0, else NOT-SHOWN; one look (screen_rule.py, X67's rule)" ;;
  x67) FAMILY=ce3d_swap_vs_x67_h2h; REQUEST=ce3d_swap_vs_x67
       RULE="ce3d_magnet_screen_2026-11-11/PREREG.md §5.4, Amendments 2 and 3 (play pin ${PIN:0:8}): REPORTED, DECIDES NOTHING, NO CLASS. swap forks (P*+20M, --ent-coef 0 --magnet-kl-coef 0.05, pin ${SWAP_PIN:0:8}) x X67 finals (ent_coef 0.0125, pin f4b25238), 3 x 3 cross, $COMMON; a reading only, quoted when the registered verdict is PASS: lower end > 0 = evidence the anchor adds something beyond lower entropy, else the gain over C0 is plausibly the lower uniform push alone; one look (screen_rule.report_x67)" ;;
  *)   echo "usage: drive_cross.sh <PIN_DIR> <WORK_DIR> cpu|cuda [c0|x67]" >&2; exit 2 ;;
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
n=$("$PY" "$HERE/cross/plan_cross.py" --cols "$CROSS" --out "$W/cells_plan.json" --shas "$W/shas.json" 2>>"$LOG") \
  || { log "HOLD: plan REFUSED (see $LOG)"; exit 3; }
log "plan ($CROSS): $n cells; registering family $FAMILY and playing (1000 pairs a cell, batches of 500, $DEV)"
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
log "played; reading and applying the coded $([ "$CROSS" = c0 ] && echo rule || echo report)"
V=$("$PY" "$HERE/cross/read_cross.py" "$W/shas.json" "$W/result" "$HERE/screen_rule.py" 2>>"$LOG") \
  || { log "HOLD: read failed (see $LOG)"; exit 6; }
log "DONE ($CROSS): $(echo "$V" | cut -c1-400)"
