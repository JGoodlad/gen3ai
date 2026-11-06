#!/usr/bin/env bash
# X65 learnability probe (DESCRIPTIVE; README.md here): the four GPU arms, in order, each resumable. A chunk is
# <= 50 min (exit 3 = budget reached: the loop re-runs it); any other non-zero exit HOLDs. Run by the session holding
# the GPU lease, from a checkout that carries `main.x65.student probe` (the commit that shipped this folder or later).
#   GEN3AI_GPU_LEASE_TOKEN_FILE=<lease token file> gpu_arms.sh [arm ...]   # default: curve_h ctrl_h cap_h curve_p
set -u
REPO=${REPO:-/home/goodlad/dev/gen3ai}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
BANK=/home/goodlad/gen3ai_archive/x65_exit0_2026-11-05/bank
BSHA=d13fd55f10ba649ad8e728ce1585e4203c4200148d772919eb175d3273992b7a
PSTAR=$REPO/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip
PCTRL=$REPO/models/rb_x26_s2601/checkpoints/checkpoint_100000037_steps.zip
OUT=/home/goodlad/gen3ai_archive/x65_learnability_2026-11-06
: "${GEN3AI_GPU_LEASE_TOKEN_FILE:?set it to the GPU lease token file}"
cd "$REPO" || exit 2
export PYTHONPATH="$REPO/src"
mkdir -p "$OUT"
say() { echo "[$(date -Is)] X65L: $*" | tee -a "$OUT/driver.log"; }

knobs() {   # <arm>: the arm's knobs (README "The arms")
    case $1 in
        curve_h) echo "--fold heldout72_hpfix --lr 1e-4 --beta 0.1 --epochs 12 --evals-per-epoch 2" ;;
        ctrl_h)  echo "--fold heldout72_hpfix --lr 1e-4 --beta 0.1 --epochs 12 --evals-per-epoch 2 --target-model $PCTRL" ;;
        cap_h)   echo "--fold heldout72_hpfix --lr 3e-4 --beta 0 --epochs 24 --evals-per-epoch 1 --train-panels forum72" ;;
        curve_p) echo "--fold p72 --lr 1e-4 --beta 0.1 --epochs 12 --evals-per-epoch 2" ;;
        *) return 1 ;;
    esac
}

for arm in ${@:-curve_h ctrl_h cap_h curve_p}; do
    k=$(knobs "$arm") || { say "HOLD: unknown arm $arm"; exit 2; }
    finished=0
    for try in 1 2 3 4 5; do
        say "$arm try $try: start"
        # shellcheck disable=SC2086
        env -u CUDA_VISIBLE_DEVICES -u GEN3AI_TEST_ALLOW_GPU taskset -c 0-4,8-12 scripts/ops/gpu_lock.sh \
            "$PY" -m main.x65.student probe --bank "$BANK" --bank-sha256 "$BSHA" --name "$arm" $k --init "$PSTAR" \
            --out "$OUT/$arm" --device cuda --budget-min 50 >> "$OUT/$arm.log" 2>&1
        rc=$?
        say "$arm try $try: rc=$rc"
        [ "$rc" -eq 0 ] && { finished=1; break; }
        [ "$rc" -eq 3 ] || { say "HOLD: $arm exited rc=$rc (see $OUT/$arm.log)"; exit 1; }
    done
    [ "$finished" -eq 1 ] || { say "HOLD: $arm not finished after 5 chunks"; exit 1; }
done
say "all arms done"
