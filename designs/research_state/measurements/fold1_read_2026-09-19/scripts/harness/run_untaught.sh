#!/usr/bin/env bash
# ROW 1 — THE UNTAUGHT METER (the off-slice / collateral read) on the ERA-1 FOLD.
#
# Same tool, same registry OPPONENT name, same untaught-8 manifest in its canonical order, same
# --seed 0 and concurrency 1 as arm S's 2026-09-14 read, the pair read's 2026-09-16 one and the
# floor read's 2026-09-18 one. ALL FIVE REFS IN ONE INVOCATION, so every ref sees the identical
# 8 teams and the identical 200 games each (CRN) and every ref-vs-ref contrast is PAIRED.
#
# arm W appears here AND in the floor read's file, so its level is this run's reproduction check:
# a meter that is deterministic at seed 0 / concurrency 1 must return 46.19 pp / 739 of 1600 again.
#
# CPU only, nice, from the MAIN checkout. Nothing is written under models/.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/9ab51de6/tmp/fold1_read/untaught
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai

CFG="${1:-registry}"
ARGS=()
[ "$CFG" = "auto" ] && ARGS=(--config auto)
echo "=== UNTAUGHT METER config=$CFG  $(date -Is) ==="
nice -n 15 $P -m main.untaught_meter \
  fold_p1M=models/ai_v13_07_fold1/checkpoints/checkpoint_76005984_steps.zip \
  fold_p3M=models/ai_v13_07_fold1/checkpoints/checkpoint_78006048_steps.zip \
  fold_p6M=models/ai_v13_07_fold1/final_model.zip \
  armW=models/ai_v13_02_flywheel_winprob/final_model.zip \
  armWb=models/ai_v13_04_flywheel_winprob_b/final_model.zip \
  --opponent untaught_meter_opponent "${ARGS[@]}" \
  --workers 8 --seed 0 \
  --json "$OUT/untaught_${CFG}.json" --md "$OUT/untaught_${CFG}.md" 2>&1
echo "=== EXIT $? config=$CFG  $(date -Is) ==="
