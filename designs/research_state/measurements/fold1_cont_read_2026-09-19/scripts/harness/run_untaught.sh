#!/usr/bin/env bash
# ROW 1 — THE UNTAUGHT METER (the off-slice / collateral read) across the WHOLE FOLD PATH.
#
# REUSED VERBATIM from fold1_read_2026-09-19/scripts/harness/run_untaught.sh: same tool, same
# registry OPPONENT name, same untaught-8 manifest in its canonical order, same --seed 0 and
# concurrency 1 as arm S's 2026-09-14 read, the pair read's 2026-09-16 one, the floor read's
# 2026-09-18 one and the +6M fold read's. The ONLY change is the ref list.
#
# 🚨 ALL EIGHT REFS IN ONE INVOCATION, so every ref sees the identical 8 teams and the identical
# 200 games each (CRN): the six-point trajectory +1M/+3M/+6M/+7.59M/+9.09M/+12.09M is PAIRED on ONE
# index set, and so is every contrast against the parent and the seed arm.
#
# FIVE REPRODUCTION CHECKS ride in it: armW must return 46.19 pp (739/1600), armWb 49.88 (798/1600),
# and the fold's three depths 47.62 / 51.56 / 51.31 — to the last win, on the PER-TEAM rows.
#
# CPU only, nice, from the MAIN checkout. Nothing is written under models/.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/9ab51de6/tmp/fold1c_read/untaught
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai

CFG="${1:-registry}"
ARGS=()
[ "$CFG" = "auto" ] && ARGS=(--config auto)
[ -f "$OUT/untaught_${CFG}.json" ] && { echo "SKIP $CFG"; exit 0; }
echo "=== UNTAUGHT METER config=$CFG  $(date -Is) ==="
nice -n 15 $P -m main.untaught_meter \
  fold_p1M=models/ai_v13_07_fold1/checkpoints/checkpoint_76005984_steps.zip \
  fold_p3M=models/ai_v13_07_fold1/checkpoints/checkpoint_78006048_steps.zip \
  fold_p6M=models/ai_v13_07_fold1/final_model.zip \
  cont_p7_5M=models/ai_v13_08_fold1_cont/checkpoints/checkpoint_82600848_steps.zip \
  cont_p9M=models/ai_v13_08_fold1_cont/checkpoints/checkpoint_84100896_steps.zip \
  cont_p12M=models/ai_v13_08_fold1_cont/final_model.zip \
  armW=models/ai_v13_02_flywheel_winprob/final_model.zip \
  armWb=models/ai_v13_04_flywheel_winprob_b/final_model.zip \
  --opponent untaught_meter_opponent "${ARGS[@]}" \
  --workers 8 --seed 0 \
  --json "$OUT/untaught_${CFG}.json" --md "$OUT/untaught_${CFG}.md" 2>&1
echo "=== EXIT $? config=$CFG  $(date -Is) ==="
