#!/usr/bin/env bash
# ROW 1 -- THE UNTAUGHT METER (the off-slice / collateral read) on the SPLIT arm.
#
# REUSED VERBATIM from wcont_control_read_2026-09-20/scripts/harness/run_untaught.sh, which reused
# it verbatim from fold1_cont_read_2026-09-19's, which reused fold1_read_2026-09-19's: same tool,
# same registry OPPONENT name, same untaught-8 manifest in its canonical order, same --seed 0 and
# concurrency 1 as arm S's 2026-09-14 read, the pair read's 2026-09-16 one, the floor read's
# 2026-09-18 one, the fold's +6M read, the convergence read and the control read.
# The ONLY change is the ref list.
#
# ALL FOUR MEASURED REFS IN ONE INVOCATION, so every ref sees the identical 8 teams and the
# identical 200 games each (CRN). The CONTROL path, the FOLD path and W_b are DECLARED IMPORTS --
# PREDICTION.md sec 1.2 -- warranted by the armW reproduction check that rides in this file.
#
# THE REPRODUCTION CHECK: armW must return 46.19 pp (739/1600) and all EIGHT per-team rows
# identical to the banked ones. A failure VOIDS every import; the job STOPS and reports.
#
# CPU only, nice 15, 8 workers (the GPU carries ai_v13_12_plateau block 1 and has first call).
# From the MAIN checkout. Nothing is written under models/.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/split_read_20260920/tmp/untaught
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai

CFG="${1:-registry}"
[ -f "$OUT/untaught_${CFG}.json" ] && { echo "SKIP $CFG"; exit 0; }
echo "=== UNTAUGHT METER config=$CFG  $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
nice -n 15 $P -m main.untaught_meter \
  split_p3M=models/ai_v13_11_split_lossoff/checkpoints/checkpoint_78006048_steps.zip \
  split_p6M=models/ai_v13_11_split_lossoff/checkpoints/checkpoint_81404208_steps.zip \
  split_p12M=models/ai_v13_11_split_lossoff/final_model.zip \
  armW=models/ai_v13_02_flywheel_winprob/final_model.zip \
  --opponent untaught_meter_opponent \
  --workers 8 --seed 0 \
  --json "$OUT/untaught_${CFG}.json" --md "$OUT/untaught_${CFG}.md" 2>&1
echo "=== EXIT $? config=$CFG  $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
