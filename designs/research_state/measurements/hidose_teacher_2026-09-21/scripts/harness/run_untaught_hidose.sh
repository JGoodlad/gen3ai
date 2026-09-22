#!/usr/bin/env bash
# THE COLLATERAL CELL -- the UNTAUGHT METER (off-slice) on the high-dose teacher.
#
# REUSED VERBATIM from split_lossoff_read_2026-09-20/scripts/harness/run_untaught.sh, which reused
# wcont_control_read_2026-09-20's, which reused fold1_cont_read_2026-09-19's: same tool, same
# registry OPPONENT name, the untaught-8 manifest in its canonical order (the tool's default --
# do NOT pass --teams here), same --seed 0 and concurrency 1. THE ONLY CHANGE IS THE REF LIST.
#
# BOTH refs in ONE invocation (CRN). The plateau parent's banked level is 60.19 pp (ledger
# 2026-09-20, `3459ecce`); the registered off-slice floor is 3.69 pp. Read this cell WHICHEVER WAY
# the admission cell goes -- ../../PREDICTION.md sec 3.2 and branches (c)/(d).
#
# CPU only, nice 15, 8 workers. From the MAIN checkout. Nothing is written under models/.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$D/../../out}"
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai

[ -f "$OUT/untaught_hidose.json" ] && { echo "SKIP untaught"; exit 0; }
echo "=== UNTAUGHT METER hidose  $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
nice -n 15 $P -m main.untaught_meter \
  hidose=models/ai_v13_18_teach5_offense_hidose/final_model.zip \
  plateau_parent=models/ai_v13_12_plateau/final_model.zip \
  --opponent untaught_meter_opponent \
  --workers 8 --seed 0 \
  --json "$OUT/untaught_hidose.json" --md "$OUT/untaught_hidose.md" 2>&1
echo "=== EXIT $? untaught  $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
