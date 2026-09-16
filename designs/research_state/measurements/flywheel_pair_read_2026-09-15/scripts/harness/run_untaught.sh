#!/usr/bin/env bash
# THE UNTAUGHT METER at run end — arm W against arm S (the pair), under BOTH config resolutions
# (registration sec 8.5: a level that only exists under one resolution is not a level).
# Same tool, same registry OPPONENT name, same seed and concurrency as arm S's 2026-09-14 read.
# Offline, CPU only. Nothing is written under models/.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/untaught
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai

for CFG in registry auto; do
  ARGS=()
  [ "$CFG" = "auto" ] && ARGS=(--config auto)
  if [ -f "$OUT/untaught_${CFG}.json" ]; then echo "SKIP $CFG"; continue; fi
  echo "=== UNTAUGHT METER config=$CFG  $(date -Is) ==="
  nice -n 15 $P -m main.untaught_meter \
    armW=models/ai_v13_02_flywheel_winprob/final_model.zip \
    armS=models/ai_v13_01_flywheel_shaped/final_model.zip \
    --opponent untaught_meter_opponent "${ARGS[@]}" \
    --workers 4 --seed 0 \
    --json "$OUT/untaught_${CFG}.json" --md "$OUT/untaught_${CFG}.md" 2>&1
  echo "=== EXIT $? config=$CFG  $(date -Is) ==="
done
echo "=== UNTAUGHT DONE $(date -Is) ==="
