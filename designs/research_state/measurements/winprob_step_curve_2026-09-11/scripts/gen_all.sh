#!/usr/bin/env bash
# THE STEP CURVE — 8 offline eval cycles: 4 checkpoints x 2 independent draws.
# Run from the MAIN checkout (land.sh removes worktrees out from under a running generation).
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
ROOT=/home/goodlad/.claude/jobs/9ab51de6/tmp/stepcurve
cd /home/goodlad/dev/gen3ai

# draw1 first for every checkpoint, so a failure still leaves a complete single-draw curve.
for DRAW_SEED in "draw1 20260909" "draw2 20260910"; do
  set -- $DRAW_SEED; DRAW=$1; SEED=$2
  for STEP in 9969408 18660864 40935168 73121280; do
    OUT=$ROOT/$STEP/$DRAW
    if [ -f "$OUT/eval_traces/step_$STEP/eval_manifest.json" ] && \
       grep -q '"complete": true' "$OUT/eval_traces/step_$STEP/eval_manifest.json" 2>/dev/null; then
      echo "=== SKIP $STEP $DRAW (already complete) ==="; continue
    fi
    echo "=== GEN $STEP $DRAW seed $SEED  $(date -Is) ==="
    nice -n 15 $P -m main.ops.eval_trace_gen "ai_v12_02_winprob_critic@$STEP" \
      --games 400 --sentinels 3 --include-current-snapshot --no-eval-sentinel-greedy \
      --out "$OUT" --workers 4 --concurrency 1 --nice 15 --seed "$SEED" \
      --shard-games 25 --force 2>&1
    echo "=== EXIT $? for $STEP $DRAW  $(date -Is) ==="
  done
done
echo "=== ALL GENERATION DONE $(date -Is) ==="
