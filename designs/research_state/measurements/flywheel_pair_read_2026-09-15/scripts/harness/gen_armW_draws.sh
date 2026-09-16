#!/usr/bin/env bash
# ARM W run-end critic draws: two independent offline 800-game full-capture cycles.
# Identical harness, seeds and specs to arm S's (gen_armS_draws.sh, 2026-09-14).
# Run from the MAIN checkout. Nothing is written under models/.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
ROOT=/home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/traces
cd /home/goodlad/dev/gen3ai

for SPEC in "draw1 20260910" "draw2 20260911"; do
  set -- $SPEC; DRAW=$1; SEED=$2
  OUT=$ROOT/$DRAW
  if [ -f "$OUT/eval_traces/step_74000016/eval_manifest.json" ] && \
     grep -q '"complete": true' "$OUT/eval_traces/step_74000016/eval_manifest.json" 2>/dev/null; then
    echo "=== SKIP $DRAW (already complete) ==="; continue
  fi
  echo "=== GEN armW $DRAW seed $SEED  $(date -Is) ==="
  nice -n 15 $P -m main.ops.eval_trace_gen ai_v13_02_flywheel_winprob \
    --games 800 --sentinels 3 \
    --out "$OUT" --workers 4 --concurrency 1 --nice 15 --seed "$SEED" \
    --shard-games 25 --force 2>&1
  echo "=== EXIT $? for $DRAW  $(date -Is) ==="
done
echo "=== ARM W DRAWS DONE $(date -Is) ==="
