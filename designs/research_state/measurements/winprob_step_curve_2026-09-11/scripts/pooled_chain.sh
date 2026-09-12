#!/usr/bin/env bash
# THE STRENGTH-ROBUST ROW: value_pooled -> opponent-class AUC, per checkpoint per draw.
# Reuses the N-curve's OWN extractor + decoder in place (measurements/winprob_refit_ncurve_2026-09-10).
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
NC=/home/goodlad/dev/gen3ai/designs/research_state/measurements/winprob_refit_ncurve_2026-09-10
ROOT=/home/goodlad/.claude/jobs/9ab51de6/tmp/stepcurve
cd /home/goodlad/dev/gen3ai
declare -A LB=( [9969408]=10M [18660864]=20M [40935168]=40M [73121280]=73M )

complete() {
  local m="$ROOT/$1/$2/eval_traces/step_$1/eval_manifest.json"
  [ -f "$m" ] && grep -q '"complete": true' "$m" 2>/dev/null
}

ARGS=()
for STEP in 9969408 18660864 40935168 73121280; do
  for DRAW in draw1 draw2; do
    TAG="${LB[$STEP]}_$DRAW"
    D=$ROOT/pooled/$TAG
    if [ ! -f "$D/pooled.npy" ]; then
      echo "=== WAIT cycle $STEP/$DRAW  $(date -Is) ==="
      while ! complete "$STEP" "$DRAW"; do
        grep -q "=== ALL GENERATION DONE" "$ROOT/gen_all.log" 2>/dev/null && \
          { complete "$STEP" "$DRAW" || { echo "=== GIVE UP $TAG ==="; continue 2; }; }
        sleep 60
      done
      echo "=== EXTRACT $TAG  $(date -Is) ==="
      mkdir -p "$D"
      nice -n 15 $P $NC/extract.py \
        --tree c=$ROOT/$STEP/$DRAW/eval_traces/step_$STEP \
        --snapshot $ROOT/$STEP/$DRAW/eval_traces/step_$STEP/snapshot.zip \
        --out-dir "$D" --threads 2 --batch 256 2>&1 | tail -3
      echo "=== EXTRACT EXIT $? for $TAG ==="
    else
      echo "=== SKIP extract $TAG (exists) ==="
    fi
    [ -f "$D/pooled.npy" ] && ARGS+=(--dir "$TAG=$D")
  done
done

echo "=== FRAME CHECK over ${#ARGS[@]} dirs  $(date -Is) ==="
nice -n 15 $P $NC/frame_check.py "${ARGS[@]}" --out $ROOT/pooled/frame_check.json --cap 4000 2>&1
echo "=== POOLED CHAIN DONE $(date -Is) ==="
