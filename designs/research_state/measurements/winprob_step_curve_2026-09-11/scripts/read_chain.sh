#!/usr/bin/env bash
# Pipelined within-run reads: each pair is read as soon as BOTH its cycles are complete.
# One read at a time — the box carries a peer session's pipeline as well.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
ROOT=/home/goodlad/.claude/jobs/9ab51de6/tmp/stepcurve
READS=$ROOT/reads
FLOOR=/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/reads/hp400_floor.json
mkdir -p "$READS"; cd /home/goodlad/dev/gen3ai

complete() {  # $1=step $2=draw  — cycle finished and certified complete
  local m="$ROOT/$1/$2/eval_traces/step_$1/eval_manifest.json"
  [ -f "$m" ] && grep -q '"complete": true' "$m" 2>/dev/null
}

for DRAW in draw1 draw2; do
  for STEP in 18660864 40935168 73121280; do
    OUT=$READS/${DRAW}_${STEP}_vs_10M
    [ -f "$OUT/critic_read.json" ] && { echo "=== SKIP $DRAW $STEP ==="; continue; }
    echo "=== WAIT for $STEP/$DRAW and 9969408/$DRAW  $(date -Is) ==="
    while ! { complete "$STEP" "$DRAW" && complete 9969408 "$DRAW"; }; do
      # if generation has finished and the cycle still is not there, stop waiting forever
      if grep -q "=== ALL GENERATION DONE" "$ROOT/gen_all.log" 2>/dev/null; then
        complete "$STEP" "$DRAW" && complete 9969408 "$DRAW" || {
          echo "=== GIVE UP on $DRAW $STEP — generation done, cycle not complete ==="; break 2; }
      fi
      sleep 60
    done
    echo "=== READ $DRAW $STEP vs 9969408  $(date -Is) ==="
    nice -n 15 $P -m main.ops.critic_read ai_v12_02_winprob_critic \
      --control ai_v12_02_winprob_critic \
      --step "$STEP" --control-step 9969408 \
      --arm-traces "$ROOT/$STEP/$DRAW" --control-traces "$ROOT/9969408/$DRAW" \
      --floor-json "$FLOOR" --out "$OUT" --nice 15 --ledger-line 2>&1
    echo "=== READ EXIT $? for $DRAW $STEP  $(date -Is) ==="
  done
done
echo "=== ALL WITHIN-RUN READS DONE $(date -Is) ==="
