#!/usr/bin/env bash
# ROW 2 -- PER-SLICE PILOTING, the matched-extraction row.
#
# The untaught meter's engine pointed at the TWO TAUGHT TEAMS instead of the untaught 8: every ref
# pilots the SAME pinned team against the SAME fixed opponent (`untaught_meter_opponent`
# = ai_v9_29_rev1_0823@24,000,000) drawing the SAME 800-long pool-team sequence under the SAME
# dice, so a ref-vs-ref difference is PAIRED on the same games and is a PILOTING read, not a
# head-to-head.
#
# WHY THIS OPPONENT AND NOT ARM W's POOL SENTINELS: a sentinel is the trainee's OWN snapshot, so
# arm W's and W_b's sentinels are different checkpoints (the floor read's hazard F-G) and the seed
# floor would then be measured against different opponents from the treatment contrast.
#
# SHARDING. A cell is a pure function of (ref, team index, battle index), so splitting the REFS
# across processes cannot move a number -- the same property that licenses --workers. Three
# processes x 2 workers keeps the box at six battle workers beside the live GPU arm.
#   A: the fold at +6M and the PARENT          (the treatment contrast)
#   B: the fold at +3M and W_b                 (the depth row and the SEED FLOOR on this cell)
#   C: both TEACHERS                           (the teacher's ceiling on its own team)
#
# CPU only, nice 15, from the MAIN checkout. Nothing is written under models/.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/9ab51de6/tmp/fold1_read/slice
MAN=/home/goodlad/dev/gen3ai-wt/fold1_read_20260919/designs/research_state/measurements/fold1_read_2026-09-19/scripts/harness/taught_slice_teams.json
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai

SHARD="$1"
case "$SHARD" in
  A) REFS=(fold_p6M=models/ai_v13_07_fold1/final_model.zip
           armW=models/ai_v13_02_flywheel_winprob/final_model.zip) ;;
  B) REFS=(fold_p3M=models/ai_v13_07_fold1/checkpoints/checkpoint_78006048_steps.zip
           armWb=models/ai_v13_04_flywheel_winprob_b/final_model.zip) ;;
  C) REFS=(t1_big5=models/ai_v13_05_exploit_big5starmie/final_model.zip
           t2_ddtar=models/ai_v13_06_exploit_ddtar_spikes/final_model.zip) ;;
  *) echo "usage: run_slice.sh A|B|C"; exit 2 ;;
esac
[ -f "$OUT/slice_${SHARD}.json" ] && { echo "SKIP $SHARD"; exit 0; }
echo "=== SLICE shard $SHARD  $(date -Is) ==="
nice -n 15 $P -m main.untaught_meter "${REFS[@]}" \
  --teams "$MAN" --opponent untaught_meter_opponent \
  --games-per-team 800 --workers 2 --seed 0 \
  --json "$OUT/slice_${SHARD}.json" --md "$OUT/slice_${SHARD}.md" 2>&1
echo "=== EXIT $? shard $SHARD  $(date -Is) ==="
