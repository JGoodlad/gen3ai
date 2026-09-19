#!/usr/bin/env bash
# ROW 2 -- PER-SLICE PILOTING, the matched-extraction row, REUSED VERBATIM from
# fold1_read_2026-09-19/scripts/harness/run_slice.sh: same engine, same 2-team taught manifest in
# TEACHER ORDER (the order is the seed offset -- re-ordering it changes every game), same fixed
# opponent (`untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000), same 800 games/cell, same
# --seed 0, same 3 shards x 2 workers. The ONLY change is the ref list.
#
# WHY THIS OPPONENT AND NOT ARM W's POOL SENTINELS: a sentinel is the trainee's OWN snapshot, so
# arm W's and W_b's are different checkpoints (the floor read's hazard F-G) and the seed floor
# would then be measured against different opponents from the treatment contrast.
#
# SHARDING. A cell is a pure function of (ref, team index, battle index), so splitting the REFS
# across processes cannot move a number -- the same property that licenses --workers.
#   A: the continuation at +7.59M (the STOP POINT) and the PARENT arm W (the reproduction check)
#   B: the continuation at +12.09M (the ENDPOINT) and at +9.09M (the trajectory point)
#   C: the fold at +1M (the trajectory's first point)
# fold_p3M, fold_p6M and W_b are DECLARED IMPORTS from the +6M read -- see PREDICTION.md sec 3.2.
#
# CPU only, nice 15, from the MAIN checkout. Nothing is written under models/.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/9ab51de6/tmp/fold1c_read/slice
MAN=/home/goodlad/dev/gen3ai-wt/fold1cont_read/designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/harness/taught_slice_teams.json
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai

SHARD="$1"
case "$SHARD" in
  A) REFS=(cont_p7_5M=models/ai_v13_08_fold1_cont/checkpoints/checkpoint_82600848_steps.zip
           armW=models/ai_v13_02_flywheel_winprob/final_model.zip) ;;
  B) REFS=(cont_p12M=models/ai_v13_08_fold1_cont/final_model.zip
           cont_p9M=models/ai_v13_08_fold1_cont/checkpoints/checkpoint_84100896_steps.zip) ;;
  C) REFS=(fold_p1M=models/ai_v13_07_fold1/checkpoints/checkpoint_76005984_steps.zip) ;;
  *) echo "usage: run_slice.sh A|B|C"; exit 2 ;;
esac
[ -f "$OUT/slice_${SHARD}.json" ] && { echo "SKIP $SHARD"; exit 0; }
echo "=== SLICE shard $SHARD  $(date -Is) ==="
nice -n 15 $P -m main.untaught_meter "${REFS[@]}" \
  --teams "$MAN" --opponent untaught_meter_opponent \
  --games-per-team 800 --workers 2 --seed 0 \
  --json "$OUT/slice_${SHARD}.json" --md "$OUT/slice_${SHARD}.md" 2>&1
echo "=== EXIT $? shard $SHARD  $(date -Is) ==="
