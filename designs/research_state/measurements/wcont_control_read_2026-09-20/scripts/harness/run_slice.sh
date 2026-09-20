#!/usr/bin/env bash
# ROW 2 -- PER-SLICE PILOTING on the CONTINUATION CONTROL: the TEAM-DIFFERENTIAL continuation read.
#
# REUSED VERBATIM from fold1_cont_read_2026-09-19/scripts/harness/run_slice.sh: same engine, same
# 2-team taught manifest in TEACHER ORDER (the order is the seed offset -- re-ordering it changes
# every game), same fixed opponent (`untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000),
# same 800 games/cell, same --seed 0, same shard x 2-worker layout. The ONLY change is the ref list.
#
# WHY THIS OPPONENT AND NOT ARM W's POOL SENTINELS: a sentinel is the trainee's OWN snapshot, so
# arm W's and W_b's are different checkpoints (the floor read's hazard F-G) and the seed floor
# would then be measured against different opponents from the treatment contrast.
#
# SHARDING. A cell is a pure function of (ref, team index, battle index), so splitting the REFS
# across processes cannot move a number -- the same property that licenses --workers.
#   A: the control at +12M (the ENDPOINT, the bar-carrying point)
#   B: the control at +6M (the bar-carrying point matched to the fold's +6M read)
#   C: the control at +3M (a trajectory point, registered as such)
#   D: the PARENT arm W -- THE REPRODUCTION CHECK that warrants every import
# fold_p3M, fold_p6M, cont_p12M, armWb, t1 and t2 are DECLARED IMPORTS -- PREDICTION.md sec 1.2.
#
# CPU only, nice 15, from the MAIN checkout. Nothing is written under models/.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/9ab51de6/tmp/wcont_read/slice
MAN=/home/goodlad/dev/gen3ai-wt/wcont_control_read/designs/research_state/measurements/wcont_control_read_2026-09-20/scripts/harness/taught_slice_teams.json
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai

SHARD="$1"
case "$SHARD" in
  A) REFS=(wcont_p12M=models/ai_v13_09_wcont/final_model.zip) ;;
  B) REFS=(wcont_p6M=models/ai_v13_09_wcont/checkpoints/checkpoint_81143280_steps.zip) ;;
  C) REFS=(wcont_p3M=models/ai_v13_09_wcont/checkpoints/checkpoint_78006048_steps.zip) ;;
  D) REFS=(armW=models/ai_v13_02_flywheel_winprob/final_model.zip) ;;
  *) echo "usage: run_slice.sh A|B|C|D"; exit 2 ;;
esac
[ -f "$OUT/slice_${SHARD}.json" ] && { echo "SKIP $SHARD"; exit 0; }
echo "=== SLICE shard $SHARD  $(date -Is) ==="
nice -n 15 $P -m main.untaught_meter "${REFS[@]}" \
  --teams "$MAN" --opponent untaught_meter_opponent \
  --games-per-team 800 --workers 2 --seed 0 \
  --json "$OUT/slice_${SHARD}.json" --md "$OUT/slice_${SHARD}.md" 2>&1
echo "=== EXIT $? shard $SHARD  $(date -Is) ==="
