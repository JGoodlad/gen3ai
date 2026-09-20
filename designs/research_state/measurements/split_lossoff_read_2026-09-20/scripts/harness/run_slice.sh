#!/usr/bin/env bash
# ROW 2 -- PER-SLICE PILOTING on the SPLIT arm: the arm that got the teacher's TEAMS and OPPONENTS
# but not the teacher's ACTIONS.
#
# REUSED VERBATIM from wcont_control_read_2026-09-20/scripts/harness/run_slice.sh: same engine,
# same 2-team taught manifest in TEACHER ORDER (the order is the seed offset -- re-ordering it
# changes every game), same fixed opponent (`untaught_meter_opponent` = ai_v9_29_rev1_0823@24M),
# same 800 games/cell, same --seed 0, same shard x 2-worker layout. The ONLY change is the refs.
#
# WHY THIS OPPONENT AND NOT A POOL SENTINEL: a sentinel is the trainee's OWN snapshot, so the arms'
# sentinels differ (the floor read's hazard F-G) and the seed floor would then be measured against
# different opponents from the treatment contrast.
#
# SHARDING. A cell is a pure function of (ref, team index, battle index), so splitting the REFS
# across processes cannot move a number -- the same property that licenses --workers.
#   A: the split at +12M (the ENDPOINT, the BRANCH-CARRYING point)
#   B: the split at +6M (a trajectory point; it carries the 260,928-step offset, hazard S-D)
#   C: the split at +3M (the registered ROBUSTNESS depth -- the same step as both other paths)
#   D: the PARENT arm W -- THE REPRODUCTION CHECK that warrants every import
# Every wcont_*, fold_*, cont_p12M, armWb, t1 and t2 cell is a DECLARED IMPORT -- PREDICTION sec 1.2.
#
# 4 shards x 2 workers = 8 processes, the registered cap. CPU only, nice 15, MAIN checkout.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/split_read_20260920/tmp/slice
MAN=/home/goodlad/dev/gen3ai-wt/split_read/designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/harness/taught_slice_teams.json
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai

SHARD="$1"
case "$SHARD" in
  A) REFS=(split_p12M=models/ai_v13_11_split_lossoff/final_model.zip) ;;
  B) REFS=(split_p6M=models/ai_v13_11_split_lossoff/checkpoints/checkpoint_81404208_steps.zip) ;;
  C) REFS=(split_p3M=models/ai_v13_11_split_lossoff/checkpoints/checkpoint_78006048_steps.zip) ;;
  D) REFS=(armW=models/ai_v13_02_flywheel_winprob/final_model.zip) ;;
  *) echo "usage: run_slice.sh A|B|C|D"; exit 2 ;;
esac
[ -f "$OUT/slice_${SHARD}.json" ] && { echo "SKIP $SHARD"; exit 0; }
echo "=== SLICE shard $SHARD  $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
nice -n 15 $P -m main.untaught_meter "${REFS[@]}" \
  --teams "$MAN" --opponent untaught_meter_opponent \
  --games-per-team 800 --workers 2 --seed 0 \
  --json "$OUT/slice_${SHARD}.json" --md "$OUT/slice_${SHARD}.md" 2>&1
echo "=== EXIT $? shard $SHARD  $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
