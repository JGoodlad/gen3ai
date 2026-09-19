#!/usr/bin/env bash
# THE TWO FORK DATASETS.
#
#   branch  the ALPHAGO set — `branch_forks.py` (this directory): a uniformly-random legal move at
#           a uniformly-random depth, plus the policy's own top-1 sibling under COMMON RANDOM
#           NUMBERS. This is the TRAINING data and nothing is read from it.
#   eval    the CONTESTED set — `paired_refit_discrimination_2026-09-14/forks.py`, invoked BY PATH
#           and UNMODIFIED, at byte-identical flags to `fork_arm_read_2026-09-16/run_forks.sh` and
#           `exploiter_discrimination_2026-09-18/run_forks.sh` except the seed. This is the
#           instrument the 0.568–0.578 reference levels are on; a copy of it would be a different
#           instrument.
#
#   usage: run_forks.sh branch|eval [tmp] [shards] [max_forks_per_shard]
set -uo pipefail
MAIN=/home/goodlad/dev/gen3ai
HERE=$MAIN/designs/research_state/measurements/offline_leaf_fit_2026-09-18
BUILDER14=$MAIN/designs/research_state/measurements/paired_refit_discrimination_2026-09-14/forks.py
WHICH=${1:?branch|eval}
T=${2:-/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit}
W=${3:-6}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
SNAP=$T/src_run/ai_v13_02_flywheel_winprob/eval_traces/step_75005952/snapshot.zip
export PYTHONPATH=$MAIN/src
export POKESIM_SIM_BRIDGE_BIN=$MAIN/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$MAIN" || exit 1

if [ "$WHICH" = branch ]; then
  TREE=$T/tree_train/eval_traces/step_75005952; OUT=$T/forks_branch; MAXF=${4:-3000}
else
  TREE=$T/tree_eval/eval_traces/step_75005952;  OUT=$T/forks_eval;   MAXF=${4:-840}
fi
[ -d "$TREE" ] || { echo "REFUSED: no tree at $TREE"; exit 1; }
mkdir -p "$OUT/logs"
echo "=== FORKS[$WHICH] start $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
for i in $(seq 0 $((W-1))); do
  if [ "$WHICH" = branch ]; then
    nice -n 15 "$PY" "$HERE/branch_forks.py" --tree "$TREE" --snapshot "$SNAP" --out "$OUT" \
       --opponents sentinel_0 --shard "$i/$W" \
       --forks-per-battle 3 --min-legal 2 --min-turn 1 --max-turn 60 \
       --max-forks "$MAXF" --determinism-check-every 50 --impl rust --threads 1 \
       --seed 20260918 > "$OUT/logs/shard_$i.log" 2>&1 &
  else
    nice -n 15 "$PY" "$BUILDER14" --tree "$TREE" --snapshot "$SNAP" --out "$OUT" \
       --opponents sentinel_0 --shard "$i/$W" \
       --forks-per-battle 4 --gap-quantile 0.40 --min-legal 3 --min-turn 2 --max-turn 40 \
       --max-forks "$MAXF" --determinism-check-every 50 --impl rust --threads 1 \
       --seed 20260919 > "$OUT/logs/shard_$i.log" 2>&1 &
  fi
  echo "  shard $i pid $!"
done
wait
echo "=== FORKS[$WHICH] done $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
