#!/usr/bin/env bash
# The FRESH three-branch CRN fork datasets — one per POLICY, each from that policy's OWN offline
# full-capture eval tree (seed 20260911 draw), sentinel opponents only.
#
# 🚨 The builder, the contested selector and the CRN discipline are the 2026-09-14 record's,
# REUSED VERBATIM by import — `forks.py` is invoked from
# `designs/research_state/measurements/paired_refit_discrimination_2026-09-14/`, not copied, so it
# cannot drift from the instrument the 0.5872 baseline is measured on.
#
#   usage: run_forks.sh <arm|ctrl> [out_root] [shards] [max_forks_per_shard]
set -uo pipefail
MAIN=/home/goodlad/dev/gen3ai
BUILDER=$MAIN/designs/research_state/measurements/paired_refit_discrimination_2026-09-14/forks.py
WHICH=${1:?usage: run_forks.sh <arm|ctrl> [out] [shards] [max_forks]}
OUT=${2:-/home/goodlad/.claude/jobs/9ab51de6/tmp/fork_read}/$WHICH
W=${3:-6}
MAXF=${4:-840}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$MAIN/src
export POKESIM_SIM_BRIDGE_BIN=$MAIN/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$MAIN" || exit 1

H=/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b
if [ "$WHICH" = arm ]; then
  RUN=ai_v13_03_fork;            SNAP=models/ai_v13_03_fork/snapshots/snapshot_000010000032.zip
else
  RUN=ai_v12_11_ladder_ctrl10M;  SNAP=models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip
fi
TREE=$H/$RUN/eval_traces/step_10000032
[ -d "$TREE" ] || { echo "REFUSED: no tree at $TREE"; exit 1; }
mkdir -p "$OUT/logs"
echo "=== FORKS[$WHICH] $RUN start $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
for i in $(seq 0 $((W-1))); do
  nice -n 15 "$PY" "$BUILDER" --tree "$TREE" --snapshot "$SNAP" --out "$OUT" \
     --opponents sentinel_2,sentinel_1,sentinel_0 --shard "$i/$W" \
     --forks-per-battle 4 --gap-quantile 0.40 --min-legal 3 --min-turn 2 --max-turn 40 \
     --max-forks "$MAXF" --determinism-check-every 50 --impl rust --threads 1 --seed 20260916 \
     > "$OUT/logs/shard_$i.log" 2>&1 &
  echo "  shard $i pid $!"
done
wait
echo "=== FORKS[$WHICH] done $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
