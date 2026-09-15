#!/usr/bin/env bash
# PART B step 1 — the PAIRED SIBLING fork dataset. 6 disjoint shards over the frozen
# ai_v12_11_ladder_ctrl10M@10000032 offline full-capture eval tree, sentinel opponents only
# (reloadable EXACTLY from the plan's snapshot paths, and RECORDED GREEDY, so the CRN pairing is
# exact in both the dice and the policy draws).
set -uo pipefail
ROOT=/home/goodlad/dev/gen3ai
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/paired_refit/forks}
W=${2:-6}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$ROOT/src
export POKESIM_SIM_BRIDGE_BIN=$ROOT/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
cd "$ROOT" || exit 1
mkdir -p "$OUT/logs"
TREE=/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032
SNAP=models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip
echo "=== FORKS start $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
for i in $(seq 0 $((W-1))); do
  nice -n 15 "$PY" "$HERE/forks.py" --tree "$TREE" --snapshot "$SNAP" --out "$OUT" \
     --opponents sentinel_2,sentinel_1,sentinel_0 --shard "$i/$W" \
     --forks-per-battle 4 --gap-quantile 0.40 --min-legal 3 --min-turn 2 --max-turn 40 \
     --determinism-check-every 50 --impl rust --threads 1 \
     > "$OUT/logs/shard_$i.log" 2>&1 &
  echo "  shard $i pid $!"
done
wait
echo "=== FORKS done $(date -u +%FT%TZ)"
