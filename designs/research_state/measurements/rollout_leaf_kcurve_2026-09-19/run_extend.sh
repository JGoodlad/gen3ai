#!/usr/bin/env bash
# JOB 1's dice — 16 FRESH re-rolls per branch (indices 8..23) over the six BANKED fork shards,
# one worker each, the same six-way split the banked K'=8 pass used.
# RESUMABLE: re-running picks up forks not yet complete; a stop leaves fewer forks at FULL K.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L=/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit
B=/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_control/reroll
O=${O:-/home/goodlad/.claude/jobs/9ab51de6/tmp/kcurve/rerollx}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
KS="${KS:-8}"; KE="${KE:-24}"; VERIFY="${VERIFY:-4}"; SHARDS="${SHARDS:-0 1 2 3 4 5}"
export PYTHONPATH="${PYTHONPATH:-}:/home/goodlad/dev/gen3ai/src"
export CUDA_VISIBLE_DEVICES=""
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
mkdir -p "$O/logs"
SNAP="$L/src_run/ai_v13_02_flywheel_winprob/eval_traces/step_75005952/snapshot.zip"
for i in $SHARDS; do
  nice -n 15 "$PY" "$HERE/reroll_extend.py" \
     --forks "$L/forks_eval" --banked "$B" --snapshot "$SNAP" --out "$O" --shard "$i/6" \
     --k-start "$KS" --k-end "$KE" --seed 20260919 --verify "$VERIFY" --impl rust \
     > "$O/logs/shard_$i.log" 2>&1 &
done
wait
echo "[run_extend] all shards returned"
