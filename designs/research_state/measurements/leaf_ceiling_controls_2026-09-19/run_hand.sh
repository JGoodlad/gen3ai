#!/usr/bin/env bash
# CONTROL 1 — the omniscient board capture over the six BANKED fork shards, one worker each.
# Run from the MAIN checkout (the rollouts must ride main's code and main's rust binary).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L=/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit
O=/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_control/hand
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH="${PYTHONPATH:-}:/home/goodlad/dev/gen3ai/src"
export CUDA_VISIBLE_DEVICES=""                    # the GPU carries a training arm
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
mkdir -p "$O/logs"
SNAP="$L/src_run/ai_v13_02_flywheel_winprob/eval_traces/step_75005952/snapshot.zip"
for i in 0 1 2 3 4 5; do
  nice -n 15 "$PY" "$HERE/hand_capture.py" \
     --forks "$L/forks_eval" --snapshot "$SNAP" --out "$O" --shard "$i/6" \
     --to-terminal 8 --impl rust > "$O/logs/shard_$i.log" 2>&1 &
done
wait
echo "[run_hand] all six shards returned"
