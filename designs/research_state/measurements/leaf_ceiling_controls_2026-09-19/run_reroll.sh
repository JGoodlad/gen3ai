#!/usr/bin/env bash
# CONTROL 2 — K-rollout averaged labels over the six BANKED fork shards, one worker each.
# FORK-OUTER / K-INNER and RESUMABLE: re-running picks up forks not yet complete, and a stop
# leaves fewer forks at FULL K rather than all forks at a ragged K.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L=/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit
O=/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_control/reroll
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
K="${K:-8}"; SAMPLE="${SAMPLE:-1000}"
export PYTHONPATH="${PYTHONPATH:-}:/home/goodlad/dev/gen3ai/src"
export CUDA_VISIBLE_DEVICES=""
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
mkdir -p "$O/logs"
SNAP="$L/src_run/ai_v13_02_flywheel_winprob/eval_traces/step_75005952/snapshot.zip"
for i in 0 1 2 3 4 5; do
  nice -n 15 "$PY" "$HERE/reroll_labels.py" \
     --forks "$L/forks_eval" --snapshot "$SNAP" --out "$O" --shard "$i/6" \
     --k "$K" --sample "$SAMPLE" --seed 20260919 --impl rust \
     > "$O/logs/shard_$i.log" 2>&1 &
done
wait
echo "[run_reroll] all six shards returned"
