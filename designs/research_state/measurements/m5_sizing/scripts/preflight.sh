#!/usr/bin/env bash
# REGISTRATION §5.1 pre-flight at N (not a read): one real FRESH launch to its first completed update,
# run dir in ~/gen3ai_archive/m5_sizing_preflight/ (never models/), this worktree at the arms' pin.
#   preflight.sh <N> <buckets|->
set -u
N=$1; BK=$2
WT=/home/goodlad/dev/gen3ai-wt/m5-sizing
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
OUT=/home/goodlad/gen3ai_archive/m5_sizing_preflight/n$N
mkdir -p "$(dirname $OUT)"; rm -rf "$OUT"
cd /home/goodlad/dev/gen3ai || exit 2          # cwd = main, as a pinned launch runs (data/ from main)
export PYTHONPATH=$WT/src
X=(--arch production --env-core rust --n-envs "$N" --steps 491520 --seed 1001 --checkpoint-every-steps 2000000
   --device cuda --log-level periodic --run-dir "$OUT")
[ "$N" != 48 ] && X+=(--n-steps $((98304 / N)))
[ "$BK" != - ] && X+=(--t2-buckets "$BK")
exec "$WT/scripts/ops/mem_cap.sh" --name "preflight_n$N" 56 "$WT/scripts/ops/gpu_lock.sh" timeout 1500 \
    nice -n 5 "$PY" "$WT/src/main/train_rl_agent.py" "${X[@]}"
