#!/usr/bin/env bash
# The targeted lock / trap game set against Showdown, on THIS checkout's self-check build (the audit's B-C1 re-run).
#   run_targeted_set.sh <outdir> <seed> <n>
# Generates <n> targeted games (`targeted_selfcheck.py`: both sides external, uniform over their masks), BULK-replays them through
# the Node bridge (`showdown_replay_selfcheck.py`), then re-checks every bulk-DIFFER game PACED (`showdown_replay_paced.py`, one
# command at a time): a bulk replay of a script carrying refused hidden-trap switches trips Node's reject-streak guard, which is
# a replay-tool artifact and not an engine divergence. Run it from the worktree root; the build must be current
# (`cd src/rust_env && CARGO_TARGET_DIR=$PWD/target cargo build --profile selfcheck --features emission-selfcheck --lib --bin rust_env_proc`).
set -u
OUT=$1; SEED=$2; N=$3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=""
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
LOW="nice -n 19 taskset -c 5-7,13-15"
mkdir -p "$OUT"
$LOW $PY "$HERE/targeted_selfcheck.py" "$N" "$SEED" "$OUT/games_s${SEED}_${N}.json" > "$OUT/gen_s${SEED}.log" 2>&1
$LOW $PY "$HERE/showdown_replay_selfcheck.py" "$OUT/games_s${SEED}_${N}.json" > "$OUT/bulk_s${SEED}.out" 2>&1
echo "bulk: $(head -1 "$OUT/bulk_s${SEED}.out")" > "$OUT/summary_s${SEED}.txt"
for g in $(grep -o '^game [0-9]*' "$OUT/bulk_s${SEED}.out" | awk '{print $2}'); do
  $LOW $PY "$HERE/showdown_replay_paced.py" "$OUT/games_s${SEED}_${N}.json" "$g" $((g+1)) > "$OUT/paced_s${SEED}_g${g}.out" 2>&1
  echo "paced game $g: $(grep -c 'DIFFER' "$OUT/paced_s${SEED}_g${g}.out") differ" >> "$OUT/summary_s${SEED}.txt"
done
echo DONE >> "$OUT/summary_s${SEED}.txt"
