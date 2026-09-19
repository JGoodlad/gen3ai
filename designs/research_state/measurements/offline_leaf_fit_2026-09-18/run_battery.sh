#!/usr/bin/env bash
# THE MIRROR BATTERY at >=400 PAIRS (rule 25) — RUN ONLY IF A FIT PASSES ITS REGISTERED BAR.
#
# The fitted head is installed as the SEARCH LEAF via `main.search_dividend --leaf-head`; the
# CONTEMPORANEOUS control is the same checkpoint with its ORIGINAL head, in the same window, the
# same 4-shard geometry and the same `--games-seed 7` game indices, so every contrast is paired at
# the battle level and both cells buy their realized search width K from the same box at the same
# time (rule 23). Both cells are arm W piloting the FULL 719-team pool — the same population the
# fits' states come from.
#
# 🚨 `--leaf-head` loads a BARE state_dict whose keys must match the checkpoint's `win_head`, so
# only the WinProbHead-shaped fits (a / b / a0 / c0) are battery-eligible. A passing WIDE fit
# (c / d / e) has no runnable battery without an `src/` change, and that is reported rather than
# worked around.
#
#   usage: run_battery.sh <leafhead_*.pt> [out] [pairs] [shards]
set -uo pipefail
MAIN=/home/goodlad/dev/gen3ai
HEAD=${1:?a leafhead_*.pt written by fit_heads.py}
OUT=${2:-/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit/battery}
N=${3:-400}
SH=${4:-4}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
CKPT=$MAIN/models/ai_v13_02_flywheel_winprob/final_model.zip
export PYTHONPATH=$MAIN/src
export POKESIM_SIM_BRIDGE_BIN=$MAIN/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$MAIN" || exit 1
mkdir -p "$OUT/logs"

COMMON="--arm honest --budget 1 --opponents self --games-seed 7 --battle-timeout-s 1800 --battle-idle-s 120"
DEF="--root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0 --defensive-contested-deadline-s 3.0"
PER=$((N / SH))

shard() { local head=$1 cell=$2 lo=$3 n=$4; shift 4
  local extra=()
  [ "$head" = fit ] && extra=(--leaf-head "$HEAD")
  nice -n 15 "$PY" -m main.search_dividend "$CKPT" $COMMON "${extra[@]}" "$@" \
      --games-start "$lo" --games "$n" \
      --out "$OUT/${cell}__${head}__s${lo}.jsonl" \
      > "$OUT/logs/${cell}__${head}__s${lo}.log" 2>&1 &
  echo "  launched $cell $head [$lo,$((lo+n))) pid $!"
}

echo "=== BATTERY start $(date -u +%FT%TZ) head=$HEAD N=$N shards=$SH load=$(cut -d' ' -f1-3 /proc/loadavg)"
for s in $(seq 0 $((SH-1))); do
  lo=$((s * PER))
  shard fit  defB $lo $PER $DEF
  shard ctrl defB $lo $PER $DEF
  shard fit  grid $lo $PER --root-strategy grid
  shard ctrl grid $lo $PER --root-strategy grid
done
wait
echo "=== BATTERY done $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
