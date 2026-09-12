#!/usr/bin/env bash
# Phase 3: the third head (`cflabels`) on the registered rung-B cell + its OWN contemporaneous
# ctrl10M anchor window (indices 150-299), and the `grid` @1 s reference cell for all three new
# heads. Same protocol, same --games-seed 7, same game indices as every head already read.
set -uo pipefail
ROOT=/home/goodlad/dev/gen3ai
OUT=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/search_dividend2}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$ROOT/src
export POKESIM_SIM_BRIDGE_BIN=$ROOT/src/rust_sim/target/release/sim_bridge
cd "$ROOT" || exit 1
mkdir -p "$OUT/logs"

cflabels=models/ai_v12_12_ladder_cflabels/snapshots/snapshot_000010000032.zip
strata=models/ai_v12_17_ladder_strata/snapshots/snapshot_000010000032.zip
denseaux=models/ai_v12_20_ladder_denseaux/snapshots/snapshot_000010000032.zip
anchor=models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip

COMMON="--arm honest --budget 1 --opponents self --games-seed 7 --battle-timeout-s 1800 --battle-idle-s 120"
DEF="--root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0 --defensive-contested-deadline-s 3.0"

shard() {
  local head=$1 ckpt=$2 cell=$3 lo=$4 n=$5; shift 5
  nice -n 15 "$PY" -m main.search_dividend "$ckpt" $COMMON "$@" \
      --games-start "$lo" --games "$n" \
      --out "$OUT/${cell}__${head}__s${lo}.jsonl" \
      > "$OUT/logs/${cell}__${head}__s${lo}.log" 2>&1 &
  echo "  launched $cell $head [$lo,$((lo+n))) pid $!"
}

echo "=== PHASE3 start $(date -u +%FT%TZ)"
for lo in 0 200 400 600; do shard cflabels "$cflabels" defB "$lo" 200 $DEF; done
shard anchor10Mb "$anchor" defBanchor2 150 150 $DEF
# NOTE: the strata / denseaux `grid` cells are run by run_battery2.sh's own grid phase. Launching
# them here too would put two processes on the SAME out file and replay the not-yet-done games.
for lo in 0 50; do shard cflabels "$cflabels" grid "$lo" 50 --root-strategy grid; done
wait
echo "=== PHASE3 done $(date -u +%FT%TZ)"
