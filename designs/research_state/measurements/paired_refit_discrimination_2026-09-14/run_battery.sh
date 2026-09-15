#!/usr/bin/env bash
# PART A — the mirror battery's `grid` and rung-B defensive cells on three arms never read there,
# WITH a contemporaneous ai_v12_11_ladder_ctrl10M anchor in the same window (UNDERSTANDING rule 23).
#
# Exactly the 2026-09-11 registered operating point, same --games-seed 7, same game indices, so
# every contrast is paired at the battle level with the six heads already on this instrument.
set -uo pipefail
ROOT=/home/goodlad/dev/gen3ai
OUT=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/paired_refit/battery}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$ROOT/src
export POKESIM_SIM_BRIDGE_BIN=$ROOT/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
cd "$ROOT" || exit 1
mkdir -p "$OUT/logs"

rollout=models/ai_v12_23_ladder_rollout/snapshots/snapshot_000010000032.zip
ent05=models/ai_v12_28_ladder_ent05/snapshots/snapshot_000010000032.zip
vf025=models/ai_v12_29_ladder_vf025/snapshots/snapshot_000010000032.zip
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

PHASE=${2:-grid}
echo "=== PART A $PHASE start $(date -u +%FT%TZ)  load=$(cut -d' ' -f1-3 /proc/loadavg)"
if [ "$PHASE" = grid ]; then
  shard rollout "$rollout" grid 0 100 --root-strategy grid
  shard ent05   "$ent05"   grid 0 100 --root-strategy grid
  shard vf025   "$vf025"   grid 0 100 --root-strategy grid
  shard anchor  "$anchor"  grid 0 100 --root-strategy grid
else
  shard rollout "$rollout" defB 0 100 $DEF
  shard ent05   "$ent05"   defB 0 100 $DEF
  shard vf025   "$vf025"   defB 0 100 $DEF
  shard anchor  "$anchor"  defB 0 100 $DEF
fi
wait
echo "=== PART A $PHASE done $(date -u +%FT%TZ)  load=$(cut -d' ' -f1-3 /proc/loadavg)"
