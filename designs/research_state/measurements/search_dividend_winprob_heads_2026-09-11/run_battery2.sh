#!/usr/bin/env bash
# Follow-up: the registered rung-B cell + the `grid` reference on two more heads (PREDICTION.md §7),
# plus a CONTEMPORANEOUS ctrl10M anchor because the box is no longer quiet.
#
# Same protocol, same --games-seed 7, same game indices as the three heads already read, so every
# contrast is paired at the battle level. Shard counts are unequal on purpose: strata forces ~96%
# of decisions and costs ~11 s/battle where denseaux costs ~35 s.
set -uo pipefail
ROOT=/home/goodlad/dev/gen3ai
OUT=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/search_dividend2}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$ROOT/src
export POKESIM_SIM_BRIDGE_BIN=$ROOT/src/rust_sim/target/release/sim_bridge
cd "$ROOT" || exit 1
mkdir -p "$OUT/logs"

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

echo "=== PHASE2 defB start $(date -u +%FT%TZ)"
for lo in 0 160 320 480 640; do shard denseaux "$denseaux" defB "$lo" 160 $DEF; done
for lo in 0 400;             do shard strata   "$strata"   defB "$lo" 400 $DEF; done
shard anchor10M "$anchor" defBanchor 0 150 $DEF
wait
echo "=== PHASE2 defB done $(date -u +%FT%TZ)"

echo "=== PHASE2 grid start $(date -u +%FT%TZ)"
for lo in 0 50; do
  shard denseaux "$denseaux" grid "$lo" 50 --root-strategy grid
  shard strata   "$strata"   grid "$lo" 50 --root-strategy grid
done
wait
echo "=== PHASE2 grid done $(date -u +%FT%TZ)"
echo "PHASE2 COMPLETE $(date -u +%FT%TZ)"
