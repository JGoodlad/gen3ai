#!/usr/bin/env bash
# The search-dividend battery for the three current WIN-PROB heads.
#
# Protocol: the registered defensive-search cells (defensive_search_first_cell_2026-08-29.md =
# rung A, defensive_search_iter2_2026-08-29.md = rung B) plus the naive `grid` reference cell,
# all in the MIRROR (`--opponents self`, side-swap on by default, null = 0.50 by construction)
# at the historical salt `--games-seed 7`, so every head plays the SAME battles as every other
# head and as the historical v9 cells.
#
# Runs from the MAIN checkout (models/ lives only there). CPU only, nice 15, BLAS pinned by the
# CLI itself. Shards take disjoint --games-start windows and write SEPARATE files: the row schema
# records neither the checkpoint nor --defensive-contested-deadline-s, so a pooled file could not
# tell rung A from rung B or one head from another. File separation IS the cell identity here.
set -uo pipefail
ROOT=/home/goodlad/dev/gen3ai
OUT=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/search_dividend}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$ROOT/src
export POKESIM_SIM_BRIDGE_BIN=$ROOT/src/rust_sim/target/release/sim_bridge
cd "$ROOT" || exit 1
mkdir -p "$OUT/logs"

ctrl10M=models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip
lambda09=models/ai_v12_19_ladder_lambda09/snapshots/snapshot_000010000032.zip
wp73M=models/ai_v12_02_winprob_critic/checkpoints/checkpoint_73121280_steps.zip

COMMON="--arm honest --budget 1 --opponents self --games-seed 7 --battle-timeout-s 1800 --battle-idle-s 120"
DEF="--root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0"

# shard <head> <ckpt> <cell> <games_start> <games> <extra...>
shard() {
  local head=$1 ckpt=$2 cell=$3 lo=$4 n=$5; shift 5
  nice -n 15 "$PY" -m main.search_dividend "$ckpt" $COMMON "$@" \
      --games-start "$lo" --games "$n" \
      --out "$OUT/${cell}__${head}__s${lo}.jsonl" \
      > "$OUT/logs/${cell}__${head}__s${lo}.log" 2>&1 &
  echo "  launched $cell $head [$lo,$((lo+n))) pid $!"
}

# phase <cell> <lo:n,lo:n,...>  <flags...>   — one process per (head, shard); 3 shards/head so the
# three heads always see the SAME contention (the budget is a wall clock, so symmetry is the point)
phase() {
  local cell=$1 windows=$2; shift 2
  echo "=== PHASE $cell start $(date -u +%FT%TZ)"
  for h in ctrl10M lambda09 wp73M; do
    for w in ${windows//,/ }; do
      shard "$h" "${!h}" "$cell" "${w%%:*}" "${w##*:}" "$@"
    done
  done
  wait
  echo "=== PHASE $cell done  $(date -u +%FT%TZ)"
}

# Rung B — iteration 2's operating point (contested decisions get 3 s), 400 pairs / head.
phase defB 0:134,134:133,267:133 $DEF --defensive-contested-deadline-s 3.0
# Rung A — iteration 1's operating point (contested decisions get the uniform 1 s), 150 pairs.
phase defA 0:50,50:50,100:50 $DEF
# Reference — naive grid search, no gate, no race: the purest form of the question, 100 pairs.
phase grid 0:34,34:33,67:33 --root-strategy grid

echo "ALL PHASES COMPLETE $(date -u +%FT%TZ)"
