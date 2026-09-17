#!/usr/bin/env bash
# PART 2 — the MIRROR BATTERY at 400 PAIRS (rule 25 retires the 100-pair cell for L2 claims).
# The fork arm as the LEAF against a CONTEMPORANEOUS ctrl10M cell — same window, same shard
# geometry, same `--games-seed 7` game indices, so every contrast is paired at the battle level
# and the two cells' realized K is bought from the same box at the same time (rule 23).
#
# The operating point is the 2026-09-11 registered one, unchanged, so these cells join the nine
# heads already on this instrument.
set -uo pipefail
MAIN=/home/goodlad/dev/gen3ai
OUT=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/fork_read/battery}
N=${2:-400}
SH=${3:-4}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$MAIN/src
export POKESIM_SIM_BRIDGE_BIN=$MAIN/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$MAIN" || exit 1
mkdir -p "$OUT/logs"

arm=$MAIN/models/ai_v13_03_fork/snapshots/snapshot_000010000032.zip
ctrl=$MAIN/models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip
COMMON="--arm honest --budget 1 --opponents self --games-seed 7 --battle-timeout-s 1800 --battle-idle-s 120"
DEF="--root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0 --defensive-contested-deadline-s 3.0"
PER=$((N / SH))

shard() {
  local head=$1 ckpt=$2 cell=$3 lo=$4 n=$5; shift 5
  nice -n 15 "$PY" -m main.search_dividend "$ckpt" $COMMON "$@" \
      --games-start "$lo" --games "$n" \
      --out "$OUT/${cell}__${head}__s${lo}.jsonl" \
      > "$OUT/logs/${cell}__${head}__s${lo}.log" 2>&1 &
  echo "  launched $cell $head [$lo,$((lo+n))) pid $!"
}

echo "=== BATTERY start $(date -u +%FT%TZ) N=$N shards=$SH load=$(cut -d' ' -f1-3 /proc/loadavg)"
for s in $(seq 0 $((SH-1))); do
  lo=$((s * PER))
  shard fork "$arm"  defB $lo $PER $DEF
  shard ctrl "$ctrl" defB $lo $PER $DEF
  shard fork "$arm"  grid $lo $PER --root-strategy grid
  shard ctrl "$ctrl" grid $lo $PER --root-strategy grid
done
wait
echo "=== BATTERY done $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
