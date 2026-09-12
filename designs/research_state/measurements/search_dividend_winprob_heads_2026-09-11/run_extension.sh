#!/usr/bin/env bash
# The registered EXTENSION (PREDICTION.md §6) + the exact-50 BASE control.
#
# Extension: rung B to 800 pairs on the two heads whose 400-pair CI straddled 0.50 with a point
# estimate >= 0.51 (ctrl10M, lambda09), over FRESH game indices 400-799. 4 shards per head = 8
# concurrent processes, so the box carries the same load rung B's own 9 shards did — the budget
# is a wall clock, and a second look taken under different contention is not a second look at the
# same cell.
#
# Base control: `--arm base` is the policy on BOTH sides with search structurally off. Its paired
# rate must be exactly 0.5000 (every pair splits) — the harness's own no-effect check. It searches
# nothing, so contention cannot touch it; it runs last and alone.
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

COMMON="--budget 1 --opponents self --games-seed 7 --battle-timeout-s 1800 --battle-idle-s 120"
DEF="--root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0"

shard() {
  local head=$1 ckpt=$2 cell=$3 lo=$4 n=$5; shift 5
  nice -n 15 "$PY" -m main.search_dividend "$ckpt" $COMMON "$@" \
      --games-start "$lo" --games "$n" \
      --out "$OUT/${cell}__${head}__s${lo}.jsonl" \
      > "$OUT/logs/${cell}__${head}__s${lo}.log" 2>&1 &
  echo "  launched $cell $head [$lo,$((lo+n))) pid $!"
}

echo "=== PHASE defB-extension start $(date -u +%FT%TZ)"
for h in ctrl10M lambda09; do
  for lo in 400 500 600 700; do
    shard "$h" "${!h}" defB "$lo" 100 --arm honest $DEF --defensive-contested-deadline-s 3.0
  done
done
wait
echo "=== PHASE defB-extension done $(date -u +%FT%TZ)"

echo "=== PHASE base start $(date -u +%FT%TZ)"
for h in ctrl10M lambda09 wp73M; do
  shard "$h" "${!h}" base 0 100 --arm base --root-strategy grid
done
wait
echo "=== PHASE base done $(date -u +%FT%TZ)"
echo "EXTENSION COMPLETE $(date -u +%FT%TZ)"
