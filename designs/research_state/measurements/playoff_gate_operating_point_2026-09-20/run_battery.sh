#!/usr/bin/env bash
# THE LIVE BATTERY at the gate's chosen OPERATING POINT, with its two contemporaneous controls,
# in ONE window on the SAME game indices (rule 23).
#
#   bash run_battery.sh [outdir]
#
# THE POINT is R = 4, --playoff-se-k 0.5, --playoff-min-pairs 4, chosen by PREDICTION.md §3's
# registered rule (R pinned to 4 by COST before the curve existed; then the SE multiple maximising
# EV gain per DECISION on the banked dice: +0.0157 win-prob units at a 0.4556 resolve rate,
# against +0.0025 at 0.0451 for the production k = 2.0).
#
# 🚨 `--impl node`: the playoff's nested counterfactual rollouts FAIL under `--impl rust` and the
# failure is INVISIBLE at the row level (2026-09-19 hazard 1, backlog P1).
# 🚨 `--budget 120`: the deadline must buy 2R = 8 rollouts or `--playoff-rollouts` is INERT and
# `short_r_refusal` fires on the first game. 120 s realized exactly R = 4.00 on 2026-09-19.
#
# Cell files are `<cell>__<head>__s<lo>.jsonl` -- the 2026-09-11 battery's convention, so its
# overlap guard and pairing logic apply unchanged.
set -u
ROOT=/home/goodlad/dev/gen3ai
OUT=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/gatecurve/battery}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
CKPT=$ROOT/models/ai_v13_02_flywheel_winprob/final_model.zip
PAIRS="${PAIRS:-200}"; SH="${SH:-8}"
export PYTHONPATH=$ROOT/src
export CUDA_VISIBLE_DEVICES=""
export POKESIM_SIM_BRIDGE_BIN=$ROOT/src/rust_sim/target/release/sim_bridge
cd "$ROOT" || exit 1
mkdir -p "$OUT/logs"

COMMON="--opponents self --games-seed 7 --impl node --device cpu"
PF="--arm playoff --budget 120 --max-worlds 4 --max-dice 2 --max-opp 6 --max-depth 1 \
 --playoff-rollouts 4 --playoff-se-k 0.5 --playoff-min-pairs 4 \
 --battle-timeout-s 7200 --battle-idle-s 300"
DEF="--arm honest --budget 1 --root-strategy defensive --defensive-leaf winprob \
 --defensive-wp-margin 0.15 --defensive-confirm 0 --defensive-contested-deadline-s 3.0 \
 --battle-timeout-s 1800 --battle-idle-s 120"

shard() {  # cell head lo n extra...
  local cell=$1 head=$2 lo=$3 n=$4; shift 4
  nice -n 15 "$PY" -m main.search_dividend "$CKPT" $COMMON "$@" \
      --games-start "$lo" --games "$n" \
      --out "$OUT/${cell}__${head}__s${lo}.jsonl" \
      > "$OUT/logs/${cell}__${head}__s${lo}.log" 2>&1 &
  echo "  launched $cell [$lo,$((lo+n))) pid $!"
}

echo "=== BATTERY start $(date -u +%FT%TZ)  PAIRS=$PAIRS shards=$SH"
# Contiguous windows so a clock-stop leaves each shard a complete prefix and the controls -- which
# run the whole range -- always cover whatever the playoff completed.
W=$(( (PAIRS + SH - 1) / SH ))
for ((i=0;i<SH;i++)); do shard pfk05 armW $((i*W)) "$W" $PF; done
shard defB armW 0 "$PAIRS" $DEF
shard base armW 0 "$PAIRS" --arm base --battle-timeout-s 1800
wait
echo "=== BATTERY done $(date -u +%FT%TZ)"
