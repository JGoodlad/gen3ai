#!/usr/bin/env bash
# JOB 2 — the ROLLOUT-LEAF battery cell and its TWO CONTEMPORANEOUS CONTROLS, in ONE window on
# the SAME game indices (rule 23). `--impl node`: the playoff's nested rollouts FAIL under rust
# (AMENDMENT.md F1) and the failure is silent at the row level.
#
#   R=<4|8> PAIRS=<n> bash run_battery.sh [outdir]
#
# Cell file names are `<cell>__<head>__s<lo>.jsonl`, the 2026-09-11 battery's own convention, so
# its overlap guard and pairing logic apply unchanged.
set -u
ROOT=/home/goodlad/dev/gen3ai
OUT=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/kcurve/battery}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
CKPT=$ROOT/models/ai_v13_02_flywheel_winprob/final_model.zip
PAIRS="${PAIRS:-400}"; SH4="${SH4:-3}"; SH8="${SH8:-3}"
export PYTHONPATH=$ROOT/src
export CUDA_VISIBLE_DEVICES=""
export POKESIM_SIM_BRIDGE_BIN=$ROOT/src/rust_sim/target/release/sim_bridge
cd "$ROOT" || exit 1
mkdir -p "$OUT/logs"

# the mirror, side-swap pairs, one pinned game-seed — every cell plays the SAME battles
COMMON="--opponents self --games-seed 7 --impl node --device cpu"
# the playoff SCREEN is a nominator, not a width meter: the caps bind before the clock does
# (realized k_worlds 1.0, m_opp 5.2, 1.44 s of the 20 s budget in the smoke), so the budget's
# remainder is what buys the ROLLOUTS -- `deadline.fits(2 x rollout_cost)` is what stops them.
PFCOMMON="--arm playoff --budget 20 --max-worlds 4 --max-dice 2 --max-opp 6 --max-depth 1 \
 --battle-timeout-s 7200 --battle-idle-s 300"
DEF="--arm honest --budget 1 --root-strategy defensive --defensive-leaf winprob \
 --defensive-wp-margin 0.15 --defensive-confirm 0 --defensive-contested-deadline-s 3.0 \
 --battle-timeout-s 1800 --battle-idle-s 120"

shard() {  # cell ckptlabel lo n extra...
  local cell=$1 head=$2 lo=$3 n=$4; shift 4
  nice -n 15 "$PY" -m main.search_dividend "$CKPT" $COMMON "$@" \
      --games-start "$lo" --games "$n" \
      --out "$OUT/${cell}__${head}__s${lo}.jsonl" \
      > "$OUT/logs/${cell}__${head}__s${lo}.log" 2>&1 &
  echo "  launched $cell $head [$lo,$((lo+n))) pid $!"
}

echo "=== BATTERY start $(date -u +%FT%TZ)  PAIRS=$PAIRS shards pf4=$SH4 pf8=$SH8"
# The playoff shards divide [0,PAIRS) so a stop leaves each shard with a CONTIGUOUS window and the
# controls, which run the whole range, always cover whatever the playoff completed.
W4=$(( (PAIRS + SH4 - 1) / SH4 ))
for ((i=0;i<SH4;i++)); do shard pf4 armW $((i*W4)) "$W4" $PFCOMMON --playoff-rollouts 4; done
W8=$(( (PAIRS + SH8 - 1) / SH8 ))
for ((i=0;i<SH8;i++)); do shard pf8 armW $((i*W8)) "$W8" $PFCOMMON --playoff-rollouts 8; done
shard defB armW 0 "$PAIRS" $DEF
shard base armW 0 "$PAIRS" --arm base --battle-timeout-s 1800
wait
echo "=== BATTERY done $(date -u +%FT%TZ)"
