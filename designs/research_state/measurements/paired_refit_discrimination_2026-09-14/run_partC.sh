#!/usr/bin/env bash
# PART C — the REFIT head on the leaf instrument, against the ORIGINAL head as its
# CONTEMPORANEOUS control in the SAME window on the SAME game indices (rule 23).
#
# Both cells load the SAME checkpoint (ai_v12_11_ladder_ctrl10M@10000032). The only difference is
# `--leaf-head`: the win head is a leak-safe SIDE readout, never in pi/vf, so the swap changes the
# SEARCH LEAF and nothing about how either side plays unsearched — the mirror's 0.50 null stays
# structural. The `orig` cell runs `--leaf-head head_original.pt` rather than no flag at all, so
# both cells take the identical code path and a difference cannot be the hook.
set -uo pipefail
# 🚨 Part C runs from the WORKTREE, not the main checkout: `--leaf-head` is new code and main's
# `src/` does not have it (the first launch died on `unrecognized arguments: --leaf-head`). The
# checkpoint is named by ABSOLUTE path so nothing depends on which tree we stand in, and
# POKESIM_SIM_BRIDGE_BIN still points at the binary built in the MAIN checkout (never build into
# main's target from a worktree).
ROOT=/home/goodlad/dev/gen3ai-wt/pairedrefit
MAIN=/home/goodlad/dev/gen3ai
OUT=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/paired_refit/partC}
HEADS=${2:-/home/goodlad/.claude/jobs/9ab51de6/tmp/paired_refit/refit}
BEST=${3:?usage: run_partC.sh <out> <refit_dir> <best_head_file>}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$ROOT/src
export POKESIM_SIM_BRIDGE_BIN=$MAIN/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
cd "$ROOT" || exit 1
mkdir -p "$OUT/logs"

ckpt=$MAIN/models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip
COMMON="--arm honest --budget 1 --opponents self --games-seed 7 --battle-timeout-s 1800 --battle-idle-s 120"
DEF="--root-strategy defensive --defensive-leaf winprob --defensive-wp-margin 0.15 --defensive-confirm 0 --defensive-contested-deadline-s 3.0"

shard() {
  local head=$1 hp=$2 cell=$3 lo=$4 n=$5; shift 5
  nice -n 15 "$PY" -m main.search_dividend "$ckpt" $COMMON --leaf-head "$hp" "$@" \
      --games-start "$lo" --games "$n" \
      --out "$OUT/${cell}__${head}__s${lo}.jsonl" \
      > "$OUT/logs/${cell}__${head}__s${lo}.log" 2>&1 &
  echo "  launched $cell $head [$lo,$((lo+n))) pid $!"
}

echo "=== PART C start $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "    refit head: $BEST"
shard refit "$BEST" grid 0 100 --root-strategy grid
shard orig  "$HEADS/head_original.pt" grid 0 100 --root-strategy grid
shard refit "$BEST" defB 0 100 $DEF
shard orig  "$HEADS/head_original.pt" defB 0 100 $DEF
wait
echo "=== PART C done $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
