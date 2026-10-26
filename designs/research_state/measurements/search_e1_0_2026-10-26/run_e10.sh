#!/usr/bin/env bash
# Run play_e10.py at the maturity battery's X26 pin (~/.cache/gen3ai/hib_mat/pin_x26, 9c033489), CPU only, one thread,
# at SCHED_IDLE + nice 19 on the agent cores, in its own mem_cap scope (root rule 5b). A chunk is bounded:
#
#   run_e10.sh <timeout> [--max-s S] -- <play_e10.py args...>
#
# --max-s S puts the chunk in mem_cap's SHORT lane (S <= 600; killed past S + 30 s, exit 88 = re-run as a long chunk).
set -euo pipefail
TO="$1"; shift
MAXS=()
if [ "${1:-}" = "--max-s" ]; then MAXS=(--max-s "$2"); shift 2; fi
[ "${1:-}" = "--" ] && shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
DIR=/home/goodlad/.cache/gen3ai/hib_mat/pin_x26
WANT=9c0334890d05785bbfadcb2ecbe3fe2d548efe9b
HEAD_SHA="$(git -C "$DIR" rev-parse HEAD)"
[ "$HEAD_SHA" = "$WANT" ] || { echo "checkout $DIR is at $HEAD_SHA, not $WANT" >&2; exit 3; }
[ -z "$(git -C "$DIR" status --porcelain)" ] || { echo "checkout $DIR is not clean" >&2; exit 3; }
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH="$DIR/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VDP_PIN="$DIR" VDP_PIN_SHA="$HEAD_SHA" VDP_THREADS=1 VDP_PY="$PY"
export HIB_MAT_KIND=pin HIB_MAT_CHECKOUT_HEAD="$HEAD_SHA" HIB_MAT_CHECKOUT_DIR="$DIR"
export E10_DRIVER_SHA="${E10_DRIVER_SHA:-$(git -C "$REPO" rev-parse HEAD)}"
cd "$DIR"
exec chrt --idle 0 nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name e10_play "${MAXS[@]}" 6 timeout "$TO" \
  taskset -c 5-7,13-15 "$PY" "$HERE/play_e10.py" "$@"
