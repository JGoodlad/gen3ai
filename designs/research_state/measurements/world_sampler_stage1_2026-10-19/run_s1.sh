#!/usr/bin/env bash
# Run one of this folder's scripts at the checkout stage 1 registers for it: CPU only (CUDA hidden), SCHED_IDLE and
# nice 19, pinned to the agent cores, one torch / OMP thread, inside its own memory-capped scope and under a timeout
# (root rule 5b: a training run is live).
#
#   run_s1.sh <where> <threads> <GB> <timeout> <script> [args...]
#
# <where>:
#   builder  $WS1_BUILDER (default ~/.cache/gen3ai/ws_stage1/builder): every arm's worlds ($WS1_BUILDER_SHA, when set,
#            is the commit it must be at)
#   pin      ~/.cache/gen3ai/ws_stage1/pin_rolls: X26's pin 9c033489 + the damage-roll control (play_s1.py checks it)
# The measurement scripts always come from THIS folder's checkout (the DRIVER); `main` from the selected checkout.
# CPUs: $WS1_CPUS (default 5-7,13-15: physical cores 5-7 and their SMT siblings).
set -euo pipefail
WHERE="$1"; THREADS="$2"; GB="$3"; TO="$4"; SCRIPT="$5"; shift 5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
C=/home/goodlad/.cache/gen3ai/ws_stage1
case "$WHERE" in
  builder) DIR="${WS1_BUILDER:-$C/builder}"; WANT="${WS1_BUILDER_SHA:-}"; KIND=builder ;;
  pin) DIR=$C/pin_rolls; WANT=""; KIND=pin ;;
  *) echo "unknown checkout $WHERE" >&2; exit 2 ;;
esac
HEAD_SHA="$(git -C "$DIR" rev-parse HEAD)"
if [ -n "$WANT" ]; then
  case "$HEAD_SHA" in "$WANT"*) ;; *) echo "checkout $DIR is at $HEAD_SHA, not $WANT" >&2; exit 3 ;; esac
fi
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH="$DIR/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
# the pilot's modules read VDP_* (its thread count, the pin dir for the data tables, the interpreter)
export VDP_PIN="$DIR" VDP_PIN_SHA="$HEAD_SHA" VDP_THREADS="$THREADS" VDP_PY="$PY"
export WS1_KIND="$KIND" WS1_CHECKOUT_HEAD="$HEAD_SHA" WS1_CHECKOUT_DIR="$DIR"
cd "$DIR"
case "$SCRIPT" in /*) ;; *) SCRIPT="$HERE/$SCRIPT" ;; esac
CPUS="${WS1_CPUS:-5-7,13-15}"
exec chrt --idle 0 nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name "ws1_$WHERE" "$GB" timeout "$TO" \
  taskset -c "$CPUS" "$PY" "$SCRIPT" "$@"
