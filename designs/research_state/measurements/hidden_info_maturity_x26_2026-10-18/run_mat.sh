#!/usr/bin/env bash
# Run one of this folder's scripts at the checkout the battery registers for it, CPU only (CUDA hidden), at SCHED_IDLE
# and nice 19, pinned to the agent cores, one torch / OMP thread per process unless told otherwise, inside its own
# memory-capped scope and under a timeout (root rule 5b: the X26 run is live).
#
#   run_mat.sh <where> <threads> <GB> <timeout> <script> [args...]
#
# <where>:
#   pin    ~/.cache/gen3ai/hib_mat/pin_x26 (the series' pin, 9c033489; it holds the gender draw pad, so it plays the
#          worlds too): the bank, the roots, the truth, the belief read, every world playout
#   head   $HIB_MAT_HEAD (default ~/.cache/gen3ai/hib_mat/head): the world builder ($HIB_MAT_HEAD_SHA, when set, is
#          the commit it must be at)
# A checkout whose HEAD is not the expected commit is refused. A <script> ending in .sh runs under bash, anything else
# under python. CPUs: $HIB_CPUS (default 5-7,13-15: physical cores 5-7 and their SMT siblings).
set -euo pipefail
WHERE="$1"; THREADS="$2"; GB="$3"; TO="$4"; SCRIPT="$5"; shift 5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
C=/home/goodlad/.cache/gen3ai/hib_mat
case "$WHERE" in
  pin) DIR=$C/pin_x26; WANT=9c0334890d05785bbfadcb2ecbe3fe2d548efe9b; KIND=pin ;;
  head) DIR="${HIB_MAT_HEAD:-$C/head}"; WANT="${HIB_MAT_HEAD_SHA:-}"; KIND=head ;;
  *) echo "unknown checkout $WHERE" >&2; exit 2 ;;
esac
if [ -f "$DIR/.git" ]; then GITDIR="$(sed 's/^gitdir: //' "$DIR/.git")"; else GITDIR="$DIR/.git"; fi
HEAD_SHA="$(cat "$GITDIR/HEAD")"
case "$HEAD_SHA" in ref:*) HEAD_SHA="$(git -C "$DIR" rev-parse HEAD)" ;; esac
if [ -n "$WANT" ]; then
  case "$HEAD_SHA" in "$WANT"*) ;; *) echo "checkout $DIR is at $HEAD_SHA, not $WANT" >&2; exit 3 ;; esac
fi
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH="$DIR/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
# the pilot's modules read VDP_* (its thread count, the pin dir for the data tables, the interpreter)
export VDP_PIN="$DIR" VDP_PIN_SHA="$HEAD_SHA" VDP_THREADS="$THREADS" VDP_PY="$PY"
export HIB_MAT_KIND="$KIND" HIB_MAT_CHECKOUT_HEAD="$HEAD_SHA" HIB_MAT_CHECKOUT_DIR="$DIR"
cd "$DIR"
case "$SCRIPT" in *.sh) RUNNER=bash ;; *) RUNNER="$PY" ;; esac
case "$SCRIPT" in /*) ;; *) SCRIPT="$HERE/$SCRIPT" ;; esac
CPUS="${HIB_CPUS:-5-7,13-15}"
exec chrt --idle 0 nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name "hibmat_$WHERE" "$GB" timeout "$TO" \
  taskset -c "$CPUS" "$RUNNER" "$SCRIPT" "$@"
