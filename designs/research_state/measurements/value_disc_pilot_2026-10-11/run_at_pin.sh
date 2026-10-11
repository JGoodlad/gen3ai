#!/usr/bin/env bash
# Run one of this folder's scripts against a checkpoint's OWN pin code (a detached checkout of the
# pin, never main), CPU only (CUDA hidden), nice 19, under a memory cap and a timeout.
#
#   run_at_pin.sh <end2|prod> <threads> <GB> <timeout> <script> [args...]
# A <script> ending in .sh runs under bash (with $VDP_PY the interpreter), anything else under python.
#
# Pins: end2 = rb_es_end2_s2001 @ 62085750 (tag closing-test-pend2); prod = rb_es_prod_s2001 @
# 95d014fa. The pin checkouts live at $VDP_PIN_ROOT/pin_<cp> (default ~/.cache/gen3ai/vdp; made with
# `git worktree add --detach`), each with its own src/rust_env and src/rust_sim target/ (never another
# checkout's). The script refuses a checkout whose HEAD is not the pin.
set -euo pipefail
CP="$1"; THREADS="$2"; GB="$3"; TO="$4"; SCRIPT="$5"; shift 5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
ROOT="${VDP_PIN_ROOT:-$HOME/.cache/gen3ai/vdp}"
case "$CP" in
  end2) PIN="$ROOT/pin_end2"; WANT=62085750baab1ca275d810fbe0bb5a960904e183 ;;
  prod) PIN="$ROOT/pin_prod"; WANT=95d014faeeb4d699703264a19094afee6f4eafc2 ;;
  *) echo "unknown checkpoint $CP" >&2; exit 2 ;;
esac
GITDIR="$(sed 's/^gitdir: //' "$PIN/.git")"
HEAD_SHA="$(cat "$GITDIR/HEAD")"
if [ "$HEAD_SHA" != "$WANT" ]; then echo "pin checkout $PIN is at $HEAD_SHA, not $WANT" >&2; exit 3; fi
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH="$PIN/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export VDP_CP="$CP" VDP_PIN="$PIN" VDP_PIN_SHA="$HEAD_SHA" VDP_THREADS="$THREADS" VDP_PY="$PY"
cd "$PIN"
case "$SCRIPT" in *.sh) RUNNER=bash ;; *) RUNNER="$PY" ;; esac
exec nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name "vdp_$CP" "$GB" timeout "$TO" "$RUNNER" "$HERE/$SCRIPT" "$@"
