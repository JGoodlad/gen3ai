#!/usr/bin/env bash
# Run one of this folder's scripts at the checkout E2 registers for it, CPU only (CUDA hidden), at SCHED_IDLE and nice 19,
# pinned to the agent cores, one torch / OMP thread per process unless told otherwise, inside its own memory-capped
# scope and under a timeout (root rule 5b: the X26 run is live).
#
#   run_e2.sh <where> <threads> <GB> <timeout> <script> [args...]
#
# <where>:
#   gen_end2 / gen_prod  ~/.cache/gen3ai/hib_e2/pin_<cp>  (the pin commit + the Showdown submodule): the GAMES
#   end2 / prod          ~/.cache/gen3ai/vdp/pin_<cp>     (the UNMODIFIED pins Tier 1 used): the BANK, PHASE 1
#   end2_pad / prod_pad  ~/.cache/gen3ai/hib_t1/pin_<cp>_pad (the pin + the gender draw pad): every WORLD playout
#   head                 $HIB_E2_HEAD (default ~/.cache/gen3ai/hib_e2/head): the WORLD builder
# A checkout whose HEAD is not the expected commit is refused. A <script> ending in .sh runs under bash, anything else
# under python. CPUs: $HIB_CPUS (default 5-7,13-15: physical cores 5-7 and their SMT siblings).
set -euo pipefail
WHERE="$1"; THREADS="$2"; GB="$3"; TO="$4"; SCRIPT="$5"; shift 5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
C=/home/goodlad/.cache/gen3ai
case "$WHERE" in
  gen_end2) CP=end2; DIR=$C/hib_e2/pin_end2; WANT=62085750baab1ca275d810fbe0bb5a960904e183; KIND=gen ;;
  gen_prod) CP=prod; DIR=$C/hib_e2/pin_prod; WANT=95d014faeeb4d699703264a19094afee6f4eafc2; KIND=gen ;;
  end2) CP=end2; DIR=$C/vdp/pin_end2; WANT=62085750baab1ca275d810fbe0bb5a960904e183; KIND=pin ;;
  prod) CP=prod; DIR=$C/vdp/pin_prod; WANT=95d014faeeb4d699703264a19094afee6f4eafc2; KIND=pin ;;
  end2_pad) CP=end2; DIR=$C/hib_t1/pin_end2_pad; WANT=67f3f7fa; KIND=pad ;;
  prod_pad) CP=prod; DIR=$C/hib_t1/pin_prod_pad; WANT=9a698801; KIND=pad ;;
  head) CP=""; DIR="${HIB_E2_HEAD:-$C/hib_e2/head}"; WANT="${HIB_E2_HEAD_SHA:-}"; KIND=head ;;
  *) echo "unknown checkout $WHERE" >&2; exit 2 ;;
esac
GITDIR="$(sed 's/^gitdir: //' "$DIR/.git")"
HEAD_SHA="$(cat "$GITDIR/HEAD")"
if [ -n "$WANT" ]; then
  case "$HEAD_SHA" in "$WANT"*) ;; *) echo "checkout $DIR is at $HEAD_SHA, not $WANT" >&2; exit 3 ;; esac
fi
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH="$DIR/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
if [ -n "$CP" ]; then
  PILOT_SHA=$([ "$CP" = end2 ] && echo 62085750baab1ca275d810fbe0bb5a960904e183 || echo 95d014faeeb4d699703264a19094afee6f4eafc2)
  # the pilot's modules read VDP_*; VDP_PIN_SHA is the pin's sha for the pilot's own check
  export VDP_CP="$CP" VDP_PIN="$DIR" VDP_PIN_SHA="$PILOT_SHA" VDP_THREADS="$THREADS" VDP_PY="$PY"
fi
export HIB_E2_KIND="$KIND" HIB_E2_CHECKOUT_HEAD="$HEAD_SHA" HIB_E2_CHECKOUT_DIR="$DIR" HIB_E2_CP="$CP"
cd "$DIR"
case "$SCRIPT" in *.sh) RUNNER=bash ;; *) RUNNER="$PY" ;; esac
case "$SCRIPT" in /*) ;; *) SCRIPT="$HERE/$SCRIPT" ;; esac
CPUS="${HIB_CPUS:-5-7,13-15}"
exec chrt --idle 0 nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name "hibe2_$WHERE" "$GB" timeout "$TO" \
  taskset -c "$CPUS" "$RUNNER" "$SCRIPT" "$@"
