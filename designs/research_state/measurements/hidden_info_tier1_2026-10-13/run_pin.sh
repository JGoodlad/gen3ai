#!/usr/bin/env bash
# Run one of this folder's scripts against a checkpoint's OWN pin code, CPU only (CUDA hidden), at SCHED_IDLE and
# nice 19, pinned to the agent cores, under a memory cap and a timeout.
#
#   run_pin.sh <end2|prod|end2_pad|prod_pad> <threads> <GB> <timeout> <script> [args...]
#
# end2 / prod      = the UNMODIFIED pin checkouts the pilot built (~/.cache/gen3ai/vdp/pin_<cp>, 62085750 / 95d014fa):
#                    the truth (Phase 1), the census, the supplement bank, the network-belief dump, the O / P hand eval.
# end2_pad / prod_pad = the pin + the gender draw pad's code (98146ddb), ~/.cache/gen3ai/hib_t1/pin_<cp>_pad
#                    (67f3f7fa / 9a698801): every WORLD playout (a world team may carry the pad marker `M*` / `F*` / `N*`).
# A checkout whose HEAD is not the expected commit is refused. CPUs: $HIB_CPUS (default 5-7,13-15: physical cores 5-7
# and their SMT siblings, the agent cores while a GPU run is live; rule 5b).
set -euo pipefail
WHICH="$1"; THREADS="$2"; GB="$3"; TO="$4"; SCRIPT="$5"; shift 5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
case "$WHICH" in
  end2) CP=end2; PIN=/home/goodlad/.cache/gen3ai/vdp/pin_end2; WANT=62085750baab1ca275d810fbe0bb5a960904e183; KIND=pin ;;
  prod) CP=prod; PIN=/home/goodlad/.cache/gen3ai/vdp/pin_prod; WANT=95d014faeeb4d699703264a19094afee6f4eafc2; KIND=pin ;;
  end2_pad) CP=end2; PIN=/home/goodlad/.cache/gen3ai/hib_t1/pin_end2_pad; WANT=67f3f7fa; KIND=pad ;;
  prod_pad) CP=prod; PIN=/home/goodlad/.cache/gen3ai/hib_t1/pin_prod_pad; WANT=9a698801; KIND=pad ;;
  *) echo "unknown pin $WHICH" >&2; exit 2 ;;
esac
GITDIR="$(sed 's/^gitdir: //' "$PIN/.git")"
HEAD_SHA="$(cat "$GITDIR/HEAD")"
case "$HEAD_SHA" in "$WANT"*) ;; *) echo "pin checkout $PIN is at $HEAD_SHA, not $WANT" >&2; exit 3 ;; esac
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH="$PIN/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
# the pilot's modules read VDP_*; VDP_PIN_SHA is the pin's sha for the pilot's own check (a pad pin's own HEAD is in
# HIB_T1_PIN_HEAD, and this unit's scripts check THAT instead of calling the pilot's check_pin)
PILOT_SHA=$([ "$CP" = end2 ] && echo 62085750baab1ca275d810fbe0bb5a960904e183 || echo 95d014faeeb4d699703264a19094afee6f4eafc2)
export VDP_CP="$CP" VDP_PIN="$PIN" VDP_PIN_SHA="$PILOT_SHA" VDP_THREADS="$THREADS" VDP_PY="$PY"
export HIB_T1_PIN_KIND="$KIND" HIB_T1_PIN_HEAD="$HEAD_SHA" HIB_T1_PIN_DIR="$PIN"
cd "$PIN"
case "$SCRIPT" in *.sh) RUNNER=bash ;; *) RUNNER="$PY" ;; esac
CPUS="${HIB_CPUS:-5-7,13-15}"
exec chrt --idle 0 nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name "hibt1_$WHICH" "$GB" timeout "$TO" taskset -c "$CPUS" "$RUNNER" "$HERE/$SCRIPT" "$@"
