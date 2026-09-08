#!/usr/bin/env bash
#
# watch_run.sh — SOP §2 LAYER 1: the OS-level watcher for one training run.
#
#     scripts/ops/watch_run.sh <run-name|run-dir> [options]
#     scripts/ops/watch_run.sh --help
#
# `designs/ops/TRAINING_RUN_SOP.md` §2 names four watch layers and says layers 1-2 are OS
# processes that keep the machine working if the Claude session dies. This is layer 1: it polls
# progress, bounds the IDLE gap (the wedge limit), matches FAILURE WORDS as well as progress,
# checks the arm-INVALIDATING condition, and appends one line per tick to a status file that a
# `Monitor` (layer 3) or a later session reads.
#
# Every tick carries MARGINAL fps measured from CHECKPOINT MTIMES (owner, 2026-08-23), never
# SB3's cumulative `time/fps` and never the `T = step / fps` derivative — `fps` is logged as an
# INTEGER, so that derivative explodes at high step counts (6,008 fps was observed) and rots
# silently as the run gets longer.
#
# Promoted 2026-09-07 from the Training Run session's `watch_arm2.sh`, where the run name, the
# model directory, the pid file, the status file and the launcher log were all literals.
set -u

usage() {
    cat <<'EOF'
watch_run.sh — the SOP §2 layer-1 OS watcher for one training run.

USAGE
    scripts/ops/watch_run.sh <run-name|run-dir> [options]
    scripts/ops/watch_run.sh --help

ARGUMENTS
    <run-name|run-dir>   a bare name is looked up in the run archive (models/); anything
                         with a slash is taken as a directory and must exist.

OPTIONS
    --pid-file PATH      file holding the LAUNCHER pid. When it names a pid that is gone,
                         the watcher reports FAILURE and exits. Required unless
                         --no-pid-check is given.
    --no-pid-check       do not watch a launcher pid (progress + failure words only).
    --status-file PATH   where to append ticks (default: <run-dir>/watch_status.txt).
    --launcher-log PATH  the LAUNCHER's own log, scanned for the arm-INVALIDATING
                         `--sync-to-main` line. Optional; its absence is reported once and
                         then the check CANNOT fire — an absent file is not a clean one.
    --wedge-seconds N    no step progress for this long = WEDGED (default 2100 = 35 min).
    --interval N         seconds between ticks (default 300).

WHAT IT WRITES
    one line per tick:  [<timestamp>] ok step=<n> ckpt=<n> marginal_fps=<n> ep_len=<n>
    and on the way out: FAILURE: <reason>  then  WATCHER EXIT

ENVIRONMENT
    GEN3AI_MODELS_DIR  the run archive, if it is not the main checkout's models/
    GEN3AI_PYTHON      interpreter for the marginal-fps helper

EXIT
    0  the watcher exited after reporting a terminal condition     2  bad usage
EOF
}

case "${1:-}" in
    -h|--help|"") usage; [ -z "${1:-}" ] && exit 2 || exit 0 ;;
esac

# shellcheck source=scripts/ops/_common.sh
. "$(dirname "$(readlink -f "$0")")/_common.sh"

RUN_ARG="$1"; shift
PIDF=""; NO_PID=0; STATUS=""; LAUNCHER_LOG=""; WEDGE=2100; INTERVAL=300
while [ $# -gt 0 ]; do
    case "$1" in
        --pid-file)       PIDF="$2"; shift 2 ;;
        --no-pid-check)   NO_PID=1; shift ;;
        --status-file)    STATUS="$2"; shift 2 ;;
        --launcher-log)   LAUNCHER_LOG="$2"; shift 2 ;;
        --wedge-seconds)  WEDGE="$2"; shift 2 ;;
        --interval)       INTERVAL="$2"; shift 2 ;;
        *) echo "REFUSING: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

D="$(ops_resolve_run "$RUN_ARG")" || exit 2
CLOG="$D/launcher_child.log"
STATUS="${STATUS:-$D/watch_status.txt}"
PY="$(ops_python)"

if [ "$NO_PID" -eq 0 ] && [ -z "$PIDF" ]; then
    echo "REFUSING: --pid-file is required (or pass --no-pid-check to watch progress only)." >&2
    exit 2
fi

say() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

last_step=-1; last_move=$(date +%s)
say "WATCHER START (pid $$) run=$D — wedge limit ${WEDGE}s, interval ${INTERVAL}s"
# An ABSENT launcher log is not a clean one: `grep -q ... 2>/dev/null` on a missing file returns
# non-zero, so the arm-INVALIDATING check would read exactly like "no --sync-to-main found".
# Say so once, at the top, where the reader of the status file will see it.
if [ -n "$LAUNCHER_LOG" ] && [ ! -f "$LAUNCHER_LOG" ]; then
    say "WARN: launcher log $LAUNCHER_LOG is ABSENT — the --sync-to-main invalidation check CANNOT fire"
elif [ -z "$LAUNCHER_LOG" ]; then
    say "WARN: no --launcher-log given — the --sync-to-main invalidation check is NOT ARMED"
fi

while true; do
    if [ "$NO_PID" -eq 0 ]; then
        pid=$(cat "$PIDF" 2>/dev/null || echo 0)
        if ! kill -0 "$pid" 2>/dev/null; then
            say "FAILURE: launcher pid $pid GONE"; break
        fi
    fi
    # FAILURE words, not just progress
    if grep -qiE 'OutOfMemory|CUDA out of memory|FATAL|Traceback|FATAL_CONFIG' "$CLOG" 2>/dev/null; then
        say "FAILURE: error text in child log — $(grep -oiE 'OutOfMemory|CUDA out of memory|FATAL_CONFIG|FATAL|Traceback' "$CLOG" | tail -1)"
        break
    fi
    # arm-INVALIDATING: the pin must be the registered sha. A `--sync-to-main` line voids the arm.
    if [ -n "$LAUNCHER_LOG" ] && grep -q 'sync-to-main' "$LAUNCHER_LOG" 2>/dev/null; then
        say "FAILURE: --sync-to-main in the launcher log — the arm is VOID, stop it"; break
    fi
    s=$(grep -oE 'total_timesteps *\| *[0-9]+' "$CLOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    now=$(date +%s)
    if [ -n "$s" ] && [ "$s" != "$last_step" ]; then last_step=$s; last_move=$now; fi
    if [ $((now-last_move)) -gt "$WEDGE" ]; then
        say "FAILURE: WEDGED — no step progress for $(( (now-last_move)/60 )) min at step ${last_step}"
        break
    fi
    # MARGINAL fps from checkpoint mtimes (owner, 2026-08-23), not from the log's own fps line
    marg="n/a"
    if [ -n "$(ls -t "$D"/checkpoints/*.zip 2>/dev/null | head -2)" ]; then
        marg=$("$PY" - "$D" <<'PY' 2>/dev/null || echo n/a
import sys, glob, os, re
cks = sorted(glob.glob(sys.argv[1] + '/checkpoints/*.zip'),
             key=lambda p: int(re.search(r'_(\d+)_steps', os.path.basename(p)).group(1)))
if len(cks) >= 2:
    a, b = cks[-2], cks[-1]
    sa, sb = (int(re.search(r'_(\d+)_steps', os.path.basename(x)).group(1)) for x in (a, b))
    ta, tb = os.path.getmtime(a), os.path.getmtime(b)
    print(f"{(sb-sa)/(tb-ta):.0f}" if tb > ta else "n/a")
else:
    print("n/a")
PY
)
    fi
    el=$(grep -oE 'ep_len_mean *\| *[0-9.]+' "$CLOG" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')
    ck=$(ls "$D"/checkpoints/*.zip 2>/dev/null | wc -l)
    say "ok step=${s:-?} ckpt=${ck} marginal_fps=${marg} ep_len=${el:-?}"
    sleep "$INTERVAL"
done
say "WATCHER EXIT"
