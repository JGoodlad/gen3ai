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
    --launcher-log PATH  the LAUNCHER's own log. Scanned for the arm-INVALIDATING
                         `--sync-to-main` line, AND it is the AUTHORITY on crashes and the
                         run's end (below). Optional; its absence is reported once, the
                         sync-to-main check CANNOT fire, and the watcher falls back to the
                         old rule (the first child-log error text ends it) — an absent file
                         is not a clean one.
    --wedge-seconds N    no step progress for this long = WEDGED (default 2100 = 35 min).
    --interval N         seconds between ticks (default 300).
    --dir-wait-seconds N with --pid-file and a bare run NAME: if the run directory does not
                         exist yet, WAIT for it (up to N s, default 900) while the launcher pid
                         is alive, instead of refusing. A launch creates the dir only once its
                         child starts, so a watcher started beside the launch used to refuse and
                         leave the arm unwatched. A dir that never appears is still a refusal.

WHAT IT WRITES
    one line per tick:  [<timestamp>] ok step=<n> ckpt=<n> marginal_fps=<n> ep_len=<n>
                        (+ crashes=<n> when a launcher log is readable)
    and on the way out: FAILURE: <reason>  or  DONE: <reason>,  then  WATCHER EXIT

RESUME-AWARE (with --launcher-log)
    The launcher RESTARTS a crashed child from its checkpoint by itself, and the child log
    RESTARTS with it (a new incarnation, its old traceback gone). So the watcher does NOT
    exit on a child-log error line; it reports it (`ERROR: ...`) and lets the LAUNCHER LOG
    decide. Crash counts come from the launcher log, never the child log:
        CRASH: ...    a `Child crashed (exit N) ... crash #K` line - an event, watching goes on
        RESUMED: ...  an `Auto-restart #N after crash` line - the wedge clock restarts
        DONE: ...     the launcher printed `Training complete`, or a final_model.zip newer than
                      the watcher appeared, or the launcher pid is gone after `Training complete`
        FAILURE: ...  the launcher will NOT restart (`will NOT restart`, `giving up`,
                      `cannot restart`), the launcher pid is gone without a completion line,
                      the arm is voided (`--sync-to-main`), or the run is WEDGED
    A Monitor filter wants all of CRASH|RESUMED|DONE|FAILURE|WATCHER EXIT.

ENVIRONMENT
    GEN3AI_MODELS_DIR  the run archive, if it is not the main checkout's models/
    GEN3AI_PYTHON      interpreter for the marginal-fps helper

EXIT
    0  the watcher exited after reporting a terminal condition (DONE or FAILURE)     2  bad usage
EOF
}

case "${1:-}" in
    -h|--help|"") usage; [ -z "${1:-}" ] && exit 2 || exit 0 ;;
esac

# shellcheck source=scripts/ops/_common.sh
. "$(dirname "$(readlink -f "$0")")/_common.sh"

RUN_ARG="$1"; shift
PIDF=""; NO_PID=0; STATUS=""; LAUNCHER_LOG=""; WEDGE=2100; INTERVAL=300; DIR_WAIT=900
while [ $# -gt 0 ]; do
    case "$1" in
        --pid-file)       PIDF="$2"; shift 2 ;;
        --no-pid-check)   NO_PID=1; shift ;;
        --status-file)    STATUS="$2"; shift 2 ;;
        --launcher-log)   LAUNCHER_LOG="$2"; shift 2 ;;
        --wedge-seconds)  WEDGE="$2"; shift 2 ;;
        --interval)       INTERVAL="$2"; shift 2 ;;
        --dir-wait-seconds) DIR_WAIT="$2"; shift 2 ;;
        *) echo "REFUSING: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$NO_PID" -eq 0 ] && [ -z "$PIDF" ]; then
    echo "REFUSING: --pid-file is required (or pass --no-pid-check to watch progress only)." >&2
    exit 2
fi

# A watcher started beside a launch runs BEFORE the launcher's child has created models/<run>/.
# With a pid file and a bare NAME, wait for the directory while the launcher lives (2026-09-28:
# the T32 arm's watcher refused at start and the arm ran unwatched). ops_resolve_run below still
# refuses a directory that never appeared, with its reason.
case "$RUN_ARG" in
    */*|.*) ;;
    *)
        if [ "$NO_PID" -eq 0 ] && m_dir="$(ops_models_dir 2>/dev/null)" && [ ! -d "$m_dir/$RUN_ARG" ]; then
            waited=0
            while [ ! -d "$m_dir/$RUN_ARG" ] && [ "$waited" -lt "$DIR_WAIT" ]; do
                kill -0 "$(cat "$PIDF" 2>/dev/null || echo 0)" 2>/dev/null || break
                sleep 5; waited=$((waited + 5))
            done
        fi
        ;;
esac

D="$(ops_resolve_run "$RUN_ARG")" || exit 2
CLOG="$D/launcher_child.log"
STATUS="${STATUS:-$D/watch_status.txt}"
PY="$(ops_python)"

# FAILURE lines of the child log, ANCHORED to what the trainer really prints (2026-10-06: the old
# case-insensitive bare `FATAL|Traceback` false-fired on every current run: Python's harmless
# "Enable tracemalloc to get the object allocation traceback" warning, the `[CompileCanary] armed`
# banner ("... before it FATALs") and the `[SUPPLY] ... else FATAL_SUPPLY (5)` banner).
#   * `^Traceback (most recent call last)` - a real traceback starts a line, case-sensitive
#   * `] FATAL`  - every typed fatal is `[Tag] FATAL` (`[ModelVersion]`, `[SUPPLY]`, `[CompileSentinel]`,
#                  `[CompileCanary]`, `[Learner]`, `[LearnerLifecycle]`, `[Resume]`, `[Recipe]`,
#                  `[TorchFloor]`, `[DesktopGpu]`, `[RunArchive]`, ...); no armed/announce banner puts
#                  a bracket tag directly before FATAL
#   * `FATAL ERROR DETECTED` - train_rl_agent.py's fail-fast banner
#   * CUDA / host out-of-memory
# Pinned by src/main/ops/watch_run_failure_grep_test.py (banners must NOT fire, real lines must).
FAIL_RE='^[[:space:]]*Traceback \(most recent call last\)|\] FATAL|FATAL ERROR DETECTED|CUDA out of memory|OutOfMemoryError'

say() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }

# --- the LAUNCHER-LOG half (resume-aware) ---------------------------------------------------
# Patterns are the launcher's own event text (src/main/launcher/run.py), emoji-free so a
# change of glyph does not blind them. Pinned by src/main/ops/watch_run_resume_test.py.
CRASH_RE='Child crashed \(exit'
RESUME_RE='Auto-restart #[0-9]+ after crash'
DONE_RE='Training complete — all steps done'
GIVEUP_RE='will NOT restart|rapid crashes in a row|cannot restart'
lcount() { local c; c=$(grep -cE "$1" "$LAUNCHER_LOG" 2>/dev/null); echo "${c:-0}"; }
have_launcher() { [ -n "$LAUNCHER_LOG" ] && [ -f "$LAUNCHER_LOG" ]; }
START_EPOCH=$(date +%s)
seen_crash=0; seen_resume=0; first_pass=1; first_pass_err=0; err_seen=0; n_crash=0

# Emit each NEW launcher-log crash / resume line; return 0 (and set TERMINAL=DONE|FAILURE and
# TERMINAL_MSG) when the log says the run has ended.
TERMINAL=""; TERMINAL_MSG=""
launcher_events() {
    have_launcher || return 0
    local n line note=""
    [ "$first_pass" -eq 1 ] && note=" (already in the launcher log when the watcher started)"
    n=$(lcount "$CRASH_RE")
    if [ "$n" -gt "$seen_crash" ]; then
        while IFS= read -r line; do
            say "CRASH: launcher log: ${line:0:200} — crashes in the launcher log: ${n};${note} the launcher handles a crash itself, still watching"
        done < <(grep -E "$CRASH_RE" "$LAUNCHER_LOG" | tail -n +$((seen_crash + 1)))
        seen_crash=$n; last_move=$(date +%s)
    fi
    n_crash=$n
    n=$(lcount "$RESUME_RE")
    if [ "$n" -gt "$seen_resume" ]; then
        while IFS= read -r line; do
            say "RESUMED: launcher log: ${line:0:200}${note}"
        done < <(grep -E "$RESUME_RE" "$LAUNCHER_LOG" | tail -n +$((seen_resume + 1)))
        seen_resume=$n; last_move=$(date +%s); last_step=-1; err_seen=0   # a new child incarnation
    fi
    if [ "$(lcount "$GIVEUP_RE")" -gt 0 ]; then
        TERMINAL=FAILURE
        TERMINAL_MSG="the launcher will not restart — $(grep -E "$GIVEUP_RE" "$LAUNCHER_LOG" | tail -1 | cut -c1-200)"
    elif [ "$(lcount "$DONE_RE")" -gt 0 ]; then
        TERMINAL=DONE; TERMINAL_MSG="the launcher reports training complete (${n_crash} crash(es) auto-resumed)"
    fi
    first_pass=0
}

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
    TERMINAL=""; TERMINAL_MSG=""
    launcher_events
    if [ -z "$TERMINAL" ] && [ -f "$D/final_model.zip" ] \
            && [ "$(stat -c %Y "$D/final_model.zip" 2>/dev/null || echo 0)" -ge "$START_EPOCH" ]; then
        TERMINAL=DONE; TERMINAL_MSG="final_model.zip written since the watcher started"
    fi
    if [ "$TERMINAL" = DONE ]; then say "DONE: $TERMINAL_MSG"; break; fi
    if [ "$TERMINAL" = FAILURE ]; then say "FAILURE: $TERMINAL_MSG"; break; fi
    if [ "$NO_PID" -eq 0 ]; then
        pid=$(cat "$PIDF" 2>/dev/null || echo 0)
        if ! kill -0 "$pid" 2>/dev/null; then
            say "FAILURE: launcher pid $pid GONE (no completion line in the launcher log)"; break
        fi
    fi
    # FAILURE words, not just progress. With a readable launcher log the LAUNCHER decides: it
    # restarts a crashed child from its checkpoint and the child log restarts with it, so a
    # traceback here is an EVENT, not the end (2026-10-08: the first FAILURE line ended the
    # watcher and a launcher crash-resume then ran unwatched). Without one, the old rule holds.
    if have_launcher; then
        nerr=$(grep -cE "$FAIL_RE" "$CLOG" 2>/dev/null || true); nerr=${nerr:-0}
        if [ "$first_pass_err" != 1 ]; then
            first_pass_err=1
            [ "$nerr" -gt 0 ] && say "WARN: ${nerr} error line(s) already in the child log at watcher start - ignored (the launcher log decides)"
            err_seen=$nerr
        elif [ "$nerr" -gt "$err_seen" ]; then
            say "ERROR: error text in child log (${nerr} this incarnation, crashes in the launcher log: ${n_crash}) — $(grep -oE "$FAIL_RE" "$CLOG" | tail -1); the launcher log decides, still watching"
            err_seen=$nerr
        elif [ "$nerr" -lt "$err_seen" ]; then
            err_seen=$nerr          # the child log restarted
        fi
    elif grep -qE "$FAIL_RE" "$CLOG" 2>/dev/null; then
        say "FAILURE: error text in child log — $(grep -oE "$FAIL_RE" "$CLOG" | tail -1)"
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
    cr=""; have_launcher && cr=" crashes=${n_crash}"
    say "ok step=${s:-?} ckpt=${ck} marginal_fps=${marg} ep_len=${el:-?}${cr}"
    sleep "$INTERVAL"
done
say "WATCHER EXIT"
