#!/usr/bin/env bash
#
# gate_lock.sh — run ONE command holding one of the box's N TEST-GATE slots (utils.gate_lock).
#
# At most N (declared: 1; $GEN3AI_GATE_SLOTS overrides) routine gates run at once; the rest wait.
# One at a time is the measured policy (designs/ops/testing.md). A live training run gets a one-line
# rule-5b warning — nice 19, the agent cores, the measured -n (utils.gate_lock.live_run_warning) —
# never an action.
# 2026-09-30: seven concurrent `pytest -n 2` gates put load ~36 on 8 cores / 16 threads and every
# gate crawled. The command runs as a CHILD of the slot holder with GEN3AI_GATE_LOCK_HELD exported,
# so a gate_lock take inside it is a verified no-op (never a second slot, never a self-deadlock).
set -u
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'USAGE_EOF'
gate_lock.sh — run a command while holding one of N test-gate slots (re-entrant; utils/gate_lock.py)

USAGE
    scripts/ops/gate_lock.sh [--timeout-s S] <cmd> [args...]
    scripts/ops/gate_lock.sh --status [--json] [--sample-s S]
                         # who holds each slot, since when, its ETA (the median wall of past gates of
                         # the same -n on the same cpu count), its tree's CPU, and who is queued

Slots are ~/.claude/jobs/gate_slots/slot<i>.lock ($GEN3AI_GATE_LOCK_DIR overrides). The wait and the
acquisition time are logged to stderr; while waiting, the holders are printed every 60 s.
--timeout-s bounds the WAIT; a wall timeout goes INSIDE: gate_lock.sh timeout 3000 <cmd> (never
timeout 3000 gate_lock.sh <cmd>, which counts queue time).
Exit status: the command's own (128+N if signal N killed it); 2 = usage; 3 = self-deadlock;
4 = --timeout-s expired; 5 = lock order (an ancestor holds a heavy slot and the gate is busy: put
gate_lock.sh OUTSIDE mem_cap.sh). While a training run is live the gate also holds one HEAVY slot
(utils/heavy_slots.py; scripts/ops/mem_cap.sh --status).
USAGE_EOF
}

opts=()
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; echo "REFUSING: gate_lock.sh needs a command to run" >&2; exit 2 ;;
    --status) ;;                     # --status [--json] [--sample-s S]: passed through as given
    *)
        while [ "${1:-}" = "--timeout-s" ]; do opts+=("$1" "${2:-}"); shift 2; done
        set -- "${opts[@]}" -- "$@" ;;
esac

exec "$(ops_python)" -c 'import sys; sys.path.insert(0, sys.argv[1]); from utils.gate_lock import main; sys.exit(main(sys.argv[2:]))' \
    "$(ops_repo_root)/src" "$@"
