#!/usr/bin/env bash
#
# gate_lock.sh — run ONE command holding one of the box's N TEST-GATE slots (utils.gate_lock).
#
# At most N (declared: 2; $GEN3AI_GATE_SLOTS overrides) routine gates run at once; the rest wait.
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
    scripts/ops/gate_lock.sh --status          # who holds each slot now

Slots are ~/.claude/jobs/gate_slots/slot<i>.lock ($GEN3AI_GATE_LOCK_DIR overrides). The wait and the
acquisition time are logged to stderr; while waiting, the holders are printed every 60 s.
--timeout-s bounds the WAIT; a wall timeout goes INSIDE: gate_lock.sh timeout 3000 <cmd> (never
timeout 3000 gate_lock.sh <cmd>, which counts queue time).
Exit status: the command's own (128+N if signal N killed it); 2 = usage; 3 = self-deadlock;
4 = --timeout-s expired.
USAGE_EOF
}

opts=()
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; echo "REFUSING: gate_lock.sh needs a command to run" >&2; exit 2 ;;
    --status) set -- --status ;;
    *)
        while [ "${1:-}" = "--timeout-s" ]; do opts+=("$1" "${2:-}"); shift 2; done
        set -- "${opts[@]}" -- "$@" ;;
esac

exec "$(ops_python)" -c 'import sys; sys.path.insert(0, sys.argv[1]); from utils.gate_lock import main; sys.exit(main(sys.argv[2:]))' \
    "$(ops_repo_root)/src" "$@"
