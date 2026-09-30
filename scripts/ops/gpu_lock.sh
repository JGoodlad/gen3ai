#!/usr/bin/env bash
#
# gpu_lock.sh — run ONE command holding the box's GPU lock, RE-ENTRANTLY (utils.gpu_lock).
#
# Replaces a bare `flock ~/.claude/jobs/gpu.lock <cmd>`. The difference is the class fix for the
# 2026-09-30 self-deadlock: a bare flock exports nothing, so a command that takes the lock itself
# (main.rust_core_m5 gates --gpu, throughput/fanout on cuda, policy_spectrum truth --lock) opened
# the file again and waited forever on its own ancestor. Under this wrapper the child inherits
# GEN3AI_GPU_LOCK_HELD=<holder pid>, which utils.gpu_lock VERIFIES (ancestor + /proc/locks) and
# then does not re-acquire.
set -u
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'USAGE_EOF'
gpu_lock.sh — run a command while holding the box's GPU lock (re-entrant; see utils/gpu_lock.py)

USAGE
    scripts/ops/gpu_lock.sh <cmd> [args...]
    scripts/ops/gpu_lock.sh --status          # who holds it now (pid + command line)

The lock is ~/.claude/jobs/gpu.lock ($GEN3AI_GPU_LOCK overrides). The command runs as a CHILD of
the lock holder with GEN3AI_GPU_LOCK_HELD=<holder pid> exported, so any take of the same lock inside
it is a verified no-op. While waiting, the holder's pid and command line are printed every 60 s; a
holder that is this command's own ANCESTOR (a bare `flock` around it) is refused at once (exit 3).
Exit status: the command's own (128+N if signal N killed it); 2 = no command; 3 = self-deadlock.
USAGE_EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; echo "REFUSING: gpu_lock.sh needs a command to run" >&2; exit 2 ;;
    --status) set -- --status ;;
    *) set -- -- "$@" ;;
esac

# Import the helper from THIS checkout's src/ without touching the child's PYTHONPATH (a pinned
# run's import precedence is PYTHONPATH order — the wrapper must not reorder it).
exec "$(ops_python)" -c 'import sys; sys.path.insert(0, sys.argv[1]); from utils.gpu_lock import main; sys.exit(main(sys.argv[2:]))' \
    "$(ops_repo_root)/src" "$@"
