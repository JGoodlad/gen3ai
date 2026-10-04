#!/usr/bin/env bash
#
# gpu_lock.sh — run ONE command holding the box's GPU lock, RE-ENTRANTLY and FAIL-FAST (utils.gpu_lock).
#
# The GPU is LEASED for an agent's lifetime (scripts/ops/gpu_lease.sh); this wrapper NEVER waits by
# default: a lease held by someone else exits 6, a one-off holder exits 5, both AT ONCE. The caller's
# GEN3AI_GPU_LEASE_TOKEN (or ..._TOKEN_FILE) matching the lease passes straight through.
#
# Replaces a bare `flock ~/.claude/jobs/gpu.lock <cmd>`. The difference is the class fix for the
# 2026-09-30 self-deadlock: a bare flock exports nothing, so a command that takes the lock itself
# (the deleted main.rust_core_m5 harness's gates --gpu, a cuda throughput probe, policy_spectrum truth --lock) opened
# the file again and waited forever on its own ancestor. Under this wrapper the child inherits
# GEN3AI_GPU_LOCK_HELD=<holder pid>, which utils.gpu_lock VERIFIES (ancestor + /proc/locks) and
# then does not re-acquire.
set -u
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'USAGE_EOF'
gpu_lock.sh — run a command while holding the box's GPU lock (re-entrant, FAIL-FAST; see utils/gpu_lock.py)

USAGE
    scripts/ops/gpu_lock.sh <cmd> [args...]
    scripts/ops/gpu_lock.sh --wait <cmd>      # DELIBERATE kernel wait on a one-off holder (orchestrator / training launch only)
    scripts/ops/gpu_lock.sh --status          # LEASED / BUSY / FREE, with owner or pid + command line
    scripts/ops/gpu_lock.sh timeout 3000 <cmd>  # a WALL TIMEOUT goes INSIDE (outside, it counts lock-wait)

The lock is ~/.claude/jobs/gpu.lock ($GEN3AI_GPU_LOCK overrides). The command runs as a CHILD of
the lock holder with GEN3AI_GPU_LOCK_HELD=<holder pid> exported, so any take of the same lock inside
it is a verified no-op. If the caller's GEN3AI_GPU_LEASE_TOKEN owns the lease the command runs without
taking the flock. Otherwise: a lease held by someone else = exit 6 at once (even with --wait); a one-off
holder = exit 5 at once unless --wait (then the holder's pid and command line are printed every 60 s); a
holder that is this command's own ANCESTOR (a bare `flock` around it) is refused at once (exit 3).
Exit status: the command's own (128+N if signal N killed it); 2 = no command; 3 = self-deadlock;
4 = --timeout-s expired; 5 = GPU busy (one-off holder); 6 = GPU leased to someone else.
USAGE_EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; echo "REFUSING: gpu_lock.sh needs a command to run" >&2; exit 2 ;;
    --status) set -- --status ;;
    --wait) shift; set -- --wait -- "$@" ;;
    *) set -- -- "$@" ;;
esac

# Import the helper from THIS checkout's src/ without touching the child's PYTHONPATH (a pinned
# run's import precedence is PYTHONPATH order — the wrapper must not reorder it).
exec "$(ops_python)" -c 'import sys; sys.path.insert(0, sys.argv[1]); from utils.gpu_lock import main; sys.exit(main(sys.argv[2:]))' \
    "$(ops_repo_root)/src" "$@"
