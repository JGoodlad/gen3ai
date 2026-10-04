#!/usr/bin/env bash
#
# gpu_lease.sh — the GPU LEASE: one agent/session owns the GPU for its whole lifetime; nobody waits.
# A thin wrapper over src/utils/gpu_lease.py (see its docstring for the mechanism and the WHY).
set -u
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'USAGE_EOF'
gpu_lease.sh — lease the GPU for an agent's whole lifetime (nobody waits; see utils/gpu_lease.py)

USAGE
    scripts/ops/gpu_lease.sh acquire --owner <name> [--note <text>] [--max-hours H] [--watch-pid N] [--token-file F]
    scripts/ops/gpu_lease.sh status
    scripts/ops/gpu_lease.sh release [--force]

acquire   takes the lease NOW or fails AT ONCE naming the holder (lease owner, pid, command, since when):
          exit 6 = leased to someone else, 5 = a one-off hold is in progress. On success stdout is
          `export GEN3AI_GPU_LEASE_TOKEN=<token>` (the human line is on stderr). A detached holder keeps the
          GPU's flock until `release`, SIGTERM, --max-hours (default 12; 0 = none) or --watch-pid's death.
          Re-running it with your token renews the expiry. An agent's Bash tool does not keep exports between
          calls: pass --token-file F once and put GEN3AI_GPU_LEASE_TOKEN_FILE=F in front of each GPU call.
status    LEASED / BUSY / FREE; a stale lease record (dead holder, reused pid) is detected and cleared.
release   idempotent; needs your token in the environment, or --force (the ORCHESTRATOR's, for a crashed
          owner); exit 7 = not your lease. Returns only after the GPU is free.

Every GPU command then runs as `scripts/ops/gpu_lock.sh <cmd>` with the token in the environment: it passes
straight through. Without the token it is refused at once (exit 6), never queued.
The lock is ~/.claude/jobs/gpu.lock ($GEN3AI_GPU_LOCK overrides); the record sits beside it as gpu.lock.lease.json.
USAGE_EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; echo "REFUSING: gpu_lease.sh needs a command (acquire | release | status)" >&2; exit 2 ;;
esac

exec "$(ops_python)" -c 'import sys; sys.path.insert(0, sys.argv[1]); from utils.gpu_lease import main; sys.exit(main(sys.argv[2:]))' \
    "$(ops_repo_root)/src" "$@"
