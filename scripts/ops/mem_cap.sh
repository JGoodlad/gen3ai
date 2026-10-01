#!/usr/bin/env bash
#
# mem_cap.sh — run ONE command under a hard memory cap, in its OWN systemd scope inside the shared
# gen3ai-heavy.slice (utils.mem_cap), so a runaway job is OOM-killed ALONE.
#
# WHY: on 2026-09-30 three global OOM kills each took a 74-82 GB python3 — and, because every agent
# and the Claude session shared one tmux scope with OOMPolicy=stop, systemd then tore the whole
# session down. Any exploratory or one-off heavy job (traces, benchmarks, measurement drivers,
# anything that loads many checkpoints) runs under this wrapper.
set -u
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'USAGE_EOF'
mem_cap.sh — run a command under a memory cap in its own scope (see src/utils/mem_cap.py)

USAGE
    scripts/ops/mem_cap.sh [--name N] [--slice-gb G] <GB> <cmd> [args...]
    scripts/ops/mem_cap.sh --status           # the heavy slice and every capped job now

The command runs in gen3ai-capped-<name>-<pid>.scope (MemoryMax=<GB>G, MemorySwapMax=0,
OOMPolicy=stop) inside gen3ai-heavy.slice ($GEN3AI_HEAVY_SLICE), whose aggregate MemoryMax is
declared at every launch (default 64 GB; --slice-gb / $GEN3AI_HEAVY_SLICE_GB), with oom_score_adj
+500. Prints the cap and unit at start; peak memory and whether it was OOM-killed at the end.
A timeout goes INSIDE (mem_cap.sh 24 timeout 2h python ...); composes with gpu_lock.sh either way.
Exit status: the command's own (128+N if signal N killed it); 86 = OOM-killed at its cap or the
slice's; 85 = the cap could not be put in force (the command did NOT run); 2 = usage.
USAGE_EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; echo "REFUSING: mem_cap.sh needs a cap in GB and a command" >&2; exit 2 ;;
esac

# Import the helper from THIS checkout's src/ without touching the child's PYTHONPATH.
exec "$(ops_python)" -c 'import sys; sys.path.insert(0, sys.argv[1]); from utils.mem_cap import main; sys.exit(main(sys.argv[2:]))' \
    "$(ops_repo_root)/src" "$@"
