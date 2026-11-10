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
    scripts/ops/mem_cap.sh [--name N] [--slice-gb G] [--term-grace S] [--slot] [--no-wait] [--blocking WHY] [--max-s S] [--max-run-s S] <GB> <cmd> [args...]
    scripts/ops/mem_cap.sh --status           # the heavy slice, every capped job, the HEAVY SLOTS

HEAVY SLOTS (rule 5b, utils/heavy_slots.py): while a training run is live or the GPU is leased (or
with --slot) the job first takes one of the box's heavy slots, waiting with a visible line; --no-wait
refuses at once (exit 87). ADMISSION IS BY MEMORY (utils/heavy_admission.py): the job runs while the caps of
the running jobs plus its own <GB> fit the box's memory less a reserve, at most 4 at once; so declare the cap
the job needs, not a habit. `python -m utils.heavy_slots --admission count` is the fixed count of 2 again. A job the orchestrator marks BLOCKING ($GEN3AI_BLOCKING=<why> or --blocking
<why>) bypasses them; a job inside a slot holder (the routine gate holds one) takes none. Lock order:
gate_lock.sh OUTSIDE mem_cap.sh.

SHORT LANE: a job of at most 10 minutes DECLARES it with --max-s S (0 < S <= 600; or
$GEN3AI_HEAVY_SLOT_MAX_S=S) so it need not wait out the 40-45-minute chunks ahead of it. It queues in the
short lane (FIFO among shorts, FIFO among longs) and may jump ahead of a queued LONG job, at most 3 times
per long job. The price is a HARD TIMEOUT of S + 30 s (SIGTERM, then SIGKILL after --term-grace, which
a short job caps at 60 s). An overrun prints "SHORT-LANE OVERRUN ... re-queue it as a LONG job" and exits
88: its work is incomplete; re-run it WITHOUT --max-s, in bounded chunks (<= 45 min). The deadline binds
only a job that took a slot of its own: not one that needed none, bypassed (BLOCKING, gpu_lock) or runs
inside a holder. A declaration above 600 s, or --term-grace above 60 s, is refused (exit 2).

RUN-TIME BOUND: --max-run-s S (S > 0, any lane) stops the job S seconds after ITS COMMAND STARTS: the wait
for a heavy slot (or the gate lock) is never counted. SIGTERM, then SIGKILL of the whole scope after
--term-grace; prints "RUN-TIME BOUND ..." and exits 89. NEVER put `timeout N` OUTSIDE a job that may
queue: it counts the queue wait, fires while the job is still waiting and loses its place. Use this.

The command runs in gen3ai-capped-<name>-<pid>.scope (MemoryMax=<GB>G, MemorySwapMax=0,
OOMPolicy=stop) inside gen3ai-heavy.slice ($GEN3AI_HEAVY_SLICE), whose aggregate MemoryMax is
declared at every launch (default 64 GB; --slice-gb / $GEN3AI_HEAVY_SLICE_GB), with oom_score_adj
+500. Prints the cap and unit at start; peak memory and whether it was OOM-killed at the end.
STOPPING: bound the run with --max-run-s (above), never an outer `timeout` (it counts the slot wait). A
stop signal reaching the wrapper stops the job: SIGINT/SIGTERM/SIGHUP are forwarded to the command, and --term-grace S seconds (default 60;
$GEN3AI_MEM_CAP_TERM_GRACE_S) after the first one everything left in the scope is SIGKILLed, a setsid-ed
worker too. The wrapper dying by SIGKILL (timeout -k, kill -9) stops the job the same way. Exit 143 on
SIGTERM, 137 once the grace killed it. Composes with gpu_lock.sh either way.
Exit status: the command's own (128+N if signal N killed it); 86 = OOM-killed at its cap or the
slice's; 85 = the cap could not be put in force (the command did NOT run); 87 = --no-wait and every
heavy slot is held (the command did NOT run); 88 = a declared-short job overran its hard timeout and
was killed; 89 = --max-run-s stopped it; 2 = usage.
USAGE_EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; echo "REFUSING: mem_cap.sh needs a cap in GB and a command" >&2; exit 2 ;;
esac

# Import the helper from THIS checkout's src/ without touching the child's PYTHONPATH.
exec "$(ops_python)" -c 'import sys; sys.path.insert(0, sys.argv[1]); from utils.mem_cap import main; sys.exit(main(sys.argv[2:]))' \
    "$(ops_repo_root)/src" "$@"
