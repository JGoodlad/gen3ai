#!/usr/bin/env bash
# One measured REAL launch through the launcher (gpu_checks_endstate, 2026-10-09).
#
#   drive.sh <tag> <scratch dir> <stop after N canary-and-update rows> <launcher args...>
#
# Starts `python -m main.launcher <args>` under scripts/ops/gpu_lock.sh (the lease token passes through,
# GEN3AI_GPU_LEASE_TOKEN_FILE must be set), in its own session (setsid) so it has its own process group,
# with a contention sampler beside it. Stops when the run's TensorBoard-backed metrics table has printed
# N per-update tables AND canary_verdicts.jsonl has a row (or the launcher exits), then SIGTERMs the
# LAUNCHER by its explicit pid (the launcher routes SIGTERM to its clean save-and-exit path), waits for
# it, and lists any leftover process of this session group by pid (killed by pid, never by pattern).
set -u
tag=$1; scratch=$2; want=$3; shift 3
repo=$(cd "$(dirname "$0")/../../../../.." && pwd)
py=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
log="$scratch/$tag.launcher.log"
echo "drive $tag start $(date +%s.%N) args: $*" > "$scratch/$tag.drive.log"
cd "$repo" || exit 2
PYTHONPATH="$repo/src" setsid "$repo/scripts/ops/gpu_lock.sh" "$py" -m main.launcher "$@" < /dev/null > "$log" 2>&1 &
wrapper=$!
echo "wrapper pid $wrapper (session/pgid leader)" >> "$scratch/$tag.drive.log"
PYTHONPATH="$repo/src" "$py" "$(dirname "$0")/sampler.py" --root "$wrapper" --out "$scratch/$tag.cpu.jsonl" &
sampler=$!
rundir="${GEN3AI_MODELS_DIR:-/home/goodlad/dev/gen3ai/models}/$tag"   # the run name IS the tag
i=0
while kill -0 "$wrapper" 2>/dev/null; do
    sleep 15; i=$((i+1))
    n=0; c=0
    if [ -f "$rundir/launcher_child.full.log" ]; then
        n=$(grep -c "|    train_ms  " "$rundir/launcher_child.full.log" 2>/dev/null || true)
        [ -f "$rundir/canary_verdicts.jsonl" ] && c=$(wc -l < "$rundir/canary_verdicts.jsonl")
    fi
    if [ "${n:-0}" -ge "$want" ] && [ "${c:-0}" -ge "${DRIVE_MIN_CANARY:-1}" ]; then
        echo "reached $n update tables, $c canary verdict(s) at i=$i" >> "$scratch/$tag.drive.log"
        break
    fi
    if [ "$i" -ge 220 ]; then echo "drive cap (55 min) at n=$n c=$c" >> "$scratch/$tag.drive.log"; break; fi
done
# the launcher is the gpu_lock wrapper's child: find it by parent pid, signal it by explicit pid
launcher=$(ps -o pid= --ppid "$wrapper" | head -1 | tr -d ' ')
echo "launcher pid ${launcher:-none}; SIGTERM at $(date +%s.%N)" >> "$scratch/$tag.drive.log"
[ -n "$launcher" ] && kill -TERM "$launcher" 2>/dev/null
for _ in $(seq 1 60); do kill -0 "$wrapper" 2>/dev/null || break; sleep 5; done
echo "end $(date +%s.%N); wrapper alive: $(kill -0 "$wrapper" 2>/dev/null && echo yes || echo no)" >> "$scratch/$tag.drive.log"
echo "leftover in session $wrapper:" >> "$scratch/$tag.drive.log"
ps -o pid=,pgid=,etime=,args= -s "$wrapper" | cut -c1-200 >> "$scratch/$tag.drive.log"
kill -0 "$sampler" 2>/dev/null && kill "$sampler"
