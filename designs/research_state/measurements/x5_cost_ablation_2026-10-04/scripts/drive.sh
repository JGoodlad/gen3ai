#!/usr/bin/env bash
# One measured launch (X5 cost ablation, 2026-10-04): the CPU sampler, nvidia-smi at 2 Hz, the trainer
# through the bottleneck profile's phase-probe wrapper (the trainer's own main(), unchanged; the
# script's name never appears in the argv), stopped by SIGTERM to its explicit PID after N update rows.
#   drive.sh <tag> <n_update_rows> <trainer args...>
# env: S (scratch dir), W (worktree), PY, GEN3AI_GPU_LEASE_TOKEN_FILE
set -u
tag=$1; nupd=$2; shift 2
cd "$W"
export PYTHONPATH=$W/src GEN3AI_MODELS_DIR=$S/models PROF_PHASE_LOG=$S/$tag.phases.jsonl
echo "drive $tag start $(date +%s.%N) args: $*" > "$S/$tag.drive.log"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 500 > "$S/$tag.smi.csv" 2>/dev/null &
smi=$!
scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 48 timeout 3000 "$PY" "$S/phase_probe_run.py" \
    --run-name "$tag" "$@" > "$S/$tag.log" 2>&1 &
wrap=$!
echo "wrapper pid $wrap" >> "$S/$tag.drive.log"
tp=""
for i in $(seq 1 60); do
  tp=$(pgrep -f "^$PY $S/[p]hase_probe_run.py --run-name $tag " | head -1); [ -n "$tp" ] && break; sleep 2
done
echo "trainer pid $tp" >> "$S/$tag.drive.log"
"$PY" "$S/cpu_sampler.py" --root-cmd-match "phase_probe_run.py --run-name $tag " --out "$S/$tag.cpu.jsonl" --interval 2 --max-hours 1.2 > /dev/null 2>&1 &
samp=$!
for i in $(seq 1 900); do        # <= 45 min
  n=$(cat "$S/$tag.phases.jsonl" 2>/dev/null | grep -c '"phase": "update"')
  if [ "$n" -ge "$nupd" ]; then echo "reached $n update rows at i=$i" >> "$S/$tag.drive.log"; break; fi
  kill -0 "$tp" 2>/dev/null || { echo "trainer died at i=$i (n=$n)" >> "$S/$tag.drive.log"; break; }
  sleep 3
done
desc() { local kids; kids=$(pgrep -P "$1"); for k in $kids; do echo "$k"; desc "$k"; done; }
pre=$(desc "$tp")                 # the trainer's descendants BEFORE the stop (F-G-10: detached children outlive it)
if kill -0 "$tp" 2>/dev/null; then echo "SIGTERM -> $tp at $(date +%s.%N)" >> "$S/$tag.drive.log"; kill -TERM "$tp"; fi
for i in $(seq 1 60); do kill -0 "$wrap" 2>/dev/null || break; sleep 2; done
kill -0 "$wrap" 2>/dev/null && { echo "wrapper still alive; SIGKILL trainer $tp" >> "$S/$tag.drive.log"; kill -KILL "$tp" 2>/dev/null; }
kill "$smi" 2>/dev/null; kill "$samp" 2>/dev/null
for p in $pre $(pgrep -f "[s]napshot_ladder.*$S/models/$tag"); do
  kill -0 "$p" 2>/dev/null || continue
  echo "orphan $p: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | head -c 200)" >> "$S/$tag.drive.log"; kill -TERM "$p" 2>/dev/null
done
echo "end $(date +%s.%N); wrapper alive: $(kill -0 $wrap 2>/dev/null && echo yes || echo no)" >> "$S/$tag.drive.log"
