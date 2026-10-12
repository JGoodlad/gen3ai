#!/usr/bin/env bash
# usage: run.sh <pin_name> <run>
set -u
PIN=/home/goodlad/.cache/gen3ai/vdp/pin_$1
RUN=/home/goodlad/dev/gen3ai/models/$2
OUT=/home/goodlad/gen3ai_archive/policy_drift/$2
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH=$PIN/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$PIN"
WRAP="nice -n 19 /home/goodlad/dev/gen3ai/scripts/ops/mem_cap.sh --name hib_drift 8"
echo "=== collect $2 $(date)"
$WRAP timeout 7200 taskset -c 5-7,13-15 "$PY" /tmp/hib_drift/drift_entry.py collect "$RUN/snapshots/snapshot_000004000000.zip" --battles 40 --seed-base 0 --threads 1 || exit 1
echo "=== watch $2 $(date)"
$WRAP timeout 10800 taskset -c 5-7,13-15 "$PY" -m main.policy_drift watch "$RUN" --source both --once --threads 1 --nice 19 --device cpu || exit 2
echo "=== report $2 $(date)"
"$PY" -m main.policy_drift report "$2" > "$OUT/report.txt" 2>&1
echo "=== done $2 $(date)"
