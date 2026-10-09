#!/usr/bin/env bash
# Capture + probe every checkpoint in plan.txt, ONE at a time (one heavy job), resumable: a label whose
# probe JSON exists is skipped. CPU only, nice 19, 4 threads.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
A=/home/goodlad/gen3ai_archive/static_diag_2026-10-09
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
while read -r zip label; do
  [ -z "$label" ] && continue
  if [ -f "$A/probe/$label.json" ]; then echo "skip $label"; continue; fi
  "$HERE/run_at_pin.sh" capture.py --ckpt "$zip" --label "$label" 2>&1 | grep -E '^\[cap\]|Error|error' || true
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 19 timeout 1800 "$PY" -I "$HERE/probe.py" "$label" || echo "PROBE FAIL $label"
done < "$A/plan.txt"
echo ALL_DONE
