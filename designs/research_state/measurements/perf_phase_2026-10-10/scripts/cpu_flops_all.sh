#!/usr/bin/env bash
# CPU only: the analytic FLOP read + one eager CPU micro-step wall for every attribution config (configs.py).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
for c in P S SF R E EK; do
  CUDA_VISIBLE_DEVICES= timeout 900 "$ROOT/scripts/ops/mem_cap.sh" 12 nice -n 10 "$PY" "$HERE/cpu_flops.py" \
      --cfg "$c" --rows 256 --out "$HERE/../results/cpu_flops_$c.json" 2>&1 | grep -E '"wall_s"|"GFLOP"' | tr '\n' ' '
  echo " <- $c"
done
