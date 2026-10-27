#!/usr/bin/env bash
# The whole N2 identity test, CPU only, rule 5b's shape (nice 19 on the agent cores, mem_cap-capped). Read-only on models/.
# Runs with MAIN's src on PYTHONPATH (its release Rust build passes its stamp there; see README F-N2-H1).
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
SRC=${N2_SRC:-/home/goodlad/dev/gen3ai/src}
export PYTHONPATH=$SRC:$D CUDA_VISIBLE_DEVICES="" PROBE_THREADS=3
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
CAP=/home/goodlad/dev/gen3ai/scripts/ops/mem_cap.sh
for step in ${*:-build_draws probe ablate decide}; do
  echo "== $step"
  "$CAP" --name "n2-$step" --max-s 600 8 nice -n 19 taskset -c 5-7,13-15 "$PY" "$D/$step.py" 2>&1 | grep -v Warning
done
