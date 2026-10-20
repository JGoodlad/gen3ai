#!/bin/bash
# The whole L1 + L5 read (PREREG §7): the cache, the L1 read, the L5 read. CPU only, light (measured 1.3 GB, 31 s).
#   nice -n 19 taskset -c 5-7,13-15 bash -c 'ulimit -v 8000000; .../scripts/run_all.sh'
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
D=$(cd "$(dirname "$0")" && pwd)
"$PY" -I "$D/l1_read.py" selfcheck
"$PY" -I "$D/l1l5_data.py" build
"$PY" -I "$D/l1_read.py" read
"$PY" -I "$D/l5_props.py" > /dev/null
echo "L1L5 DONE"
