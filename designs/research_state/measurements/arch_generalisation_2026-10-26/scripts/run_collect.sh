#!/usr/bin/env bash
# Collect the four panels in ONE short-lane capped job (rule 5b: nice 19 on the agent cores), then analyse them in a
# second one. Each stage re-runs only what is missing.
set -u
cd "$(dirname "$0")/../../../../.."
D=designs/research_state/measurements/arch_generalisation_2026-10-26/scripts
O=designs/research_state/measurements/arch_generalisation_2026-10-26
export PYTHONPATH=$PWD/src:$PWD/$D CUDA_VISIBLE_DEVICES="" PROBE_THREADS=3
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
scripts/ops/mem_cap.sh --name argen-collect --max-s 600 6 nice -n 19 taskset -c 5-7,13-15 bash -c "
seed=2610
for p in home72 F H P72; do seed=\$((seed + 1)); [ -f $O/rows_\$p.npz ] || $PY $D/collect.py \$p 48 \$seed; done" 2>&1 | grep -v Warning
scripts/ops/mem_cap.sh --name argen-analyse --max-s 600 6 nice -n 19 taskset -c 5-7,13-15 bash -c "
for p in home72 F H P72; do [ -f $O/rows_\${p}_analysis.json ] || $PY $D/analyze_rows.py \$p 640 > /dev/null; echo analysed \$p; done" 2>&1 | grep -v Warning
