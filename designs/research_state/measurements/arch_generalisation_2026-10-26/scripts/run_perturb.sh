#!/usr/bin/env bash
# The perturbation probe, one short-lane capped job (rule 5b), once the home72 rows exist.
set -u
cd "$(dirname "$0")/../../../../.."
D=designs/research_state/measurements/arch_generalisation_2026-10-26/scripts
O=designs/research_state/measurements/arch_generalisation_2026-10-26
export PYTHONPATH=$PWD/src:$PWD/$D CUDA_VISIBLE_DEVICES="" PROBE_THREADS=3
timeout 3300 bash -c "until [ -f $O/rows_home72.npz ]; do sleep 20; done" || exit 3
scripts/ops/mem_cap.sh --name argen-perturb --max-s 600 6 nice -n 19 taskset -c 5-7,13-15 \
  /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 $D/perturb.py 2>&1 | grep -v Warning
