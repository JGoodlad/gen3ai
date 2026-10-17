#!/usr/bin/env bash
# env wrapper for these CPU reads (rule 5b: run it as `nice -n 19 taskset -c 5-7,13-15 run.sh <script> ...`):
# CPU only, one thread, THIS checkout's src first. Run from the checkout root.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export PYTHONPATH="$(git rev-parse --show-toplevel)/src"
exec "${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}" "$@"
