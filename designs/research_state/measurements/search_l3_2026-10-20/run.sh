#!/usr/bin/env bash
# L3's run (PREREGISTRATION.md §8): the driver over the 287 registered states, then the determinism recheck (G3).
# Launch from this checkout, inside ONE heavy slot (rule 5b), CPU only:
#   CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --slot --name l3 12 \
#       designs/research_state/measurements/search_l3_2026-10-20/run.sh
# $L3_PIN_CORE_EVENTS names X26's pin's core_events (G4); without it G4 is not read.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}"
export CUDA_VISIBLE_DEVICES=""
cd "$HERE"
"$PY" -I l3_run.py
"$PY" -I l3_run.py --recheck 10
echo "[l3 run] RUN_DONE"
