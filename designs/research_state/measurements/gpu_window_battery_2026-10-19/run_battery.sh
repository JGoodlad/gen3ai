#!/usr/bin/env bash
# run_battery.sh — the 2026-10-19 GPU-window battery (README.md beside this file). A thin wrapper: it pins the
# interpreter and THIS checkout's src/, then runs battery.py with the arguments given.
#
#   export GEN3AI_GPU_LEASE_TOKEN_FILE=<lease token file>      # the leaseholder (the training session) only
#   designs/research_state/measurements/gpu_window_battery_2026-10-19/run_battery.sh --out <dir> [--budget-min N]
#   designs/research_state/measurements/gpu_window_battery_2026-10-19/run_battery.sh --out <dir> --dry-run   # CPU only
#
# Exit: 0 DONE · 1 HOLD (battery_status.json names the reason; the next cell did NOT run) · 10 BUDGET_STOP (re-run
# resumes) · 2 usage / refused --out.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
export PY="${PY:-${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}}"
export PYTHONPATH="$ROOT/src"
case " $* " in
    *" --dry-run "*|*" --list "*) export CUDA_VISIBLE_DEVICES="" ;;   # the dry run never sees the GPU
esac
exec "$PY" "$HERE/battery.py" "$@"
