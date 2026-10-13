#!/usr/bin/env bash
# Run one of this folder's pure-numpy reads (reads the pilot archive; no model, no playout) under the rule-5b caps:
# nice 19, taskset 5-7,13-15, a memory cap and a timeout.
#
#   run_read.sh <script.py> [args...]      e.g. run_read.sh by_reveal_count.py
set -euo pipefail
SCRIPT="$1"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
exec nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name by_reveal_count "${BRC_GB:-12}" timeout "${BRC_TIMEOUT:-50m}" \
  taskset -c "${BRC_CPUS:-5-7,13-15}" "$PY" -I "$HERE/$SCRIPT" "$@"
