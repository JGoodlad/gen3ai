#!/usr/bin/env bash
# Run one model_input step AFTER another process exits (keeps at most two runner processes alive at once).
#   queue_after.sh <pid> <max_wait_s> <end2|prod> [HIB_CKPT]
# Waits at most <max_wait_s> for <pid>; starts the step only if <pid> is gone by then (else exits 9, nothing started).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID="$1"; MAX="$2"; PIN="$3"; CK="${4:-}"
deadline=$((SECONDS + MAX))
while kill -0 "$PID" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do sleep 30; done
if kill -0 "$PID" 2>/dev/null; then echo "queue_after: $PID still alive after ${MAX}s; not starting" >&2; exit 9; fi
echo "=== $(date -Is) start $PIN ${CK}"
HIB_CKPT="$CK" exec "$HERE/run_at_pin.sh" "$PIN" 2 8 20h model_input.py
