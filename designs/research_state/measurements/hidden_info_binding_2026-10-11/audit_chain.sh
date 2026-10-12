#!/usr/bin/env bash
# Run audit scripts back to back, each through run_at_pin.sh (resumable; a failed step does not stop the next).
#   audit_chain.sh <log> <threads> "<cp> <script> [args]" ["<cp> <script> [args]" ...]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$1"; T="$2"; shift 2
for step in "$@"; do
  # shellcheck disable=SC2086
  set -- $step
  cp="$1"; shift
  echo "=== $(date -Is) $cp $*" >> "$LOG"
  "$HERE/run_at_pin.sh" "$cp" "$T" 8 3h "$@" >> "$LOG" 2>&1 < /dev/null
  echo "=== $(date -Is) exit $?" >> "$LOG"
done
