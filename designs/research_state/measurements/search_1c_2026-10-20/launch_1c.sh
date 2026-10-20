#!/usr/bin/env bash
# Launch (or relaunch) STAGE 1c's detached runner from THIS checkout (the driver commit). Phase A writes the registered
# STOP once (first launch + 30 h; a relaunch keeps it); phase B writes STOP_B once (its first launch + 24 h).
#   launch_1c.sh [a|play]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-a}"
ARCH="${S1C_ARCHIVE:-$HOME/gen3ai_archive/search_1c}"
mkdir -p "$ARCH/logs"
if [ "$PHASE" = a ]; then
  [ -f "$ARCH/STOP_AT" ] || echo "$(( $(date +%s) + 30 * 3600 ))" > "$ARCH/STOP_AT"
else
  [ -f "$ARCH/STOP_B" ] || echo "$(( $(date +%s) + 24 * 3600 ))" > "$ARCH/STOP_B"
fi
nohup setsid "$HERE/loop_1c.sh" "$PHASE" "${S1C_BUDGET_MIN:-30}" "${S1C_TIMEOUT:-45m}" \
  >> "$ARCH/logs/loop_$PHASE.out" 2>&1 < /dev/null &
echo $! > "$ARCH/loop_$PHASE.pid"
echo "launched loop $PHASE: pid $(cat "$ARCH/loop_$PHASE.pid")"
