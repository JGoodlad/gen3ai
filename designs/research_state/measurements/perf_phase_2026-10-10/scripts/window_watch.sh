#!/usr/bin/env bash
# THE FILE-BASED WINDOW HANDOFF (orchestrator, 2026-10-10). A BOUNDED watcher: it polls every 30 s until BOTH
#   (a) ~/.claude/jobs/es_closing/PERF_WINDOW_GO exists and is NEWER than PERF_WINDOW_DONE (if any), and
#   (b) scripts/ops/gpu_lease.sh status reads FREE,
# then IMMEDIATELY acquires the lease as perf_T25, runs `window.sh <WIN> <token> <MINUTES>`, releases, and writes
# PERF_WINDOW_DONE (removing nothing). It exits on its own after MAX_S seconds (default 3300, the 55-min rule) if
# the window never opens; re-arm it then.
#
#   bash window_watch.sh B 44 [MAX_S]
set -u
WIN=${1:?window}
MINUTES=${2:-44}
MAX_S=${3:-3300}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
D="$HOME/.claude/jobs/es_closing"
GO="$D/PERF_WINDOW_GO"
DONE="$D/PERF_WINDOW_DONE"
TOK=/tmp/perf_T25_lease.tok
END=$(( $(date +%s) + MAX_S ))
echo "[watch $WIN] armed at $(date +%T), until $(date -d @$END +%T)"
while (( $(date +%s) < END )); do
  if [[ -e "$GO" && ( ! -e "$DONE" || "$GO" -nt "$DONE" ) ]] && "$ROOT/scripts/ops/gpu_lease.sh" status 2>/dev/null | grep -q '^FREE'; then
    echo "[watch $WIN] GO + FREE at $(date +%T): acquiring"
    if ! "$ROOT/scripts/ops/gpu_lease.sh" acquire --owner perf_T25 --max-hours 1 --token-file "$TOK"; then
      echo "[watch $WIN] acquire REFUSED at $(date +%T); polling on"
      sleep 30
      continue
    fi
    bash "$HERE/window.sh" "$WIN" "$TOK" "$MINUTES"
    GEN3AI_GPU_LEASE_TOKEN_FILE="$TOK" "$ROOT/scripts/ops/gpu_lease.sh" release
    date +%FT%T > "$DONE"
    echo "[watch $WIN] DONE at $(date +%T): lease released, $DONE written"
    exit 0
  fi
  sleep 30
done
echo "[watch $WIN] expired at $(date +%T) without a window: re-arm"
exit 3
