#!/usr/bin/env bash
# Worker 3 of the 3-worker schedule: re-run end2 P's 3 failed states (final code), then the queue below.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIR_IDS="$HOME/gen3ai_archive/hidden_info_replica/rerun_end2_P.txt" HIR_TAG=_rerun1 \
  "$HERE/run_chain.sh" end2 0 1 5-7,13-15 P
exec "$HERE/seq_remaining.sh" _g 5-7,13-15 prod:F:interim end2:F prod:F+M
