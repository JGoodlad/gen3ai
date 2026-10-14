#!/usr/bin/env bash
# One checkpoint's world builds in the registered priority order, ONE process at a time (resumable at every step):
#   build_chain.sh <cp>
# 1. the registered 600: the spec arms + FN (F's screen shared in-process), 2. POOLSETS and POOL, 3. the supplement's
# arms. Touches <archive>/BUILDS_DONE_<cp> at the end.
set -uo pipefail
CP="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS=/home/goodlad/gen3ai_archive/hidden_info_tier1/logs
bash "$HERE/run_head.sh" 8 24h build_worlds.py --cp "$CP" --states registered \
  --arms F,D50,D90,W90,MAP100,F+U,FN,F+M,O-M,O-U >> "$LOGS/worlds_${CP}_reg_spec.log" 2>&1
bash "$HERE/run_head.sh" 8 24h build_worlds.py --cp "$CP" --states registered --arms POOLSETS,POOL \
  >> "$LOGS/worlds_${CP}_reg_pool.log" 2>&1
bash "$HERE/run_head.sh" 8 24h build_worlds.py --cp "$CP" --states supp >> "$LOGS/worlds_${CP}_supp.log" 2>&1
touch "/home/goodlad/gen3ai_archive/hidden_info_tier1/BUILDS_DONE_${CP}"
echo "chain done $(date)" >> "$LOGS/worlds_${CP}_reg_spec.log"
