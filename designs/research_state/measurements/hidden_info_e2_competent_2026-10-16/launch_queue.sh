#!/usr/bin/env bash
# Launch E2's runner DETACHED from the E2 HEAD checkout (the commit that ships this harness): survives the launching
# session. Prints the runner's pid. Refuses if a runner already holds the pid file.
#   launch_queue.sh [queue_e2.py args...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${HIB_E2_ARCHIVE:-$HOME/gen3ai_archive/hidden_info_e2}"
mkdir -p "$ARCH/logs"
PIDF="$ARCH/queue.pid"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "a runner is already up: pid $(cat "$PIDF")" >&2; exit 3
fi
REPO="$(cd "$HERE/../../../.." && pwd)"
GITDIR="$(sed 's/^gitdir: //' "$REPO/.git")"
SHA="$(cat "$GITDIR/HEAD")"
case "$SHA" in ref:*) echo "the E2 HEAD checkout must be DETACHED at the shipped commit ($REPO is on ${SHA#ref: })" >&2; exit 4 ;; esac
# the world builder runs in THIS checkout, at THIS commit (run_e2.sh head refuses any other)
export HIB_E2_HEAD="$REPO" HIB_E2_HEAD_SHA="$SHA"
echo "E2 HEAD: $REPO at $SHA"
cd "$HERE"
nohup setsid nice -n 19 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -I "$HERE/queue_e2.py" "$@" \
  >> "$ARCH/logs/queue_e2.out" 2>&1 < /dev/null &
echo "$!" > "$PIDF"
echo "runner pid $! (pid file $PIDF; log $ARCH/logs/queue_e2.log)"
