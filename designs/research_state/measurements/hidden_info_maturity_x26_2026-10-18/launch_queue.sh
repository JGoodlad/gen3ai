#!/usr/bin/env bash
# Launch the battery's runner DETACHED from the HEAD checkout (a detached checkout of the commit that ships the launch):
# survives the launching session. Prints the runner's pid. Refuses if a runner already holds the pid file.
#   launch_queue.sh [queue_mat.py args...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${HIB_MAT_ARCHIVE:-$HOME/gen3ai_archive/hidden_info_maturity}"
mkdir -p "$ARCH/logs"
PIDF="$ARCH/queue.pid"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "a runner is already up: pid $(cat "$PIDF")" >&2; exit 3
fi
REPO="$(cd "$HERE/../../../.." && pwd)"
if [ -f "$REPO/.git" ]; then GITDIR="$(sed 's/^gitdir: //' "$REPO/.git")"; else GITDIR="$REPO/.git"; fi
SHA="$(cat "$GITDIR/HEAD")"
case "$SHA" in ref:*) echo "the HEAD checkout must be DETACHED at the shipped commit ($REPO is on ${SHA#ref: })" >&2; exit 4 ;; esac
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "the HEAD checkout $REPO has local changes: refused (the builder must be the commit's)" >&2; exit 5
fi
# the world builder runs in THIS checkout, at THIS commit (run_mat.sh head refuses any other)
export HIB_MAT_HEAD="$REPO" HIB_MAT_HEAD_SHA="$SHA"
unset HIB_MAT_PILOT
echo "HEAD: $REPO at $SHA"
cd "$HERE"
nohup setsid nice -n 19 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -I "$HERE/queue_mat.py" "$@" \
  >> "$ARCH/logs/queue_mat.out" 2>&1 < /dev/null &
echo "$!" > "$PIDF"
echo "runner pid $! (pid file $PIDF; log $ARCH/logs/queue_mat.log)"
