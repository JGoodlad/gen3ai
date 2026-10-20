#!/usr/bin/env bash
# Launch stage 1's runner DETACHED (survives the launching session) from a DRIVER checkout: a DETACHED, clean checkout
# of the commit that ships the launch, which is also the world BUILDER (one commit builds every arm). Writes the
# REGISTERED STOP once (PREREGISTRATION §5: STOP_HOURS after the first launch; a relaunch keeps it). Prints the pid.
#   launch_s1.sh [queue_s1.py args...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
ARCH="${WS1_ARCHIVE:-$HOME/gen3ai_archive/world_sampler_stage1}"
STOP_HOURS=24
mkdir -p "$ARCH/logs"
PIDF="$ARCH/queue.pid"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "a runner is already up: pid $(cat "$PIDF")" >&2; exit 3
fi
if [ -n "$(git -C "$REPO" symbolic-ref -q HEAD || true)" ]; then
  echo "the driver checkout $REPO must be DETACHED at a shipped commit" >&2; exit 4
fi
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "the driver checkout $REPO has local changes: refused" >&2; exit 5
fi
SHA="$(git -C "$REPO" rev-parse HEAD)"
if ! git -C "$REPO" merge-base --is-ancestor "$SHA" origin/main; then
  echo "the driver commit $SHA is not on origin/main (ship first)" >&2; exit 6
fi
if [ ! -f "$ARCH/STOP_AT" ]; then
  python3 -c "import time; t=time.time()+$STOP_HOURS*3600; print(f'{t:.0f}', time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t)))" > "$ARCH/STOP_AT"
fi
export WS1_BUILDER="$REPO" WS1_BUILDER_SHA="$SHA" WS1_DRIVER_SHA="$SHA" WS1_ARCHIVE="$ARCH"
echo "DRIVER = BUILDER: $REPO at $SHA; STOP_AT $(cat "$ARCH/STOP_AT")"
cd "$HERE"
nohup setsid nice -n 19 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -I "$HERE/queue_s1.py" "$@" \
  >> "$ARCH/logs/queue.out" 2>&1 < /dev/null &
echo "$!" > "$PIDF"
echo "runner pid $! (pid file $PIDF; log $ARCH/logs/queue.log)"
