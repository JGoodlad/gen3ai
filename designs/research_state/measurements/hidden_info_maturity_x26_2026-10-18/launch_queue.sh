#!/usr/bin/env bash
# Launch the battery's runner DETACHED from a DRIVER checkout (a detached checkout of the commit that ships the launch):
# survives the launching session. Prints the runner's pid. Refuses if a runner already holds the pid file.
#   launch_queue.sh [queue_mat.py args...]
# Amendment A1 (2026-10-19): the DRIVER (this folder's scripts, stamped into every row as `driver`) and the world
# BUILDER are separate checkouts. The builder stays the registration commit (mat_common.BUILDER_SHA, 826bee89;
# §12.13.7 (f): a changed builder starts a new series) in $HIB_MAT_HEAD (default ~/.cache/gen3ai/hib_mat/head).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${HIB_MAT_ARCHIVE:-$HOME/gen3ai_archive/hidden_info_maturity}"
mkdir -p "$ARCH/logs"
PIDF="$ARCH/queue.pid"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "a runner is already up: pid $(cat "$PIDF")" >&2; exit 3
fi
head_of() {  # the commit a checkout is DETACHED at (refuses a branch)
  local dir="$1" gitdir sha
  if [ -f "$dir/.git" ]; then gitdir="$(sed 's/^gitdir: //' "$dir/.git")"; else gitdir="$dir/.git"; fi
  sha="$(cat "$gitdir/HEAD")"
  case "$sha" in ref:*) echo "checkout $dir must be DETACHED at a shipped commit (it is on ${sha#ref: })" >&2; return 4 ;; esac
  if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
    echo "checkout $dir has local changes: refused" >&2; return 5
  fi
  echo "$sha"
}
REPO="$(cd "$HERE/../../../.." && pwd)"
DRIVER_SHA="$(head_of "$REPO")"
BUILDER="${HIB_MAT_HEAD:-$HOME/.cache/gen3ai/hib_mat/head}"
BUILDER_SHA="$(head_of "$BUILDER")"
WANT="$(sed -n 's/^BUILDER_SHA = "\([0-9a-f]*\)"$/\1/p' "$HERE/mat_common.py")"
if [ "$BUILDER_SHA" != "$WANT" ]; then
  echo "the builder checkout $BUILDER is at $BUILDER_SHA, not the registered builder $WANT" >&2; exit 6
fi
export HIB_MAT_HEAD="$BUILDER" HIB_MAT_HEAD_SHA="$BUILDER_SHA" HIB_MAT_DRIVER_SHA="$DRIVER_SHA"
unset HIB_MAT_PILOT
echo "DRIVER: $REPO at $DRIVER_SHA"
echo "BUILDER: $BUILDER at $BUILDER_SHA"
cd "$HERE"
nohup setsid nice -n 19 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -I "$HERE/queue_mat.py" "$@" \
  >> "$ARCH/logs/queue_mat.out" 2>&1 < /dev/null &
echo "$!" > "$PIDF"
echo "runner pid $! (pid file $PIDF; log $ARCH/logs/queue_mat.log)"
