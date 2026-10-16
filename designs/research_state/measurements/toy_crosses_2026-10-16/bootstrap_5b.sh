#!/usr/bin/env bash
# bootstrap this worktree (--skip-env --with-rust) under rule 5b; log to /tmp/toyx_bootstrap.log
HERE=$(cd "$(dirname "$0")" && pwd); TOP=$(git -C "$HERE" rev-parse --show-toplevel)
cd "$TOP" || exit 2
nice -n 19 taskset -c 5-7,13-15 ./scripts/bootstrap.sh --skip-env --with-rust > /tmp/toyx_bootstrap.log 2>&1
rc=$?; echo "rc=$rc"; tail -25 /tmp/toyx_bootstrap.log; exit $rc
