#!/usr/bin/env bash
#
# showdown_supervisor.sh — run the training Showdown server in the foreground
# and auto-restart it if it crashes. Built to own a tmux pane.
#
#   bash scripts/training/showdown_supervisor.sh [PORT]      # default 8001
#
# Everything logs right here in the terminal. Ctrl-C stops it for good (it will
# NOT restart). That's the only command you need to remember.
#
set -u

PORT="${1:-8001}"
# Repo root derived from this script's location (works from main or a worktree);
# override with SHOWDOWN_ROOT=... if ever needed.
ROOT="${SHOWDOWN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="$ROOT/deps/pokemon-showdown/pokemon-showdown"

# Ctrl-C (SIGINT) or SIGTERM => stop for good, don't treat it as a crash.
stop=0
trap 'stop=1; echo; echo "[sup] stopping — no restart"; kill "$child" 2>/dev/null' INT TERM

recent=()   # crash timestamps, for the "give up if it won't stay up" guard
while :; do
  echo "[sup] $(date '+%F %T') starting showdown on :$PORT"
  NODE_ENV=production node --turbo-fast-api-calls --max-old-space-size=5120 \
      "$BIN" start --no-security "$PORT" &
  child=$!
  wait "$child"; code=$?

  (( stop )) && exit 0          # Ctrl-C'd → leave cleanly

  echo "[sup] server exited (code=$code) — restarting in 3s…"

  # If it crashes >5 times in under a minute it's a real problem (port in use,
  # bad build, …), so stop and leave the error on screen instead of spamming.
  now=$(date +%s); recent+=("$now")
  recent=($(for t in "${recent[@]}"; do (( now - t < 60 )) && echo "$t"; done))
  if (( ${#recent[@]} >= 5 )); then
    echo "[sup] 5 crashes in under a minute — giving up so you can read why above."
    exit 1
  fi

  sleep 3
done
