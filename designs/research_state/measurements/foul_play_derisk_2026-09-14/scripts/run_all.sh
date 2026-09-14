#!/usr/bin/env bash
# The whole de-risk campaign: 8 sessions x 10 games = 80 gen3ou games.
# Our side pins ONE pool team per session (rotating across 8); Foul Play draws a
# random pool team every battle from the SAME exported folder.
set -u
ROOT=/home/goodlad/.claude/jobs/9ab51de6/tmp/foul_play
POOL=/home/goodlad/dev/foul-play/fp/teams/teams/gen3/ou/gen3ai_pool
mapfile -t ALL < <(ls "$POOL"/*.txt | sort)
N_SESS=8
GAMES=10
STEP=$(( ${#ALL[@]} / N_SESS ))
for i in $(seq 0 $((N_SESS - 1))); do
  T="${ALL[$(( i * STEP ))]}"
  echo "=== session $i : $(date -Is) ==="
  bash "$ROOT/scripts/run_session.sh" "$i" "$T" "$GAMES"
done
echo "=== ALL DONE $(date -Is) ==="
