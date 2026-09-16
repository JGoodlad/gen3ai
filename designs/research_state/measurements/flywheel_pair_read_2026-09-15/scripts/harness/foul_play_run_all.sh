#!/usr/bin/env bash
# The whole arm-W campaign: 8 sessions x 10 games = 80 gen3ou games, the SAME 8 pinned pool
# teams arm S used (same sorted export, same stride).
set -u
ROOT=/home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/anchors/foul_play
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
