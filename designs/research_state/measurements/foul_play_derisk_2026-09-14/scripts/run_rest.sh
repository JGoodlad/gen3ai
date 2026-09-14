#!/usr/bin/env bash
# Resume the campaign at session 4 (sessions 0-3 completed before main gained
# --forfeit-turn-limit under us; see README hazard H3).
set -u
ROOT=/home/goodlad/.claude/jobs/9ab51de6/tmp/foul_play
POOL=/home/goodlad/dev/foul-play/fp/teams/teams/gen3/ou/gen3ai_pool
mapfile -t ALL < <(ls "$POOL"/*.txt | sort)
N_SESS=8
GAMES=10
STEP=$(( ${#ALL[@]} / N_SESS ))
for i in 4 5 6 7; do
  T="${ALL[$(( i * STEP ))]}"
  echo "=== session $i : $(date -Is) ==="
  bash "$ROOT/scripts/run_session.sh" "$i" "$T" "$GAMES"
done
echo "=== ALL DONE $(date -Is) ==="
