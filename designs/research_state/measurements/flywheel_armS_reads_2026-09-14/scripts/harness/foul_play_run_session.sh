#!/usr/bin/env bash
# One head-to-head SESSION: N games, our checkpoint (fixed pool team) accepting
# challenges from Foul Play (random pool team per battle).
#
#   run_session.sh <session_idx> <our_team_file> <n_games>
set -u
SESSION=$1; OUR_TEAM=$2; N=$3
ROOT=/home/goodlad/.claude/jobs/9ab51de6/tmp/armS_reads/anchors/foul_play
GEN3AI=/home/goodlad/dev/gen3ai
FP=/home/goodlad/dev/foul-play
PORT=9417
MODEL=$GEN3AI/models/ai_v13_01_flywheel_shaped/final_model.zip
# 🚨 KEEP THESE UNDER 19 CHARACTERS. Both clients guest-login through Smogon's
# action.php even against a --no-security LOCAL server, and a 19-char name comes back as
# `;;Your username must be less than 19 characters long.` — which Foul Play's guest path
# (fp/websocket_client.py:121) accepts as an assertion and /trn's, then hangs forever.
OURS=G3aiS914s${SESSION}
THEIRS=FpS914s${SESSION}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3

export PYTHONPATH=${PYTHONPATH:-}:$GEN3AI/src

# preflight: refuse a name action.php will not issue an assertion for (see above)
for NAME in "$OURS" "$THEIRS"; do
  A=$(curl -s --data "act=getassertion&userid=$NAME&challstr=4|0" \
        "https://play.pokemonshowdown.com/action.php?")
  case "$A" in
    ";"*) echo "[session $SESSION] ABORT: action.php refused $NAME -> $A"; exit 2;;
  esac
done

nohup nice -n 10 $PY $ROOT/scripts/run_derisk.py \
  --model "$MODEL" --port $PORT --username "$OURS" --opponent "$THEIRS" \
  --team "$OUR_TEAM" --n-battles "$N" --out "$ROOT/games_raw.jsonl" \
  --session-timeout 5400 \
  > "$ROOT/logs/ours_s${SESSION}.log" 2>&1 < /dev/null &
OURS_PID=$!
echo "[session $SESSION] ours pid=$OURS_PID team=$(basename "$OUR_TEAM")"
sleep 40

cd "$FP" || exit 1
PYTHONUNBUFFERED=1 timeout 2400 nice -n 10 ./.conda/bin/python -u run.py \
  --websocket-uri ws://localhost:$PORT/showdown/websocket \
  --ps-username "$THEIRS" \
  --bot-mode challenge_user --user-to-challenge "$OURS" \
  --pokemon-format gen3ou \
  --team-name gen3/ou/gen3ai_pool \
  --search-time-ms 1000 --search-parallelism 1 --search-threads 1 \
  --run-count "$N" --log-level INFO 2>&1 \
  | nice -n 10 $PY $ROOT/scripts/ts.py > "$ROOT/logs/fp_s${SESSION}.log"
FP_EXIT=${PIPESTATUS[0]}
echo "[session $SESSION] foul-play exit=$FP_EXIT"

# our side exits on its own once n_battles are done; give it a moment, then reap
for _ in $(seq 1 60); do kill -0 $OURS_PID 2>/dev/null || break; sleep 2; done
kill -0 $OURS_PID 2>/dev/null && { echo "[session $SESSION] killing our client $OURS_PID"; kill $OURS_PID; }
echo "[session $SESSION] done"
