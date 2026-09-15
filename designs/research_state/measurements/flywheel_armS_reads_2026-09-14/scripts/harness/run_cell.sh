#!/usr/bin/env bash
# ONE HALF-CELL of the matched-regime 2x2: <n> gen3ou games between a Metamon pretrained policy
# and OUR checkpoint, at a fixed (regime, team set, who-challenges) on a LOCAL Showdown server
# built from deps/pokemon-showdown.
#
#   bash run_cell.sh <agent> <greedy|t1> <home|away> <ours|metamon> <n> <port> <out_dir> \
#                    <our_team_seed> <their_team_seed> <tag>
#
# ROLE ORDER IS LOAD-BEARING. The ACCEPTOR must be logged in before the CHALLENGER sends its
# first `/challenge`: Showdown drops a challenge aimed at a user who is not online, and the
# challenger then waits forever. So the acceptor is started first and we wait for its banner.
# When Metamon is the challenger the margin is large by construction — `_pretrained_on_ladder`
# BUILDS the policy (seconds for SmallRL, minutes for the 200M SyntheticRLV2) before its env is
# constructed and its client connects.
#
# PYTHONUNBUFFERED is mandatory on the Metamon side: its own prints carry no flush=True, so a
# redirected stdout is block-buffered and the banner a driver waits on never lands (de-risk H5).
set -uo pipefail

AGENT="${1:?agent}"; REGIME="${2:?greedy|t1}"; TEAMS="${3:?home|away}"
WHO="${4:?ours|metamon}"; N="${5:?n}"; PORT="${6:?port}"; OUT="${7:?out_dir}"
OUR_SEED="${8:?our_team_seed}"; THEIR_SEED="${9:?their_team_seed}"; TAG="${10:?tag}"

case "$PORT" in 8000|8001) echo "refusing port $PORT (dev/training server)"; exit 2;; esac

REPO=/home/goodlad/dev/gen3ai
WT=/home/goodlad/dev/gen3ai
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${GEN3AI_MODEL:-models/ai_v12_02_winprob_critic/final_model.zip}"
PY_META=/home/goodlad/miniconda3/envs/metamon/bin/python
PY_OURS=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
COMPETITIVE_DIR=/home/goodlad/dev/metamon/cache/teams/competitive/gen3ou
MT="${CELL_TIMEOUT:-7200}"; OT="${CELL_TIMEOUT:-7200}"

mkdir -p "$OUT"
MU="Metamon${TAG}"; OU="Gen3AI${TAG}"

# --- the team-set arms, one per side -------------------------------------------------------
if [ "$TEAMS" = "home" ]; then
  META_TEAMS=(--team_set gen3ai_pool)
  OUR_TEAMS=(--team-pool)
else
  META_TEAMS=(--team_set competitive)
  OUR_TEAMS=(--team-dir "$COMPETITIVE_DIR")
fi

# --- the regime arms, one per side ---------------------------------------------------------
# `mixed` reproduces the 2026-09-14 DE-RISK's regime: OUR side greedy, Metamon at its OWN eval
# default (temperature 1.0). It is the only cell here whose two sides are NOT at the same setting,
# and it exists because the de-risk's banked 0.742 is a mixed-regime number and a comparator must
# be quoted in its own regime.
case "$REGIME" in
  greedy) OUR_TEMP=0.0; META_REGIME=greedy ;;
  t1)     OUR_TEMP=1.0; META_REGIME=t1 ;;
  mixed)  OUR_TEMP=0.0; META_REGIME=t1 ;;
  *) echo "unknown regime $REGIME"; exit 2 ;;
esac

start_metamon() {
  cd /home/goodlad/dev/metamon
  PYTHONUNBUFFERED=1 METAMON_CACHE_DIR=/home/goodlad/dev/metamon/cache \
    timeout "$MT" nice -n 10 "$PY_META" "$HERE/run_metamon_side.py" \
    --port "$PORT" --agent "$AGENT" --username "$MU" --opponent_username "$OU" \
    --role "$1" --total_battles "$N" "${META_TEAMS[@]}" --regime "$META_REGIME" \
    --team-seed "$THEIR_SEED" \
    --save_results_to "$OUT" --timing_out "$OUT/metamon_timing.json" \
    --draws_out "$OUT/metamon_draws.json" \
    > "$OUT/metamon.log" 2>&1 &
  META_PID=$!
  echo "[cell] metamon pid=$META_PID role=$1"
}

start_ours() {
  cd "$REPO"
  PYTHONUNBUFFERED=1 PYTHONPATH="$WT/src" CUDA_VISIBLE_DEVICES="" \
    timeout "$OT" nice -n 10 "$PY_OURS" "$HERE/run_gen3ai_side.py" \
    --mode "$1" --port "$PORT" --opponent "$MU" --username "$OU" --model "$MODEL" \
    "${OUR_TEAMS[@]}" --temperature "$OUR_TEMP" --team-seed "$OUR_SEED" \
    --n-battles "$N" --games-out "$OUT/games_raw.jsonl" --series "$AGENT" \
    --timing-out "$OUT/our_timing.json" --draws-out "$OUT/our_draws.json" \
    > "$OUT/gen3ai.log" 2>&1 &
  OUR_PID=$!
  echo "[cell] ours pid=$OUR_PID mode=$1"
}

wait_for() {  # wait_for <logfile> <pattern> <pid> <label>
  for _ in $(seq 1 360); do
    grep -q "$2" "$1" 2>/dev/null && { echo "[cell] $4 online"; return 0; }
    kill -0 "$3" 2>/dev/null || { echo "[cell] $4 died before it was ready"; tail -25 "$1"; return 1; }
    sleep 5
  done
  echo "[cell] $4 never came online"; return 1
}

echo "[cell] $AGENT regime=$REGIME teams=$TEAMS challenger=$WHO n=$N port=$PORT tag=$TAG"
echo "[cell] model=$MODEL our_temp=$OUR_TEMP seeds ours=$OUR_SEED theirs=$THEIR_SEED"

if [ "$WHO" = "ours" ]; then
  # Metamon accepts; it must be online first.
  start_metamon acceptor
  wait_for "$OUT/metamon.log" "Made Challenge Env" "$META_PID" "metamon(acceptor)" || {
    kill "$META_PID" 2>/dev/null; exit 1; }
  start_ours challenge
else
  # We accept; we must be online first. The extra settle is for the LOGIN, which the banner
  # does not prove — the banner prints before `accept_challenges` awaits the websocket.
  start_ours accept
  wait_for "$OUT/gen3ai.log" "connect-or-raise deadline" "$OUR_PID" "ours(acceptor)" || {
    kill "$OUR_PID" 2>/dev/null; exit 1; }
  sleep 20
  start_metamon challenger
fi

wait "$OUR_PID"; OUR_RC=$?
wait "$META_PID"; META_RC=$?
echo "[cell] ours exit=$OUR_RC metamon exit=$META_RC"

cat > "$OUT/cell.json" <<JSON
{"agent":"$AGENT","regime":"$REGIME","team_set":"$TEAMS","challenger":"$WHO",
 "n_requested":$N,"our_rc":$OUR_RC,"metamon_rc":$META_RC,"model":"$MODEL","port":$PORT,
 "our_temperature":$OUR_TEMP,"metamon_regime":"$META_REGIME","our_team_seed":$OUR_SEED,"their_team_seed":$THEIR_SEED,"tag":"$TAG"}
JSON
