#!/usr/bin/env bash
# One head-to-head series: a Metamon pretrained policy (acceptor) vs OUR checkpoint (challenger)
# on a LOCAL Showdown server built from deps/pokemon-showdown.
#
#   bash run_series.sh <agent> <n_battles> <port> <out_dir> [metamon_timeout_s] [our_timeout_s]
#
# The acceptor must be online before the challenger starts, so this starts Metamon, waits for its
# "Made Challenge Env" banner, then starts ours. PYTHONUNBUFFERED is mandatory: Metamon's own
# prints have no flush=True, so a redirected stdout is block-buffered and the banner never lands.
set -uo pipefail

AGENT="${1:?agent}"; N="${2:?n_battles}"; PORT="${3:?port}"; OUT="${4:?out_dir}"
MT="${5:-7200}"; OT="${6:-7200}"

case "$PORT" in 8000|8001) echo "refusing port $PORT (dev/training server)"; exit 2;; esac

REPO=/home/goodlad/dev/gen3ai
WT=/home/goodlad/dev/gen3ai-wt/metamon-derisk
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${GEN3AI_MODEL:-models/ai_v12_02_winprob_critic/final_model.zip}"
PY_META=/home/goodlad/miniconda3/envs/metamon/bin/python
PY_OURS=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3

mkdir -p "$OUT"
MU="Meta${AGENT}"; OU="Gen3AIv12"

echo "[series] $AGENT vs $MODEL — $N battles on :$PORT"
cd /home/goodlad/dev/metamon
PYTHONUNBUFFERED=1 METAMON_CACHE_DIR=/home/goodlad/dev/metamon/cache \
  timeout "$MT" nice -n 10 "$PY_META" "$HERE/run_metamon_side.py" \
  --port "$PORT" --agent "$AGENT" --username "$MU" --opponent_username "$OU" \
  --role acceptor --total_battles "$N" --team_set gen3ai_pool \
  --save_results_to "$OUT" --timing_out "$OUT/metamon_timing.json" \
  > "$OUT/metamon.log" 2>&1 &
META_PID=$!
echo "[series] metamon pid=$META_PID"

# Wait for the acceptor to be online (or to die).
for _ in $(seq 1 240); do
  grep -q "Made Challenge Env" "$OUT/metamon.log" && break
  kill -0 "$META_PID" 2>/dev/null || { echo "[series] metamon died before it was ready"; tail -20 "$OUT/metamon.log"; exit 1; }
  sleep 5
done
grep -q "Made Challenge Env" "$OUT/metamon.log" || { echo "[series] metamon never came online"; kill "$META_PID"; exit 1; }
echo "[series] acceptor online"

cd "$REPO"
PYTHONUNBUFFERED=1 PYTHONPATH="$WT/src" timeout "$OT" nice -n 10 "$PY_OURS" \
  "$HERE/run_gen3ai_side.py" --mode challenge --port "$PORT" \
  --opponent "$MU" --username "$OU" --model "$MODEL" --team-pool \
  --n-battles "$N" --games-out "$OUT/games_raw.jsonl" --series "$AGENT" \
  --timing-out "$OUT/our_timing.json" > "$OUT/gen3ai.log" 2>&1
OUR_RC=$?
echo "[series] our side exit=$OUR_RC"

wait "$META_PID"; META_RC=$?
echo "[series] metamon exit=$META_RC"
echo "{\"agent\":\"$AGENT\",\"n_requested\":$N,\"our_rc\":$OUR_RC,\"metamon_rc\":$META_RC,\"model\":\"$MODEL\",\"port\":$PORT}" > "$OUT/series.json"
