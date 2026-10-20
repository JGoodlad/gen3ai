#!/usr/bin/env bash
# X50 S0 driver: the census over every date folder of the corpus, RESUMABLE (a date with a shard is skipped),
# CHUNKED (one heavy slot per chunk of X50_CHUNK_DATES dates, each chunk bounded at <= 45 min of work) and
# STOPPED at a REGISTERED epoch (X50_DEADLINE_EPOCH, required). CPU only, rule 5b: nice 19 on the agent cores.
#
#   X50_DEADLINE_EPOCH=$(( $(date +%s) + 8*3600 )) setsid nohup timeout 30000 run_census.sh > log 2>&1 &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
SHARDS=${X50_SHARDS:-$HERE/shards}
DEADLINE=${X50_DEADLINE_EPOCH:?X50_DEADLINE_EPOCH (the registered stop, epoch seconds) is required}
CHUNK=${X50_CHUNK_DATES:-20}
export PYTHONPATH="$REPO/src" CUDA_VISIBLE_DEVICES=""
export POKESIM_LIVE_READER_BIN="$REPO/src/rust_sim/target/release/live_reader"
export X50_CORE_EVENTS_BIN="$REPO/src/rust_sim/target/release/core_events"
for b in "$POKESIM_LIVE_READER_BIN" "$X50_CORE_EVENTS_BIN"; do
    [ -x "$b" ] || { echo "[x50] missing $b (cargo build --release --bin live_reader --bin core_events)"; exit 2; }
done
mapfile -t DATES < <("$PY" "$HERE/census.py" dates)
echo "[x50] $(date -Is) ${#DATES[@]} date folders; stop at $(date -Is -d @"$DEADLINE"); shards $SHARDS"
while :; do
    todo=()
    for d in "${DATES[@]}"; do [ -e "$SHARDS/$d.json.gz" ] || todo+=("$d"); done
    [ ${#todo[@]} -eq 0 ] && { echo "[x50] $(date -Is) all ${#DATES[@]} dates done"; exit 0; }
    now=$(date +%s)
    [ "$now" -ge "$DEADLINE" ] && { echo "[x50] $(date -Is) REGISTERED STOP reached, ${#todo[@]} dates left"; exit 3; }
    batch=$(IFS=,; echo "${todo[*]:0:$CHUNK}")
    left=$(( DEADLINE - now )); bound=$(( left < 2700 ? left : 2700 ))
    echo "[x50] $(date -Is) chunk: ${todo[0]} .. (${CHUNK} dates max), bound ${bound}s"
    before=${#todo[@]}
    "$REPO/scripts/ops/mem_cap.sh" --name x50census --slot 8 timeout "$bound" \
        nice -n 19 taskset -c 5-7,13-15 "$PY" "$HERE/census.py" run --dates "$batch" --shards "$SHARDS" --workers 6
    rc=$?
    after=0
    for d in "${DATES[@]}"; do [ -e "$SHARDS/$d.json.gz" ] || after=$(( after + 1 )); done
    if [ "$after" -ge "$before" ]; then
        echo "[x50] $(date -Is) chunk made no progress (rc=$rc): HOLD"; exit 4
    fi
done
