#!/usr/bin/env bash
# Gate (b) of P6, the NEW path's series (after legacy_shadow.py's old-path shadow read):
#   1. --server node (the pinned deps/pokemon-showdown this tool starts on 9561, stopped by PID), our checkpoint on the
#      live client, 20 games vs metamon:SmallRL greedy, home — under the poke-env import blocker;
#   2. --server node, our side bot:heuristic2 (the Rust bot on the live client), 10 games, away — blocked;
#   3. --server-uri: a Node Showdown at MASTER (P4's build, ~/.cache/gen3ai/p4/start_master.sh) started HERE on 9570
#      and stopped by its PID, our checkpoint on the live client, 10 games, away — blocked.
here=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/tmp/p6s2/work/gate_b}
MODEL=${MODEL:-$HOME/.cache/gen3ai/p4/models/p4_shadow_v144/final_model.zip}
mkdir -p "$OUT"
timeout 3000 "$here/read.sh" 1 --our-transport live --server node --port 9561 --model "$MODEL" --model-load bare \
    --opponent metamon:SmallRL --regime greedy --teamset home --games 20 --out "$OUT/new_node_model" \
    > "$OUT/new_node_model.log" 2>&1
echo "new_node_model rc=$?" >> "$OUT/status.txt"
timeout 3000 "$here/read.sh" 1 --server node --port 9562 --our-side bot:heuristic2 --bot-seed 77 \
    --opponent metamon:SmallRL --regime greedy --teamset away --games 10 --out "$OUT/new_node_bot" \
    > "$OUT/new_node_bot.log" 2>&1
echo "new_node_bot rc=$?" >> "$OUT/status.txt"
"$HOME/.cache/gen3ai/p4/start_master.sh" 9570 > "$OUT/master9570.log" 2>&1 &
mpid=$!
echo "master pid=$mpid" >> "$OUT/status.txt"
for _ in $(seq 1 120); do
  grep -q "Worker 1 now listening\|now listening" "$OUT/master9570.log" 2>/dev/null && break
  sleep 1
done
timeout 3000 "$here/read.sh" 1 --our-transport live --server-uri ws://localhost:9570/showdown/websocket \
    --model "$MODEL" --model-load bare --opponent metamon:SmallRL --regime greedy --teamset away --games 10 \
    --out "$OUT/new_master_uri_model" > "$OUT/new_master_uri_model.log" 2>&1
echo "new_master_uri_model rc=$?" >> "$OUT/status.txt"
kill "$mpid"
wait "$mpid" 2>/dev/null
echo "master stopped (pid $mpid)" >> "$OUT/status.txt"
echo DONE >> "$OUT/status.txt"
