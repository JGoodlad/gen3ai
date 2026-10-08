#!/usr/bin/env bash
# usage: model_cell.sh <out root> <cell name> <port> <anchors args...>
# The CHECKPOINT on the live client vs the legacy RLPlayer client, the SAME seeded read on the `--server rust`
# front end (subprocess form for both; ports p / p+1): new = `--our-transport live` under the poke-env blocker,
# old = `--our-transport poke-env` (no hook: a greedy checkpoint draws nothing). This is the client the
# `--server node` / `--server-uri` shapes now use, compared on byte-identical battles (a Node battle cannot be seeded).
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1; name=$2; port=$3; shift 3
mkdir -p "$OUT/$name"
rm -rf "$OUT/$name/new" "$OUT/$name/old"
timeout 3000 "$here/read.sh" 1 --our-transport live --port "$port" --capture-dir "$OUT/$name/new/cap" \
    --out "$OUT/$name/new" "$@" > "$OUT/$name.new.log" 2>&1 &
a=$!
sleep 15
timeout 3000 "$here/read.sh" 0 --our-transport poke-env --port $((port + 1)) --capture-dir "$OUT/$name/old/cap" \
    --out "$OUT/$name/old" "$@" > "$OUT/$name.old.log" 2>&1 &
b=$!
wait $a; echo "$name new rc=$?" >> "$OUT/status.txt"
wait $b; echo "$name old rc=$?" >> "$OUT/status.txt"
