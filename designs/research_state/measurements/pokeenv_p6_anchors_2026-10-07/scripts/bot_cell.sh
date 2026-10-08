#!/usr/bin/env bash
# usage: bot_cell.sh <out root> <cell name> <bot> <anchors args...>
# One cell of the BOT identity proof (gate (a)): the SAME seeded read on BOTH paths, concurrently, on the
# `--server rust` front end (each its own auto-picked 95XX port):
#   new = `--our-transport live` (the Rust bot over the bot_reader session) under the poke-env import blocker;
#   old = `--our-transport poke-env` (the legacy client + the Python bot, streams seeded by legacy_seeded_bot.py).
# Same --seed-base (each battle's sim seed), --team-seed (both sides' draws), --bot-seed (the bot's streams).
# PORT=<p> pins the two front ends to p (new) and p+1 (old): two cells started at once otherwise race for the same
# auto-picked port (the first batch lost 6 of 18 reads to `address already in use`, rc 3, before a game).
here=$(cd "$(dirname "$0")" && pwd)
OUT=$1; name=$2; bot=$3; shift 3
pn=(); po=()
if [ -n "$PORT" ]; then pn=(--port "$PORT"); po=(--port $((PORT + 1))); fi
mkdir -p "$OUT/$name"
rm -rf "$OUT/$name/new" "$OUT/$name/old"
timeout 3000 "$here/read.sh" 1 --our-side "bot:$bot" "${pn[@]}" --capture-dir "$OUT/$name/new/cap" \
    --out "$OUT/$name/new" "$@" > "$OUT/$name.new.log" 2>&1 &
a=$!
sleep 15
timeout 3000 "$here/read.sh" legacybot --our-transport poke-env --our-side "bot:$bot" "${po[@]}" \
    --capture-dir "$OUT/$name/old/cap" --out "$OUT/$name/old" "$@" > "$OUT/$name.old.log" 2>&1 &
b=$!
wait $a; echo "$name new rc=$?" >> "$OUT/status.txt"
wait $b; echo "$name old rc=$?" >> "$OUT/status.txt"
