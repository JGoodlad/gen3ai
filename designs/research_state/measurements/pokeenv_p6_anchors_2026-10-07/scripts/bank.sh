#!/usr/bin/env bash
# Bank the reads' small artifacts (games.jsonl + summary.json per path) and the analyses beside this README.
# The captures (cap/) and the raw shadow logs are NOT banked (tens of MB); analyze.py / shadow_digest.py re-derive
# every count from them. usage: bank.sh <work dir (holds gate_a, gate_b, gate_c)> <dest = the measurement dir>
W=$1; D=$2
here=$(cd "$(dirname "$0")" && pwd)
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
for gate in gate_a gate_c; do
  for cell in "$W/$gate"/*/; do
    c=$(basename "$cell")
    for side in new old; do
      [ -f "$cell/$side/summary.json" ] || continue
      mkdir -p "$D/$gate/$c/$side"
      cp "$cell/$side/games.jsonl" "$cell/$side/summary.json" "$D/$gate/$c/$side/"
    done
  done
done
for r in old_shadow new_node_model new_node_bot new_master_uri_model; do
  mkdir -p "$D/gate_b/$r"
  cp "$W/gate_b/$r/games.jsonl" "$W/gate_b/$r/summary.json" "$D/gate_b/$r/"
done
GEN3AI_SRC="$(cd "$here/../../../../../src" && pwd)"
export GEN3AI_SRC
"$PY" "$here/analyze.py" "$W/gate_a" "$D/gate_a/analysis.json" > /dev/null
"$PY" "$here/analyze.py" "$W/gate_c" "$D/gate_c/analysis.json" > /dev/null
"$PY" "$here/shadow_digest.py" "$W/gate_b/old_shadow" "$D/gate_b/old_shadow/shadow_digest.json" > /dev/null
"$PY" "$here/race_kinds.py" "$W/gate_b/old_shadow" > "$D/gate_b/old_shadow/race_kinds.json"
"$PY" "$here/read_digest.py" "$W/gate_b/old_shadow" "$W/gate_b/new_node_model" "$W/gate_b/new_node_bot" \
    "$W/gate_b/new_master_uri_model" --json "$D/gate_b/read_digest.json" > /dev/null
