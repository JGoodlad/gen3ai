#!/usr/bin/env bash
# The P3 matched comparison: each cell played on BOTH paths (core slot under the blocker; legacy
# poke-env client) at one seed base, one team seed, CPU, local servers on auto-picked 95XX ports.
WT=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a92f44ba7fef947d3
OUT=/tmp/p3work/gate
MODEL=/home/goodlad/dev/gen3ai/models/rb_x5ab_blob_s1008/final_model.zip
mkdir -p "$OUT"
cell() {  # name extra-args...
  local name=$1; shift
  "$WT/scripts/ops/mem_cap.sh" --name "p3$name-new" 12 timeout 3000 /tmp/p3work/read.sh 1 \
      --model "$MODEL" --opponent metamon:SmallRL --seed-base 914007 \
      --capture-dir "$OUT/$name/new/cap" --out "$OUT/$name/new" "$@" > "$OUT/$name.new.log" 2>&1 &
  local a=$!
  sleep 20
  "$WT/scripts/ops/mem_cap.sh" --name "p3$name-old" 12 timeout 3000 /tmp/p3work/read.sh 0 \
      --our-transport poke-env \
      --model "$MODEL" --opponent metamon:SmallRL --seed-base 914007 \
      --capture-dir "$OUT/$name/old/cap" --out "$OUT/$name/old" "$@" > "$OUT/$name.old.log" 2>&1 &
  local b=$!
  wait $a; echo "$name new rc=$?" >> "$OUT/status.txt"
  wait $b; echo "$name old rc=$?" >> "$OUT/status.txt"
}
cell away_greedy --regime greedy --teamset away --games 40
cell home_greedy --regime greedy --teamset home --games 40
GEN3AI_POLICY_SEED=11 cell away_t1seeded --regime greedy --our-temperature 1.0 --teamset away --games 20
echo DONE >> "$OUT/status.txt"
