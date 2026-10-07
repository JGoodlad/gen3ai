#!/bin/bash
# P2 paired read at 60 games/team: the NEW stack (one engine, all 8 teams) beside the OLD stack (4 team processes at
# a time). Both schedules are prefix-consistent, so each 60-game cell contains its 20-game read.
cd /home/goodlad/dev/gen3ai/.claude/worktrees/agent-a95ca950ad27679ff || exit 9
D=designs/research_state/measurements/pokeenv_p2_ladder_2026-10-06/untaught
scripts/ops/mem_cap.sh 16 /tmp/claude-1000/py.sh /tmp/untaught_p2_compare.py new all 60 "$D/new_rust_eval_g60.json" \
  > /tmp/untaught_p2_new60.log 2>&1 &
NEW=$!
for batch in "0 1 2 3" "4 5 6 7"; do
  pids=()
  for t in $batch; do
    scripts/ops/mem_cap.sh 8 /tmp/claude-1000/py.sh /tmp/untaught_p2_compare.py old "$t" 60 "$D/old_python_bridge_g60_t$t.json" \
      > "/tmp/untaught_p2_old60_$t.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
wait "$NEW"
echo ALLDONE
