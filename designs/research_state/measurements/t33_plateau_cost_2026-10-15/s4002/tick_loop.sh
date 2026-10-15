#!/bin/bash
# T33: operate the toy's Tier-1 plateau meter (Δ 5M, lag 4) one check at a time, timing each check.
# Rule 5b: py.sh pins taskset -c 5-7,13-15 + nice -n 19; mem_cap gives each check its own capped scope.
# Exits when no check is due and the run's newest checkpoint is > 60 min old (the run ended / stopped).
RUN=${1:-rb_toy_ent_s4001}
WT=${WT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
MAIN=/home/goodlad/dev/gen3ai
D=/tmp/t33_s4002
TLOG=$D/ticks.tsv
PLAN="--delta-steps 5000000 --lag 4"
COMPUTE="--threads 4 --torch-threads 2 --n-envs 64"
[ -f "$TLOG" ] || printf "start\tend\twall_s\tuser_s\tsys_s\tmaxrss_kb\trc\tload1\n" > "$TLOG"
while true; do
  due=$($D/py.sh -m main.plateau tick "$RUN" $PLAN --dry-run 2>&1 >/dev/null | sed -n 's/.*: \([0-9]*\) due check.*/\1/p')
  if [ "${due:-0}" -gt 0 ]; then
    s=$(date +%s); l=$(cut -d' ' -f1 /proc/loadavg)
    "$MAIN/scripts/ops/mem_cap.sh" --name t33s4002tick 12 /usr/bin/time -o $D/time.tmp -f "%e\t%U\t%S\t%M" \
      $D/py.sh -m main.plateau tick "$RUN" $PLAN $COMPUTE --max-checks 1 >> $D/tick_stdout.log 2>> $D/tick_stderr.log
    rc=$?; e=$(date +%s)
    printf "%s\t%s\t%s\t%s\t%s\n" "$s" "$e" "$(tail -1 $D/time.tmp)" "$rc" "$l" >> "$TLOG"
    $D/py.sh -m main.plateau status "$RUN" $PLAN --json > $D/status_$e.json 2>/dev/null
    $D/py.sh -m main.plateau status "$RUN" $PLAN > $D/status_latest.txt 2>/dev/null
    [ "$rc" -ne 0 ] && { echo "tick rc=$rc at $e; stopping the loop" >> $D/tick_stderr.log; exit "$rc"; }
  else
    newest=$(ls -t "$MAIN/models/$RUN/checkpoints/"checkpoint_*_steps.zip 2>/dev/null | head -1)
    age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
    if [ "$age" -gt 3600 ]; then echo "no due check and newest checkpoint ${age}s old: done" >> $D/tick_stderr.log; exit 0; fi
    sleep 120
  fi
done
