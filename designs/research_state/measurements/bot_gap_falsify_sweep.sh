#!/bin/bash
# SI-2 measurement 1: falsify-scan every bot-loss cell in rev1's last 3 eval steps.
cd /home/goodlad/dev/gen3ai/.claude/worktrees/simplify-p0
export PYTHONPATH=$PYTHONPATH:src
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=tmp/si2/falsify
mkdir -p $OUT
RUN=/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823
for s in 20000016 22000032 24000000; do
  for b in heuristic heuristic2 staller staller_v2 aggressive aggressive_v2 setup_sweep setup_sweep_v2 random; do
    f=$OUT/falsify_${s}_${b}.json
    [ -s "$f" ] && continue
    echo "=== $s $b $(date +%H:%M:%S)"
    nice -n 15 $PY -m main.prober.query --impl rust falsify-scan $RUN --opponent $b --step $s --limit 20 > "$f" 2> $OUT/falsify_${s}_${b}.err || echo "FAILED $s $b"
  done
done
echo DONE $(date +%H:%M:%S)
