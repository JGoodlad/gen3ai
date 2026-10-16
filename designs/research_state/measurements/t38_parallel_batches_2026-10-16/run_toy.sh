#!/bin/bash
# T38 toy speed-up: the same recorded T-ENT check (rb_toy_ent_s4002 65M vs 45M, recorded GAIN at 680 pairs) ticked
# into scratch ledgers at several in-flight counts, sequentially, on the agent cores (rule 5b).
WT=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a45be0fb7c4b9b031
cd $WT
for K in ${KS:-1 4 2 8}; do
  out=/tmp/t38/ledger_toy_k$K
  rm -rf $out
  echo "=== K=$K start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
  /usr/bin/time -f "WALL %e s USER %U SYS %S MAXRSS %M KB" \
    $WT/scripts/ops/mem_cap.sh --name t38toy 16 /tmp/t38_py.sh -m main.plateau tick /tmp/t38/runs/rb_toy_ent_s4002 \
      --delta-steps 5000000 --lag 4 --out $out --threads 4 --torch-threads 2 --n-envs 64 --inflight-batches $K \
      > /tmp/t38/toy_k$K.log 2>&1
  echo "=== K=$K exit $? end $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
  tail -3 /tmp/t38/toy_k$K.log
done
