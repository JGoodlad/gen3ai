#!/bin/bash
# T33: time ONE production-network Tier-1 check on the agent cores (scratch ledger, not the archive's).
mkdir -p /tmp/t33/ledger_prod
/home/goodlad/dev/gen3ai/scripts/ops/mem_cap.sh --name t33prod 12 timeout 900 \
  /usr/bin/time -o /tmp/t33/time_prod.txt -f "%e	%U	%S	%M" \
  "$(dirname "$0")"/py.sh -m main.plateau tick "${1:-rb_bf_eb_s2001}" --delta-steps 5000000 --lag 2 \
  --out /tmp/t33/ledger_prod --threads 4 --torch-threads 2 --n-envs 64 --max-checks 1 \
  > /tmp/t33/prod_stdout.log 2> /tmp/t33/prod_stderr.log
echo "rc=$?"
