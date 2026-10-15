#!/bin/bash
# T33: production-network CPU game rate with ONE 160-pair batch (vs the plateau's 40-pair batches): the tail effect.
mkdir -p /tmp/t33/ledger_big
M=/home/goodlad/dev/gen3ai/models/rb_bf_eb_s2001/checkpoints
/home/goodlad/dev/gen3ai/scripts/ops/mem_cap.sh --name t33big 12 timeout 1500 \
  /usr/bin/time -o /tmp/t33/time_big.txt -f "%e	%U	%S	%M" \
  "$(dirname "$0")"/py.sh -m main.h2h play --player $M/checkpoint_15000029_steps.zip --opponent $M/checkpoint_10000194_steps.zip \
  --pairs "${1:-160}" --batch-pairs "${1:-160}" --out /tmp/t33/ledger_big --label t33_rate --purpose audit \
  --threads 4 --torch-threads 2 --n-envs 64 > /tmp/t33/big_stdout.log 2> /tmp/t33/big_stderr.log
echo "rc=$?"
