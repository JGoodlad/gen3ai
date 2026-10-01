#!/usr/bin/env bash
# K8 acceptance (TF32, torch 2.8, the learner benchmark's pinned buffer, arm C's checkpoint):
# the time stage of the compile inventory, prewarm + lock KEPT (the real arm_compile_sentinel, so
# the K8 tree installs its regions), one unbracketed full update. Interleaved A (main, pre-regions)
# / B (K8). Resumable: a unit with a status row is skipped.
L=/home/goodlad/gen3ai_archive/k6_k8/accept
BUF=/home/goodlad/gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
while kill -0 2854123 2>/dev/null; do sleep 30; done      # the K8.3 routine gate: keep the box quiet
for unit in A1:k6c B1:k84 A2:k6c B2:k84; do
  name=${unit%%:*}; wt=/home/goodlad/dev/gen3ai-wt/${unit##*:}
  grep -q "^$name " $L/status 2>/dev/null && continue
  cd "$wt" || exit 2
  git -C "$wt" rev-parse --short HEAD > $L/$name.head
  $wt/scripts/ops/gpu_lock.sh timeout 1200 $wt/scripts/ops/mem_cap.sh --name accept$name 40 \
    env PYTHONPATH=$wt/src $PY -m main.compile_inventory run --stage time --device cuda \
    --matmul-precision high --buffer $BUF --keep-prewarm --unbracketed --worker-timeout-min 19 \
    --out-root $L/$name > $L/$name.log 2>&1
  echo "$name exit=$? head=$(cat $L/$name.head) at=$(date -Is)" >> $L/status
done
