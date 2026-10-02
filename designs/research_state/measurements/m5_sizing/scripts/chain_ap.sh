#!/usr/bin/env bash
# A' after the switch-prep holds: drain (bounded 120 min; free on 3 checks 60 s apart), then the arm.
G=/home/goodlad/dev/gen3ai-wt/m5-sizing/scripts/ops/gpu_lock.sh
e=$((SECONDS + 7200)); f=0
sleep 120
while [ $SECONDS -lt $e ] && [ $f -lt 3 ]; do
  if "$G" --status 2>&1 | grep -q "held by"; then f=0; else f=$((f + 1)); fi
  sleep 60
done
echo "CHAINAP drain done free=$f $(date -Is)"
cd /home/goodlad/dev/gen3ai
bash /home/goodlad/dev/gen3ai-wt/m5-sizing/designs/research_state/measurements/m5_sizing/scripts/arm.sh \
    sizing_Ap_n48_e10_s1002 48 10 1002 - > /home/goodlad/.cache/gen3ai/tmp/sizing/arm_sizing_Ap_n48_e10_s1002.log 2>&1
echo "CHAINAP arm exit=$? $(date -Is)"
