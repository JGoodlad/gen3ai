#!/usr/bin/env bash
# B then C at the memory-fix pin 7ef99979, each one GPU hold (arm.sh blocks on gpu_lock), untyped buckets
# (production's declaration at 256, D-18). A 180 s gap between the arms lets a queued holder (the switch
# agent's 256 pre-flight) take the lock first.
A=/home/goodlad/dev/gen3ai-wt/m5-sizing/designs/research_state/measurements/m5_sizing/scripts/arm.sh
PIN=7ef99979ad38f2676cace24dc29e546ccfc7c8da
L=/home/goodlad/.cache/gen3ai/tmp/sizing
cd /home/goodlad/dev/gen3ai
echo "CHAINBC B start $(date -Is)"
bash $A sizing_B_n256_e10_s1001 256 10 1001 - $PIN > $L/arm_sizing_B_n256_e10_s1001.log 2>&1
echo "CHAINBC B exit=$? $(date -Is)"
sleep 180
echo "CHAINBC C start $(date -Is)"
bash $A sizing_C_n256_e5_s1001 256 5 1001 - $PIN > $L/arm_sizing_C_n256_e5_s1001.log 2>&1
echo "CHAINBC C exit=$? $(date -Is)"
