#!/usr/bin/env bash
# (T-a') at N* = 256 after the B/C meters end (bounded 9 h), then part_t.sh nstar 256 (quiet box, gate slot held,
# quiet wait bounded by QWAIT).
L=/home/goodlad/.cache/gen3ai/tmp/sizing
e=$((SECONDS + 32400))
until grep -q "CHAINM_ALL_DONE" $L/chain_m.log || [ $SECONDS -ge $e ]; do sleep 60; done
grep -q "CHAINM_ALL_DONE" $L/chain_m.log || { echo "CHAINT_FAIL meters never ended $(date -Is)"; exit 1; }
echo "CHAINT nstar start $(date -Is)"
QWAIT=21600 bash /home/goodlad/dev/gen3ai-wt/m5-sizing/designs/research_state/measurements/m5_sizing/scripts/part_t.sh nstar 256
echo "CHAINT nstar exit=$? $(date -Is)"
