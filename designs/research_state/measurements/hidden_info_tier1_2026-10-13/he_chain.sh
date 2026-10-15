#!/usr/bin/env bash
# The hand-evaluator re-plays of R-EVAL (§6.5, §12.6), one job at a time in the world builder's slot once <pid> (the last
# build chain) has exited:  he_chain.sh <pid> <F's code>
# HE@O and HE@P (det2) at the unmodified pins (with the driver's identity check against the archived rows), HE@F at the
# PAD pins (F's worlds carry the gender pad). Resumable: each driver skips the states its directory already holds.
set -uo pipefail
PID="$1"; FCODE="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A=/home/goodlad/gen3ai_archive/hidden_info_tier1
LOGS=$A/logs
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for CP in end2 prod; do
  bash "$HERE/run_pin.sh" "$CP" 2 8 12h he_run.py --world O --out-root "$A/$CP/hand_eval" --identity \
    >> "$LOGS/he_O_$CP.log" 2>&1
  bash "$HERE/run_pin.sh" "$CP" 2 8 12h he_run.py --world P --worlds-dir "/home/goodlad/gen3ai_archive/value_disc_pilot/$CP/det2" \
    --out-root "$A/$CP/hand_eval" --identity >> "$LOGS/he_P_$CP.log" 2>&1
  bash "$HERE/run_pin.sh" "${CP}_pad" 2 8 12h he_run.py --world F --worlds-dir "$A/$CP/worlds/$FCODE" \
    --out-root "$A/$CP/hand_eval" >> "$LOGS/he_F_$CP.log" 2>&1
done
echo "he chain done $(date)" >> "$LOGS/he_O_end2.log"
