#!/usr/bin/env bash
# The model-input axis end to end, ONE process at a time (2 threads): the injection teeth on each checkpoint, then
# the runner on each. Stops at the first failure (a teeth FAIL exits 4 / 3). Resumable: rerun it.
#   setsid nohup ./chain.sh > ~/gen3ai_archive/hidden_info_binding/chain.log 2>&1 &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$HERE/run_at_pin.sh"
STEPS="${HIB_STEPS:-teeth run}"
for step in $STEPS; do
  for cp in end2 prod end2_s2002; do
    pin="$cp"; ck=""
    if [ "$cp" = end2_s2002 ]; then pin=end2; ck=end2_s2002; fi
    echo "=== $(date -Is) $step $cp"
    if [ "$step" = teeth ]; then
      HIB_CKPT="$ck" "$R" "$pin" 2 8 2h inject_teeth.py --need 64 --max 200
    else
      HIB_CKPT="$ck" "$R" "$pin" 2 8 20h model_input.py
    fi
  done
done
echo "=== $(date -Is) chain done"
