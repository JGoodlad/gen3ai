#!/usr/bin/env bash
# The PAD-PIN re-gate of every arm on one checkpoint (README §2): 60 registered states per arm, and 30 supplement states
# per arm that runs on the supplement. No outcome. One job per checkpoint.
#   regate_chain.sh <cp>
set -uo pipefail
CP="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS=/home/goodlad/gen3ai_archive/hidden_info_tier1/logs
cd "$HERE"
CODES=$(/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -I -c "
import sys; sys.path.insert(0, '.')
import t1_common as T
k = T.load_key()
print(' '.join(f'{k[l]}:{T.ARMS[l][\"states\"]}' for l in T.ARMS))")
for cs in $CODES; do
  C="${cs%%:*}"; S="${cs##*:}"
  bash "$HERE/run_pin.sh" "${CP}_pad" 1 6 2h pin_regate.py --code "$C" --n 60 --out "pin_regate_${CP}_${C}.json" \
    >> "$LOGS/regate_${CP}.log" 2>&1
  if [ "$S" = all ]; then
    bash "$HERE/run_pin.sh" "${CP}_pad" 1 6 2h pin_regate.py --code "$C" --states supp --n 30 \
      --out "pin_regate_${CP}_${C}_supp.json" >> "$LOGS/regate_${CP}.log" 2>&1
  fi
done
echo "regate chain done $(date)" >> "$LOGS/regate_${CP}.log"
