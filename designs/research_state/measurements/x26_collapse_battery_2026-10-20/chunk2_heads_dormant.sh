#!/usr/bin/env bash
# CHUNKS 2 / 3 of the X26 collapse battery (one heavy slot each, <= 45 min, forward passes only, CPU, rule 5b).
#     chunk2_heads_dormant.sh 2     the ride-along read of 10M (the reference) + the five read points on the Lane S
#                                   bank (its first 200 of 580 battles, so six checkpoints fit one 45-min chunk)
#     chunk2_heads_dormant.sh 3     B10: dormant.py on 9 checkpoints (X26's bank)
# The ride-along reader (main.ridealong_read: B5's A-head meters, B8's RND coverage / saturation, the ensemble) takes
# its cross-checkpoint reads against the FIRST checkpoint named, so the invocation leads with 10M.
# Read-only on models/; every output goes under $A. Run detached, after chunk 1 (which builds core_events).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
A="${A:-$HOME/gen3ai_archive/x26_collapse_battery_2026-10-20}"
C=/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints
P=$ROOT/designs/research_state/measurements/collapse_pilot_2026-10-17
cd "$ROOT"
ck() { echo "$C/checkpoint_${1}_steps.zip"; }
LIMIT=200

if [ "${1:-}" = 2 ]; then
  echo "=== $(date -Is) ride-along read 10M + 180M / 230M / 280M / 330M / 380M"
  "$HERE/run.sh" -m main.ridealong_read --out "$A/ridealong" --archive "" --no-rnd-choice --threads 3 --workers 1 \
    --limit-battles $LIMIT --checkpoint "$(ck 10000089)=X26@10M" --checkpoint "$(ck 180000055)=X26@180M" \
    --checkpoint "$(ck 230000074)=X26@230M" --checkpoint "$(ck 280000170)=X26@280M" \
    --checkpoint "$(ck 330000202)=X26@330M" --checkpoint "$(ck 380000195)=X26@380M" || echo "FAILED ridealong"
elif [ "${1:-}" = 3 ]; then
  if [ ! -e "$A/dormant.json" ]; then
    echo "=== $(date -Is) B10 dormant"
    "$HERE/run.sh" "$HERE/dormant.py" --bank "$P/bank_x26" --out "$A/dormant.json" --threads 3 \
      --ckpt "$(ck 10000089)=X26@10M" --ckpt "$(ck 30000045)=X26@30M" --ckpt "$(ck 90000067)=X26@90M" \
      --ckpt "$(ck 160000085)=X26@160M" --ckpt "$(ck 180000055)=X26@180M" --ckpt "$(ck 230000074)=X26@230M" \
      --ckpt "$(ck 280000170)=X26@280M" --ckpt "$(ck 330000202)=X26@330M" --ckpt "$(ck 380000195)=X26@380M" \
      || echo "FAILED dormant"
  fi
else
  echo "usage: $0 2|3"; exit 2
fi
echo "=== $(date -Is) DONE chunk ${1}"
