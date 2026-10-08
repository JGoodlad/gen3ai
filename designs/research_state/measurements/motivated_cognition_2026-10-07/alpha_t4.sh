#!/usr/bin/env bash
# FOLLOW-UP T4 driver (README §0.7): the 8 fixed_mass finals at the 706fa536 pin, CPU, under a memory cap.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MODELS=${MODELS:-/home/goodlad/dev/gen3ai/models}
OPS=${OPS:-$HERE/../../../../scripts/ops}
CK=()
for s in 1001 1002 1003 1004 1005 1006 1007 1008; do
  r=rb_x5ab_fm_s$s; [ "$s" = 1006 ] && r=rb_x5ab_fm_s1006b
  CK+=(--ckpt "$MODELS/$r/final_model.zip=fm_s$s")
done
echo "=== $(date -Is) start"
"$OPS/mem_cap.sh" --name alphat4 16 timeout 1h nice -n 15 "$HERE/run_pin.sh" "$HERE/alpha_t4.py" \
  --out "$HERE" "${CK[@]}" || { echo "FAILED ($(date -Is))"; exit 4; }
echo "=== $(date -Is) DONE"
