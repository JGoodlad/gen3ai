#!/usr/bin/env bash
# The full extraction: all 16 X5 A/B finals (README §0.2), at the 706fa536 pin, CPU, under a memory cap.
# Resumable: re-run the same line; finished labels are skipped. Run DETACHED:
#   nohup setsid bash extract_all.sh > extract.log 2>&1 < /dev/null &
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MODELS=${MODELS:-/home/goodlad/dev/gen3ai/models}
OPS=${OPS:-$HERE/../../../../scripts/ops}
SEEDS="1001 1002 1003 1004 1005 1006 1007 1008"
run_of() { case "$1_$2" in fm_1006) echo rb_x5ab_fm_s1006b ;; *) echo "rb_x5ab_$1_s$2" ;; esac; }
CK=()
for s in $SEEDS; do CK+=(--ckpt "$MODELS/$(run_of blob "$s")/final_model.zip=blob_s$s"); done
for s in $SEEDS; do CK+=(--ckpt "$MODELS/$(run_of fm "$s")/final_model.zip=fm_s$s"); done
echo "=== $(date -Is) start"
"$OPS/mem_cap.sh" --name motivcog 24 timeout 3h nice -n 15 "$HERE/run_pin.sh" "$HERE/extract.py" \
  --out "$HERE/${ROWS:-rows}" --threads 4 --workers 2 "${CK[@]}" || { echo "FAILED ($(date -Is))"; exit 4; }
echo "=== $(date -Is) DONE"
