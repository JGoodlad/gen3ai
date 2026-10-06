#!/usr/bin/env bash
# Step 1 of X5 A/B look 1: the DEFERRED GPU identity checks. play-many (a same-architecture cell on a two-group
# engine), the two-architecture cross and the per-side oracle reveal (one_sided + both_sided), each played on the
# CPU (T2 eager) and on the GPU (T2 graph) on the SAME seeds, into scratch ledger roots OUTSIDE models/ (schedule
# seed 911, not the read's seed 0). compare.py then requires identical rows.
# Usage (repo root, PYTHONPATH=$PWD/src, GEN3AI_GPU_LEASE_TOKEN set): run.sh <scratch dir>
set -u
S=${1:?scratch dir}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
M=/home/goodlad/dev/gen3ai/models
B1=$M/rb_x5ab_blob_s1001/final_model.zip
B2=$M/rb_x5ab_blob_s1002/final_model.zip
F1=$M/rb_x5ab_fm_s1001/final_model.zip
OS=$M/rb_x5ab_oracle_sp_s1002/final_model.zip
OF=$M/rb_x5ab_oracle_full_s1002/final_model.zip
mkdir -p "$S"
printf '[["%s","%s"],["%s","%s"]]\n' "$F1" "$B1" "$B2" "$B1" > "$S/cells_off.json"
printf '[["%s","%s"],["%s","%s"]]\n' "$OS" "$B1" "$OF" "$F1" > "$S/cells_one.json"
printf '[["%s","%s"]]\n' "$OS" "$F1" > "$S/cells_both.json"
COMMON="--pairs 50 --batch-pairs 50 --seed 911 --label x5ab_look1_gpu_identity --n-envs 64"
rc=0
for dev in ${DEVICES:-cpu cuda}; do
  for mode in off one both; do
    case $mode in off) m=off;; one) m=one_sided;; both) m=both_sided;; esac
    out="$S/ledger_${dev}_${mode}"
    if [ "$dev" = cuda ]; then pre="scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24"; else pre="scripts/ops/mem_cap.sh 24"; fi
    echo "=== $dev $m $(date -Is)"
    $pre "$PY" -m main.h2h play-many --cells "$S/cells_${mode}.json" $COMMON --device "$dev" \
      --oracle-reveal-mode "$m" --out "$out" > "$S/${dev}_${mode}.log" 2>&1 || { echo "FAILED $dev $m"; rc=1; }
    tail -3 "$S/${dev}_${mode}.log" | cut -c1-300
  done
done
exit $rc
