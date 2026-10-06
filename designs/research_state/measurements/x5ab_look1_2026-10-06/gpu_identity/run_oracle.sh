#!/usr/bin/env bash
# The one_sided identity cells, split: run.sh's first pass put oracle-species and oracle-full in ONE engine, which
# the pre-flight REFUSES (an oracle level is an architecture toggle: `oracle_reveal` 'species' vs 'full' are two
# groups, plus the opponent's = three). one = oracle-species vs blob, one2 = oracle-full vs fixed_mass.
# Usage as run.sh: DEVICES="cpu cuda" run_oracle.sh <scratch dir>
set -u
S=${1:?scratch dir}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
M=/home/goodlad/dev/gen3ai/models
B1=$M/rb_x5ab_blob_s1001/final_model.zip
F1=$M/rb_x5ab_fm_s1001/final_model.zip
OS=$M/rb_x5ab_oracle_sp_s1002/final_model.zip
OF=$M/rb_x5ab_oracle_full_s1002/final_model.zip
mkdir -p "$S"
printf '[["%s","%s"]]\n' "$OS" "$B1" > "$S/cells_one.json"
printf '[["%s","%s"]]\n' "$OF" "$F1" > "$S/cells_one2.json"
COMMON="--pairs 50 --batch-pairs 50 --seed 911 --label x5ab_look1_gpu_identity --n-envs 64 --oracle-reveal-mode one_sided"
rc=0
for dev in ${DEVICES:-cpu cuda}; do
  for mode in one one2; do
    out="$S/ledger_${dev}_${mode}"
    if [ "$dev" = cuda ]; then pre="scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24"; else pre="scripts/ops/mem_cap.sh 24"; fi
    echo "=== $dev $mode $(date -Is)"
    $pre "$PY" -m main.h2h play-many --cells "$S/cells_${mode}.json" $COMMON --device "$dev" --out "$out" \
      > "$S/${dev}_${mode}.log" 2>&1 || { echo "FAILED $dev $mode"; rc=1; }
    tail -3 "$S/${dev}_${mode}.log" | cut -c1-300
  done
done
exit $rc
