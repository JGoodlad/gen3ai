#!/bin/bash
# D3 look 2, AM-D3-L2 item 11: C0 s3501 batch 0 vs x26_pstar / tstar at the pin, on the pool (G1) and on home72 (D1 P).
PIN=/home/goodlad/dev/gen3ai-reads/d3l2_pin
OUT=/home/goodlad/gen3ai_archive/x41_d3_read/look2_s_identity
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
FAM=x41_d3_l2_s_identity_2026-10-28
RULE="main.reads S on the frozen R (population_score of design_compute_efficiency §5.6.4)"
cd $PIN
export PYTHONPATH=$PIN/src CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
A=/home/goodlad/gen3ai_archive/x35_g1/ckpts
for lane in pool home72; do
  mkdir -p $OUT/$lane/ledger
  $PY -m main.eval_ledger family-register --root $OUT/$lane/ledger --id $FAM --decision-kind ab_verdict --rule "$RULE" --protocol gen3_eval_protocol_v1_h2h
  echo "family-register $lane exit $?"
  PANEL=(); [ $lane = home72 ] && PANEL=(--panel home72)
  for m in x26_pstar/checkpoint_280000170_steps.zip tstar/checkpoint_290000087_steps.zip; do
    $PIN/scripts/ops/mem_cap.sh --slot --max-s 600 --name d3l2_s_identity 12 nice -n 19 taskset -c 5-7,13-15 \
      $PY -m main.h2h play-many --players $A/finals/c0_s3501/final_model.zip --opponents $A/members/$m \
      --pairs 100 --batch-pairs 100 --purpose ab --family $FAM --request d3l2_s_identity_c0_s3501_${lane}_b100 \
      --out $OUT/$lane/ledger --label reads_x41_d3_l2_s_identity --device cpu --torch-threads 1 --threads 2 "${PANEL[@]}"
    echo "exit $? for $lane $m"
  done
done
