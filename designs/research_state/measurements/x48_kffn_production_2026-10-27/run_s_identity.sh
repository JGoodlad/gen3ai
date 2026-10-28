#!/bin/bash
# X48 look 1 (PREREG NOTE 4): the S lane's identity at the play pin 86cfdf7ef, before any X48 read game, as LR-F did
# (lrf_read/run_identity.sh): C0 s3501's G1 batch 0 re-played against x26_pstar and tstar, then compared with G1's
# banked rows by s_identity_compare.py. CPU, nice 19 on the agent cores, one torch thread.
set -u
W=/home/goodlad/dev/gen3ai-reads/x48_play_86cfdf7e
cd $W
export PYTHONPATH=$W/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
LEDGER=/home/goodlad/gen3ai_archive/x35_x48_read/s_identity/ledger
mkdir -p $LEDGER
nice -n 19 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m main.eval_ledger family-register --root $LEDGER \
  --id x35_x48_s_identity_2026-10-28 --decision-kind ab_verdict \
  --rule "main.reads S on the frozen R (population_score of design_compute_efficiency §5.6.4)" \
  --protocol gen3_eval_protocol_v1_h2h || exit 2
for m in x26_pstar/checkpoint_280000170_steps.zip tstar/checkpoint_290000087_steps.zip; do
  $W/scripts/ops/mem_cap.sh --slot --max-s 600 --name x48_s_identity 12 nice -n 19 taskset -c 5-7,13-15 \
    /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m main.h2h play-many \
    --players /home/goodlad/gen3ai_archive/x35_g1/ckpts/finals/c0_s3501/final_model.zip \
    --opponents /home/goodlad/gen3ai_archive/x35_g1/ckpts/members/$m --pairs 100 --batch-pairs 100 --purpose ab \
    --family x35_x48_s_identity_2026-10-28 --request x48_s_identity_c0_s3501_b100 \
    --out /home/goodlad/gen3ai_archive/x35_x48_read/s_identity/ledger --label reads_x35_x48_s_identity \
    --device cpu --torch-threads 1 --threads 2
  echo "exit $? for $m"
done
