#!/bin/bash
cd /home/goodlad/dev/gen3ai-reads/lrf_play_86cfdf7e
export PYTHONPATH=/home/goodlad/dev/gen3ai-reads/lrf_play_86cfdf7e/src CUDA_VISIBLE_DEVICES=""
for m in x26_pstar/checkpoint_280000170_steps.zip tstar/checkpoint_290000087_steps.zip; do
  /home/goodlad/dev/gen3ai-reads/lrf_play_86cfdf7e/scripts/ops/mem_cap.sh --slot --max-s 600 --name lrf_s_identity 12 nice -n 19 taskset -c 5-7,13-15     /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m main.h2h play-many --players /home/goodlad/gen3ai_archive/x35_g1/ckpts/finals/c0_s3501/final_model.zip     --opponents /home/goodlad/gen3ai_archive/x35_g1/ckpts/members/$m --pairs 100 --batch-pairs 100 --purpose ab --family x35_lrf_s_identity_2026-10-27     --request lrf_s_identity_c0_s3501_b100 --out /home/goodlad/gen3ai_archive/x35_lrf_read/s_identity/ledger --label reads_x35_lrf_s_identity --device cpu --torch-threads 1 --threads 2
  echo "exit $? for $m"
done
