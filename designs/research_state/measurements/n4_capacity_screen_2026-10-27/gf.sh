#!/bin/bash
# N4 PREREG §6 item 3 (reported, never deciding): G-F, the 3x3 head-to-head N4 finals (players) x C0' finals (opponents),
# 500 mirrored pairs per cell, schedule seed 0, at the read's play commit 31bd9ced9 (one engine: two architectures).
# Plays the state's sha-checked copies; ledger outside models/. CPU, rule 5b, heavy slot (long lane).
set -u
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
PLAY=/home/goodlad/dev/gen3ai-reads/n4_play_31bd9ced
C=/home/goodlad/gen3ai_archive/n4_capacity_look1_2026-10-31/ckpts
OUT=/home/goodlad/gen3ai_archive/n4_capacity_look1_2026-10-31_ops/gf
export PYTHONPATH=$PLAY/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
cd "$PLAY"
echo "[$(date +%FT%T)] start gf"
nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --name n4gf 12 timeout 10800 "$PY" -m main.h2h play-many \
  --players $C/n4ffn_s3641/final_model.zip $C/n4ffn_s3642/final_model.zip $C/n4ffn_s3643/final_model.zip \
  --opponents $C/c0n4_s3641/final_model.zip $C/c0n4_s3642/final_model.zip $C/c0n4_s3643/final_model.zip \
  --pairs 500 --batch-pairs 100 --seed 0 --out "$OUT/ledger" --label n4_gf_look1 --purpose matrix \
  --device cpu --threads 1 --torch-threads 1 > "$OUT/gf.log" 2>&1
echo "[$(date +%FT%T)] end gf rc $?"
