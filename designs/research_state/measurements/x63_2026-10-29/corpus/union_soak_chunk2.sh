#!/usr/bin/env bash
# X63 union soak, chunk 2 of 2 (rounds 5-9, seed 3802+1), then the merged record (N1-P's 10-round dose: 178,760 battles).
set -euo pipefail
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
M=designs/research_state/measurements/x63_2026-10-29/corpus
echo "== soak chunk 2 $(date -Is)"
"$PY" -m utils.team_corpora.soak x63_union_v1 --rounds 5 --first-round 5 --seed 3803 --out $M/soak_chunk_r5-9_s3803.json | tail -c 600
echo
"$PY" -m utils.team_corpora.soak merge $M/soak_chunk_r0-4_s3802.json $M/soak_chunk_r5-9_s3803.json \
    --out src/utils/team_corpora/x63_union_v1/soak_selfcheck.json
echo "== done $(date -Is)"
