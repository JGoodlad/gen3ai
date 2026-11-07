#!/bin/bash
# S'-HONEST stage 1b: one run of seq_probe.py, in bounded chunks (rule 5b: each chunk <= 45 min under mem_cap, a heavy
# slot held per chunk; the probe resumes by group, so a chunk killed at its bound loses only its unfinished group).
# Usage: run.sh <checkout> <out dir> <shard i/k> [seq_probe run args...]
set -u
WT=$1; OUT=$2; SHARD=$3; shift 3
MODEL=/home/goodlad/gen3ai_archive/sprime_live_play_2026-11-01/ckpts/pstar_s/checkpoint_280000170_steps.zip
SHA=814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed
cd "$WT" || exit 90
export PYTHONPATH=$WT/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
for chunk in $(seq 1 40); do
  echo "[run] chunk $chunk $(date -Is)" >> "$OUT/run.$(echo "$SHARD" | tr / _).log"
  nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 timeout 2700 "$PY" \
    designs/research_state/measurements/sprime_honest_inc_stage1b_2026-11-06/seq_probe.py run \
    --model "$MODEL" --model-sha256 "$SHA" --out "$OUT" --shard "$SHARD" "$@" \
    >> "$OUT/run.$(echo "$SHARD" | tr / _).log" 2>&1
  rc=$?
  echo "[run] chunk $chunk exit $rc $(date -Is)" >> "$OUT/run.$(echo "$SHARD" | tr / _).log"
  [ $rc -eq 0 ] && break
  [ $rc -ne 124 ] && [ $rc -ne 88 ] && exit $rc
done
