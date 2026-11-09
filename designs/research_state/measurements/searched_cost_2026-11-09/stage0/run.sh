#!/bin/bash
# Q19 stage 0: one arm of screen.py, in bounded chunks (rule 5b: each chunk under mem_cap.sh holding a heavy slot,
# bounded by `timeout`; the screen resumes by battle and mid-battle from the saved carry, so a chunk killed at its
# bound loses only the decision it was on). CHUNK_S <= 600 runs in mem_cap's short lane (--max-s CHUNK_S + 30 s hard
# kill, exit 88 = overrun, re-run); above it, a long chunk (the draft's <= 45 min).
# Usage: CHUNK_S=2700 run.sh <checkout> <out dir> <shard i/k> --preset P@10 --sample sample.json [--salt S] [...]
set -u
WT=$1; OUT=$2; SHARD=$3; shift 3
CHUNK_S=${CHUNK_S:-2700}
MAX_CHUNKS=${MAX_CHUNKS:-200}
MODEL=/home/goodlad/gen3ai_archive/sprime_live_play_2026-11-01/ckpts/pstar_s/checkpoint_280000170_steps.zip
SHA=814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed
ENGINE='port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130'
cd "$WT" || exit 90
export PYTHONPATH=$WT/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LANE=()
if [ "$CHUNK_S" -le 600 ]; then LANE=(--max-s "$CHUNK_S"); fi
mkdir -p "$OUT"
LOG="$OUT/run.$(echo "$SHARD" | tr / _).log"
for chunk in $(seq 1 "$MAX_CHUNKS"); do
  echo "[run] chunk $chunk $(date -Is)" >> "$LOG"
  nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh "${LANE[@]}" 8 timeout "$((CHUNK_S - 15))" "$PY" \
    designs/research_state/measurements/searched_cost_2026-11-09/stage0/screen.py run \
    --model "$MODEL" --model-sha256 "$SHA" --engine "$ENGINE" --out "$OUT" --shard "$SHARD" "$@" >> "$LOG" 2>&1
  rc=$?
  echo "[run] chunk $chunk exit $rc $(date -Is)" >> "$LOG"
  [ $rc -eq 0 ] && break
  [ $rc -ne 124 ] && [ $rc -ne 88 ] && exit $rc
done
