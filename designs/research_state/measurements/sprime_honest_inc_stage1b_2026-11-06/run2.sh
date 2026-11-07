#!/bin/bash
# S'-HONEST stage 1b: TWO probe workers inside ONE mem_cap scope (one heavy slot: each worker peaks at ~1 GB of the
# scope's 8 GB), in bounded chunks like run.sh. Usage: run2.sh <checkout> <out dir> <shard a/k> <shard b/k> [args...]
set -u
WT=$1; OUT=$2; SA=$3; SB=$4; shift 4
MODEL=/home/goodlad/gen3ai_archive/sprime_live_play_2026-11-01/ckpts/pstar_s/checkpoint_280000170_steps.zip
SHA=814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed
cd "$WT" || exit 90
export PYTHONPATH=$WT/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=$WT/designs/research_state/measurements/sprime_honest_inc_stage1b_2026-11-06/seq_probe.py
LOG="$OUT/run2.$(echo "$SA" | tr / _).log"
mkdir -p "$OUT"
for chunk in $(seq 1 60); do
  echo "[run2] chunk $chunk $(date -Is)" >> "$LOG"
  nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 timeout 2700 bash -c \
    "\"$PY\" $P run --model $MODEL --model-sha256 $SHA --out $OUT --shard $SA $* & a=\$!;
     \"$PY\" $P run --model $MODEL --model-sha256 $SHA --out $OUT --shard $SB $* & b=\$!;
     wait \$a; ra=\$?; wait \$b; rb=\$?; [ \$ra -eq 0 ] && [ \$rb -eq 0 ]" >> "$LOG" 2>&1
  rc=$?
  echo "[run2] chunk $chunk exit $rc $(date -Is)" >> "$LOG"
  [ $rc -eq 0 ] && break
  [ $rc -ne 124 ] && [ $rc -ne 88 ] && [ $rc -ne 1 ] && exit $rc
done
