#!/usr/bin/env bash
# F-ED-22 before/after: perfbench.py on the pre-fix code (the main checkout at ecf9eeca, no index) and the fixed code
# (this worktree), on tmpfs (CPU shape) and on the NVMe the archive lives on (fsync-bound), at 1.7k / 5.1k / 50k events.
#   bash run_all.sh <before_src> <after_src> <out.jsonl>
set -u
BEFORE_SRC=${1:?before src}; AFTER_SRC=${2:?after src}; OUT=${3:?out}
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
BENCH="$(cd "$(dirname "$0")" && pwd)/perfbench.py"
export PYTHONDONTWRITEBYTECODE=1
for fs in tmpfs nvme; do
  if [ "$fs" = tmpfs ]; then T=/tmp; else T=/home/goodlad/.cache/gen3ai/tmp; mkdir -p "$T"; fi
  for phase in before after; do
    SRC=$BEFORE_SRC; [ "$phase" = after ] && SRC=$AFTER_SRC
    for n in 1700 5100 50000; do
      r=5; [ "$n" -ge 50000 ] && [ "$phase" = before ] && r=3
      TMPDIR=$T PYTHONPATH=$SRC "$PY" "$BENCH" --events "$n" --reps "$r" --label "$phase-$fs" --out "$OUT"
    done
  done
done
