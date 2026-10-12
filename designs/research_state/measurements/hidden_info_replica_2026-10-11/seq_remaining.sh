#!/usr/bin/env bash
# ONE worker: for each <cp>:<arm>, list the states with no row yet and run them (row-file tag <tag>).
# Used when the CPU allowance is cut (rule 5b: <= 3 workers on cores 5-7 / 13-15 at nice 19).
#   seq_remaining.sh <tag> <cpus> <cp>:<arm>[:<first N>] ...   (first N: only the registered prefix)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
ROWS="$HOME/gen3ai_archive/hidden_info_replica/rows"
tag="$1"; cpus="$2"; shift 2
for job in "$@"; do
  IFS=: read -r cp arm first <<< "$job"
  first="${first:-0}"
  ids="$HOME/gen3ai_archive/hidden_info_replica/remaining_${cp}_${arm}${tag}_${first}.txt"
  only=""; if [ "$first" = "interim" ]; then only="$HERE/interim_ids_${cp}.txt"; first=0; fi
  PYTHONPATH="$HERE" "$PY" -m replica.remaining --cp "$cp" --arm "$arm" --rows "$ROWS" --out "$ids" --first "$first" ${only:+--only "$only"}
  echo "== $cp $arm $(date -Is)"
  PYTHONPATH="$HERE" "$REPO/scripts/ops/mem_cap.sh" --name "hir-$cp-seq" 6 \
    taskset -c "$cpus" nice -n 19 "$PY" -m replica.play --cp "$cp" --arm "$arm" \
    --shard 0 --nshards 1 --out "$ROWS" --ids "$ids" --tag "$tag"
  echo "== done $cp $arm $(date -Is)"
done
