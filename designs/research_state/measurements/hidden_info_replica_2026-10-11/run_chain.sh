#!/usr/bin/env bash
# One worker: run the named arms for one checkpoint in order, under the brief's CPU rules while a training
# run is live (cores 5-7 + SMT siblings, nice 19, a memory cap). Resumable: re-running skips finished states.
#   run_chain.sh <cp> <shard> <nshards> <cpus> <arm> [<arm> ...]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
OUT="${HIR_OUT:-$HOME/gen3ai_archive/hidden_info_replica/rows}"
cp="$1"; shard="$2"; nshards="$3"; cpus="$4"; shift 4
for arm in "$@"; do
  echo "== $cp $arm shard $shard/$nshards $(date -Is)"
  PYTHONPATH="$HERE" "$REPO/scripts/ops/mem_cap.sh" --name "hir-$cp-$shard" 6 \
    taskset -c "$cpus" nice -n 19 "$PY" -m replica.play --cp "$cp" --arm "$arm" \
    --shard "$shard" --nshards "$nshards" --out "$OUT"
  echo "== done $cp $arm $(date -Is) exit $?"
done
