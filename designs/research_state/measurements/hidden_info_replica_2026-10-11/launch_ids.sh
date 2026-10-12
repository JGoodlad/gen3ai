#!/usr/bin/env bash
# Re-shard the remaining states of one arm over n workers (a new row-file tag, so files never collide).
#   launch_ids.sh <cp> <arm> <ids file> <tag> <n> <cpus>
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HOME/gen3ai_archive/hidden_info_replica/logs"
cp="$1"; arm="$2"; ids="$3"; tag="$4"; n="$5"; cpus="$6"
for ((i = 0; i < n; i++)); do
  HIR_IDS="$ids" HIR_TAG="$tag" nohup "$HERE/run_chain.sh" "$cp" "$i" "$n" "$cpus" "$arm" \
    > "$LOG/final_${cp}_${arm}_${i}of${n}${tag}.log" 2>&1 &
  echo "$cp $arm $i/$n$tag pid $!"
done
