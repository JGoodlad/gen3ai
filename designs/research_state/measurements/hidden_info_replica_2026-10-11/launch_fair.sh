#!/usr/bin/env bash
# Launch the screened arms on the freed box (after the closing test, 2026-10-12 09:00): F and F+M for both
# checkpoints, sharded, every worker on the given CPU list at nice 19 under a memory cap.
#   launch_fair.sh <cpus> <shards_end2> <shards_prod>
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HOME/gen3ai_archive/hidden_info_replica/logs"
cpus="$1"; ne="$2"; np="$3"
for arm in F F+M; do
  for ((i = 0; i < ne; i++)); do
    nohup "$HERE/run_chain.sh" end2 "$i" "$ne" "$cpus" "$arm" > "$LOG/final_end2_${arm}_${i}of${ne}.log" 2>&1 &
    echo "end2 $arm $i/$ne pid $!"
  done
  for ((i = 0; i < np; i++)); do
    nohup "$HERE/run_chain.sh" prod "$i" "$np" "$cpus" "$arm" > "$LOG/final_prod_${arm}_${i}of${np}.log" 2>&1 &
    echo "prod $arm $i/$np pid $!"
  done
done
