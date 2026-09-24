#!/usr/bin/env bash
# Interleaved A/B of the fork-clone bench: BEFORE (pre-split session) vs AFTER (engine split).
# usage: run_pairs.sh <before_bin> <after_bin> <transcript> <pairs> <rows.jsonl>
# Each run appends ONE durable row: {"arm","pair","load1_start","load1_end","result":{...}}.
# Resumable: pairs already present in <rows.jsonl> for both arms are skipped.
set -euo pipefail
B=$1; A=$2; T=$3; N=$4; OUT=$5
touch "$OUT"
for p in $(seq 1 "$N"); do
  if [ $((p % 2)) -eq 1 ]; then order="before after"; else order="after before"; fi
  for arm in $order; do
    if grep -q "\"arm\":\"$arm\",\"pair\":$p," "$OUT"; then continue; fi
    bin=$B; [ "$arm" = after ] && bin=$A
    l0=$(cut -d' ' -f1 /proc/loadavg)
    res=$(nice -n 10 "$bin" 101 < "$T")
    l1=$(cut -d' ' -f1 /proc/loadavg)
    echo "{\"arm\":\"$arm\",\"pair\":$p,\"load1_start\":$l0,\"load1_end\":$l1,\"result\":$res}" >> "$OUT"
  done
done
