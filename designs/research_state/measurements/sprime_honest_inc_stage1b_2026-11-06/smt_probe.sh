#!/bin/bash
# S'-HONEST stage 1b: a CHEAP SMT probe for the read's whole-box wall estimate (the coordinator, 2026-11-06). One fixed
# workload (the incremental builder's steps over one battle's first decisions, `smoke_group.py`: the builder only, the
# stub-free path, no network) timed pinned: ALONE on one logical CPU; TWO copies on two PHYSICAL cores; TWO copies on
# the two SMT SIBLINGS of one core (CPU n and n+8 share a core on this box). Twice each, interleaved. Inside the agent
# cores (rule 5b), one mem_cap scope for the pair. Prints one JSON line a run: config, per-copy wall and CPU seconds.
# Usage: smt_probe.sh <checkout> <workload.py> <group prefix> <stop turn> <out.jsonl>
set -u
WT=$1; WL=$2; G=$3; T=$4; OUT=$5
cd "$WT" || exit 90
export PYTHONPATH=$WT/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
one() {   # <cpu> <tag>: one pinned copy; prints "<wall> <user+sys of it and its children>"
  /usr/bin/time -f "%e %U %S" -o "/tmp/smt.$$.$2" nice -n 19 taskset -c "$1" "$PY" "$WL" "$G" "$T" > /dev/null 2>&1
  awk '{printf "%s %.2f\n", $1, $2 + $3}' "/tmp/smt.$$.$2"; rm -f "/tmp/smt.$$.$2"
}
for rep in 1 2; do
  for cfg in alone physical siblings; do
    case $cfg in
      alone) a=$(one 6 a); echo "{\"rep\":$rep,\"config\":\"alone\",\"cpus\":[6],\"runs\":[\"$a\"]}" ;;
      physical) one 6 a > /tmp/smt.$$.ra & one 7 b > /tmp/smt.$$.rb; wait
        echo "{\"rep\":$rep,\"config\":\"physical\",\"cpus\":[6,7],\"runs\":[\"$(cat /tmp/smt.$$.ra)\",\"$(cat /tmp/smt.$$.rb)\"]}" ;;
      siblings) one 6 a > /tmp/smt.$$.ra & one 14 b > /tmp/smt.$$.rb; wait
        echo "{\"rep\":$rep,\"config\":\"siblings\",\"cpus\":[6,14],\"runs\":[\"$(cat /tmp/smt.$$.ra)\",\"$(cat /tmp/smt.$$.rb)\"]}" ;;
    esac
  done
done >> "$OUT"
rm -f /tmp/smt.$$.ra /tmp/smt.$$.rb
