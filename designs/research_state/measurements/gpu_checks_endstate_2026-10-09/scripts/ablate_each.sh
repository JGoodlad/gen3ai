#!/usr/bin/env bash
# One FRESH process per ablation variant (no cross-variant compile / autotune state). Run from <checkout>/src under
# scripts/ops/gpu_lock.sh with the lease token.   ablate_each.sh <out.jsonl> <variant> [<variant> ...]
out=$1; shift
py=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
here=$(dirname "$0")
for v in "$@"; do
    timeout 900 "$py" "$here/ablate.py" --out "$out" --variants "$v" 2>&1 | grep -E "^(E|R|P)[-+ ]|Error" | cut -c1-400
done
