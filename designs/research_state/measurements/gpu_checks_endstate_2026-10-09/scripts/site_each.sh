#!/usr/bin/env bash
# One FRESH process per SDPA site restriction. Run from <checkout>/src under scripts/ops/gpu_lock.sh with the lease.
#   site_each.sh <dump.json> <site> [<site> ...]   (the first site's run also dumps the SDPA mask statistics)
dump=$1; shift
py=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
here=$(dirname "$0")
first=1
for s in "$@"; do
    extra=()
    [ "$first" = 1 ] && extra=(--dump-inputs "$dump")
    first=0
    timeout 900 "$py" "$here/site_probe.py" --site "$s" "${extra[@]}" 2>&1 | grep -E "^site=|^SDPA call|Error" | cut -c1-600
done
