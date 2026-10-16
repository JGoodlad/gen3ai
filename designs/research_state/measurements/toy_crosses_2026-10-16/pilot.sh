#!/usr/bin/env bash
# CPU-cost PILOT (not a registered read): one R-T6-shaped cell (toy 100M vs EB s2001 15M), N pairs, into a SCRATCH ledger
# root (/tmp/toyx_pilot_ledger), under rule 5b + mem_cap. Usage: pilot.sh <pairs> <rust_threads> <n_envs|""> <tag>
HERE=$(cd "$(dirname "$0")" && pwd); TOP=$(git -C "$HERE" rev-parse --show-toplevel)
M=/home/goodlad/dev/gen3ai/models
NE=${3:+--n-envs $3}
/usr/bin/time -v "$TOP/scripts/ops/mem_cap.sh" 16 "$HERE/py.sh" -m main.h2h play-many \
  --players "$M/rb_toy_ent_s4002/final_model.zip" --opponents "$M/rb_bf_eb_s2001/final_model.zip" \
  --pairs "$1" --batch-pairs "$1" --device cpu --torch-threads 1 --threads "$2" $NE \
  --out /tmp/toyx_pilot_ledger_$4 --label toyx_pilot --oracle-reveal-mode off > "/tmp/toyx_pilot_$4.log" 2>&1
echo "rc=$?"; tail -30 "/tmp/toyx_pilot_$4.log"
