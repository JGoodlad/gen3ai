#!/usr/bin/env bash
# THE BANK (PREREGISTRATION §3-§4): the three parts of the cycle, each in ONE heavy slot for at most 45 min (rule 5b),
# CPU only, then the draw. Resumable: a part whose directory holds a manifest is skipped (an INCOMPLETE part is refused by
# make_bank.py; remove its directory and rerun, same seed). `--extension` plays cyc2 and redraws (§4's one extension).
#   nohup nice -n 19 taskset -c 5-7,13-15 ./run_bank.sh <8fe4fb89 checkout> [--extension] > <log> 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="${1:?the checkout of the L3 code commit 8fe4fb89}"
EXT="${2:-}"
PY="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}"
CAP=/home/goodlad/dev/gen3ai/scripts/ops/mem_cap.sh
ARCH="${OFFPOOL_ARCHIVE:-$HOME/gen3ai_archive/search_offpool_bank}"
export CUDA_VISIBLE_DEVICES=""
cd "$CODE"
CYCLES="cyc1"; [ "$EXT" = "--extension" ] && CYCLES="cyc1 cyc2"
for cyc in $CYCLES; do
  for part in bots sentA sentB; do
    [ -f "$ARCH/traces/${cyc}_${part}/eval_traces/step_280000170/eval_manifest.json" ] && continue
    echo "[offpool bank] $cyc $part"
    "$CAP" --slot --name "offpool_${part}" 12 timeout 45m "$PY" -I "$HERE/gen_cycle.py" --code "$CODE" --cycle "$cyc" \
      --part "$part" --timeout-min 42 || { echo "[offpool bank] $cyc $part FAILED rc=$?"; exit 3; }
  done
done
"$CAP" --slot --name offpool_draw 12 timeout 45m "$PY" -I "$HERE/make_bank.py" --code "$CODE" $EXT
echo "[offpool bank] BANK_DONE rc=$?"
