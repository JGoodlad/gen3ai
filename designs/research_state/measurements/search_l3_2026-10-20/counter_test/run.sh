#!/usr/bin/env bash
# L3's familiarity counter-test (PREREGISTRATION.md §7): the driver over the 287 registered states in chunks, each in
# ONE heavy slot for at most 45 min (rule 5b), CPU only; then the determinism recheck (G3). Resumable: rerun to continue.
#   nohup nice -n 19 taskset -c 5-7,13-15 counter_test/run.sh <8fe4fb89 checkout> > <log> 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="${1:?the checkout of the L3 code commit 8fe4fb89}"
PY="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}"
CAP=/home/goodlad/dev/gen3ai/scripts/ops/mem_cap.sh
ARCH="${L3D_ARCHIVE:-$HOME/gen3ai_archive/search_l3_decoy}"
export CUDA_VISIBLE_DEVICES=""
cd "$CODE"
n_done() { [ -f "$ARCH/rows.jsonl" ] && "$PY" -c "import json,sys; print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip()}))" "$ARCH/rows.jsonl" || echo 0; }
for chunk in 1 2 3 4 5 6 7 8; do
  [ "$(n_done)" -ge 287 ] && break
  echo "[l3d run] chunk $chunk: $(n_done) done"
  "$CAP" --slot --name l3decoy 12 timeout 45m "$PY" -I "$HERE/decoy_run.py" --code "$CODE" --budget-min 38
done
[ "$(n_done)" -ge 287 ] || { echo "[l3d run] INCOMPLETE after 8 chunks"; exit 3; }
"$CAP" --slot --name l3decoy_recheck 12 timeout 45m "$PY" -I "$HERE/decoy_run.py" --code "$CODE" --recheck 10
echo "[l3d run] RUN_DONE"
