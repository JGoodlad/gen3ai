#!/usr/bin/env bash
# The blob policy's masked probabilities at every bank root row (rows/reads/blob.probs.npz): the
# contested selector and the starved definition read them (README §3). CPU, capped, nice 15.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../../.." && pwd)"
cd "$WT"
export PYTHONPATH="$WT/src"
exec nice -n 15 "$WT/scripts/ops/mem_cap.sh" --name crnread 12 timeout 30m \
    /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m main.policy_spectrum read \
    --bank "$HERE/../m5_laneS/bank_v1" --out "$HERE/rows/reads" \
    --ckpt /home/goodlad/dev/gen3ai/models/rb_x5ab_blob_s1001/final_model.zip=blob --threads 2 --workers 2
