#!/usr/bin/env bash
# One resumable CRN truth job (README §5): every legal action x S shared-dice seeds, greedy blob
# continuation on both sides, CPU, under a memory cap, nice 15.
#   run_truth.sh <subset.json> <S> <out.jsonl> <workers> <threads> <log>
# Launch DETACHED:  setsid nohup run_truth.sh ... > /dev/null 2>&1 < /dev/null &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../../.." && pwd)"
SUBSET="$(realpath -m "$1")"; S="$2"; OUT="$(realpath -m "$3")"; WORKERS="$4"; THREADS="$5"
LOG="$(realpath -m "$6")"
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
CKPT=/home/goodlad/dev/gen3ai/models/rb_x5ab_blob_s1001/final_model.zip
cd "$WT"
export PYTHONPATH="$WT/src"
export CUDA_VISIBLE_DEVICES=   # CPU only: torch cannot create a CUDA context (no GPU lease)
exec nice -n 15 "$WT/scripts/ops/mem_cap.sh" --name crnrefit 12 timeout 12h \
    "$PY" -m main.policy_spectrum.truth run \
    --bank "$HERE/../m5_laneS/bank_v1" --subset "$SUBSET" \
    --continuation "$CKPT=blob_s1001_final" --out "$OUT" --seeds "$S" \
    --threads "$THREADS" --workers "$WORKERS" --chunk 8 --device cpu --lib-profile release \
    >> "$LOG" 2>&1
