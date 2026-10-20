#!/usr/bin/env bash
# AM-X47b-2 (F-AMX-3): B2's ground truth re-branched under a TOY continuation (T* = rb_toy_ent_s4002_deep @ 290,000,087,
# greedy on both sides), on the SAME bank (collapse pilot bank_x26) and the SAME 306-turn subset (subset_x26.json), 32
# shared dice seeds -- exactly READ A's recipe with the continuation swapped. CPU only, root CLAUDE.md rule 5b: nice 19,
# the 3-core agent pin, one OpenMP thread, no GPU; each chunk takes a HEAVY SLOT (mem_cap.sh) and runs <= 45 min; the
# truth runner skips every turn already in the rows file, so the loop RESUMES after any stop.
#     run_truth.sh <archive dir holding tstar_290000087.zip + model_config.json>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
A="$1"
P="$ROOT/designs/research_state/measurements/collapse_pilot_2026-10-17"
PY="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}"
OUT="$A/rows_toyTstar290M_S32.jsonl"
WANT=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["ids"]))' "$P/subset_x26.json")
export PYTHONPATH="$ROOT/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
for chunk in 1 2 3 4 5 6 7 8; do
    have=$( [ -f "$OUT" ] && "$PY" -c 'import json,sys; print(len({json.loads(l)["id"] for l in open(sys.argv[1]) if l.strip()}))' "$OUT" || echo 0)
    echo "[toy-b2-truth] $(date -Is) chunk $chunk: $have / $WANT turns in $OUT"
    [ "$have" -ge "$WANT" ] && { echo "[toy-b2-truth] DONE"; exit 0; }
    rc=0
    "$ROOT/scripts/ops/mem_cap.sh" --name toyb2truth 8 timeout 45m nice -n 19 taskset -c 5-7,13-15 "$PY" \
        -m main.policy_spectrum.truth run --bank "$P/bank_x26" --subset "$P/subset_x26.json" \
        --continuation "$A/tstar_290000087.zip=toyTstar@290M" --out "$OUT" \
        --seeds 32 --threads 1 --workers 3 --chunk 6 || rc=$?
    echo "[toy-b2-truth] $(date -Is) chunk $chunk exit $rc"
    case "$rc" in 0|124|143) ;; *) echo "[toy-b2-truth] HOLD: exit $rc"; exit "$rc" ;; esac
done
echo "[toy-b2-truth] HOLD: 8 chunks and not done"; exit 3
