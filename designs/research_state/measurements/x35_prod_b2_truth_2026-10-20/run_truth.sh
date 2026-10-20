#!/usr/bin/env bash
# AM-X47c-2 (F-5): production's B2 ground truth re-branched under P* (rb_x26_s2601 @ 280,000,170, greedy on both
# sides), section 5.6.5's own refresh rule (P* >= the old truth's 150M + 50M). The SAME bank (collapse pilot bank_x26),
# the SAME 306-turn subset (subset_x26.json), 32 shared dice seeds -- exactly READ A's recipe and the toy refresh's
# procedure with the continuation swapped. CPU only, root CLAUDE.md rule 5b: nice 19, the 3-core agent pin, one OpenMP
# thread, no GPU; each chunk takes a HEAVY SLOT (mem_cap.sh) and runs <= 45 min; the truth runner skips every turn
# already in the rows file, so the loop RESUMES after any stop.
#     run_truth.sh <archive dir holding pstar_280000170.zip + model_config.json>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
A="$1"
P="$ROOT/designs/research_state/measurements/collapse_pilot_2026-10-17"
PY="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}"
OUT="$A/rows_X26at280M_S32.jsonl"
WANT=$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["ids"]))' "$P/subset_x26.json")
export PYTHONPATH="$ROOT/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
for chunk in 1 2 3 4 5 6 7 8; do
    have=$( [ -f "$OUT" ] && "$PY" -c 'import json,sys; print(len({json.loads(l)["id"] for l in open(sys.argv[1]) if l.strip()}))' "$OUT" || echo 0)
    echo "[prod-b2-truth] $(date -Is) chunk $chunk: $have / $WANT turns in $OUT"
    [ "$have" -ge "$WANT" ] && { echo "[prod-b2-truth] DONE"; exit 0; }
    rc=0
    "$ROOT/scripts/ops/mem_cap.sh" --name prodb2truth 8 timeout 45m nice -n 19 taskset -c 5-7,13-15 "$PY" \
        -m main.policy_spectrum.truth run --bank "$P/bank_x26" --subset "$P/subset_x26.json" \
        --continuation "$A/pstar_280000170.zip=X26@280M" --out "$OUT" \
        --seeds 32 --threads 1 --workers 3 --chunk 6 || rc=$?
    echo "[prod-b2-truth] $(date -Is) chunk $chunk exit $rc"
    case "$rc" in 0|124|143) ;; *) echo "[prod-b2-truth] HOLD: exit $rc"; exit "$rc" ;; esac
done
echo "[prod-b2-truth] HOLD: 8 chunks and not done"; exit 3
