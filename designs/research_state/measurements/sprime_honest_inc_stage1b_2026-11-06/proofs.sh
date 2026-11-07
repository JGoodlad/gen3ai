#!/bin/bash
# S'-HONEST stage 1b, the IDENTITY PROOFS at this build's engine stamp, in bounded chunks (rule 5b: <= 45 min each
# under mem_cap; both tools resume from their rows files):
#   tooth   - the replication designer's stage-1 tooth: r0s4_honest@10 at its DEFAULT budget on 24 of stage 1's B3
#             states, every non-timing field equal to stage 1's (`sprime_honest_play_2026-11-06/inputs/stage1_tooth.py`)
#   rescore - S'-LIVE's registered identity re-score of r0s4_rolls@10, its 600 decisions (`live_identity rescore`, the
#             stage-1 `rescore.sh` arguments)
# Usage: proofs.sh <checkout> <tooth|rescore> <expected engine stamp> <out dir>
set -u
WT=$1; WHAT=$2; STAMP=$3; OUT=$4
STATE=/home/goodlad/gen3ai_archive/sprime_live_play_2026-11-01
MODEL=$STATE/ckpts/pstar_s/checkpoint_280000170_steps.zip
SHA=814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed
TOOTH=/home/goodlad/dev/gen3ai-worktrees/sprime_honest_prereg/designs/research_state/measurements/sprime_honest_play_2026-11-06/inputs/stage1_tooth.py
STAGE1=/home/goodlad/dev/gen3ai/designs/research_state/measurements/sprime_honest_stage1_2026-11-05
cd "$WT" || exit 90
export PYTHONPATH=$WT/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
s=$(nice -n 19 taskset -c 5-7,13-15 "$PY" -c "from main.anchors.server import engine_stamp as e; print(e())") || exit 91
echo "[proofs] engine $s"; [ "$s" = "$STAMP" ] || { echo "[proofs] ENGINE NOT $STAMP"; exit 92; }
mkdir -p "$OUT"
for chunk in $(seq 1 12); do
  echo "[proofs] $WHAT chunk $chunk $(date -Is)" >> "$OUT/$WHAT.log"
  if [ "$WHAT" = tooth ]; then
    nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 timeout 2700 "$PY" "$TOOTH" \
      --rows "$STAGE1/curve_b3_rows.jsonl.gz" --selected /home/goodlad/gen3ai_archive/dlive_read_2026-10-28/live/selected.jsonl \
      --model "$MODEL" --model-sha256 "$SHA" --out "$OUT/tooth" --n 24 >> "$OUT/$WHAT.log" 2>&1
  else
    nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 timeout 2700 "$PY" -m main.value_search.live_identity rescore \
      --cells "$STATE/anchor/u/pstar_s/cells" --panels heldout72_hpfix,forum72,p72 --model "$MODEL" \
      --model-sha256 "$SHA" --engine "$STAMP" --max 600 --min 300 --out "$OUT/rescore.json" >> "$OUT/$WHAT.log" 2>&1
  fi
  rc=$?
  echo "[proofs] $WHAT chunk $chunk exit $rc $(date -Is)" >> "$OUT/$WHAT.log"
  [ $rc -eq 0 ] && exit 0
  [ $rc -ne 124 ] && [ $rc -ne 88 ] && exit $rc
done
exit 124
