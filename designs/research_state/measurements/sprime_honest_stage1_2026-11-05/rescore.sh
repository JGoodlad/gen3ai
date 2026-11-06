#!/bin/bash
# S'-HONEST stage 1: r0s4_rolls's BYTE IDENTITY at the NEW engine stamp. S'-LIVE's registered identity re-score
# (its PREREG §12: the hashed sample's first 600 decisions in the registered order, the live searcher's own
# `_search`, every compared field equal) re-run from this build's checkout, on the stamp this build moved the engine to.
# Usage: rescore.sh <checkout> <expected engine stamp> <out dir>
set -u
WT=$1; STAMP=$2; OUT=$3
STATE=/home/goodlad/gen3ai_archive/sprime_live_play_2026-11-01
cd "$WT" || exit 90
export PYTHONPATH=$WT/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
s=$(nice -n 19 taskset -c 5-7,13-15 "$PY" -c "from main.anchors.server import engine_stamp as e; print(e())") || exit 91
echo "[rescore] engine $s"; [ "$s" = "$STAMP" ] || { echo "[rescore] ENGINE NOT $STAMP"; exit 92; }
mkdir -p "$OUT"
exec nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 "$PY" -m main.value_search.live_identity rescore \
  --cells "$STATE/anchor/u/pstar_s/cells" --panels heldout72_hpfix,forum72,p72 \
  --model "$STATE/ckpts/pstar_s/checkpoint_280000170_steps.zip" \
  --model-sha256 814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed --engine "$STAMP" \
  --max 600 --min 300 --out "$OUT/rescore.json"
