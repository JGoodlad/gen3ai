#!/usr/bin/env bash
# The SOP's tier-B AWAY cell (`designs/ops/EXTERNAL_ANCHORS_SOP.md` sec 1: report BOTH team sets,
# always — they disagreed in SIGN for SmallRL), taken through the TOOL OF RECORD `main.anchors`
# on BOTH arms so the pair row exists on the away set too. greedy-vs-greedy, verified per decision.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/anchors/away
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai
for SPEC in "armW ai_v13_02_flywheel_winprob" "armS ai_v13_01_flywheel_shaped"; do
  set -- $SPEC; TAG=$1; RUN=$2
  [ -f "$OUT/${TAG}_smallrl_greedy_away/summary.json" ] && { echo "SKIP $TAG"; continue; }
  echo "=== $TAG greedy away $(date -Is) ==="
  nice -n 12 $P -m main.anchors --model "models/$RUN/final_model.zip" \
    --opponent metamon:SmallRL --regime greedy --teamset away --games 100 \
    --device cpu --nice 12 --out "$OUT/${TAG}_smallrl_greedy_away" 2>&1
  echo "=== EXIT $? $TAG $(date -Is) ==="
done
echo "=== AWAY CELLS DONE $(date -Is) ==="
