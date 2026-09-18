#!/usr/bin/env bash
# The anchors SOP's tier-B AWAY cell (sec 1: report BOTH team sets, always — they disagreed in
# SIGN for SmallRL), taken through the TOOL OF RECORD `python -m main.anchors` on W_b, exactly
# as the pair read took it on arm W and arm S. greedy-vs-greedy, verified per decision.
# 🚨 --out is under the job tmp, NEVER the cwd: a stray anchors_out in the main checkout dirties
# main, which happened on 2026-09-16.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/anchors/away
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai
for SPEC in "armWb ai_v13_04_flywheel_winprob_b"; do
  set -- $SPEC; TAG=$1; RUN=$2
  [ -f "$OUT/${TAG}_smallrl_greedy_away/summary.json" ] && { echo "SKIP $TAG"; continue; }
  echo "=== $TAG greedy away $(date -Is) ==="
  nice -n 12 $P -m main.anchors --model "models/$RUN/final_model.zip" \
    --opponent metamon:SmallRL --regime greedy --teamset away --games 100 \
    --device cpu --nice 12 --out "$OUT/${TAG}_smallrl_greedy_away" 2>&1
  echo "=== EXIT $? $TAG $(date -Is) ==="
done
echo "=== AWAY CELL DONE $(date -Is) ==="
