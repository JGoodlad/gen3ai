#!/usr/bin/env bash
# ROW 3 -- the anchors SOP's tier-B AWAY cell through the TOOL OF RECORD `python -m main.anchors`,
# on the CONTINUATION CONTROL's endpoint (+12.09M). greedy-vs-greedy, verified per decision,
# 100 games as two role-balanced half-cells. This is the convergence read's own run_away_cell.sh
# with the ref swapped and the port moved inside this job's 9400-9499 band.
#
# 🚨 --out is under the job tmp, NEVER the cwd: a stray anchors_out in the main checkout dirties
# main, which happened on 2026-09-16.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/9ab51de6/tmp/wcont_read/anchors/away
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai
TAG=wcont_p12M
[ -f "$OUT/${TAG}_smallrl_greedy_away/summary.json" ] && { echo "SKIP $TAG"; exit 0; }
echo "=== $TAG greedy away $(date -Is) ==="
nice -n 15 $P -m main.anchors --model models/ai_v13_09_wcont/final_model.zip \
  --opponent metamon:SmallRL --regime greedy --teamset away --games 100 \
  --device cpu --nice 15 --port 9450 --out "$OUT/${TAG}_smallrl_greedy_away" 2>&1
echo "=== EXIT $? $TAG $(date -Is) ==="
