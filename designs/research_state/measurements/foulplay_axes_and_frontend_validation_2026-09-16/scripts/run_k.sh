#!/usr/bin/env bash
set -u
cd /home/goodlad/dev/gen3ai
export PYTHONPATH="${PYTHONPATH:-}:/home/goodlad/dev/gen3ai/src"
TMP=/home/goodlad/.claude/jobs/9ab51de6/tmp/fp_axes
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
echo "=== CELL k1000 START $(date -Is) load: $(cut -d' ' -f1-3 /proc/loadavg) head: $(git rev-parse --short HEAD)" | tee $TMP/cells/k1000.meta
rm -rf $TMP/cells/k1000
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="" GEN3AI_ANCHORS_CONFIG=$TMP/cfg/anchors_noknow.json \
  nice -n 10 $PY -m main.anchors \
    --model models/ai_v12_02_winprob_critic/final_model.zip --opponent foulplay \
    --regime greedy --teamset home --games 80 --search-time-ms 1000 --search-parallelism 1 \
    --team-seed 916001 --port 9744 --username AxK1k --peer-username FoulPlayKK \
    --progress-timeout 1800 --first-game-timeout 1800 \
    --out $TMP/cells/k1000 > $TMP/cells/k1000.log 2>&1
echo "=== CELL k1000 END $(date -Is) load: $(cut -d' ' -f1-3 /proc/loadavg) rc=$?" | tee -a $TMP/cells/k1000.meta
grep -E '^  status|^  WIN RATE|^  search ' $TMP/cells/k1000.log | tee -a $TMP/cells/k1000.meta
echo "K CELL DONE"
