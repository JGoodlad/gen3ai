#!/usr/bin/env bash
# The four Foul Play cells, ONE AT A TIME, in the registered order.
set -u
cd /home/goodlad/dev/gen3ai
export PYTHONPATH="${PYTHONPATH:-}:/home/goodlad/dev/gen3ai/src"
TMP=/home/goodlad/.claude/jobs/9ab51de6/tmp/fp_axes
MODEL=models/ai_v12_02_winprob_critic/final_model.zip
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3

run_cell () {  # name cfg ms port ourname peername
  local name=$1 cfg=$2 ms=$3 port=$4 un=$5 pn=$6
  echo "=== CELL $name START $(date -Is) load: $(cut -d' ' -f1-3 /proc/loadavg) head: $(git rev-parse --short HEAD)" \
      | tee $TMP/cells/$name.meta
  rm -rf $TMP/cells/$name
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES="" GEN3AI_ANCHORS_CONFIG=$cfg \
    nice -n 10 $PY -m main.anchors \
      --model $MODEL --opponent foulplay --regime greedy --teamset home \
      --games 80 --search-time-ms $ms --search-parallelism 1 --team-seed 916001 \
      --port $port --username $un --peer-username $pn \
      --progress-timeout 1800 --first-game-timeout 1800 \
      --out $TMP/cells/$name > $TMP/cells/$name.log 2>&1
  echo "=== CELL $name END $(date -Is) load: $(cut -d' ' -f1-3 /proc/loadavg) rc=$?" | tee -a $TMP/cells/$name.meta
  grep -E '^  status|^  WIN RATE|^  search ' $TMP/cells/$name.log | tee -a $TMP/cells/$name.meta
}

mkdir -p $TMP/cells
run_cell w100  $TMP/cfg/anchors_std.json     100  9741 AxW100  FoulPlayW1
run_cell w300  $TMP/cfg/anchors_std.json     300  9742 AxW300  FoulPlayW3
run_cell w1000 $TMP/cfg/anchors_std.json    1000  9743 AxW1k   FoulPlayWK
run_cell k1000 $TMP/cfg/anchors_noknow.json 1000  9744 AxK1k   FoulPlayKK
echo "ALL CELLS DONE $(date -Is)"
