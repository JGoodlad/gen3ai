#!/usr/bin/env bash
# JOB 2 — one SOP tier-B read (metamon:SmallRL, greedy, BOTH team sets, 100 games each)
# taken TWICE in the same window: once through the in-repo websocket FRONT END, once through
# the Node path (main.anchors starting its own deps/pokemon-showdown server). Same team seed.
set -u
cd /home/goodlad/dev/gen3ai
export PYTHONPATH="${PYTHONPATH:-}:/home/goodlad/dev/gen3ai/src"
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
TMP=/home/goodlad/.claude/jobs/9ab51de6/tmp/fp_axes
OUT=$TMP/frontend
MODEL=models/ai_v12_02_winprob_critic/final_model.zip
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
SEED=916002
mkdir -p $OUT

echo "=== JOB2 START $(date -Is) load: $(cut -d' ' -f1-3 /proc/loadavg) head: $(git rev-parse --short HEAD)"

# ---- the front end, on 9751 -------------------------------------------------------------
env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" nice -n 10 \
  $PY -m utils.bridge.ws_frontend --port 9751 --impl rust > $OUT/ws_frontend.log 2>&1 &
FE_PID=$!
echo "ws_frontend pid $FE_PID"
for i in $(seq 1 60); do
  grep -q "\[ws_frontend\] READY" $OUT/ws_frontend.log 2>/dev/null && break
  sleep 1
done
grep "READY" $OUT/ws_frontend.log || { echo "FRONT END NEVER READY"; kill $FE_PID; exit 2; }

run () {  # label teamset extra_args...
  local label=$1 teamset=$2; shift 2
  local t0=$(date +%s)
  echo "--- $label ($teamset) START $(date -Is) load: $(cut -d' ' -f1-3 /proc/loadavg)" \
      | tee $OUT/$label.meta
  rm -rf $OUT/$label
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES="" POKESIM_SIM_BRIDGE_BIN=$POKESIM_SIM_BRIDGE_BIN \
    nice -n 10 $PY -m main.anchors \
      --model $MODEL --opponent metamon:SmallRL --regime greedy --teamset $teamset \
      --games 100 --team-seed $SEED --device cpu \
      --username AxFe${label:0:6} --peer-username MmFe${label:0:6} \
      "$@" > $OUT/$label.log 2>&1
  local rc=$? t1=$(date +%s)
  echo "--- $label END $(date -Is) rc=$rc wall_s=$((t1-t0)) load: $(cut -d' ' -f1-3 /proc/loadavg)" \
      | tee -a $OUT/$label.meta
  grep -E '^  status|^  WIN RATE' $OUT/$label.log | tee -a $OUT/$label.meta
}

run fehome home --server-uri ws://localhost:9751/showdown/websocket --out $OUT/fehome
run feaway away --server-uri ws://localhost:9751/showdown/websocket --out $OUT/feaway
run ndhome home --port 9752 --out $OUT/ndhome
run ndaway away --port 9753 --out $OUT/ndaway

kill $FE_PID 2>/dev/null && echo "stopped ws_frontend $FE_PID"
echo "=== JOB2 DONE $(date -Is)"
