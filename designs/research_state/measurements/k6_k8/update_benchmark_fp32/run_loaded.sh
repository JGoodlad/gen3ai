#!/usr/bin/env bash
# L1: ONE unit of the benchmark with 6 CPU-bound bystander processes (nice 0) running for its whole duration —
# what a run inside the slow tier beside other tests / a routine gate (`-n 6`) would see. Reads how much a
# loaded CPU moves the regions' update.
L=/home/goodlad/gen3ai_archive/lane_c_perf_guard/bank
BUF=/home/goodlad/gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl
CKPT=/home/goodlad/dev/gen3ai/models/ai_v14_02_lbat_ctrl/final_model.zip
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
B=/home/goodlad/dev/gen3ai-wt/lane-c-perf-guard
name=L1
cd "$B" || exit 2
for _ in $(seq 1 60); do awk '{exit !($1 < 0.5)}' /proc/loadavg && break; sleep 10; done
hogs=()
for _ in 1 2 3 4 5 6; do "$PY" -c "while True: pass" & hogs+=($!); done
{ echo "== $name tree=$B head=$(git rev-parse --short HEAD) dirty=$(git status --short | wc -l) bystanders=${hogs[*]}"
  echo "before: load $(cut -d' ' -f1-3 /proc/loadavg); gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"; } > $L/$name.env
$B/scripts/ops/gpu_lock.sh timeout 1260 $B/scripts/ops/mem_cap.sh --name bank$name 48 \
  env PYTHONPATH=$B/src $PY -m main.compile_inventory run --stage time --device cuda --buffer $BUF \
  --model $CKPT --keep-prewarm --unbracketed --worker-timeout-min 25 \
  --out-root $L/$name > $L/$name.log 2>&1
rc=$?
kill "${hogs[@]}" 2>/dev/null
echo "after: load $(cut -d' ' -f1-3 /proc/loadavg); gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" >> $L/$name.env
echo "$name exit=$rc at=$(date -Is)" >> $L/status
