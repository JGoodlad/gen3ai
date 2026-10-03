#!/usr/bin/env bash
# fp32 'highest' UPDATE BENCHMARK read (deletion pass lane C, 2026-10-02) — the compile inventory's time stage
# (the retired compiled_perf_guard_test's command) on arm C's final checkpoint and the learner benchmark's
# pinned buffer, the production compile sentinel kept, one un-bracketed update; one unit per gpu_lock hold. B = the tree as committed; P = a throwaway worktree with
# ONE planted regression (compile_regions.install's R1 dispatcher runs the EAGER micro-step once locked,
# uncounted). Interleaved B P B P B B B on an idle box. Resumable: a unit with a status row is skipped.
# Records, per unit: the tree head, the load average and the GPU's compute apps before and after.
L=/home/goodlad/gen3ai_archive/lane_c_perf_guard/bank
BUF=/home/goodlad/gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl
CKPT=/home/goodlad/dev/gen3ai/models/ai_v14_02_lbat_ctrl/final_model.zip
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
B=/home/goodlad/dev/gen3ai-wt/lane-c-perf-guard
P=/home/goodlad/dev/gen3ai-wt/lane-c-plant
mkdir -p "$L"
for unit in B1:$B P1:$P B2:$B P2:$P B3:$B B4:$B B5:$B; do
  name=${unit%%:*}; wt=${unit#*:}
  grep -q "^$name " $L/status 2>/dev/null && continue
  cd "$wt" || exit 2
  # a quiet box: load1 < 0.5 (waits up to 10 min), nothing on the GPU
  for _ in $(seq 1 60); do
    awk '{exit !($1 < 0.5)}' /proc/loadavg && break; sleep 10
  done
  { echo "== $name tree=$wt head=$(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)"
    echo "before: load $(cut -d' ' -f1-3 /proc/loadavg); gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"; } > $L/$name.env
  $wt/scripts/ops/gpu_lock.sh timeout 1260 $wt/scripts/ops/mem_cap.sh --name bank$name 48 \
    env PYTHONPATH=$wt/src $PY -m main.compile_inventory run --stage time --device cuda --buffer $BUF \
    --model $CKPT --keep-prewarm --unbracketed --worker-timeout-min 25 \
    --out-root $L/$name > $L/$name.log 2>&1
  rc=$?
  echo "after: load $(cut -d' ' -f1-3 /proc/loadavg); gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" >> $L/$name.env
  echo "$name exit=$rc at=$(date -Is)" >> $L/status
done
