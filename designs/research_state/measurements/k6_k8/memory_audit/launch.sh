#!/usr/bin/env bash
# The GPU memory audit's launch: sizing arm A's argv (rust core, N = 48, production, fp32), 3 updates,
# with the CUDA ledger; arg 1 = tag, the rest = extra flags (e.g. the X26 ride-along heads).
TAG=$1; shift
cd /home/goodlad/dev/gen3ai-wt/memaudit || exit 2
/home/goodlad/dev/gen3ai-wt/memaudit/scripts/ops/gpu_lock.sh timeout 1200 /home/goodlad/dev/gen3ai-wt/memaudit/scripts/ops/mem_cap.sh --name mem_$TAG 48 env PYTHONPATH=/home/goodlad/dev/gen3ai-wt/memaudit/src \
  /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m main.launcher --no-pin --restart-interval-hours 0 \
  --max-crash-restarts 1 --env-core rust --n-envs 48 --steps 294912 --seed 1001 --device cuda \
  --log-level periodic --arch production "$@" --run-name memaudit_${TAG}_$(date +%s) > /home/goodlad/gen3ai_archive/k6_k8/memaudit/launch_$TAG.log 2>&1
echo "$TAG exit=$? at=$(date -Is)" >> /home/goodlad/gen3ai_archive/k6_k8/memaudit/status
