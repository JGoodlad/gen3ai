#!/usr/bin/env bash
# Run measure.py cells in sequence. Usage (repo root):
#   designs/research_state/measurements/k9_bf16_2026-10-10/run_cells.sh <out_dir> <device> <cell>...
# where a cell is <state>:<behaviour> (state fresh|perturb05|perturb10|trained; behaviour bf16|fp32);
# "trained" reads $K9BF16_TRAINED (the train_state.py output). A cell already written is skipped (resumable).
# CPU: each cell under scripts/ops/mem_cap.sh 24, CUDA_VISIBLE_DEVICES empty. CUDA (the GPU window, lease
# held): set GEN3AI_GPU_LEASE_TOKEN and run this whole script under scripts/ops/gpu_lock.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$1"; dev="$2"; shift 2
py="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}"
mkdir -p "$out"
for cell in "$@"; do
  state="${cell%%:*}"; beh="${cell##*:}"
  name="${state}_${beh}"
  if [ -f "$out/$name.json" ]; then echo "[run_cells] $name exists, skipped"; continue; fi
  extra=()
  if [ "$state" = "trained" ]; then extra=(--state weights --weights-from "${K9BF16_TRAINED:?set K9BF16_TRAINED}")
  elif [ "$state" = "prod9m" ]; then
    # a REAL long-trained checkpoint (read only) at ANOTHER arch: reported on its own, never pooled
    extra=(--state weights --weights-from "${K9BF16_PROD9M:?set K9BF16_PROD9M}" --arch production)
  else extra=(--state "$state"); fi
  if [ "$beh" = "bf16" ]; then extra+=(--compiled-rounds); fi
  if [ "$dev" = "cpu" ]; then
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/k9bf16_inductor}" \
      OMP_NUM_THREADS=4 timeout 2400 nice -n 19 scripts/ops/mem_cap.sh 24 "$py" "$here/measure.py" "${extra[@]}" \
      --behaviour "$beh" --device cpu --out "$out/$name" > "$out/$name.log" 2>&1
  else
    PYTHONPATH=src timeout 900 "$py" "$here/measure.py" "${extra[@]}" --behaviour "$beh" --device "$dev" \
      --t2-backend "${K9BF16_T2_BACKEND:-eager}" --out "$out/$name" > "$out/$name.log" 2>&1
  fi
  echo "[run_cells] $name done: $(tail -2 "$out/$name.log" | head -1)"
done
