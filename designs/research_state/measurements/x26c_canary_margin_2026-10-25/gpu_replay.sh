#!/usr/bin/env bash
# gpu_replay.sh — the CUDA half of the x26c u1000 clip-margin read (WRITTEN, NOT RUN; 2026-10-25). The CPU read
# (margin_read.py) cannot run the run's Triton / cuBLAS kernels; this re-reads the same checkpoints on the card: eager
# E32 / Ebf per row + the 16-seed dither null on CUDA, and the PRODUCTION compiled arm (`compile_regions.install`) per
# fixture row at the canary's B = 2048 (Cbf vs E32, Ebf vs E32, Cbf vs Ebf). It answers: does CUDA compiled bf16 move row
# 50 by ~0.7 at 478.0M as the canary read at 478.31M, and is that inside CUDA's own fp32 dither null for the row?
#
#   GEN3AI_GPU_LEASE_TOKEN=<token> gpu_replay.sh <pin checkout at f4b25238> <out dir>
#
# Under the caller's GPU LEASE only (root rule 5): gpu_lock.sh refuses everyone else (exit 5 / 6), mem_cap.sh caps
# memory, a hard wall timeout bounds it. Writes nothing under models/. Est. 10-20 min (two compiles at B = 2048).
set -u
PIN="${1:?pin checkout (git worktree at f4b25238)}"; OUT="${2:?out dir}"
if [ -z "${GEN3AI_GPU_LEASE_TOKEN:-}" ] && [ -z "${GEN3AI_GPU_LEASE_TOKEN_FILE:-}" ]; then
    echo "REFUSING: no GEN3AI_GPU_LEASE_TOKEN / _FILE — the GPU is LEASED; run this as the lease holder" >&2; exit 2
fi
[ "$(git -C "$PIN" rev-parse HEAD 2>/dev/null)" = "f4b25238def57900c0305bb2a3a5a8f9edaef285" ] \
    || { echo "REFUSING: $PIN is not a checkout of the run's pin f4b25238" >&2; exit 2; }
ROOT=/home/goodlad/dev/gen3ai
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C=$ROOT/models/rb_x26c_s2601/checkpoints
mkdir -p "$OUT"
cd "$PIN" && export PYTHONPATH="$PIN/src" GEN3AI_TEST_ALLOW_GPU=1
"$ROOT/scripts/ops/gpu_lock.sh" "$ROOT/scripts/ops/mem_cap.sh" --name x26c-margin-gpu 24 timeout 1800 \
    "$PY" "$HERE/margin_read.py" --device cuda --compiled --dither 16 --out "$OUT" \
    --ckpt 478M=$C/checkpoint_478000208_steps.zip --ckpt 479M=$C/checkpoint_479000039_steps.zip \
    > "$OUT/gpu_replay.log" 2>&1
RC=$?
echo "[gpu_replay] exit $RC -> $OUT (log $OUT/gpu_replay.log)"
exit "$RC"
