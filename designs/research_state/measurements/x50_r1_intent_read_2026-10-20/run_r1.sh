#!/usr/bin/env bash
# X50 R1 driver: extract (humans: the held-out dates; traces: X26's cycles >= 360M), then the registered read.
# Each step holds ONE heavy slot, bounded at 45 min, nice 19 on the agent cores, CPU only. Resumable: the
# extractor skips finished parts; the read re-runs whole (forward passes only).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
MAIN=$(git -C "$REPO" worktree list | head -1 | awk '{print $1}')
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
OUT=${X50R1_OUT:-/home/goodlad/gen3ai_archive/x50_r1_intent}
export PYTHONPATH="$REPO/src" CUDA_VISIBLE_DEVICES=""
step() { "$REPO/scripts/ops/mem_cap.sh" --name "x50r1_$1" --slot 8 timeout 2700 nice -n 19 taskset -c 5-7,13-15 "${@:2}"; }
step humans "$PY" -m main.x50_r1.run humans --dates 2026-09-21,2026-09-22,2026-09-23 \
    --corpus "$MAIN/replays/showdown/gen3ou" --out "$OUT/humans" --workers 3 || exit 1
step traces "$PY" -m main.x50_r1.run traces --run "$MAIN/models/rb_x26_s2601" --min-step 360000000 \
    --out "$OUT/traces" --workers 3 || exit 1
step read "$PY" -m main.x50_r1.read --humans "$OUT/humans" --traces "$OUT/traces" \
    --checkpoint "$MAIN/models/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip" --out "$HERE" --threads 3 || exit 1
echo "[x50r1] $(date -Is) done"
