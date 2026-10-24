#!/usr/bin/env bash
# X50 rung 1b R0 driver (design_x50_rung1b_human_opponent_model.md §8.1): extract R1's populations again WITH the
# completed own side (--complete: modes 2 C-GAME and 3 own-true beside R1's 0 spectator and 1 full), then the read.
# Extraction: the SHORT LANE (<= 600 s; resumable, a finished part is skipped: re-run on exit 88). The read: one
# heavy slot, <= 45 min, resumable through its per-part forward cache. nice 19 on the agent cores, CPU only.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
MAIN=$(git -C "$REPO" worktree list | head -1 | awk '{print $1}')
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
OUT=${X50R1B_OUT:-/home/goodlad/gen3ai_archive/x50_r1b}
export PYTHONPATH="$REPO/src" CUDA_VISIBLE_DEVICES=""
short() { "$REPO/scripts/ops/mem_cap.sh" --name "x50r1b_$1" --slot --max-s 600 8 nice -n 19 taskset -c 5-7,13-15 "${@:2}"; }
long() { "$REPO/scripts/ops/mem_cap.sh" --name "x50r1b_$1" --slot 12 timeout 2700 nice -n 19 taskset -c 5-7,13-15 "${@:2}"; }
step="${1:-all}"
if [ "$step" = all ] || [ "$step" = humans ]; then
  short humans "$PY" -m main.x50_r1.run humans --complete --dates 2026-09-21,2026-09-22,2026-09-23 \
      --corpus "$MAIN/replays/showdown/gen3ou" --out "$OUT/humans" --workers 3 || exit $?
fi
if [ "$step" = all ] || [ "$step" = traces ]; then
  short traces "$PY" -m main.x50_r1.run traces --complete --run "$MAIN/models/rb_x26_s2601" --min-step 360000000 \
      --out "$OUT/traces" --workers 3 || exit $?
fi
if [ "$step" = all ] || [ "$step" = read ]; then
  long read "$PY" -m main.x50_r1.read_r0 --humans "$OUT/humans" --traces "$OUT/traces" \
      --checkpoint "$MAIN/models/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip" --out "$HERE" --threads 6 || exit $?
fi
echo "[x50r1b] $(date -Is) $step done"
