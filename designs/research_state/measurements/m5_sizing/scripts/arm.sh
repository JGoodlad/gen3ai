#!/usr/bin/env bash
# REGISTRATION §5.1: ONE learning arm through the launcher, from the MAIN checkout, pinned, in ONE GPU hold
# (<= 120 min inside the lock; orchestrator 2026-10-01), under a memory cap. RESUMABLE: when the run dir
# already holds a checkpoint, the same call RESUMES from the newest one (--model, no FRESH-only --arch).
#   arm.sh <run-name> <N> <epochs> <seed> <t2-buckets|-> [pin]
set -u
NAME=$1; N=$2; EP=$3; SEED=$4; BK=$5; PIN=${6:-277f318fad93a587e5ad1a3f476fab917c563ed0}
MAIN=/home/goodlad/dev/gen3ai
WT=/home/goodlad/dev/gen3ai-wt/m5-sizing
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
LOGD=/home/goodlad/.cache/gen3ai/tmp/sizing/arms
D=98304
STEPS=8060928
mkdir -p "$LOGD"
cd "$MAIN" || exit 2
export PYTHONPATH=$MAIN/src GEN3AI_PYTHON=$PY
ARGS=(--restart-interval-hours 3 --pin-commit "$PIN" --env-core rust --n-envs "$N" --steps "$STEPS"
      --run-name "$NAME" --seed "$SEED" --checkpoint-every-steps 2000000 --device cuda --log-level periodic)
[ "$N" != 48 ] && ARGS+=(--n-steps $((D / N)))
[ "$EP" != 10 ] && ARGS+=(--n-epochs "$EP")
[ "$BK" != - ] && ARGS+=(--t2-buckets "$BK")
CK=$(ls -1 "$MAIN/models/$NAME/checkpoints/"*_steps.zip 2>/dev/null | sed 's/.*checkpoint_\([0-9]*\)_steps.zip/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
if [ -n "$CK" ]; then
    echo "[arm] RESUME $NAME from $CK"; ARGS+=(--model "$CK")
elif [ -e "$MAIN/models/$NAME" ]; then
    echo "[arm] REFUSING: models/$NAME exists with no checkpoint — inspect it by hand"; exit 3
else
    echo "[arm] FRESH $NAME"; ARGS+=(--arch production)
fi
echo "[arm] $(date -Is) argv: ${ARGS[*]}"
"$PY" -m main.checkargs --argv "${ARGS[*]}" > "$LOGD/$NAME.checkargs.txt" 2>&1 || { echo "[arm] checkargs REFUSED"; exit 4; }
exec "$WT/scripts/ops/mem_cap.sh" --name "arm_$NAME" 56 "$WT/scripts/ops/gpu_lock.sh" timeout 7200 \
    nice -n 5 "$PY" -m main.launcher "${ARGS[@]}"
