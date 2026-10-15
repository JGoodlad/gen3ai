#!/usr/bin/env bash
# E10: checkargs + launcher --dry-run of the X26 argv at this checkout's HEAD (no GPU; scratch models dir).
set -u
# rule 5b: low priority, confined to the agent cores (children inherit both)
renice -n 19 -p $$ >/dev/null
taskset -cp 5-7,13-15 $$ >/dev/null
WT=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a4a0f0b499e8a107d
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
OUT=/tmp/x26exit
cd "$WT"
export PYTHONPATH="$WT/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
export GEN3AI_MODELS_DIR="$OUT/scratch_models"
mkdir -p "$GEN3AI_MODELS_DIR"
FLAGS="--arch production --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all --checkpoint-every-steps 1000000 --steps 500000000 --seed 2601 --device cuda --run-name rb_x26_s2601"
HEAD=$(git -C "$WT" rev-parse --short=8 HEAD)
echo "HEAD $HEAD"
timeout 900 "$PY" -m main.checkargs --argv "$FLAGS" > "$OUT/checkargs_$HEAD.log" 2>&1
echo "checkargs exit $?" | tee -a "$OUT/checkargs_$HEAD.log"
# shellcheck disable=SC2086
timeout 900 "$PY" -m main.launcher --dry-run $FLAGS > "$OUT/dryrun_$HEAD.log" 2>&1
echo "dry-run exit $?" | tee -a "$OUT/dryrun_$HEAD.log"
