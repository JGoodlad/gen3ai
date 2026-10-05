#!/usr/bin/env bash
# F-XC-5 re-check, launch 2 retry (fmA5b): fmA5 crashed at its first logger dump (the stdout table's key
# truncation collision, a FINDING); this adds the measurement-only PROF_LOG_KEY_MAX override. Same pool
# construction from fmB5's checkpoint, same argv.
set -u
export S=/home/goodlad/.cache/gen3ai/x5fxc4/fxc5
export W=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a12467d9f7faa4fe4
export PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export GEN3AI_GPU_LEASE_TOKEN=$(cat /home/goodlad/.cache/gen3ai/x5fxc4/lease_token)
export PROF_FIT_HEADROOM_MIB=256 PROF_SLOT_LOAD_TOL_MIB=8 PROF_LOG_KEY_MAX=80
X26="--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all"
COMMON="--arch production --device cuda --steps 6000000 --eval-freq 50000000 $X26 --belief-tokens fixed_mass --allow-nonproduction-arch --no-compile-trainer --behaviour-check warn"
tpl="$S/models/fmB5/final_model_interrupted.zip"
cd "$W" && PYTHONPATH=$W/src GEN3AI_MODELS_DIR=$S/models "$PY" "$S/build_pool.py" --template "$tpl" \
    --out "$S/models/fmA5b/snapshots" >> "$S/fxc5.progress" 2>&1
bash "$S/drive.sh" fmA5b 6 $COMMON
echo "A5b done" >> "$S/fxc5.progress"
