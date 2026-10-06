#!/usr/bin/env bash
# x5-perf-nan step 5: a short fixed_mass regime-A run (seeded 20-snapshot pool, the X26 heads), COMPILED
# (--compile-trainer default on cuda), through the startup gate, the update-10 canary and >= 2 logger dumps.
# Measurement-only overrides: PROF_FIT_HEADROOM_MIB=256, PROF_SLOT_LOAD_TOL_MIB=8 (F-XC-2 / F-XC-3 stand).
# NO PROF_LOG_KEY_MAX: the stdout table is the production one (the logger fix is under test).
set -u
export S=/home/goodlad/.cache/gen3ai/x5pn/e2e
export W=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a30fcd2b9887c1fda
export PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export GEN3AI_GPU_LEASE_TOKEN=$(cat /home/goodlad/.cache/gen3ai/x5pn/lease_token)
export PROF_FIT_HEADROOM_MIB=256 PROF_SLOT_LOAD_TOL_MIB=8
X26="--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all"
COMMON="--arch production --device cuda --steps 6000000 --eval-freq 50000000 $X26 --belief-tokens fixed_mass --allow-nonproduction-arch --behaviour-check warn"
tag=${1:-fmA_nanfix}
tpl=/home/goodlad/.cache/gen3ai/x5fxc4/fxc5/models/fmB5/final_model_interrupted.zip
cd "$W" && PYTHONPATH=$W/src GEN3AI_MODELS_DIR=$S/models "$PY" "$S/build_pool.py" --template "$tpl" \
    --out "$S/models/$tag/snapshots" >> "$S/e2e.progress" 2>&1
bash "$S/drive.sh" "$tag" 12 $COMMON
echo "$tag done $(date +%s)" >> "$S/e2e.progress"
