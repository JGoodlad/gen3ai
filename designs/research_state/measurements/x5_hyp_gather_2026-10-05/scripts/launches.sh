#!/usr/bin/env bash
# x5-perf-encode: the measured launches (cost ablation's argv + the nanfix e2e's), drive.sh unchanged.
#   1. fmA_gather — fixed_mass, regime A, the SAME 20-snapshot pool construction as fmA_nanfix (the BEFORE,
#      x5_fxc4_nanfix_2026-10-05/e2e: template fmB5), --behaviour-check warn as there;
#   2. blobB — blob, regime B (empty pool): the same-day blob comparator.
# Measurement-only overrides on the fixed_mass launch: PROF_FIT_HEADROOM_MIB=256 (so a launch under the floor
# still REPORTS its headroom instead of refusing), PROF_SLOT_LOAD_TOL_MIB=8 (F-XC-3 stands, not this unit's).
set -u
export S=/home/goodlad/.cache/gen3ai/x5pe/launch
export W=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a72eda887a3a57819
export PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export GEN3AI_GPU_LEASE_TOKEN=$(cat /home/goodlad/.cache/gen3ai/x5pe/lease_token)
X26="--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all"
BASE="--arch production --device cuda --steps 6000000 --eval-freq 50000000 $X26"
tpl=/home/goodlad/.cache/gen3ai/x5fxc4/fxc5/models/fmB5/final_model_interrupted.zip
which=${1:-all}
if [ "$which" = all ] || [ "$which" = fm ]; then
  cd "$W" && PYTHONPATH=$W/src GEN3AI_MODELS_DIR=$S/models "$PY" "$S/build_pool.py" --template "$tpl" \
      --out "$S/models/fmA_gather/snapshots" >> "$S/progress" 2>&1
  PROF_FIT_HEADROOM_MIB=256 PROF_SLOT_LOAD_TOL_MIB=8 bash "$S/drive.sh" fmA_gather 12 $BASE \
      --belief-tokens fixed_mass --allow-nonproduction-arch --behaviour-check warn
  echo "fmA_gather done $(date +%s)" >> "$S/progress"
fi
if [ "$which" = all ] || [ "$which" = blob ]; then
  bash "$S/drive.sh" blobB 11 $BASE
  echo "blobB done $(date +%s)" >> "$S/progress"
fi
echo LAUNCHES_DONE >> "$S/progress"
