#!/usr/bin/env bash
# F-XC-5 re-check at HEAD: K9(b) excluded share, fixed_mass, regime A (seeded 20-snapshot self-play pool).
# Launch 1 (fmB5, regime B) only to make the pool template; launch 2 (fmA5) is the read.
# DEVIATIONS from the cost ablation (reported): --no-compile-trainer (a fixed_mass compile gate FATALs at
# HEAD, F-XC-4); the template is fmB5's checkpoint after its first real update (not 1.2M steps).
set -u
export S=/home/goodlad/.cache/gen3ai/x5fxc4/fxc5
export W=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a12467d9f7faa4fe4
export PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export GEN3AI_GPU_LEASE_TOKEN=$(cat /home/goodlad/.cache/gen3ai/x5fxc4/lease_token)
export PROF_FIT_HEADROOM_MIB=256 PROF_SLOT_LOAD_TOL_MIB=8
X26="--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all"
COMMON="--arch production --device cuda --steps 6000000 --eval-freq 50000000 $X26 --belief-tokens fixed_mass --allow-nonproduction-arch --no-compile-trainer --behaviour-check warn"
step=${1:-all}
if [ "$step" = all ] || [ "$step" = B ]; then
  bash "$S/drive.sh" fmB5 2 $COMMON
  echo "B done" >> "$S/fxc5.progress"
fi
if [ "$step" = all ] || [ "$step" = A ]; then
  tpl=$(ls -t "$S"/models/fmB5/final_model*.zip "$S"/models/fmB5/checkpoints/*.zip 2>/dev/null | head -1)
  echo "template $tpl" >> "$S/fxc5.progress"
  cd "$W" && PYTHONPATH=$W/src GEN3AI_MODELS_DIR=$S/models "$PY" "$S/build_pool.py" --template "$tpl" \
      --out "$S/models/fmA5/snapshots" >> "$S/fxc5.progress" 2>&1
  bash "$S/drive.sh" fmA5 6 $COMMON
  echo "A done" >> "$S/fxc5.progress"
fi
