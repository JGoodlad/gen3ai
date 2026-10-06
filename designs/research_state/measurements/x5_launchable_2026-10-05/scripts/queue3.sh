#!/usr/bin/env bash
# x5-perf-encode unit 2 (2026-10-05): the blob identity proof for the F-XC-3 fix, the REAL fixed_mass launch
# (no PROF_* overrides; slot loads LOGGED only), then the paired ABAB speed benchmark (regime A, X26 heads).
set -u
export S=/home/goodlad/.cache/gen3ai/x5pe/launch
export W=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a72eda887a3a57819
export PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
export GEN3AI_GPU_LEASE_TOKEN=$(cat /home/goodlad/.cache/gen3ai/x5pe/lease_token)
X26="--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all"
AB="--arch production --device cuda --steps 15000000 --seed 1001 $X26 --snapshot-ladder-games 0 --checkpoint-every-steps 1000000"
FM="--belief-tokens fixed_mass --allow-nonproduction-arch"
TPL_FM=/home/goodlad/.cache/gen3ai/x5fxc4/fxc5/models/fmB5/final_model_interrupted.zip
TPL_BLOB=$S/models/blobB/final_model_interrupted.zip
log() { echo "$1 $(date +%s)" >> "$S/progress3"; }
pool() {  # pool <tag> <template>
  cd "$W" && PYTHONPATH=$W/src GEN3AI_MODELS_DIR=$S/models "$PY" "$S/build_pool.py" --template "$2" \
      --out "$S/models/$1/snapshots" >> "$S/progress3.pool" 2>&1
}
log "queue3 start"
if [ "${SKIP_IDENTITY:-}" = "" ]; then
  bash /tmp/x5pe/identity2.sh > /tmp/x5pe/identity2.log 2>&1
  log "identity2 done"
fi
if [ "${SKIP_REAL:-}" = "" ]; then
  # 2. the REAL D-6 / slot-load / gate / canary launch: no PROF_* overrides (PROF_LOG_SLOT_LOADS only LOGS)
  pool fmREAL "$TPL_FM"
  PROF_LOG_SLOT_LOADS=1 bash "$S/drive.sh" fmREAL 12 $AB $FM --eval-freq 400000 --promote-threshold 0.0
  log "fmREAL done"
fi
# 3. the paired benchmark: A = blob, B = fixed_mass, ABABAB, the A/B argv (eval at its default, no eval cycle
#    inside 10 updates), each block a fresh run into a seeded 20-snapshot pool (regime A)
for blk in 1 2 3; do
  pool "sB_blob_$blk" "$TPL_BLOB"
  PROF_LOG_SLOT_LOADS=1 bash "$S/drive.sh" "sB_blob_$blk" 10 $AB
  log "sB_blob_$blk done"
  pool "sB_fm_$blk" "$TPL_FM"
  PROF_LOG_SLOT_LOADS=1 bash "$S/drive.sh" "sB_fm_$blk" 10 $AB $FM
  log "sB_fm_$blk done"
done
log "QUEUE3_DONE"
