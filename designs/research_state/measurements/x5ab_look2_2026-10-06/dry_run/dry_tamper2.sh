#!/usr/bin/env bash
# Tamper 2: plant the look-1 cell fm1 x blob1 under x5ab_look2_steps at ANOTHER schedule seed (the ledger refused the
# same-seed replay itself: DuplicateBatchError). The plan must refuse (stray cell), the read must be INCONCLUSIVE.
set -u
W=$(cd "$(dirname "$0")/../../../../.." && pwd)   # the repo root of this checkout
D=$W/designs/research_state/measurements/x5ab_look2_2026-10-06
S=/home/goodlad/.cache/gen3ai/x5look2/dry
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
cd "$W"
export PYTHONPATH=$W/src CUDA_VISIBLE_DEVICES=
echo "##### 7. TAMPER 2 (schedule seed 7)"
scripts/ops/mem_cap.sh 24 "$PY" -m main.h2h play-many --cells "$S/plan/tamper.json" --pairs 20 --batch-pairs 10 \
  --device cpu --purpose ab --family x5ab_strength_steps --request x5ab_look2_steps --label tamper --seed 7 \
  --oracle-reveal-mode off --out "$S/ledger" 2>&1 | grep "play-many:\|batch\|Error"
ROOT=$S/ledger MODELS=$S/models PLAN_DIR=$S/plan DEVICE=cpu PAIRS=20 BATCH=10 SEEDS="1001 1002" LOOK1_SEEDS="1001" \
  EMIT=look2 STEPS=a bash "$D/play_look2.sh"; echo "exit $? (expect 3)"
"$PY" "$D/read_look2.py" --root "$S/ledger" --models "$S/models" --seeds 1001 1002 --look1-seeds 1001 \
  --min-pairs 20 --out-dir "$S/read_tamper" --reads steps; echo "exit $?"
grep "reasons" "$S/read_tamper/result.md"
echo "##### DONE $(date -Is)"
