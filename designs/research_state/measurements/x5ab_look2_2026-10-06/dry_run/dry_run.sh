#!/usr/bin/env bash
# The look-2 CPU dry run: scratch ledger + scratch models, 2 x 2 seeds, look 1 = 1 x 1, 20 pairs per cell.
set -u
W=$(cd "$(dirname "$0")/../../../../.." && pwd)   # the repo root of this checkout
D=$W/designs/research_state/measurements/x5ab_look2_2026-10-06
S=/home/goodlad/.cache/gen3ai/x5look2/dry
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
cd "$W"
export PYTHONPATH=$W/src CUDA_VISIBLE_DEVICES=
export ROOT=$S/ledger MODELS=$S/models PLAN_DIR=$S/plan DEVICE=cpu PAIRS=20 BATCH=10 SEEDS="1001 1002" LOOK1_SEEDS="1001"
echo "##### 1. look 2 BEFORE look 1 exists: the plan must refuse"
EMIT=look2 STEPS=a bash "$D/play_look2.sh"; r=$?; echo "exit $r"
echo "##### 2. DRY RUN look 1 (1 cell per read)"
EMIT=look1 STEPS="a b" LABEL=x5ab_look1 bash "$D/play_look2.sh"; r=$?; echo "exit $r"
[ "$r" = 0 ] || exit 1
echo "##### 3. look 2 (3 new cells per read)"
EMIT=look2 STEPS="a b" bash "$D/play_look2.sh"; r=$?; echo "exit $r"
[ "$r" = 0 ] || exit 1
echo "##### 4. look 2 again: every batch recorded, nothing replayed"
EMIT=look2 STEPS="a b" bash "$D/play_look2.sh"; r=$?; echo "exit $r"
[ "$r" = 0 ] || exit 1
echo "##### 5. the read"
"$PY" "$D/read_look2.py" --root "$S/ledger" --models "$S/models" --seeds 1001 1002 --look1-seeds 1001 \
  --min-pairs 20 --out-dir "$S/read"; echo "exit $?"
"$PY" "$D/read_look2.py" --root "$S/ledger" --models "$S/models" --seeds 1001 1002 --look1-seeds 1001 \
  --min-pairs 20 --out-dir "$S/read" --progress
"$PY" -m main.eval_ledger audit --root "$S/ledger"; echo "audit exit $?"
echo "##### 6. TAMPER: play look 1 cell fm1 x blob1 under x5ab_look2_steps; the plan must refuse, the read must be INCONCLUSIVE"
printf "[[\"%s\", \"%s\"]]\n" "$S/models/rb_x5ab_fm_s1001/final_model.zip" "$S/models/rb_x5ab_blob_s1001/final_model.zip" > "$S/plan/tamper.json"
scripts/ops/mem_cap.sh 24 "$PY" -m main.h2h play-many --cells "$S/plan/tamper.json" --pairs 20 --batch-pairs 10 --device cpu --purpose ab --family x5ab_strength_steps --request x5ab_look2_steps --label tamper --oracle-reveal-mode off --out "$S/ledger" 2>&1 | grep -v -i "warn\|deserialize"
EMIT=look2 STEPS=a bash "$D/play_look2.sh"; echo "exit $? (expect 3)"
"$PY" "$D/read_look2.py" --root "$S/ledger" --models "$S/models" --seeds 1001 1002 --look1-seeds 1001 --min-pairs 20 --out-dir "$S/read_tamper" --reads steps; echo "exit $?"
grep -o "reasons[^;]*" "$S/read_tamper/result.md"
echo "##### DONE $(date -Is)"
