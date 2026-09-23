#!/bin/bash
# H18 — one cell. $1 = arm (U|P), $2 = teamset (away|home), $3 = port
set -u
T=/home/goodlad/.claude/jobs/9ab51de6/tmp/h18
ARM=$1; TS=$2; PORT=$3
case "$ARM" in
  U) PY=$T/python_unpatched ;;
  P) PY=$T/python_patched ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
OUT=$T/cell_${ARM}_${TS}
export PYTHONPATH=/home/goodlad/dev/gen3ai-wt/h18/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GEN3AI_ANCHORS_CONFIG=$T/anchors_h18.json
export GEN3AI_METAMON_PYTHON=$PY
cd /home/goodlad/dev/gen3ai-wt/h18
exec nice -n 15 /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.anchors \
  --model /home/goodlad/dev/gen3ai/models/ai_v13_02_flywheel_winprob/final_model.zip \
  --opponent metamon:SmallRL --regime greedy --teamset "$TS" --games 400 \
  --device cpu --server rust --port "$PORT" \
  --seed-base 20260922 --team-seed 20260914 \
  --capture-dir "$OUT/captures" --out "$OUT" --nice 15
