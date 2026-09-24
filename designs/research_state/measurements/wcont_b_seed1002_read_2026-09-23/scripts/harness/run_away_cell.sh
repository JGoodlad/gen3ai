#!/usr/bin/env bash
# wcont_b ROW 3 — the SmallRL greedy AWAY cell, the wcont read's run_away_cell.sh with the ref swapped,
# the code tree at 58389caa (its anchors default = the Node server the wcont cell used) and port 9451.
set -u
WT=/home/goodlad/dev/gen3ai-wt/wcontb_read
M=/home/goodlad/dev/gen3ai/models
export PYTHONPATH=$WT/src
export POKESIM_SIM_BRIDGE_BIN=$WT/src/rust_sim/target/release/sim_bridge
export GEN3AI_MODELS_DIR=$M
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/wcontb_read_2026-09-23/out/anchors/away
mkdir -p "$OUT"
cd $WT
TAG=wcontb_p12M
[ -f "$OUT/${TAG}_smallrl_greedy_away/summary.json" ] && { echo "SKIP $TAG"; exit 0; }
echo "=== $TAG greedy away $(date -Is) ==="
nice -n 15 $P -m main.anchors --model $M/ai_v13_21_wcont_b/final_model.zip \
  --opponent metamon:SmallRL --regime greedy --teamset away --games 100 \
  --device cpu --nice 15 --port 9451 --out "$OUT/${TAG}_smallrl_greedy_away" 2>&1
echo "=== EXIT $? $TAG $(date -Is) ==="
