#!/usr/bin/env bash
# wcont_b (seed 1002) ROW 1 — the untaught 8, the wcont read's run_untaught.sh VERBATIM except:
# the code tree (a worktree at 58389caa, the wcont read's own commit, with ITS OWN sim_bridge build,
# because main now carries 65f22334's reading fixes + the port's Solar Beam fix) and the ref list.
# armW = the REPRODUCTION CHECK (must be 46.19 pp, 739/1600, all 8 rows identical to the banked ones);
# wcont_p12M = a second reproduction (banked 61.69) AND the same-draw seed-1001 replicate.
set -u
WT=/home/goodlad/dev/gen3ai-wt/wcontb_read
M=/home/goodlad/dev/gen3ai/models
export PYTHONPATH=$WT/src
export POKESIM_SIM_BRIDGE_BIN=$WT/src/rust_sim/target/release/sim_bridge
export GEN3AI_MODELS_DIR=$M
export CUDA_VISIBLE_DEVICES=""
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=/home/goodlad/.claude/jobs/wcontb_read_2026-09-23/out/untaught
cd $WT
[ -f "$OUT/untaught_registry.json" ] && { echo "SKIP"; exit 0; }
echo "=== UNTAUGHT METER wcont_b $(date -Is) @ $(git -C $WT rev-parse --short HEAD) ==="
nice -n 15 $P -m main.untaught_meter \
  wcontb_p3M=$M/ai_v13_21_wcont_b/checkpoints/checkpoint_78006048_steps.zip \
  wcontb_p6M=$M/ai_v13_21_wcont_b/checkpoints/checkpoint_80912688_steps.zip \
  wcontb_p12M=$M/ai_v13_21_wcont_b/final_model.zip \
  armW=$M/ai_v13_02_flywheel_winprob/final_model.zip \
  wcont_p12M=$M/ai_v13_09_wcont/final_model.zip \
  --opponent untaught_meter_opponent \
  --workers 8 --seed 0 \
  --json "$OUT/untaught_registry.json" --md "$OUT/untaught_registry.md" 2>&1
echo "=== EXIT $? $(date -Is) ==="
