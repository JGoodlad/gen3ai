#!/usr/bin/env bash
# G-A — the external-anchor KILL guard of population-loop ROUND 2 (round-2 registration §4.3), INCREMENTAL.
# Round 1's driver with the models (B2 = ai_v13_27_popr2_loop, the round-2 loop generalist; C2 =
# ai_v13_28_popr2_ctrl, the round-2 no-exploiter control), the tree path and the niceness (19,
# coordinator 2026-09-25; changes no number) changed: 6 seeds x 2 teamsets x 2 models = 24 units.
# G0 is NOT re-played: the registration's descriptors reuse round 1's banked units (same tree, seeds).
# Round 1's header follows.
#
# Registered: main.anchors --opponent metamon:SmallRL --regime greedy, 1,200 games per model =
# 600 --teamset away + 600 --teamset home, --device cpu, OMP_NUM_THREADS=1, one cell at a time, for
# B (ai_v13_22_popr1_loop, the loop generalist), C (ai_v13_23_popr1_ctrl, the no-exploiter control)
# and G0 (ai_v13_12_plateau, the plateau parent). KILL iff B - C (pooled 1,200, Newcombe) is OUTSIDE
# and BELOW the 0.110 run floor.
#
# UNITS (owner rule, ORCHESTRATOR_SOP §2): one 100-game main.anchors cell per (team seed, teamset,
# model) — 6 seeds x 2 teamsets x 3 models = 36 units, ~5-8 min each, ONE AT A TIME (the SOP's
# one-cell rule). DECLARED DEVIATION: the registration names "--team-seed 0" for a single 600-game
# cell per teamset; 100-game units need distinct draws, so each unit takes --team-seed in
# {0,10,20,30,40,50} (stride 10: the tool seeds THEIR draw at seed+1, so adjacent seeds would share
# a stream), SHARED ACROSS B, C and G0 (the banked anchor_ab_continuation_2026-09-20 convention) and
# --seed-base = the same value, so the three models face the same team draws and the same dice.
# Order is seed-major, model-minor: all three models advance together.
#
# TREE: 7c511161 (the orchestrator's ruling 2026-09-24): the last commit before the 65f22334
# training-input boundary; contains 6eb9c776 plus regime_verified_decisions (6e25c380), --server
# rust (33338688) and serial challenges (558586d0). Its own release sim_bridge.
#
# RESUME: a unit whose summary.json says status OK is skipped; any other existing unit dir is MOVED
# aside to <dir>.partial.<epoch> (never deleted) and the unit re-runs from game 1.
# Run DETACHED:  setsid nohup ga_driver.sh > <rows>/_driver.log 2>&1 < /dev/null &
set -u
TREE=/home/goodlad/dev/gen3ai-wt/pin-7c511161-popr2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROWS="${GA_ROWS:-$HERE/../ga_rows}"
ONLY="${GA_ONLY:-}"            # optional: restrict to units whose name contains this string (tests)
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH=$TREE/src
export POKESIM_SIM_BRIDGE_BIN=$TREE/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$ROWS"
cd "$TREE" || exit 9
echo $$ > "$ROWS/_driver.pid"

declare -A MODEL=(
  [B2_loop]=/home/goodlad/dev/gen3ai/models/ai_v13_27_popr2_loop/final_model.zip
  [C2_ctrl]=/home/goodlad/dev/gen3ai/models/ai_v13_28_popr2_ctrl/final_model.zip
)
ORDER=(B2_loop C2_ctrl)

ok() { [ -f "$1/summary.json" ] && $P -c "import json,sys; d=json.load(open('$1/summary.json')); sys.exit(0 if d.get('status')=='OK' and d.get('n')==100 else 1)"; }

for SEED in 0 10 20 30 40 50; do
  for TS in away home; do
    for M in "${ORDER[@]}"; do
      U="${M}_${TS}_s${SEED}"
      [ -n "$ONLY" ] && [[ "$U" != *"$ONLY"* ]] && continue
      D="$ROWS/$U"
      if ok "$D"; then echo "SKIP $U (OK on disk)"; continue; fi
      [ -e "$D" ] && mv "$D" "$D.partial.$(date +%s)"
      echo "=== $U  $(date -Is)  load=$(cut -d' ' -f1-3 /proc/loadavg)"
      nice -n 19 $P -m main.anchors --model "${MODEL[$M]}" --opponent metamon:SmallRL \
        --server rust --regime greedy --teamset "$TS" --games 100 --team-seed "$SEED" --seed-base "$SEED" \
        --device cpu --nice 19 --out "$D" > "$ROWS/$U.log" 2>&1
      rc=$?
      echo "=== $U rc=$rc  $(date -Is)"
      if ! ok "$D"; then echo "UNIT FAILED $U (rc=$rc) — stopping; re-run the driver to resume"; exit 3; fi
    done
  done
done
echo "ALL UNITS DONE $(date -Is)"
