#!/usr/bin/env bash
# THE REPRESENTATION-LEVEL OPPONENT-CLASS DECODE, end to end. Offline, CPU only, niced,
# nothing written under models/.
#
#   1. extract     one frozen forward of `step_10000032` over ONE eval tree per (run, draw) ->
#                  meta.npy + pooled.npy (`value_pooled`, the win head's literal input).
#                  `../winprob_refit_ncurve_2026-09-10/extract.py`, UNMODIFIED.
#   2. frame_check the out-of-fold, battle-grouped decode: `pooled_to_opp_class_AUC` and
#                  `V_to_opp_class_AUC` on THE SAME ROWS AND FOLDS, plus the own-team leak.
#                  `../winprob_refit_ncurve_2026-09-10/frame_check.py`, UNMODIFIED.
#   3. tabulate    deltas vs ctrl10M, the control-pair floor, the branch test of PREDICTION.md §3.
#
# 🚨 Run the EXTRACTION from the MAIN checkout's code (PYTHONPATH -> main's src). A worktree has no
# models/ and the rust bridge binary lives in main's target/.
set -euo pipefail
MAIN=/home/goodlad/dev/gen3ai
HERE="$(cd "$(dirname "$0")" && pwd)"
NC="$MAIN/designs/research_state/measurements/winprob_refit_ncurve_2026-09-10"
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3}
HP=${HP_EVAL:-/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval}
OUT=${REPDECODE_OUT:-/home/goodlad/.claude/jobs/9ab51de6/tmp/repdecode}
export PYTHONPATH="$MAIN/src"
export POKESIM_SIM_BRIDGE_BIN="$MAIN/src/rust_sim/target/release/sim_bridge"
export CUDA_VISIBLE_DEVICES=""
RUNS="ai_v12_17_ladder_strata ai_v12_10_ladder_vf15 ai_v12_11_ladder_ctrl10M ai_v12_15_ladder_ctrl10M_b ai_v12_16_ladder_ctrl10M_c"
mkdir -p "$OUT"

for RUN in $RUNS; do
  for DRAW in hp800 hp800b; do
    T="$HP/$DRAW/$RUN/eval_traces/step_10000032"
    [ -d "$T" ] || { echo "skip $RUN/$DRAW (no tree)"; continue; }
    D="$OUT/$RUN/$DRAW"
    [ -f "$D/pooled.npy" ] && { echo "have $RUN/$DRAW"; continue; }
    mkdir -p "$D"
    nice -n 15 "$PY" "$NC/extract.py" --tree "$DRAW=$T" --snapshot "$T/snapshot.zip" \
      --out-dir "$D" --threads 2 --batch 256
  done
done

ARGS=()
for RUN in $RUNS; do for DRAW in hp800 hp800b; do
  [ -f "$OUT/$RUN/$DRAW/pooled.npy" ] && ARGS+=(--dir "$RUN@$DRAW=$OUT/$RUN/$DRAW")
done; done
nice -n 15 "$PY" "$NC/frame_check.py" "${ARGS[@]}" --out "$HERE/frame_check.json" \
  --cap 20000 --seed 20260910
nice -n 15 "$PY" "$HERE/tabulate.py" --in "$HERE/frame_check.json" \
  --out-json "$HERE/branch_test.json" --out-md "$HERE/TABLE.md"
