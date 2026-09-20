#!/usr/bin/env bash
# THE A/B CONTINUATION ANCHOR CAMPAIGN — one LANE of the job list.
#
#   bash run_lane.sh <0|1> <jobset>        jobset: smallrl | synthv2
#
# Every cell goes through the TOOL OF RECORD, `python -m main.anchors`, which starts and stops its
# OWN Showdown server (a unique 9521-9560 port per sub-cell, by PID) and verifies the greedy regime
# PER DECISION on both sides. 8000/8001 are refused in the tool's own code and never touched here.
#
# 🚨 FOUR 100-GAME SUB-CELLS PER CELL, not one 400-game invocation. Hazard H5 (Metamon's
# RecursionError on a forfeit desync) kills a whole invocation ~0.5 %/game; four sub-cells bound the
# loss at 100 games and the `[ -f summary.json ]` guard makes the lane resumable. The four
# `--team-seed` values are SHARED ACROSS ALL THREE ARMS, so the arms face the same four draws.
#
# 🚨 SEEDS ARE DIFFERENT FROM THE 2026-09-20 READ, which ran at the tool's default 20260914.
# Spacing is 10 because the peer draws at `seed + 1`.
#
# 🚨 RUN FROM THE WORKTREE (the code is pinned to this branch), with ABSOLUTE model paths into the
# MAIN checkout's models/ — a worktree has no models/. SOP: "run a campaign from its own worktree".
# --out is ALWAYS explicit and outside the repo; a stray anchors_out dirties main.
set -u
LANE="${1:?lane parity 0 or 1}"
JOBSET="${2:-smallrl}"

WT=/home/goodlad/dev/gen3ai-wt/anchor_ab
MODELS=/home/goodlad/dev/gen3ai/models
OUT="${ANCHOR_AB_OUT:-/home/goodlad/.claude/jobs/anchor_ab/out}"
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3

export PYTHONPATH="$WT/src"
export CUDA_VISIBLE_DEVICES=""       # a GPU training arm owns this box
export OMP_NUM_THREADS=1             # hazard H13: ~20x on a shared box
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

declare -A ZIP=(
  [W]="$MODELS/ai_v13_02_flywheel_winprob/final_model.zip"   # arm W, the frozen parent, 75,005,952
  [C]="$MODELS/ai_v13_09_wcont/final_model.zip"              # the continuation,          87,097,344
  [F]="$MODELS/ai_v13_08_fold1_cont/final_model.zip"         # the fold path,             87,097,344
)

# arm|opponent|teamset|games|team-seed  — round-robin by seed so every arm gains n together and a
# lane stopped early still leaves a balanced table.
if [ "$JOBSET" = "smallrl" ]; then
  JOBS=()
  for TS in away home; do
    for S in 20260919 20260929 20260939 20260949; do
      for A in W C F; do JOBS+=("$A|metamon:SmallRL|$TS|100|$S"); done
    done
  done
else
  JOBS=()
  for S in 20260919 20260929; do
    for A in W C F; do JOBS+=("$A|metamon:SyntheticRLV2|away|100|$S"); done
  done
fi

mkdir -p "$OUT"
cd "$WT"
PORT_BASE=9521
[ "$JOBSET" = "synthv2" ] && PORT_BASE=9561

for i in "${!JOBS[@]}"; do
  [ $(( i % 2 )) -ne "$LANE" ] && continue
  IFS='|' read -r ARM OPP TS GAMES SEED <<< "${JOBS[$i]}"
  SHORT="${OPP#metamon:}"
  TAG="${ARM}_${SHORT}_${TS}_s${SEED}"
  D="$OUT/$TAG"
  if [ -f "$D/summary.json" ]; then echo "SKIP $TAG (done)"; continue; fi
  PORT=$(( PORT_BASE + i ))
  echo "=== START $TAG port=$PORT $(date -Is) ==="
  nice -n 15 "$P" -m main.anchors \
      --model "${ZIP[$ARM]}" --opponent "$OPP" --regime greedy --teamset "$TS" \
      --games "$GAMES" --device cpu --nice 15 --port "$PORT" --team-seed "$SEED" \
      --out "$D" > "$OUT/$TAG.log" 2>&1
  echo "=== END $TAG rc=$? $(date -Is) ==="
  tail -25 "$OUT/$TAG.log"
done
echo "=== LANE $LANE $JOBSET COMPLETE $(date -Is) ==="
