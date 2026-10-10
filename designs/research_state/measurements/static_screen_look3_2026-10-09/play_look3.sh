#!/usr/bin/env bash
# STATIC-TOKEN SCREEN LOOK 3 (the FINAL look): the 39 NEW cells of the registered 8 x 8 head-to-head CROSS
# (design_static_tokens.md §8.1 / §8.2), INCREMENTAL and RESUMABLE, exactly as looks 1 and 2 played
# (../static_screen_look2_2026-10-09/play_look2.sh): 1,000 mirrored pairs per cell in batches of 500, both sides
# greedy, schedule seed 0, purpose `ab`, oracle reveal off, ONE engine (two architecture groups: static rows, legacy
# columns).
#
#   st_look3_steps    static 15M final x legacy 15M final    8 x 8 - 5 x 5 = 39 cells    family st_screen_strength_steps
#
# Look 1's 9 cells (st_look1_steps) and look 2's 16 (st_look2_steps) are REUSED from the ledger: same engine, regime,
# pin and checkpoints. plan_look3.py refuses unless those requests hold exactly those cells and look 3's holds nothing
# outside the plan. Every batch is one durable, fsynced row in the eval COUNT ledger under its claim; a re-run re-plans
# and the ledger's resume skips every recorded batch, so this script is safe to kill and to start again.
#
# THE PIN (§8.2: "Every cross and read plays at P_st, never at HEAD"): every game is played by the code of a checkout
# DETACHED at 6c6d2e09 (PIN_DIR), PYTHONPATH=$PIN_DIR/src. This script REFUSES (exit 5) unless `git rev-parse HEAD`
# in PIN_DIR is 6c6d2e09 with src/ and data/ clean, and logs it.
#
# Env: PIN_DIR, ROOT (ledger root; empty = the run archive's models/_ledger, the registered home), MODELS, PAIRS (1000),
# BATCH (500), DEVICE (cuda), LABEL (st_look3), PLAN_DIR, OPS (main's scripts/ops: gpu_lock, mem_cap).
# Run DETACHED, with the GPU lease token (file) for cuda.
set -u
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=6c6d2e0942e2111703a7e6d79bfadb31c8e51f01
PIN_DIR=${PIN_DIR:-/home/goodlad/dev/gen3ai/.claude/worktrees/st-look1-pin-6c6d2e09}
MODELS=${MODELS:-/home/goodlad/dev/gen3ai/models}
OPS=${OPS:-/home/goodlad/dev/gen3ai/scripts/ops}
PAIRS=${PAIRS:-1000}
BATCH=${BATCH:-500}
DEVICE=${DEVICE:-cuda}
LABEL=${LABEL:-st_look3}
ROOT=${ROOT:-}
PLAN_DIR=${PLAN_DIR:-$HOME/.cache/gen3ai/st_look3/plan}
mkdir -p "$PLAN_DIR"
OUT=(); [ -n "$ROOT" ] && OUT=(--out "$ROOT")
RROOT=(); [ -n "$ROOT" ] && RROOT=(--root "$ROOT")

cd "$PIN_DIR" || { echo "REFUSED: no pin checkout at $PIN_DIR"; exit 5; }
export PYTHONPATH=$PIN_DIR/src
HEAD=$(git rev-parse HEAD)
echo "=== $(date -Is) static-screen look 3 (FINAL): play checkout $PIN_DIR at git rev-parse HEAD = $HEAD (required $PIN)"
if [ "$HEAD" != "$PIN" ]; then echo "REFUSED: the play checkout is not at the registered P_st 6c6d2e09"; exit 5; fi
if [ -n "$(git status --porcelain --untracked-files=no -- src data)" ]; then
  echo "REFUSED: the pin checkout's src/ or data/ is modified"; git status --short -- src data; exit 5
fi

# The family was registered at look 1 with EXACTLY these terms; family-register is idempotent only on identical terms,
# so the rule string below is look 1's, byte for byte.
FAMILY=st_screen_strength_steps
RULE="design_static_tokens.md §8.1+§8.2 (P_st 6c6d2e09; amendments 2026-10-08 1-2): static 15M x legacy 15M cross, t_NI=(Δ̂+3.5)/√V̂, t_SUP=Δ̂/√V̂, df=2(n-1), looks 3/5/8 seeds at OBF 5.761/2.683/1.874, rule 8; estimator main.h2h.cross"
"$PY" -m main.eval_ledger family-register "${RROOT[@]}" --id "$FAMILY" --decision-kind ab_verdict --rule "$RULE" \
  --protocol gen3_eval_protocol_v1_h2h --commit "$PIN" || exit 3

if [ "$DEVICE" = cuda ]; then PRE=("$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32); else PRE=("$OPS/mem_cap.sh" 32); fi
CELLS="$PLAN_DIR/steps_look3.json"
n=$("$PY" "$HERE/plan_look3.py" --out "$CELLS" --models "$MODELS" "${RROOT[@]}") \
  || { echo "PLAN REFUSED ($(date -Is))"; exit 3; }
echo "=== $(date -Is) st_look3_steps: $n cell(s) planned ($CELLS)"
"${PRE[@]}" "$PY" -m main.h2h play-many --cells "$CELLS" --pairs "$PAIRS" \
  --batch-pairs "$BATCH" --device "$DEVICE" --purpose ab --family "$FAMILY" --request st_look3_steps \
  --label "$LABEL" --oracle-reveal-mode off "${OUT[@]}" || { echo "FAILED st_look3_steps ($(date -Is))"; exit 4; }
echo "=== $(date -Is) DONE"
