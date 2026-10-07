#!/usr/bin/env bash
# X5 A/B LOOK 3: the NEW cells of the registered head-to-head cross (design_x5_belief_tokens.md §7.4, §7.8-§7.9),
# INCREMENTAL and RESUMABLE, exactly as looks 1-2 play a cell (`../x5ab_look2_2026-10-06/play_look2.sh`): 1,000
# mirrored pairs per cell in batches of 500, both sides greedy, schedule seed 0, purpose `ab`, oracle reveal off.
#
#   (a) x5ab_look3_steps    fixed_mass 15M x blob 15M   39 new cells   family x5ab_strength_steps      (registered)
#   (b) x5ab_look3_wall     fixed_mass 12M x blob 15M   39 new cells   family x5ab_strength_wall       (registered)
#   (c) x5ab_look3_wall13   fixed_mass 13M x blob 15M   64 new cells   family x5ab_sensitivity_wall13  (SENSITIVITY)
#
# Same FAMILIES as looks 1-2 for (a)/(b) (registered at bcb0296c with all three looks' boundaries in the rule), a NEW
# REQUEST per look; looks 1-2's 9 + 16 cells per read are REUSED from the ledger, never replayed (`plan_look3.py`'s
# uniqueness guard, by checkpoint sha256). (c) is the 13M steady-state SENSITIVITY line of the 2026-10-06 Decision
# row "MATCHED-WALL-TIME definition fixed BEFORE look 3 is read" (11d27574): its own family, registered at the commit
# it is played at, its rule string saying it is never the read. A re-run re-plans and the ledger's resume skips every
# recorded batch, so this script is safe to kill and to start again. One engine per read (one `play-many`).
#
# THE PIN (Decision record 2026-10-06 "The LOOK-3 CROSS plays at the TRAINING code", bef16d61): every game is played
# by the code of a checkout DETACHED at 706fa536 (PIN_DIR), with PYTHONPATH=$PIN_DIR/src. This script REFUSES (exit 5)
# unless `git rev-parse HEAD` in PIN_DIR is 706fa536, and logs it. HEAD-of-main code (2d29c4c0's legality fix,
# f0d673fd's op fix) is NOT used for any game.
#
# Env: PIN_DIR (the 706fa536 checkout), ROOT (ledger root; empty = the run archive's models/_ledger, the registered
# home), MODELS (the archive's models/), PAIRS (1000), BATCH (500), DEVICE (cuda), STEPS (a b c), LABEL (x5ab_look3),
# SEEDS (1001..1008), LOOK_SEEDS ("3 5"), PLAN_DIR, OPS (the scripts/ops of the main checkout: gpu_lock, mem_cap),
# DRY (1 = a CPU dry run into a scratch ROOT/MODELS; EMIT_LOOK honoured only then),
# EMIT_LOOK (3; 1|2 = DRY RUN only: play an earlier look's block into the scratch ledger first).
# Run DETACHED, with the GPU lease token for cuda.
set -u
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=706fa536ef53d9a680d2461f16613e0d1430d176
PIN_DIR=${PIN_DIR:-/home/goodlad/dev/gen3ai/.claude/worktrees/x5-look3-pin-706fa536}
MODELS=${MODELS:-/home/goodlad/dev/gen3ai/models}
OPS=${OPS:-/home/goodlad/dev/gen3ai/scripts/ops}
PAIRS=${PAIRS:-1000}
BATCH=${BATCH:-500}
DEVICE=${DEVICE:-cuda}
STEPS=${STEPS:-a b c}
LABEL=${LABEL:-x5ab_look3}
SEEDS=${SEEDS:-1001 1002 1003 1004 1005 1006 1007 1008}
LOOK_SEEDS=${LOOK_SEEDS:-3 5}
EMIT_LOOK=${EMIT_LOOK:-3}
DRY=${DRY:-0}
ROOT=${ROOT:-}
PLAN_DIR=${PLAN_DIR:-$HOME/.cache/gen3ai/x5look3/plan}
mkdir -p "$PLAN_DIR"
OUT=(); [ -n "$ROOT" ] && OUT=(--out "$ROOT")
RROOT=(); [ -n "$ROOT" ] && RROOT=(--root "$ROOT")
[ "$DRY" = 1 ] || [ "$EMIT_LOOK" = 3 ] || { echo "REFUSED: EMIT_LOOK=$EMIT_LOOK is a DRY-RUN option"; exit 2; }

cd "$PIN_DIR" || { echo "REFUSED: no pin checkout at $PIN_DIR"; exit 5; }
export PYTHONPATH=$PIN_DIR/src
HEAD=$(git rev-parse HEAD)
echo "=== $(date -Is) look 3: play checkout $PIN_DIR at git rev-parse HEAD = $HEAD (required $PIN)"
if [ "$HEAD" != "$PIN" ]; then echo "REFUSED: the play checkout is not at the registered pin 706fa536"; exit 5; fi
if [ -n "$(git status --porcelain --untracked-files=no -- src data)" ]; then
  echo "REFUSED: the pin checkout's src/ or data/ is modified"; git status --short -- src data; exit 5
fi

# The families' REGISTERED terms (family-register is idempotent only on identical terms; never HEAD's commit).
FAMILY_COMMIT=bcb0296c0edf9d9f6602f23bcfbc6b14e5df8ef7
RULE="design_x5_belief_tokens.md §7.4 (+§7.6-§7.9 Amendments 2-5): cross t=(Δ̂+3.5)/√V̂, df=2(n-1), looks 5.761/2.683/1.874; estimator main.h2h.cross"
SENS_RULE="design_x5_belief_tokens.md Decision record 2026-10-06 (matched wall = END-TO-END, 12M): SENSITIVITY line only, fixed_mass 13M (steady-state s +14.3%) x blob 15M, reported beside the 12M read and NEVER a verdict; estimator main.h2h.cross at look 3"

# ENGINE DRIFT: looks 1-2 played at bcb0296c / 131f3412; look 3 plays at 706fa536. Every commit since bcb0296c under
# src/ or data/ must be on the REVIEWED list (the look-2 driver's five + six reviewed 2026-10-07, before this play):
#   b7309869  load: never unpickle the ride-along optimizers on a load; a CPU load that initialises CUDA throws —
#             the optimizer state was never restored by any load; the weights and every forward are untouched.
#   38ed7d0d  --policy-readout {tower,trunk}: every X5 checkpoint migrates to `tower` (pre-v138), which builds
#             nothing new; K9 goldens (blob + fixed_mass) unchanged; compiled R1 / T2 FX graphs byte-identical.
#   fe237eac  K9(b): `stable_order(..., consumed=)` is a declaration the tie recorder reads; it changes NOTHING the
#             sort returns (fixed_mass learner golden post_params_sha256 identical) — a training-gate change only.
#   9c58c0d1  tests + an optional `python_seed` kwarg on the offline parity tool; no game path.
#   5dcbcddd, 5151f2bf  main.belief_roles (the purpose-metric reader); not imported by h2h.
REVIEWED="131f3412 19bdf034 96eedfb5 54f3bb64 e1e81456 b7309869 38ed7d0d fe237eac 9c58c0d1 5dcbcddd 5151f2bf"
ALL=$(git log --format=%h --abbrev=8 "$FAMILY_COMMIT"..HEAD -- src data 2>&1)
DRIFT=$(for c in $ALL; do case " $REVIEWED " in *" $c "*) ;; *) git log -1 --oneline "$c";; esac; done)
echo "=== engine-path commits since $FAMILY_COMMIT: ${ALL:-none} (reviewed: $REVIEWED); unreviewed: ${DRIFT:-none}"
if [ -n "$DRIFT" ]; then echo "REFUSED: an unreviewed engine-path commit (above)"; exit 5; fi

if [ "$DEVICE" = cuda ]; then PRE=("$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32); else PRE=("$OPS/mem_cap.sh" 32); fi

register() {  # family rule commit
  "$PY" -m main.eval_ledger family-register "${RROOT[@]}" --id "$1" --decision-kind ab_verdict --rule "$2" \
    --protocol gen3_eval_protocol_v1_h2h --commit "$3" || exit 3
}

req() {  # read → this run's request
  case $EMIT_LOOK in 3) echo "x5ab_look3_$1" ;; *) echo "x5ab_look${EMIT_LOOK}_$1" ;; esac
}

play() {  # read family
  local read=$1 fam=$2 rq cells n split part
  rq=$(req "$read"); cells="$PLAN_DIR/${read}_look$EMIT_LOOK.json"; split="$PLAN_DIR/${read}_look${EMIT_LOOK}_engines"
  # shellcheck disable=SC2086
  n=$("$PY" "$HERE/plan_look3.py" --read "$read" --out "$cells" "${RROOT[@]}" --models "$MODELS" \
      --seeds $SEEDS --look-seeds $LOOK_SEEDS --emit-look "$EMIT_LOOK" --split-dir "$split") \
      || { echo "PLAN REFUSED $read ($(date -Is))"; exit 3; }
  echo "=== $(date -Is) $rq: $n cell(s) planned ($cells)"
  [ "$n" = 0 ] && return 0
  # ONE ENGINE PER (row, column) ARCHITECTURE KEY PAIR (plan_look3.arch_key): the 706fa536-trained seeds record two
  # defaulted kwargs the older ones lack, so the engine's forward fingerprint refuses to co-serve them (2026-10-07).
  for part in "$split"/engine_*.json; do
    echo "=== $(date -Is) $rq: engine $(basename "$part") ($("$PY" -c 'import json,sys;print(len(json.load(open(sys.argv[1]))))' "$part") cell(s))"
    "${PRE[@]}" "$PY" -m main.h2h play-many --cells "$part" --pairs "$PAIRS" \
      --batch-pairs "$BATCH" --device "$DEVICE" --purpose ab --family "$fam" --request "$rq" \
      --label "$LABEL" --oracle-reveal-mode off "${OUT[@]}" || { echo "FAILED $rq $part ($(date -Is))"; exit 4; }
  done
}

for s in $STEPS; do
  case $s in
    a) register x5ab_strength_steps "$RULE" "$FAMILY_COMMIT"; play steps x5ab_strength_steps ;;
    b) register x5ab_strength_wall "$RULE" "$FAMILY_COMMIT";  play wall x5ab_strength_wall ;;
    c) [ "$EMIT_LOOK" = 3 ] || continue
       register x5ab_sensitivity_wall13 "$SENS_RULE" "$PIN"; play wall13 x5ab_sensitivity_wall13 ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac
done
echo "=== $(date -Is) DONE"
