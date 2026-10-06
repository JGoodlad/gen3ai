#!/usr/bin/env bash
# X5 A/B LOOK 2: the NEW cells of the registered head-to-head cross (design_x5_belief_tokens.md §7.4, §7.8-§7.9),
# INCREMENTAL and RESUMABLE, exactly as look 1's driver (`../x5ab_look1_2026-10-06/play_look1.sh`) plays a cell:
# 1,000 mirrored pairs per cell in batches of 500, both sides greedy, schedule seed 0, purpose `ab`.
#
#   (a) x5ab_look2_steps   fixed_mass 15M x blob 15M   16 new cells   family x5ab_strength_steps
#   (b) x5ab_look2_wall    fixed_mass 12M x blob 15M   16 new cells   family x5ab_strength_wall
#
# Same FAMILIES as look 1 (each registered once, at bcb0296c, with all three looks' boundaries in its rule; a family is
# the eval ledger's group-sequential read across its looks), a NEW REQUEST per look. Look 1's 9 + 9 cells are REUSED
# from the ledger, never replayed: `plan_look2.py` resolves the checkpoints NOW (refusing a missing / short run — e.g.
# rb_x5ab_fm_s1005 before it finishes), checks the ledger's look-1 request holds EXACTLY the registered 3 x 3 cells,
# and emits the 5 x 5 minus 3 x 3 = 16 new cells; it refuses a planned cell that is already a look-1 cell (by sha256)
# and a look-2 request holding any cell outside the plan. A re-run re-plans (same 16) and the ledger's resume skips
# every recorded batch, so this script is safe to kill and to start again. One engine per read (one `play-many`).
#
# Env: ROOT (ledger root; empty = the run archive's models/_ledger, the registered home), MODELS (the archive's
# models/), PAIRS (1000), BATCH (500), DEVICE (cuda), STEPS (a b), LABEL (x5ab_look2), SEEDS / LOOK1_SEEDS (the
# registered 1001..1005 / 1001..1003), PLAN_DIR (where the cell lists go), EMIT (look2; `look1` = DRY RUN only, into a
# scratch ROOT + scratch MODELS), ALLOW_ENGINE_DRIFT (1 = play although the engine path changed since bcb0296c).
# Run from the repo root with PYTHONPATH=<tree>/src (and the GPU lease token for cuda), DETACHED.
set -u
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
MODELS=${MODELS:-/home/goodlad/dev/gen3ai/models}
PAIRS=${PAIRS:-1000}
BATCH=${BATCH:-500}
DEVICE=${DEVICE:-cuda}
STEPS=${STEPS:-a b}
LABEL=${LABEL:-x5ab_look2}
SEEDS=${SEEDS:-1001 1002 1003 1004 1005}
LOOK1_SEEDS=${LOOK1_SEEDS:-1001 1002 1003}
EMIT=${EMIT:-look2}
ROOT=${ROOT:-}
PLAN_DIR=${PLAN_DIR:-$HOME/.cache/gen3ai/x5look2/plan}
mkdir -p "$PLAN_DIR"
OUT=(); [ -n "$ROOT" ] && OUT=(--out "$ROOT")
RROOT=(); [ -n "$ROOT" ] && RROOT=(--root "$ROOT")
# the families' REGISTERED terms (family-register is idempotent only on identical terms; never HEAD's commit)
FAMILY_COMMIT=bcb0296c0edf9d9f6602f23bcfbc6b14e5df8ef7
RULE="design_x5_belief_tokens.md §7.4 (+§7.6-§7.9 Amendments 2-5): cross t=(Δ̂+3.5)/√V̂, df=2(n-1), looks 5.761/2.683/1.874; estimator main.h2h.cross"

# Look 1 was played at bcb0296c. Refuse to play look 2 on an engine path that changed since, unless consented.
# The scope is ALL of src/ and data/ (any import the engine reaches). REVIEWED (2026-10-06, before the GPU play):
#   19bdf034  snapshot.py save_model_snapshot keeps init_num_threads; single_thread_build yields an int — SAVE-time /
#             fresh-build only; h2h never saves or builds a model.
#   131f3412  play.py: plan_edge's spec now comes from edge_spec(); without a consumer (many.py passes none) it
#             serialises byte-identically to the recorded look-1 spec {"batch_pairs":500,"producer":"h2h",
#             "schedule_seed":0} (checked with schema.canonical); no game-path function changed; eval_ledger gains
#             read_decisions / append_decision(unique=) / the plateau verify rule — additive, not on the row path.
#   96eedfb5, 54f3bb64, e1e81456  ops scripts / gpu_lease acquire / their tests — not imported by h2h.
REVIEWED="131f3412 19bdf034 96eedfb5 54f3bb64 e1e81456"
ALL=$(git log --format=%h --abbrev=8 "$FAMILY_COMMIT"..HEAD -- src data 2>&1)
DRIFT=$(for c in $ALL; do case " $REVIEWED " in *" $c "*) ;; *) git log -1 --oneline "$c";; esac; done)
echo "=== $(date -Is) look 2 at $(git rev-parse HEAD); engine-path commits since $FAMILY_COMMIT: ${ALL:-none}" \
     "(reviewed: $REVIEWED); unreviewed: ${DRIFT:-none}"
if [ -n "$DRIFT" ] && [ "${ALLOW_ENGINE_DRIFT:-0}" != 1 ]; then
  echo "REFUSED: the engine path changed since look 1 (above); review, then ALLOW_ENGINE_DRIFT=1 to play"; exit 5
fi

if [ "$DEVICE" = cuda ]; then PRE=(scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24); else PRE=(scripts/ops/mem_cap.sh 24); fi

register() {  # family
  "$PY" -m main.eval_ledger family-register "${RROOT[@]}" --id "$1" --decision-kind ab_verdict --rule "$RULE" \
    --protocol gen3_eval_protocol_v1_h2h --commit "$FAMILY_COMMIT" || exit 3
}

play() {  # read family request
  local read=$1 fam=$2 req=$3 cells="$PLAN_DIR/$1_$EMIT.json" n
  # shellcheck disable=SC2086
  n=$("$PY" "$HERE/plan_look2.py" --read "$read" --out "$cells" "${RROOT[@]}" --models "$MODELS" \
      --seeds $SEEDS --look1-seeds $LOOK1_SEEDS --emit "$EMIT") || { echo "PLAN REFUSED $read ($(date -Is))"; exit 3; }
  echo "=== $(date -Is) $req: $n cell(s) planned ($cells)"
  "${PRE[@]}" "$PY" -m main.h2h play-many --cells "$cells" --pairs "$PAIRS" \
    --batch-pairs "$BATCH" --device "$DEVICE" --purpose ab --family "$fam" --request "$req" \
    --label "$LABEL" --oracle-reveal-mode off "${OUT[@]}" || { echo "FAILED $req ($(date -Is))"; exit 4; }
}

req() { [ "$EMIT" = look1 ] && echo "x5ab_look1_$1" || echo "x5ab_look2_$1"; }

for s in $STEPS; do
  case $s in
    a) register x5ab_strength_steps; play steps x5ab_strength_steps "$(req steps)" ;;
    b) register x5ab_strength_wall;  play wall  x5ab_strength_wall  "$(req wall)" ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac
done
echo "=== $(date -Is) DONE"
