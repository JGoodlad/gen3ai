#!/usr/bin/env bash
# X5 A/B LOOK 1: the registered head-to-head cross (design_x5_belief_tokens.md §7.4, §7.6–§7.9), INCREMENTAL and
# RESUMABLE. Every cell is a `main.h2h play-many` cell; every batch (500 mirrored pairs) is one durable, fsynced row
# in the eval COUNT ledger under its claim, for its look's request in its registered family. A re-run skips every
# recorded batch of every cell (the ledger's resume), so this script is safe to kill and to start again.
#
#   (a) x5ab_look1_steps          fixed_mass 15M × blob 15M          off           family x5ab_strength_steps
#   (b) x5ab_look1_wall           fixed_mass 12M × blob 15M          off           family x5ab_strength_wall
#   (c) x5ab_look1_oracle_one     oracle {sp, full} × {blob, fm} 15M one_sided     family x5ab_oracle_one_sided
#   (d) x5ab_look1_oracle_both    oracle {sp, full} × {blob, fm} 15M both_sided    family x5ab_oracle_both_sided
#
# One engine per (request, oracle level, opponent arm): an oracle level is its own architecture group to the
# pre-flight (`oracle_reveal` is an architecture toggle), so an engine holds one oracle level and one opponent arm.
#
# Env: ROOT (ledger root; empty = the run archive's models/_ledger, the registered home), PAIRS (1000),
# BATCH (500), DEVICE (cuda), STEPS (which of a b c d, default all), LABEL (x5ab_look1).
# Run from the repo root with PYTHONPATH=<tree>/src (and the GPU lease token for cuda), DETACHED.
set -u
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
M=/home/goodlad/dev/gen3ai/models
PAIRS=${PAIRS:-1000}
BATCH=${BATCH:-500}
DEVICE=${DEVICE:-cuda}
STEPS=${STEPS:-a b c d}
LABEL=${LABEL:-x5ab_look1}
ROOT=${ROOT:-}
OUT=(); [ -n "$ROOT" ] && OUT=(--out "$ROOT")
RROOT=(); [ -n "$ROOT" ] && RROOT=(--root "$ROOT")
COMMIT=$(git rev-parse HEAD)
RULE="design_x5_belief_tokens.md §7.4 (+§7.6-§7.9 Amendments 2-5): cross t=(Δ̂+3.5)/√V̂, df=2(n-1), looks 5.761/2.683/1.874; estimator main.h2h.cross"

f() { local s; for s in "$@"; do echo "$M/rb_x5ab_${s}/final_model.zip"; done; }
BLOB=$(f blob_s1001 blob_s1002 blob_s1003)
FM=$(f fm_s1001 fm_s1002 fm_s1003)
FM12="$M/rb_x5ab_fm_s1001/checkpoints/checkpoint_12000021_steps.zip
$M/rb_x5ab_fm_s1002/checkpoints/checkpoint_12000070_steps.zip
$M/rb_x5ab_fm_s1003/checkpoints/checkpoint_12000119_steps.zip"
OSP=$(f oracle_sp_s1001b oracle_sp_s1002 oracle_sp_s1003)
OFU=$(f oracle_full_s1001b oracle_full_s1002 oracle_full_s1003)

if [ "$DEVICE" = cuda ]; then PRE=(scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 24); else PRE=(scripts/ops/mem_cap.sh 24); fi

register() {  # family protocol
  "$PY" -m main.eval_ledger family-register "${RROOT[@]}" --id "$1" --decision-kind ab_verdict --rule "$RULE" \
    --protocol "$2" --commit "$COMMIT" || exit 3
}

play() {  # request family mode players opponents
  local req=$1 fam=$2 mode=$3 players=$4 opps=$5
  echo "=== $(date -Is) $req mode=$mode"
  # shellcheck disable=SC2086
  "${PRE[@]}" "$PY" -m main.h2h play-many --players $players --opponents $opps --pairs "$PAIRS" \
    --batch-pairs "$BATCH" --device "$DEVICE" --purpose ab --family "$fam" --request "$req" \
    --label "$LABEL" --oracle-reveal-mode "$mode" "${OUT[@]}" || { echo "FAILED $req $mode ($(date -Is))"; exit 4; }
}

for s in $STEPS; do
  case $s in
    a) register x5ab_strength_steps gen3_eval_protocol_v1_h2h
       play x5ab_look1_steps x5ab_strength_steps off "$FM" "$BLOB" ;;
    b) register x5ab_strength_wall gen3_eval_protocol_v1_h2h
       play x5ab_look1_wall x5ab_strength_wall off "$FM12" "$BLOB" ;;
    c) register x5ab_oracle_one_sided gen3_eval_protocol_v1_h2h_oracle_one_sided
       for O in "$OSP" "$OFU"; do for P in "$BLOB" "$FM"; do
         play x5ab_look1_oracle_one x5ab_oracle_one_sided one_sided "$O" "$P"; done; done ;;
    d) register x5ab_oracle_both_sided gen3_eval_protocol_v1_h2h_oracle_both_sided
       for O in "$OSP" "$OFU"; do for P in "$BLOB" "$FM"; do
         play x5ab_look1_oracle_both x5ab_oracle_both_sided both_sided "$O" "$P"; done; done ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac
done
echo "=== $(date -Is) DONE"
