#!/usr/bin/env bash
# X35's R-FREEZE BAND CELL (design_compute_efficiency.md §5.6.4): P* (X26 @ 280,000,170, the TIER1_PLATEAU call of
# PREREGISTRATION Amendment 5) against each X-block candidate, 500 mirrored pairs (1,000 games), greedy vs greedy,
# oracle reveal off, on CPU, INCREMENTAL and RESUMABLE: every 50-pair batch is one durable row in a SCRATCH eval
# ledger (never under models/); a re-run re-plans and the ledger's resume skips every recorded batch.
#
# P* is played from a read-only copy of the verified ~/.cache/gen3ai/x26_plateau/ file, laid out as
# x35_refpop/pstar/rb_x26_s2601/{checkpoints/<zip>, model_config.json (copied from X26)} because h2h needs a sibling
# arch record; the script REFUSES (exit 5) if the
# copy's sha256 is not the call's. Rule 5b: run under `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh`, ONE
# lane (--threads 1, --torch-threads 1), CUDA_VISIBLE_DEVICES="". Writes $STATE/DONE (or $STATE/FAILED) at the end.
#
# Env: CODE (the checkout whose src/ plays; its HEAD is logged), ROOT (ledger root), STATE (marker dir).
set -u
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
CODE=${CODE:?}
ROOT=${ROOT:-$HOME/.cache/gen3ai/x35_refpop/ledger}
STATE=${STATE:-$HOME/.cache/gen3ai/x35_refpop}
PSTAR=$HOME/.cache/gen3ai/x35_refpop/pstar/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip
PSTAR_SHA=814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed
OPPS=("$@")
[ ${#OPPS[@]} -ge 1 ] || { echo "usage: play_band.sh <opponent.zip> ..."; exit 2; }
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=$CODE/src
mkdir -p "$ROOT" "$STATE"; rm -f "$STATE/DONE" "$STATE/FAILED"
cd "$CODE" || exit 5
echo "=== $(date -Is) band cells; code $CODE at $(git rev-parse HEAD)"
got=$(sha256sum "$PSTAR" | cut -d' ' -f1)
[ "$got" = "$PSTAR_SHA" ] || { echo "REFUSED: P* copy sha $got != $PSTAR_SHA"; touch "$STATE/FAILED"; exit 5; }
FAMILY=x35_band
RULE="design_compute_efficiency.md §5.6.4 (X35 R freeze, X-block band [0.20, 0.80], rule-8 eps 1e-9, 500 mirrored pairs, greedy vs greedy); P* = rb_x26_s2601 @ 280,000,170 (sha 814823c5)"
"$PY" -m main.eval_ledger family-register --root "$ROOT" --id "$FAMILY" --decision-kind ab_verdict --rule "$RULE" \
  --protocol gen3_eval_protocol_v1_h2h --commit "$(git rev-parse HEAD)" || { touch "$STATE/FAILED"; exit 3; }
"$PY" -m main.h2h play-many --players "$PSTAR" --opponents "${OPPS[@]}" --pairs 500 --batch-pairs 50 \
  --purpose ab --family "$FAMILY" --request x35_band_production --label x35_refpop_band --device cpu \
  --torch-threads 1 --threads 1 --oracle-reveal-mode off --out "$ROOT" \
  || { echo "FAILED ($(date -Is))"; touch "$STATE/FAILED"; exit 4; }
echo "=== $(date -Is) DONE"; touch "$STATE/DONE"
