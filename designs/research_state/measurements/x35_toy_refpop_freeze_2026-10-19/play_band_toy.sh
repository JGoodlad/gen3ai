#!/usr/bin/env bash
# X35's TOY-SIDE R-FREEZE BAND CELL (design_compute_efficiency.md §5.6.4): T* (X34 @ 290,000,087, the plateau call)
# against the toy side's one X candidate, P* (X26 @ 280,000,170), 500 mirrored pairs (1,000 games), greedy vs greedy,
# oracle reveal off, on CPU, INCREMENTAL and RESUMABLE: every 50-pair batch is one durable row in the SCRATCH eval
# ledger the production band cell used (never under models/); a re-run re-plans and the ledger's resume skips every
# recorded batch. T* is the PLAYER (refpop's `_band_from_ledger` matches player == T*'s sha256, opponent == the X
# candidate's); the production band cell had P* as the player, so this cell is its own request.
#
# Both checkpoints are played from read-only copies laid out as <cache>/<tstar|pstar>/<run>/{checkpoints/<zip>,
# model_config.json} because h2h needs a sibling arch record; the script REFUSES (exit 5) if either copy's sha256 is
# not the call's. The two are different ARCHITECTURES (toy vs production); the h2h engine serves up to two slot groups.
# Rule 5b: run under `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 16`, ONE lane (--threads 1,
# --torch-threads 1), CUDA_VISIBLE_DEVICES="". Writes $STATE/DONE_TOY (or $STATE/FAILED_TOY) at the end.
#
# Env: CODE (the checkout whose src/ plays; its HEAD is logged), ROOT (ledger root), STATE (marker dir).
set -u
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
CODE=${CODE:?}
ROOT=${ROOT:-$HOME/.cache/gen3ai/x35_refpop/ledger}
STATE=${STATE:-$HOME/.cache/gen3ai/x35_refpop}
TSTAR=$STATE/tstar/rb_toy_ent_s4002_deep/checkpoints/checkpoint_290000087_steps.zip
TSTAR_SHA=cfa7f7bee7b7a43959f496ce6d5a34556bae914de9c80f90f9342a05a22965e7
PSTAR=$STATE/pstar/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip
PSTAR_SHA=814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=$CODE/src
mkdir -p "$ROOT" "$STATE"; rm -f "$STATE/DONE_TOY" "$STATE/FAILED_TOY"
cd "$CODE" || exit 5
echo "=== $(date -Is) toy band cell; code $CODE at $(git rev-parse HEAD)"
for pair in "$TSTAR:$TSTAR_SHA" "$PSTAR:$PSTAR_SHA"; do
  f=${pair%:*}; want=${pair#*:}; got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$got" = "$want" ] || { echo "REFUSED: $f sha $got != $want"; touch "$STATE/FAILED_TOY"; exit 5; }
done
FAMILY=x35_band_toy
RULE="design_compute_efficiency.md §5.6.4 (X35 toy R freeze, X-block band [0.20, 0.80], rule-8 eps 1e-9, 500 mirrored pairs, greedy vs greedy); T* = rb_toy_ent_s4002_deep @ 290,000,087 (sha cfa7f7be) vs P* = rb_x26_s2601 @ 280,000,170 (sha 814823c5)"
"$PY" -m main.eval_ledger family-register --root "$ROOT" --id "$FAMILY" --decision-kind ab_verdict --rule "$RULE" \
  --protocol gen3_eval_protocol_v1_h2h --commit "$(git rev-parse HEAD)" || { touch "$STATE/FAILED_TOY"; exit 3; }
"$PY" -m main.h2h play-many --players "$TSTAR" --opponents "$PSTAR" --pairs 500 --batch-pairs 50 \
  --purpose ab --family "$FAMILY" --request x35_band_toy --label x35_refpop_band_toy --device cpu \
  --torch-threads 1 --threads 1 --oracle-reveal-mode off --out "$ROOT" \
  || { echo "FAILED ($(date -Is))"; touch "$STATE/FAILED_TOY"; exit 4; }
echo "=== $(date -Is) DONE"; touch "$STATE/DONE_TOY"
