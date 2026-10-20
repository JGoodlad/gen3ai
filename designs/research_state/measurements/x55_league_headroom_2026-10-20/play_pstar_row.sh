#!/usr/bin/env bash
# X55 / LG-0 part 2: P*'s SCORE ROW against the archive (design_alphastar_league_path.md §6 row LG-0 (c)).
# P* (X26 @ 280,000,170) vs 18 X26 nodes, 200 mirrored pairs (400 games) each, greedy vs greedy, oracle reveal off,
# main.h2h play-many on CPU into a SCRATCH ledger (never under models/; models/ is only READ). Resumable: every
# 50-pair batch is one durable row; a re-run skips every recorded batch, so a chunk cut by its timeout loses at most
# the batch in flight. Cells are played in PRIORITY order (a partial run keeps the most useful cells):
#   1. A1's early archive nodes (30, 60, 100M)       - no in-house P* row exists for them
#   2. R's A-block below P* (270, 210, 140, 80M)       - S(P*)'s own baseline
#   3. A1's late archive nodes (130, 160, 190, 230, 260M)
#   4. the archive's Nash-support nodes (pool snapshots 268, 244, 194, 254, 224, 246M)
# Rule 5b: run as   scripts/ops/mem_cap.sh --name x55_row 8 timeout 2700 nice -n 19 taskset -c 5-7,13-15 <this>
# (mem_cap takes a heavy slot while a run is live; <= 45 min per chunk). CUDA_VISIBLE_DEVICES="" is forced here.
# Env: CODE (the checkout whose src/ plays), ROOT (ledger root), THREADS (default 3), PAIRS (default 200).
# Arguments (optional): opponent .zip paths to play INSTEAD of the full list, in the order given (a later chunk passes
# the cells an earlier chunk has not reached, so two chunks holding the two heavy slots never play the same cell).
set -u
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
CODE=${CODE:?}
ROOT=${ROOT:-$HOME/.cache/gen3ai/x55_league_headroom/ledger}
THREADS=${THREADS:-3}
PAIRS=${PAIRS:-200}
RUN=/home/goodlad/dev/gen3ai/models/rb_x26_s2601
PSTAR=$RUN/checkpoints/checkpoint_280000170_steps.zip
PSTAR_SHA=814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=$CODE/src
mkdir -p "$ROOT"
cd "$CODE" || exit 5
echo "=== $(date -Is) X55 P* row; code $CODE at $(git rev-parse HEAD); threads $THREADS"
got=$(sha256sum "$PSTAR" | cut -d' ' -f1)
[ "$got" = "$PSTAR_SHA" ] || { echo "REFUSED: P* sha $got != $PSTAR_SHA"; exit 5; }
C=$RUN/checkpoints
S=$RUN/snapshots
OPPS=(
  $C/checkpoint_30000045_steps.zip $C/checkpoint_60000135_steps.zip $C/checkpoint_100000037_steps.zip
  $C/checkpoint_270000128_steps.zip $C/checkpoint_210000149_steps.zip $C/checkpoint_140000062_steps.zip
  $C/checkpoint_80000111_steps.zip
  $C/checkpoint_130000065_steps.zip $C/checkpoint_160000085_steps.zip $C/checkpoint_190000228_steps.zip
  $C/checkpoint_230000074_steps.zip $C/checkpoint_260000235_steps.zip
  $S/snapshot_000268000171.zip $S/snapshot_000244000047.zip $S/snapshot_000194000195.zip
  $S/snapshot_000254000018.zip $S/snapshot_000224000053.zip $S/snapshot_000246000132.zip
)
[ $# -ge 1 ] && OPPS=("$@")
echo "=== cells: ${#OPPS[@]} at $PAIRS pairs"
CELLS=$(mktemp)
"$PY" - "$PSTAR" "${OPPS[@]}" > "$CELLS" <<'EOF'
import json, sys
p, opps = sys.argv[1], sys.argv[2:]
print(json.dumps([[p, o] for o in opps]))
EOF
"$PY" -m main.h2h play-many --cells "$CELLS" --pairs "$PAIRS" --batch-pairs 50 --label x55_league_headroom \
  --device cpu --threads "$THREADS" --torch-threads "$THREADS" --oracle-reveal-mode off --out "$ROOT"
rc=$?
rm -f "$CELLS"
echo "=== $(date -Is) exit $rc"
exit $rc
