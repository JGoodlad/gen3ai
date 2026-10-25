#!/usr/bin/env bash
# gpu_census.sh — the CUDA CENSUS the per-row dither-null margin needs before it can bind (WRITTEN, NOT RUN; 2026-10-25;
# README here). Per checkpoint of the X26 lineage (X26 100M-380M, the X26 continuation 390M-482M, and the archived u1000 neighbours 478M / 479M), on the card: the fp32
# reference per fixture row, its 32-seed half-bf16-ulp dither null per row and per seed, eager bf16 per row, and the
# PRODUCTION compiled arm per fixture row at the canary's B = 2048 (`census_read.py` -> the x26c `margin_read.py`).
# Then the null's cost on the card (`null_cost.py`, K = 16). It answers the design's open questions 1-3: the healthy
# CUDA compiled / null ratio (k), the null's K-sensitivity (K), the floor, and the null's GPU seconds.
#
#   GEN3AI_GPU_LEASE_TOKEN=<token> gpu_census.sh <checkout> <out dir>
#
# <checkout>: a clean checkout of the code to calibrate (main at or after this file's commit). Under the caller's GPU
# LEASE only (root rule 5): gpu_lock.sh refuses everyone else (exit 5 / 6), mem_cap.sh caps memory, a hard wall timeout
# bounds each checkpoint. Writes nothing under models/; resumable (a finished checkpoint is skipped). Est. 45-75 min
# (20 checkpoints; the first compile ~5 min, later ones from the Inductor cache). Then, on CPU:
#   python <checkout>/designs/research_state/measurements/canary_row_null_design_2026-10-25/derive_k.py <out dir>
set -u
CO="${1:?checkout (git worktree of the code to calibrate)}"; OUT="${2:?out dir}"
if [ -z "${GEN3AI_GPU_LEASE_TOKEN:-}" ] && [ -z "${GEN3AI_GPU_LEASE_TOKEN_FILE:-}" ]; then
    echo "REFUSING: no GEN3AI_GPU_LEASE_TOKEN / _FILE — the GPU is LEASED; run this as the lease holder" >&2; exit 2
fi
[ -z "$(git -C "$CO" status --porcelain 2>/dev/null)" ] || { echo "REFUSING: $CO is not a clean checkout" >&2; exit 2; }
ROOT=/home/goodlad/dev/gen3ai
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
HERE="$CO/designs/research_state/measurements/canary_row_null_design_2026-10-25"
[ -f "$HERE/census_read.py" ] || { echo "REFUSING: $CO has no census_read.py (older than this script)" >&2; exit 2; }
mkdir -p "$OUT"
git -C "$CO" rev-parse HEAD > "$OUT/checkout_head.txt"
cd "$CO" && export PYTHONPATH="$CO/src" GEN3AI_TEST_ALLOW_GPU=1
run() {   # <run dir> <at>
    local lab; lab="$(basename "$1")_$2"
    if [ -f "$OUT/read_${lab}.json" ]; then echo "[census] $lab: done already"; return 0; fi
    "$ROOT/scripts/ops/gpu_lock.sh" "$ROOT/scripts/ops/mem_cap.sh" --name "rownull-$lab" 24 timeout 1200 \
        "$PY" "$HERE/census_read.py" --run-dir "$1" --at "$2" --out "$OUT" --device cuda --dither 32 --vf-coef 0.5 \
        >> "$OUT/census.log" 2>&1
    echo "[census] $lab: exit $?"
}
for at in 100M 180M 230M 280M 330M 380M; do run "$ROOT/models/rb_x26_s2601" "$at"; done
for at in 390M 400M 410M 420M 430M 440M 450M 460M 470M 480M 481M 482M; do run "$ROOT/models/rb_x26c_s2601" "$at"; done
# the u1000 neighbours (478.0M / 479.0M), pruned from the run dir but archived: a scratch run dir under $OUT holding COPIES
# (a symlink resolves to the archive dir, which has no metadata.json) and the live run's config + metadata (read only)
ARC=$HOME/gen3ai_archive/x26c_canary_u1000_2026-10-25
U=$OUT/x26c_u1000_run
if [ -f "$ARC/checkpoint_478000208_steps.zip" ]; then
    mkdir -p "$U/checkpoints"
    cp -n "$ROOT/models/rb_x26c_s2601/model_config.json" "$ROOT/models/rb_x26c_s2601/metadata.json" "$U/"
    cp -n "$ARC"/checkpoint_47[89]*_steps.zip "$U/checkpoints/"
    for at in 478M 479M; do run "$U" "$at"; done
fi
LAST=$(ls "$ROOT"/models/rb_x26c_s2601/checkpoints/checkpoint_*_steps.zip | sort | tail -1)
[ -f "$OUT/cost_cuda.json" ] || "$ROOT/scripts/ops/gpu_lock.sh" "$ROOT/scripts/ops/mem_cap.sh" --name rownull-cost 24 \
    timeout 900 "$PY" "$HERE/null_cost.py" --ckpt "$LAST" --out "$OUT/cost_cuda.json" --device cuda --k 16 --vf-coef 0.5 \
    >> "$OUT/census.log" 2>&1
echo "[census] cost: exit $? -> $OUT (log $OUT/census.log); next: derive_k.py $OUT"
