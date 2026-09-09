#!/usr/bin/env bash
# Matched-quota re-read of the cf-label arm's conditioning rows. CPU only, niced, READ-ONLY on
# models/ — the subsampled views are symlink trees under $TMP, never a write into the archive.
#
#   bash run.sh [TMP_DIR]
#
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
TMP="${1:-/tmp/cf_matched}"
PY="${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3}"
export PYTHONPATH="${PYTHONPATH:-}:$REPO/src"
mkdir -p "$TMP/out"

# 30 subsample seeds on the two matched rungs, 20 on each frame-size rung, 2000 bootstrap draws
# per block, block seed 0 — the registered read's own seed, so the FULL rung reproduces
# `../critic_read.json` exactly (it does: that is this script's own control).
nice -n 15 "$PY" "$HERE/subsample.py" --out "$TMP/out" --tmp "$TMP/views" \
    --seeds 30 --curve-seeds 20 --boot 2000
nice -n 15 "$PY" "$HERE/analyze.py" --json "$TMP/out/matched_quota.json" > "$HERE/tables.md"

# The identity row is a weighted MEAN, not a fit — this is the stability check, not a re-read.
nice -n 15 "$PY" "$HERE/identity_matched.py" \
    --payload "$HERE/../identity_payload.json" \
    --out "$TMP/out/identity_matched.json" --tmp "$TMP/idviews" \
    --seeds 20 --boot 2000 > "$HERE/identity_tables.md"
