#!/usr/bin/env bash
# Run a PLAY script of this folder at the PIN checkout (X26's pin 9c033489 + the damage-roll control fc010d7d3, stage
# 1's play recipe; `play_s1.check_pin` verifies it): `main` from the pin's src/, the measurement scripts from THIS
# folder's checkout. Recreate the checkout: `git worktree add --detach <dir> 9c033489 && git -C <dir> cherry-pick
# fc010d7d3` (take the pin's side of the src/rust_sim/CLAUDE.md conflict), then build its env core
# (`src/rust_env`: cargo build --release). $S1C_PIN overrides the directory.
#   run_play.sh <script> [args...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="${S1C_PIN:-$HOME/.cache/gen3ai/search_1c/pin_rolls}"
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
SCRIPT="$1"; shift
HEAD_SHA="$(git -C "$DIR" rev-parse HEAD)"
export PYTHONPATH="$DIR/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VDP_PIN="$DIR" VDP_PIN_SHA="$HEAD_SHA" VDP_THREADS=1 VDP_PY="$PY"
export WS1_KIND=pin WS1_CHECKOUT_HEAD="$HEAD_SHA" WS1_CHECKOUT_DIR="$DIR"
cd "$DIR"
exec "$PY" "$HERE/$SCRIPT" "$@"
