#!/usr/bin/env bash
# PHASE 2 of one checkpoint in ONE capped job: the worlds, then the determinized playouts (both resumable).
#   run_at_pin.sh <cp> 1 <GB> <timeout> phase2.sh <W>        (needs Phase 1's rows: other_open per root)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$HERE/shards.sh" "$1" --worlds || exit $?
bash "$HERE/shards.sh" "$1" --det
