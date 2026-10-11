#!/usr/bin/env bash
# PHASE 2 of one checkpoint in ONE capped job: the worlds, then the determinized playouts (both resumable).
#   run_at_pin.sh <cp> 1 <GB> <timeout> phase2.sh <W> [--belief]     (needs Phase 1's rows: other_open per root)
# --belief = Phase 2b (Amendment 5): species from the network's own X5 belief, into the archive's <cp>/det_b/.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$1"; shift
WARGS=(); DARGS=()
if [ "${1:-}" = "--belief" ]; then WARGS=(--sampler belief --det-dir det_b); DARGS=(--det-dir det_b); fi
bash "$HERE/shards.sh" "$W" --worlds "${WARGS[@]}" || exit $?
bash "$HERE/shards.sh" "$W" --det "${DARGS[@]}"
