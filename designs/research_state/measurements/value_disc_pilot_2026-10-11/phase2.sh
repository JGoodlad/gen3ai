#!/usr/bin/env bash
# PHASE 2 of one checkpoint in ONE capped job: the worlds, then the determinized playouts (both resumable).
#   run_at_pin.sh <cp> 1 <GB> <timeout> phase2.sh <W> [--belief | --moves] [--worlds-only | --playouts-only]
#   (needs Phase 1's rows: other_open per root)
# --belief = Phase 2b (Amendment 5): species from the network's own X5 belief, into the archive's <cp>/det_b/.
# --moves  = Phase 2c (Amendment 6): that AND the hidden moves from the network's own per-mon move belief, 8 worlds
#            built per state (the 8 played) with up to 60 tries each, into <cp>/det_c/.
# --truemoves = Phase 2c-T (Amendment 6, the CEILING arm): as --moves, but every SEEN opponent mon keeps its TRUE
#            set (a perfect move belief for the revealed mons), into <cp>/det_ct/.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$1"; shift
WARGS=(); DARGS=()
if [ "${1:-}" = "--belief" ]; then WARGS=(--sampler belief --det-dir det_b); DARGS=(--det-dir det_b); shift; fi
if [ "${1:-}" = "--moves" ]; then WARGS=(--sampler moves --det-dir det_c --J 8 --max-tries 60); DARGS=(--det-dir det_c); shift; fi
if [ "${1:-}" = "--truemoves" ]; then WARGS=(--sampler truemoves --det-dir det_ct --J 8 --max-tries 60); DARGS=(--det-dir det_ct); shift; fi
STEP="${1:-}"
if [ "$STEP" != "--playouts-only" ]; then bash "$HERE/shards.sh" "$W" --worlds "${WARGS[@]}" || exit $?; fi
if [ "$STEP" != "--worlds-only" ]; then bash "$HERE/shards.sh" "$W" --det "${DARGS[@]}"; fi
