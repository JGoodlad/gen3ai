#!/usr/bin/env bash
# W Phase 1 shards of the SUPPLEMENT in ONE capped job (the pilot's shards.sh pattern), run through run_pin.sh at the
# UNMODIFIED pin:
#   run_pin.sh <end2|prod> <threads per shard> <GB> <timeout> phase1_supp.sh <W> [phase1_supp.py args...]
# Each shard is resumable (its own fsync'd JSONL under the archive's <cp>/rows_supp/); logs go to <cp>/logs/.
# Keep W x threads within the CPU set run_pin.sh pins to ($HIB_CPUS, default 6 hardware threads).
set -uo pipefail
W="$1"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$HIB_T1_PIN_KIND" in pin) ;; *) echo "phase1_supp runs at the plain pin only" >&2; exit 2 ;; esac
LOGS="${HIB_T1_ARCHIVE:-$HOME/gen3ai_archive/hidden_info_tier1}/$VDP_CP/logs"
mkdir -p "$LOGS"
pids=()
for ((i = 0; i < W; i++)); do
  "$VDP_PY" "$HERE/phase1_supp.py" --shard "$i" --of "$W" "$@" >> "$LOGS/phase1_supp_shard_${i}_of_${W}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
exit "$rc"
