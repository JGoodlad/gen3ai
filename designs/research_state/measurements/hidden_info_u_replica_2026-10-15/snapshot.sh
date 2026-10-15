#!/usr/bin/env bash
# Copy everything the runner reads from the repo into <archive>/code_<sha>/, so the detached workers never read a
# worktree that may be removed: the replica's package + engine probes, this folder's package, the registered
# states, and the CURRENT learnset table (sha256-checked by hiu.fix). Prints the code dir.
#   snapshot.sh <sha> [archive]
set -eu
sha="$1"; arc="${2:-$HOME/gen3ai_archive/hidden_info_u_replica}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
dst="$arc/code_$sha"
mkdir -p "$dst/hidden_info_replica_2026-10-11" "$dst/hidden_info_u_replica_2026-10-15" "$dst/value_disc_pilot_2026-10-11"
cp -r "$M/hidden_info_replica_2026-10-11/replica" "$dst/hidden_info_replica_2026-10-11/"
cp "$M"/hidden_info_replica_2026-10-11/engine_probe_*.jsonl "$dst/hidden_info_replica_2026-10-11/"
cp -r "$HERE/hiu" "$HERE/run_worker.sh" "$dst/hidden_info_u_replica_2026-10-15/"
cp "$M"/value_disc_pilot_2026-10-11/states_end2.json "$M"/value_disc_pilot_2026-10-11/states_prod.json \
   "$dst/value_disc_pilot_2026-10-11/"
cp "$REPO/data/pokemon/gen3_learnset.json" "$dst/gen3_learnset.json"
find "$dst" -name __pycache__ -prune -exec rm -rf {} +
echo "$dst"
