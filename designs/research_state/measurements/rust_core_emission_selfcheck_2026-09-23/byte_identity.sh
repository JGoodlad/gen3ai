#!/usr/bin/env bash
# Byte identity of sim_bridge's stdout on the recorded 41-battle training transcript, for the three
# builds of run_pairs.py; also each binary's sha256, the baked CARGO_MANIFEST_DIR (the dex path)
# and whether the self-check's panic MARKER / symbols are in it at all.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$HERE/../m1_transport_throughput_2026-09-23/transcript_env_seed0.txt"
WT=/home/goodlad/dev/gen3ai-wt
for arm in "A $WT/esc-base/src/rust_sim/target/release/sim_bridge" \
           "B $WT/emission-selfcheck/src/rust_sim/target/release/sim_bridge" \
           "C $WT/emission-selfcheck/src/rust_sim/target/selfcheck/sim_bridge"; do
  set -- $arm
  out=$(mktemp); err=$(mktemp)
  "$2" < "$T" > "$out" 2> "$err"
  echo "$1 bin_sha256=$(sha256sum < "$2" | cut -c1-16) baked=$(strings "$2" | grep -o '/home/goodlad/dev/gen3ai-wt/[a-z-]*/src/rust_sim' | sort -u)" \
       "marker_strings=$(strings "$2" | grep -c 'EMISSION SELF-CHECK' || true)" \
       "emission_check_symbols=$(nm -C "$2" | grep -c 'emission_check' || true)"
  echo "   stdout_sha256=$(sha256sum < "$out" | cut -c1-16) bytes=$(wc -c < "$out") __END__=$(grep -c __END__ "$out") __ERR__=$(grep -c __ERR__ "$out" || true) stderr_bytes=$(wc -c < "$err")"
  rm -f "$out" "$err"
done
