#!/usr/bin/env bash
# The four A/B fuzzers on the EMISSION SELF-CHECK build (each builds `--profile selfcheck
# --features emission-selfcheck` itself). Fixed master seeds (reproducible); isolated target dirs so
# no other session's shared /tmp target is rebuilt from this worktree. Exit status = green gate.
set -u
WT=/home/goodlad/dev/gen3ai-wt/emission-selfcheck
H=$WT/src/rust_sim/harness
OUT=${OUT:-/tmp/esc_fuzz}
S=20260923
export POKESIM_BRIDGE_TARGET=/tmp/esc_target_bridge POKESIM_SIMBRIDGE_TARGET=/tmp/esc_target_simbridge
cd "$WT"
run() {
  local name=$1; shift
  local t0=$(date +%s)
  nice -n 10 node "$@" --master-seed $S --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc wall_s=$(( $(date +%s) - t0 )) selfcheck_failures=$(grep -rc 'EMISSION SELF-CHECK FAILED' "$OUT/$name.log" "$OUT/$name" 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')"
  tail -1 "$OUT/$name.log" | cut -c1-300
}
mkdir -p "$OUT"
# (1) ab_fuzz --protocol --mode pool --battles 720 ran separately (ab_pool_protocol.log).
run ab_ourandom_protocol  $H/ab_fuzz.js --mode ourandom --protocol --format gen3ou --battles 400
run ab_randbats_state     $H/ab_fuzz.js --mode randbats --battles 300
run bridge_ab_pool        $H/bridge_ab_fuzz.js --mode pool --format gen3ou --battles 400
run bridge_ab_trapping    $H/bridge_ab_fuzz.js --mode trapping --battles 200
run sim_bridge_diff_pool  $H/gen_sim_bridge_diff.js --mode pool --format gen3ou --battles 300
run sim_bridge_diff_randbats $H/gen_sim_bridge_diff.js --mode randbats --battles 150
