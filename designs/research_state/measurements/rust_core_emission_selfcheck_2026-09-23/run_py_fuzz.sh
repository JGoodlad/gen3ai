#!/usr/bin/env bash
# The Python fuzz scripts that drive the rust bridge / search driver, run DIRECTLY as scripts —
# so `sim_bridge_bin._auto_selfcheck` puts every child on the EMISSION SELF-CHECK build
# (`target/selfcheck/`). One log per script under $OUT; a script's exit status is its verdict.
set -u
WT=/home/goodlad/dev/gen3ai-wt/emission-selfcheck
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
OUT=${OUT:-/tmp/esc_pyfuzz}
mkdir -p "$OUT"
cd "$WT"
export PYTHONPATH=${PYTHONPATH:-}:src
run() {
  local name=$1; shift
  local t0=$(date +%s)
  nice -n 10 "$PY" "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc wall_s=$(( $(date +%s) - t0 )) selfcheck_failures=$(grep -c 'EMISSION SELF-CHECK FAILED' "$OUT/$name.log")"
}
run action_fuzz                 src/agents/action/fuzz_test.py 60
run obs_roundtrip_fuzz          src/agents/training/obs_roundtrip_fuzz_test.py
run event_log_fuzz              src/agents/battle/event_log_fuzz_test.py 80
run live_view_memo_fuzz         src/agents/battle/live_view_memo_fuzz_test.py 40
run one_sided_view_parity_fuzz  src/agents/battle/one_sided_view_parity_fuzz_test.py 16
run event_fold_parity_fuzz      src/agents/battle/event_fold_parity_fuzz_test.py 8
run bridge_session_fuzz         src/utils/bridge/bridge_session_fuzz_test.py
run search_clone_parity_fuzz    src/utils/bridge/search_clone_parity_fuzz_test.py 6 --impl rust --record-impl rust
run counterfactual_fuzz         src/utils/bridge/counterfactual_fuzz_test.py 4 --impl rust --record-impl rust
run turn_delta_fold_fuzz        src/agents/training/turn_delta_fold_equivalence_fuzz_test.py --minutes 5
