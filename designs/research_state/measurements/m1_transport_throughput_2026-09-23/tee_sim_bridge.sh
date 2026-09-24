#!/usr/bin/env bash
# Capture shim: set as $POKESIM_SIM_BRIDGE_BIN for ONE env_step_rust.py run to record the exact
# stdin stream each spawned sim_bridge child receives (one file per child, in spawn order), then
# exec the real binary named by $M1_REAL_SIM_BRIDGE (exec, so the process the parent waits on IS the
# child and exits with it; tee reads from the parent pipe in a process substitution).
# Used only to BUILD the replay transcript; never timed.
set -euo pipefail
dir="${M1_TRANSCRIPT_DIR:?}"
f="$dir/$(date +%s%N)_$$.in"
exec "${M1_REAL_SIM_BRIDGE:?}" "$@" < <(tee "$f")
