#!/usr/bin/env bash
# Frozen record of the 2026-10-27 lean-screen review (run from /tmp/lws_probe; outputs probe_results.jsonl).
# Read-only probe on main HEAD (code == the review worktree); outputs only under /tmp/lws_probe.
cd /home/goodlad/dev/gen3ai || exit 1
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES="" POKESIM_CORE_EVENTS_BIN=$PWD/src/rust_sim/target/release/core_events
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
run() { (
  ulimit -v 6000000; timeout 560 nice -n 19 taskset -c 5-7,13-15 \
    /usr/bin/time -f "maxrss_kb %M wall %e" "$PY" /tmp/lws_probe/probe.py "$1" "$2"
  echo "exit $? for $1" )
}
for spec in "$@"; do run "${spec%%@*}" "/tmp/lws_probe/out_${spec##*@}.jsonl"; done
echo CHAIN_DONE
