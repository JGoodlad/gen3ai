#!/usr/bin/env bash
#
# ship_gate.sh — THE SHIP GATE: the routine gate (`-m "not slow and not e2e"`, every test under src/,
# the static gates included) in the shape the box can afford right now, through gate_lock.sh +
# mem_cap.sh with CUDA_VISIBLE_DEVICES="". A live training run ⇒ rule 5b (nice 19, the agent cores,
# the measured -n); none ⇒ the quiet -n 6. It never selects tests by the change.
#
# USAGE (--help prints it)
#     scripts/ops/ship_gate.sh run [--max-run-s S] [-- <extra pytest args>]
#     scripts/ops/ship_gate.sh --dry-run [--max-run-s S] [-- <extra pytest args>]
# No argument refuses (exit 2). NEVER wrap it in `timeout` (it queues; the timeout counts the wait and
# loses the gate's place): --max-run-s S bounds the run from pytest's start (exit 89).
#
# Detail: designs/ops/testing.md "The SHIP GATE"; the code is src/utils/ship_gate.py.
set -u
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export GEN3AI_PYTHON="${GEN3AI_PYTHON:-$(ops_python)}"
exec "$GEN3AI_PYTHON" -c 'import sys; sys.path.insert(0, sys.argv[1]); from utils.ship_gate import main; sys.exit(main(sys.argv[2:]))' \
    "$(ops_repo_root)/src" "$@"
