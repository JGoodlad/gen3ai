#!/usr/bin/env bash
# The G-U environment: the ARMS' OWN TREE (pin 6eb9c776) — its src/, its cwd, its own release
# sim_bridge (built in that worktree, never main's target). CPU only, one thread, nice 19 (coordinator 2026-09-25; niceness changes no number).
# Usage: gu_env.sh <command...>   (e.g. gu_env.sh python3 gu_driver.py status)
PIN=/home/goodlad/dev/gen3ai-wt/pin-6eb9c776-popr2
export PYTHONPATH=$PIN/src
export POKESIM_SIM_BRIDGE_BIN=$PIN/src/rust_sim/target/release/sim_bridge
export GU_TREE_COMMIT=$(git -C $PIN rev-parse HEAD)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd $PIN || exit 9
exec nice -n 19 "$@"
