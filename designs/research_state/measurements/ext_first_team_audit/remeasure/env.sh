#!/usr/bin/env bash
# The arms' OWN tree (pin 6eb9c776), its own release sim_bridge, CPU only, one thread, nice 19.
PIN=/home/goodlad/dev/gen3ai-wt/ext-audit-pin
export PYTHONPATH=$PIN/src
export POKESIM_SIM_BRIDGE_BIN=$PIN/src/rust_sim/target/release/sim_bridge
export AUDIT_TREE_COMMIT=$(git -C $PIN rev-parse HEAD)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd $PIN || exit 9
exec nice -n 19 "$@"
