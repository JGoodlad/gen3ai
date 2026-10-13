#!/usr/bin/env bash
#
# cuda_build_validate.sh — the GPU validation of the CUDA 12.8 twin env against the default (cu126) env, on the
# LOCAL card: the update cells (the baseline the cloud ladder compares against) and the numerics comparison.
# A thin wrapper over src/main/ops/cuda_build_validate.py (the plan, the cells, the summary; its docstring is the
# design). Every GPU cell runs through scripts/ops/gpu_lock.sh + mem_cap.sh, so the GPU LEASE rules hold: this
# script never touches the card unleased and waits for nobody.
set -u
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'USAGE_EOF'
cuda_build_validate.sh — torch 2.8.0+cu126 vs +cu128 on the local GPU (update cells, numerics, a baseline table)

USAGE
    scripts/ops/cuda_build_validate.sh plan [opts]          # the cells, their full commands, the time; touches nothing
    scripts/ops/cuda_build_validate.sh cpu-smoke --out D    # the CODE PATH on CPU in both envs; no GPU, no lease
    GEN3AI_GPU_LEASE_TOKEN=<token> scripts/ops/cuda_build_validate.sh run [opts]   # THE GPU WINDOW (~70 min, resumable)
    scripts/ops/cuda_build_validate.sh summarize --out D    # re-render SUMMARY.md / summary.json from the cell JSONs

OPTIONS (see src/main/ops/cuda_build_validate.py)
    --out D            results directory (default designs/research_state/measurements/cuda_build_<date>)
    --archs E,P        perf-phase configs: E = the end state, P = production (both, with the ride-along flags)
    --precs fp32,bf16  trunk precisions            --only SUBSTR[,..]   run only matching cells   --force   redo done cells
    --epochs 1,3 --warm 1 --reps 2     update_bench's schedule        --mem-gb 32    mem_cap per cell
    --cell-timeout-min 50   wall timeout per cell, INSIDE the lock    --cpus LIST / --nice N   taskset / nice for every cell
    --py-cu126 P / --py-cu128 P        interpreters (default: the sibling envs gen3ai_torch28 / gen3ai_torch28_cu128)

`run` REFUSES (exit 2) without GEN3AI_GPU_LEASE_TOKEN (or GEN3AI_GPU_LEASE_TOKEN_FILE): the lease is granted by the
orchestrator to ONE agent; a card held by someone else makes gpu_lock.sh exit 6 (a one-off holder: 5) and this script stops
at once. A finished cell is skipped on a re-run; a failed cell is reported and the rest still run (exit 1). Run it from a
CLEAN checkout of the landed commit, detached, and poll its log with waits of at most 55 minutes.
USAGE_EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") usage >&2; echo "REFUSING: cuda_build_validate.sh needs a mode: plan | cpu-smoke | run | summarize" >&2; exit 2 ;;
esac

# This checkout's src/ FIRST (a worktree's code, not main's — the editable install names main), without touching the
# cells' own PYTHONPATH (the driver sets each cell's explicitly).
exec env PYTHONPATH="$(ops_repo_root)/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$(ops_python)" -m main.ops.cuda_build_validate "$@"
