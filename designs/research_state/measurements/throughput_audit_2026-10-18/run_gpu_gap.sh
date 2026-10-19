#!/usr/bin/env bash
# run_gpu_gap.sh — the throughput audit's GPU measurements for the 2026-10-19 GPU gap (X26 stops at ~380M).
#
# NOT RUN by the audit (2026-10-18: X26 held the GPU). Needs a GPU LEASE granted by the orchestrator to the agent
# that runs it (root CLAUDE.md standing rule 5):
#
#     scripts/ops/gpu_lease.sh acquire --owner <agent>      # prints the token
#     export GEN3AI_GPU_LEASE_TOKEN=<token>
#     export PYTHONPATH=$PWD/src PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
#     designs/research_state/measurements/throughput_audit_2026-10-18/run_gpu_gap.sh /tmp/gpu_gap_2026-10-19
#     scripts/ops/gpu_lease.sh release --owner <agent>
#
# Order = value per GPU-minute; each step is INCREMENTAL (a cell whose JSON reads "ok" is skipped on a re-run):
#   1. the FROZEN SUITE, all phases, on the local 3080 Ti — fills the PENDING row of cloud_gpu_benchmark.md §0
#      (~40-60 min; most of the collect phase is T2's cold compile);
#   2. the update at smaller effective batches (B_eff 8,192 and 2,048; ~8 min each) — the per-optimizer-step cost
#      of the "fewer epochs at the same optimizer-step count" lever;
#   3. collection at 512 / 1,024 envs and against a 5- / 1-snapshot pool (~15-20 min each, mostly T2 startup).
# Total ~2.5-3 h. Stop after step 2 if the gap is shorter; step 3's cells are independent.
# A BENCHMARK WARNS, IT NEVER STRETCHES: run nothing else on the GPU, and keep agent CPU work to rule 5b's cores.
set -euo pipefail
OUT="${1:?usage: run_gpu_gap.sh <out-dir> (never under models/)}"
: "${GEN3AI_GPU_LEASE_TOKEN:?the GPU lease token (scripts/ops/gpu_lease.sh acquire) — no lease, no GPU}"
: "${PY:?PY=<the gen3ai_torch28 python>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
LOCK="$ROOT/scripts/ops/gpu_lock.sh"
mkdir -p "$OUT/logs"

# 1. the frozen suite (its own driver: subprocess per phase, nvidia-smi sampling, per-phase wall bounds)
"$LOCK" "$PY" -m agents.training.gpu_bench_suite run --out "$OUT/suite_3080ti" \
    --tag "RTX 3080 Ti (local, 9800X3D)" 2>&1 | tee "$OUT/logs/suite.log"

# 2 + 3. the audit's cells: the suite's own `_phase` worker on a config with ONE field changed (gpu_gap_cells.py)
"$PY" "$HERE/gpu_gap_cells.py" --out "$OUT/cells"
"$PY" "$HERE/gpu_gap_cells.py" --out "$OUT/cells" --list | while read -r kind name; do
    res="$OUT/cells/phases/$name.json"
    if [ -f "$res" ] && "$PY" -c "import json,sys; sys.exit(json.load(open('$res')).get('status') != 'ok')"; then
        echo "[gpu_gap] $name: ok already — skipped"; continue
    fi
    echo "[gpu_gap] $name ($kind): START $(date +%H:%M:%S)"
    "$LOCK" timeout 2700 "$PY" -u -m agents.training.gpu_bench_suite _phase "$kind" "$OUT/cells/phases/$name.cfg.json" \
        > "$OUT/logs/$name.log" 2>&1 || echo "[gpu_gap] $name: exit $? (see $OUT/logs/$name.log)"
done
echo "[gpu_gap] done; read: python3 $HERE/read_gpu_gap.py $OUT"
