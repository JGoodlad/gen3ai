#!/usr/bin/env bash
# P0 driver: the head-to-head round-robin of the four banked finals, ONE edge per invocation of `main.h2h play`,
# each under the GPU lock (held for that edge only, ~5 min) and a memory cap, the wall timeout INSIDE both.
# Resumable: `main.h2h play` skips the batches already in rows/, so a killed driver is simply re-run.
#
#   run_edges.sh                    # every edge below, in order
#   run_edges.sh A2:Ap A:A2         # just these (player:opponent)
#
# Edges (player:opponent; the PLAYER keeps seat p1): the three replicate pairs in both directions (the two
# directions of a pair draw the SAME team pairs, so d_ij + d_ji = 2 x the seat effect), B against each (sensitivity,
# one direction), and the replicate runs against THEMSELVES (the seat effect and the exact-cancellation diagnostic).
set -u
ROOT=${ROOT:-/home/goodlad/dev/gen3ai-wt/p0h2h}
MODELS=${MODELS:-/home/goodlad/dev/gen3ai/models}
HERE="$ROOT/designs/research_state/measurements/x5_p0_h2h_2026-10-03"
PY=${PY:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
PAIRS=${PAIRS:-5000}
export PYTHONPATH="$ROOT/src"
declare -A RUN=( [A]=sizing_A_n48_e10_s1001 [A2]=sizing_A2_n48_e10_s1001 [Ap]=sizing_Ap_n48_e10_s1002 [B]=sizing_B_n256_e10_s1001 )
DEFAULT_EDGES=(A2:Ap A:A2 A:Ap Ap:A2 A2:A Ap:A B:A B:A2 B:Ap A2:A2 Ap:Ap A:A)
EDGES=("$@"); [ ${#EDGES[@]} -eq 0 ] && EDGES=("${DEFAULT_EDGES[@]}")
mkdir -p "$HERE/rows" "$HERE/logs"
cd "$ROOT" || exit 2
for e in "${EDGES[@]}"; do
    p=${e%%:*}; o=${e##*:}
    tmp=$(mktemp /tmp/p0h2h_edge_XXXXXX.log)
    echo "[run_edges] $(date -u +%FT%TZ) $p vs $o"
    scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 20 timeout 1500 "$PY" -m main.h2h play \
        --player "$MODELS/${RUN[$p]}" --opponent "$MODELS/${RUN[$o]}" --pairs "$PAIRS" --batch-pairs 500 \
        --out "$HERE/rows" --label x5_p0_h2h_2026-10-03 --purpose audit --seed 0 \
        --device cuda --n-envs 64 --threads 6 > "$tmp" 2>&1
    rc=$?
    { grep -E '^\[h2h\]|^\[mem_cap\]|^\[gpu_lock\]' "$tmp"; tail -n 2 "$tmp" | grep -v '^{' ; echo "[run_edges] exit $rc"; } >> "$HERE/logs/${p}_vs_${o}.log"
    [ $rc -ne 0 ] && { echo "[run_edges] $p vs $o FAILED rc=$rc (full log kept: $tmp)"; exit $rc; }
    rm -f "$tmp"
done
echo "[run_edges] done $(date -u +%FT%TZ)"
