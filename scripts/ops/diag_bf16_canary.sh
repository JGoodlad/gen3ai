#!/usr/bin/env bash
#
# diag_bf16_canary.sh — the GPU half of the bf16-canary diagnosis (2026-10-15, X26 BLOCKER): `rb_toy_ent_s4001` ended
# FATAL at update 700 when the in-run canary found the COMPILED bf16 learner's pointer-head gradient 14x farther from
# the fp32 reference than eager bf16's own error. This runs `python -m main.ops.diag_bf16_canary gpu` ONCE on the
# local card under the caller's GPU LEASE: on the canary's own weights (`final_model_canary_fatal.zip`) and a growth
# curve of checkpoints, every arm (eager fp32 / bf16; compiled bf16 = the production compile, compiled fp32, compiled
# bf16 with `emulate_precision_casts`, through `aot_eager`, and with the pointer head eager) on the canary's own rows,
# the pairwise per-parameter distances, the canary's verdict replayed, the forward-vs-backward read, the per-row
# conditioning and a dithered null. Hypotheses and how each read decides them:
# designs/training/compile_flags.md "The bf16 canary FATAL on trained weights".
#
# One process inside `gpu_lock.sh` (the lease token passes straight through; anyone else's lease = exit 6 at once)
# and `mem_cap.sh`, with a hard wall timeout. Nothing here waits for the GPU and nothing writes under models/.
set -u
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'USAGE_EOF'
diag_bf16_canary.sh — the bf16 canary FATAL on trained weights: the GPU diagnostic (<= 20 min)

USAGE
    GEN3AI_GPU_LEASE_TOKEN=<token> scripts/ops/diag_bf16_canary.sh RUN [--out DIR] [--ckpts LIST] [--budget-min 14]
                                   [--timeout-s 1200] [--mem-gb 16] [--cpus LIST]
    scripts/ops/diag_bf16_canary.sh RUN --plan       # print the command; touches nothing

    RUN       the run NAME under models/ (rb_toy_ent_s4001) or a run directory
    --ckpts   comma list for `main.ops.diag_bf16_canary gpu --ckpts` (default: fatal,72M,70M,66M,60M,45M,30M,15M —
              the FIRST builds the learners; `fatal` = final_model_canary_fatal.zip, the canary's own weights)
    --budget-min  stop STARTING checkpoints after this many minutes (default 14); the hard wall is --timeout-s

OUTPUT (DIR, default ~/.cache/gen3ai/diag_bf16_canary/<sha>):
    gpu_<label>.json per checkpoint (written as each finishes: a cut-off window keeps its head), gpu_meta.json,
    gpu_status.json, VERDICT.md (the one-page template, filled), diag.log. Exit 0 = every checkpoint read, 3 = the
    wall budget left some unread (INCOMPLETE, never a semantic outcome), 2 = refused, 5 / 6 = the GPU is not ours.

Run from a CLEAN checkout of the landed commit. REFUSES (exit 2) without GEN3AI_GPU_LEASE_TOKEN / _TOKEN_FILE.
USAGE_EOF
}

if [ $# -eq 0 ]; then
    usage >&2
    echo "REFUSING: no run given (pass the run name, e.g. rb_toy_ent_s4001)" >&2
    exit 2
fi
case "$1" in -h|--help) usage; exit 0 ;; esac
RUN_ARG="$1"; shift
OUT="" CKPTS="" BUDGET=14 TIMEOUT_S=1200 MEM_GB=16 CPUS="" PLAN=0
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --out) OUT="$2"; shift 2 ;;
        --ckpts) CKPTS="$2"; shift 2 ;;
        --budget-min) BUDGET="$2"; shift 2 ;;
        --timeout-s) TIMEOUT_S="$2"; shift 2 ;;
        --mem-gb) MEM_GB="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --plan) PLAN=1; shift ;;
        *) usage >&2; echo "REFUSING: unknown argument $1" >&2; exit 2 ;;
    esac
done

RUN_DIR="$(ops_resolve_run "$RUN_ARG")" || exit 2
ROOT="$(ops_repo_root)"
PY="$(ops_python)"
SHA="$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
DIRTY="$(git -C "$ROOT" status --porcelain 2>/dev/null | head -c1)"
[ -n "$OUT" ] || OUT="${XDG_CACHE_HOME:-$HOME/.cache}/gen3ai/diag_bf16_canary/$SHA"
PIN=()
[ -n "$CPUS" ] && PIN=(taskset -c "$CPUS")
CK=()
[ -n "$CKPTS" ] && CK=(--ckpts "$CKPTS")
CMD=("$ROOT/scripts/ops/gpu_lock.sh" "$ROOT/scripts/ops/mem_cap.sh" --name diag-bf16-canary "$MEM_GB"
     timeout "$TIMEOUT_S" "${PIN[@]}" "$PY" -m main.ops.diag_bf16_canary gpu --run "$RUN_DIR" --out "$OUT"
     --budget-min "$BUDGET" "${CK[@]}")

if [ "$PLAN" = 1 ]; then
    printf 'cd %s && GEN3AI_TEST_ALLOW_GPU=1 PYTHONPATH=%s/src \\\n    ' "$ROOT" "$ROOT"
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi
if [ -z "${GEN3AI_GPU_LEASE_TOKEN:-}" ] && [ -z "${GEN3AI_GPU_LEASE_TOKEN_FILE:-}" ]; then
    echo "REFUSING: no GEN3AI_GPU_LEASE_TOKEN / GEN3AI_GPU_LEASE_TOKEN_FILE — the GPU is LEASED; run this as the lease holder" >&2
    exit 2
fi

mkdir -p "$OUT"
export GEN3AI_TEST_ALLOW_GPU=1 PYTHONPATH="$ROOT/src"
echo "[diag] commit $SHA${DIRTY:+ (DIRTY tree)} run $RUN_DIR -> $OUT (log: $OUT/diag.log); est. 12-16 min, wall $TIMEOUT_S s"
START=$(date +%s)
(cd "$ROOT" && "${CMD[@]}") > "$OUT/diag.log" 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - START ))
echo "[diag] exit $RC after ${ELAPSED}s"
case "$RC" in
    5|6) echo "[diag] the GPU is not ours (gpu_lock.sh exit $RC): nothing ran" >&2; exit "$RC" ;;
    124) echo "[diag] the wall timeout ($TIMEOUT_S s) cut the read: the finished checkpoints are in $OUT (INCOMPLETE)" >&2 ;;
    86) echo "[diag] OOM-killed at the ${MEM_GB} GB cap: the finished checkpoints are in $OUT (INCOMPLETE)" >&2 ;;
esac
tail -n 12 "$OUT/diag.log"
# a cut-off read never rendered its template: render it from whatever finished (CPU only, seconds)
(cd "$ROOT" && "$PY" -m main.ops.diag_bf16_canary verdict --out "$OUT" > /dev/null 2>&1) || true
[ -f "$OUT/VERDICT.md" ] && echo "[diag] verdict template: $OUT/VERDICT.md"
exit "$RC"
