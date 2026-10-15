#!/usr/bin/env bash
#
# verify_compile_gate_cuda.sh — the GPU half of the 2026-10-15 R1-gate fix: the two CUDA test files of region R1's
# startup gate, run ONCE on the local card under the caller's GPU LEASE, plus the bf16 gate's measured RESOLUTION
# (the smallest planted backward miscompile it refuses, per parameter; `compile_regions_resolution_test`).
#
#   src/agents/model/compile_regions_fixed_mass_cuda_test.py   fresh weights: the fp32 gate's teeth (a 10 % plant is
#                                                             FATAL) + the bf16 rule (pass, NaN named, resolution on 4
#                                                             representative parameters, confirmed at 0.5x / 2x)
#   src/agents/model/compile_regions_trained_cuda_test.py      the `production` baseline's trained weights (by name):
#                                                             the fp32 gate + the bf16 rule (+ its resolution there)
#
# One pytest process, serial, `slow` + GPU (GEN3AI_TEST_ALLOW_GPU=1), inside `gpu_lock.sh` (the lease token passes
# straight through; anyone else's lease = exit 6 at once) and `mem_cap.sh` (peak measured 7.5 GB for the old pair).
# ~12-14 min on the RTX 3080 Ti (estimate: the old pair took 4.2 min; the two bf16 tests add a compile each and ~40
# micro-steps at B = 2,048). Nothing here waits for the GPU.
set -u
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() {
    cat <<'USAGE_EOF'
verify_compile_gate_cuda.sh — R1's startup gate on CUDA: the fp32 gate's teeth, the bf16 rule, its measured resolution

USAGE
    GEN3AI_GPU_LEASE_TOKEN=<token> scripts/ops/verify_compile_gate_cuda.sh [--out DIR] [--mem-gb 32] [--timeout-s 2400] [--cpus LIST]
    scripts/ops/verify_compile_gate_cuda.sh --plan       # print the command; touches nothing

OUTPUT (DIR, default ~/.cache/gen3ai/verify_compile_gate_cuda/<sha>):
    pytest.log, junit.xml, bf16_fresh.json + bf16_trained.json (the resolution records), summary.json
    summary.json: {"verdict": "PASS"|"FAIL"|"INCOMPLETE", "tests": {id: pass|fail|skip}, the resolution records'
    headline numbers, the commit, whether the tree was clean}. Exit 0 only when all four tests PASSED (a SKIP is
    INCOMPLETE, exit 3: a skipped GPU test verified nothing).

Run from a CLEAN checkout of the landed commit. From the MAIN checkout the slow-tier verdicts are NOT banked into
designs/ops/slow_tier_status.json (GEN3AI_SKIP_SLOW_STATUS_RECORD=1: main is never dirty) — they are in summary.json;
from a worktree they are banked as usual. REFUSES (exit 2) without GEN3AI_GPU_LEASE_TOKEN / _TOKEN_FILE.
USAGE_EOF
}

OUT="" MEM_GB=32 TIMEOUT_S=2400 CPUS="" PLAN=0
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --out) OUT="$2"; shift 2 ;;
        --mem-gb) MEM_GB="$2"; shift 2 ;;
        --timeout-s) TIMEOUT_S="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --plan) PLAN=1; shift ;;
        *) usage >&2; echo "REFUSING: unknown argument $1" >&2; exit 2 ;;
    esac
done

ROOT="$(ops_repo_root)"
PY="$(ops_python)"
SHA="$(git -C "$ROOT" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
DIRTY="$(git -C "$ROOT" status --porcelain 2>/dev/null | head -c1)"
[ -n "$OUT" ] || OUT="${XDG_CACHE_HOME:-$HOME/.cache}/gen3ai/verify_compile_gate_cuda/$SHA"
TESTS=(src/agents/model/compile_regions_fixed_mass_cuda_test.py src/agents/model/compile_regions_trained_cuda_test.py)
PIN=()
[ -n "$CPUS" ] && PIN=(taskset -c "$CPUS")
CMD=("$ROOT/scripts/ops/gpu_lock.sh" "$ROOT/scripts/ops/mem_cap.sh" --name verify-r1-gate "$MEM_GB"
     timeout "$TIMEOUT_S" "${PIN[@]}" "$PY" -m pytest "${TESTS[@]}" -m slow -p no:xdist -q -rA
     --junitxml "$OUT/junit.xml")

if [ "$PLAN" = 1 ]; then
    printf 'cd %s && GEN3AI_TEST_ALLOW_GPU=1 GEN3AI_GATE_RESOLUTION_OUT=%s PYTHONPATH=%s/src \\\n    ' "$ROOT" "$OUT" "$ROOT"
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi
if [ -z "${GEN3AI_GPU_LEASE_TOKEN:-}" ] && [ -z "${GEN3AI_GPU_LEASE_TOKEN_FILE:-}" ]; then
    echo "REFUSING: no GEN3AI_GPU_LEASE_TOKEN / GEN3AI_GPU_LEASE_TOKEN_FILE — the GPU is LEASED; run this as the lease holder" >&2
    exit 2
fi

mkdir -p "$OUT"
export GEN3AI_TEST_ALLOW_GPU=1 GEN3AI_GATE_RESOLUTION_OUT="$OUT" PYTHONPATH="$ROOT/src"
if [ "$(ops_main_checkout)" = "$ROOT" ]; then
    export GEN3AI_SKIP_SLOW_STATUS_RECORD=1
    echo "[verify] MAIN checkout: slow-tier verdicts NOT banked (main is never dirty) — read $OUT/summary.json"
fi
echo "[verify] commit $SHA${DIRTY:+ (DIRTY tree)} -> $OUT (log: $OUT/pytest.log); est. 12-14 min"
START=$(date +%s)
(cd "$ROOT" && "${CMD[@]}") > "$OUT/pytest.log" 2>&1
RC=$?
ELAPSED=$(( $(date +%s) - START ))
echo "[verify] pytest exit $RC after ${ELAPSED}s"
case "$RC" in
    5|6) echo "[verify] the GPU is not ours (gpu_lock.sh exit $RC): nothing ran" >&2; exit "$RC" ;;
esac

"$PY" - "$OUT" "$SHA" "${DIRTY:+dirty}" "$RC" "$ELAPSED" <<'PY'
import json, os, sys
import xml.etree.ElementTree as ET
out, sha, dirty, rc, elapsed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
tests = {}
unreadable = None
try:
    for case in ET.parse(os.path.join(out, "junit.xml")).getroot().iter("testcase"):
        tid = f"{case.get('classname')}::{case.get('name')}"
        kids = {c.tag for c in case}
        tests[tid] = "fail" if kids & {"failure", "error"} else "skip" if "skipped" in kids else "pass"
except (OSError, ET.ParseError) as exc:
    unreadable = f"{type(exc).__name__}: {exc}"
records = {}
for name in ("bf16_fresh", "bf16_trained"):
    p = os.path.join(out, f"{name}.json")
    if os.path.exists(p):
        r = json.load(open(p))
        records[name] = {"what": r.get("what"), "regime": r.get("regime"), "gate_resolution": r.get("gate"),
                         "confirmed": r.get("confirmed"),
                         "per_rung": {k: {"resolution": v["resolution"], "same_precision": v["same_precision"]}
                                      for k, v in r.get("rungs", {}).items()}}
want = 4
# a recorded failure is FAIL; four passes and a clean exit is PASS; anything else (a skip, a timeout — exit 124 —,
# an unreadable junit, a missing test) verified nothing: INCOMPLETE. A timeout is never a semantic outcome.
if any(v == "fail" for v in tests.values()):
    verdict = "FAIL"
elif rc == 0 and len(tests) == want and all(v == "pass" for v in tests.values()):
    verdict = "PASS"
else:
    verdict = "INCOMPLETE"
summary = {"verdict": verdict, "commit": sha, "tree": dirty or "clean", "pytest_exit": rc, "elapsed_s": elapsed,
           "tests": tests, "junit_unreadable": unreadable, "resolution": records,
           "read_me": "designs/training/compile_flags.md 'The bf16 gate's resolution' says what these numbers mean"}
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("verdict", "commit", "tree", "elapsed_s", "tests")}, indent=1))
for name, r in records.items():
    print(f"[{name}] gate resolution {r['gate_resolution']}")
sys.exit(0 if verdict == "PASS" else 3 if verdict == "INCOMPLETE" else 1)
PY
