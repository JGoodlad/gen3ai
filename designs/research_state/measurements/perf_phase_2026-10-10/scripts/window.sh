#!/usr/bin/env bash
# THE PERF-PHASE GPU WINDOW DRIVER (2026-10-10). Runs one window's jobs in PRIORITY order, one at a time, each under
# the granted lease (gpu_lock.sh, the token from --token-file) and a memory cap, with a per-job timeout, and never
# STARTS a job that cannot finish before the window's deadline. Results land incrementally (one JSON per job), so
# a cut-off window loses only its tail.
#
#   scripts/ops/gpu_lease.sh acquire --owner <name> --token-file /tmp/perf_lease.tok   # the orchestrator's grant
#   bash window.sh A /tmp/perf_lease.tok 44                                            # window A, 44-minute deadline
#
# Window A = the COST ATTRIBUTION (production -> static -> +facts -> +depth -> end state; exact KO) + E's fresh profile.
# Window B = the LEVERS on E (bf16 trunk, Inductor presets) + perf item 1's deferred A/B (957d4dbd^ vs 957d4dbd).
set -u
WIN=${1:?window A or B}
TOKFILE=${2:?the lease token file}
MINUTES=${3:-44}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
RES="$HERE/../results/window_$WIN"
mkdir -p "$RES/logs"
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
TREES=/home/goodlad/gen3ai_archive/perf_phase_2026-10-10/trees
DEADLINE=$(( $(date +%s) + MINUTES * 60 ))
export GEN3AI_GPU_LEASE_TOKEN_FILE="$TOKFILE"

# job <tag> <budget-minutes> <src-or-HEAD> <update_bench args...>
job() {
  local tag=$1 budget=$2 src=$3; shift 3
  local now left
  now=$(date +%s); left=$(( (DEADLINE - now) / 60 ))
  if (( left < budget )); then
    echo "[window $WIN] SKIP $tag: needs ${budget} min, ${left} min left" | tee -a "$RES/window.log"
    return
  fi
  echo "[window $WIN] START $tag at $(date +%T) (budget ${budget} min, ${left} min left)" | tee -a "$RES/window.log"
  local env=()
  [[ "$src" != HEAD ]] && env=(env "DIAG_SRC=$src")
  "${env[@]}" timeout $(( budget * 60 )) "$ROOT/scripts/ops/gpu_lock.sh" "$ROOT/scripts/ops/mem_cap.sh" 40 \
      "$PY" "$HERE/update_bench.py" --tag "$tag" --out "$RES/$tag.json" "$@" > "$RES/logs/$tag.log" 2>&1
  echo "[window $WIN] END $tag rc=$? at $(date +%T): $(grep -h 'DONE' "$RES/logs/$tag.log" | tail -1)" \
      | tee -a "$RES/window.log"
}
cfg() { "$PY" "$HERE/configs.py" --argv "$1"; }

case "$WIN" in
  A)
    job E   11 HEAD --argv "$(cfg E)"  --t2 256 --profile
    job P    9 HEAD --argv "$(cfg P)"  --t2 256
    job R    9 HEAD --argv "$(cfg R)"  --t2 256
    job EK   7 HEAD --argv "$(cfg EK)"
    job SF   7 HEAD --argv "$(cfg SF)"
    job S    7 HEAD --argv "$(cfg S)"
    ;;
  B)
    job E_bf16       8 HEAD --argv "$(cfg E)" --trunk-precision bf16 --canary --profile
    job E_coordesc  10 HEAD --argv "$(cfg E)" --r1-preset coordesc --canary
    job E_cudagraphs 8 HEAD --argv "$(cfg E)" --r1-preset cudagraphs --canary
    job P_pre_item1  7 "$TREES/pre_957d4dbd/src" --argv "--arch production $(cfg P | sed 's/--arch production//')" --profile
    job P_at_item1   7 "$TREES/at_957d4dbd/src"  --argv "--arch production $(cfg P | sed 's/--arch production//')" --profile
    job E_combo      8 HEAD --argv "$(cfg E)" --r1-preset combo --canary
    ;;
  *) echo "unknown window $WIN" >&2; exit 2 ;;
esac
echo "[window $WIN] FINISHED at $(date +%T)" | tee -a "$RES/window.log"
