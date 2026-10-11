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

# envjob <tag> <budget-minutes> <src> <bf16_envelope args...>: the bf16 parity ENVELOPE (no timing; PLAN.md §2a)
envjob() {
  local tag=$1 budget=$2 src=$3; shift 3
  local now left
  now=$(date +%s); left=$(( (DEADLINE - now) / 60 ))
  if (( left < budget )); then
    echo "[window $WIN] SKIP $tag: needs ${budget} min, ${left} min left" | tee -a "$RES/window.log"
    return
  fi
  echo "[window $WIN] START $tag at $(date +%T) (budget ${budget} min, ${left} min left)" | tee -a "$RES/window.log"
  DIAG_SRC="$src" timeout $(( budget * 60 )) "$ROOT/scripts/ops/gpu_lock.sh" "$ROOT/scripts/ops/mem_cap.sh" 40 \
      "$PY" "$HERE/bf16_envelope.py" --out "$RES/$tag.json" "$@" > "$RES/logs/$tag.log" 2>&1
  echo "[window $WIN] END $tag rc=$? at $(date +%T): $(grep -h 'DONE' "$RES/logs/$tag.log" | tail -1)" \
      | tee -a "$RES/window.log"
}

# launchjob <tag> <budget-minutes>: a FRESH one-update `--arch endstate` launch through the launcher's normal pinned
# path (a scratch run archive, never models/; an EMPTY shared compile root of this day's own), its startup lines kept.
CC_ROOT="$HOME/.cache/gen3ai/tmp/t25_cc_$(date +%Y%m%d)"
RUNS=/home/goodlad/gen3ai_archive/perf_phase_2026-10-10/runs
launchjob() {
  local tag=$1 budget=$2
  local now left
  now=$(date +%s); left=$(( (DEADLINE - now) / 60 ))
  if (( left < budget )); then
    echo "[window $WIN] SKIP $tag: needs ${budget} min, ${left} min left" | tee -a "$RES/window.log"
    return
  fi
  echo "[window $WIN] START $tag at $(date +%T) (budget ${budget} min, ${left} min left)" | tee -a "$RES/window.log"
  local run="rb_t25_${tag}_$(date +%H%M)"
  ( cd "$ROOT" && GEN3AI_MODELS_DIR="$RUNS" GEN3AI_COMPILE_SHARED_DIR="$CC_ROOT" PYTHONPATH="$ROOT/src" \
      timeout $(( budget * 60 )) "$ROOT/scripts/ops/gpu_lock.sh" "$ROOT/scripts/ops/mem_cap.sh" 40 "$PY" -m main.launcher \
      --restart-interval-hours 0 --device cuda --arch endstate --steps 98304 --snapshot-ladder-games 0 \
      --run-name "$run" ) > "$RES/logs/$tag.log" 2>&1
  local rc=$?
  grep -hE "CompileCache\]|reset \+ prewarm took|T2 (up|SHARED) in|Training complete" \
      "$RUNS/$run/launcher_child.log" > "$RES/$tag.startup.txt" 2>/dev/null
  echo "[window $WIN] END $tag rc=$rc at $(date +%T): $(head -3 "$RES/$tag.startup.txt" | tr '\n' ' ' | cut -c1-300)" \
      | tee -a "$RES/window.log"
}

# t2job <tag> <budget-minutes> <tree-root> <lead_lanes_benchmark args...>: T2 lever 4 (`--t2-lead-trainee`), the paired
# OFF/ON service benchmark (`agents.inference.service.lead_lanes_benchmark`), on a pinned tree, its own compile cache.
t2job() {
  local tag=$1 budget=$2 tree=$3; shift 3
  local now left
  now=$(date +%s); left=$(( (DEADLINE - now) / 60 ))
  if (( left < budget )); then
    echo "[window $WIN] SKIP $tag: needs ${budget} min, ${left} min left" | tee -a "$RES/window.log"
    return
  fi
  echo "[window $WIN] START $tag at $(date +%T) (budget ${budget} min, ${left} min left)" | tee -a "$RES/window.log"
  local cc="$HOME/.cache/gen3ai/tmp/t25_t2lead_$(date +%Y%m%d)"
  mkdir -p "$cc"
  ( cd "$tree" && TORCHINDUCTOR_CACHE_DIR="$cc" PYTHONPATH="$tree/src" timeout $(( budget * 60 )) \
      "$ROOT/scripts/ops/gpu_lock.sh" "$ROOT/scripts/ops/mem_cap.sh" 40 "$PY" -m agents.inference.service.lead_lanes_benchmark \
      --json "$RES/$tag.json" "$@" ) > "$RES/logs/$tag.log" 2>&1
  echo "[window $WIN] END $tag rc=$? at $(date +%T)" | tee -a "$RES/window.log"
}

TA="$TREES/at_6e0a1a7a/src"     # window A's pinned tree (the 16-roll exact KO)
case "$WIN" in
  A)
    # Window A is PINNED to 6e0a1a7a (the 16-roll exact KO; main moves to the closed form after it), extracted by
    # `git archive 6e0a1a7a src data designs scripts`.
    TA="$TREES/at_6e0a1a7a/src"
    job E   10 "$TA" --argv "$(cfg E)"  --t2 256 --profile
    job P    6 "$TA" --argv "$(cfg P)"
    job R    7 "$TA" --argv "$(cfg R)"
    job EK   7 "$TA" --argv "$(cfg EK)"
    job SF   7 "$TA" --argv "$(cfg SF)"
    job S    7 "$TA" --argv "$(cfg S)"
    ;;
  B)
    # Budgets from window A's measured times (E: compile + gate + prewarm 277 s, the timed schedule ~85 s).
    job E_base       8 HEAD --argv "$(cfg E)" --canary --profile   # the levers' paired baseline (closed-form KO)
    job E_bf16       8 HEAD --argv "$(cfg E)" --trunk-precision bf16 --canary --profile
    job E_cudagraphs 8 HEAD --argv "$(cfg E)" --r1-preset cudagraphs --canary --profile
    job P_pre_item1  5 "$TREES/pre_957d4dbd/src" --argv "--arch production $(cfg P | sed 's/--arch production//')" --profile
    job P_at_item1   5 "$TREES/at_957d4dbd/src"  --argv "--arch production $(cfg P | sed 's/--arch production//')" --profile
    job E_coordesc  11 HEAD --argv "$(cfg E)" --r1-preset coordesc --canary
    ;;
  C)
    # WHO SYNCS (F-PB-4) + T2 per-step for E; then the SHARED compile cache's cold vs warm fresh launch
    # (`designs/training/compile_flags.md` "The SHARED compile cache, by KEY"; the per-run control is skipped: a
    # cold shared start compiles exactly what a per-run cold start compiles); then T2 per-step for P and R.
    job E_stack     11 HEAD --argv "$(cfg E)" --profile --profile-epochs 3 --sync-sites --t2 256,64
    launchjob cc_cold 18
    launchjob cc_warm  9
    job T2_P         5 "$TA" --argv "$(cfg P)" --t2-only --t2 256,64
    job T2_R         5 "$TA" --argv "$(cfg R)" --t2-only --t2 256,64
    ;;
  D)
    # The ride-along host-read fix (3545b868) A/B against its parent (5e399caf), both PINNED to extracted trees (the
    # worktree's own src/ is mid-edit), then the bf16 parity envelope (PLAN.md §2a) on the fix's tree.
    TD="$TREES/at_3545b868/src"
    job E_ride_before 11 "$TREES/at_5e399caf/src" --argv "$(cfg E)" --profile --profile-epochs 3 --sync-sites
    job E_ride_after  11 "$TD" --argv "$(cfg E)" --profile --profile-epochs 3 --sync-sites
    envjob E_bf16_envelope 15 "$TD" --argv "$(cfg E)"
    ;;
  E)
    # The LAST window during the closing test. Lever C (--learner-cudagraphs on) and lever A on its flag path
    # (--trunk-precision bf16: the bf16 parity rule at startup + the canary), all on the extracted tree of the commit
    # that landed them ($TREES/at_lever_a), in priority order; then T2 lever 4's paired OFF/ON service benchmark; the
    # paired baseline last but one (window D's E_ride_after, 53.24 s, is the fallback comparator, ~0.2 s drift).
    TE="$TREES/at_lever_a/src"
    job E_cg        13 "$TE" --argv "$(cfg E) --learner-cudagraphs on" --canary --profile
    job E_bf16_flag 12 "$TE" --argv "$(cfg E) --trunk-precision bf16" --canary --profile
    t2job T2_lead   15 "$TREES/at_lever_a"
    job E_base_e     8 "$TE" --argv "$(cfg E)" --canary --profile
    job E_cg_bf16   13 "$TE" --argv "$(cfg E) --learner-cudagraphs on --trunk-precision bf16" --canary --profile
    ;;
  *) echo "unknown window $WIN" >&2; exit 2 ;;
esac
echo "[window $WIN] FINISHED at $(date +%T)" | tee -a "$RES/window.log"
