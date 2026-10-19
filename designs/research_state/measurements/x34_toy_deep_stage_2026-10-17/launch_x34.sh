#!/usr/bin/env bash
# X34 — THE TOY DEEP RUN, STAGED 2026-10-17 (not run by the staging agent; the Training session runs it).
# A FORK of rb_toy_ent_s4002 from its 100M cap (final_model.zip) into rb_toy_ent_s4002_deep, PINNED at main, to its
# own Tier-1 plateau (design_compute_efficiency.md §4.3: Δ 10M, W 50M, two FLAT checks) or the 500M CAP.
# Run it ONLY after X26 has ended or been stopped and the GPU is free (the Training session's lease / window rules).
# Every pre-check below must pass or the script HOLDs (exit 7) without creating anything.
#   usage: launch_x34.sh [PIN]     (default PIN: the validated 6dcc769c; a newer main is fine once these checks pass)
set -u
REPO=/home/goodlad/dev/gen3ai; J=/home/goodlad/.claude/jobs/x34; ST=$J/status.txt; RUN=rb_toy_ent_s4002_deep
STAGE=$REPO/designs/research_state/measurements/x34_toy_deep_stage_2026-10-17
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3; OPS=$REPO/scripts/ops
PIN=$(git -C "$REPO" rev-parse "${1:-6dcc769c4539096d6544683fa262ecc980b60029}") || exit 7
A="$(cat "$STAGE/fork_argv_minimal.txt") --behaviour-check warn"   # §7 (2026-10-19): warn + registered ceilings; a fork otherwise inherits the parent's fatal
mkdir -p "$J"; log() { echo "[$(date -Is)] $1" >> "$ST"; }
hold() { log "HOLD: $1"; echo "HOLD: $1" >&2; exit 7; }

# ---- pre-checks (nothing is created until all pass) ------------------------------------------------------------
for fix in 5236eb22 9c033489; do   # F10: T36's canary fix + the memory fix
  git -C "$REPO" merge-base --is-ancestor "$fix" "$PIN" || hold "pin ${PIN:0:8} lacks $fix"
done
[ -e "$REPO/models/$RUN" ] && hold "models/$RUN already exists (a restart RESUMES: use the launcher's own restart, not this script)"
[ "$(sha256sum "$REPO/models/rb_toy_ent_s4002/final_model.zip" | cut -c1-64)" = \
  60af6b8814f80f72853a5fe6c28655620cd8640884cedf17ba7eec8bd95648fb ] || hold "the parent final_model.zip changed"
( cd "$REPO" && PYTHONPATH="$REPO/src" timeout 600 "$PY" -m main.checkargs --pin "$PIN" --argv "$A" ) \
  > "$J/checkargs.txt" 2>&1 || hold "checkargs refused (see $J/checkargs.txt)"
( cd "$REPO" && PYTHONPATH="$REPO/src" timeout 900 "$PY" -m main.launcher --dry-run --restart-interval-hours 24 \
  --pin-commit "$PIN" $A ) > "$J/dryrun.txt" 2>&1 || hold "launcher --dry-run refused (see $J/dryrun.txt)"
grep -q "would launch" "$J/dryrun.txt" || hold "dry-run did not say 'would launch'"
grep -q "role        : FORK of $REPO/models/rb_toy_ent_s4002" "$J/dryrun.txt" || hold "dry-run role is not FORK of s4002"
grep -q "+399,918,572 steps" "$J/dryrun.txt" || hold "dry-run step arithmetic moved (expected +399,918,572)"
# the identity proof at the PIN (structure + forward vs the toy's own pin 3899277a); CPU, one job, rule 5b
( cd "$REPO" && PYTHONPATH="$REPO/src" CUDA_VISIBLE_DEVICES="" timeout 900 nice -n 19 taskset -c 5-7,13-15 \
  "$PY" -m pytest src/agents/model/toy_network_identity_test.py -q -p no:cacheprovider ) > "$J/identity_test.txt" 2>&1 \
  || hold "toy_network_identity_test failed at main (see $J/identity_test.txt)"
grep -q "2 passed" "$J/identity_test.txt" || hold "identity test did not run both tests (a skip is not a pass)"

# ---- launch -----------------------------------------------------------------------------------------------------
echo $$ > "$J/chain.pid"
log "START X34 $RUN: fork of rb_toy_ent_s4002/final_model.zip @100,081,428 → 500M, pin ${PIN:0:8}, restart 24 h"
( cd "$REPO" && export PYTHONPATH="$REPO/src"; exec taskset -c 0-4,8-12 "$PY" -m main.launcher \
  --restart-interval-hours 24 --pin-commit "$PIN" $A ) >> "$J/$RUN.launcher.log" 2>&1 < /dev/null &
LP=$!; echo "$LP" > "$J/$RUN.pid"
( # wait (<= 10 min) for the TRAINER to create the run dir: the watcher refuses a missing dir (2026-10-19 race)
  for _ in $(seq 200); do [ -d "$REPO/models/$RUN" ] && break; kill -0 "$LP" 2>/dev/null || break; sleep 3; done
  [ -d "$REPO/models/$RUN" ] || { log "watcher NOT started: $RUN dir never appeared in 10 min"; exit 0; }
  "$OPS/watch_run.sh" "$REPO/models/$RUN" --pid-file "$J/$RUN.pid" --launcher-log "$J/$RUN.launcher.log" \
    --status-file "$J/$RUN.repo.watch.txt" ) >> "$J/$RUN.repo.watch.out" 2>&1 < /dev/null &
wait "$LP"; RC=$?
log "END $RUN launcher_rc=$RC"
