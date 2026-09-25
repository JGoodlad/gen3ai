# Sourced by ../launch_<arm>.sh — the shared pre-flight + launch for round 2's READERS and the
# convergence side-check. Round 1's scripts/_launch_common.sh, with two changes: the fork-parent
# check is named PARENT_* (for a reader the parent IS the target; for an extension it is the round-1
# reader), and an optional EXPLOITER_ZIP/EXPLOITER_STEP pins the target of an extension.
# Not executable on its own. Each caller sets: ARM, RUN, ARGV_FILE, PREREQ_ZIPS (array), PARENT_ZIP,
# PARENT_STEP, optionally EXPLOITER_ZIP + EXPLOITER_STEP, STOP_LIST, then calls `launch_main "$@"`.
#
# 🚨 PREPARED BUT NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session, VERBATIM.
set -euo pipefail

REPO=/home/goodlad/dev/gen3ai
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
LOG_DIR="${LOG_DIR:-/home/goodlad/.claude/jobs/popr2_2026-09-24/launch}"

_num_timesteps() {
  "$P" -c "import json,sys; print(json.load(open(sys.argv[1])).get('num_timesteps'))" "$1/metadata.json"
}

_require_finished_at() {   # <zip> <step> <role>
  local z="$1" want="$2" role="$3" d got
  d="$(dirname "$z")"
  got="$(_num_timesteps "$d")"
  [ "$got" = "$want" ] || {
    echo "FATAL: $role $d num_timesteps=$got, registered $want. The post-fork budget would not"
    echo "       be the registered one and best_response_gap would REFUSE the read. Do NOT edit"
    echo "       --steps here: it is a registration change — record it in the ledger BEFORE launching."
    exit 4; }
  grep -aq "Training complete" "$d/launcher_child.full.log" 2>/dev/null || {
    echo "FATAL: $d/launcher_child.full.log has no 'Training complete' — the $role has not finished."
    exit 4; }
}

launch_main() {
  export PYTHONPATH="${PYTHONPATH:-}:$REPO/src"
  cd "$REPO"
  [ -f "$ARGV_FILE" ] || { echo "FATAL: no argv at $ARGV_FILE"; exit 2; }
  ARGV="$(cat "$ARGV_FILE")"
  grep -q -- "--run-name $RUN " <<<"$ARGV " || { echo "FATAL: $ARGV_FILE does not name --run-name $RUN"; exit 2; }
  grep -q -- "--model $PARENT_ZIP " <<<"$ARGV " || { echo "FATAL: $ARGV_FILE does not fork from $PARENT_ZIP"; exit 2; }

  # --- inputs must exist, as FILES (a bare run dir would mean 'last snapshot') ---
  for z in "${PREREQ_ZIPS[@]}"; do
    [ -f "$z" ] || { echo "FATAL: prerequisite $z does not exist — this arm cannot launch yet."; exit 4; }
  done
  # --- the fork parent must be FINISHED at exactly the registered step ---
  _require_finished_at "$PARENT_ZIP" "$PARENT_STEP" "fork parent"
  # --- an extension's exploiter target is pinned too (it is the round-1 target, unchanged) ---
  if [ -n "${EXPLOITER_ZIP:-}" ]; then
    grep -q -- "--exploiter $EXPLOITER_ZIP " <<<"$ARGV " || { echo "FATAL: $ARGV_FILE does not target $EXPLOITER_ZIP"; exit 2; }
    _require_finished_at "$EXPLOITER_ZIP" "$EXPLOITER_STEP" "exploiter target"
  fi

  # --- pre-flight, always, even on a real launch ---
  echo "=== [$ARM] checkargs $(date -Is) ==="
  nice -n 15 "$P" -m main.checkargs --argv "$ARGV"
  echo "=== [$ARM] launcher --dry-run $(date -Is) ==="
  # shellcheck disable=SC2086
  nice -n 15 "$P" -m main.launcher --dry-run $ARGV

  if [ "${1:-}" = "--dry-run" ]; then
    echo "=== [$ARM] --dry-run requested: stopping here. Nothing was created. ==="
    exit 0
  fi

  # --- refuse to clobber, refuse to share the GPU ---
  if [ -e "models/$RUN" ]; then
    echo "FATAL: models/$RUN already exists — this script only starts a FRESH fork."
    echo "       Resuming or re-forking is a decision, not a re-run."
    exit 3
  fi
  if pgrep -f "train_rl_agent.py" >/dev/null 2>&1 && [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    echo "FATAL: a train_rl_agent.py process is already running — the GPU is single-tenant."
    pgrep -af "train_rl_agent.py" | cut -c1-200
    exit 5
  fi

  mkdir -p "$LOG_DIR"
  local LOG="$LOG_DIR/${ARM}_${RUN}.log"
  echo "=== [$ARM] LAUNCH $(date -Is) -> $LOG ==="
  # shellcheck disable=SC2086
  nohup "$P" -m main.launcher $ARGV > "$LOG" 2>&1 < /dev/null &
  echo "launcher pid $!"
  echo "watch:  tail -f $LOG"
  echo
  echo "THE FIRST TWO MINUTES ARE THE ONLY TEST OF THE PRELOAD LAYER. Confirm, in the log:"
  printf '%s\n' "${STOP_LIST[@]}"
}
