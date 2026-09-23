# Sourced by ../launch_<arm>.sh — the shared pre-flight + launch for round 1 of the population loop.
# Not executable on its own. Each caller sets: ARM, RUN, ARGV_FILE, PREREQ_ZIPS (array), and
# optionally TARGET_ZIP + TARGET_STEP (readers), then calls `launch_main "$@"`.
#
# 🚨 PREPARED, NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session, VERBATIM.
set -euo pipefail

REPO=/home/goodlad/dev/gen3ai
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
LOG_DIR="${LOG_DIR:-/home/goodlad/.claude/jobs/popr1_2026-09-23/launch}"

launch_main() {
  export PYTHONPATH="${PYTHONPATH:-}:$REPO/src"
  cd "$REPO"
  [ -f "$ARGV_FILE" ] || { echo "FATAL: no argv at $ARGV_FILE"; exit 2; }
  ARGV="$(cat "$ARGV_FILE")"
  grep -q -- "--run-name $RUN " <<<"$ARGV " || { echo "FATAL: $ARGV_FILE does not name --run-name $RUN"; exit 2; }

  # --- inputs must exist, as FILES (a bare run dir would mean 'last snapshot') ---
  for z in "${PREREQ_ZIPS[@]}"; do
    [ -f "$z" ] || { echo "FATAL: prerequisite $z does not exist — this arm cannot launch yet."; exit 4; }
  done
  # --- a reader's target must be a FINISHED generalist at exactly the registered step, so the
  #     reader's post-fork budget is 8,060,928 — the value main.best_response_gap matches on ---
  if [ -n "${TARGET_ZIP:-}" ]; then
    local tdir; tdir="$(dirname "$TARGET_ZIP")"
    local got
    got="$("$P" -c "import json,sys; print(json.load(open('$tdir/metadata.json')).get('num_timesteps'))")"
    [ "$got" = "$TARGET_STEP" ] || {
      echo "FATAL: $tdir num_timesteps=$got, registered $TARGET_STEP. The reader budget would not"
      echo "       match A's 8,060,928 and best_response_gap would REFUSE it. Recompute --steps as"
      echo "       <target final> + 8,000,000 and record the change in the ledger BEFORE launching."
      exit 4; }
    grep -q "Training complete" "$tdir/launcher_child.full.log" 2>/dev/null || {
      echo "FATAL: $tdir/launcher_child.full.log has no 'Training complete' — the target has not finished."
      exit 4; }
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
