#!/usr/bin/env bash
# LAUNCH — the NEW LINEAGE's base run, ai_v14_01_base: a FRESH run from scratch on the clean-input
# boundary (b0a28b5b; obs 2761, model v121), the last production lineage's fresh-root recipe
# (ai_v13_02_flywheel_winprob's recorded original_command) with four registered moves:
# --run-name, --pin-commit, + --arch production, + --obs-source core.
#
# 🚨 PREPARED BUT NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session,
#    VERBATIM, after the orchestrator's go. It takes the GPU for ~38 h at ~550 fps.
#    Registration: designs/research_state/new_lineage_2026-09-26.md (§2 the recipe, §4 the watch list)
#
# Usage:  bash designs/research_state/measurements/new_lineage_2026-09-26/launch_base.sh [--dry-run]
#   --dry-run : checkargs + launcher --dry-run, then stop. Creates nothing. No argument LAUNCHES.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/home/goodlad/dev/gen3ai
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
RUN=ai_v14_01_base
ARGV_FILE="$HERE/argv_base.txt"
LOG_DIR="${LOG_DIR:-/home/goodlad/.claude/jobs/new_lineage_2026-09-26/launch}"

export PYTHONPATH="${PYTHONPATH:-}:$REPO/src"
cd "$REPO"
[ -f "$ARGV_FILE" ] || { echo "FATAL: no argv at $ARGV_FILE"; exit 2; }
ARGV="$(cat "$ARGV_FILE")"
grep -q -- "--run-name $RUN " <<<"$ARGV " || { echo "FATAL: $ARGV_FILE does not name --run-name $RUN"; exit 2; }
grep -q -- "--pin-commit " <<<"$ARGV" || { echo "FATAL: $ARGV_FILE has no --pin-commit"; exit 2; }
grep -q -- "--model " <<<"$ARGV" && { echo "FATAL: the base run is FRESH; $ARGV_FILE names --model"; exit 2; }

echo "=== [base] checkargs $(date -Is) ==="
nice -n 15 "$P" -m main.checkargs --argv "$ARGV"
echo "=== [base] launcher --dry-run $(date -Is) ==="
# shellcheck disable=SC2086
nice -n 15 "$P" -m main.launcher --dry-run $ARGV

if [ "${1:-}" = "--dry-run" ]; then
  echo "=== [base] --dry-run requested: stopping here. Nothing was created. ==="
  exit 0
fi

# --- refuse to clobber, refuse to share the GPU ---
if [ -e "models/$RUN" ]; then
  echo "FATAL: models/$RUN already exists — this script only starts a FRESH run."
  echo "       A restart / resume is a decision, not a re-run (and is DESTRUCTIVE if wrong)."
  exit 3
fi
if pgrep -f "[t]rain_rl_agent.py" >/dev/null 2>&1 && [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
  echo "FATAL: a trainer process is already running — the GPU is single-tenant."
  echo "       (The core-obs burn-in ai_v13_33_core_burnin may still hold it: ending it is the orchestrator's call.)"
  pgrep -af "[t]rain_rl_agent.py" | cut -c1-200
  exit 5
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/base_${RUN}.log"
echo "=== [base] LAUNCH $(date -Is) -> $LOG ==="
# shellcheck disable=SC2086
nohup "$P" -m main.launcher $ARGV > "$LOG" 2>&1 < /dev/null &
echo "launcher pid $!"
echo "watch:  tail -f $LOG"
echo
echo "THE FIRST TWO MINUTES ARE THE ONLY TEST OF THE PRELOAD LAYER. Confirm, in the log (registration §4.1):"
cat <<'EOF'
  - role FRESH, run dir models/ai_v14_01_base, pin 8d07051a…, obs source core, transport rust
  - arch: --arch production applied; "✓ every ARCH-surface key matches the production mirror"
  - no [ModelVersion] FATAL (do NOT wait for "Round-trip smoke test PASSED": a real launch never prints it)
  - the observation dimension line reads 2761 and the arch signature gen3_event_record_v2
  - NO 🐴 [STABLE] line (no stable opponents), NO [ForkLR] pin (fresh), self-play pool EMPTY at start
  - no CoreObsMismatch / "__OBS__ frame nobody asked for" anywhere (a single one = STOP, §4.2)
  - first rollout lands; time/fps in TensorBoard within ~20 % of the burn-in's ~550 once compile settles
  - after the FIRST eval (2,000,000): eval_results.jsonl has a row with sentinel_regime greedy + symmetric
EOF
