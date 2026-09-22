#!/usr/bin/env bash
# LAUNCH — ai_v13_18_teach5_offense_hidose, THE HIGH-DOSE OFFENSE TEACHER.
#
# 🚨 THIS SCRIPT WAS PREPARED BUT NEVER EXECUTED by the session that wrote it. It is for the
#    TRAINING RUN session to run VERBATIM. It takes the GPU.
#
# WHAT IT IS. `ai_v13_13_exploit5_offense`'s own resolved argv (its metadata.json
# `original_command`), with a FOUR-TOKEN diff:
#     --run-name  ai_v13_13_exploit5_offense -> ai_v13_18_teach5_offense_hidose
#     + --fork-lr 2.5e-4 --fork-lr-freeze
# Same plateau-parent fork, same five offense teams, same +8,000,000 steps (--steps 103158272 is a
# TOTAL and the parent sits at 95,158,272), same seed 1001, same --team-block-episodes 1, same
# --exploiter models/ai_v13_12_plateau/final_model.zip --exploiter-keep-bots, no --self-play.
#
# WHY 2.5e-4. Every run in this lineage runs --batch-size 2048 --grad-accum-steps 32 --n-epochs 10,
# so updates/step = 10 / 65,536 = 1.52588e-4 and
#     2.5e-4 * 1.52588e-4 = 3.8147e-8 = 1.78x the v8 reference (2.145e-8)
# which is the era-1 exploiters' measured dose to the digit. --fork-lr-freeze is REQUIRED: without
# it the KL controller runs live and the lr is not stationary within the arm (that is exactly how
# era-2 drifted 2.8e-5 -> 8.36e-5, ledger 2026-09-20 `7afa2b34`).
#
# THE FULL REGISTRATION, INCLUDING THE READ AND THE BRANCHES: ./PREDICTION.md — read it first.
#
# Usage:   bash designs/research_state/measurements/hidose_teacher_2026-09-21/launch.sh [--dry-run]
#          (--dry-run resolves and creates NOTHING; no argument LAUNCHES)
set -euo pipefail

REPO=/home/goodlad/dev/gen3ai
ARGV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/argv_hidose.txt"
LOG=/home/goodlad/.claude/jobs/1046b1d6/tmp/launch/teach5_hidose.log
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3

export PYTHONPATH="${PYTHONPATH:-}:$REPO/src"
cd "$REPO"

[ -f "$ARGV_FILE" ] || { echo "FATAL: no argv at $ARGV_FILE"; exit 2; }
ARGV="$(cat "$ARGV_FILE")"

# --- pre-flight, always, even on a real launch: the argv must still parse and still be the arch ---
echo "=== checkargs $(date -Is) ==="
nice -n 15 $P -m main.checkargs --argv "$ARGV"

echo "=== launcher --dry-run $(date -Is) ==="
nice -n 15 $P -m main.launcher --dry-run $ARGV

if [ "${1:-}" = "--dry-run" ]; then
  echo "=== --dry-run requested: stopping here. Nothing was created. ==="
  exit 0
fi

# --- refuse to clobber ---
if [ -e "models/ai_v13_18_teach5_offense_hidose" ]; then
  echo "FATAL: models/ai_v13_18_teach5_offense_hidose already exists — this script only starts a"
  echo "       FRESH fork. Resuming or re-forking is a decision, not a re-run."
  exit 3
fi

mkdir -p "$(dirname "$LOG")"
echo "=== LAUNCH $(date -Is) -> $LOG ==="
nohup $P -m main.launcher $ARGV > "$LOG" 2>&1 < /dev/null &
echo "launcher pid $!"
echo "watch:  tail -f $LOG"
echo
echo "THE FIRST TWO MINUTES ARE THE ONLY TEST OF THE PRELOAD LAYER. Confirm, in the log:"
echo "  - 🎯 [MULTI-SPECIALIST] trainee pinned to 5 teams: 9eb3abdc52876a63, ac17a9dde5,"
echo "    a185b2d193, 9ba039ba8a, b904dbe059"
echo "  - 🎚️ [ForkLR] ... pinning LR to 2.50e-04 and FREEZING the KL controller"
echo "  - exploiter target: models/ai_v13_12_plateau/final_model.zip | bots mixed in 50%"
echo "  - 🧭 [MATCHUP 80dea7c93b]  (ai_v13_13's own era — the SAME five teams, so the SAME hash;"
echo "    a DIFFERENT hash means the team set is not the one that was registered: STOP)"
echo "  - [ModelVersion] Round-trip smoke test PASSED, and no [ModelVersion] FATAL"
echo "  - no [Untaught] FATAL (gen3_untaught_teacher_guard_v1 — none of these five is untaught-8)"
echo
echo "Then, when it finishes, read it with ./PREDICTION.md sec 3 — the admission cell AND the"
echo "collateral untaught-8 cell, both of them, whichever way the primary goes."
