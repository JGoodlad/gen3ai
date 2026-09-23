#!/usr/bin/env bash
# LAUNCH — RC, ai_v13_25_popr1_read_ctrl: the FRESH OFFENSE EXPLOITER that reads C (the CONTROL).
#
# 🚨 PREPARED BUT NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session,
#    VERBATIM. It takes the GPU. It CANNOT launch until ai_v13_23_popr1_ctrl has FINISHED at
#    103,219,200 — the script refuses otherwise.
#    Registration: designs/research_state/population_loop_round1_2026-09-23.md
#
# WHAT IT IS. Arm A's recipe TOKEN-EXACT (ai_v13_18_teach5_offense_hidose's recorded
# original_command) with a four-value diff: --run-name, --model and --exploiter both ->
# models/ai_v13_23_popr1_ctrl/final_model.zip, --steps 103158272 -> 111219200 (a TOTAL: the target's
# 103,219,200 + 8,000,000 -> lands 111,280,128, post-fork budget 8,060,928 == A's). Named frozen
# --fork-lr 2.5e-4 (1.78x), seed 1001, the same five offense teams, --exploiter-keep-bots at 50 %,
# pin 6eb9c776. gap(RC) = its pooled greedy vs-target rate - 0.5, read by main.best_response_gap.
# Validated on a STAND-IN target (scripts/argv_RC_STANDIN.txt, ai_v13_16_teach5_offense_dist @103,219,200).
#
# Usage:  bash designs/research_state/measurements/population_loop_r1_2026-09-23/launch_RC.sh [--dry-run]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=RC
RUN=ai_v13_25_popr1_read_ctrl
ARGV_FILE="$HERE/argv_RC.txt"
TARGET_ZIP=models/ai_v13_23_popr1_ctrl/final_model.zip
TARGET_STEP=103219200
PREREQ_ZIPS=("$TARGET_ZIP")
STOP_LIST=(
"  - 🎯 [MULTI-SPECIALIST] trainee pinned to 5 teams: 9eb3abdc52876a63, ac17a9dde5, a185b2d193,"
"    9ba039ba8a, b904dbe059"
"  - 🎚️ [ForkLR] ... pinning LR to 2.50e-04 and FREEZING the KL controller"
"  - exploiter target: models/ai_v13_23_popr1_ctrl/final_model.zip | bots mixed in 50%"
"  - 🧭 [MATCHUP <hash>] — a NEW hash, EXPECTED: the exploiter-target path is part of the hash"
"    (matchup_spec.to_dict), so RB, RC and A (80dea7c93b) all differ. Record it."
"  - NO 🐴 [STABLE] line (exploiter mode; stable opponents are not inherited from the target)"
"  - [ModelVersion] Round-trip smoke test PASSED, and no [ModelVersion] FATAL"
)
# shellcheck source=scripts/_launch_common.sh
source "$HERE/scripts/_launch_common.sh"
launch_main "$@"
