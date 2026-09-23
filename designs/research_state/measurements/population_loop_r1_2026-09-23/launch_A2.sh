#!/usr/bin/env bash
# LAUNCH — A2, ai_v13_26_popr0_exploit5_offense_s1002: arm A's SEED REPLICATE against G0.
#
# 🚨 PREPARED BUT NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session,
#    VERBATIM. It takes the GPU. Registration: designs/research_state/population_loop_round1_2026-09-23.md
#
# WHAT IT IS. ai_v13_18_teach5_offense_hidose's (= arm A's) recorded original_command with a
# two-value diff: --run-name, --seed 1001 -> 1002. Same G0 fork and target, same five offense teams,
# same +8,000,000, same named frozen --fork-lr 2.5e-4 (3.815e-8 = 1.78x), same --exploiter-keep-bots.
# Its job: |gap(A) - gap(A2)| is the reader instrument's RUN-level floor at a fixed target, the one
# variance component the per-reader Newcombe interval does not contain. Depends on nothing.
#
# Usage:  bash designs/research_state/measurements/population_loop_r1_2026-09-23/launch_A2.sh [--dry-run]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=A2
RUN=ai_v13_26_popr0_exploit5_offense_s1002
ARGV_FILE="$HERE/argv_A2.txt"
PREREQ_ZIPS=(models/ai_v13_12_plateau/final_model.zip)
STOP_LIST=(
"  - 🎯 [MULTI-SPECIALIST] trainee pinned to 5 teams: 9eb3abdc52876a63, ac17a9dde5, a185b2d193,"
"    9ba039ba8a, b904dbe059"
"  - 🎚️ [ForkLR] ... pinning LR to 2.50e-04 and FREEZING the KL controller"
"  - exploiter target: models/ai_v13_12_plateau/final_model.zip | bots mixed in 50%"
"  - 🧭 [MATCHUP 80dea7c93b]  (A's own era: same five teams => same hash; different: STOP)"
"  - [ModelVersion] Round-trip smoke test PASSED, and no [ModelVersion] FATAL"
)
# shellcheck source=scripts/_launch_common.sh
source "$HERE/scripts/_launch_common.sh"
launch_main "$@"
