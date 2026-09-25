#!/usr/bin/env bash
# LAUNCH — RC+, ai_v13_32_popr1_read_ctrl_ext: the CONVERGENCE SIDE-CHECK extension of RC (ai_v13_25_popr1_read_ctrl), +50 % budget.
#
# 🚨 PREPARED BUT NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session,
#    VERBATIM. It takes the GPU (~2 GPU-h). Run RB+ and RC+ ADJACENT (matched contention), in either
#    order, anywhere in the queue; never one without the other.
#    Registration: designs/research_state/population_loop_round2_2026-09-24.md §4.4 — a DESCRIPTOR,
#    NOT a re-verdict of round 1.
#
# WHAT IT IS. RC's OWN recorded original_command TOKEN-EXACT with a three-value diff: --run-name,
# --model -> models/ai_v13_25_popr1_read_ctrl/final_model.zip (a FORK of the reader's final into a NEW run dir — NEVER a
# resume of ai_v13_25_popr1_read_ctrl itself, which would move its recorded budget and void every round-1 read), and
# --steps 111219200 -> 115280128 (a TOTAL: 111,280,128 + 4,000,000 -> lands 115,310,592 = +4,030,464,
# exactly +50 % of 8,060,928). --exploiter is UNCHANGED: models/ai_v13_23_popr1_ctrl/final_model.zip (C, the round-1 control).
# --fork-lr 2.5e-4 --fork-lr-freeze (1.78x, the parent's own frozen value), seed 1001, the same five
# offense teams, --exploiter-keep-bots at 50 %, --eval-battles 100, pin 6eb9c776. No
# --warmstart-consensus, so no warm-start runs on the fork. Validated against its REAL parent.
#
# Usage:  bash designs/research_state/measurements/population_loop_r2_2026-09-24/launch_RCX.sh [--dry-run]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=RCX
RUN=ai_v13_32_popr1_read_ctrl_ext
ARGV_FILE="$HERE/argv_RCX.txt"
PARENT_ZIP=models/ai_v13_25_popr1_read_ctrl/final_model.zip
PARENT_STEP=111280128
EXPLOITER_ZIP=models/ai_v13_23_popr1_ctrl/final_model.zip
EXPLOITER_STEP=103219200
PREREQ_ZIPS=("$PARENT_ZIP" "$EXPLOITER_ZIP")
STOP_LIST=(
"  - 🎯 [MULTI-SPECIALIST] trainee pinned to 5 teams: 9eb3abdc52876a63, ac17a9dde5, a185b2d193,"
"    9ba039ba8a, b904dbe059"
"  - 🎚️ [ForkLR] FORK from a checkpoint outside models/ai_v13_32_popr1_read_ctrl_ext — pinning LR to 2.50e-04 and FREEZING the KL controller"
"  - exploiter target: models/ai_v13_23_popr1_ctrl/final_model.zip | bots mixed in 50%"
"  - 🧭 [MATCHUP <hash>] — record it; the same target path and teams as RC, so its hash is EXPECTED to match RC's (see README)"
"  - NO 🐴 [STABLE] line; NO 🌱 [WARMSTART] line"
"  - no [ModelVersion] FATAL. (Do NOT wait for 'Round-trip smoke test PASSED': a real launch never prints it; finding P-2)"
"  - after the FIRST eval (~112M): a FRESH eval_results.jsonl in models/ai_v13_32_popr1_read_ctrl_ext (it must NOT contain RC's"
"    104M-110M rows) carrying externals.ext_ai_v13_23_popr1_ctrl with counts [w, 100]"
)
# shellcheck source=scripts/_launch_common.sh
source "$HERE/scripts/_launch_common.sh"
launch_main "$@"
