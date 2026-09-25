#!/usr/bin/env bash
# LAUNCH — RB2, ai_v13_29_popr2_read_loop: the FRESH OFFENSE EXPLOITER that reads B2 (the round-2 LOOP).
#
# 🚨 PREPARED BUT NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session,
#    VERBATIM. It takes the GPU. It CANNOT launch until ai_v13_27_popr2_loop (B2, the round-2 LOOP generalist) has FINISHED at
#    111,280,128 — the script refuses otherwise.
#    Registration: designs/research_state/population_loop_round2_2026-09-24.md (§1, §4.2)
#
# WHAT IT IS. Arm A's recipe TOKEN-EXACT (ai_v13_18_teach5_offense_hidose's recorded
# original_command) with a four-value diff: --run-name, --model and --exploiter both ->
# models/ai_v13_27_popr2_loop/final_model.zip, --steps 103158272 -> 119280128 (a TOTAL: the target's 111,280,128
# + 8,000,000 -> lands 119,341,056, post-fork budget 8,060,928 == A's, RB's and RC's). Named frozen
# --fork-lr 2.5e-4 (1.78x), seed 1001, the same five offense teams, --exploiter-keep-bots at 50 %,
# --eval-battles 100, pin 6eb9c776. gap(RB2) = its pooled greedy vs-target rate - 0.5, read by
# main.best_response_gap. Validated on a STAND-IN target (scripts/argv_RB2_STANDIN.txt:
# ai_v13_24_popr1_read_loop @111,280,128 — an exploiter, not a generalist; finding K-2).
#
# Usage:  bash designs/research_state/measurements/population_loop_r2_2026-09-24/launch_RB2.sh [--dry-run]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=RB2
RUN=ai_v13_29_popr2_read_loop
ARGV_FILE="$HERE/argv_RB2.txt"
PARENT_ZIP=models/ai_v13_27_popr2_loop/final_model.zip
PARENT_STEP=111280128
PREREQ_ZIPS=("$PARENT_ZIP")
STOP_LIST=(
"  - 🎯 [MULTI-SPECIALIST] trainee pinned to 5 teams: 9eb3abdc52876a63, ac17a9dde5, a185b2d193,"
"    9ba039ba8a, b904dbe059"
"  - 🎚️ [ForkLR] FORK from a checkpoint outside models/ai_v13_29_popr2_read_loop — pinning LR to 2.50e-04 and FREEZING the KL controller"
"  - exploiter target: models/ai_v13_27_popr2_loop/final_model.zip | bots mixed in 50%"
"  - 🧭 [MATCHUP <hash>] — a NEW hash, EXPECTED (the exploiter-target path is hashed; finding P-4). Record it."
"  - NO 🐴 [STABLE] line (exploiter mode; the target's stable opponents are not inherited)"
"  - no [ModelVersion] FATAL. (Do NOT wait for 'Round-trip smoke test PASSED': a real launch never prints it; finding P-2)"
"  - after the FIRST eval (~112M): eval_results.jsonl carries externals.ext_ai_v13_27_popr2_loop with counts [w, 100]"
)
# shellcheck source=scripts/_launch_common.sh
source "$HERE/scripts/_launch_common.sh"
launch_main "$@"
