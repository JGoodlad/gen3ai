#!/usr/bin/env bash
# LAUNCH — RB+, ai_v13_31_popr1_read_loop_ext: the CONVERGENCE SIDE-CHECK extension of RB (ai_v13_24_popr1_read_loop), +50 % budget.
#
# 🚨 PREPARED BUT NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session,
#    VERBATIM. It takes the GPU (~2 GPU-h). Run RB+ and RC+ ADJACENT (matched contention), in either
#    order, anywhere in the queue; never one without the other.
#    Registration: designs/research_state/population_loop_round2_2026-09-24.md §4.4 — a DESCRIPTOR,
#    NOT a re-verdict of round 1.
#
# WHAT IT IS. RB's OWN recorded original_command TOKEN-EXACT with a three-value diff: --run-name,
# --model -> models/ai_v13_24_popr1_read_loop/final_model.zip (a FORK of the reader's final into a NEW run dir — NEVER a
# resume of ai_v13_24_popr1_read_loop itself, which would move its recorded budget and void every round-1 read), and
# --steps 111219200 -> 115280128 (a TOTAL: 111,280,128 + 4,000,000 -> lands 115,310,592 = +4,030,464,
# exactly +50 % of 8,060,928). --exploiter is UNCHANGED: models/ai_v13_22_popr1_loop/final_model.zip (B, the round-1 loop generalist).
# --fork-lr 2.5e-4 --fork-lr-freeze (1.78x, the parent's own frozen value), seed 1001, the same five
# offense teams, --exploiter-keep-bots at 50 %, --eval-battles 100, pin 6eb9c776. No
# --warmstart-consensus, so no warm-start runs on the fork. Validated against its REAL parent.
#
# Usage:  bash designs/research_state/measurements/population_loop_r2_2026-09-24/launch_RBX.sh [--dry-run]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=RBX
RUN=ai_v13_31_popr1_read_loop_ext
ARGV_FILE="$HERE/argv_RBX.txt"
PARENT_ZIP=models/ai_v13_24_popr1_read_loop/final_model.zip
PARENT_STEP=111280128
EXPLOITER_ZIP=models/ai_v13_22_popr1_loop/final_model.zip
EXPLOITER_STEP=103219200
PREREQ_ZIPS=("$PARENT_ZIP" "$EXPLOITER_ZIP")
STOP_LIST=(
"  - 🎯 [MULTI-SPECIALIST] trainee pinned to 5 teams: 9eb3abdc52876a63, ac17a9dde5, a185b2d193,"
"    9ba039ba8a, b904dbe059"
"  - 🎚️ [ForkLR] FORK from a checkpoint outside models/ai_v13_31_popr1_read_loop_ext — pinning LR to 2.50e-04 and FREEZING the KL controller"
"  - exploiter target: models/ai_v13_22_popr1_loop/final_model.zip | bots mixed in 50%"
"  - 🧭 [MATCHUP <hash>] — record it; the same target path and teams as RB, so its hash is EXPECTED to match RB's (see README)"
"  - NO 🐴 [STABLE] line; NO 🌱 [WARMSTART] line"
"  - no [ModelVersion] FATAL. (Do NOT wait for 'Round-trip smoke test PASSED': a real launch never prints it; finding P-2)"
"  - after the FIRST eval (~112M): a FRESH eval_results.jsonl in models/ai_v13_31_popr1_read_loop_ext (it must NOT contain RB's"
"    104M-110M rows) carrying externals.ext_ai_v13_22_popr1_loop with counts [w, 100]"
)
# shellcheck source=scripts/_launch_common.sh
source "$HERE/scripts/_launch_common.sh"
launch_main "$@"
