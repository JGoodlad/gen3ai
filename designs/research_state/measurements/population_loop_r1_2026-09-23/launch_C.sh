#!/usr/bin/env bash
# LAUNCH — C, ai_v13_23_popr1_ctrl: THE CONTROL ARM of population-loop round 1.
#
# 🚨 PREPARED BUT NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session,
#    VERBATIM. It takes the GPU. Registration: designs/research_state/population_loop_round1_2026-09-23.md
#
# WHAT IT IS. B's argv with exactly ONE value changed: --stable-opponent-selfplay-share 0.4 -> 0.0
# (plus --run-name). The two offense specialists are still LOADED by every worker and still PLAYED
# in every eval cycle (so C's own eval series reads its rate against them — the manipulation-check
# comparator — and eval load and the per-episode stable-coin RNG draw match B's), but the coin
# `rng.random() < 0.0` is never true, so C never TRAINS against them (wrappers.py
# _pick_challenge_opponent, identical at pin 6eb9c776). Pool 0.90 / bots 0.10 of episodes — the
# plateau block's own mix. Same parent, dose (0.20x frozen), budget, seed 1001, pin 6eb9c776.
#
# Usage:  bash designs/research_state/measurements/population_loop_r1_2026-09-23/launch_C.sh [--dry-run]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=C
RUN=ai_v13_23_popr1_ctrl
ARGV_FILE="$HERE/argv_C.txt"
PREREQ_ZIPS=(models/ai_v13_12_plateau/final_model.zip
             models/ai_v13_18_teach5_offense_hidose/final_model.zip
             models/ai_v13_13_exploit5_offense/final_model.zip)
STOP_LIST=(
"  - 🎚️ [ForkLR] ... pinning LR to 2.80e-05 and FREEZING the KL controller"
"  - 🌱 [SELFPLAY] [pool] seeded 20 snapshots + metadata from ai_v13_12_plateau"
"  - 🐴 [STABLE] 2 cross-run opponent(s) ... training ≤0% of self-play"
"  - 🧭 [MATCHUP ef5242cffd] and NO drift line — MUST EQUAL B's (and G0's). Different: STOP."
"  - [ModelVersion] Round-trip smoke test PASSED, and no [ModelVersion] FATAL"
"  - after the FIRST eval: tb train/stable_fraction = 0.00 EXACTLY and train/selfplay_fraction = 0.90."
"    ANY non-zero stable_fraction means the control is contaminated: STOP."
)
# shellcheck source=scripts/_launch_common.sh
source "$HERE/scripts/_launch_common.sh"
launch_main "$@"
