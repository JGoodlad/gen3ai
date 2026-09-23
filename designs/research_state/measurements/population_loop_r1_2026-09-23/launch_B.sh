#!/usr/bin/env bash
# LAUNCH — B, ai_v13_22_popr1_loop: THE LOOP ARM of population-loop round 1.
#
# 🚨 PREPARED BUT NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session,
#    VERBATIM. It takes the GPU. Registration: designs/research_state/population_loop_round1_2026-09-23.md
#
# WHAT IT IS. The plateau parent G0 (models/ai_v13_12_plateau/final_model.zip @95,158,272) continued
# +8,000,000 steps (--steps 103158272 is a TOTAL -> lands 103,219,200) from ai_v13_12_plateau's OWN
# recorded original_command, at G0's OWN named frozen dose (--fork-lr 2.8e-5 --fork-lr-freeze =
# 4.272e-9 = 0.20x v8), NO distillation (--distill-coef 0.0), with the two OFFENSE exploiters of G0
# as stable opponents, each piloting its own five pinned teams:
#     A    = models/ai_v13_18_teach5_offense_hidose/final_model.zip  (named frozen 1.78x, gap +14.00 pp)
#     A13  = models/ai_v13_13_exploit5_offense/final_model.zip        (unnamed 0.39x, gap +15.75 pp)
# at --stable-opponent-selfplay-share 0.4 (= 0.36 of ALL episodes at self_play_fraction 0.90;
# ~0.18 per specialist), --stable-opponent-pfsp (loss-weighted within the slice) and
# --stable-opponent-mastered-wr 1.01 (retirement disabled: a win rate can never reach 1.01).
# Diff vs the recorded plateau argv (scripts/build_argvs.py prints it): --run-name, --model, --steps,
# --stable-opponent-selfplay-share 0.2->0.4, --stable-opponent-mastered-wr 0.8->1.01,
# +--stable-opponent-pfsp, +--stable-opponents. Seed 1001, pin 6eb9c776, --team-block-episodes 1.
#
# Usage:  bash designs/research_state/measurements/population_loop_r1_2026-09-23/launch_B.sh [--dry-run]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM=B
RUN=ai_v13_22_popr1_loop
ARGV_FILE="$HERE/argv_B.txt"
PREREQ_ZIPS=(models/ai_v13_12_plateau/final_model.zip
             models/ai_v13_18_teach5_offense_hidose/final_model.zip
             models/ai_v13_13_exploit5_offense/final_model.zip)
STOP_LIST=(
"  - 🎚️ [ForkLR] ... pinning LR to 2.80e-05 and FREEZING the KL controller"
"  - 🌱 [SELFPLAY] [pool] seeded 20 snapshots + metadata from ai_v13_12_plateau"
"  - 🐴 [STABLE] 2 cross-run opponent(s): ext_ai_v13_18_teach5_offense_hidose,"
"    ext_ai_v13_13_exploit5_offense — training ≤40% of self-play until mastered (win_rate ≥ 101%)"
"  - 🧭 [MATCHUP ef5242cffd] and NO drift line — G0's own era (the hash covers teams, mix kind and"
"    exploiter target, NOT stable opponents or the share: matchup_spec.to_dict). C prints the same."
"  - [ModelVersion] Round-trip smoke test PASSED, and no [ModelVersion] FATAL"
"  - after the FIRST eval (~+0.84M): tb train/stable_fraction = 0.36, train/selfplay_fraction = 0.54,"
"    and eval rows carry ext_ai_v13_18_teach5_offense_hidose + ext_ai_v13_13_exploit5_offense."
"    stable_fraction 0.18 means the share did not take (0.2 default): STOP."
)
# shellcheck source=scripts/_launch_common.sh
source "$HERE/scripts/_launch_common.sh"
launch_main "$@"
