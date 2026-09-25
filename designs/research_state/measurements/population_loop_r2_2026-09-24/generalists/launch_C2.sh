#!/usr/bin/env bash
# LAUNCH — C2, ai_v13_28_popr2_ctrl: population-loop ROUND 2 (orchestrator spec 2026-09-24, branch N+ of record a6637f29).
# Round 1's argv_C.txt with ONLY --run-name, --model (ai_v13_23_popr1_ctrl @103,219,200), --steps 111219200
# (= parent final + 8,000,000, a TOTAL) and RB added to --stable-opponents changed. Pin 6eb9c776, seed 1001.
ARM=C2
RUN=ai_v13_28_popr2_ctrl
ARGV_FILE=/home/goodlad/.claude/jobs/popr2_2026-09-24/argv_C2.txt
LOG_DIR=/home/goodlad/.claude/jobs/popr2_2026-09-24/launch
PREREQ_ZIPS=(models/ai_v13_23_popr1_ctrl/final_model.zip
             models/ai_v13_18_teach5_offense_hidose/final_model.zip
             models/ai_v13_13_exploit5_offense/final_model.zip
             models/ai_v13_24_popr1_read_loop/final_model.zip)
STOP_LIST=(
"  - 🎚️ [ForkLR] ... pinning LR to 2.80e-05 and FREEZING the KL controller"
"  - 🌱 [SELFPLAY] [pool] seeded 20 snapshots + metadata from ai_v13_23_popr1_ctrl"
"  - 🐴 [STABLE] 3 cross-run opponent(s): ext_ai_v13_18_teach5_offense_hidose, ext_ai_v13_13_exploit5_offense,"
"    ext_ai_v13_24_popr1_read_loop — each [pilots ITS OWN pin: 9eb3abdc52876a63.txt]; training ≤0% of self-play"
"  - 🧭 [MATCHUP ef5242cffd] and NO drift line (the hash omits stable opponents and the share)"
"  - [ModelVersion] Round-trip smoke test PASSED, and no [ModelVersion] FATAL"
"  - after the FIRST eval: tb train/stable_fraction = 0.00; eval rows carry all THREE ext_ opponents"
)
source /home/goodlad/dev/gen3ai/designs/research_state/measurements/population_loop_r1_2026-09-23/scripts/_launch_common.sh
launch_main "$@"
