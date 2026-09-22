#!/usr/bin/env bash
# THE ADMISSION GATE for ai_v13_18_teach5_offense_hidose — its own five offense teams, against
# THE PLATEAU PARENT, played by a fixed THIRD PARTY's opponent.
#
# REUSED VERBATIM from the 2026-09-21 gate (~/.claude/jobs/1046b1d6/tmp/admission/run_admission.sh),
# which reused split_lossoff_read_2026-09-20/scripts/harness/run_slice.sh: same engine, same fixed
# third-party opponent (`untaught_meter_opponent` = ai_v9_29_rev1_0823@24M -- a sentinel would be
# the trainee's OWN snapshot and so would differ between refs), same --games-per-team 800, same
# --seed 0, concurrency 1, nice 15, CPU-only, MAIN checkout. THE ONLY CHANGE IS THE TEACHER REF.
#
# BOTH refs in ONE invocation so the teacher and the plateau parent see identical teams and
# identical dice (CRN). The team manifest is the offense set in REGISTERED order -- the order IS
# the seed offset, so it is reused byte-identically and must not be re-sorted.
#
# THE RULE (../../PREDICTION.md sec 3.3, registered before the first battle): OUTSIDE iff
# |delta| > floor AND the 95% CI excludes the floor point; ADMITTED iff OUTSIDE *and* ABOVE the
# parent, at BOTH floors 4.75 / 8.50 pp. Not admitted => report, do not fold.
set -u
export PYTHONPATH=/home/goodlad/dev/gen3ai/src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$D/../../out}"
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai

[ -f "$OUT/admission_hidose.json" ] && { echo "SKIP hidose"; exit 0; }
echo "=== ADMISSION hidose  $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
nice -n 15 $P -m main.untaught_meter \
  teacher=models/ai_v13_18_teach5_offense_hidose/final_model.zip \
  plateau_parent=models/ai_v13_12_plateau/final_model.zip \
  --teams "$D/teams_offense.json" --opponent untaught_meter_opponent \
  --games-per-team 800 --workers 5 --seed 0 \
  --json "$OUT/admission_hidose.json" --md "$OUT/admission_hidose.md" 2>&1
echo "=== EXIT $? hidose  $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg) ==="
