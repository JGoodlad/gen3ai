#!/usr/bin/env bash
# VALIDATE BY EXECUTING — the LEARNER BATTERY's four arms on the STAND-IN parent (N0's latest periodic
# checkpoint, N0 being unfinished): build_argvs --standin, then per arm checkargs + launcher --dry-run +
# `STANDIN=1 launch_arm.sh <ARM> --dry-run`; then the model-free readers on N0 (mechanics checks).
# Creates NOTHING under models/ (asserted before and after).
#
#   bash designs/research_state/measurements/learner_battery_2026-09-26/scripts/validate_all.sh \
#        [<standin zip> <standin step> <standin D_g>]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D="$(dirname "$HERE")"
SRC="$(cd "$D/../../../.." && pwd)/src"
OUT="$D/validation"
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
ZIP="${1:-models/ai_v14_01_base/checkpoints/checkpoint_4800000_steps.zip}"
STEP="${2:-4800000}"
DG="${3:-4.3e-04}"
export PYTHONPATH="${PYTHONPATH:-}:$SRC"
mkdir -p "$OUT"
cd /home/goodlad/dev/gen3ai
before="$(ls models | wc -l)"
for r in ai_v14_02_lbat_ctrl ai_v14_03_lbat_e5 ai_v14_04_lbat_t32 ai_v14_05_lbat_l95; do
  [ -e "models/$r" ] && { echo "FATAL: models/$r exists — validation must not run against a live arm"; exit 3; }
done
rcs=0
echo "=== build_argvs --standin ($ZIP @ $STEP, D_g $DG) + --final (placeholder) ($(date -Is))"
"$P" "$HERE/build_argvs.py" --standin --parent "$ZIP" --parent-step "$STEP" --dg "$DG" > "$OUT/build_argvs_standin.txt" 2>&1; r=$?; rcs=$((rcs+r)); echo "    rc=$r"
"$P" "$HERE/build_argvs.py" --final > "$OUT/build_argvs_final_placeholder.txt" 2>&1; r=$?; rcs=$((rcs+r)); echo "    rc=$r"
for A in C E5 T32 L95; do
  ARGV="$(cat "$HERE/argv_${A}_STANDIN.txt")"
  echo "=== $A: checkargs ($(date -Is))"
  nice -n 15 "$P" -m main.checkargs --argv "$ARGV" > "$OUT/checkargs_$A.txt" 2>&1; r=$?; rcs=$((rcs+r)); echo "    rc=$r"
  echo "=== $A: launcher --dry-run ($(date -Is))"
  # shellcheck disable=SC2086
  nice -n 15 "$P" -m main.launcher --dry-run $ARGV > "$OUT/dryrun_$A.txt" 2>&1; r=$?; rcs=$((rcs+r)); echo "    rc=$r"
  echo "=== $A: STANDIN=1 launch_arm.sh --dry-run ($(date -Is))"
  STANDIN=1 bash "$D/launch_arm.sh" "$A" --dry-run > "$OUT/launchsh_$A.txt" 2>&1; r=$?; rcs=$((rcs+r)); echo "    rc=$r"
done
echo "=== the committed placeholder argv must REFUSE ($(date -Is))"
bash "$D/launch_arm.sh" C --dry-run > "$OUT/launchsh_C_placeholder_refusal.txt" 2>&1; r=$?
echo "    rc=$r (expected 2)"; [ "$r" = "2" ] || rcs=$((rcs+1))
echo "=== STANDIN without --dry-run must REFUSE"
STANDIN=1 bash "$D/launch_arm.sh" C > "$OUT/launchsh_standin_nodryrun_refusal.txt" 2>&1; r=$?
echo "    rc=$r (expected 2)"; [ "$r" = "2" ] || rcs=$((rcs+1))
echo "=== readers: speed_read on N0 (identity pair; bots-only window and self-play window), power table"
"$P" "$HERE/speed_read.py" --fork-step 0 --futility X C=ai_v14_01_base X=ai_v14_01_base > "$OUT/speed_read_standin_N0_from_0.txt" 2>&1; r=$?; rcs=$((rcs+r)); echo "    rc=$r"
"$P" "$HERE/speed_read.py" --fork-step 4000032 --futility X C=ai_v14_01_base X=ai_v14_01_base > "$OUT/speed_read_standin_N0_from_4000032.txt" 2>&1; r=$?; rcs=$((rcs+r)); echo "    rc=$r"
"$P" "$HERE/battery_rule.py" --power > "$OUT/power_table.txt" 2>&1; r=$?; rcs=$((rcs+r)); echo "    rc=$r"
after="$(ls models | wc -l)"
if [ "$before" != "$after" ] || ls -d models/ai_v14_0[2-5]_lbat_* >/dev/null 2>&1; then
  echo "FATAL: validation CREATED something under models/ ($before -> $after entries)"; exit 9
fi
echo "=== models/ untouched ($after entries; no ai_v14_0[2-5]_lbat_* dir); summed rc of the checks = $rcs"
exit $rcs
