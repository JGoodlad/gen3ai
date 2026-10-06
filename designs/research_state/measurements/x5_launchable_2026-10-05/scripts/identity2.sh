#!/usr/bin/env bash
# The blob byte-identity proof for gen3_reference_state_released_v1 (the nanfix unit's method):
# codehash blob + fixed_mass at the fix and with decision.py put back to HEAD; the new test at HEAD must FAIL.
set -u
W=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a72eda887a3a57819
N=$W/designs/research_state/measurements/x5_fxc4_nanfix_2026-10-05/scripts
X=/home/goodlad/.cache/gen3ai/x5pe/identity2
R=/tmp/x5pe/identity2
mkdir -p $X $R
FILES="src/agents/inference/service/decision.py"
cd "$W" || exit 9
for arm in blob fixed_mass; do
  /tmp/x5pe/gpu_run.sh 28 1500 "$R/codehash_${arm}_fix.log" "$N/codehash.py" "$arm" "$R/codehash_${arm}_fix.json" "$X/dump_fix_$arm"
done
mkdir -p "$X/fixsrc"
for f in $FILES; do cp "$f" "$X/fixsrc/$(basename "$f")"; done
restore() { for f in $FILES; do cp "$X/fixsrc/$(basename "$f")" "$f"; done; echo RESTORED; }
trap restore EXIT
for f in $FILES; do git show "HEAD:$f" > "$f"; done
for arm in blob fixed_mass; do
  /tmp/x5pe/gpu_run.sh 28 1500 "$R/codehash_${arm}_head.log" "$N/codehash.py" "$arm" "$R/codehash_${arm}_head.json" "$X/dump_head_$arm"
done
echo IDENTITY2_DONE
