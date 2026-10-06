#!/usr/bin/env bash
# The blob byte-identity proof (x5-perf-encode, 2026-10-05), the nanfix unit's method unchanged:
#   1. codehash blob + fixed_mass at the FIX;
#   2. the changed non-test src files put back to HEAD (backups kept; restored on any exit);
#   3. codehash blob + fixed_mass at HEAD; then the new unit tests at HEAD (must FAIL — the revert teeth).
set -u
W=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a72eda887a3a57819
N=$W/designs/research_state/measurements/x5_fxc4_nanfix_2026-10-05/scripts
X=/home/goodlad/.cache/gen3ai/x5pe/identity
R=/tmp/x5pe/identity
mkdir -p $X $R
FILES="src/agents/model/extractor_forward.py src/agents/model/hypothesis_tokens.py src/agents/model/selection_sites.py"
cd "$W" || exit 9
for arm in blob fixed_mass; do
  /tmp/x5pe/gpu_run.sh 28 1500 "$R/codehash_${arm}_fix.log" "$N/codehash.py" "$arm" "$R/codehash_${arm}_fix.json" "$X/dump_fix_$arm"
  tail -1 "$R/codehash_${arm}_fix.log"
done
mkdir -p "$X/fixsrc"
for f in $FILES; do cp "$f" "$X/fixsrc/$(basename "$f")"; done
restore() { for f in $FILES; do cp "$X/fixsrc/$(basename "$f")" "$f"; done; echo RESTORED; }
trap restore EXIT
for f in $FILES; do git show "HEAD:$f" > "$f"; done
for arm in blob fixed_mass; do
  /tmp/x5pe/gpu_run.sh 28 1500 "$R/codehash_${arm}_head.log" "$N/codehash.py" "$arm" "$R/codehash_${arm}_head.json" "$X/dump_head_$arm"
  tail -1 "$R/codehash_${arm}_head.log"
done
/tmp/x5pe/py -m pytest src/agents/model/hypothesis_encode_test.py -q -p no:cacheprovider > "$R/unit_tests_at_head.log" 2>&1
echo "unit tests at HEAD exit $?" >> "$R/unit_tests_at_head.log"
tail -3 "$R/unit_tests_at_head.log"
echo IDENTITY_DONE
