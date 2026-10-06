#!/usr/bin/env bash
# The blob byte-identity proof + the revert teeth, in one GPU sequence (x5-perf-nan, 2026-10-05):
#   1. codehash blob + fixed_mass at the FIX;
#   2. the changed non-test src files put back to HEAD (backups kept; restored on any exit);
#   3. codehash blob + fixed_mass at HEAD; the CUDA test at HEAD (must FAIL; its slow-tier row redirected;
#      NO_TEST=1 skips it).
set -u
W=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a30fcd2b9887c1fda
D=$W/designs/research_state/measurements/x5_fxc4_nanfix_2026-10-05
X=/home/goodlad/.cache/gen3ai/x5pn
FILES="src/agents/model/damage_op.py src/agents/model/compile_canary.py src/agents/model/compile_regions.py
src/agents/model/compile_trainer.py src/agents/model/selection_sites.py src/agents/training/instrumented_ppo/flat_intent_fold.py"
cd "$W" || exit 9
for arm in blob fixed_mass; do
  "$D/scripts/gpu_run.sh" 28 1500 "$D/results/codehash_${arm}_fix.log" "$D/scripts/codehash.py" "$arm" \
      "$D/results/codehash_${arm}_fix.json" "$X/dump_fix_$arm"
done
mkdir -p "$X/fixsrc"
for f in $FILES; do cp "$f" "$X/fixsrc/$(basename "$f")"; done
restore() { for f in $FILES; do cp "$X/fixsrc/$(basename "$f")" "$f"; done; echo RESTORED; }
trap restore EXIT
for f in $FILES; do git show "HEAD:$f" > "$f"; done
for arm in blob fixed_mass; do
  "$D/scripts/gpu_run.sh" 28 1500 "$D/results/codehash_${arm}_head.log" "$D/scripts/codehash.py" "$arm" \
      "$D/results/codehash_${arm}_head.json" "$X/dump_head_$arm"
done
[ -n "${NO_TEST:-}" ] || GEN3AI_SLOW_STATUS_FILE=$X/slow_status_revert.json "$X/gpu_pytest.sh" "$D/results/cuda_test_at_head.log" \
    src/agents/model/compile_regions_fixed_mass_cuda_test.py -q -p no:cacheprovider
echo IDENTITY_DONE
