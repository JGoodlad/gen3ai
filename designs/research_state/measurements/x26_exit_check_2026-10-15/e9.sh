#!/usr/bin/env bash
# E9's two launch-gate tests (Amendment 2 clause 12) at this checkout's HEAD, rule 5b.
renice -n 19 -p $$ >/dev/null
taskset -cp 5-7,13-15 $$ >/dev/null
WT=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a4a0f0b499e8a107d
cd "$WT" || exit 9
export PYTHONPATH="$WT/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
timeout 3000 /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m pytest -q -p no:cacheprovider \
  "src/agents/training/ridealong_update_test.py::test_BF16_one_update_is_BIT_IDENTICAL_with_the_heads_on_and_off" \
  "src/agents/training/diagnostics_cadence_test.py::test_the_K9b_excluded_share_reaches_TENSORBOARD_on_EVERY_update_not_on_the_diagnostics_cadence" \
  > /tmp/x26exit/e9_tests.log 2>&1
echo "pytest exit $?" >> /tmp/x26exit/e9_tests.log
