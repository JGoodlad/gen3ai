#!/usr/bin/env bash
# Four FRESH processes through the real startup gate (fixed_mass): 2 at fresh init, 2 at the golden's
# perturbed (trained-like) weights. Usage: gate4.sh <tag>
D=/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a30fcd2b9887c1fda/designs/research_state/measurements/x5_fxc4_nanfix_2026-10-05
TAG=${1:-fix}
for i in 1 2; do
  for w in fresh golden; do
    "$D/scripts/gpu_run.sh" 24 1500 "$D/results/gate_${TAG}_fm_${w}_p$i.txt" "$D/scripts/gate.py" fixed_mass "$w" 1
  done
done
echo ALLDONE
