#!/bin/bash
# OPPONENT-CHECKPOINT isolation cell: greedy / team set M / rev-1 FINAL (25M) opponent.
# Delta vs greedy_meter_inputs (greedy / set M / 24M snapshot) isolates the opponent-ckpt
# axis exactly, holding regime + teams fixed. Pre-registered read:
#   |delta| small (<~1.5pp) => probe Q's -7.06 vs -3.44 gap is mostly TEAM COMPOSITION
#   |delta| large           => the opponent checkpoint carries it
D=/tmp/oppckpt_probe
cd /home/goodlad/dev/gen3ai/.claude/worktrees/simplify-p0
export PYTHONPATH=$PYTHONPATH:src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GEN3AI_TIMEOUT_SCALE=12
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
run_arm () {
  for a in $(seq 1 40); do
    n=$($PY -c "import json;print(len([k for k in json.load(open('$3')) if k.startswith('U_')]))" 2>/dev/null || echo 0)
    [ "$n" -ge 8 ] && { echo "[$2] COMPLETE ($a attempts)"; return 0; }
    echo "[$2] attempt $a (have $n/8)"
    nice -n 15 $PY $D/arm_25M.py "$1" "$2" "$3" 200 3
    sleep 5
  done
  echo "[$2] GAVE UP"
}
run_arm models/ai_v9_29_rev1_0823/final_model.zip   O_REV1FIN  $D/o25_REV1FIN.json  > $D/logs/rev1fin.log 2>&1
run_arm models/ai_v9_59_R2ACTION_0827/final_model.zip O_R2ACTION $D/o25_R2ACTION.json > $D/logs/r2action.log 2>&1
echo "[oppckpt] BOTH ARMS DONE $(date +%H:%M:%S)"
