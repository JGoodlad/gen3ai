#!/usr/bin/env bash
# THE GUARD — `cond.opp_class_auc.t4_10`, fork arm vs ctrl10M, on BOTH offline 800-game draws,
# each against that draw's own v6 floor file. Both trees are 12 opponents (9 bots + the SAME three
# sentinel steps 4,000,032 / 6,000,000 / 8,000,016), both at step 10,000,032, both
# eval_sentinel_greedy RECORDED — the frame is matched by construction.
set -uo pipefail
MAIN=/home/goodlad/dev/gen3ai
cd "$MAIN" || exit 1
export PYTHONPATH=$MAIN/src
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
H=/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval
ARM=ai_v13_03_fork
CTL=ai_v12_11_ladder_ctrl10M
OUTROOT=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/fork_read/guard}
mkdir -p "$OUTROOT"
for D in hp800 hp800b; do
  O=$H/$D
  [ "$D" = hp800 ] && FL=$O/reads/hp800_floor_v6.json || FL=$O/reads/hp800b_floor_v6.json
  echo "=== GUARD $D  $ARM vs $CTL  $(date -Is)"
  nice -n 15 $PY -m main.ops.critic_read "$ARM" --control "$CTL" --step 10000032 \
      --arm-traces "$O/$ARM" --control-traces "$O/$CTL" --floor-json "$FL" \
      --out "$OUTROOT/$D" --nice 15 --ledger-line > "$OUTROOT/$D.read.log" 2>&1
  echo "GUARD $D exit=$? $(date -Is)"
  grep -E "opp_class_auc" "$OUTROOT/$D/critic_read.md" 2>/dev/null | cut -c1-220 | head -6
done
echo "GUARD DONE $(date -Is)"
