#!/bin/bash
# usage: score_one.sh <run>   (resumable: skips a finished score file)
set -u
RUN=$1
A=/home/goodlad/gen3ai_archive/b1_offpool_belief_50M_20261118
B=/home/goodlad/gen3ai_archive/offpool_belief_2026-11-09/bank
WT=/home/goodlad/dev/gen3ai_wt_b1opb
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
CK=$(awk -v r="$RUN" '$1==r{print $2}' $A/ckpts.txt)
[ -f $A/scores/$RUN.json ] && { echo "$RUN already done"; exit 0; }
cd $WT
export PYTHONPATH=$WT/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
CPUS=$(PYTHONPATH=$WT/src $PY -m utils.core_split agent-cpus)
S=$(date +%s)
nice -n 19 taskset -c "$CPUS" scripts/ops/mem_cap.sh ${CAP_GB:-6} /usr/bin/time -v -o $A/logs/time_$RUN.txt \
  $PY -m main.offpool_belief score --bank $B/home72 --bank $B/heldout72_hpfix --bank $B/forum72 --bank $B/p72 \
  --ckpt $CK --label $RUN --mix 0.25 --out $A/scores/$RUN.json.tmp > $A/logs/score_$RUN.log 2>&1
RC=$?
E=$(date +%s)
if [ $RC -eq 0 ] && [ -s $A/scores/$RUN.json.tmp ]; then mv $A/scores/$RUN.json.tmp $A/scores/$RUN.json; fi
echo "$RUN exit=$RC wall=$((E-S))s" >> $A/logs/driver.log
