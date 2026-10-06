#!/usr/bin/env bash
# The ablation before/after: replicate 1 in arm order, replicate 2 reversed (the cost ablation's protocol).
set -u
R=/tmp/x5pe/res
mkdir -p $R
for rep in 1 2; do
  if [ $rep = 1 ]; then ARMS="blob fm_perrow fm"; else ARMS="fm fm_perrow blob"; fi
  for arm in $ARMS; do
    T2=""
    [ $rep = 1 ] && T2="--t2-batches 8,64,256"
    /tmp/x5pe/gpu_run.sh 28 1800 $R/ablate_${arm}_r$rep.log /tmp/x5pe/ablate2.py --arm $arm \
      --obs /tmp/x5pe/real_obs.npz --batch 2048 $T2 --out $R/ablate_${arm}_r$rep.json
    tail -2 $R/ablate_${arm}_r$rep.log
  done
done
echo ABLATE_DONE
