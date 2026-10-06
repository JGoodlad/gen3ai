#!/usr/bin/env bash
# The final code: T2 replays (fm, fm_nohyp floor), the blob identity proof, the GPU-tier compile test, launches.
set -u
R=/tmp/x5pe/res
echo "queue2 start $(date +%s)"
for arm in fm fm_nohyp; do
  /tmp/x5pe/gpu_run.sh 28 1800 $R/final_${arm}.log /tmp/x5pe/ablate2.py --arm $arm \
    --obs /tmp/x5pe/real_obs.npz --batch 2048 --t2-batches 8,64,256 --out $R/final_${arm}.json
  tail -1 $R/final_${arm}.log
done
echo "t2 done $(date +%s)"
/tmp/x5pe/identity.sh > /tmp/x5pe/identity.log 2>&1
echo "identity done $(date +%s)"
/tmp/x5pe/gpu_run.sh 28 1500 /tmp/x5pe/cuda_test.log -m pytest \
  src/agents/model/compile_regions_fixed_mass_cuda_test.py -q -p no:cacheprovider
tail -3 /tmp/x5pe/cuda_test.log
echo "cuda test done $(date +%s)"
bash /home/goodlad/.cache/gen3ai/x5pe/launch/launches.sh all
echo "QUEUE_DONE $(date +%s)"
