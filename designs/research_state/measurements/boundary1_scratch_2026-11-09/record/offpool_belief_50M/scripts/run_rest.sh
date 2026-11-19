#!/bin/bash
A=/home/goodlad/gen3ai_archive/b1_offpool_belief_50M_20261118
export CAP_GB=4
awk '{print $1}' $A/ckpts.txt | xargs -P 3 -n 1 $A/score_one.sh
echo ALLDONE >> $A/logs/driver.log
