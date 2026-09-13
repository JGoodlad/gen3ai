#!/bin/bash
# vf025's REGISTERED offline read (ledger 2026-09-12 VERDICT vf15_b PASSES): PASS = the delta CI on cond.opp_class_auc.t4_10 clears +0.022 UPWARD at both draws vs all three controls; else NOT DETECTED.
cd /home/goodlad/dev/gen3ai; export PYTHONPATH=/home/goodlad/dev/gen3ai/src; export CUDA_VISIBLE_DEVICES=""; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3; H=/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval; ARM=ai_v12_29_ladder_vf025
gen () { local O=$1 RUN=$2 SEED=$3; echo "=== GEN $RUN -> $O seed $SEED $(date -Is)"; nice -n 15 $PY -m main.ops.eval_trace_gen "$RUN@10000032" --games 800 --sentinels 3 --out "$O/$RUN" --workers 4 --concurrency 1 --nice 15 --seed $SEED --shard-games 25 --force > "$O/$RUN.gen.log" 2>&1; echo "GEN $RUN -> $(basename $O) exit=$?"; }
LT=/home/goodlad/dev/gen3ai/models/$ARM/latest.txt; [ -f "$LT" ] || { echo "NO latest.txt exit=1"; exit 1; }; echo "latest.txt -> $(cat $LT)"
gen $H/hp800  $ARM 20260910 &
gen $H/hp800b $ARM 20260911 &
wait
for D in hp800 hp800b; do O=$H/$D; R=$H/$D/reads; [ $D = hp800 ] && FL=$R/hp800_floor_v6.json || FL=$R/hp800b_floor_v6.json
  for CTL in ai_v12_11_ladder_ctrl10M ai_v12_15_ladder_ctrl10M_b ai_v12_16_ladder_ctrl10M_c; do
    T=${CTL#ai_v12_??_ladder_}; echo "=== READ $D $ARM vs $CTL $(date -Is)"; nice -n 15 $PY -m main.ops.critic_read "$ARM" --control "$CTL" --step 10000032 --arm-traces "$O/$ARM" --control-traces "$O/$CTL" --floor-json "$FL" --out "$R/${ARM}_vs_$T" --nice 15 --ledger-line > "$R/$ARM.vs_$T.read.log" 2>&1; echo "READ $D vs $CTL exit=$?"; grep -E "cond.opp_class_auc.t4_10.*MATCHED · battle|cond.opp_class_auc.t1_3.*MATCHED · battle|^\| resolution ⭐ \| \`(bot|all)\`|CROSSING" "$R/${ARM}_vs_$T/critic_read.md" | cut -c1-200 | head -5
  done
done
echo "VF025 HP DONE $(date -Is)"
