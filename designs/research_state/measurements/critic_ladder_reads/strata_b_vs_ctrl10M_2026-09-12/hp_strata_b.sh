#!/bin/bash
# strata_b's REGISTERED offline read: 400/800 cycles + v6 reads vs all three controls. Waits for final_model.zip AND the step curve's last generation cycle (serial generator).
cd /home/goodlad/dev/gen3ai; export PYTHONPATH=/home/goodlad/dev/gen3ai/src; export CUDA_VISIBLE_DEVICES=""; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3; H=/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval; ARM=ai_v12_24_ladder_strata_b
while [ ! -f /home/goodlad/dev/gen3ai/models/$ARM/final_model.zip ] || ! grep -q "EXIT 0 for 73121280 draw2" /home/goodlad/.claude/jobs/9ab51de6/tmp/stepcurve/gen_all.log 2>/dev/null; do sleep 120; done
for G in 400 800; do
  if [ $G = 400 ]; then O=$H; R=$H/reads; SEED=20260909; FL=$R/hp400_floor.json; else O=$H/hp800; R=$H/hp800/reads; SEED=20260910; FL=$R/hp800_floor_v6.json; fi
  echo "=== GEN$G $ARM $(date -Is)"; nice -n 15 $PY -m main.ops.eval_trace_gen "$ARM@10000032" --games $G --sentinels 3 --out "$O/$ARM" --workers 4 --concurrency 1 --nice 15 --seed $SEED --shard-games 25 --force > "$O/$ARM.gen.log" 2>&1; echo "GEN$G exit=$?"
  for CTL in ai_v12_11_ladder_ctrl10M ai_v12_15_ladder_ctrl10M_b ai_v12_16_ladder_ctrl10M_c; do
    T=${CTL#ai_v12_??_ladder_}; echo "=== READ$G $ARM vs $CTL $(date -Is)"; nice -n 15 $PY -m main.ops.critic_read "$ARM" --control "$CTL" --step 10000032 --arm-traces "$O/$ARM" --control-traces "$O/$CTL" --floor-json "$FL" --out "$R/${ARM}_vs_$T" --nice 15 --ledger-line > "$R/$ARM.vs_$T.read.log" 2>&1; echo "READ$G vs $CTL exit=$?"; grep -E "cond.opp_class_auc.t4_10.*MATCHED · battle|cond.opp_class_auc.t1_3.*MATCHED · battle|^\| resolution ⭐ \| \`(bot|all)\`|CROSSING" "$R/${ARM}_vs_$T/critic_read.md" | cut -c1-200 | head -5
  done
done
echo "STRATA_B HP DONE $(date -Is)"
