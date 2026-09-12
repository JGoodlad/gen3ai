#!/bin/bash
# vf15_b's REGISTERED offline read (ledger 2026-09-12 VERDICT strata_b FAILS): two 800-game draws (seeds 20260910 / 20260911) vs all three controls, floors hp800_floor_v6.json / hp800b_floor_v6.json (the latter built here from the two hp800b control pairs by the same max|delta| rule). PASS = the delta CI on cond.opp_class_auc.t4_10 clears -0.022 DOWNWARD at both draws against all three.
cd /home/goodlad/dev/gen3ai; export PYTHONPATH=/home/goodlad/dev/gen3ai/src; export CUDA_VISIBLE_DEVICES=""; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3; H=/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval; ARM=ai_v12_25_ladder_vf15_b
gen () { local O=$1 RUN=$2 SEED=$3; echo "=== GEN $RUN -> $O seed $SEED $(date -Is)"; nice -n 15 $PY -m main.ops.eval_trace_gen "$RUN@10000032" --games 800 --sentinels 3 --out "$O/$RUN" --workers 4 --concurrency 1 --nice 15 --seed $SEED --shard-games 25 --force > "$O/$RUN.gen.log" 2>&1; echo "GEN $RUN -> $(basename $O) exit=$?"; }
gen $H/hp800  $ARM 20260910 &
gen $H/hp800b $ARM 20260911 &
gen $H/hp800b ai_v12_16_ladder_ctrl10M_c 20260911 &
wait
# hp800b v6 floor from the two control pairs
R=$H/hp800b/reads; O=$H/hp800b
for CTL in ai_v12_15_ladder_ctrl10M_b ai_v12_16_ladder_ctrl10M_c; do T=${CTL#ai_v12_??_ladder_}; echo "=== FLOORPAIR $CTL vs ctrl10M $(date -Is)"; nice -n 15 $PY -m main.ops.critic_read "$CTL" --control ai_v12_11_ladder_ctrl10M --step 10000032 --arm-traces "$O/$CTL" --control-traces "$O/ai_v12_11_ladder_ctrl10M" --out "$R/floor_v6_${T}_vs_ctrl10M" --nice 15 > "$R/floor_v6_$T.log" 2>&1; echo "FLOORPAIR $CTL exit=$?"; done
$PY - <<'PYE'
import json,os
R="/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval/hp800b/reads"; fl={}
for t in ("ctrl10M_b","ctrl10M_c"):
    d=json.load(open(f"{R}/floor_v6_{t}_vs_ctrl10M/critic_read.json"))
    for r in d["deltas"]:
        if r.get("delta") is not None: fl[r["key"]]=max(fl.get(r["key"],0.0),abs(r["delta"]))
json.dump({"floors":fl,"provenance":"800-game OFFLINE floor, SECOND eval draw (seed 20260911) = the WIDER |delta| of two control pairs (ctrl10M_b-ctrl10M, ctrl10M_c-ctrl10M; ctrl10M_c's hp800b tree generated 2026-09-12 for the vf15_b read), main.ops.critic_read v6; two pairs BOUND a floor, no CI; demote-only. Same rule as hp800_floor_v6.json."},open(f"{R}/hp800b_floor_v6.json","w"),indent=1)
print("hp800b_floor_v6 t4_10:",fl.get("cond.opp_class_auc.t4_10"),"t1_3:",fl.get("cond.opp_class_auc.t1_3"),"n keys",len(fl))
PYE
for D in hp800 hp800b; do O=$H/$D; R=$H/$D/reads; [ $D = hp800 ] && FL=$R/hp800_floor_v6.json || FL=$R/hp800b_floor_v6.json
  for CTL in ai_v12_11_ladder_ctrl10M ai_v12_15_ladder_ctrl10M_b ai_v12_16_ladder_ctrl10M_c; do
    T=${CTL#ai_v12_??_ladder_}; echo "=== READ $D $ARM vs $CTL $(date -Is)"; nice -n 15 $PY -m main.ops.critic_read "$ARM" --control "$CTL" --step 10000032 --arm-traces "$O/$ARM" --control-traces "$O/$CTL" --floor-json "$FL" --out "$R/${ARM}_vs_$T" --nice 15 --ledger-line > "$R/$ARM.vs_$T.read.log" 2>&1; echo "READ $D vs $CTL exit=$?"; grep -E "cond.opp_class_auc.t4_10.*MATCHED · battle|cond.opp_class_auc.t1_3.*MATCHED · battle|^\| resolution ⭐ \| \`(bot|all)\`|CROSSING" "$R/${ARM}_vs_$T/critic_read.md" | cut -c1-200 | head -5
  done
done
echo "VF15_B HP DONE $(date -Is)"
