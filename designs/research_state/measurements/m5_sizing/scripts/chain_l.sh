#!/usr/bin/env bash
# Part L after A2: the memory pre-flight at N* = 256 (D-6/D-10), then B, C, A' — each its own GPU hold,
# released between units with a 2-min gap. Stops at the first failure.
S=/home/goodlad/dev/gen3ai-wt/m5-sizing/designs/research_state/measurements/m5_sizing/scripts
T=/home/goodlad/.cache/gen3ai/tmp/sizing
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
M=/home/goodlad/dev/gen3ai/models
G=/home/goodlad/dev/gen3ai-wt/m5-sizing/scripts/ops/gpu_lock.sh
drain() {  # orchestrator 2026-10-01: before re-taking the GPU let queued holders go — bounded 45 min;
           # "drained" = the lock free on 3 consecutive checks 60 s apart
    local e=$((SECONDS + ${1:-2700})) f=0
    sleep 120
    while [ $SECONDS -lt $e ] && [ $f -lt 3 ]; do
        if "$G" --status 2>&1 | grep -q "held by"; then f=0; else f=$((f + 1)); fi
        sleep 60
    done
    echo "CHAINL drain done free=$f $(date -Is)"
}
end=$((SECONDS + 21600))
while kill -0 2025369 2>/dev/null && [ $SECONDS -lt $end ]; do sleep 30; done
echo "CHAINL A2 chain gone $(date -Is)"
[ -f $M/sizing_A2_n48_e10_s1001/final_model.zip ] || { echo "CHAINL_FAIL A2 has no final_model.zip"; exit 1; }
drain 3600      # orchestrator: after A2 the K9(b) root-cause sweep + K8's two memory launches (~55 min)
bash $T/preflight.sh 256 8,16,32,256 > $T/preflight_n256.log 2>&1 || { echo "CHAINL_FAIL preflight256 rc=$?"; exit 1; }
PYTHONPATH=/home/goodlad/dev/gen3ai-wt/m5-sizing/src "$PY" $S/mem_rule.py /home/goodlad/gen3ai_archive/m5_sizing_preflight/n256 \
    $M/sizing_A2_n48_e10_s1001 > $T/mem_rule_n256.json 2>&1
rc=$?; echo "CHAINL mem_rule n256 rc=$rc $(cat $T/mem_rule_n256.json | tail -1)"
[ $rc -eq 0 ] || { echo "CHAINL_FAIL memory rule at 256"; exit 1; }
for spec in "sizing_B_n256_e10_s1001 256 10 1001" "sizing_C_n256_e5_s1001 256 5 1001" "sizing_Ap_n48_e10_s1002 48 10 1002"; do
  set -- $spec
  drain
  BK=-; [ "$2" = 256 ] && BK=8,16,32,256
  bash $S/arm.sh "$1" "$2" "$3" "$4" "$BK" > $T/arm_$1.log 2>&1
  rc=$?; echo "CHAINL arm $1 exit=$rc $(date -Is)"
  [ -f $M/$1/final_model.zip ] || { echo "CHAINL_FAIL $1 has no final_model.zip"; exit 1; }
done
echo CHAINL_ALL_DONE
