#!/usr/bin/env bash
# The meter queue (REGISTRATION §5.2): each arm's U + G-A as soon as its final_model.zip exists, one arm at a
# time (6 workers, nice 15, CPU only). Bounded wait per arm (12 h total).
S=/home/goodlad/dev/gen3ai-wt/m5-sizing/designs/research_state/measurements/m5_sizing/scripts
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
M=/home/goodlad/dev/gen3ai/models
end=$((SECONDS + 43200))
for lab in ${LABS:-A:sizing_A_n48_e10_s1001 Ap:sizing_Ap_n48_e10_s1002 B:sizing_B_n256_e10_s1001 C:sizing_C_n256_e5_s1001}; do
  L=${lab%%:*}; R=${lab#*:}
  until [ -f $M/$R/final_model.zip ] || [ $SECONDS -ge $end ]; do sleep 60; done
  [ -f $M/$R/final_model.zip ] || { echo "CHAINM_FAIL no model for $L"; exit 1; }
  sleep 60                                   # the zip is written whole before the json; let the save finish
  echo "CHAINM start $L $(date -Is)"
  PYTHONPATH=/home/goodlad/dev/gen3ai-wt/m5-sizing/src "$PY" $S/meters.py "$L" $M/$R/final_model.zip --workers 6
  echo "CHAINM done $L rc=$? $(date -Is)"
done
echo CHAINM_ALL_DONE
