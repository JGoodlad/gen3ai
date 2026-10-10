#!/usr/bin/env bash
# The probe battery's first read, as run (2026-10-09). CPU only, nice 19, one heavy job at a time; every model-side
# step runs at the screen pin 6c6d2e09 through the worker (its checkout read-only). Resumable: each step skips what
# it already wrote. Bulk lands in ~/gen3ai_archive/probe_battery/ (never models/).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
PIN=/home/goodlad/dev/gen3ai/.claude/worktrees/st-look1-pin-6c6d2e09
A=/home/goodlad/gen3ai_archive/probe_battery
M=/home/goodlad/dev/gen3ai/models
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
export PYTHONPATH="$REPO/src" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$REPO"
L=""; for i in 1 2 3 4 5 6 7 8; do L="$L L$i=$M/rb_st_legacy_s100$i/final_model.zip"; done
S=""; for i in 1 2 3 4 5 6 7; do S="$S S$i=$M/rb_st_static_s100$i/final_model.zip"; done
R=""; for s in 1 2 3; do R="$R Lr$s=$M/rb_st_legacy_s1001/final_model.zip@rand$s Sr$s=$M/rb_st_static_s1001/final_model.zip@rand$s"; done
ARMS=(--arm legacy=L1,L2,L3,L4,L5,L6,L7,L8 --arm static=S1,S2,S3,S4,S5,S6,S7)
RAND=(--random legacy=Lr1,Lr2,Lr3 --random static=Sr1,Sr2,Sr3)
W=(--checkout "$PIN" --expect-commit 6c6d2e09)
nice -n 19 "$PY" -m main.probe_battery bank "${W[@]}" --legacy $L --static $S --pairs 16 --seed 20261009 \
  --target 25000 --out "$A/bank_v1"                                                   # ~11 min play + ~1 min replay
nice -n 19 "$PY" -m main.probe_battery capture "${W[@]}" --bank "$A/bank_v1" --ckpt $L $S $R --out "$A/caps_v1"   # ~10 min
nice -n 19 "$REPO/scripts/ops/mem_cap.sh" --name probebat-probe 24 timeout 5h \
  "$PY" -m main.probe_battery probe --bank "$A/bank_v1" --caps "$A/caps_v1" --out "$A/probes_v1"   # ~3.5 min / label
nice -n 19 "$PY" -m main.probe_battery report --probes "$A/probes_v1" "${ARMS[@]}" "${RAND[@]}" --out "$A/report_v1"
nice -n 19 "$PY" -m main.probe_battery behaviour "${W[@]}" --bank "$A/bank_v1" --ckpt $L $S "${ARMS[@]}" \
  --out "$A/behaviour_v1"                                                             # ~6 min
nice -n 19 "$PY" -m main.probe_battery depth "${W[@]}" --bank "$A/bank_v1" --ckpt $L $S $R --out "$A/depth_v1"     # ~35 min
nice -n 19 "$PY" -m main.probe_battery depth-report --depth "$A/depth_v1" "${ARMS[@]}" "${RAND[@]}" \
  --out "$A/depth_report_v1"
D=designs/research_state/measurements/probe_battery_2026-10-09
nice -n 19 "$PY" "$D/label_check.py" "$A/bank_v1" > "$D/label_check.json"
nice -n 19 "$PY" "$D/critic_auc.py" > "$D/critic_auc.json"
