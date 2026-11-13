#!/usr/bin/env bash
# CE-3d: the two registered CPU crosses in sequence (deciding first, then the reported x67 one). Resumable: re-run as is.
set -u
D=/home/goodlad/dev/gen3ai/designs/research_state/measurements/ce3d_magnet_screen_2026-11-11/cross/drive_cross.sh
PIN_DIR=/home/goodlad/dev/gen3ai-reads/ce3d_play_333a73dd
A=/home/goodlad/gen3ai_archive/ce3d_cross_2026-11-12
S=/tmp/claude-1000/-home-goodlad-dev-gen3ai/4b7f33cf-798a-4092-8287-751835e7f926/scratchpad/ce3d_cross
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
unset GEN3AI_TEST_ALLOW_GPU GEN3AI_GPU_LEASE_TOKEN GEN3AI_GPU_LEASE_TOKEN_FILE GEN3AI_MODELS_DIR GEN3AI_BLOCKING
echo "[$(date -Is)] driver pid $$ start" >> "$S/driver.log"
"$D" "$PIN_DIR" "$A/deciding" cpu >> "$S/driver.log" 2>&1; rc1=$?
echo "[$(date -Is)] deciding cross rc=$rc1" >> "$S/driver.log"
"$D" "$PIN_DIR" "$A/x67" cpu x67 >> "$S/driver.log" 2>&1; rc2=$?
echo "[$(date -Is)] reported (x67) cross rc=$rc2" >> "$S/driver.log"
echo "[$(date -Is)] ALL DONE rc1=$rc1 rc2=$rc2" >> "$S/driver.log"
