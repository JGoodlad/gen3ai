#!/usr/bin/env bash
# bf16 STRENGTH A/B STAGE 2: the registered 8 x 8 CROSS (design_bf16_strength_ab.md §5), run by the Training agent
# on its own GPU lease right after the stage-2 chain's CHAIN DONE.
#   8 x 8 = 64 cells (rows rb_bf_eb_s2001..s2008, cols rb_bf_e_s2001..s2008, each run's 15M final_model.zip),
#   1,000 mirrored pairs per cell (batches of 500), both sides greedy, schedule seed 0, --oracle-reveal-mode off,
#   purpose ab, family bf16_strength_steps, request bf16_s2_steps, label bf16_ab, the archive's ledger.
# PLAYED AT P_bf16 f45d0c27 (a detached checkout PIN_DIR verified by rev-parse + clean src/data), never at HEAD.
# Holds the GPU lease: started BEFORE the chain ends, the lease is re-pointed to THIS pid, so it survives the chain.
# Resumable: the ledger's resume skips recorded batches.
set -u
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
J=/home/goodlad/.claude/jobs/bf16_ab; C=$J/cross/s2; ST=$J/chain_status.txt
PIN=f45d0c2797b892292355dc8b784bb985099f10c1; PIN_DIR=/home/goodlad/.cache/gen3ai/bf_cross/pin_bf16
OPS=/home/goodlad/dev/gen3ai/scripts/ops; T=/home/goodlad/.claude/jobs/gpu_lease_training.token
RULE_PY=$PIN_DIR/designs/research_state/measurements/endstate_closing_test_2026-10-09/closing_rule.py
log() { echo "[$(date -Is)] $1" >> "$ST"; echo "[$(date -Is)] $1"; }
echo $$ > "$C/drive.pid"
log "CROSS armed (pid $$): waits for the end-state chain's clean finish, then plays the 8x8 stage-2 cross at P_bf16 ${PIN:0:8}"
until tail -n 1 "$ST" | grep -qE '^\[[^]]+\] CHAIN DONE$'; do
  if tail -n 3 "$ST" | grep -qE 'CHAIN STOP|HOLD:'; then log "CROSS HOLD: the chain did not finish cleanly — cross NOT played; the session decides"; exit 1; fi
  sleep 30
done
cd "$PIN_DIR" || { log "CROSS HOLD: no pin checkout"; exit 5; }
export PYTHONPATH=$PIN_DIR/src GEN3AI_SHARED_COMPILE_CACHE=0 GEN3AI_GPU_LEASE_TOKEN_FILE=$T
HEAD=$(git rev-parse HEAD)
if [ "$HEAD" != "$PIN" ] || [ -n "$(git status --porcelain --untracked-files=no -- src data)" ]; then
  log "CROSS HOLD: play checkout HEAD=$HEAD (need $PIN) or src/data dirty"; exit 5; fi
"$PY" -c "import main.h2h, sys; print('main.h2h from', main.h2h.__file__)" >> "$C/run.log" 2>&1
n=$("$PY" "$C/plan_s2.py" --out "$C/cells_plan.json" --shas "$C/shas.json" 2>>"$C/run.log") || { log "CROSS HOLD: plan REFUSED (see cross/run.log)"; exit 3; }
log "CROSS plan: $n cells; playing (family registered at stage 1) (1000 pairs/cell, batches of 500, cuda)"
RULE="design_bf16_strength_ab.md §4-5 (P_bf16 f45d0c27, EB=--trunk-precision bf16 vs E=fp32, --arch endstate): stage 1 = EB 5M x E 5M 4x4 cross, Δ̂=mean(h)-50, V̂=(s²_R+s²_C)/4, df 6, 90% CI Δ̂±1.9432√V̂, STOP iff upper<-2.0pp (rule 8); stage 2 = 8 x 8 at 15M, the closing test rule (PASS iff upper>0 AND lower>-2.0pp, df 14); estimator main.h2h.cross; coded rule futility_rule.py"
# family bf16_strength_steps was registered once, by the stage-1 driver (ledger seq 11713, its rule names stage 2)
"$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 taskset -c 0-4,8-12 "$PY" -m main.h2h play-many --cells "$C/cells_plan.json" --pairs 1000 \
  --batch-pairs 500 --device cuda --purpose ab --family bf16_strength_steps --request bf16_s2_steps \
  --label bf16_ab --oracle-reveal-mode off >> "$C/run.log" 2>&1 || { log "CROSS HOLD: play-many failed (see cross/run.log)"; exit 4; }
log "CROSS played; reading + applying the coded rule"
V=$("$PY" "$C/read_s2.py" "$C/shas.json" "$C/result" "$RULE_PY" 2>>"$C/run.log") || { log "CROSS HOLD: read failed (see cross/run.log)"; exit 6; }
log "CROSS DONE: $(echo "$V" | cut -c1-300)"
