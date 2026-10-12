#!/usr/bin/env bash
# END-STATE closing test: the registered CROSS (design_endstate_closing_test.md §3.1-3.3), run by the Training agent
# on its own GPU lease right after the end-state chain's CHAIN DONE (orchestrator 10-12 ~05:00).
#   8 x 8 = 64 cells (rows rb_es_end2_s2001..s2008, cols rb_es_prod_s2001..s2008, each run's final_model.zip),
#   1,000 mirrored pairs per cell (batches of 500), both sides greedy, schedule seed 0, --oracle-reveal-mode off,
#   purpose ab, family es_closing_strength_steps, request es_closing_steps, label es_closing, the archive's ledger.
# PLAYED AT P_end 62085750 (a detached checkout PIN_DIR verified by rev-parse + clean src/data), never at HEAD.
# Holds the GPU lease: started BEFORE the chain ends, the lease is re-pointed to THIS pid, so it survives the chain.
# Resumable: the ledger's resume skips recorded batches.
set -u
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
J=/home/goodlad/.claude/jobs/es_closing; C=$J/cross; ST=$J/chain_status.txt
PIN=62085750baab1ca275d810fbe0bb5a960904e183; PIN_DIR=/home/goodlad/.cache/gen3ai/es_cross/pin_end2
OPS=/home/goodlad/dev/gen3ai/scripts/ops; T=/home/goodlad/.claude/jobs/gpu_lease_training.token
RULE_PY=$PIN_DIR/designs/research_state/measurements/endstate_closing_test_2026-10-09/closing_rule.py
log() { echo "[$(date -Is)] $1" >> "$ST"; echo "[$(date -Is)] $1"; }
echo $$ > "$C/drive.pid"
log "CROSS armed (pid $$): waits for the end-state chain's clean finish, then plays the 8x8 closing cross at P_end ${PIN:0:8}"
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
n=$("$PY" "$C/plan_cross.py" --out "$C/cells_plan.json" --shas "$C/shas.json" 2>>"$C/run.log") || { log "CROSS HOLD: plan REFUSED (see cross/run.log)"; exit 3; }
log "CROSS plan: $n cells; registering family + playing (1000 pairs/cell, batches of 500, cuda)"
RULE="design_endstate_closing_test.md §3.1-3.3 (P_prod 95d014fa, P_end 62085750 = b132b099 + compat fix; amendments 1-2): end-state 15M x production 15M 8x8 cross, Δ̂=mean(h)-50, V̂=(s²_R+s²_C)/8, df 14, 90% CI Δ̂±1.7613√V̂; PASS iff upper>0 AND lower>-2.0pp, rule 8; estimator main.h2h.cross; coded rule closing_rule.py"
"$PY" -m main.eval_ledger family-register --id es_closing_strength_steps --decision-kind ab_verdict --rule "$RULE" \
  --protocol gen3_eval_protocol_v1_h2h --commit "$PIN" >> "$C/run.log" 2>&1 || { log "CROSS HOLD: family-register failed (see cross/run.log)"; exit 3; }
"$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 "$PY" -m main.h2h play-many --cells "$C/cells_plan.json" --pairs 1000 \
  --batch-pairs 500 --device cuda --purpose ab --family es_closing_strength_steps --request es_closing_steps \
  --label es_closing --oracle-reveal-mode off >> "$C/run.log" 2>&1 || { log "CROSS HOLD: play-many failed (see cross/run.log)"; exit 4; }
log "CROSS played; reading + applying the coded rule"
V=$("$PY" "$C/read_cross.py" "$C/shas.json" "$C/result" "$RULE_PY" 2>>"$C/run.log") || { log "CROSS HOLD: read failed (see cross/run.log)"; exit 6; }
log "CROSS DONE: $(echo "$V" | cut -c1-300)"
