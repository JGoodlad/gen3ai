#!/usr/bin/env bash
# HIB knock-in forks: the §8.4 crosses (primary per lever, one_sided; v8, off), run by the Training agent on its own
# GPU lease right after the fork chain's CHAIN DONE (orchestrator 10-14). Played at P_hib 64a695dc from a detached
# checkout (verified by rev-parse + clean src/data). Read with the committed read_hib.py (4ffb8824). The lease is
# re-pointed to THIS pid at arming so it survives the chain's exit. Resumable (the ledger skips recorded batches).
set -u
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
J=/home/goodlad/.claude/jobs/hib_forks; C=$J/cross; ST=$J/chain_status.txt
PIN=64a695dc23c946eaaf4e81cae04c4e18dfff787d; PIN_DIR=/home/goodlad/.cache/gen3ai/hib_cross/pin
OPS=/home/goodlad/dev/gen3ai/scripts/ops; T=/home/goodlad/.claude/jobs/gpu_lease_training.token
log() { echo "[$(date -Is)] $1" >> "$ST"; echo "[$(date -Is)] $1"; }
echo $$ > "$C/drive.pid"
log "CROSS armed (pid $$): waits for the fork chain's clean finish, then plays the §8.4 crosses at P_hib ${PIN:0:8}"
until tail -n 1 "$ST" | grep -qE '^\[[^]]+\] CHAIN DONE$'; do
  if tail -n 3 "$ST" | grep -qE 'CHAIN STOP|HOLD:'; then log "CROSS HOLD: the fork chain did not finish cleanly — not played"; exit 1; fi
  sleep 30
done
cd "$PIN_DIR" || { log "CROSS HOLD: no pin checkout"; exit 5; }
export PYTHONPATH=$PIN_DIR/src GEN3AI_SHARED_COMPILE_CACHE=0 GEN3AI_GPU_LEASE_TOKEN_FILE=$T
HEAD=$(git rev-parse HEAD)
if [ "$HEAD" != "$PIN" ] || [ -n "$(git status --porcelain --untracked-files=no -- src data)" ]; then log "CROSS HOLD: pin checkout HEAD=$HEAD or dirty"; exit 5; fi
INVALID=$(python3 - <<'PY'
import subprocess, json
bad = []
for arm in ("moves", "items", "spreads", "species", "ctrl"):
    for s in (3101, 3102, 3103):
        r = f"rb_hib_{arm}_s{s}"
        out = subprocess.run(["python3", "/home/goodlad/.claude/jobs/hib_forks/bank_hib.py", r], capture_output=True, text=True, cwd="/home/goodlad/dev/gen3ai").stdout
        if "STRICT-BY-READ: PASS" not in out: bad.append(f"{r}: strict-by-read not PASS")
print(json.dumps(bad))
PY
)
log "CROSS K-VALID audit: invalid=$INVALID"
CNT=$("$PY" "$C/plan_hib.py" "$C" 2>>"$C/run.log") || { log "CROSS HOLD: plan REFUSED (see cross/run.log)"; exit 3; }
log "CROSS plan: $CNT"
RULE="design_hidden_info_binding.md §8.4-8.5 as coded in designs/research_state/measurements/hidden_info_forks_2026-10-14/read_hib.py (4ffb8824): per lever L, L 3 seeds x ctrl 3 seeds one_sided (L told at its recorded level), cross_stat n=3 df 4, two-sided 95% interval; WORTH_FIRST_CLAUSE_ONLY (lower>+0.05; K-L-WORTH not declarable, no placebo arm) / NULL (inside ±2.0) / NOT_DETECTED; moves extension if NOT_DETECTED with half-width>2.5; K-RANK paired; rule 8 ±0.05"
"$PY" -m main.eval_ledger family-register --id hib_forks_primary --decision-kind ab_verdict --rule "$RULE" \
  --protocol gen3_eval_protocol_v1_h2h_oracle_one_sided --commit "$PIN" >> "$C/run.log" 2>&1 || { log "CROSS HOLD: family-register (primary) failed"; exit 3; }
"$PY" -m main.eval_ledger family-register --id hib_forks_v8 --decision-kind ab_verdict \
  --rule "design_hidden_info_binding.md §8.4 v8 read: ctrl 3 seeds x the parent rb_es_end2_s2001, off; DESCRIPTIVE, CONFOUNDED by the learnset data change + 3M (decision (c))" \
  --protocol gen3_eval_protocol_v1_h2h --commit "$PIN" >> "$C/run.log" 2>&1 || { log "CROSS HOLD: family-register (v8) failed"; exit 3; }
for LV in moves items spreads species; do
  "$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 taskset -c 0-4,8-12 "$PY" -m main.h2h play-many --cells "$C/cells_$LV.json" --pairs 1000 \
    --batch-pairs 500 --device cuda --purpose ab --family hib_forks_primary --request "hib_primary_$LV" \
    --label hib_forks --oracle-reveal-mode one_sided >> "$C/run.log" 2>&1 || { log "CROSS HOLD: play-many $LV failed (see cross/run.log)"; exit 4; }
  log "CROSS played $LV (9 cells)"
done
"$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 taskset -c 0-4,8-12 "$PY" -m main.h2h play-many --cells "$C/cells_v8.json" --pairs 1000 \
  --batch-pairs 500 --device cuda --purpose ab --family hib_forks_v8 --request hib_v8 --label hib_forks \
  --oracle-reveal-mode off >> "$C/run.log" 2>&1 || { log "CROSS HOLD: play-many v8 failed (see cross/run.log)"; exit 4; }
log "CROSS played v8 (3 cells)"
# HONEST (withheld): its own detached checkout at the withheld build (orchestrator 10-15), behind a pin check.
PIN2=8272c6cb5f35d26b172e609fc96ce60e09b3d0d4; PIN2_DIR=/home/goodlad/.cache/gen3ai/hib_cross/pin_withheld
cd "$PIN2_DIR" || { log "CROSS HOLD: no withheld checkout"; exit 5; }
HEAD2=$(git rev-parse HEAD)
if [ "$HEAD2" != "$PIN2" ] || [ -n "$(git status --porcelain --untracked-files=no -- src data)" ]; then log "CROSS HOLD: withheld checkout HEAD=$HEAD2 or dirty"; exit 5; fi
export PYTHONPATH=$PIN2_DIR/src
"$PY" -m main.eval_ledger family-register --id hib_forks_honest --decision-kind ab_verdict \
  --rule "design_hidden_info_binding.md §8.4 honest-play cross: lever L seed i with its reveal WITHHELD at eval vs ctrl seed i (seed-paired, 12 cells), DESCRIPTIVE (no rule); read by read_hib.py (f68f686b)" \
  --protocol gen3_eval_protocol_v1_h2h_oracle_withheld --commit "$PIN2" >> "$C/run.log" 2>&1 || { log "CROSS HOLD: family-register (honest) failed"; exit 3; }
for LV in moves items spreads species; do
  "$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 taskset -c 0-4,8-12 "$PY" -m main.h2h play-many --cells "$C/cells_honest_$LV.json" --pairs 1000 \
    --batch-pairs 500 --device cuda --purpose ab --family hib_forks_honest --request "hib_honest_$LV" \
    --label hib_forks --oracle-reveal-mode withheld >> "$C/run.log" 2>&1 || { log "CROSS HOLD: play-many honest $LV failed (see cross/run.log)"; exit 4; }
  log "CROSS played honest $LV (3 cells)"
done
log "CROSS played honest (12 cells); reading with read_hib.decide (at the withheld checkout's main.h2h.cross)"
V=$("$PY" "$C/read_cross_hib.py" "$C" "$C/result" "$INVALID" 2>>"$C/run.log") || { log "CROSS HOLD: read failed (see cross/run.log)"; exit 6; }
log "CROSS DONE: $(echo "$V" | cut -c1-300)"
