#!/usr/bin/env bash
# Amendment-2 switch for the HIB fork chain (orchestrator 10-14 ~15:25, pre-approved, uniform across all 15 arms):
# K9(b) |d log pi| stays STRICT by READ (every probe's max < 1e-4, audited per fork; a violation makes the fork
# INCONCLUSIVE), the excluded_frac clause runs in WARN (--behaviour-check warn). rb_hib_moves_s3101 is switched NOW
# (its whole buffer sits above the tie ceiling, so the 3-crash breaker would trip); every later fork starts in warn
# (chain_hib2.sh, which also has the shared compile cache ON). Usage: hib_switch.sh <swap_cache pid>
set -u
REPO=/home/goodlad/dev/gen3ai; Q=/home/goodlad/.claude/jobs/hib_forks; ST=$Q/chain_status.txt; T=/home/goodlad/.claude/jobs/gpu_lease_training.token
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3; PIN=64a695dc23c946eaaf4e81cae04c4e18dfff787d; RUN=rb_hib_moves_s3101; R=$REPO/models/$RUN
SWAPPID=${1:?swap pid}
log() { echo "[$(date -Is)] $1" >> "$ST"; }
GEN3AI_GPU_LEASE_TOKEN_FILE="$T" "$REPO/scripts/ops/gpu_lease.sh" acquire --owner "Training Oct" --max-hours 96 --watch-pid $$ --token-file "$T" >/dev/null 2>&1
W=$(python3 -c "import json;print(json.load(open('/home/goodlad/.claude/jobs/gpu.lock.lease.json'))['watch_pid'])"); [ "$W" = "$$" ] || { log "SWITCH HOLD: lease not re-pointed ($W)"; exit 1; }
OLDCHAIN=$(cat "$Q/chain.pid"); LP=$(cat "$Q/$RUN.pid")
echo $$ > "$Q/chain.pid"
log "SWITCH (pid $$): Amendment 2 for the fork chain — stopping swap $SWAPPID, old chain $OLDCHAIN, then SIGTERM $RUN launcher $LP (it saves)"
kill -TERM "$SWAPPID" 2>/dev/null; kill -TERM "$OLDCHAIN"; sleep 1
kill -TERM "$LP"
for _ in $(seq 120); do kill -0 "$LP" 2>/dev/null || break; sleep 5; done
kill -0 "$LP" 2>/dev/null && { log "SWITCH HOLD: $RUN launcher $LP still alive after 600 s"; exit 1; }
for _ in $(seq 24); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { log "SWITCH HOLD: GPU not clear after the launcher exit"; exit 1; }
rm -f "$Q/PAUSE"
M=$(ls -t "$R"/checkpoints/checkpoint_*_steps.zip "$R"/final_model.zip "$R"/final_model_exception.zip 2>/dev/null | head -1)
log "SWITCH: $RUN stopped cleanly; resuming from $(basename "$M") with --behaviour-check warn (typed recipe, reveal + bf16)"
export GEN3AI_SHARED_COMPILE_CACHE=1
cd "$REPO" && PYTHONPATH="$REPO/src" taskset -c 0-4,8-12 "$PY" -m main.launcher --restart-interval-hours 6 --pin-commit "$PIN" \
  --model "$M" --steps 18047835 --trunk-precision bf16 \
  --n-envs 256 --n-steps 384 --rollout-target-samples 98304 --batch-size 2048 --grad-accum-steps 32 --n-epochs 10 --ent-coef 0.05 --self-play true \
  --oracle-reveal ki_moves --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all \
  --snapshot-ladder-games 0 --checkpoint-every-steps 1000000 --device cuda --behaviour-check warn \
  > "$Q/$RUN.warn.launcher.log" 2>&1 < /dev/null &
LP2=$!; echo "$LP2" > "$Q/$RUN.pid"
nohup setsid "$REPO/scripts/ops/watch_run.sh" "$R" --pid-file "$Q/$RUN.pid" --launcher-log "$Q/$RUN.warn.launcher.log" --status-file "$Q/$RUN.warn.repo.watch.txt" > "$Q/$RUN.warn.repo.watch.out" 2>&1 < /dev/null &
wait "$LP2"; RC=$?
reached() { python3 -c "import json,sys;m=json.load(open(sys.argv[1]));sys.exit(0 if max([int(p.get('last_step') or 0) for p in m.get('pin_history',[])] or [0])>=18047835 else 1)" "$R/metadata.json"; }
log "END $RUN (warn resume) launcher_rc=$RC"
reached || { log "CHAIN STOP: $RUN warn resume did not reach 18,047,835 (rc=$RC)"; exit 1; }
exec env PIN="$PIN" "$Q/chain_hib2.sh" "$Q/queue_hib_rest.txt"
