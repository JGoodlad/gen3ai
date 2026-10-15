#!/usr/bin/env bash
# chain_hib2 = chain_hib + the shared compile cache ON. HIDDEN-INFO KNOCK-IN FORKS chain (design_hidden_info_binding.md §8.3; orchestrator 10-14). STAGED, NOT LAUNCHED.
# Usage: PIN=<data-landed main sha> chain_hib.sh <queue>   queue line: RUN SEED <oracle-reveal spec or ->
# Every fork: parent rb_es_end2_s2001/final_model.zip (15,047,835) → --steps 18047835 (+3M), --fork-lr 0.00036, the
# PARENT'S RECORDED RECIPE TYPED (a fork does not inherit the cli_args recipe: untyped it would fall to the parser
# defaults — n_envs 32, self_play False, ent_coef 0.02, grad_accum 1 …), training tree pinned to 0-4,8-12.
# Same PAUSE / disk-gate / CHAIN STOP / CHAIN DONE conventions as the bf16 chains.
set -u
export GEN3AI_SHARED_COMPILE_CACHE=1   # shared compile cache ON from the 2nd fork (orchestrator 10-14 15:25; clean pin 64a695dc)
PIN="${PIN:?PIN unset (the data-landed main commit)}"; MIN_FREE_GB=12
REPO=/home/goodlad/dev/gen3ai; Q=/home/goodlad/.claude/jobs/hib_forks
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
PARENT=$REPO/models/rb_es_end2_s2001/final_model.zip
RECIPE="--n-envs 256 --n-steps 384 --rollout-target-samples 98304 --batch-size 2048 --grad-accum-steps 32 --n-epochs 10 --ent-coef 0.05 --self-play true"
COMMON="--steps 18047835 --fork-lr 0.00036 --trunk-precision bf16 --behaviour-check warn $RECIPE --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all --snapshot-ladder-games 0 --checkpoint-every-steps 1000000 --device cuda"
CONSENT="hib-knockin-forks-design_hidden_info_binding-s8"
cd "$REPO" || exit 2; export PYTHONPATH="$REPO/src"
echo $$ > "$Q/chain.pid"; STATUS="$Q/chain_status.txt"; log() { echo "[$(date -Is)] $1" >> "$STATUS"; }
reached() { python3 -c "import json,sys;m=json.load(open(sys.argv[1]));sys.exit(0 if max([int(p.get('last_step') or 0) for p in m.get('pin_history',[])] or [0])>=18047835 else 1)" "$REPO/models/$1/metadata.json" 2>/dev/null; }
log "HIB chain armed (pid $$, pin $PIN)"
for QUEUE in "$@"; do
  mapfile -t LINES < "$QUEUE"
  for LINE in "${LINES[@]}"; do
    read -r RUN SEED SPEC <<< "$LINE"
    reached "$RUN" && { log "SKIP $RUN (already at 18,047,835)"; continue; }
    if [ -f "$Q/PAUSE" ]; then log "PAUSED before $RUN"; while [ -f "$Q/PAUSE" ]; do sleep 20; done; log "RESUMED: launching $RUN"; fi
    FREE_KB=$(df --output=avail -k / | tail -n 1 | tr -d ' ')
    [ "$FREE_KB" -lt $((MIN_FREE_GB * 1024 * 1024)) ] && { log "HOLD: disk free $((FREE_KB/1024/1024)) GB before $RUN"; exit 3; }
    LEVER=""; [ "$SPEC" != "-" ] && LEVER="--oracle-reveal $SPEC --diagnostic-fork-consent $CONSENT"
    log "START $RUN seed=$SEED lever='${SPEC}' pin=$PIN disk_free_gb=$((FREE_KB/1024/1024))"
    taskset -c 0-4,8-12 "$PY" -m main.launcher --restart-interval-hours 6 --pin-commit "$PIN" \
      --model "$PARENT" $COMMON --seed "$SEED" $LEVER --run-name "$RUN" > "$Q/$RUN.launcher.log" 2>&1 < /dev/null &
    LPID=$!; echo "$LPID" > "$Q/$RUN.pid"
    ( for _ in $(seq 60); do [ -d "$REPO/models/$RUN" ] && break; sleep 30; done
      "$REPO/scripts/ops/watch_run.sh" "$REPO/models/$RUN" --pid-file "$Q/$RUN.pid" --launcher-log "$Q/$RUN.launcher.log" --status-file "$Q/$RUN.repo.watch.txt" ) \
      > "$Q/$RUN.repo.watch.out" 2>&1 < /dev/null &
    wait "$LPID"; RC=$?
    log "END $RUN launcher_rc=$RC"
    reached "$RUN" || { log "CHAIN STOP: $RUN did not reach 18,047,835 (rc=$RC) — the session decides"; exit 1; }
  done
  log "ARM DONE $(basename "$QUEUE")"
done
log "CHAIN DONE"
