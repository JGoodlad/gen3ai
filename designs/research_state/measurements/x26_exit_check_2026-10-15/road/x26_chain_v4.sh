#!/usr/bin/env bash
# THE ROAD TO X26 (design_x26_readiness.md §4.1 Launch plan; orchestrator 10-15 ~15:30): W1 -> W2/W3 (E7 sanity seeds, E6) ->
# W4 (E7 cross + toy crosses R-T1 / R-T1b) -> E10 -> X26 READY (waits for the GO file) -> W5 launch.
# v4 (10-16 00:15, after the memfix): starts at once at W1 (no s4003 / W0b waits). Waits for: x26/pin.txt (P_X26, the full sha; written when the orchestrator sends it), x26/w1_cmds.sh (W1's named CUDA
# checks), and T-ENT s4003's NEW END line. Every GPU step on the Training lease (lease_holder.pid holds it); launches from the
# MAIN checkout with --pin-commit P_X26; the h2h cells at a clean detached checkout of P_X26. Any failed gate => HOLD.
set -u
REPO=/home/goodlad/dev/gen3ai; J=/home/goodlad/.claude/jobs/x26; TJ=/home/goodlad/.claude/jobs/toy_ent; ST=$J/status.txt
T=/home/goodlad/.claude/jobs/gpu_lease_training.token; PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3; OPS=$REPO/scripts/ops
EC=$REPO/designs/research_state/measurements/x26_exit_check_2026-10-15
log() { echo "[$(date -Is)] $1" >> "$ST"; }
echo $$ > "$J/chain.pid"
log "X26 CHAIN armed v4 (pid $$): P_X26 $(cat "$J/pin.txt" | cut -c1-8) (the memfix) -> checkout + selfcheck build -> W1 -> W2/W3 -> W4 -> E10 -> READY/GO -> W5"
P=$(cat "$J/pin.txt"); P8=${P:0:8}; PW=/home/goodlad/.cache/gen3ai/x26/pin_$P8
idle() { for _ in $(seq 40); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 15; done; return 1; }
[ -f "$J/PAUSE" ] && { log "PAUSED on $J/PAUSE"; while [ -f "$J/PAUSE" ]; do sleep 20; done; }
# the P_X26 checkout (W1 + W4 run there; clean, verified)
if [ ! -d "$PW" ]; then mkdir -p "$(dirname "$PW")"; git -C "$REPO" worktree add -q --detach "$PW" "$P" >> "$J/run.log" 2>&1
  ( cd "$PW" && git submodule update --init -q && for n in dist node_modules; do [ -e "deps/pokemon-showdown/$n" ] || ln -s "$REPO/deps/pokemon-showdown/$n" "deps/pokemon-showdown/$n"; done ) >> "$J/run.log" 2>&1; fi
log "building the selfcheck Rust binaries in $PW (no run is live: full machine)"
( cd "$PW" && cargo build -q --profile selfcheck --features emission-selfcheck --bin sim_bridge --bin search_driver --bin core_events --manifest-path src/rust_sim/Cargo.toml ) >> "$J/run.log" 2>&1 || { log "HOLD: selfcheck build failed (run.log)"; exit 5; }
[ "$(git -C "$PW" rev-parse HEAD)" = "$P" ] && [ -z "$(git -C "$PW" status --porcelain --untracked-files=no -- src data)" ] || { log "HOLD: the P_X26 checkout $PW is not $P8 / clean"; exit 5; }
# ---- W1: the CUDA checks the orchestrator named for P_X26 (w1_cmds.sh runs in $PW; PW, T, OPS, PY exported)
idle || { log "HOLD: GPU busy before W1"; exit 2; }
log "W1: CUDA checks at $P8 (w1_cmds.sh)"
( cd "$PW" && export PW T OPS PY GEN3AI_GPU_LEASE_TOKEN_FILE="$T" && bash "$J/w1_cmds.sh" ) > "$J/w1.log" 2>&1; W1RC=$?
log "W1 rc=$W1RC (w1.log)"; [ "$W1RC" = 0 ] || { log "HOLD: W1 failed (rc $W1RC) — nothing further runs"; exit 6; }
# ---- W2 / W3: E7 sanity seeds, one at a time, fresh --arch production at P_X26 to 5M
XS_ARGV=$(cat "$J/xs_argv_base.txt")
for S in 2701 2702; do
  RUN=rb_xs_s$S; A="$XS_ARGV --seed $S --run-name $RUN $(cat "$J/xs_extra_$S.txt" 2>/dev/null)"
  [ -d "$REPO/models/$RUN" ] && { log "HOLD: models/$RUN exists"; exit 3; }
  ( cd "$REPO" && PYTHONPATH="$REPO/src" timeout 600 "$PY" -m main.checkargs --argv "$A" ) > "$J/${RUN}_checkargs.txt" 2>&1 || { log "HOLD: checkargs refused $RUN"; exit 7; }
  ( cd "$REPO" && PYTHONPATH="$REPO/src" timeout 600 "$PY" -m main.launcher --dry-run --restart-interval-hours 6 --pin-commit "$P" $A ) > "$J/${RUN}_dryrun.txt" 2>&1 || { log "HOLD: dry-run refused $RUN"; exit 7; }
  idle || { log "HOLD: GPU busy before $RUN"; exit 2; }
  log "START $RUN pin=$P (main HEAD $(git -C "$REPO" rev-parse --short=8 HEAD))"
  ( cd "$REPO" && export PYTHONPATH="$REPO/src"; exec taskset -c 0-4,8-12 "$PY" -m main.launcher --restart-interval-hours 6 --pin-commit "$P" $A ) > "$J/$RUN.launcher.log" 2>&1 < /dev/null &
  LP=$!; echo "$LP" > "$J/$RUN.pid"
  ( for _ in $(seq 60); do [ -d "$REPO/models/$RUN" ] && break; sleep 30; done
    "$OPS/watch_run.sh" "$REPO/models/$RUN" --pid-file "$J/$RUN.pid" --launcher-log "$J/$RUN.launcher.log" --status-file "$J/$RUN.repo.watch.txt" ) > "$J/$RUN.repo.watch.out" 2>&1 < /dev/null &
  wait "$LP"; RC=$?; log "END $RUN launcher_rc=$RC"
  [ "$RC" = 0 ] || { log "HOLD: $RUN ended rc=$RC"; exit 8; }
  [ "$S" = 2701 ] && { E6=$("$PY" "$J/e6_read.py" "$REPO/models/$RUN" 2>&1); log "E6: $E6"; }
  if [ "$S" = 2701 ]; then  # T37 (b): the RED-band command parses / resolves at P_X26 — a same-run RESUME with --canary-shift-max 0.25 (creates nothing)
    CK=$(ls -t "$REPO/models/$RUN/checkpoints/"checkpoint_*_steps.zip 2>/dev/null | head -1)
    RA="--model ${CK#$REPO/} --steps 6000000 --device cuda --canary-shift-max 0.25"
    ( cd "$REPO" && PYTHONPATH="$REPO/src" timeout 600 "$PY" -m main.checkargs --argv "$RA" ) > "$J/t37b_checkargs.txt" 2>&1; B1=$?
    ( cd "$REPO" && PYTHONPATH="$REPO/src" timeout 600 "$PY" -m main.launcher --dry-run --restart-interval-hours 6 --pin-commit "$P" $RA ) > "$J/t37b_dryrun.txt" 2>&1; B2=$?
    log "T37(b) resume of $RUN with --canary-shift-max 0.25: checkargs rc=$B1, dry-run rc=$B2 ($(grep -c 'would launch' "$J/t37b_dryrun.txt") 'would launch')"
  fi
done
# ---- W4: the ONE h2h window at the P_X26 checkout
export PYTHONPATH="$PW/src"
N=$("$PY" "$J/plan_w4.py" "$J/w4" 2>> "$J/run.log") || { log "HOLD: W4 plan REFUSED (run.log)"; exit 9; }
log "W4 plan: $N (every reference loaded at $P8)"
reg() { "$PY" -m main.eval_ledger family-register --id "$1" --decision-kind ab_verdict --rule "$2" --protocol gen3_eval_protocol_v1_h2h --commit "$P" >> "$J/run.log" 2>&1; }
reg x26_e7_sanity "design_x26_readiness.md §4 E7: rb_xs_s2701/s2702 (P_X26 $P8, 5M) x the 8 rb_bf_eb 5M models (F30), 500 mirrored pairs/cell; coded rule sanity_rule.py (pooled Δ̂ >= -3.0, every seed >= -6.0, run gates clean; rule 8)" || { log "HOLD: register e7"; exit 9; }
reg x26_toy_rt1 "design_x26_readiness.md §3.5 R-T1: T-ENT s4001-s4003 at 15M x rb_bf_eb s2001-s2008 15M finals, 500 pairs/cell, cross_stat (pre-registered §3.5)" || { log "HOLD: register rt1"; exit 9; }
reg x26_toy_rt1b "design_x26_readiness.md §3.5 R-T1b: T-ENT s4001-s4003 at 5M x rb_bf_eb 5M (F30), 250 pairs/cell, DESCRIPTIVE" || { log "HOLD: register rt1b"; exit 9; }
for X in "e7 x26_e7_sanity x26_e7 500" "rt1 x26_toy_rt1 x26_rt1 500" "rt1b x26_toy_rt1b x26_rt1b 250"; do
  set -- $X; idle || { log "HOLD: GPU busy before W4 $1"; exit 2; }
  ( cd "$PW" && GEN3AI_GPU_LEASE_TOKEN_FILE="$T" "$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 taskset -c 0-4,8-12 "$PY" -m main.h2h play-many \
    --cells "$J/w4/cells_$1.json" --pairs "$4" --batch-pairs "$4" --device cuda --purpose ab --family "$2" --request "$3" \
    --label x26_exit --oracle-reveal-mode off ) >> "$J/run.log" 2>&1 || { log "HOLD: W4 play-many $1 failed (run.log)"; exit 10; }
  log "W4 played $1"
done
V=$("$PY" "$J/read_w4.py" "$J/w4" "$J/w4/result" "$EC/sanity_rule.py" 2>> "$J/run.log") || { log "HOLD: W4 read failed"; exit 11; }
log "W4 READ — E7: $(echo "$V" | cut -c1-400)"
unset PYTHONPATH
E7OK=$(echo "$V" | python3 -c "import json,sys;print(json.loads(sys.stdin.read()).get('outcome'))")
E6OK=$(grep -oE 'E6: .*' "$ST" | tail -1 | grep -o '"verdict": "[A-Z]*' | grep -o '[A-Z]*$')
# ---- E10: checkargs + dry-run of X26's argv from MAIN at P_X26
XA=$(cat "$J/x26_argv.txt")
( cd "$REPO" && PYTHONPATH="$REPO/src" timeout 600 "$PY" -m main.checkargs --argv "$XA" ) > "$J/e10_checkargs.txt" 2>&1; C1=$?
( cd "$REPO" && PYTHONPATH="$REPO/src" timeout 600 "$PY" -m main.launcher --dry-run --restart-interval-hours 6 --pin-commit "$P" $XA ) > "$J/e10_dryrun.txt" 2>&1; C2=$?
log "E10: checkargs rc=$C1, dry-run rc=$C2 ($(grep -c 'would launch' "$J/e10_dryrun.txt") 'would launch')"
if [ "$E7OK" != PASS ] || [ "$E6OK" != WITHIN ] || [ "$C1" != 0 ] || [ "$C2" != 0 ]; then
  log "HOLD: X26 NOT READY — E7=$E7OK E6=$E6OK E10 checkargs=$C1 dry-run=$C2"; exit 12; fi
log "X26 READY: E7 PASS, E6 WITHIN, E10 OK — waiting for $J/GO (the data/ freeze is announced at launch)"
until [ -f "$J/GO" ]; do sleep 20; done
idle || { log "HOLD: GPU busy before X26"; exit 2; }
RUN=rb_x26_s2601
log "START $RUN pin=$P (main HEAD $(git -C "$REPO" rev-parse --short=8 HEAD)) — data/ FROZEN from now"
( cd "$REPO" && export PYTHONPATH="$REPO/src"; exec taskset -c 0-4,8-12 "$PY" -m main.launcher --restart-interval-hours 6 --pin-commit "$P" $XA ) > "$J/$RUN.launcher.log" 2>&1 < /dev/null &
LP=$!; echo "$LP" > "$J/$RUN.pid"
( for _ in $(seq 60); do [ -d "$REPO/models/$RUN" ] && break; sleep 30; done
  "$OPS/watch_run.sh" "$REPO/models/$RUN" --pid-file "$J/$RUN.pid" --launcher-log "$J/$RUN.launcher.log" --status-file "$J/$RUN.repo.watch.txt" ) > "$J/$RUN.repo.watch.out" 2>&1 < /dev/null &
wait "$LP"; RC=$?; log "END $RUN launcher_rc=$RC"
