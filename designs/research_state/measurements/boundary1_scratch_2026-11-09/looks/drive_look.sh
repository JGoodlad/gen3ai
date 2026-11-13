#!/usr/bin/env bash
# BOUNDARY-1, the 2 x 2 from scratch: ONE LOOK of the registered head-to-head cross (PREREG §5), played and read.
#   54 cells at three runs an arm (every pair of runs from different arms, the later arm in C, W, Br, WB playing;
#   PREREG §18 Amendment 1), the look's checkpoints
#   (the first checkpoint each run saved at or after the look's step, never final_model.zip), both sides greedy,
#   schedule seed 0, --oracle-reveal-mode off, purpose ab, on FOUR panels: home72 at 1,000 mirrored pairs a cell,
#   heldout72_hpfix / forum72 / p72 at 500 (batches of 500). One ledger family a panel
#   (b1_scratch_h2h_<panel>), one request a look and panel (the plan's own names, WORK_DIR/look.json), the
#   archive's ledger.
# PLAYED AT THE REGISTERED PIN (PREREG §3; an ARGUMENT here, never a constant: a commit cannot name itself), in a
# detached checkout PIN_DIR verified by rev-parse + clean src/data, never at HEAD:
#   git worktree add --detach "$PIN_DIR" "$PIN"
#   (cd "$PIN_DIR" && ./scripts/bootstrap.sh --skip-env --with-rust --no-check)   # once, under mem_cap, rule 5b
# PLANNED AND READ BY THE KIT AT THE KIT COMMIT (PREREG §18 Amendment 4; an argument too): the guards, the cells, the
# read and the team bootstrap (python -m main.b1.*) run from a second detached checkout KIT_DIR, verified the same
# way. The pin's own kit refuses every run (it wants a pin entry to END at the checkpoint, which no checkpoint's
# does), and the runs train at the pin, so the kit is read from a later commit. KIT may differ from PIN under src/
# and data/ ONLY inside src/main/b1/ (checked below): the parser that resolves each arm's levers, the K9(b) reader's
# rule, the ledger and everything a game runs are then the pin's own bytes. KIT = PIN is allowed (one checkout).
#   git worktree add --detach "$KIT_DIR" "$KIT" && (cd "$KIT_DIR" && ./scripts/bootstrap.sh --skip-env --no-rust --no-check)
# usage: drive_look.sh <KIT_DIR> <KIT> <PIN_DIR> <PIN> <WORK_DIR> <50M|75M|100M|EQ> cpu|cuda [plan options]
#   plan options: --eq-step N (the EQ look) | --replace <the REGISTERED seed the spare replaced, e.g. 5103> |
#                 --survivor width|breadth (after the OTHER factor's STOP-HARM: two arms, 9 cells, no class)
#   cuda: the caller holds the GPU lease (GEN3AI_GPU_LEASE_TOKEN or _FILE in the environment).
#   cpu : rule 5b's shape (nice 19, the agent cores, CUDA hidden). No outer timeout: the job is slot-queued by mem_cap.
#   ONE device a look: the read refuses a look whose rows name two devices or backends (INCONCLUSIVE).
# Resumable: the plan is rewritten identically and the ledger's resume skips recorded batches.
# Exit 0 = played and read (WORK_DIR/result/verdict.json); 3 = the plan REFUSED a run (a question for the
# orchestrator, never a silent drop); 4 = a panel's play failed; 5 = no pin or kit checkout (or a kit that differs
# from the pin outside src/main/b1/); 6 = the read failed.
# REPORTED in run.log beside the plan's own guards, never a gate: main.sidecar_audit --strict and main.checkargs
# on every run of the look; and, after the read, the team-cluster bootstrap of each factor's statistic
# (python -m main.b1.team_bootstrap -> WORK_DIR/result/team_bootstrap.json; PREREG §5.5: it decides nothing and
# never changes this script's exit; a survivor look has none).
# The 100M look's CONFIRMING meter is added afterwards, once the Kakuna read has its classes (PREREG §6):
#   python -m main.b1.look read --work <WORK_DIR> --kakuna width=<class> --kakuna breadth=<class>
set -u
KIT_DIR=${1:?KIT_DIR}; KIT=${2:?KIT (the 40-hex kit commit of PREREG §18 Amendment 4)}
PIN_DIR=${3:?PIN_DIR}; PIN=${4:?PIN (the 40-hex pin of PREREG §3)}; W=${5:?WORK_DIR}; LOOK=${6:?50M|75M|100M|EQ}
DEV=${7:?cpu|cuda}; shift 7
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
MAIN=/home/goodlad/dev/gen3ai; OPS=$MAIN/scripts/ops
mkdir -p "$W"; W=$(cd "$W" && pwd); LOG=$W/run.log   # absolute: the commands below run in two checkouts
log() { echo "[$(date -Is)] $1" | tee -a "$LOG"; }
case "$DEV" in cpu|cuda) ;; *) log "HOLD: device '$DEV' is not cpu or cuda"; exit 2;; esac
[[ "$PIN" =~ ^[0-9a-f]{40}$ ]] || { log "HOLD: pin '$PIN' is not a full 40-hex commit"; exit 2; }
[[ "$KIT" =~ ^[0-9a-f]{40}$ ]] || { log "HOLD: kit commit '$KIT' is not a full 40-hex commit"; exit 2; }
[ -d "$KIT_DIR" ] || { log "HOLD: no kit checkout at $KIT_DIR"; exit 5; }
KIT_DIR=$(cd "$KIT_DIR" && pwd)
KHEAD=$(git -C "$KIT_DIR" rev-parse HEAD)
if [ "$KHEAD" != "$KIT" ] || [ -n "$(git -C "$KIT_DIR" status --porcelain --untracked-files=no -- src data)" ]; then
  log "HOLD: kit checkout HEAD=$KHEAD (need $KIT) or src/data dirty"; exit 5; fi
# The kit may differ from the pin only in the kit itself (the header's rule): a failed diff is a HOLD too.
OUTSIDE=$(git -C "$KIT_DIR" diff --name-only "$PIN" "$KIT" -- src data) \
  || { log "HOLD: cannot diff the pin against the kit commit in $KIT_DIR"; exit 5; }
OUTSIDE=$(printf '%s\n' "$OUTSIDE" | grep -v '^src/main/b1/' | grep -c .)
[ "$OUTSIDE" = 0 ] || { log "HOLD: kit $KIT differs from pin $PIN in $OUTSIDE file(s) of src/ or data/ outside src/main/b1/"; exit 5; }
cd "$PIN_DIR" || { log "HOLD: no pin checkout at $PIN_DIR"; exit 5; }
HEAD=$(git rev-parse HEAD)
if [ "$HEAD" != "$PIN" ] || [ -n "$(git status --porcelain --untracked-files=no -- src data)" ]; then
  log "HOLD: play checkout HEAD=$HEAD (need $PIN) or src/data dirty"; exit 5; fi
export PYTHONPATH=$PIN_DIR/src GEN3AI_SHARED_COMPILE_CACHE=0
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=""
kit() { ( cd "$KIT_DIR" && PYTHONPATH=$KIT_DIR/src "$PY" "$@" ); }   # a kit command: the kit commit's main.b1
"$PY" -c "import main.h2h; print('main.h2h from', main.h2h.__file__)" >> "$LOG" 2>&1 \
  || { log "HOLD: the pin checkout does not import main.h2h (see $LOG)"; exit 5; }
kit -c "import main.b1.look, main.b1.team_bootstrap; print('main.b1.look from', main.b1.look.__file__)" >> "$LOG" 2>&1 \
  || { log "HOLD: the kit checkout does not import main.b1.look (see $LOG)"; exit 5; }
log "kit $KIT at $KIT_DIR; play at pin $PIN in $PIN_DIR"
n=$(kit -m main.b1.look plan --look "$LOOK" --pin "$PIN" --out "$W" "$@" 2>>"$LOG") \
  || { log "HOLD: plan REFUSED (see $LOG)"; exit 3; }
log "plan: $n cells at look $LOOK ($*); four panels on $DEV"
field() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['names'][sys.argv[2]][sys.argv[3]])" "$W/look.json" "$1" "$2"; }
RUNS=$("$PY" -c "import json,os,sys; d=json.load(open(sys.argv[1])); print(' '.join(sorted({os.path.dirname(os.path.dirname(f['checkpoint'])) for fs in d['runs'].values() for f in fs})))" "$W/look.json")
# shellcheck disable=SC2086
"$PY" -m main.sidecar_audit --strict $RUNS >> "$LOG" 2>&1; log "REPORTED: main.sidecar_audit --strict exit $? (see $LOG)"
for R in $RUNS; do "$PY" -m main.checkargs "$R" >> "$LOG" 2>&1; log "REPORTED: main.checkargs $(basename "$R") exit $?"; done
for PANEL in home72 heldout72_hpfix forum72 p72; do
  FAMILY=$(field "$PANEL" family); REQUEST=$(field "$PANEL" request); PAIRS=$(field "$PANEL" pairs); RULE=$(field "$PANEL" rule)
  "$PY" -m main.eval_ledger family-register --id "$FAMILY" --decision-kind ab_verdict --rule "$RULE" \
    --protocol gen3_eval_protocol_v1_h2h --commit "$PIN" >> "$LOG" 2>&1 \
    || { log "HOLD: family-register $FAMILY failed (see $LOG)"; exit 3; }
  PLAY=("$PY" -m main.h2h play-many --cells "$W/cells_plan.json" --pairs "$PAIRS" --batch-pairs 500 --panel "$PANEL"
        --seed 0 --device "$DEV" --purpose ab --family "$FAMILY" --request "$REQUEST" --label "$REQUEST"
        --oracle-reveal-mode off)
  log "panel $PANEL: $PAIRS pairs a cell, family $FAMILY, request $REQUEST"
  if [ "$DEV" = cuda ]; then
    "$OPS/gpu_lock.sh" "$OPS/mem_cap.sh" 32 "${PLAY[@]}" >> "$LOG" 2>&1 \
      || { log "HOLD: play-many failed on $PANEL (see $LOG)"; exit 4; }
  else
    AGENT_CPUS=$("$PY" -m utils.core_split agent-cpus) || { log "HOLD: utils.core_split named no agent CPUs"; exit 4; }
    NT=$(( $(taskset -c "$AGENT_CPUS" nproc) / 2 )); [ "$NT" -ge 1 ] || NT=1   # one thread a physical core
    nice -n 19 taskset -c "$AGENT_CPUS" "$OPS/mem_cap.sh" 24 "${PLAY[@]}" --threads "$NT" --torch-threads "$NT" >> "$LOG" 2>&1 \
      || { log "HOLD: play-many failed on $PANEL (see $LOG)"; exit 4; }
  fi
done
log "played; reading and applying the coded rule"
V=$(kit -m main.b1.look read --work "$W" 2>>"$LOG") || { log "HOLD: read failed (see $LOG)"; exit 6; }
kit -m main.b1.team_bootstrap --work "$W" > /dev/null 2>>"$LOG"
log "REPORTED, never a gate: main.b1.team_bootstrap exit $? (0 = $W/result/team_bootstrap.json; 2 = refused, as for a survivor look; 3 = the look is INCONCLUSIVE)"
log "DONE: $(echo "$V" | cut -c1-600)"
