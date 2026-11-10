#!/usr/bin/env bash
# X66 ExIt-0H (PREREG §10): the three registered students on the GPU, in order, each resumable. Run by the session
# holding the GPU lease (Training Oct), from anywhere, once the GPU is free. Preconditions first (any failure is a
# HOLD: nothing trains); a chunk is <= 50 min (exit 3 = budget reached: the loop re-runs it); any other non-zero exit
# HOLDs. GPU only: the pick, the export and the delivery gate are CPU work for the read operator (gate.sh).
#   GEN3AI_GPU_LEASE_TOKEN_FILE=<lease token file> gpu_students.sh [fold ...]   # default: all three, H then F then P72
set -u
PIN_SHA=c0b95914e239ff8c66da5a06afe9b42a43d41cf6   # the side pin (PREREG §5.1), tag x66-pin-c0b95914e
PINWT=/home/goodlad/dev/gen3ai-reads/x66_pin
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
X=/home/goodlad/gen3ai_archive/x66_exit0h_2026-11-09
BANK=$X/bank
BSHA=0812768c9fcecf586c5973e23ba600cc9b3da99a704b2aeab567e89e7a185344   # the registered bank (PREREG Amendment 1)
PSTAR=/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip
PSTAR_SHA=814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed
LOG=$X/gpu
: "${GEN3AI_GPU_LEASE_TOKEN_FILE:?set it to the GPU lease token file}"
mkdir -p "$LOG"
ST=/home/goodlad/.claude/jobs/x35/status.txt              # the training session's status file (as X65's driver)
say() { echo "[$(date -Is)] X66S: $*" | tee -a "$LOG/driver.log" >> "$ST"; }
hold() { say "HOLD: $*"; exit 1; }
E6="port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130"   # S'-HONEST's engine (PREREG §5.1); a fresh worktree can miss it
eng() { (cd "$1" && PYTHONPATH="$1/src" CUDA_VISIBLE_DEVICES="" "$PY" -c \
    "from main.anchors.server import engine_stamp as e; print(e())" 2>/dev/null | tail -1); }
need_e6() { local got; got=$(eng "$1"); [ "$got" = "$E6" ] || hold "the engine at $1 is '$got', not E6 $E6 (a missing \
src/rust_env/Cargo.lock or release lib: re-run bootstrap.sh --with-rust there; never copy a lock in without recording it)"; }
[ -e "$LOG/NO_GO" ] && hold "$LOG/NO_GO present"
case "$BSHA" in *PENDING*) hold "the bank sha256 is still PENDING: Amendment 1 has not landed" ;; esac

# --- preconditions (a HOLD, never a train) ---------------------------------------------------------------------
if [ ! -d "$PINWT" ]; then
    git -C /home/goodlad/dev/gen3ai fetch -q origin "refs/tags/x66-pin-${PIN_SHA:0:9}:refs/tags/x66-pin-${PIN_SHA:0:9}" \
        || hold "cannot fetch the tag x66-pin-${PIN_SHA:0:9}"
    git -C /home/goodlad/dev/gen3ai worktree add --detach "$PINWT" "x66-pin-${PIN_SHA:0:9}" >/dev/null 2>&1 \
        || hold "cannot create the pin worktree $PINWT"
    (cd "$PINWT" && CUDA_VISIBLE_DEVICES="" nice -n 19 ./scripts/bootstrap.sh --skip-env --with-rust --no-check \
        >> "$LOG/bootstrap.log" 2>&1) || hold "bootstrap of $PINWT failed (see $LOG/bootstrap.log)"
fi
cd "$PINWT" || hold "no $PINWT"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$PIN_SHA^{commit}")" ] || hold "$PINWT is not at the pin $PIN_SHA"
[ -z "$(git status --porcelain --untracked-files=no)" ] || hold "$PINWT is dirty"
need_e6 "$PINWT"
export PYTHONPATH="$PINWT/src"
[ "$(sha256sum "$PSTAR" | cut -c1-64)" = "$PSTAR_SHA" ] || hold "P*'s zip is not $PSTAR_SHA"
CUDA_VISIBLE_DEVICES="" "$PY" -m main.x65.bank check --out "$BANK" --manifest-sha256 "$BSHA" >> "$LOG/driver.log" 2>&1 \
    || hold "the bank is not the registered $BSHA"
say "preconditions PASS: pin $PIN_SHA on E6, bank $BSHA, P* $PSTAR_SHA"

for fold in ${@:-heldout72_hpfix forum72 p72}; do
    case $fold in heldout72_hpfix) s=s_h ;; forum72) s=s_f ;; p72) s=s_p ;; *) hold "unknown fold $fold" ;; esac
    finished=0
    for try in 1 2 3 4 5 6; do
        [ -e "$LOG/NO_GO" ] && hold "$LOG/NO_GO present"
        say "$s try $try: start"
        env -u CUDA_VISIBLE_DEVICES -u GEN3AI_TEST_ALLOW_GPU taskset -c 0-4,8-12 scripts/ops/gpu_lock.sh \
            "$PY" -m main.x65.student train --recipe x66 --bank "$BANK" --bank-sha256 "$BSHA" --fold "$fold" \
            --lr 1e-4 --init "$PSTAR" --out "$X/students/$s" --device cuda --budget-min 50 >> "$LOG/$s.log" 2>&1
        rc=$?
        say "$s try $try: rc=$rc"
        [ "$rc" -eq 0 ] && { finished=1; break; }
        [ "$rc" -eq 3 ] || hold "$s exited rc=$rc (see $LOG/$s.log)"
    done
    [ "$finished" -eq 1 ] || hold "$s not finished after 6 chunks"
done
say "DONE: every student trained; the read operator runs gate.sh (CPU)"
