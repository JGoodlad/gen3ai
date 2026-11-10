#!/usr/bin/env bash
# X66 ExIt-0H (PREREG §4, §10): the CPU steps after gpu_students.sh, run by the read operator under rule 5b (nice 19
# on the agent cores, the heavy slots). In order: pick (val Δagree) -> export at the play pin (bit-identity) ->
# delivery score on each held-out panel -> the gate -> the by-hand check -> the learnability report (REPORTED). Any
# failure is a HOLD. No Kakuna game: on DELIVERED the read operator next fills the specs (the students amendment).
#   gate.sh
set -u
PIN_SHA=c0b95914e239ff8c66da5a06afe9b42a43d41cf6   # the side pin (PREREG §5.1)
PINWT=/home/goodlad/dev/gen3ai-reads/x66_pin
PLAYWT=/home/goodlad/dev/gen3ai-reads/x66_play_e46e3ddf8  # the play pin e46e3ddf8 (S'-HONEST's)
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
X=/home/goodlad/gen3ai_archive/x66_exit0h_2026-11-09
BANK=$X/bank
BSHA=0812768c9fcecf586c5973e23ba600cc9b3da99a704b2aeab567e89e7a185344   # the registered bank (PREREG Amendment 1)
PSTAR=/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=$X/gate
mkdir -p "$LOG" "$X/delivery"
say() { echo "[$(date -Is)] X66G: $*" | tee -a "$LOG/gate.log"; }
hold() { say "HOLD: $*"; exit 1; }
E6="port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130"   # S'-HONEST's engine (PREREG §5.1); a fresh worktree can miss it
eng() { (cd "$1" && PYTHONPATH="$1/src" CUDA_VISIBLE_DEVICES="" "$PY" -c \
    "from main.anchors.server import engine_stamp as e; print(e())" 2>/dev/null | tail -1); }
need_e6() { local got; got=$(eng "$1"); [ "$got" = "$E6" ] || hold "the engine at $1 is '$got', not E6 $E6 (a missing \
src/rust_env/Cargo.lock or release lib: re-run bootstrap.sh --with-rust there; never copy a lock in without recording it)"; }
case "$BSHA" in *PENDING*) hold "the bank sha256 is still PENDING" ;; esac
cd "$PINWT" || hold "no $PINWT"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "$PIN_SHA^{commit}")" ] || hold "$PINWT is not at the pin"
[ "$(git -C "$PLAYWT" rev-parse --short=9 HEAD)" = "e46e3ddf8" ] || hold "$PLAYWT is not at the play pin e46e3ddf8"
need_e6 "$PINWT"
need_e6 "$PLAYWT"
export PYTHONPATH="$PINWT/src" CUDA_VISIBLE_DEVICES=""
C="nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh --name x66gate 16"
run() { say "$*"; $C "$@" >> "$LOG/gate.log" 2>&1 || hold "rc=$? from: $*"; }
for spec in "heldout72_hpfix s_h" "forum72 s_f" "p72 s_p"; do
    set -- $spec
    run "$PY" -m main.x65.student pick --recipe x66 --out "$X/students/$2"
    run "$PY" -m main.x65.export write --fold-dir "$X/students/$2" --init "$PSTAR" --pin-root "$PLAYWT"
    run "$PY" -m main.x65.delivery score --bank "$BANK" --bank-sha256 "$BSHA" --fold "$1" \
        --export "$X/students/$2/model/export.json" --init "$PSTAR" --out "$X/delivery/$1.npz"
done
run "$PY" -m main.x65.delivery gate --scores "$X/delivery" --bank-sha256 "$BSHA" --out "$X/delivery.json"
say "by hand"
$C "$PY" "$HERE/delivery_by_hand.py" "$X/delivery" "$X/delivery.json" > "$X/delivery_by_hand.json" 2>> "$LOG/gate.log" \
    || hold "the by-hand check disagrees with the gate (see $X/delivery_by_hand.json)"
run "$PY" -m main.x65.learnability report --out "$X/learnability_report.json" \
    --arm s_h="$X/students/s_h/lr0.0001" --arm s_f="$X/students/s_f/lr0.0001" --arm s_p="$X/students/s_p/lr0.0001"
say "DONE: $(python3 -c "import json;d=json.load(open('$X/delivery.json'));print({k:d[k] for k in ('pooled','lb','delivered','boundary')})")"
