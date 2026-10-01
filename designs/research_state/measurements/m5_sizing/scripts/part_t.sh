#!/usr/bin/env bash
# REGISTRATION Part T driver (M5 sizing study). Resumable: a unit with an "ok" row in $STATE/status is skipped.
#   part_t.sh update            (T-c) the update cost at fp32 highest, 2 units
#   part_t.sh n <N>             (T-b) then (T-a) at N; (T-b) re-run when the rule's buckets differ
#   part_t.sh nstar <N>         (T-a') serial vs _p8 vs _p1 at N*
#   part_t.sh abq <N>           (T-a) re-run on a QUIET box (load1 < 3 before the unit), *_quiet.json
#   part_t.sh threads <N>       addendum: serial at 8 / 12 / 16 core threads (quiet, gate slot held)
# Every GPU unit: scripts/ops/gpu_lock.sh timeout <=1200 inside, under scripts/ops/mem_cap.sh.
set -u
WT=/home/goodlad/dev/gen3ai-wt/m5-sizing
K=$WT/designs/research_state/measurements/m5_sizing
RES=$K/results
STATE=/home/goodlad/.cache/gen3ai/tmp/sizing/part_t
PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
R=/home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix
BUF=/home/goodlad/gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl
mkdir -p "$RES" "$STATE"
cd "$WT" || exit 2
export PYTHONPATH=$WT/src

done_unit() { grep -q "^$1 ok" "$STATE/status" 2>/dev/null; }
mark() { echo "$1 $2 rc=$3 at=$(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$STATE/status"; }
gpu() {  # gpu <name> <timeout_s> <cmd...>
    local name=$1 t=$2; shift 2
    scripts/ops/mem_cap.sh --name "sz_$name" 48 scripts/ops/gpu_lock.sh timeout "$t" nice -n 5 "$@" \
        > "$STATE/$name.log" 2>&1
}
gpuq() {  # gpuq <name> <timeout_s> <cmd...>: as gpu(), HOLDING the gate_lock slot for the unit (orchestrator
          # 2026-10-01: routine gates queue behind a quiet-box unit instead of colliding with it)
    local name=$1 t=$2; shift 2
    scripts/ops/gate_lock.sh scripts/ops/mem_cap.sh --name "sz_$name" 48 scripts/ops/gpu_lock.sh timeout "$t" \
        nice -n 5 "$@" > "$STATE/$name.log" 2>&1
}
COMMON=(--inference learner --ckpt "$R/final_model.zip" --device cuda --opponent production
        --collector complete_game --pool "$R/snapshots" --pool-size 20 --self-play-fraction 0.95 --target 98304)

case "${1:-}" in
update)
    for u in U1 U2; do
        done_unit "update_$u" && continue
        gpu "update_$u" 1200 "$PY" -m main.compile_inventory run --stage time --device cuda \
            --matmul-precision highest --buffer "$BUF" --keep-prewarm --unbracketed --worker-timeout-min 19 \
            --out-root "$STATE/update_$u"
        rc=$?; [ $rc -eq 0 ] && mark "update_$u" ok $rc || { mark "update_$u" fail $rc; exit $rc; }
    done ;;
n)
    N=$2
    PROV=$("$PY" "$K/scripts/bucket_rule.py" provisional "$N")
    if ! done_unit "split1_n$N"; then
        gpu "split1_n$N" 1200 "$PY" -m main.rust_core_m5.fanout --out "$STATE/split1_n$N.json" \
            --pool "$R/snapshots" --ckpt "$R/final_model.zip" --pool-size 20 --self-play-fraction 0.95 \
            --n-envs "$N" --threads 8 --slots "" --lanes 0 --warmup-steps 300 --rounds 6 --real-flushes 24 \
            --real-reps 5 --t2-buckets "$PROV"
        rc=$?; [ $rc -eq 0 ] && mark "split1_n$N" ok $rc || { mark "split1_n$N" fail $rc; exit $rc; }
    fi
    RULE=$("$PY" "$K/scripts/bucket_rule.py" rule "$STATE/split1_n$N.json")
    echo "$RULE" > "$RES/buckets_n$N.json"
    BK=$("$PY" -c 'import json,sys; print(",".join(str(b) for b in json.loads(sys.argv[1])["buckets"]))' "$RULE")
    if [ "$BK" = "$PROV" ]; then
        cp "$STATE/split1_n$N.json" "$RES/split_n$N.json"
    elif ! done_unit "split2_n$N"; then
        gpu "split2_n$N" 1200 "$PY" -m main.rust_core_m5.fanout --out "$STATE/split2_n$N.json" \
            --pool "$R/snapshots" --ckpt "$R/final_model.zip" --pool-size 20 --self-play-fraction 0.95 \
            --n-envs "$N" --threads 8 --slots "" --lanes 0 --warmup-steps 300 --rounds 6 --real-flushes 24 \
            --real-reps 5 --t2-buckets "$BK"
        rc=$?; [ $rc -eq 0 ] && mark "split2_n$N" ok $rc || { mark "split2_n$N" fail $rc; exit $rc; }
        cp "$STATE/split2_n$N.json" "$RES/split_n$N.json"
    fi
    if ! done_unit "ab_n$N"; then
        gpu "ab_n$N" 1200 "$PY" -m main.rust_core_m5 throughput --out "$RES/throughput_n$N.json" \
            --n-envs "$N" --threads 8 --arms rust_serial_keyed,rust_overlap_keyed --pairs 6 --block-seconds 20 \
            --warmup-steps 120 "${COMMON[@]}" --t2-buckets "$BK"
        rc=$?; [ $rc -eq 0 ] && mark "ab_n$N" ok $rc || { mark "ab_n$N" fail $rc; exit $rc; }
    fi ;;
nstar)
    N=$2
    BK=$("$PY" -c 'import json,sys; print(",".join(str(b) for b in json.load(open(sys.argv[1]))["buckets"]))' "$RES/buckets_n$N.json")
    if ! done_unit "nstar_n$N"; then
        gpu "nstar_n$N" 1200 "$PY" -m main.rust_core_m5 throughput --out "$RES/throughput_nstar_n$N.json" \
            --n-envs "$N" --threads 8 --arms rust_serial_keyed,rust_serial_keyed_p8,rust_serial_keyed_p1 \
            --pairs 6 --block-seconds 20 --warmup-steps 120 "${COMMON[@]}" --t2-buckets "$BK"
        rc=$?; [ $rc -eq 0 ] && mark "nstar_n$N" ok $rc || { mark "nstar_n$N" fail $rc; exit $rc; }
    fi ;;
threads)
    # PROGRESS addendum (owner's question, not a registered arm): serial at N on 8 / 12 / 16 core threads
    N=$2
    BK=$("$PY" -c 'import json,sys; print(",".join(str(b) for b in json.load(open(sys.argv[1]))["buckets"]))' "$RES/buckets_n$N.json")
    if ! done_unit "threads_n$N"; then
        end=$((SECONDS + 3600))
        until awk '{exit !($1 < 3.0)}' /proc/loadavg; do
            [ $SECONDS -ge $end ] && { mark "threads_n$N" noquiet 9; exit 9; }
            sleep 30
        done
        gpuq "threads_n$N" 1200 "$PY" -m main.rust_core_m5 throughput --out "$RES/throughput_threads_n$N.json" \
            --n-envs "$N" --threads 8 --arms rust_serial_keyed,rust_serial_keyed_t12,rust_serial_keyed_t16 \
            --pairs 6 --block-seconds 20 --warmup-steps 120 "${COMMON[@]}" --t2-buckets "$BK"
        rc=$?; [ $rc -eq 0 ] && mark "threads_n$N" ok $rc || { mark "threads_n$N" fail $rc; exit $rc; }
    fi ;;
abq)
    # QUIET re-run of (T-a) at N (PROGRESS D-5): wait, bounded (60 min), for load1 < 3 before the unit
    N=$2
    BK=$("$PY" -c 'import json,sys; print(",".join(str(b) for b in json.load(open(sys.argv[1]))["buckets"]))' "$RES/buckets_n$N.json")
    if ! done_unit "abq_n$N"; then
        end=$((SECONDS + 3600))
        until awk '{exit !($1 < 3.0)}' /proc/loadavg; do
            [ $SECONDS -ge $end ] && { mark "abq_n$N" noquiet 9; exit 9; }
            sleep 30
        done
        echo "abq_n$N quiet at $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$STATE/status"
        gpuq "abq_n$N" 1200 "$PY" -m main.rust_core_m5 throughput --out "$RES/throughput_n${N}_quiet.json" \
            --n-envs "$N" --threads 8 --arms "${ARMS:-rust_serial_keyed,rust_overlap_keyed}" --pairs 6 \
            --block-seconds 20 --warmup-steps 120 "${COMMON[@]}" --t2-buckets "$BK"
        rc=$?; [ $rc -eq 0 ] && mark "abq_n$N" ok $rc || { mark "abq_n$N" fail $rc; exit $rc; }
    fi ;;
*) sed -n 2,6p "$0"; exit 2 ;;
esac
