#!/usr/bin/env bash
# One anchors cell, with the SERVER PROCESS TREE's RSS sampled beside it.
#
#   run_cell.sh <label> <server:rust|node> <out-root> [extra anchors args...]
#
# The RSS is the owner's stated reason for wanting off Node, so it is measured rather than
# asserted — and it is measured over the server's whole TREE, because `--server rust` backs each
# battle with a `sim_bridge` CHILD and a comparison that counted only the parent would flatter it.
set -u
LABEL=$1; SERVER=$2; ROOT=$3; shift 3
WT=/home/goodlad/dev/gen3ai-wt/anchors_rust
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
MODEL=/home/goodlad/dev/gen3ai/models/ai_v13_02_flywheel_winprob/final_model.zip
OUT=$ROOT/$LABEL
mkdir -p "$OUT"
cd "$WT" || exit 2
export PYTHONPATH="$WT/src"
export GEN3AI_MODELS_DIR=/home/goodlad/dev/gen3ai/models
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

echo "=== $LABEL START $(date -Is) server=$SERVER load=$(cut -d' ' -f1-3 /proc/loadavg)" \
    | tee "$OUT/meta.txt"
T0=$(date +%s)
nice -n 15 $PY -m main.anchors --server "$SERVER" --model "$MODEL" \
    --device cpu --out "$OUT" "$@" > "$OUT/cell.log" 2>&1 &
CELL=$!

# The PID is the tool's own announcement, so the sampler watches exactly the process the tool
# started — never a pgrep pattern, which is how a sampler ends up measuring somebody else's run.
SRV=""
for _ in $(seq 1 120); do
  SRV=$(grep -oP '(?<=pid=)\d+' "$OUT/cell.log" 2>/dev/null | head -1)
  [ -n "$SRV" ] && break
  kill -0 $CELL 2>/dev/null || break
  sleep 1
done
echo "server pid=$SRV" | tee -a "$OUT/meta.txt"
if [ -n "$SRV" ]; then
  ( while kill -0 "$SRV" 2>/dev/null; do
      TOT=0; N=0
      for p in "$SRV" $(cat /proc/$SRV/task/*/children 2>/dev/null); do
        R=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
        [ -n "$R" ] && { TOT=$((TOT+R)); N=$((N+1)); }
      done
      echo "$(date +%s) $TOT $N" >> "$OUT/rss.txt"
      sleep 2
    done ) &
  SAMP=$!
fi

wait $CELL; RC=$?
[ -n "${SAMP:-}" ] && kill "$SAMP" 2>/dev/null
T1=$(date +%s)
echo "=== $LABEL END $(date -Is) rc=$RC wall_s=$((T1-T0)) load=$(cut -d' ' -f1-3 /proc/loadavg)" \
    | tee -a "$OUT/meta.txt"
if [ -s "$OUT/rss.txt" ]; then
  awk '{s+=$2; if($2>m)m=$2; n++} END {printf "server tree RSS: peak %.1f MB  mean %.1f MB  samples %d\n", m/1024, s/n/1024, n}' \
      "$OUT/rss.txt" | tee -a "$OUT/meta.txt"
fi
grep -E '^  status|^  WIN RATE|^  transport|^  verified' "$OUT/cell.log" | tee -a "$OUT/meta.txt"
exit $RC
