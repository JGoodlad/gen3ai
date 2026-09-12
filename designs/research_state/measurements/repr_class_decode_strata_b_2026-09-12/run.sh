#!/usr/bin/env bash
# `strata_b` AT THE REPRESENTATION, end to end. Offline, CPU only, niced, nothing under models/.
#
#   0. gen         strata_b's hp800b draw (seed 20260911) did not exist. Generated with the SAME
#                  `main.ops.eval_trace_gen` invocation `hp_eval/hp800b.sh` used for every other
#                  run on that draw token. Its hp800 draw (seed 20260910) already existed and is
#                  used as-is.
#   1. extract     one frozen forward of `ai_v12_24_ladder_strata_b@step_10000032` over each of the
#                  two eval trees -> meta.npy + pooled.npy (`value_pooled`, the win head's literal
#                  input). `../winprob_refit_ncurve_2026-09-10/extract.py`, UNMODIFIED.
#   2. frame_check the out-of-fold, battle-grouped decode: `pooled_to_opp_class_AUC` and
#                  `V_to_opp_class_AUC` on THE SAME ROWS AND FOLDS, plus the own-team leak.
#                  `../winprob_refit_ncurve_2026-09-10/frame_check.py`, UNMODIFIED.
#   3. merge       the two new frames are MERGED into the NINE frames
#                  `../repr_class_decode_2026-09-11/frame_check.json` already holds. Those nine are
#                  REUSED VERBATIM and never recomputed — `frame_check.py` treats each extraction
#                  dir independently, so a merge cannot change an existing row.
#   4. tabulate    PREDICTION.md §3's TRUNK / HEAD / PARTIAL-TRUNK branch test, in code.
#
# 🚨 Run from the MAIN checkout's code (PYTHONPATH -> main's src). A worktree has no models/ and
# the rust bridge binary lives in main's target/ — never build into it from a worktree.
set -euo pipefail
MAIN=/home/goodlad/dev/gen3ai
HERE="$(cd "$(dirname "$0")" && pwd)"
NC="$MAIN/designs/research_state/measurements/winprob_refit_ncurve_2026-09-10"
PARENT="$MAIN/designs/research_state/measurements/repr_class_decode_2026-09-11"
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3}
HP=${HP_EVAL:-/home/goodlad/.claude/jobs/9ab51de6/tmp/hp_eval}
OUT=${REPDECODE_OUT:-/home/goodlad/.claude/jobs/9ab51de6/tmp/repdecode_strata_b}
export PYTHONPATH="$MAIN/src"
export POKESIM_SIM_BRIDGE_BIN="$MAIN/src/rust_sim/target/release/sim_bridge"
export CUDA_VISIBLE_DEVICES=""
RUN=ai_v12_24_ladder_strata_b
mkdir -p "$OUT"

# ── 0. the hp800b draw (skipped if the tree is already there)
if [ ! -d "$HP/hp800b/$RUN/eval_traces/step_10000032" ]; then
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  nice -n 15 "$PY" -m main.ops.eval_trace_gen "$RUN@10000032" --games 800 --sentinels 3 \
    --out "$HP/hp800b/$RUN" --workers 4 --concurrency 1 --nice 15 --seed 20260911 \
    --shard-games 25 > "$HP/hp800b/$RUN.gen.log" 2>&1
  unset OMP_NUM_THREADS MKL_NUM_THREADS
fi

# ── 1. extract
for DRAW in hp800 hp800b; do
  T="$HP/$DRAW/$RUN/eval_traces/step_10000032"
  [ -d "$T" ] || { echo "MISSING TREE $RUN/$DRAW" >&2; exit 1; }
  D="$OUT/$RUN/$DRAW"
  [ -f "$D/pooled.npy" ] && { echo "have $RUN/$DRAW"; continue; }
  mkdir -p "$D"
  nice -n 15 "$PY" "$NC/extract.py" --tree "$DRAW=$T" --snapshot "$T/snapshot.zip" \
    --out-dir "$D" --threads 2 --batch 256
done

# ── 2. frame_check, ONLY on the two new frames
nice -n 15 "$PY" "$NC/frame_check.py" \
  --dir "$RUN@hp800=$OUT/$RUN/hp800" --dir "$RUN@hp800b=$OUT/$RUN/hp800b" \
  --out "$OUT/frame_check_strata_b.json" --cap 20000 --seed 20260910

# ── 3. merge the nine REUSED frames in front of the two new ones
"$PY" - "$PARENT/frame_check.json" "$OUT/frame_check_strata_b.json" \
       "$HERE/frame_check.json" <<'PYEOF'
import json, sys
old, new, dst = sys.argv[1:4]
a = json.load(open(old)); b = json.load(open(new))
assert not (set(a) & set(b)), f"a merge must not overwrite a reused frame: {set(a) & set(b)}"
json.dump({**a, **b}, open(dst, "w"), indent=1)
print(f"merged {len(a)} reused + {len(b)} new = {len(a) + len(b)} frames -> {dst}")
PYEOF

# ── 4. the branch test
nice -n 15 "$PY" "$HERE/tabulate.py" --in "$HERE/frame_check.json" \
  --out-json "$HERE/branch_test.json" --out-md "$HERE/TABLE.md"
