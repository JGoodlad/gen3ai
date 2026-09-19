#!/usr/bin/env bash
# THE THREE TREES, all of the FROZEN POLICY `ai_v13_02_flywheel_winprob/final_model.zip` (arm W,
# step 75,005,952) — CPU only, nothing written under models/.
#
#  train  self-play, arm W's final posed as the cycle's SOLE sentinel  -> the branched AlphaGo set
#  eval   the same, a DIFFERENT seed and different games               -> the contested read set
#  guard  arm W's REAL run dir, the default bot roster + 3 pool sentinels -> the frame
#         `cond.opp_class_auc.t4_10` is computed on (a bots-free tree has ONE opponent class and
#         the AUC is undefined; `refit.conditioning_frame` also needs a `plan.json`, which a LIVE
#         eval tree does not have).
#
# 🚨 DECLARED DEVIATION — the SELF-PLAY posing. `eval_trace_gen` draws sentinels from the run's own
# `snapshots/`, and the newest of those is 72,000,000, not the frozen 75,005,952 final. A SHADOW
# SOURCE RUN under $T/src_run therefore carries arm W's `model_config.json` and `metadata.json`
# byte-for-byte, `eval_traces/step_75005952/snapshot.zip` -> arm W's `final_model.zip`, and
# `snapshots/snapshot_000075005952.zip` -> the SAME file, with arm W's own `model_config.json`
# beside it. With `--sentinels 1 --include-current-snapshot` the cycle's sole opponent IS the
# frozen policy, so both seats are the frozen policy and both are GREEDY (the run's own RECORDED
# `eval_sentinel_greedy: true`). The regime is read from the record, never declared.
#
#   usage: run_trees.sh [tmp] [train_games] [eval_games] [guard_games] [workers]
set -uo pipefail
MAIN=/home/goodlad/dev/gen3ai
T=${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit}
NTR=${2:-6000}
NEV=${3:-1300}
NGU=${4:-80}
W=${5:-6}
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
RUN=$MAIN/models/ai_v13_02_flywheel_winprob
SRC=$T/src_run/ai_v13_02_flywheel_winprob
export PYTHONPATH=$MAIN/src
export POKESIM_SIM_BRIDGE_BIN=$MAIN/src/rust_sim/target/release/sim_bridge
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$MAIN" || exit 1

# --- the shadow source run (the self-play posing) -----------------------------------------
rm -rf "$T/src_run"
mkdir -p "$SRC/snapshots" "$SRC/eval_traces/step_75005952"
cp "$RUN/model_config.json" "$RUN/metadata.json" "$SRC/"
cp "$RUN/model_config.json" "$SRC/snapshots/"
ln -sf "$RUN/final_model.zip" "$SRC/eval_traces/step_75005952/snapshot.zip"
ln -sf "$RUN/final_model.zip" "$SRC/snapshots/snapshot_000075005952.zip"

gen () { # out  games  seed  extra...
  local O=$1 G=$2 S=$3; shift 3
  echo "=== GEN $O games=$G seed=$S $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  nice -n 15 "$PY" -m main.ops.eval_trace_gen "$@" --out "$O" --games "$G" --seed "$S" \
      --workers "$W" --concurrency 1 --shard-games 25 --nice 15 --force \
      > "$O.gen.log" 2>&1
  echo "  exit=$? $(date -u +%FT%TZ)"; tail -3 "$O.gen.log"
}

gen "$T/tree_train" "$NTR" 20260918 "$SRC" --sentinels 1 --include-current-snapshot --opponents ,
gen "$T/tree_eval"  "$NEV" 20260919 "$SRC" --sentinels 1 --include-current-snapshot --opponents ,
gen "$T/tree_guard" "$NGU" 20260920 "$RUN@74000016" --sentinels 3
echo "=== TREES DONE $(date -u +%FT%TZ)"
