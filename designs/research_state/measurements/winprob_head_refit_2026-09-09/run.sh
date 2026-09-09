#!/usr/bin/env bash
# Reproduce the head refit end to end. Read-only over models/; CPU only, no GPU, no server.
# Fixed seeds throughout; every number in README.md comes out of these calls.
#
# ~2 min of head extraction + ~15 min of fits + ~40 min of readouts on two niced cores.
# Nothing is written under models/.
#
# 🚨 THE FEATURES ARE NOT RE-EXTRACTED HERE. This measurement reuses the probe read's frozen
# forward verbatim — `pooled.npy` (`stash.value_pooled`, the tensor the win head literally reads)
# and `meta.npy` (outcome, turn, opponent, team, HT weight, the recorded and re-forwarded V). Run
# `../winprob_probe_read_2026-09-09/run.sh $PROBE` first if $PROBE is empty; re-deriving the
# forward here would risk a second, subtly different feature tensor and make the two measurements
# incomparable for no gain.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(git rev-parse --show-toplevel)/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
NICE="nice -n 10"
PROBE="${PROBE:-/home/goodlad/.claude/jobs/9ab51de6/tmp/probe}"
TMP="${1:-/home/goodlad/.claude/jobs/9ab51de6/tmp/refit}"
M=/home/goodlad/dev/gen3ai/models
mkdir -p "$TMP"

A=$M/ai_v12_02_winprob_critic
C=$M/ai_v12_11_ladder_ctrl10M

# 1. lift the ONLINE win head out of each snapshot — and REFUSE unless
#    sigmoid(win_head(value_pooled)) reproduces the frozen forward's own V. The whole
#    measurement rests on that identity.
$NICE $PY dump_head.py --snapshot "$A/eval_traces/step_74000016/snapshot.zip" \
    --dir "$PROBE/A"    --out "$TMP/head_A.npz"
$NICE $PY dump_head.py --snapshot "$C/eval_traces/step_10000032/snapshot.zip" \
    --dir "$PROBE/CTRL" --out "$TMP/head_CTRL.npz"

# 2. the seven conditions, out-of-fold, battle-grouped, HT-weighted
$NICE $PY refit.py --dir "$PROBE/A"    --head "$TMP/head_A.npz"    --out "$TMP/preds_A.npz"
$NICE $PY refit.py --dir "$PROBE/CTRL" --head "$TMP/head_CTRL.npz" --out "$TMP/preds_CTRL.npz"

# 3. the four meters + every delta's own battle-clustered CI
$NICE $PY readout.py --dir "$PROBE/A"    --preds "$TMP/preds_A.npz"    --out "$TMP/read_A.json"
$NICE $PY readout.py --dir "$PROBE/CTRL" --preds "$TMP/preds_CTRL.npz" --out "$TMP/read_CTRL.json"

# 4. the counter-hypotheses, each as a WHOLE re-run of the pipeline rather than a footnote
#    (a) overfitting to ~1k battles: the same readout on the IN-FOLD prediction columns
$NICE $PY readout.py --dir "$PROBE/A" --preds "$TMP/preds_A.npz" --train --no-decode \
    --out "$TMP/read_A_train.json"
#    (b) the HT weights: the entire fit + readout unweighted
$NICE $PY refit.py --dir "$PROBE/A" --head "$TMP/head_A.npz" --no-ipw --out "$TMP/preds_A_raw.npz"
$NICE $PY readout.py --dir "$PROBE/A" --preds "$TMP/preds_A_raw.npz" --no-decode \
    --out "$TMP/read_A_raw.json"
#    (c) label leakage through the conditional target: the literal per-(opponent, team) LOO cell
#        mean wherever the cell has >= 3 battles, in place of the additive one
$NICE $PY refit.py --dir "$PROBE/A" --head "$TMP/head_A.npz" --cond raw_cell \
    --out "$TMP/preds_A_cell.npz"
$NICE $PY readout.py --dir "$PROBE/A" --preds "$TMP/preds_A_cell.npz" --no-decode \
    --out "$TMP/read_A_cell.json"

# 5. the committed summary (small JSON + the tables; no prediction or feature arrays)
$NICE $PY summarize.py --tmp "$TMP" --out-dir .
