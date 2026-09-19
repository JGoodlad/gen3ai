#!/usr/bin/env bash
# THE READ — both controls, every column, plus rule 25's indexing clause and the omniscience
# post-hoc. Everything lands in the job tmp; the JSON copies committed beside this file are what
# the README quotes.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L=/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit
O=/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_control
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH="${PYTHONPATH:-}:/home/goodlad/dev/gen3ai/src"
export CUDA_VISIBLE_DEVICES=""
SNAP="$L/src_run/ai_v13_02_flywheel_winprob/eval_traces/step_75005952/snapshot.zip"

# rule 25's indexing clause, by the 2026-09-18 verifier, invoked BY PATH and unmodified
nice -n 15 "$PY" "$HERE/../exploiter_discrimination_2026-09-18/verify_indexing.py" \
    --forks "$L/forks_eval" --snapshot "$SNAP" --expect 0.5723320246 \
    --out "$O/verify_indexing.json" --show 4

nice -n 15 "$PY" "$HERE/posthoc_omni.py" --hand "$O/hand" --out "$O/posthoc_omni.json"

nice -n 15 "$PY" "$HERE/score_controls.py" \
    --forks "$L/forks_eval" --snapshot "$SNAP" \
    --hand "$O/hand" --reroll "$O/reroll" --fit "$L/fit" \
    --out "$O/score.json" --threads 4
echo "[run_score] -> $O/score.json"
