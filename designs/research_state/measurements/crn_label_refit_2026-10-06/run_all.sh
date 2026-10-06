#!/usr/bin/env bash
# The detached driver (README §5): capture -> held-out truth (S = 32) -> train truth (S = 8). Every
# step is resumable, so re-running this script after a kill continues where it stopped.
#   setsid nohup ./run_all.sh <workers> <threads> > /dev/null 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../../.." && pwd)"
W="${1:-3}"; T="${2:-2}"
R="$HERE/rows"
echo "[run_all] start $(date -Is) workers=$W threads=$T" >> "$R/run_all.log"
(cd "$WT" && PYTHONPATH="$WT/src" nice -n 15 "$WT/scripts/ops/mem_cap.sh" --name crncap 10 timeout 3h \
    /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 "$HERE/capture.py" \
    --subset "$R/subset_held.json" --subset "$R/subset_train.json" --out "$R/cap" \
    --workers "$W" --threads "$T" >> "$R/capture.log" 2>&1)
echo "[run_all] capture exit $? $(date -Is)" >> "$R/run_all.log"
"$HERE/run_truth.sh" "$R/subset_held.json" 32 "$R/truth_held_S32.jsonl" "$W" "$T" "$R/truth_held.log"
echo "[run_all] held exit $? $(date -Is)" >> "$R/run_all.log"
"$HERE/run_truth.sh" "$R/subset_train.json" 8 "$R/truth_train_S8.jsonl" "$W" "$T" "$R/truth_train.log"
echo "[run_all] train exit $? $(date -Is)" >> "$R/run_all.log"
echo "[run_all] DONE $(date -Is)" >> "$R/run_all.log"
