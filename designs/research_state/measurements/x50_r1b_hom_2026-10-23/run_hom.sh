#!/usr/bin/env bash
# X50 rung 1b use (b), the HUMAN OPPONENT MODEL (design_x50_rung1b_human_opponent_model.md §3-§4, §8.2):
#   genc      G-ENC at this commit on R1's traces (a FAIL stops everything)
#   train     the training dates 2026-05-18..2026-09-20, completed rows only, the team-memory exclusion (F-1b-2);
#             long lane, <= 45-min resumable chunks (a finished part is skipped)
#   heldout   the held-out dates 09-21..23 re-extracted at this commit (the Mimic fix), completed rows only, no
#             exclusion; short lane. Nothing trains or selects on them.
# CPU only, nice 19 on the agent cores, one heavy slot.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
MAIN=$(git -C "$REPO" worktree list | head -1 | awk '{print $1}')
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
OUT=${X50R1B_OUT:-/home/goodlad/gen3ai_archive/x50_r1b}
TEAMS="$REPO/designs/research_state/measurements/heldout72_hpfix_kakuna_2026-10-18/heldout72_hpfix_manifest.json"
CORPUS="$MAIN/replays/showdown/gen3ou"
export PYTHONPATH="$REPO/src" CUDA_VISIBLE_DEVICES=""
CPU=(nice -n 19 taskset -c 5-7,13-15)
short() { "$REPO/scripts/ops/mem_cap.sh" --name "x50r1b_$1" --slot --max-s 600 8 "${CPU[@]}" "${@:2}"; }
long() { "$REPO/scripts/ops/mem_cap.sh" --name "x50r1b_$1" --slot 12 timeout 2700 "${CPU[@]}" "${@:2}"; }
step="${1:?genc | train | heldout}"
mkdir -p "$OUT/hom_build"
case "$step" in
genc)
  short genc "$PY" -m main.x50_r1.genc --run "$MAIN/models/rb_x26_s2601" --min-step 360000000 \
      --out "$OUT/hom_build/genc.json" ;;
train)
  "$PY" -c "import json,sys; sys.exit(json.load(open(sys.argv[1]))['verdict'] != 'PASS')" "$OUT/hom_build/genc.json" \
      || { echo "[x50r1b] G-ENC has not passed: nothing is extracted"; exit 2; }
  for chunk in $(seq 1 12); do
    long train_ext "$PY" -m main.x50_r1.run humans --complete --rows completed --exclude-teams "$TEAMS" \
        --dates 2026-05-18..2026-09-20 --corpus "$CORPUS" --out "$OUT/train" --workers 3 \
        --deadline "$(( $(date +%s) + 2400 ))"
    rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -eq 3 ] || { echo "[x50r1b] train extraction chunk $chunk failed rc=$rc"; exit $rc; }
  done ;;
heldout)
  short heldout "$PY" -m main.x50_r1.run humans --complete --rows completed \
      --dates 2026-09-21,2026-09-22,2026-09-23 --corpus "$CORPUS" --out "$OUT/heldout" --workers 3 ;;
*) echo "unknown step $step"; exit 2 ;;
esac
rc=$?
echo "[x50r1b] $(date -Is) $step rc=$rc"
exit $rc
