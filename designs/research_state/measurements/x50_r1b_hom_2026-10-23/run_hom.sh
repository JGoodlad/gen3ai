#!/usr/bin/env bash
# X50 rung 1b use (b), the HUMAN OPPONENT MODEL (design_x50_rung1b_human_opponent_model.md §3-§4, §8.2):
#   genc      G-ENC at this commit on R1's traces (a FAIL stops everything)
#   train     the training dates 2026-05-18..2026-09-20, completed rows only, the team-memory exclusion (F-1b-2);
#             long lane, <= 45-min resumable chunks (a finished part is skipped)
#   heldout   the held-out dates 09-21..23 re-extracted at this commit (the Mimic fix), completed rows only, no
#             exclusion; short lane. Nothing trains or selects on them.
#   compact   the HOM's rows (mode 2, own label, >= 2 legal, the actor >= 1500), split train / validation by battle
#   hom       the registered training from P*, <= 45-min resumable chunks until it finishes; then export (the HOM zip,
#             its config, the sidecar with its sha256): the model is FROZEN before any held-out row is read
#   rb1       R-b1 on the held-out rows (refuses unless the HOM's sha256 equals the sidecar's)
#   rb2       R-b2's scoring at L3's code commit 8fe4fb89 ($OFFPOOL_CODE), two judges, then its read
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
step="${1:?genc | train | heldout | compact | hom | rb1 | rb2}"
PSTAR="$MAIN/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip"
HOM="$OUT/hom/model/hom_x50r1b.zip"
SIDE="$OUT/hom/model/hom_sidecar.json"
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
compact)
  long compact "$PY" -m main.x50_r1.hom compact --parts "$OUT/train" --out "$OUT/hom/rows" ;;
hom)
  for chunk in $(seq 1 16); do
    # threads: speed only (torch's intra-op count; not the recipe). 3 = one per physical agent core: at 6 the
    # OpenMP barriers stalled 8-20x when other agents' jobs shared the cores (2026-10-24)
    long hom "$PY" -m main.x50_r1.hom train --rows "$OUT/hom/rows" --init "$PSTAR" --out "$OUT/hom" --budget-min 38 \
        --threads "${HOM_THREADS:-3}"
    rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -eq 3 ] || [ $rc -eq 124 ] || { echo "[x50r1b] hom chunk $chunk failed rc=$rc"; exit $rc; }
  done
  [ $rc -eq 0 ] || exit $rc
  "$PY" -m main.x50_r1.hom export --rows "$OUT/hom/rows" --init "$PSTAR" --out "$OUT/hom" \
      --config "$MAIN/models/rb_x26_s2601/model_config.json" ;;
rb1)
  long rb1 "$PY" -m main.x50_r1.read_rb1 --heldout "$OUT/heldout" --hom "$HOM" --sidecar "$SIDE" --pstar "$PSTAR" \
      --out "$HERE" ;;
rb2)
  CODE=${OFFPOOL_CODE:-/home/goodlad/.cache/gen3ai/offpool/code}
  for chunk in 1 2 3; do
    (cd "$CODE" && long rb2 "$PY" -I "$HERE/rb2_run.py" --code "$CODE" --hom "$HOM" --sidecar "$SIDE" \
        --out "$OUT/rb2_rows.jsonl" --budget-min 38) || exit $?
    n=$("$PY" -c "import sys; print(sum(1 for l in open(sys.argv[1]) if l.strip()))" "$OUT/rb2_rows.jsonl")
    [ "$n" -ge 320 ] && break
  done
  (cd "$CODE" && "$PY" -I "$HERE/rb2_read.py" --rows "$OUT/rb2_rows.jsonl" --train-parts "$OUT/train" \
      --out "$HERE/rb2_read.json") ;;
*) echo "unknown step $step"; exit 2 ;;
esac
rc=$?
echo "[x50r1b] $(date -Is) $step rc=$rc"
exit $rc
