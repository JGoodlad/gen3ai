#!/usr/bin/env bash
# X35 BATCH 1 + X47 — build and VALIDATE every launch argv (BATCH1_LAUNCH_SHEET.md §4). It NEVER launches: for each
# <arm>:<seed> it builds the argv with main.x35.kit (X47 is the kit arm `toy_x47` since F-LS-1: the toy C0 argv + the ONE
# lever `--grow trunk_layers=4`, run name rb_x35t_x47d4_s<seed>), runs `main.checkargs --pin` and
# `main.launcher --dry-run` at the pin, CPU only, rule 5b, with GEN3AI_MODELS_DIR pointed at a scratch dir so nothing
# can land under models/, then asks the kit for the VERDICT (`main.x35.kit validate-run`: both tools exit 0, "would
# launch", FORK of the start node's run, the arm's +steps and K9(b) mode, the trainer's arch check, and the DECLARED
# growth line for X47 and for no other arm), and `kit corpus-check` (AMENDMENT D1-CORPUS: D1's team corpus AT the pin).
#
#   usage: validate_batch1.sh --pin <PIN> --toy-call <reference_population_toy.json> --out <scratch dir>
#                             [--reader-standin <model zip>] <arm>:<seed> ...
#   arms:  a1 c0 d1 a2 (production; read reference_population_production.json beside this script)
#          toy_a1 toy_c0 toy_a2 toy_x47 (toy; read --toy-call; `x47` is accepted for `toy_x47`)
#          ge_a1 ge_a2 ge_d1 ge_c0 (the G-E READERS, main.x35.ge; seed = the TARGET's, 3501). Their target is an arm's
#          FINISHED final, which does not exist before the arm ends, so a validation names a STAND-IN target model
#          (--reader-standin, e.g. P*): printed as `target=STANDIN:<file>` on the line, never a launch value
#          toy_r0:3631 r0:3531 (R0, AM-R0, main.x35.r0: the start node is the target, so no stand-in)
# One line per argv on stdout, ending `verdict=PASS` or `verdict=FAIL <why>`, then `SUMMARY <n> PASS, <m> FAIL`; the exit
# status is the number of FAILs (capped at 100). A --toy-call whose name says STANDIN is printed as `call=…STANDIN…` on
# every toy line: a stand-in for T* is never a launch value. Transcripts: <out>/{argv,checkargs,dryrun,corpus}_<arm>_s<seed>.txt.
set -u
STAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$STAGE" rev-parse --show-toplevel)
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
PIN=""; RTOY=""; OUT=""; SPECS=(); RSTANDIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --pin) PIN=$2; shift 2;; --toy-call) RTOY=$2; shift 2;; --out) OUT=$2; shift 2;;
    --reader-standin) RSTANDIN=$2; shift 2;;
    -*) echo "unknown flag $1" >&2; exit 2;;
    *) SPECS+=("$1"); shift;;
  esac
done
[ -n "$PIN" ] && [ -n "$OUT" ] && [ "${#SPECS[@]}" -gt 0 ] || { sed -n 2,17p "$0" >&2; exit 2; }
case "$(realpath -m "$OUT")" in */models|*/models/*|*/data|*/data/*) echo "--out must not be under models/ or data/" >&2; exit 2;; esac
PIN=$(git -C "$REPO" rev-parse --verify "$PIN^{commit}") || exit 2
RPROD=$STAGE/reference_population_production.json
mkdir -p "$OUT/models"
export PYTHONPATH=$REPO/src CUDA_VISIBLE_DEVICES="" GEN3AI_MODELS_DIR=$OUT/models
cd "$REPO" || exit 2
CPU=(nice -n 19 taskset -c 5-7,13-15)
NFAIL=0; NPASS=0
run() {  # arm seed
  local arm=$1 seed=$2 call=$RPROD tag argv tgt=()
  [ "$arm" = x47 ] && arm=toy_x47
  tag=${arm}_s$seed
  case "$arm" in toy_*) call=$RTOY; [ -n "$RTOY" ] || { echo "$tag: --toy-call is required verdict=FAIL"; NFAIL=$((NFAIL+1)); return; };; esac
  case "$arm" in ge_*) [ -n "$RSTANDIN" ] || { echo "$tag: --reader-standin is required verdict=FAIL"; NFAIL=$((NFAIL+1)); return; }
                       tgt=(--target-standin "$RSTANDIN");; esac
  argv=$("$PY" -c '
import json, shlex, sys
from main.x35.kit import describe, reader_target, ARMS
arm, seed, call = sys.argv[1], int(sys.argv[2]), json.load(open(sys.argv[3]))
t = reader_target(arm, seed, call, standin=sys.argv[4]) if len(sys.argv) > 4 else None
print(" ".join(shlex.quote(x) for x in describe(arm, seed, call, target=t)["argv"]))' "$arm" "$seed" "$call" \
    ${tgt[1]:+"${tgt[1]}"}) \
    || { echo "$tag BUILD FAIL verdict=FAIL"; NFAIL=$((NFAIL+1)); return; }
  echo "$argv" > "$OUT/argv_$tag.txt"
  timeout 900 "${CPU[@]}" "$PY" -m main.checkargs --pin "$PIN" --argv "$argv" > "$OUT/checkargs_$tag.txt" 2>&1
  local rc1=$?
  # shellcheck disable=SC2086
  timeout 900 "${CPU[@]}" "$PY" -m main.launcher --dry-run --restart-interval-hours 24 --pin-commit "$PIN" $argv \
    > "$OUT/dryrun_$tag.txt" 2>&1
  local rc2=$? line
  line=$("$PY" -m main.x35.kit validate-run --arm "$arm" --seed "$seed" --call "$call" --dryrun "$OUT/dryrun_$tag.txt" \
    --checkargs "$OUT/checkargs_$tag.txt" --rc-dryrun "$rc2" --rc-checkargs "$rc1" "${tgt[@]}" 2>&1 | tail -1)
  # AMENDMENT D1-CORPUS (2026-10-20): the arm's team corpus AT the pin is its registered identity (launch pre-check 2d)
  "$PY" -m main.x35.kit corpus-check --arm "$arm" --pin "$PIN" --repo "$REPO" > "$OUT/corpus_$tag.txt" 2>&1 \
    || line="${line%%verdict=*}verdict=FAIL corpus-check: $(tail -1 "$OUT/corpus_$tag.txt")"
  case "$line" in *"verdict=PASS"*) NPASS=$((NPASS+1));; *) NFAIL=$((NFAIL+1));; esac
  case "$arm" in toy_*) line="$line call=$(basename "$call")";; ge_*) line="$line target=STANDIN:$(basename "$RSTANDIN")";; esac
  echo "$line"
}
for spec in "${SPECS[@]}"; do run "${spec%:*}" "${spec#*:}"; done
echo "SUMMARY $NPASS PASS, $NFAIL FAIL (pin ${PIN:0:8})"
[ "$NFAIL" -gt 100 ] && NFAIL=100
exit "$NFAIL"
