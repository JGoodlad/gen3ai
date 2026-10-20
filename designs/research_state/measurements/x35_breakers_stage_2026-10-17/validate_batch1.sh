#!/usr/bin/env bash
# X35 BATCH 1 + X47 — build and VALIDATE every launch argv (BATCH1_LAUNCH_SHEET.md §4). It NEVER launches: for each
# <arm>:<seed> it builds the argv with main.x35.kit (X47 = toy_c0's argv + `--grow trunk_layers=4`, run name
# rb_x35t_x47d4_s<seed>), then runs `main.checkargs --pin` and `main.launcher --dry-run` at the pin, CPU only, rule 5b,
# with GEN3AI_MODELS_DIR pointed at a scratch dir so nothing can land under models/.
#
#   usage: validate_batch1.sh --pin <PIN> --toy-call <reference_population_toy.json> --out <scratch dir> <arm>:<seed> ...
#   arms:  a1 c0 d1 a2 (production; read reference_population_production.json beside this script)
#          toy_a1 toy_c0 toy_a2 x47 (toy; read --toy-call)
# One line per argv on stdout: checkargs / dry-run exit codes, "would launch", the +steps, the REQUIRED GiB, the role
# and (X47) the DECLARED growth line. Transcripts: <out>/{argv,checkargs,dryrun}_<arm>_s<seed>.txt.
set -u
STAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$STAGE" rev-parse --show-toplevel)
PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}
PIN=""; RTOY=""; OUT=""; SPECS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pin) PIN=$2; shift 2;; --toy-call) RTOY=$2; shift 2;; --out) OUT=$2; shift 2;;
    -*) echo "unknown flag $1" >&2; exit 2;;
    *) SPECS+=("$1"); shift;;
  esac
done
[ -n "$PIN" ] && [ -n "$OUT" ] && [ "${#SPECS[@]}" -gt 0 ] || { sed -n 2,12p "$0" >&2; exit 2; }
case "$(realpath -m "$OUT")" in */models|*/models/*|*/data|*/data/*) echo "--out must not be under models/ or data/" >&2; exit 2;; esac
PIN=$(git -C "$REPO" rev-parse --verify "$PIN^{commit}") || exit 2
RPROD=$STAGE/reference_population_production.json
mkdir -p "$OUT/models"
export PYTHONPATH=$REPO/src CUDA_VISIBLE_DEVICES="" GEN3AI_MODELS_DIR=$OUT/models
cd "$REPO" || exit 2
CPU=(nice -n 19 taskset -c 5-7,13-15)
run() {  # arm seed
  local arm=$1 seed=$2 karm=$1 call=$RPROD tag=$1_s$2 argv
  case "$arm" in toy_*|x47) call=$RTOY; [ -n "$RTOY" ] || { echo "$tag: --toy-call is required"; return; };; esac
  [ "$arm" = x47 ] && karm=toy_c0
  argv=$("$PY" -c '
import json, shlex, sys
from main.x35.kit import build_argv
arm, karm, seed, call = sys.argv[1], sys.argv[2], int(sys.argv[3]), json.load(open(sys.argv[4]))
a = build_argv(karm, seed, call)
if arm == "x47":  # X47 = the toy C0 argv + the ONE lever (design_bot_ideas_screen.md 2.1)
    i = a.index("--seed"); a = a[:i] + ["--grow", "trunk_layers=4"] + a[i:]
    a[a.index("--run-name") + 1] = f"rb_x35t_x47d4_s{seed}"
print(" ".join(shlex.quote(x) for x in a))' "$arm" "$karm" "$seed" "$call") || { echo "$tag BUILD FAIL"; return; }
  echo "$argv" > "$OUT/argv_$tag.txt"
  timeout 900 "${CPU[@]}" "$PY" -m main.checkargs --pin "$PIN" --argv "$argv" > "$OUT/checkargs_$tag.txt" 2>&1
  local rc1=$?
  # shellcheck disable=SC2086
  timeout 900 "${CPU[@]}" "$PY" -m main.launcher --dry-run --restart-interval-hours 24 --pin-commit "$PIN" $argv \
    > "$OUT/dryrun_$tag.txt" 2>&1
  local rc2=$? f=$OUT/dryrun_$tag.txt
  echo "$tag checkargs=$rc1 dryrun=$rc2 would_launch=$(grep -c 'this command would launch' "$f")" \
       "delta=[$(grep -m1 -oE '\+[0-9,]+ steps' "$f")] gib=[$(grep -m1 -oE '[0-9.]+ GiB REQUIRED' "$f")]" \
       "role=[$(grep -m1 'role        :' "$f" | sed 's/^ *role *: *//')]" \
       "k9b=[$(grep -m1 'K9(b) check' "$f" | sed 's/^ *K9(b) check *: *//' | cut -c1-20)]" \
       "grow=[$(grep -m1 '^  grow        :' "$f" | sed 's/^ *grow *: *//' | cut -c1-60)]"
}
for spec in "${SPECS[@]}"; do run "${spec%:*}" "${spec#*:}"; done
