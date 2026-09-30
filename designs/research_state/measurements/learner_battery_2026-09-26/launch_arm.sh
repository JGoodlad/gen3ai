#!/usr/bin/env bash
# LAUNCH one arm of the LEARNER BATTERY — a +8,060,928-step FORK of N0 (ai_v14_01_base, the new lineage's
# fresh base run) at its final_model.zip (75,005,952), frozen at the lineage's generalist rate.
#   C   ai_v14_02_lbat_ctrl  the control = the lineage's continuation block K1 (N0's recipe continued)
#   E5  ai_v14_03_lbat_e5    --n-epochs 5 at --fork-lr 2 D_g (the dose held at C's)
#   T32 ai_v14_04_lbat_t32   --matmul-precision high (TF32)
#   L95 ai_v14_05_lbat_l95   --policy-gae-lambda 0.95
# REGISTERED ORDER: C -> E5 -> T32 (futility look at its 11th rollout, §4.3) -> L95. One arm at a time.
#
# 🚨 PREPARED BUT NEVER EXECUTED by the session that wrote it. For the TRAINING RUN session, VERBATIM,
#    after the orchestrator's go AND after `scripts/build_argvs.py --final --dg <D_g>` has replaced the
#    __DG__ placeholders (the argv files as committed REFUSE to launch).
#    Registration: designs/research_state/learner_battery_2026-09-26.md
#
# Usage:  bash launch_arm.sh <C|E5|T32|L95> [--dry-run]
#         STANDIN=1 bash launch_arm.sh <ARM> --dry-run   # the stand-in argv (N0's periodic checkpoint);
#                                                         # REFUSED without --dry-run
#         ARGV_FILE=<file> [CONTROL_RUN=<run>] bash launch_arm.sh <ARM> [--dry-run]
#                          # launch an arm from ANOTHER argv file (a re-pinned rerun such as T32b); the
#                          # run name is READ from the argv, and CONTROL_RUN names the arm's registered
#                          # control when it is not argv_C.txt's run (T32b's matched control = C_fix).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/home/goodlad/dev/gen3ai
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
LOG_DIR="${LOG_DIR:-/home/goodlad/.claude/jobs/learner_battery_2026-09-26/launch}"
ARM="${1:?usage: launch_arm.sh <C|E5|T32|L95> [--dry-run]}"
MODE="${2:-}"
case "$ARM" in
  C|E5|T32|L95) ;;
  *) echo "FATAL: unknown arm $ARM"; exit 2 ;;
esac
PARENT_ZIP=models/ai_v14_01_base/final_model.zip
PARENT_STEP=75005952

# The run name and the control are READ, never hard-coded (2026-09-28: the re-pinned C_fix / T32b arms
# could not go through this script because it named ai_v14_0[2-5] literally).
argv_run_name() { tr -s ' \n' '\n\n' < "$1" | grep -A1 -x -- '--run-name' | tail -1; }
if [ "${STANDIN:-0}" = "1" ]; then
  [ "$MODE" = "--dry-run" ] || { echo "FATAL: STANDIN=1 is a validation mode; it never launches"; exit 2; }
  ARGV_FILE="$HERE/scripts/argv_${ARM}_STANDIN.txt"
elif [ -z "${ARGV_FILE:-}" ]; then
  ARGV_FILE="$HERE/argv_${ARM}.txt"
fi
[ -f "$ARGV_FILE" ] || { echo "FATAL: no argv at $ARGV_FILE"; exit 2; }
RUN="$(argv_run_name "$ARGV_FILE")"
[ -n "$RUN" ] || { echo "FATAL: $ARGV_FILE names no --run-name"; exit 2; }
CTRL="${CONTROL_RUN:-$(argv_run_name "$HERE/argv_C.txt")}"

export PYTHONPATH="${PYTHONPATH:-}:$REPO/src"
cd "$REPO"
ARGV="$(cat "$ARGV_FILE")"
grep -q "__DG__\|__2DG__" <<<"$ARGV" && {
  echo "FATAL: $ARGV_FILE still carries the D_g placeholder. D_g is the orchestrator's (registration §3,"
  echo "       FINDING F-1): run  scripts/build_argvs.py --final --dg <D_g>  first."; exit 2; }
grep -q -- "--run-name $RUN " <<<"$ARGV " || { echo "FATAL: $ARGV_FILE does not name --run-name $RUN"; exit 2; }
grep -q -- "--fork-lr-freeze" <<<"$ARGV" || { echo "FATAL: $ARGV_FILE is not frozen"; exit 2; }

if [ "${STANDIN:-0}" != "1" ]; then
  grep -q -- "--model $PARENT_ZIP " <<<"$ARGV " || { echo "FATAL: $ARGV_FILE does not fork $PARENT_ZIP"; exit 2; }
  [ -f "$PARENT_ZIP" ] || { echo "FATAL: $PARENT_ZIP does not exist — N0 has not finished."; exit 4; }
  got="$("$P" -c "import json; print(json.load(open('models/ai_v14_01_base/metadata.json')).get('num_timesteps'))")"
  [ "$got" = "$PARENT_STEP" ] || { echo "FATAL: N0 num_timesteps=$got, registered $PARENT_STEP."; exit 4; }
  grep -aq "Training complete" models/ai_v14_01_base/launcher_child.full.log || {
    echo "FATAL: N0's child log has no 'Training complete'."; exit 4; }
  # every arm's D_g must be the SAME number (E5 carries 2 D_g): read all four argv files
  "$P" - "$HERE" <<'PY' || exit 4
import shlex, sys
from pathlib import Path
k = Path(sys.argv[1]); lr = {}
for a in ("C", "E5", "T32", "L95"):
    t = shlex.split((k / f"argv_{a}.txt").read_text())
    lr[a] = float(t[t.index("--fork-lr") + 1]) / (2 if a == "E5" else 1)
    ep = int(t[t.index("--n-epochs") + 1])
    assert ep == (5 if a == "E5" else 10), (a, ep)
assert len({round(v, 12) for v in lr.values()}) == 1, f"D_g differs across arms: {lr}"
print(f"D_g = {lr['C']:.2e} on every arm (E5 at 2 D_g); dose lr x n_epochs identical")
PY
  if [ "$ARM" != "C" ] && [ "$RUN" != "$CTRL" ]; then
    grep -aq "Training complete" "models/$CTRL/launcher_child.full.log" 2>/dev/null || {
      echo "FATAL: the control $CTRL has not finished — the registered order is C first (both speed"
      echo "       endpoints and the T32 futility look read C's rollouts)."; exit 4; }
  fi
fi

echo "=== [$ARM] checkargs $(date -Is) ==="
nice -n 15 "$P" -m main.checkargs --argv "$ARGV"
echo "=== [$ARM] launcher --dry-run $(date -Is) ==="
# shellcheck disable=SC2086
nice -n 15 "$P" -m main.launcher --dry-run $ARGV

if [ "$MODE" = "--dry-run" ]; then
  echo "=== [$ARM] --dry-run requested: stopping here. Nothing was created. ==="
  exit 0
fi

if [ -e "models/$RUN" ]; then
  echo "FATAL: models/$RUN already exists — this script only starts a FRESH fork."
  echo "       A restart / re-fork is a decision, not a re-run (and is DESTRUCTIVE if wrong)."
  exit 3
fi
# PRECISE tenant check: only a PYTHON process running train_rl_agent.py is a trainer. The old
# `pgrep -f "[t]rain_rl_agent.py"` also matched any shell whose argv carried the text (it blocked
# two launches on 2026-09-25).
TRAINER_RE='^[^ ]*python[0-9.]* [^ ]*train_rl_agent\.py'
if pgrep -f "$TRAINER_RE" >/dev/null 2>&1 && [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
  echo "FATAL: a trainer process is already running — the GPU is single-tenant, and the battery's"
  echo "       speed endpoint assumes it (registration §4.1)."
  pgrep -af "$TRAINER_RE" | cut -c1-200
  exit 5
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${ARM}_${RUN}.log"
echo "=== [$ARM] LAUNCH $(date -Is) -> $LOG ==="
# shellcheck disable=SC2086
nohup "$P" -m main.launcher $ARGV > "$LOG" 2>&1 < /dev/null &
echo "launcher pid $!"
echo
echo "STOP-LIST — the first two minutes are the only test of the preload layer (registration §6.1):"
cat <<EOS
  - role FORK of models/ai_v14_01_base, run dir models/$RUN, pin 2cc83080, obs source core, transport rust
  - [ForkLR] "PINNED by --fork-lr to <D_g, or 2 D_g on E5> and FROZEN"; no [ForkLR] FATAL
  - no [ModelVersion] FATAL (N0 is v121, the pin v123, MIGRATION_FLOOR 121); obs dim 2761
  - 🧮 [MATMUL PRECISION] reads 'high' on T32 and 'highest' on the other three
  - the self-play pool is SEEDED from N0's (agents.training.pool_seed); no FATAL_CONFIG empty-pool exit
  - no CoreObsMismatch / "__OBS__ frame nobody asked for" (one = STOP, as for N0)
  - after the FIRST rollout (tb_read $RUN): train/approx_kl_epoch_0..9 exist (0..4 ONLY on E5);
    hparams/gae_lambda 0.95 on L95, 0.80 elsewhere; train/learning_rate constant at the pin
EOS
