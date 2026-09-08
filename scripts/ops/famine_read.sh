#!/usr/bin/env bash
#
# famine_read.sh — THE DECIDING FAMINE READ for one arm, plus the noise-aware kill-bar table.
#
#     scripts/ops/famine_read.sh <run-name|run-dir> --parent NAME --famine-comparator NAME \
#         --control "<run> <run> ..." [--bar 5M|10M] [-- <extra critic_gate flags>]
#     scripts/ops/famine_read.sh --help
#
# `designs/ops/TRAINING_RUN_SOP.md` §5: one command per registered read, and every consumer
# prints which baseline it resolved. The DECIDING instrument is the famine ladder AND-gated with
# `win_rate_vs_bots`; `python -m main.critic_gate` is what computes it.
#
# TWO THINGS THIS SCRIPT DOES NOT DO, on purpose:
#   - it does not BUILD the ladder. `agents.training.snapshot_ladder --backfill` does, and the
#     command is printed rather than run, because a backfill concurrent with the run's own
#     updater reports the previous node's verdict while looking entirely current.
#   - it does not pick the comparators. `--parent` and `--famine-comparator` are BASELINE
#     REGISTRY NAMES and `--control` is an explicit list; the SOP requires the read to state
#     what the arm is measured against by registry name, at launch, in writing.
#
# WHY THE KILL BAR RIDES ALONG. The gates' clause 2 is "ep_len rising", and a raw mean-vs-mean
# test answers YES on noise: at 4.4M it read 30.04 -> 30.35 "+0.31 RISING" on a 25.88..40.94
# series. `main.ops.killbar` asserts RISING only when the difference clears a fixed effect size
# INSIDE ONE OPPONENT REGIME.
#
# Promoted 2026-09-07 from the Training Run session's `read_10M.sh`, where the arm name, the two
# registry names, the three control runs and the interpreter were all literals. The step in the
# old name was the arm's own 10M node; the read is not specific to 10M.
set -u

usage() {
    cat <<'EOF'
famine_read.sh — the deciding famine read for one arm, plus the noise-aware kill-bar table.

USAGE
    scripts/ops/famine_read.sh <run-name|run-dir> --parent NAME --famine-comparator NAME \
        --control "<run> <run> ..." [--bar 5M|10M] [-- <extra critic_gate flags>]
    scripts/ops/famine_read.sh --help

ARGUMENTS
    <run-name|run-dir>      a bare name is looked up in the run archive (models/).

OPTIONS
    --parent NAME           BASELINE REGISTRY name of the parent (see designs/baselines.json)
    --famine-comparator N   BASELINE REGISTRY name of the famine comparator
    --control "A B C"       space-separated control runs, passed through to critic_gate
    --bar 5M|10M            which kill-bar thresholds main.ops.killbar evaluates (default 10M)
    --                      everything after this is forwarded verbatim to critic_gate

WHAT IT PRINTS
    1. the snapshots present, and the ladder-backfill command (NOT run — see the header)
    2. `python -m main.critic_gate <run> --parent … --famine-comparator … --control …`
    3. the kill-bar table from `python -m main.ops.killbar`

ENVIRONMENT
    GEN3AI_MODELS_DIR  the run archive, if it is not the main checkout's models/
    GEN3AI_PYTHON      interpreter

EXIT
    the read is printed for a human; a non-zero critic_gate status is NOT swallowed.
    2 = bad usage.
EOF
}

case "${1:-}" in
    -h|--help|"") usage; [ -z "${1:-}" ] && exit 2 || exit 0 ;;
esac

# shellcheck source=scripts/ops/_common.sh
. "$(dirname "$(readlink -f "$0")")/_common.sh"

RUN_ARG="$1"; shift
PARENT=""; COMPARATOR=""; CONTROL=""; BAR="${KILLBAR:-10M}"; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --parent)              PARENT="$2"; shift 2 ;;
        --famine-comparator)   COMPARATOR="$2"; shift 2 ;;
        --control)             CONTROL="$2"; shift 2 ;;
        --bar)                 BAR="$2"; shift 2 ;;
        --)                    shift; EXTRA=("$@"); break ;;
        *) echo "REFUSING: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$PARENT" ] || [ -z "$COMPARATOR" ]; then
    echo "REFUSING: --parent and --famine-comparator are both required. The read states what" >&2
    echo "  the arm is measured against BY REGISTRY NAME (TRAINING_RUN_SOP.md §1 step 6);" >&2
    echo "  a comparator is READ from designs/baselines.json, never assumed." >&2
    exit 2
fi

RUNDIR="$(ops_resolve_run "$RUN_ARG")" || exit 2
ARM="$(basename "$RUNDIR")"
REPO="$(ops_repo_root)"
PY="$(ops_python)"
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO/src"

echo "### snapshots present (the read wants the registered node COUNT, not a step)"
ls "$RUNDIR"/snapshots/*.zip 2>/dev/null | sed 's|.*/||' || echo "  none"
echo
echo "### ladder must exist first — run this yourself, never concurrently with the run's updater:"
echo "  $PY -m agents.training.snapshot_ladder $RUNDIR --backfill"
echo
echo "### THE READ"
set -x
"$PY" -m main.critic_gate "$ARM" \
    --parent "$PARENT" \
    --famine-comparator "$COMPARATOR" \
    ${CONTROL:+--control $CONTROL} \
    ${EXTRA[@]+"${EXTRA[@]}"}
rc=$?
set +x

# --- kill-bar evaluation, NOISE-AWARE ------------------------------------------------------
echo
"$PY" -m main.ops.killbar "$RUNDIR" --bar "$BAR" || true
exit $rc
