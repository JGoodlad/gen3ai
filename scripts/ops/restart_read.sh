#!/usr/bin/env bash
#
# restart_read.sh — THE FIRST-RESTART READ. One command, run at a restart boundary.
#
#     scripts/ops/restart_read.sh <run-name|run-dir> [--bar 5M|10M] [--baseline PATH]
#     scripts/ops/restart_read.sh --help
#
# `designs/ops/TRAINING_RUN_SOP.md` §3 registers the first-restart checks: the vf_coef decision
# from `grad/value_policy_logratio`, and `python -m main.sidecar_audit` showing the pin is
# UNCHANGED across the restart. Report even when nominal.
#
# 🚨 A SINGLE-SAMPLE READING IS NEVER A VERDICT. `grad/value_policy_logratio` swung a full decade
# between consecutive rollouts while its 20-rollout median sat at +0.15; three single-sample
# readings in one day would each have produced the wrong call. Every section below reports the
# MEDIAN over the registered window and prints the last single reading only labelled as such.
#
# WHAT IT RUNS, in order:
#   1. the vf_coef read + verdict, from the TENSORBOARD EVENTS (the only source; see 🚨 below),
#      followed by a labelled CROSS-CHECK against the child log's rendered table
#   2. `python -m main.sidecar_audit <run>` — the pin across the restart
#   3. the restart evidence: the launcher log's restart lines and the newest checkpoints
#   4. `main.ops.killbar`         — the noise-aware kill-bar table
#   5. `main.ops.vf_framings`     — all three framings of the vf_coef statistic, registered one named
#   6. `main.ops.restart_startup` — the restart's startup cost against its pre-registered reading
#
# Promoted 2026-09-07 from the Training Run session's `restart_read.sh` (run directory,
# interpreter and the three helper script paths were literals).
#
# 🚨 ONE STATISTIC, ONE SOURCE — THE EVENTS. The registered vf_coef statistic
# (`grad/value_policy_logratio`, median of the last 20 rollouts, ledger cfc72ad0) is computed
# ONCE, in step 1, from the TB events via `main.ops.tb_read`, with the verdict bar shared with
# `main.ops.vf_framings` (one definition, imported, never re-typed). Until 2026-09-07 step 1 read
# the CHILD LOG instead and step 5 read the events — one statistic, two sources, no statement of
# which to believe. The child log's `| value_policy_logratio |` table is a RENDERING: it drops the
# scalar's `grad/` group prefix, `--log-level periodic` UNDERSAMPLES the rollouts, and
# `launcher_child.log` is a ~1 MiB RING BUFFER that trims silently — so "the median of the last
# 20" taken from it is the median of the last 20 rows THAT STILL FIT, which need not be the last
# 20 rollouts. The SOP's own rule is *validate at the source*, and the events are the source.
#
# The child-log read SURVIVES, demoted to a CROSS-CHECK: it prints both medians and their
# difference, and WARNS above a tolerance. It never produces a verdict and never overrides one —
# when the two disagree, the events figure is the reading and the disagreement is the finding.
set -u

usage() {
    cat <<'EOF'
restart_read.sh — the registered first-restart read for one run.

USAGE
    scripts/ops/restart_read.sh <run-name|run-dir> [--bar 5M|10M] [--baseline PATH]
    scripts/ops/restart_read.sh --help

ARGUMENTS
    <run-name|run-dir>   a bare name is looked up in the run archive (models/).

OPTIONS
    --bar 5M|10M      which kill-bar thresholds main.ops.killbar evaluates (default 10M)
    --baseline PATH   the pre-restart baseline file main.ops.restart_startup measures the
                      startup cost against. Without it that section is SKIPPED — the
                      startup cost is not computable from a run directory alone, and a
                      skipped section is not a reading of zero.

ENVIRONMENT
    KILLBAR            default for --bar
    GEN3AI_MODELS_DIR  the run archive, if it is not the main checkout's models/
    GEN3AI_PYTHON      interpreter

EXIT
    0  printed        2  bad usage, or the run could not be resolved
EOF
}

case "${1:-}" in
    -h|--help|"") usage; [ -z "${1:-}" ] && exit 2 || exit 0 ;;
esac

# shellcheck source=scripts/ops/_common.sh
. "$(dirname "$(readlink -f "$0")")/_common.sh"

RUN_ARG="$1"; shift
BAR="${KILLBAR:-10M}"; BASELINE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --bar)       BAR="$2"; shift 2 ;;
        --baseline)  BASELINE="$2"; shift 2 ;;
        *) echo "REFUSING: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

R="$(ops_resolve_run "$RUN_ARG")" || exit 2
REPO="$(ops_repo_root)"
PY="$(ops_python)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO/src"

echo "=== vf_coef read (grad/value_policy_logratio, log10; median of the last 20 rollouts) ==="
echo "    source: the TENSORBOARD EVENTS (main.ops.tb_read). The child log is a CROSS-CHECK only."
VPL="$(mktemp)"
trap 'rm -f "$VPL"' EXIT
# The cross-check's rows: the child log's rendered table, which drops the `grad/` prefix.
grep -aE '^\| *value_policy_logratio' "$R/launcher_child.log" 2>/dev/null \
    | awk -F'|' '{gsub(/ /,"",$3); print $3}' > "$VPL"
"$PY" - "$R" "$VPL" <<'EOF'
import statistics as st
import sys
from pathlib import Path

from main.ops.tb_read import load
from main.ops.vf_framings import verdict            # ONE definition of the bar, never re-typed

TAG = "grad/value_policy_logratio"
# The cross-check tolerance, in log10 units of the same ratio. 0.10 log10 (= 1.26x) is ONE FIFTH
# of the narrowest verdict band (|med| <= 0.5 KEEP), so a difference at or below it cannot move
# the call unless the events median already sits within 0.10 of a band edge — which is reported
# separately below, because an agreeing cross-check cannot reassure about a near-boundary read.
# Anything larger means the two windows are not describing the same rollouts (the ring buffer
# trimmed, or `--log-level periodic` undersampled), and only the events figure is a measurement.
TOL = 0.10
BANDS = (0.5, 1.0)

run, rows_path = sys.argv[1], sys.argv[2]
series = load(Path(run), [TAG]).get(TAG, [])

if not series:
    print(f"  NO SCALAR {TAG} in the events  <- absence is not a zero")
    print("  NO VERDICT. The cross-check below cannot supply one.")
    med = None
else:
    last = series[-20:]
    med = st.median([v for _, v in last])
    print(f"  rollouts in events  {len(series)}   (using the last {len(last)}, "
          f"steps {last[0][0]:,}..{last[-1][0]:,})")
    print(f"  median log10        {med:+.3f}")
    print(f"  median ratio        {10**med:.2f}x  (value/policy gradient norm)")
    print(f"  last single reading {series[-1][1]:+.3f} log10 = {10**series[-1][1]:.2f}x   "
          f"<- NEVER the verdict")
    vals = [v for _, v in last]
    print(f"  min/max of last 20  {min(vals):+.3f} / {max(vals):+.3f} log10")
    first, second = vals[:len(vals)//2], vals[len(vals)//2:]
    print(f"  trend (1st half -> 2nd half of last 20): "
          f"{st.median(first):+.3f} -> {st.median(second):+.3f} log10")
    print(f"  VERDICT  {verdict(med)}   (|{med:+.3f}|)")
    near = [b for b in BANDS if abs(abs(med) - b) <= TOL]
    if near:
        print(f"  ⚠️ NEAR A BAND EDGE ({', '.join(str(b) for b in near)}): the verdict is within "
              f"{TOL} log10 of flipping — re-read at the next restart before acting.")

log = [float(x) for x in open(rows_path) if x.strip()]
print("\n  CROSS-CHECK — the child log's rendered table (NOT a source, NOT a verdict)")
if not log:
    print("    no value_policy_logratio rows in launcher_child.log  <- absence is not a zero;")
    print("    the ~1 MiB ring buffer trims silently, so an empty cross-check is expected on an")
    print("    old run and says nothing about the events read above.")
elif med is None:
    print(f"    child-log median of the last 20 rows  {st.median(log[-20:]):+.3f} log10  "
          f"(n={len(log)})")
    print("    Reported only. With no events series there is no reading to check it against,")
    print("    and this number is NOT promoted to one.")
else:
    mlog = st.median(log[-20:])
    d = mlog - med
    print(f"    child-log median {mlog:+.3f} log10 (n={len(log)} rows) vs events {med:+.3f}"
          f"  ->  difference {d:+.3f} log10 ({10**abs(d):.2f}x)")
    if abs(d) > TOL:
        print(f"    ⚠️ DISAGREEMENT above the {TOL} log10 tolerance. The events figure stands;")
        print("       the child log's window is the last 20 rows THAT STILL FIT the ring buffer,")
        print("       undersampled by --log-level periodic, so it is describing other rollouts.")
        if verdict(mlog) != verdict(med):
            print(f"       It would even have said: {verdict(mlog)}  <- and would have been WRONG.")
    else:
        print(f"    within the {TOL} log10 tolerance — the two windows agree. The verdict above")
        print("    still rests on the events alone.")
EOF

echo
echo "=== sidecar audit — the pin must be UNCHANGED across the restart ==="
"$PY" -m main.sidecar_audit "$R" 2>&1 | tail -20

echo
echo "=== restart evidence ==="
grep -aE 'restart|RESTART' "$R/launcher.log" 2>/dev/null | tail -5 \
    || echo "  no launcher.log restart lines  <- absence is not a zero"
ls -l --time-style=+%H:%M:%S "$R"/checkpoints/*.zip 2>/dev/null | tail -5 \
    || echo "  no checkpoints"

# --- kill-bar evaluation, NOISE-AWARE ------------------------------------------------------
echo
"$PY" -m main.ops.killbar "$R" --bar "$BAR" || true

# All three framings of the vf_coef statistic, with the REGISTERED one named. Reporting only.
echo
"$PY" -m main.ops.vf_framings "$R" || true

# Restart startup cost, against the pre-registered reading. Needs its baseline file.
echo
if [ -n "$BASELINE" ]; then
    "$PY" -m main.ops.restart_startup "$R" --baseline "$BASELINE" || true
else
    echo "=== RESTART STARTUP COST — SKIPPED (no --baseline given) ==="
    echo "  The cost is the TB wall-time gap across the boundary measured against a baseline"
    echo "  captured BEFORE the restart. Without that file there is nothing to difference."
    echo "  This is NOT a reading of 'startup was free'."
fi
