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
#   1. the vf_coef read + verdict, from the CHILD LOG's table (see the ⚠️ below)
#   2. `python -m main.sidecar_audit <run>` — the pin across the restart
#   3. the restart evidence: the launcher log's restart lines and the newest checkpoints
#   4. `main.ops.killbar`         — the noise-aware kill-bar table
#   5. `main.ops.vf_framings`     — all three framings of the vf_coef statistic, registered one named
#   6. `main.ops.restart_startup` — the restart's startup cost against its pre-registered reading
#
# Promoted 2026-09-07 from the Training Run session's `restart_read.sh` (run directory,
# interpreter and the three helper script paths were literals). BEHAVIOUR IS UNCHANGED.
#
# ⚠️ RECORDED, NOT FIXED — step 1 reads the CHILD LOG, and `main.ops.tb_read` exists because
# that source is wrong. The log's `| value_policy_logratio |` table is a RENDERING: it drops the
# scalar's `grad/` group prefix, `--log-level periodic` UNDERSAMPLES the rollouts, and
# `launcher_child.log` is a ~1 MiB RING BUFFER that trims silently — so "the median of the last
# 20" taken from it is the median of the last 20 rows THAT STILL FIT, which need not be the last
# 20 rollouts. The TB events carry every rollout with its step. Step 5 below (`main.ops.vf_framings`)
# already computes the SAME registered statistic from the events, so this script prints one
# statistic from two sources in one run and does not say which to believe. Changing step 1 would
# change a REGISTERED reading, which this promotion has no licence to do; it is a finding for the
# owner, not an edit. Until it is ruled on, prefer the events figure and say which source you quoted.
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

echo "=== vf_coef read (grad/value_policy_logratio, log10; median of last 20) ==="
echo "    source: the CHILD LOG's rendered table — see the ⚠️ in this script's header"
VPL="$(mktemp)"
trap 'rm -f "$VPL"' EXIT
grep -aE '^\| *value_policy_logratio' "$R/launcher_child.log" \
    | awk -F'|' '{gsub(/ /,"",$3); print $3}' > "$VPL"
"$PY" - "$VPL" <<'EOF'
import sys, statistics as st
v = [float(x) for x in open(sys.argv[1]) if x.strip()]
if not v:
    print("  NO value_policy_logratio rows in the child log  <- absence is not a zero")
    raise SystemExit(0)
last = v[-20:]
med = st.median(last)
print(f"  readings total      {len(v)}   (using last {len(last)})")
print(f"  median log10        {med:+.3f}")
print(f"  median ratio        {10**med:.2f}x  (value/policy gradient norm)")
print(f"  last single reading {v[-1]:+.3f} log10 = {10**v[-1]:.2f}x   <- NEVER the verdict")
print(f"  min/max of last 20  {min(last):+.3f} / {max(last):+.3f} log10")
first, second = last[:len(last)//2], last[len(last)//2:]
print(f"  trend (1st half -> 2nd half of last 20): {st.median(first):+.3f} -> {st.median(second):+.3f} log10")
a = abs(med)
if a <= 0.5:
    print(f"  VERDICT  KEEP vf_coef 0.5   (|{med:+.3f}| <= 0.5)")
elif a < 1.0:
    print(f"  VERDICT  KEEP + FLAG, re-read at 2nd restart   (0.5 < |{med:+.3f}| < 1.0)")
else:
    print(f"  VERDICT  NEW ARM required: --vf-coef {0.5*(10**-med):.4g}   (|{med:+.3f}| >= 1.0)")
    print("           (vf_coef is resume-immutable - cannot be changed on this run)")
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
