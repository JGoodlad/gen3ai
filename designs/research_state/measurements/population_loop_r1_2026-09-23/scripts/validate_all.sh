#!/usr/bin/env bash
# VALIDATE BY EXECUTING — checkargs + launcher --dry-run on every round-1 argv. Creates NOTHING.
#
#   bash designs/research_state/measurements/population_loop_r1_2026-09-23/scripts/validate_all.sh <outdir>
#
# Run from the checkout whose src/ should do the judging (a worktree: this one). The readers are
# validated on their STAND-IN argvs (target = ai_v13_16_teach5_offense_dist @103,219,200), because
# their real targets (B's and C's finals) do not exist until B and C have trained.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D="$(dirname "$HERE")"
SRC="$(cd "$D/../../../.." && pwd)/src"
OUTDIR="${1:-/tmp/pl_r1/validate}"
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH="${PYTHONPATH:-}:$SRC"
mkdir -p "$OUTDIR"
# relative models/… paths in the argvs resolve against the MAIN checkout's archive
cd /home/goodlad/dev/gen3ai

for arm in B C A2 RB_STANDIN RC_STANDIN; do
  case "$arm" in
    *_STANDIN) f="$HERE/argv_${arm}.txt" ;;
    *)         f="$D/argv_${arm}.txt" ;;
  esac
  ARGV="$(cat "$f")"
  echo "=== $arm: checkargs ($(date -Is))"
  nice -n 15 "$P" -m main.checkargs --argv "$ARGV" > "$OUTDIR/checkargs_${arm}.txt" 2>&1
  echo "    rc=$?  -> $OUTDIR/checkargs_${arm}.txt"
  echo "=== $arm: launcher --dry-run ($(date -Is))"
  # shellcheck disable=SC2086
  nice -n 15 "$P" -m main.launcher --dry-run $ARGV > "$OUTDIR/dryrun_${arm}.txt" 2>&1
  echo "    rc=$?  -> $OUTDIR/dryrun_${arm}.txt"
done

echo "=== resolved --stable-opponents / --exploiter entries ($(date -Is))"
nice -n 15 "$P" "$HERE/resolve_opponents.py" > "$OUTDIR/resolve_opponents.txt" 2>&1
echo "    rc=$?  -> $OUTDIR/resolve_opponents.txt"
