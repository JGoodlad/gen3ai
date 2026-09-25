#!/usr/bin/env bash
# VALIDATE BY EXECUTING — checkargs + launcher --dry-run on every round-2 reader / extension argv.
# Creates NOTHING under models/.
#
#   bash designs/research_state/measurements/population_loop_r2_2026-09-24/scripts/validate_all.sh <outdir>
#
# RB2 / RC2 are validated on their STAND-IN argvs (target = ai_v13_24_popr1_read_loop @111,280,128,
# the step B2 / C2 will land on), because B2's and C2's finals do not exist yet. RB+ / RC+ are
# validated on their REAL argvs: their parents (RB, RC) and targets (B, C) all exist.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D="$(dirname "$HERE")"
SRC="$(cd "$D/../../../.." && pwd)/src"
OUTDIR="${1:-/tmp/pl_r2/validate}"
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
export PYTHONPATH="${PYTHONPATH:-}:$SRC"
mkdir -p "$OUTDIR"
cd /home/goodlad/dev/gen3ai   # relative models/… paths resolve against the MAIN checkout's archive

for arm in RB2_STANDIN RC2_STANDIN RBX RCX; do
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
for s in RB2 RC2 RBX RCX; do
  echo "=== launch_$s.sh --dry-run ($(date -Is))"
  bash "$D/launch_$s.sh" --dry-run > "$OUTDIR/launchsh_${s}.txt" 2>&1
  echo "    rc=$?  -> $OUTDIR/launchsh_${s}.txt"
done
