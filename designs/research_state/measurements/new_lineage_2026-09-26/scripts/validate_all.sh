#!/usr/bin/env bash
# VALIDATE BY EXECUTING — checkargs + launcher --dry-run on the NEW LINEAGE base-run argv, plus the
# launch script's own --dry-run. Creates NOTHING under models/ (asserted before and after).
#
#   bash designs/research_state/measurements/new_lineage_2026-09-26/scripts/validate_all.sh [<outdir>]
#
# Default <outdir> is the kit's validation/ directory. Run from any cwd; it cd's to the MAIN checkout
# so relative models/… paths resolve against the run archive (as the launcher will).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D="$(dirname "$HERE")"
SRC="$(cd "$D/../../../.." && pwd)/src"
OUTDIR="${1:-$D/validation}"
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
RUN=ai_v14_01_base
export PYTHONPATH="${PYTHONPATH:-}:$SRC"
mkdir -p "$OUTDIR"
cd /home/goodlad/dev/gen3ai

before="$(ls models | wc -l)"
[ -e "models/$RUN" ] && { echo "FATAL: models/$RUN already exists — validation must not run against a live dir"; exit 3; }

ARGV="$(cat "$D/argv_base.txt")"
echo "=== base: checkargs ($(date -Is))"
nice -n 15 "$P" -m main.checkargs --argv "$ARGV" > "$OUTDIR/checkargs_base.txt" 2>&1
echo "    rc=$?  -> $OUTDIR/checkargs_base.txt"
echo "=== base: launcher --dry-run ($(date -Is))"
# shellcheck disable=SC2086
nice -n 15 "$P" -m main.launcher --dry-run $ARGV > "$OUTDIR/dryrun_base.txt" 2>&1
echo "    rc=$?  -> $OUTDIR/dryrun_base.txt"
echo "=== launch_base.sh --dry-run ($(date -Is))"
bash "$D/launch_base.sh" --dry-run > "$OUTDIR/launchsh_base.txt" 2>&1
echo "    rc=$?  -> $OUTDIR/launchsh_base.txt"

after="$(ls models | wc -l)"
if [ -e "models/$RUN" ] || [ "$before" != "$after" ]; then
  echo "FATAL: validation CREATED something under models/ ($before -> $after entries)"; exit 9
fi
echo "=== models/ untouched ($after entries; no models/$RUN)"
