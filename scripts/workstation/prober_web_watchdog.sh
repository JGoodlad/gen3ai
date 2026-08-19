#!/usr/bin/env bash
#
# Restart the prober web service when the code it is RUNNING no longer matches the repo.
#
# WHY THIS EXISTS, precisely. `Restart=always` in gen3ai-prober-web.service covers a process that
# DIES. It does not cover the failure this watchdog is for, because in that failure nothing dies:
# Jinja reloads a changed template from disk while Python cannot reload a changed module, so a
# long-lived server drifts into serving NEW templates against OLD code. Measured 2026-08-18: a
# service up 5 days returned HTTP 500 on every `/battle` for a fresh run — the template asked for
# a key the running `session.py` predated — while systemd reported it perfectly healthy and the
# tunnel in front of it reported it up. `auto_reload` is now off (so the hybrid cannot form), and
# this closes the other half: a pinned process must not stay pinned to a revision nobody is
# running any more.
#
# THE COMPARISON is the process's own revision (`/api/health` -> `revision`, captured at import)
# against the repo's HEAD. Not a file mtime: a `git status`, a rebuild, or a touched file changes
# mtimes without changing the code, and this must not restart on noise.
#
#   KNOWN LIMIT, stated rather than papered over: an UNCOMMITTED edit does not move HEAD, so it
#   does not trigger a restart. That is correct for this box — main is only ever advanced by a
#   commit (the worktree workflow forbids editing the main checkout), so HEAD is the real signal —
#   but anyone hand-editing files under the service must still restart it themselves.
#
# It DEFERS while a job is running: `falsify_scan` and `calibration` are minutes of Node re-rolls,
# and a restart kills them. A deferred restart is retried on the next tick; the code is stale
# either way, and silently discarding a probe someone is waiting on is the worse trade.
#
# Usage:  prober_web_watchdog.sh [--dry-run]
# Exit:   0 = nothing to do, or restarted successfully; 1 = a check failed (see stderr).
set -uo pipefail

REPO="${GEN3AI_REPO:-/home/goodlad/dev/gen3ai}"
HEALTH_URL="${GEN3AI_PROBER_HEALTH:-http://127.0.0.1:6008/api/health}"
UNIT="${GEN3AI_PROBER_UNIT:-gen3ai-prober-web.service}"
# A seam so the decision logic can be tested without a real unit — the tests stub this with a
# script that records its arguments. Untested, this file is a shell script nobody runs by hand
# guarding against a failure that already went unnoticed for five days.
SYSTEMCTL="${GEN3AI_SYSTEMCTL:-systemctl}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

log() { echo "[prober-watchdog] $*"; }
die() { echo "[prober-watchdog] $*" >&2; exit 1; }

# The unit being inactive is NOT this script's problem: systemd's own Restart=always owns that,
# and restarting a unit that is deliberately stopped would fight the operator.
state=$("$SYSTEMCTL" --user is-active "$UNIT" 2>/dev/null || true)
if [ "$state" != "active" ]; then
  log "unit is '$state' — leaving it to systemd"
  exit 0
fi

health=$(curl -fsS --max-time 15 "$HEALTH_URL" 2>/dev/null) || die "health check unreachable at $HEALTH_URL"

read -r running jobs <<<"$(printf '%s' "$health" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("unparseable 0"); raise SystemExit
print(d.get("revision", "unknown"), int(d.get("jobs_running", 0) or 0))
')"

# An older build predates the `revision` field entirely, and reports nothing. That IS staleness —
# by definition the running code is older than the code that added the field — so restart on it
# rather than treating a missing field as "fine".
if [ "$running" = "unparseable" ]; then
  die "health response could not be parsed"
fi

head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) || die "cannot read HEAD in $REPO"

if [ "$running" = "$head" ]; then
  exit 0
fi

if [ "$running" = "unknown" ]; then
  log "running revision is 'unknown' (pre-watchdog build or not a checkout) — treating as stale"
fi

log "STALE: running ${running:0:12}, repo HEAD ${head:0:12}"

if [ "$jobs" -gt 0 ]; then
  log "deferring — $jobs job(s) running; a restart would kill them. Retrying next tick."
  exit 0
fi

if [ "$DRY_RUN" = "1" ]; then
  log "--dry-run: would restart $UNIT"
  exit 0
fi

log "restarting $UNIT"
"$SYSTEMCTL" --user restart "$UNIT" || die "restart failed"

# Confirm the replacement actually came up ON THE NEW REVISION. Without this the watchdog would
# report success for a process that crash-looped straight back to the old behaviour, which is the
# same class of "healthy from outside" failure it exists to catch.
for _ in $(seq 1 20); do
  sleep 1
  now=$(curl -fsS --max-time 10 "$HEALTH_URL" 2>/dev/null | python3 -c '
import json, sys
try: print(json.load(sys.stdin).get("revision", ""))
except Exception: print("")
' 2>/dev/null)
  if [ "$now" = "$head" ]; then
    log "OK: now serving ${head:0:12}"
    exit 0
  fi
done

die "restarted, but the service is not serving HEAD (${head:0:12}) — check: journalctl --user -u $UNIT -n 50"
