"""The supervised replay-corpus pull: the one-off BACKFILL and the DAILY top-up, as systemd user units run them.

    python -m tools.replay_corpus_downloader.service backfill --out <main>/replays/showdown/gen3ou --proxy socks5h://127.0.0.1:1080
    python -m tools.replay_corpus_downloader.service daily    --out <main>/replays/showdown/gen3ou --proxy socks5h://127.0.0.1:1080

Both are thin drivers over ``sync.sync`` with its proxy, lock and politeness (``sync.py``); the units, how to stop
them and where they log: ``designs/tools/replay_corpus.md`` § Supervision.

* ``backfill`` runs the PHASES in order, each as resumable chunks of ``--chunk`` replays (a chunk that writes fewer
  than ``--chunk`` exhausted its window, so the phase is done), then writes ``BACKFILL_DONE.json`` in the corpus root
  (the unit's ``ConditionPathExists=!`` keeps it from re-walking the archive on every boot). A failure exits non-zero
  and systemd restarts it; every replay already on disk is skipped, so a restart resumes.
* ``daily`` pulls ``--since`` the newest date folder minus one day (``--max-replays`` 6000), then runs the freshness
  alarm. It exits 1 when the corpus is STALE (the unit shows as failed) and writes the alarm at journal priority
  ``err``. When the backfill holds the downloader lock it skips the pull and still checks freshness.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional

from tools.replay_corpus_downloader import freshness as F
from tools.replay_corpus_downloader import sync as S


class Phase(NamedTuple):
    name: str
    since: str                  # Los Angeles upload dates, [since, until)
    until: Optional[str]


# The restart plan (designs/tools/replay_corpus.md): the weeks since the spectator stopped first, then the two
# older holes it left, each its own phase. The spectator's last file is 2026-09-23 12:07 PDT; its 09-23 folder is partial.
PHASES = (
    Phase("since_spectator_stop", "2026-09-23", None),
    Phase("hole_2026-08-03..08-17", "2026-08-03", "2026-08-18"),
    Phase("hole_2026-06-16..06-20", "2026-06-16", "2026-06-21"),
)
DONE_MARKER = "BACKFILL_DONE.json"
DAILY_MAX_REPLAYS = 6000


def journal_err(msg: str) -> str:
    """Prefix ``<3>`` (journal priority err) when stdout/stderr is the journal, so the alarm shows in ``-p err``."""
    return f"<3>{msg}" if os.environ.get("JOURNAL_STREAM") else msg


def backfill(out: Path, fmt: str, fetch: S.Fetch, *, chunk: int, phases=PHASES,
             log: Callable[[str], None] = print) -> int:
    """Run every phase to exhaustion in chunks; returns the replays written. Raises ``FetchError`` on a failure."""
    total = 0
    for ph in phases:
        k = 0
        while True:
            k += 1
            res = S.sync(out, fmt, fetch, since=ph.since, until=ph.until, max_replays=chunk, log=lambda m: None)
            total += res.written
            log(f"[backfill] {ph.name} chunk {k}: {res.describe()}")
            if res.written < chunk:
                break
    (out / DONE_MARKER).write_text(json.dumps({
        "finished": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "phases": [ph._asdict() for ph in phases], "written_this_run": total}, indent=1) + "\n")
    log(f"[backfill] DONE: {total} written this run; {out / DONE_MARKER} written")
    return total


def daily_since(out: Path) -> Optional[str]:
    """The newest date folder minus one day (a battle that started before midnight lands in the older folder)."""
    days = F.date_folders(out)
    return (days[-1] - _dt.timedelta(days=1)).isoformat() if days else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("backfill", "daily"))
    S.add_fetch_args(ap)
    ap.add_argument("--chunk", type=int, default=4000, help="backfill: replays per chunk (default 4000)")
    ap.add_argument("--max-replays", type=int, default=DAILY_MAX_REPLAYS, help="daily: the pull's bound")
    ap.add_argument("--max-age-days", type=int, default=F.DEFAULT_MAX_AGE_DAYS, help="daily: the freshness limit")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not out.is_absolute():
        ap.error("--out must be ABSOLUTE: a relative path resolves against the cwd")
    rc = 0
    try:
        fetch = S.checked_fetch(ap, a)
        with S.single_instance():
            if a.mode == "backfill":
                backfill(out, a.format, fetch, chunk=a.chunk)
            else:
                res = S.sync(out, a.format, fetch, since=daily_since(out), max_replays=a.max_replays,
                             log=lambda m: None)
                print(f"[daily] {res.describe()}")
    except S.ProxyError as e:
        print(journal_err(f"[{a.mode}] STOPPED, no request made: {e}"), file=sys.stderr)
        rc = S.EXIT_PROXY_DOWN
    except S.Locked as e:
        print(f"[{a.mode}] no pull: {e}", file=sys.stderr)
        rc = S.EXIT_LOCKED if a.mode == "backfill" else 0      # the daily defers to a running backfill
    except S.FetchError as e:
        print(journal_err(f"[{a.mode}] STOPPED on a persistent failure: {e}"), file=sys.stderr)
        rc = S.EXIT_FAILURE
    if a.mode == "daily":
        code, msg = F.check(out, max_age_days=a.max_age_days)
        if code == F.EXIT_FRESH:
            print(f"[freshness] {msg}")
        else:
            print(journal_err(f"[freshness] REPLAY CORPUS {msg}"), file=sys.stderr)
            rc = rc or code
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
