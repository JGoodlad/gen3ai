"""READ-ONLY freshness check of the on-disk replay corpus: has the collector silently stopped again?

    python -m tools.replay_corpus_downloader.freshness [--out replays/showdown/gen3ou] [--max-age-days 3]

The corpus (``<main checkout>/replays/showdown/gen3ou/<YYYY-MM-DD>/battle-*.log``) has stopped growing
three times without anyone noticing for 16-27 days: its collector was an unsupervised tmux window and died with
every reboot (``designs/tools/replay_corpus.md``). This module is the alarm half of the fix. It lists a
directory, writes nothing and makes no request.

Exit codes: 0 fresh, 1 STALE (the newest date folder is older than ``--max-age-days``), 2 no corpus at all.
``--out`` defaults to the MAIN checkout's corpus (``PYTHONPATH=src`` for ``utils.git``). The date folders are
America/Los_Angeles dates (the convention of ``sync.py``), so "today" is the Los Angeles date too.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from tools.replay_corpus_downloader.sync import CORPUS_TZ

DEFAULT_MAX_AGE_DAYS = 3        # a daily pull leaves the newest folder at today or yesterday; 3 tolerates one missed day

EXIT_FRESH, EXIT_STALE, EXIT_MISSING = 0, 1, 2


def date_folders(out: Path) -> List[_dt.date]:
    """The corpus' date folders (names that parse as ``YYYY-MM-DD``), oldest first; anything else is ignored."""
    days: List[_dt.date] = []
    if out.is_dir():
        for p in out.iterdir():
            if p.is_dir():
                try:
                    days.append(_dt.date.fromisoformat(p.name))
                except ValueError:
                    continue
    return sorted(days)


def gap_days(days: List[_dt.date]) -> List[Tuple[_dt.date, _dt.date]]:
    """Runs of missing calendar days inside ``days`` as ``(first_missing, last_missing)`` pairs."""
    gaps = []
    for a, b in zip(days, days[1:]):
        if (b - a).days > 1:
            gaps.append((a + _dt.timedelta(days=1), b - _dt.timedelta(days=1)))
    return gaps


def check(out: Path, *, max_age_days: int = DEFAULT_MAX_AGE_DAYS, today: Optional[_dt.date] = None) -> Tuple[int, str]:
    """``(exit_code, one-line message)``. ``today`` defaults to the Los Angeles date now."""
    today = today or _dt.datetime.now(CORPUS_TZ).date()
    days = date_folders(out)
    if not days:
        return EXIT_MISSING, f"NO CORPUS: no date folders under {out}"
    newest = days[-1]
    age = (today - newest).days
    gaps = gap_days(days)
    tail = f"{len(days)} date folders, {len(gaps)} gap run(s) inside"
    if age > max_age_days:
        return EXIT_STALE, f"STALE: newest date folder {newest} is {age} days old (limit {max_age_days}); {tail}"
    return EXIT_FRESH, f"fresh: newest date folder {newest} ({age} days old, limit {max_age_days}); {tail}"


def default_out() -> Path:
    from utils.git import get_main_repo_root
    return Path(get_main_repo_root()) / "replays" / "showdown" / "gen3ou"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="corpus root (default: the main checkout's replays/showdown/gen3ou)")
    ap.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    a = ap.parse_args(argv)
    code, msg = check(Path(a.out) if a.out else default_out(), max_age_days=a.max_age_days)
    print(msg, file=sys.stdout if code == EXIT_FRESH else sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
