"""Download the PUBLIC Pokemon Showdown replay corpus for one format into the project's on-disk layout.

    <out>/<YYYY-MM-DD>/battle-<format>-<id>.log          e.g. replays/showdown/gen3ou/2026-05-18/battle-gen3ou-2612838147.log

PROVENANCE (read `README.md` first): the script that produced the existing 376k-replay corpus under
`<main checkout>/replays/showdown/gen3ou/` was never committed and could not be recovered (2026-10-08 search of the
home directory, the shell history, every session transcript and git history). THIS is a minimal re-write for the
same layout; it is not that script, and nothing here was run against the public server when it was written.

What it does, and nothing more:

* pages `https://replay.pokemonshowdown.com/search.json?format=<fmt>[&before=<uploadtime>]` newest first (a page
  is up to 51 rows; a FULL page means "more" and the next page is `before=<uploadtime of its last row>`; rows are
  de-duplicated by id, so it does not matter whether `before` is inclusive — the API's semantics were not
  verified against the live server);
* skips private replays, and every id already on disk under ANY date folder (so a re-run only fetches what is
  new, and a partial run resumes);
* fetches `https://replay.pokemonshowdown.com/<id>.log` and writes it to the date folder of the battle's START
  (`|t:|` in the log, the America/Los_Angeles calendar date — the convention measured on the existing corpus:
  100 of 100 sampled files; 32 of them differ from the UTC date), falling back to `uploadtime`;
* is POLITE by construction: ONE request at a time, a sleep between requests (default 1.0 s), a bounded
  `--max-replays`, exponential back-off on 429/5xx, and it STOPS on a persistent failure instead of hammering.

    python -m tools.replay_corpus_downloader.sync --out replays/showdown/gen3ou --format gen3ou \\
        --since 2026-09-23 --max-replays 2000 --sleep 1.0

`--list-only` fetches the search pages (still a network call) and prints what it WOULD download.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

BASE = "https://replay.pokemonshowdown.com"
USER_AGENT = "gen3ai-replay-corpus-downloader (research; contact: repo owner; 1 request at a time)"
PAGE_SIZE = 51                      # search.json returns up to 51 rows; the 51st only signals "there is more"
CORPUS_TZ = ZoneInfo("America/Los_Angeles")
_T_LINE = re.compile(r"^\|t:\|(\d+)\s*$", re.M)

Fetch = Callable[[str], str]        # url -> body text; raises FetchError on a persistent failure


class FetchError(RuntimeError):
    pass


def make_http_fetch(sleep_s: float = 1.0, retries: int = 4, timeout: float = 30.0) -> Fetch:
    """The real fetcher: one request at a time, ``sleep_s`` between requests, back-off on 429 / 5xx."""
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    last = [0.0]

    def fetch(url: str) -> str:
        delay = 2.0
        for attempt in range(retries + 1):
            wait = last[0] + sleep_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
            try:
                r = session.get(url, timeout=timeout)
            except requests.RequestException as e:
                err = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    r.encoding = "utf-8"            # name the encoding: requests would guess latin-1 for text/plain
                    return r.text
                if r.status_code == 404:
                    raise FetchError(f"404 {url}")
                err = f"HTTP {r.status_code}"
            if attempt == retries:
                raise FetchError(f"{err} {url} (gave up after {retries + 1} attempts)")
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    return fetch


def folder_date(log_text: str, uploadtime: Optional[int]) -> str:
    """The date folder of a replay: the Los Angeles date of the battle's START (`|t:|`), else of ``uploadtime``."""
    m = _T_LINE.search(log_text)
    ts = int(m.group(1)) if m else uploadtime
    if ts is None:
        raise ValueError("a replay with neither a |t:| line nor an uploadtime has no date folder")
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).astimezone(CORPUS_TZ).strftime("%Y-%m-%d")


def file_name(replay_id: str) -> str:
    """`gen3ou-2612838147` -> `battle-gen3ou-2612838147.log` (the corpus' name for it)."""
    return f"battle-{replay_id}.log"


def existing_ids(out: Path) -> Set[str]:
    """Every replay id already under ``out`` (any date folder)."""
    have: Set[str] = set()
    if out.is_dir():
        for day in out.iterdir():
            if day.is_dir():
                have.update(p.name[len("battle-"):-len(".log")] for p in day.glob("battle-*.log"))
    return have


def search_pages(fmt: str, fetch: Fetch, since_ts: Optional[int] = None) -> Iterator[dict]:
    """Yield search rows newest first, following `before=` until the archive (or ``since_ts``) is exhausted."""
    before: Optional[int] = None
    seen: Set[str] = set()
    while True:
        url = f"{BASE}/search.json?format={fmt}" + (f"&before={before}" if before is not None else "")
        rows = json.loads(fetch(url))
        if not isinstance(rows, list):
            raise FetchError(f"search.json returned {type(rows).__name__}, not a list: {url}")
        fresh = 0
        for row in rows:
            if since_ts is not None and int(row["uploadtime"]) < since_ts:
                return
            if row["id"] in seen:           # `before` may be inclusive: the seam row can come twice
                continue
            seen.add(row["id"])
            fresh += 1
            yield row
        # A short page is the last one. A full page whose rows were ALL seen (a run of identical upload
        # times as long as a page) would loop forever on an inclusive `before`: stop instead, loudly.
        if len(rows) < PAGE_SIZE:
            return
        if fresh == 0:
            raise FetchError(f"pagination made no progress at before={before}: {url}")
        before = int(rows[-1]["uploadtime"])


def sync(out: Path, fmt: str, fetch: Fetch, *, since: Optional[str] = None, max_replays: int = 1000,
         list_only: bool = False, log: Callable[[str], None] = print) -> Tuple[int, int]:
    """Download new replays; returns ``(written, skipped)``. Stops at ``max_replays`` WRITTEN."""
    since_ts = None
    if since:
        since_ts = int(_dt.datetime.fromisoformat(since).replace(tzinfo=CORPUS_TZ).timestamp())
    have = existing_ids(out)
    written = skipped = 0
    for row in search_pages(fmt, fetch, since_ts):
        rid = row["id"]
        if row.get("private") or rid in have:
            skipped += 1
            continue
        if written >= max_replays:
            break
        if list_only:
            log(f"would fetch {rid}")
            written += 1
            continue
        text = fetch(f"{BASE}/{rid}.log")
        day = out / folder_date(text, int(row["uploadtime"]))
        day.mkdir(parents=True, exist_ok=True)
        tmp = day / (file_name(rid) + ".part")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(day / file_name(rid))              # atomic: a killed run never leaves a torn .log
        have.add(rid)
        written += 1
        if written % 100 == 0:
            log(f"{written} written ({skipped} skipped)")
    return written, skipped


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="corpus root, e.g. replays/showdown/gen3ou")
    ap.add_argument("--format", default="gen3ou")
    ap.add_argument("--since", help="only replays uploaded on/after this Los Angeles date (YYYY-MM-DD)")
    ap.add_argument("--max-replays", type=int, default=1000, help="stop after writing this many (default 1000)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests (default 1.0; keep >= 0.5)")
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args(argv)
    if a.sleep < 0.5:
        ap.error("--sleep below 0.5 s is refused: this is a public server")
    try:
        written, skipped = sync(Path(a.out), a.format, make_http_fetch(a.sleep), since=a.since,
                                max_replays=a.max_replays, list_only=a.list_only)
    except FetchError as e:
        print(f"STOPPED on a persistent failure: {e}", file=sys.stderr)
        return 1
    print(f"{'would write' if a.list_only else 'wrote'} {written}, skipped {skipped} (already on disk or private)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
