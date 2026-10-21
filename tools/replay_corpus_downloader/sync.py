"""Download the PUBLIC Pokemon Showdown replay corpus for one format into the project's on-disk layout.

    <out>/<YYYY-MM-DD>/battle-<format>-<id>.log          e.g. replays/showdown/gen3ou/2026-05-18/battle-gen3ou-2612838147.log

PROVENANCE (read `README.md` first): the existing 376,410-replay corpus under `<main checkout>/replays/showdown/gen3ou/`
was written by a deleted LIVE SPECTATOR (`git show 5d85d8a20^:src/main/collect_replays.py`). THIS pulls the public
replay ARCHIVE into the same layout: a different mechanism that holds only UPLOADED replays (status and sizes:
`designs/tools/replay_corpus.md`).

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
* is POLITE by construction: ONE request at a time, a sleep between requests (default and floor 1.0 s), a bounded
  `--max-replays`, exponential back-off on 429/5xx, and it STOPS on a persistent failure instead of hammering;
* sends EVERY request through the owner's SOCKS5 proxy (`--proxy socks5h://127.0.0.1:1080`, the address
  `main.play --proxy` takes; required). It fails CLOSED: an unreachable proxy exits 3 before any request, a proxy
  that dies mid-run is a persistent failure (exit 1), and there is no direct-connection fallback (the session
  ignores the `*_PROXY` / `NO_PROXY` environment);
* holds ONE machine-wide lock: a second downloader exits 75 (EX_TEMPFAIL) without a request.

    python -m tools.replay_corpus_downloader.sync --out <main>/replays/showdown/gen3ou --format gen3ou \\
        --proxy socks5h://127.0.0.1:1080 --since 2026-09-23 --max-replays 2000

`--list-only` fetches the search pages (still a network call) and prints what it WOULD download.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fcntl
import json
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, List, NamedTuple, Optional, Set, Tuple
from zoneinfo import ZoneInfo

BASE = "https://replay.pokemonshowdown.com"
USER_AGENT = "gen3ai-replay-corpus-downloader (research; contact: repo owner; 1 request at a time)"
PAGE_SIZE = 51                      # search.json returns up to 51 rows; the 51st only signals "there is more"
CORPUS_TZ = ZoneInfo("America/Los_Angeles")
_T_LINE = re.compile(r"^\|t:\|(\d+)\s*$", re.M)
MIN_SLEEP_S = 1.0                   # the per-request floor: this is a public server we do not own
PROXY_SCHEMES = ("socks5h", "socks5")
EXIT_FAILURE, EXIT_PROXY_DOWN, EXIT_LOCKED = 1, 3, 75

Fetch = Callable[[str], str]        # url -> body text; raises FetchError on a persistent failure


class FetchError(RuntimeError):
    pass


class ProxyError(RuntimeError):
    """The proxy is malformed or unreachable: the run stops before (or instead of) any direct request."""


def redact_proxy(proxy: str) -> str:
    """``socks5h://user:pw@host:port`` -> ``socks5h://***@host:port``: never print a credential."""
    scheme, sep, rest = proxy.partition("://")
    if "@" in rest:
        rest = "***@" + rest.rsplit("@", 1)[1]
    return scheme + sep + rest


def proxy_endpoint(proxy: str) -> Tuple[str, int]:
    """Validate a SOCKS5 proxy URL and return its ``(host, port)``."""
    from urllib.parse import urlsplit
    u = urlsplit(proxy)
    if u.scheme not in PROXY_SCHEMES or not u.hostname or not u.port:
        raise ProxyError(f"--proxy must be socks5h://host:port or socks5://host:port, got {redact_proxy(proxy)!r}")
    return u.hostname, u.port


def check_proxy(proxy: str, timeout: float = 5.0) -> None:
    """Fail CLOSED before the first request: the proxy's port must accept a TCP connection."""
    host, port = proxy_endpoint(proxy)
    try:
        import socks  # noqa: F401  (PySocks: requests' SOCKS support; without it requests cannot proxy at all)
    except ImportError as e:
        raise ProxyError("PySocks is not installed: requests cannot use a SOCKS proxy") from e
    try:
        socket.create_connection((host, port), timeout=timeout).close()
    except OSError as e:
        raise ProxyError(f"proxy {redact_proxy(proxy)} is unreachable ({type(e).__name__})") from e


def make_http_fetch(sleep_s: float = 1.0, retries: int = 4, timeout: float = 30.0, *, proxy: str,
                    backoff_s: float = 2.0) -> Fetch:
    """The real fetcher: every request through ``proxy`` (SOCKS5), one at a time, ``sleep_s`` apart, back-off on
    429 / 5xx. There is no direct path: ``trust_env`` is off, so no ``NO_PROXY`` / ``*_PROXY`` variable can reroute it."""
    import requests

    proxy_endpoint(proxy)
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": proxy, "https": proxy}
    session.headers["User-Agent"] = USER_AGENT
    last = [0.0]

    def fetch(url: str) -> str:
        delay = backoff_s
        for attempt in range(retries + 1):
            wait = last[0] + sleep_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
            try:
                r = session.get(url, timeout=timeout, proxies=session.proxies)
            except requests.RequestException as e:
                err = f"{type(e).__name__}: {str(e).replace(proxy, redact_proxy(proxy))}"
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


def search_pages(fmt: str, fetch: Fetch, since_ts: Optional[int] = None,
                 until_ts: Optional[int] = None) -> Iterator[dict]:
    """Yield search rows newest first, following `before=` until the archive (or ``since_ts``) is exhausted.

    ``until_ts`` starts the walk at ``before=until_ts`` and drops any row at or after it (whether the API's
    `before` is inclusive is not relied on), so a hole far in the past costs no walk through the recent pages."""
    before: Optional[int] = until_ts
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
            if until_ts is not None and int(row["uploadtime"]) >= until_ts:
                continue
            yield row
        # A short page is the last one. A full page whose rows were ALL seen (a run of identical upload
        # times as long as a page) would loop forever on an inclusive `before`: stop instead, loudly.
        if len(rows) < PAGE_SIZE:
            return
        if fresh == 0:
            raise FetchError(f"pagination made no progress at before={before}: {url}")
        before = int(rows[-1]["uploadtime"])


def la_midnight_ts(day: str) -> int:
    """The Unix time of 00:00 America/Los_Angeles on ``day`` (YYYY-MM-DD)."""
    return int(_dt.datetime.fromisoformat(day).replace(tzinfo=CORPUS_TZ).timestamp())


class SyncResult(NamedTuple):
    written: int
    skipped: int
    folders: Tuple[str, ...]        # the date folders written into, sorted

    def describe(self, list_only: bool = False) -> str:
        span = f"{self.folders[0]}..{self.folders[-1]}" if self.folders else "none"
        verb = "would write" if list_only else "wrote"
        return f"{verb} {self.written}, skipped {self.skipped} (already on disk or private); date folders {span}"


def sync(out: Path, fmt: str, fetch: Fetch, *, since: Optional[str] = None, until: Optional[str] = None,
         max_replays: int = 1000, list_only: bool = False, log: Callable[[str], None] = print) -> SyncResult:
    """Download new replays uploaded in ``[since, until)`` (Los Angeles dates). Stops at ``max_replays`` WRITTEN."""
    since_ts = la_midnight_ts(since) if since else None
    until_ts = la_midnight_ts(until) if until else None
    have = existing_ids(out)
    written = skipped = 0
    folders: Set[str] = set()
    for row in search_pages(fmt, fetch, since_ts, until_ts):
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
        folder = folder_date(text, int(row["uploadtime"]))
        day = out / folder
        day.mkdir(parents=True, exist_ok=True)
        tmp = day / (file_name(rid) + ".part")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(day / file_name(rid))              # atomic: a killed run never leaves a torn .log
        have.add(rid)
        folders.add(folder)
        written += 1
        if written % 100 == 0:
            log(f"{written} written ({skipped} skipped)")
    return SyncResult(written, skipped, tuple(sorted(folders)))


def lock_path() -> Path:
    """ONE downloader per machine, whatever its ``--out``: the lock lives in the user's runtime dir."""
    return Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "gen3ai_replay_corpus_downloader.lock"


class Locked(RuntimeError):
    pass


@contextlib.contextmanager
def single_instance(path: Optional[Path] = None) -> Iterator[None]:
    """Hold the machine-wide downloader lock (an ``flock``: it dies with the process, so it never goes stale)."""
    path = path or lock_path()
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise Locked(f"another replay downloader holds {path}") from e
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield
    finally:
        os.close(fd)


def add_fetch_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", required=True, help="corpus root: an ABSOLUTE path to <main>/replays/showdown/gen3ou")
    ap.add_argument("--format", default="gen3ou")
    ap.add_argument("--proxy", required=True, metavar="SOCKS5_URL",
                    help="the owner's SOCKS5 proxy, e.g. socks5h://127.0.0.1:1080 (every request goes through it)")
    ap.add_argument("--sleep", type=float, default=MIN_SLEEP_S,
                    help=f"seconds between requests (default and floor {MIN_SLEEP_S})")


def checked_fetch(ap: argparse.ArgumentParser, a: argparse.Namespace) -> Fetch:
    """The polite, proxied fetcher for parsed args; ``ProxyError`` when the proxy is malformed or down."""
    if a.sleep < MIN_SLEEP_S:
        ap.error(f"--sleep below {MIN_SLEEP_S} s is refused: this is a public server")
    check_proxy(a.proxy)
    return make_http_fetch(a.sleep, proxy=a.proxy)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_fetch_args(ap)
    ap.add_argument("--since", help="only replays uploaded on/after this Los Angeles date (YYYY-MM-DD)")
    ap.add_argument("--until", help="only replays uploaded BEFORE this Los Angeles date (YYYY-MM-DD)")
    ap.add_argument("--max-replays", type=int, default=1000, help="stop after writing this many (default 1000)")
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args(argv)
    try:
        fetch = checked_fetch(ap, a)
        with single_instance():
            res = sync(Path(a.out), a.format, fetch, since=a.since, until=a.until,
                       max_replays=a.max_replays, list_only=a.list_only)
    except ProxyError as e:
        print(f"STOPPED, no request made: {e}", file=sys.stderr)
        return EXIT_PROXY_DOWN
    except Locked as e:
        print(f"STOPPED, no request made: {e}", file=sys.stderr)
        return EXIT_LOCKED
    except FetchError as e:
        print(f"STOPPED on a persistent failure: {e}", file=sys.stderr)
        return EXIT_FAILURE
    print(res.describe(a.list_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
