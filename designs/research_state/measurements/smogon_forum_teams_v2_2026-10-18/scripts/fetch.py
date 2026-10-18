#!/usr/bin/env python3
"""THROTTLED, CACHED fetch of public pages (no login, no cookies kept, >= 2 s between requests).

Usage: python -I fetch.py <cache_dir> <url> [<url> ...]   -> prints "<status> <cache_file> <final_url>"
The cache file name is the sha1 of the URL; a cached URL is never fetched again.
"""
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (X11; Linux x86_64) gen3ai-research/0.1 (read-only, throttled)"
MIN_GAP_S = 2.5


def main() -> int:
    cache = Path(sys.argv[1])
    cache.mkdir(parents=True, exist_ok=True)
    stamp = cache / ".last_request"
    for url in sys.argv[2:]:
        f = cache / (hashlib.sha1(url.encode()).hexdigest() + ".html")
        if f.exists():
            print(f"CACHED {f} {url}")
            continue
        last = float(stamp.read_text()) if stamp.exists() else 0.0
        wait = MIN_GAP_S - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        stamp.write_text(str(time.time()))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                final = r.geturl()
                code = r.status
        except Exception as e:  # noqa: BLE001 - recorded, never retried in a loop
            print(f"ERR {type(e).__name__}:{e} {url}")
            continue
        f.write_bytes(body)
        (cache / (f.name + ".url")).write_text(f"{url}\n{final}\n")
        print(f"{code} {f} {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
