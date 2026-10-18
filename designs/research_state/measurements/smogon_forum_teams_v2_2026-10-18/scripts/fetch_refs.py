#!/usr/bin/env python3
"""Fetch every paste a crawler recorded (pokepast.es JSON, pastebin.com paste PAGE) into the cache.

Per-host throttle shared with the crawlers (crawl.py throttle, >= 2 s). Robots: pokepast.es has no
robots.txt (404, 2026-10-18); pastebin.com disallows /raw/, /dl/, /print/, /embed* -- so only the plain
/<id> page is fetched. Every other host is recorded, never fetched here (README).

    python -I fetch_refs.py <cache_dir>      # idempotent: cached URLs are skipped; errors are logged
"""
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crawl  # noqa: E402

UA = "Mozilla/5.0 (X11; Linux x86_64) gen3ai-research/0.1 (read-only, throttled)"
PP = re.compile(r"https?://(?:www\.)?pokepast\.es/([0-9a-f]{16})")
PB = re.compile(r"https?://(?:www\.)?pastebin\.com/(?:raw/)?([A-Za-z0-9]{8})\b")


SPOINK = re.compile(r"^(?:https?://)?spo\.ink/([A-Za-z0-9]+)$")
RESOLVED = HERE.parent / "crawl" / "spoink_resolved.json"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def spoink_map():
    return json.loads(RESOLVED.read_text()) if RESOLVED.exists() else {}


def resolve_spoink(urls):
    """spo.ink is Smogon's own short-link service (spo.ink/<id> -> www.smogon.com/spoink/<id>, whose
    robots.txt allows it); ONE GET per id, redirects NOT followed, the Location header recorded."""
    done = spoink_map()
    opener = urllib.request.build_opener(_NoRedirect)
    for u in sorted(set(urls)):
        m = SPOINK.match(u.strip())
        if not m or m.group(1) in done:
            continue
        crawl.throttle("www.smogon.com")
        req = urllib.request.Request(f"https://www.smogon.com/spoink/{m.group(1)}", headers={"User-Agent": UA})
        try:
            opener.open(req, timeout=30)
            done[m.group(1)] = {"status": "no redirect"}
        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location") if e.code in (301, 302, 303, 307, 308) else None
            done[m.group(1)] = {"status": e.code, "location": loc}
        except Exception as e:  # noqa: BLE001
            done[m.group(1)] = {"status": f"{type(e).__name__}"}
        RESOLVED.write_text(json.dumps(done, indent=1, sort_keys=True) + "\n")
    return done


def canonical(url):
    m = SPOINK.match((url or "").strip())
    if m:
        url = (spoink_map().get(m.group(1)) or {}).get("location") or url
    m = PP.match(url or "")
    if m:
        return "pokepast.es", f"https://pokepast.es/{m.group(1)}/json"
    m = PB.match(url or "")
    if m:
        return "pastebin.com", f"https://pastebin.com/{m.group(1)}"
    return None, None


def main():
    cache = Path(sys.argv[1])
    cache.mkdir(parents=True, exist_ok=True)
    log = cache / "fetch_refs_errors.jsonl"
    resolve_spoink([r.get("url") or "" for f in (HERE.parent / "crawl" / "shards").glob("*/teams.jsonl")
                    for r in crawl._rows(f) if "spo.ink" in (r.get("url") or "")])
    urls = {}
    for f in (HERE.parent / "crawl" / "shards").glob("*/teams.jsonl"):
        for r in crawl._rows(f):
            host, u = canonical(r.get("url"))
            if u:
                urls[u] = host
    gone = {json.loads(ln)["url"] for ln in (log.read_text().splitlines() if log.exists() else [])
            if "HTTP Error 404" in ln or "HTTP Error 410" in ln}
    n_new = n_err = 0
    for u, host in sorted(urls.items()):
        f = cache / (hashlib.sha1(u.encode()).hexdigest() + ".html")
        if f.exists() or u in gone:
            continue
        crawl.throttle(host)
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=30) as r:
                body = r.read()
        except Exception as e:  # noqa: BLE001 -- recorded; never retried in a loop
            n_err += 1
            with open(log, "a") as fh:
                fh.write(json.dumps({"url": u, "error": f"{type(e).__name__}: {e}"}) + "\n")
            continue
        f.write_bytes(body)
        (cache / (f.name + ".url")).write_text(u + "\n")
        n_new += 1
    print(f"refs {len(urls)} fetched {n_new} errors {n_err}")


if __name__ == "__main__":
    main()
