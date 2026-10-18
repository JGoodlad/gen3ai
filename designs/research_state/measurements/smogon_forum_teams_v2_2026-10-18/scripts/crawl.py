#!/usr/bin/env python3
"""THE RESUMABLE CRAWL STATE for the v2 mega sweep (one shard per crawler agent).

Every crawler agent reads Smogon through WebFetch (the forum JS-checks plain clients, v1 F1) and
records what it read HERE, so a stalled agent's successor resumes from disk. Per shard:

  crawl/shards/<shard>/queue.jsonl   {"url", "kind": "listing"|"thread", "meta": {...}}  (append-only)
  crawl/shards/<shard>/pages.jsonl   one row per page READ (or refused): url, status, notes   (append-only)
  crawl/shards/<shard>/teams.jsonl   one row per team REFERENCE found on a page               (append-only)

Commands (python -I crawl.py <cmd> ...; JSON on stdin where noted):
  throttle <host>           block until >= 2.0 s since the last request to <host> by ANY agent (flock)
  enqueue <shard>           stdin: JSON list of {"url","kind","meta"}; already-queued urls are skipped
  next <shard> [n]          print the next n (default 1) queued urls not yet in pages.jsonl
  page <shard>              stdin: one JSON row {"url","status","thread_url","thread_title","page", ...}
  team <shard>              stdin: JSON list of team rows (see TEAM_FIELDS)
  status [shard]            counts per shard
  inline-check A B          two transcription files of one inline team: exit 0 iff they agree on every
                            set field (species, item, ability, nature, EVs, IVs, moves)
"""
import fcntl
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "crawl"
LOCKS = Path.home() / ".cache" / "gen3ai" / "smogon_crawl_locks"
MIN_GAP_S = 2.0
TEAM_FIELDS = ("kind", "url", "text", "page_url", "thread_url", "thread_title", "post_author",
               "post_date", "post_no", "team_name", "curator", "tier", "tier_reason", "category")


def _shard(name):
    d = ROOT / "shards" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rows(f):
    if not f.exists():
        return []
    return [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]


def _append(f, rows):
    with open(f, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def throttle(host):
    LOCKS.mkdir(parents=True, exist_ok=True)
    with open(LOCKS / f"{host}.lock", "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        stamp = LOCKS / f"{host}.last"
        last = float(stamp.read_text()) if stamp.exists() else 0.0
        wait = MIN_GAP_S - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        stamp.write_text(str(time.time()))


def main(argv):
    cmd = argv[0]
    if cmd == "throttle":
        throttle(argv[1])
    elif cmd == "enqueue":
        d = _shard(argv[1])
        have = {r["url"] for r in _rows(d / "queue.jsonl")}
        new = [r for r in json.loads(sys.stdin.read()) if r["url"] not in have]
        seen = set()
        new = [r for r in new if not (r["url"] in seen or seen.add(r["url"]))]
        _append(d / "queue.jsonl", new)
        print(f"enqueued {len(new)}")
    elif cmd == "next":
        d = _shard(argv[1])
        n = int(argv[2]) if len(argv) > 2 else 1
        done = {r["url"] for r in _rows(d / "pages.jsonl")}
        todo = [r for r in _rows(d / "queue.jsonl") if r["url"] not in done]
        for r in todo[:n]:
            print(json.dumps(r, ensure_ascii=False))
        print(f"# remaining {len(todo)}", file=sys.stderr)
    elif cmd == "page":
        row = json.loads(sys.stdin.read())
        row.setdefault("read_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        _append(_shard(argv[1]) / "pages.jsonl", [row])
    elif cmd == "team":
        rows = json.loads(sys.stdin.read())
        for r in rows:
            bad = set(r) - set(TEAM_FIELDS) - {"notes", "transcribed", "reads_agree"}
            if bad:
                raise SystemExit(f"unknown team fields {bad}")
            if r.get("kind") not in ("pokepaste", "pastebin", "inline", "other_link"):
                raise SystemExit(f"bad kind {r.get('kind')}")
            if r.get("tier") not in ("A", "B", "C"):
                raise SystemExit(f"bad tier {r.get('tier')}")
        _append(_shard(argv[1]) / "teams.jsonl", rows)
        print(f"recorded {len(rows)}")
    elif cmd == "status":
        names = [argv[1]] if len(argv) > 1 else sorted(p.name for p in (ROOT / "shards").iterdir())
        for n in names:
            d = _shard(n)
            q, p, t = (_rows(d / f) for f in ("queue.jsonl", "pages.jsonl", "teams.jsonl"))
            print(f"{n}: queued {len(q)} read {len(p)} remaining {len({r['url'] for r in q} - {r['url'] for r in p})} team-refs {len(t)}")
    elif cmd == "inline-check":
        sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))
        from main.anchors.team_overlap import parse_team
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from legacy_export import canon  # format-only canonical spelling (see its docstring)
        texts = [canon(Path(x).read_text()) for x in argv[1:3]]
        a, b = (parse_team(t) for t in texts)
        ivs = [[ln.strip() for ln in t.splitlines() if ln.strip().lower().startswith("ivs:")] for t in texts]
        same = (sorted(m.strict_key for m in a.mons) == sorted(m.strict_key for m in b.mons)
                and len(a.mons) == 6 and ivs[0] == ivs[1])
        print("AGREE" if same else "DISAGREE")
        return 0 if same else 1
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
