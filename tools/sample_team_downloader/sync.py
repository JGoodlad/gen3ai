"""Sync ``data/teams/sample/`` to EXACTLY the teams linked from the first post of Smogon's ADV OU
sample-teams thread.

``data/teams/sample/`` has ONE writer: this tool. It owns ``sample/teams.json`` outright and REFUSES
to touch a manifest holding any entry it did not write (a promoted team, a hand-added one) — the
curated set and the "legal exploiter trainee" set are different roles and live in different folders
(``data/teams/promoted/`` is ``python -m main.promote_teams``'s; see ``utils.team_loader``).

Four guarantees, each pinned by ``sample_team_downloader_test.py`` on saved HTML:

* **Structural pairing.** Each team's title is read from the line that FOLLOWS ITS OWN link
  (``<a href=pokepast><img…></a><br>Title – by Author<br>``). The old tool zipped a list of links
  with a list of " – by " lines, so one stray link or one missing title shifted every later team's
  name, author and category onto the wrong paste — silently.
* **A team link without a title is a REFUSAL**, not a skip: it means the post's structure changed
  under us. A pokepast link with no sprite images (a plain-text link in a description) is not a team
  and is ignored, and so is any link inside a spoiler or quote.
* **The team count is pinned** to the committed manifest's: a post that now links a different number
  of teams fails loudly unless ``--accept-count-change`` is passed.
* **Nothing is written unless every paste downloaded.** A network failure used to drop that team
  from the manifest; now it aborts the sync with the files untouched.

A paste the thread no longer links (the thread REPLACED it, as it did Curse RestLax's on some date
before 2026-09-23) is not deleted: archived runs recorded its path and fingerprint. It is MOVED to
``data/teams/superseded/`` (listed in that folder's ``teams.json``, which the pool does not draw
from) and ``data/teams/relocations.json`` maps its old path to the new one, so
``utils.team_loader.resolve_team_file`` keeps every archived reference working.

Usage (from the repo root):
    python tools/sample_team_downloader/sync.py [--dry-run] [--accept-count-change]
    npm run sync-teams
"""
import argparse
import datetime
import json
import os
import re
import sys

SMOGON_URL = "https://www.smogon.com/forums/threads/adv-ou-sample-teams.3687813/"
OUTPUT_DIR = "data/teams/sample"
METADATA_PATH = os.path.join(OUTPUT_DIR, "teams.json")
SUPERSEDED_DIR = "data/teams/superseded"
RELOCATIONS_PATH = "data/teams/relocations.json"

#: The exact key set of a manifest entry this tool writes (in write order).
TOOL_KEYS = ("url", "name", "author", "category", "id", "file", "source")
_POKEPASTE = re.compile(r"pokepast\.es/")
_BY = re.compile(r"\s+[–—-]\s*by\s+", re.IGNORECASE)
_BLOCK_TAGS = {"div", "p", "ul", "ol", "li", "table", "blockquote", "hr"}


class SyncError(RuntimeError):
    """The sync cannot proceed without a human decision. Nothing has been written."""


# ── parsing ─────────────────────────────────────────────────────────────────────────────────────

def _is_team_link(a) -> bool:
    """A TEAM link is a pokepast anchor that shows sprites (every team in the thread is a row of six
    Pokémon icons) and is not inside a spoiler or quote (descriptions sometimes link a variant)."""
    if a.find("img") is None:
        return False
    for parent in a.parents:
        cls = parent.get("class") or []
        if any(c.startswith("bbCodeSpoiler") or c.startswith("bbCodeBlock--quote") for c in cls):
            return False
    return True


def _title_line_after(a) -> str:
    """The text of the line that follows the anchor's own line: skip to the first ``<br>`` after the
    anchor, then gather text until the next ``<br>`` or block element. ``""`` if there is none."""
    parts = []
    seen_break = False
    for node in a.next_siblings:
        name = getattr(node, "name", None)
        if name == "br":
            if seen_break:
                break
            seen_break = True
            continue
        if name in _BLOCK_TAGS:
            break
        if not seen_break:
            # still on the anchor's own line — only whitespace belongs here
            text = node.get_text() if name else str(node)
            if text.strip():
                return ""
            continue
        parts.append(node.get_text() if name else str(node))
    return " ".join("".join(parts).split())


def _category_for(a) -> str:
    prev_bold = a.find_previous(["b", "strong"])
    if prev_bold is None:
        return ""
    cat = " ".join(prev_bold.get_text().split())
    return cat if len(cat) < 50 else ""


def extract_metadata(first_post):
    """``[{url, name, author, category}, ...]`` in post order, one per TEAM link, each title read
    from the line after its own link. Raises :class:`SyncError` for a team link with no title."""
    teams, seen = [], set()
    for a in first_post.find_all("a", href=_POKEPASTE):
        if not _is_team_link(a):
            continue
        url = a.get("href").strip().rstrip("/")
        if url in seen:
            print(f"Warning: {url} is linked twice in the first post — keeping the first")
            continue
        line = _title_line_after(a)
        parts = _BY.split(line, maxsplit=1)
        if not line or len(parts) != 2 or not parts[0].strip():
            raise SyncError(
                f"team link {url} is not followed by a 'Title – by Author' line (got {line!r}). "
                "The first post's structure has changed; refusing to guess which title belongs to "
                "which paste.")
        name = re.sub(r"[\s–—-]+$", "", parts[0]).strip()
        teams.append({"url": url, "name": name, "author": parts[1].strip(),
                      "category": _category_for(a)})
        seen.add(url)
    return teams


def find_first_post(html: str):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    post = soup.select_one(".message-inner .bbWrapper")
    if post is None:
        raise SyncError("could not find the first post's content (.message-inner .bbWrapper)")
    return post


# ── the manifest this tool owns ─────────────────────────────────────────────────────────────────

def paste_id(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def make_entry(meta: dict) -> dict:
    pid = paste_id(meta["url"])
    return {"url": meta["url"], "name": meta["name"] or f"Team {pid}", "author": meta["author"],
            "category": meta["category"], "id": pid, "file": f"teams/sample/{pid}.txt",
            "source": meta["url"]}


def is_tool_entry(entry: dict) -> bool:
    """True iff ``entry`` has exactly the shape this tool writes — its keys, a pokepast ``url`` equal
    to ``source``, and ``file == teams/sample/<id>.txt``. A promoted entry (``promoted`` key, no
    ``url``) or a hand-added one fails, and its presence makes the manifest not ours to rewrite."""
    if set(entry) != set(TOOL_KEYS):
        return False
    url, pid = entry.get("url") or "", entry.get("id") or ""
    return (bool(_POKEPASTE.search(url)) and entry.get("source") == url and paste_id(url) == pid
            and entry.get("file") == f"teams/sample/{pid}.txt")


def load_owned_manifest(path: str = METADATA_PATH):
    """The current manifest, which must be entirely this tool's. Missing file → ``[]``."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    foreign = [r for r in rows if not is_tool_entry(r)]
    if foreign:
        ids = [r.get("id") or r.get("file") for r in foreign]
        raise SyncError(
            f"{path} holds {len(foreign)} entr{'y' if len(foreign) == 1 else 'ies'} this tool did not "
            f"write (e.g. {ids[:5]}). data/teams/sample/ is EXACTLY Smogon's thread; promoted teams "
            "live in data/teams/promoted/ (python -m main.promote_teams). Move the foreign entries "
            "out before syncing — this tool will not delete them.")
    return rows


def check_count(n_post: int, n_committed: int, accept: bool) -> None:
    if n_committed and n_post != n_committed and not accept:
        raise SyncError(
            f"the first post now links {n_post} teams; the committed manifest has {n_committed}. "
            "Read the thread, then re-run with --accept-count-change if the change is real.")


def plan(post_teams, existing):
    """``{"added": [...], "removed": [...], "kept": [...]}`` by paste id, each list in its source order."""
    new_ids = [paste_id(t["url"]) for t in post_teams]
    old_ids = [e["id"] for e in existing]
    return {"added": [i for i in new_ids if i not in old_ids],
            "removed": [i for i in old_ids if i not in new_ids],
            "kept": [i for i in new_ids if i in old_ids]}


# ── I/O ─────────────────────────────────────────────────────────────────────────────────────────

def fetch_team_text(url: str) -> str:
    import requests
    res = requests.get(f"{url}/raw", timeout=10)
    if res.status_code != 200:
        raise SyncError(f"{url}/raw returned HTTP {res.status_code}")
    # Both encodings are named on purpose. `requests` falls back to ISO-8859-1 for a text/*
    # response carrying no charset (RFC 2616), which turns a UTF-8 `é` into `Ã©`; and a bare
    # `open(..., 'w')` writes in the LOCALE encoding, so the same file is not even reproducible
    # across machines. `data/teams/others/` already carries the mojibake this produces — see
    # `others_team_downloader/sync.py`.
    if (res.encoding or "iso-8859-1").lower() in ("iso-8859-1", "latin-1", "latin1"):
        res.encoding = res.apparent_encoding or "utf-8"
    if not res.text.strip():
        raise SyncError(f"{url}/raw returned an empty paste")
    return res.text


def _write_json(path: str, obj) -> None:
    # indent=2, ASCII-escaped, no trailing newline — the format every committed data/teams manifest
    # already has (this tool's predecessor and main.promote_teams both wrote it), so a no-op sync
    # is a byte-identical file.
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=2))


def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def retire(pid: str, old_entry: dict, replaced_by, stamp: str, root: str = ".") -> None:
    """Move a paste the thread no longer links into ``superseded/`` + record the relocation."""
    src = os.path.join(root, OUTPUT_DIR, f"{pid}.txt")
    dst_dir = os.path.join(root, SUPERSEDED_DIR)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{pid}.txt")
    if os.path.exists(src):
        os.replace(src, dst)
    man_path = os.path.join(dst_dir, "teams.json")
    rows = _read_json(man_path, [])
    if not any(r.get("id") == pid for r in rows):
        rows.append({**{k: old_entry[k] for k in TOOL_KEYS},
                     "file": f"teams/superseded/{pid}.txt",
                     "superseded": {"at": stamp, "replaced_by": replaced_by, "thread": SMOGON_URL,
                                    "by": "tools/sample_team_downloader/sync.py"}})
        _write_json(man_path, rows)
    rel_path = os.path.join(root, RELOCATIONS_PATH)
    reloc = _read_json(rel_path, {"moved": {}})
    reloc.setdefault("moved", {})[f"teams/sample/{pid}.txt"] = f"teams/superseded/{pid}.txt"
    _write_json(rel_path, reloc)


def sync_teams(argv=None, *, fetch_thread=None, fetch_team=None, root: str = ".") -> int:
    """Main entry point. ``fetch_thread`` / ``fetch_team`` are injectable for tests."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    ap.add_argument("--accept-count-change", action="store_true",
                    help="allow the first post's team count to differ from the committed manifest")
    args = ap.parse_args(argv)

    if fetch_thread is None:
        import requests

        def fetch_thread():
            r = requests.get(SMOGON_URL, timeout=30)
            if r.status_code != 200:
                raise SyncError(f"failed to fetch the Smogon thread: HTTP {r.status_code}")
            return r.text
    fetch_team = fetch_team or fetch_team_text

    try:
        print(f"Fetching Smogon thread: {SMOGON_URL}")
        post_teams = extract_metadata(find_first_post(fetch_thread()))
        existing = load_owned_manifest(os.path.join(root, METADATA_PATH))
        check_count(len(post_teams), len(existing), args.accept_count_change)
        p = plan(post_teams, existing)
        print(f"first post: {len(post_teams)} teams · committed: {len(existing)} · "
              f"kept {len(p['kept'])} · added {p['added'] or 'none'} · removed {p['removed'] or 'none'}")
        if args.dry_run:
            for t in post_teams:
                print(f"  [{t['category']}] {t['name']} — {t['author']} ({t['url']})")
            print("--dry-run: nothing written.")
            return 0
        # Download EVERYTHING before writing ANYTHING.
        texts = {}
        for t in post_teams:
            print(f"Fetching: [{t['category']}] {t['name']} by {t['author']} ({t['url']})")
            texts[paste_id(t["url"])] = fetch_team(t["url"])
    except SyncError as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 1

    os.makedirs(os.path.join(root, OUTPUT_DIR), exist_ok=True)
    for pid, text in texts.items():
        with open(os.path.join(root, OUTPUT_DIR, f"{pid}.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
    stamp = datetime.date.today().isoformat()
    by_name = {t["name"]: paste_id(t["url"]) for t in post_teams}
    old_by_id = {e["id"]: e for e in existing}
    for pid in p["removed"]:
        repl = by_name.get(old_by_id[pid]["name"])
        retire(pid, old_by_id[pid], repl if repl in p["added"] else None, stamp, root)
        print(f"retired {pid} ({old_by_id[pid]['name']}) → {SUPERSEDED_DIR}/"
              + (f" — replaced by {repl}" if repl in p["added"] else " — no longer linked"))
    _write_json(os.path.join(root, METADATA_PATH), [make_entry(t) for t in post_teams])
    print(f"Synced {len(post_teams)} teams to {OUTPUT_DIR} (manifest {METADATA_PATH})")
    return 0


if __name__ == "__main__":
    sys.exit(sync_teams())
