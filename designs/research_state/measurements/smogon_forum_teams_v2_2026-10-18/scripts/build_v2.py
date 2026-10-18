#!/usr/bin/env python3
"""Build v2: v1's register (unchanged, imported from the v1 folder) + every team reference the crawlers
recorded in ../crawl/shards/*/teams.jsonl -> normalise -> validate -> dedupe -> overlap -> write.

Normalisation, validation and the overlap matcher are v1's own functions (imported, not copied), so
v1 and v2 teams are processed identically. v1's manifest is never written.

    PYTHONPATH=src nice -n 19 taskset -c 5-7,13-15 python build_v2.py <cache_dir>
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent.parent / "smogon_forum_teams_2026-10-18"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(V1 / "scripts"))

import build as v1  # noqa: E402  (v1's normalisation, validation wrapper, references)
import crawl  # noqa: E402
import fetch_refs  # noqa: E402
import legacy_export  # noqa: E402
import pagetext  # noqa: E402

from main.anchors import team_overlap as to  # noqa: E402
from main.anchors.session import load_team_texts  # noqa: E402
from utils.bridge.team_validator import validate_teams_locally  # noqa: E402

OUT = HERE.parent
TIER_RANK = {"A": 0, "B": 1, "C": 2}
SEP_RE = re.compile(r"^\s*[-=_*]{2,}\s*$")
FORMAT_NOTE_RE = re.compile(r"Format:\s*(\S+)", re.IGNORECASE)


def split_separated(text):
    """A paste of several teams separated by separator-only blocks (``--``) is split ONLY when every
    segment holds exactly six mon blocks (unambiguous); otherwise it stays one candidate."""
    blocks = [b for b in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()) if b.strip()]
    if not any(SEP_RE.match(b) for b in blocks):
        return None
    segs, cur = [], []
    for b in blocks:
        if SEP_RE.match(b):
            segs.append(cur)
            cur = []
        else:
            cur.append(b)
    segs.append(cur)
    segs = [s for s in segs if s]
    if segs and all(len(s) == 6 for s in segs):
        return ["\n\n".join(s) for s in segs]
    return None


def v1_candidates(cache):
    for c in v1.candidates(cache):
        c = dict(c)
        c.setdefault("curator", c.get("credited") or c.get("post_author") or "")
        c["origin"] = "v1_register"
        yield c


def crawl_refs():
    """Every crawler team row, grouped by canonical paste URL (one candidate per paste; all posts that
    referenced it kept in ``seen_in``); inline rows stay one per post."""
    groups = {}
    for f in sorted((HERE.parent / "crawl" / "shards").glob("*/teams.jsonl")):
        shard = f.parent.name
        for r in crawl._rows(f):
            r = dict(r, shard=shard)
            host, canon = fetch_refs.canonical(r.get("url"))
            key = canon or (r.get("url") if r["kind"] != "inline" else f"inline:{r.get('url')}")
            groups.setdefault(key, []).append((canon, host, r))
    for key, refs in groups.items():
        refs.sort(key=lambda t: TIER_RANK[t[2]["tier"]])
        canon, host, best = refs[0]
        best = dict(best, seen_in=[{k: x[2].get(k) for k in ("page_url", "post_author", "post_date", "tier", "shard")}
                                   for x in refs])
        yield key, canon, host, best


def crawl_candidates(cache, unread):
    for key, canon, host, r in crawl_refs():
        base = {k: r.get(k, "") for k in crawl.TEAM_FIELDS if k != "text"}
        base.update(origin=f"crawl:{r['shard']}", source_url=r.get("url"), seen_in=r["seen_in"],
                    transcribed=bool(r.get("transcribed")), reads_agree=r.get("reads_agree"))
        if r["kind"] == "inline":
            if not r.get("reads_agree") or not r.get("text"):
                unread.append(dict(base, reason="inline transcription disagreement / no text"))
                continue
            yield dict(base, sub_index=0, sub_name="", sub_format="", text=r["text"])
            continue
        if canon is None:
            unread.append(dict(base, reason="host not fetched (not pokepast.es / pastebin.com)"))
            continue
        f = cache / (hashlib.sha1(canon.encode()).hexdigest() + ".html")
        if not f.exists():
            unread.append(dict(base, reason="paste not fetched (missing / 404 / error: see fetch log)"))
            continue
        body = f.read_text(encoding="utf-8", errors="replace")
        try:
            d = pagetext.pokepaste(body) if host == "pokepast.es" else pagetext.pastebin(body)
        except Exception as e:  # noqa: BLE001
            unread.append(dict(base, reason=f"paste unparseable: {type(e).__name__}"))
            continue
        note_fmt = FORMAT_NOTE_RE.search(d.get("paste_notes", "") or "")
        parts = v1.split_teams(d["text"])
        if len(parts) == 1 and not parts[0][0]:
            sep = split_separated(d["text"])
            if sep:
                parts = [("", "", s) for s in sep]
        for k, (fmt, name, text) in enumerate(parts):
            fmt = fmt or (note_fmt.group(1) if note_fmt else "")
            yield dict(base, sub_index=k, sub_name=name, sub_format=fmt, paste_title=d.get("paste_title", ""),
                       paste_author=d.get("paste_user", ""), paste_date=d.get("paste_date", ""),
                       paste_notes=d.get("paste_notes", ""), text=text)


def main() -> int:
    cache = Path(sys.argv[1])
    unread: list = []
    cands = list(v1_candidates(cache)) + list(crawl_candidates(cache, unread))
    cands.sort(key=lambda c: (TIER_RANK.get(c.get("tier"), 3), 0 if c["origin"] == "v1_register" else 1))
    rows = []
    for c in cands:
        flags: list = []
        row = {k: v for k, v in c.items() if k != "text"}
        row["raw_sha256"] = hashlib.sha256(c["text"].strip().encode()).hexdigest()
        try:
            if c["sub_format"] and c["sub_format"].lower() != "gen3ou":
                raise v1.Drop(f"format_tag:{c['sub_format']}")
            raw = c["text"]
            if c.get("transcribed"):
                canon_raw = legacy_export.canon(raw)
                if canon_raw.strip() != raw.strip():
                    flags.append("legacy_export_canonicalised")
                raw = canon_raw
            text = v1.fix_team(raw, flags)
        except v1.Drop as e:
            row.update(status="dropped", drop_reason=str(e))
            rows.append(row)
            continue
        row["norm_flags"] = sorted(set(flags))
        row["text"] = text
        rows.append(row)
    todo = [r for r in rows if "text" in r]
    verdicts = validate_teams_locally("gen3ou", [r["text"] for r in todo])
    assert len(verdicts) == len(todo)
    if any("Node execution error" in e or "Cannot find" in e for v in verdicts for e in (v.get("errors") or [])):
        raise SystemExit("VALIDATOR INFRASTRUCTURE FAILURE")
    seen, kept_texts = {}, []
    for r, v in zip(todo, verdicts):
        r["validator"] = {"valid": bool(v.get("valid")), "errors": v.get("errors") or []}
        if not v.get("valid"):
            r.update(status="dropped", drop_reason="validator")
            continue
        team = to.parse_team(r["text"], "")
        if team.key in seen:
            r.update(status="duplicate", duplicate_of=seen[team.key])
            continue
        r["id"] = f"sfv2_{len(seen):04d}"
        r["sha256"] = hashlib.sha256(r["text"].encode()).hexdigest()
        seen[team.key] = r["id"]
        r["status"] = "kept"
        kept_texts.append((r["id"], r["text"]))
    kept = [r for r in rows if r.get("status") == "kept"]
    by_id = {r["id"]: r for r in kept}
    for r in rows:
        if r.get("status") == "duplicate":
            by_id[r["duplicate_of"]].setdefault("also_seen_in", []).append(
                {k: r.get(k) for k in ("source_url", "thread_url", "post_author", "tier", "origin")})
    parsed = {r["id"]: to.parse_team(r["text"], r["id"]) for r in kept}
    refs = {}
    summary = {}
    for name, path in v1.REFERENCES.items():
        if path is not None and not Path(path).is_dir():
            summary[name] = {"missing": str(path)}
            continue
        refs[name] = v1.ref_index(name, path)
    refs["v1_set"] = to.Reference(to.parse_items(load_team_texts(V1 / "teams")))
    for name, ref in refs.items():
        for r in kept:
            n, lab = ref.max_shared(parsed[r["id"]])
            exact = bool(ref.keys[parsed[r["id"]].key])
            r.setdefault("overlap", {})[name] = {"status": "exact" if exact else ("near" if n >= 4 else "clean"),
                                                 "max_shared_mon_sets": n, "nearest": lab}
    ext = [n for n in refs if n != "v1_set"]
    # within-set nearest neighbour
    for r in kept:
        t = parsed[r["id"]]
        best = max(((len(t.key & o.key), o.label) for o in parsed.values() if o.label != t.label),
                   default=(0, None))
        r["within_set_max_shared"], r["within_set_nearest"] = best
    # independent cores: walk kept in (tier, register) order; clean vs every EXTERNAL reference and
    # <= 3 shared mon sets with every team already taken
    def core(tiers):
        taken = []
        for r in kept:
            if r.get("tier") not in tiers or not all(r["overlap"][n]["status"] == "clean" for n in ext):
                continue
            if all(len(parsed[r["id"]].key & parsed[c].key) <= 3 for c in taken):
                taken.append(r["id"])
        return taken
    core_ab, core_abc = core(("A", "B")), core(("A", "B", "C"))
    for r in kept:
        r["independent_core_AB"] = r["id"] in set(core_ab)
        r["independent_core_ABC"] = r["id"] in set(core_abc)
    for name, ref in refs.items():
        summary[name] = {"all": to.overlap(list(parsed.values()), ref),
                         "core_AB": to.overlap([parsed[i] for i in core_ab], ref),
                         "core_ABC": to.overlap([parsed[i] for i in core_abc], ref)}
        for t in "ABC":
            summary[name][t] = to.overlap([parsed[r["id"]] for r in kept if r.get("tier") == t], ref)
    tdir = OUT / "teams"
    tdir.mkdir(exist_ok=True)
    for f in tdir.glob("*.txt"):
        f.unlink()
    for tid, text in kept_texts:
        (tdir / f"{tid}.txt").write_text(text)

    def cnt(xs):
        return dict(Counter(xs).most_common())
    counts = {
        "candidates": len(rows), "unread_refs": len(unread),
        "by_status": cnt(r["status"] for r in rows),
        "drop_reasons": cnt((r.get("drop_reason") or "").split(":")[0] for r in rows if r["status"] == "dropped"),
        "kept_by_tier": cnt(r.get("tier") for r in kept),
        "kept_by_origin": cnt(r["origin"] for r in kept),
        "kept_by_category": cnt(r.get("category") for r in kept),
        "kept_by_curator_top30": dict(Counter(r.get("curator") or "?" for r in kept).most_common(30)),
        "pool_status_by_tier": {t: cnt(r["overlap"]["training_pool"]["status"] for r in kept if r.get("tier") == t)
                                for t in "ABC"},
        "new_vs_v1": cnt(r["overlap"]["v1_set"]["status"] for r in kept),
        "core_AB": len(core_ab), "core_ABC": len(core_abc),
        "core_ABC_by_tier": cnt(by_id[i].get("tier") for i in core_abc),
        "core_ABC_by_curator_top15": dict(Counter(by_id[i].get("curator") or "?" for i in core_abc).most_common(15)),
        "core_AB_by_curator_top15": dict(Counter(by_id[i].get("curator") or "?" for i in core_ab).most_common(15)),
        "unread_reasons": cnt(u["reason"] for u in unread),
    }
    for r in rows:
        r.pop("text", None)
    manifest = {"version": "smogon_forum_teams_v2", "built": "2026-10-18", "matcher": to.OVERLAP_VERSION,
                "v1": "smogon_forum_teams_2026-10-18 (manifest unchanged; its register is re-read here)",
                "validator": "utils.bridge.team_validator.validate_teams_locally('gen3ou'), pinned deps/pokemon-showdown",
                "rows": "manifest_teams.jsonl (one candidate per line) and manifest_unread.jsonl",
                "counts": counts, "overlap_summary": summary}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    (OUT / "manifest_teams.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                      for r in rows))
    (OUT / "manifest_unread.jsonl").write_text("".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n"
                                                       for u in unread))
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
