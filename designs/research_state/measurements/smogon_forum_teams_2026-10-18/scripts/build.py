#!/usr/bin/env python3
"""Build the Smogon-forum third-source team set: extract -> normalise -> validate -> dedupe -> overlap.

Reads ONLY the throttled page cache that ``fetch.py`` wrote (no network here) plus ``inline_teams.py``
(the transcribed rows). Writes ``teams/<id>.txt`` and ``manifest.json`` next to this directory.

    PYTHONPATH=src nice -n 19 taskset -c 5-7,13-15 python build.py <cache_dir>

Normalisation is ``make_home72.py``'s (n0_endofrun_2026-09-27), with ONE difference stated in the
README: a Hidden Power whose paste already carries an ``IVs:`` line that GIVES the bracketed type keeps
that line (home72 always replaced it); one that does not is replaced by the canonical spread, as
home72 does, and the original line is recorded in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pagetext
import sources
from inline_teams import INLINE

from main.anchors import team_overlap as to
from main.anchors.session import load_team_texts, pool_team_texts
from utils.bridge.team_validator import validate_teams_locally
from utils.gen3_utils import GEN3_HP_IVS
from utils.paths import repo_path

OUT = HERE.parent
STAT_ORDER = [("hp", "HP"), ("atk", "Atk"), ("def", "Def"), ("spa", "SpA"), ("spd", "SpD"), ("spe", "Spe")]
HP_RE = re.compile(r"^-\s*Hidden Power\s*(?:\[\s*([A-Za-z]+)\s*\]|([A-Za-z]+))?\s*$", re.IGNORECASE)
NICK_RE = re.compile(r"^(?P<nick>.+?)\s*\((?P<species>[^()]+)\)(?P<rest>\s*(\((?:M|F)\))?\s*(@.*)?)$")
HEADER_RE = re.compile(r"^===\s*(?:\[(?P<fmt>[^\]]*)\])?\s*(?P<name>.*?)\s*===\s*$")
HP_TYPES = {k.lower(): k for k in GEN3_HP_IVS}

REFERENCES = {
    "training_pool": None,  # pool_team_texts()
    "sample": repo_path("data", "teams", "sample"),
    "metamon_competitive": Path("/home/goodlad/dev/metamon/cache/teams/competitive/gen3ou"),
    "hl_05_26": Path.home() / "gen3ai_archive/metamon_cache_2026-09-24/teams/hl_05_26/gen3ou",
    "heldout72": repo_path("designs", "research_state", "measurements",
                           "heldout_teams_kakuna_2026-10-18", "teams_heldout72"),
}


class Drop(Exception):
    pass


def cache_body(cache: Path, url: str) -> str:
    f = cache / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if not f.exists():
        raise FileNotFoundError(url)
    return f.read_text(encoding="utf-8", errors="replace")


def split_teams(text: str):
    """``[(format, name, body)]``: a paste with ``=== [fmt] name ===`` headers is several teams."""
    lines = text.replace("\r\n", "\n").split("\n")
    if not any(HEADER_RE.match(ln.strip()) for ln in lines):
        return [("", "", text)]
    out, cur = [], None
    for ln in lines:
        m = HEADER_RE.match(ln.strip())
        if m:
            cur = [m.group("fmt") or "", m.group("name"), []]
            out.append(cur)
        elif cur is not None:
            cur[2].append(ln)
    return [(f, n, "\n".join(b)) for f, n, b in out]


LEGACY_STATS = {"Spd": "Spe", "SAtk": "SpA", "SDef": "SpD", "SpAtk": "SpA", "SpDef": "SpD"}


def modern_export(lines: list, flags: list) -> list:
    """The pre-2015 export spelling (``Trait:``, ``Spd`` / ``SAtk`` / ``SDef``, ``Adamant Nature (+Atk,
    -SAtk)``) in today's words — a SPELLING change only; every value is kept."""
    out = []
    for ln in lines:
        new = ln
        if new.startswith("Trait:"):
            new = "Ability:" + new[len("Trait:"):]
        if new.lower().startswith(("evs:", "ivs:")):
            for old, mod in LEGACY_STATS.items():
                new = re.sub(rf"\b{old}\b", mod, new)
        new = re.sub(r"^(\w+ Nature)\s*\([^)]*\)\s*$", r"\1", new)
        if new != ln:
            flags.append("legacy_export_spelling")
        out.append(new)
    return out


def fix_block(block: str, flags: list) -> str:
    lines = modern_export([ln.rstrip() for ln in block.strip().split("\n")], flags)
    head = lines[0].strip()
    m = NICK_RE.match(head)
    if m and m.group("species").strip() not in ("M", "F"):
        head = f"{m.group('species').strip()}{m.group('rest')}"
        flags.append("nickname_removed")
    lines[0] = head
    hp_type = None
    for i, ln in enumerate(lines):
        mm = HP_RE.match(ln.strip())
        if not mm:
            continue
        typ = (mm.group(1) or mm.group(2) or "").lower()
        if not typ:
            iv_line = next((x for x in lines if x.strip().lower().startswith("ivs:")), None)
            if iv_line is None:
                raise Drop("hidden_power_type_unknown")
            typ = to.hidden_power_type(to._parse_spread(iv_line.split(":", 1)[1]))
            flags.append("hp_type_from_page_ivs")
        if typ not in HP_TYPES:
            raise Drop(f"hidden_power_bad_type:{typ}")
        hp_type = HP_TYPES[typ]
        lines[i] = f"- Hidden Power [{hp_type}]"
    if hp_type is None:
        return "\n".join(lines)
    iv_lines = [ln for ln in lines if ln.strip().lower().startswith("ivs:")]
    if iv_lines and to.hidden_power_type(to._parse_spread(iv_lines[0].split(":", 1)[1])) == hp_type.lower():
        return "\n".join(lines)
    if iv_lines:
        flags.append(f"hp_ivs_replaced:{iv_lines[0].strip()}")
    lines = [ln for ln in lines if not ln.strip().lower().startswith("ivs:")]
    ivs = GEN3_HP_IVS[hp_type]
    parts = [f"{ivs[k]} {lab}" for k, lab in STAT_ORDER if ivs[k] != 31]
    if parts:
        first_move = next(i for i, ln in enumerate(lines) if ln.startswith("-"))
        lines.insert(first_move, "IVs: " + " / ".join(parts))
        flags.append("hp_ivs_inserted")
    return "\n".join(lines)


def fix_team(text: str, flags: list) -> str:
    blocks = [b for b in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()) if b.strip()]
    if len(blocks) != 6:
        raise Drop(f"not_6_mons:{len(blocks)}")
    return "\n\n".join(fix_block(b, flags) for b in blocks) + "\n"


def candidates(cache: Path):
    """Every team in the register, in priority order (the first copy of a duplicate is kept)."""
    for src in sources.pokepaste_sources():
        url = f"https://pokepast.es/{src['paste_id']}"
        d = pagetext.pokepaste(cache_body(cache, url + "/json"))
        for k, (fmt, name, body) in enumerate(split_teams(d["text"])):
            yield dict(src, source_url=url, sub_index=k, sub_name=name, sub_format=fmt,
                       paste_title=d["paste_title"], paste_author=d["paste_user"],
                       paste_notes=d.get("paste_notes", ""), text=body)
    for row in INLINE:
        yield dict(row, sub_index=0, sub_name="", sub_format="", paste_title="", paste_author="",
                   paste_notes="", text=row["text"])
    for src in sources.pastebin_sources():
        url = f"https://pastebin.com/{src['paste_id']}"
        d = pagetext.pastebin(cache_body(cache, url))
        for k, (fmt, name, body) in enumerate(split_teams(d["text"])):
            yield dict({k2: v for k2, v in src.items() if k2 != "multi"}, source_url=url, sub_index=k,
                       sub_name=name, sub_format=fmt, paste_title=d["paste_title"],
                       paste_author=d["paste_user"], paste_date=d["paste_date"], paste_notes="", text=body)


def ref_index(name: str, path):
    if name == "training_pool":
        items = [(f"pool{i:03d}", t) for i, t in enumerate(pool_team_texts())]
    else:
        items = load_team_texts(Path(path))
    return to.Reference(to.parse_items(items))


def main() -> int:
    cache = Path(sys.argv[1])
    rows, kept_texts = [], []
    seen_keys = {}
    for c in candidates(cache):
        flags: list = []
        row = {k: v for k, v in c.items() if k != "text"}
        row["raw_sha256"] = hashlib.sha256(c["text"].strip().encode()).hexdigest()
        try:
            if c["sub_format"] and c["sub_format"].lower() != "gen3ou":
                raise Drop(f"format_tag:{c['sub_format']}")
            text = fix_team(c["text"], flags)
        except Drop as e:
            row.update(status="dropped", drop_reason=str(e))
            rows.append(row)
            continue
        row["norm_flags"] = sorted(set(flags))
        row["text"] = text
        rows.append(row)
    todo = [r for r in rows if "text" in r]
    verdicts = validate_teams_locally("gen3ou", [r["text"] for r in todo])
    assert len(verdicts) == len(todo)
    infra = [v for v in verdicts if any("Node execution error" in e or "Cannot find" in e
                                        for e in (v.get("errors") or []))]
    if infra:  # an environment failure is never a team verdict
        raise SystemExit(f"VALIDATOR INFRASTRUCTURE FAILURE ({len(infra)}): {infra[0]['errors'][0][:400]}")
    for r, v in zip(todo, verdicts):
        r["validator"] = {"valid": bool(v.get("valid")), "errors": v.get("errors") or []}
        if not v.get("valid"):
            r["status"] = "dropped"
            r["drop_reason"] = "validator"
            continue
        team = to.parse_team(r["text"], r["source_url"])
        if team.key in seen_keys:
            r["status"] = "duplicate"
            r["duplicate_of"] = seen_keys[team.key]
            continue
        r["sha256"] = hashlib.sha256(r["text"].encode()).hexdigest()
        r["id"] = f"sft{len(seen_keys):03d}"
        seen_keys[team.key] = r["id"]
        r["status"] = "kept"
        kept_texts.append((r["id"], r["text"]))
    kept = [r for r in rows if r.get("status") == "kept"]
    parsed = {r["id"]: to.parse_team(r["text"], r["id"]) for r in kept}
    # within-set near neighbours (>= 4 shared mon sets with another KEPT team)
    self_ref = to.Reference(list(parsed.values()))
    for r in kept:
        t = parsed[r["id"]]
        best, lab = 0, None
        for o in self_ref.teams:
            if o.label == t.label:
                continue
            n = len(t.key & o.key)
            if n > best:
                best, lab = n, o.label
        r["within_set_max_shared"] = best
        r["within_set_nearest"] = lab
    summary = {}
    refs = {}
    for name, path in REFERENCES.items():
        if path is not None and not Path(path).is_dir():
            summary[name] = {"missing": str(path)}
            continue
        ref = refs[name] = ref_index(name, path)
        teams = [parsed[r["id"]] for r in kept]
        summary[name] = dict(source=str(path) if path else str(repo_path("data", "teams")),
                             all=to.overlap(teams, ref))
        for tier in ("A", "B"):
            summary[name][tier] = to.overlap([parsed[r["id"]] for r in kept if r["tier"] == tier], ref)
        for r in kept:
            n, lab = ref.max_shared(parsed[r["id"]])
            exact = bool(ref.keys[parsed[r["id"]].key])
            r.setdefault("overlap", {})[name] = {
                "status": "exact" if exact else ("near" if n >= 4 else "clean"),
                "max_shared_mon_sets": n, "nearest": lab}
    # THE INDEPENDENT CORE (a flag, never an exclusion): walk the kept teams in register order and
    # take a team iff it is clean against EVERY reference and shares <= 3 mon sets with every team
    # already taken — the heldout72 draw rule applied within this set (the dumps hold many variants).
    core = []
    for r in kept:
        t = parsed[r["id"]]
        ok = all(v["status"] == "clean" for v in r["overlap"].values()) and r["tier"] in ("A", "B")
        ok = ok and all(len(t.key & parsed[c].key) <= 3 for c in core)
        r["independent_core"] = ok
        if ok:
            core.append(r["id"])
    core_teams = [parsed[c] for c in core]
    for name, ref in refs.items():
        summary[name]["independent_core"] = to.overlap(core_teams, ref)
    # write
    tdir = OUT / "teams"
    for f in tdir.glob("*.txt"):
        f.unlink()
    for tid, text in kept_texts:
        (tdir / f"{tid}.txt").write_text(text)
    clean_pool = [r for r in kept if r["overlap"]["training_pool"]["status"] == "clean"]
    counts = {
        "candidates": len(rows),
        "by_status": dict(Counter(r["status"] for r in rows)),
        "drop_reasons": dict(Counter((r.get("drop_reason") or "").split(":")[0] for r in rows
                                     if r["status"] == "dropped")),
        "kept_by_tier": dict(Counter(r["tier"] for r in kept)),
        "kept_by_category": dict(Counter(r["category"] for r in kept)),
        "pool_status_by_tier": {t: dict(Counter(r["overlap"]["training_pool"]["status"] for r in kept
                                                if r["tier"] == t)) for t in ("A", "B", "C")},
        "clean_vs_pool_AB": sum(1 for r in clean_pool if r["tier"] in ("A", "B")),
        "independent_core": len(core),
        "independent_core_by_tier": dict(Counter(r["tier"] for r in kept if r["independent_core"])),
        "independent_core_by_category": dict(Counter(r["category"] for r in kept if r["independent_core"])),
        "clean_vs_all_refs_AB": sum(1 for r in kept if r["tier"] in ("A", "B") and all(
            v["status"] == "clean" for v in r["overlap"].values())),
    }
    for r in rows:
        r.pop("text", None)
    manifest = {"version": "smogon_forum_teams_v1", "built": "2026-10-18", "matcher": to.OVERLAP_VERSION,
                "validator": "utils.bridge.team_validator.validate_teams_locally('gen3ou') — the pinned "
                             "deps/pokemon-showdown",
                "counts": counts, "overlap_summary": summary, "teams": rows}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
