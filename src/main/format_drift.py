"""The FORMAT-SPEC drift check: does the gen3ou the LADDER runs still equal our declared spec?

``agents.gen3_data.format_spec`` declares gen3ou's rules and bans once. The public server runs Showdown
master; our engine is pinned. This module reads a Showdown tree's format files and compares them with the
spec, FAILING on any new or removed ban or clause:

* the ``[Gen 3] OU`` entry (``config/formats.ts``): its ``ruleset`` and ``banlist``;
* gen 3's ``Standard`` (``data/mods/gen3/rulesets.ts``) and the ``Standard AG`` it inherits from gen 4
  (``data/mods/gen4/rulesets.ts``);
* the Uber tier (``data/mods/gen3/formats-data.ts``);
* the clause BODIES the spec restates (``data/rulesets.ts``): the Evasion Items / Moves banlists, the One
  Boost Passer / Speed Pass boosting effects, Accuracy Trap's trapping list — compared as LISTS — and the
  board-state clauses (Sleep Clause Mod, Freeze Clause Mod) and the OHKO Clause compared as TEXT against the
  pinned engine's, because the op / move-resolution rules restate their mechanics (a changed body means
  "re-verify `status_rules`", not a list edit).

A committed snapshot of master's files (``src/main/format_drift_snapshot/``, excerpts of the two large files,
``MANIFEST.json`` naming the commit) makes the check offline and deterministic for tests;
``ladder_drift_scan`` re-runs it against a fresh master clone before any live session. Refresh the snapshot
with ``python -m main.format_drift snapshot --showdown <master checkout>`` and commit it with the spec edit
the new drift calls for.

    python -m main.format_drift check                       # the committed master snapshot (offline)
    python -m main.format_drift check --showdown <dir>      # any Showdown checkout (e.g. a master clone)
    python -m main.format_drift check --pinned              # deps/pokemon-showdown: lists the PINNED gaps

Exit 0 = the tree matches the spec, 1 = drift (each difference named).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from agents.gen3_data import format_spec as fs
from utils.paths import repo_path

SNAPSHOT_DIR = repo_path("src", "main", "format_drift_snapshot")
PINNED_ROOT = repo_path("deps", "pokemon-showdown")
FORMAT_NAME = "[Gen 3] OU"

FORMATS_TS = "config/formats.ts"
GEN3_RULESETS = "data/mods/gen3/rulesets.ts"
GEN4_RULESETS = "data/mods/gen4/rulesets.ts"
GEN3_FORMATS_DATA = "data/mods/gen3/formats-data.ts"
BASE_RULESETS = "data/rulesets.ts"
SOURCE_FILES: Tuple[str, ...] = (FORMATS_TS, GEN3_RULESETS, GEN4_RULESETS, GEN3_FORMATS_DATA, BASE_RULESETS)

#: ``data/rulesets.ts`` blocks the check reads (lists) or compares as text against the pinned engine.
LIST_CLAUSES: Tuple[str, ...] = ("evasionitemsclause", "evasionmovesclause", "oneboostpasserclause",
                                 "speedpassclause", "accuracytrapclause")
TEXT_CLAUSES: Tuple[str, ...] = ("sleepclausemod", "freezeclausemod", "ohkoclause")
#: Every ``data/rulesets.ts`` block the snapshot excerpts (the two groups above + the rules the spec cites).
EXCERPT_BLOCKS: Tuple[str, ...] = LIST_CLAUSES + TEXT_CLAUSES + (
    "standardag", "standard", "obtainable", "speciesclause", "nicknameclause", "switchpriorityclausemod",
    "endlessbattleclause", "hppercentagemod", "cancelmod", "beatupnicknamesmod")

_STR = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")


class FormatParseError(ValueError):
    """A Showdown file no longer has the shape the parser reads — itself a drift, never a silent pass."""


def _strings(src: str) -> List[str]:
    """Every JS string literal in ``src`` (single- or double-quoted, ``\\'`` unescaped), in order."""
    out: List[str] = []
    for m in _STR.finditer(src):
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        out.append(re.sub(r"\\(.)", r"\1", raw))
    return out


def _bracket_list(src: str, key: str) -> Optional[List[str]]:
    """The string list of ``key: [ … ]`` (or ``const key = [ … ]``) in ``src``; None if absent."""
    m = re.search(r"(?:\b" + re.escape(key) + r"\s*:\s*|const\s+" + re.escape(key) + r"\s*=\s*)\[", src)
    if not m:
        return None
    depth, i = 1, m.end()
    while depth and i < len(src):
        depth += {"[": 1, "]": -1}.get(src[i], 0)
        i += 1
    if depth:
        raise FormatParseError(f"unterminated list for {key!r}")
    return _strings(src[m.end():i - 1])


def top_block(text: str, key: str) -> Optional[str]:
    """The top-level ``\\t<key>: { … \\t},`` entry of a ``data/*.ts`` table (None if absent)."""
    m = re.search(r"^\t" + re.escape(key) + r": \{\n.*?^\t\},?$", text, re.M | re.S)
    return m.group(0) if m else None


def format_entry(formats_ts: str, name: str = FORMAT_NAME) -> str:
    """The object literal of the format ``name`` in ``config/formats.ts``."""
    m = re.search(r"^\t\{\n\t\tname: \"" + re.escape(name) + r"\",\n.*?^\t\},?$", formats_ts, re.M | re.S)
    if not m:
        raise FormatParseError(f"{FORMATS_TS}: no {name!r} entry")
    return m.group(0)


@dataclass(frozen=True)
class FormatFacts:
    """What a Showdown tree says about gen3ou (every field parsed from its files)."""
    format_ruleset: Tuple[str, ...]
    format_banlist: Tuple[str, ...]
    standard: Tuple[str, ...]
    standard_ag: Tuple[str, ...]
    ubers: FrozenSet[str]
    clause_lists: Dict[str, Dict[str, Tuple[str, ...]]]
    clause_text: Dict[str, str]


def _read(root: Path, rel: str) -> str:
    p = root / rel
    if not p.exists():
        raise FormatParseError(f"{p}: missing")
    return p.read_text(encoding="utf-8")


def _need(v: Optional[List[str]], what: str) -> Tuple[str, ...]:
    if v is None:
        raise FormatParseError(f"no {what}")
    return tuple(v)


def read_facts(root: Path) -> FormatFacts:
    """Parse a Showdown checkout (or the snapshot, which mirrors its layout)."""
    entry = format_entry(_read(root, FORMATS_TS))
    if "mod: 'gen3'" not in entry:
        raise FormatParseError(f"{FORMAT_NAME} no longer runs mod gen3: {entry[:200]!r}")
    g3 = top_block(_read(root, GEN3_RULESETS), "standard")
    g4 = top_block(_read(root, GEN4_RULESETS), "standardag")
    if g3 is None or g4 is None:
        raise FormatParseError("gen 3 Standard / gen 4 Standard AG block missing")
    fd = _read(root, GEN3_FORMATS_DATA)
    ubers = frozenset(m.group(1) for m in re.finditer(r"^\t(\w+): \{\n((?:\t\t.*\n)*?)\t\}", fd, re.M)
                      if re.search(r"tier: [\"']Uber[\"']", m.group(2)))
    if not ubers:
        raise FormatParseError(f"{GEN3_FORMATS_DATA}: no Uber entries parsed")
    base = _read(root, BASE_RULESETS)
    lists: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for key in LIST_CLAUSES:
        b = top_block(base, key)
        if b is None:
            raise FormatParseError(f"{BASE_RULESETS}: no {key} block")
        lists[key] = {k: tuple(v) for k in ("banlist", "boostingEffects", "trapping")
                      for v in [_bracket_list(b, k)] if v is not None}
    text: Dict[str, str] = {}
    for key in TEXT_CLAUSES:
        b = top_block(base, key)
        if b is None:
            raise FormatParseError(f"{BASE_RULESETS}: no {key} block")
        text[key] = re.sub(r"\s+", " ", b).strip()
    return FormatFacts(_need(_bracket_list(entry, "ruleset"), "format ruleset"),
                       _need(_bracket_list(entry, "banlist"), "format banlist"),
                       _need(_bracket_list(g3, "ruleset"), "gen3 Standard ruleset"),
                       _need(_bracket_list(g4, "ruleset"), "gen4 Standard AG ruleset"),
                       ubers, lists, text)


def expected(spec: fs.FormatSpec) -> Dict[str, object]:
    """The spec's side of every comparison, in the parsed facts' terms."""
    by_parent: Dict[str, List[str]] = {}
    for r in spec.rules:
        by_parent.setdefault(r.parent, []).append(r.name)
    pseudo = {"Banlist entities", "Banlist combos"}
    banlist = {"Uber"} | {b.name for r in spec.rules if r.name in pseudo for b in r.bans if b.via == "banlist"}
    return {
        "format_ruleset": frozenset(n for n in by_parent.get("format", []) if n not in pseudo),
        "format_banlist": frozenset(banlist),
        "standard": frozenset(by_parent.get("Standard", [])),
        "standard_ag": frozenset(by_parent.get("Standard AG", [])),
        "ubers": frozenset(sid for sid, _ in fs.UBER_SPECIES),
        "evasionitemsclause.banlist": frozenset(b.name for b in fs.EVASION_ITEM_BANS),
        "evasionmovesclause.banlist": frozenset(b.name for b in fs.EVASION_MOVE_BANS),
        "oneboostpasserclause.boostingEffects": fs.ONE_BOOST_PASSER_EFFECTS,
        "speedpassclause.boostingEffects": fs.SPEED_PASS_EFFECTS,
        "accuracytrapclause.trapping": fs.ACCURACY_TRAP_TRAPPING,
    }


def _observed(facts: FormatFacts) -> Dict[str, FrozenSet[str]]:
    obs: Dict[str, FrozenSet[str]] = {
        "format_ruleset": frozenset(facts.format_ruleset), "format_banlist": frozenset(facts.format_banlist),
        "standard": frozenset(facts.standard), "standard_ag": frozenset(facts.standard_ag),
        "ubers": facts.ubers,
    }
    for key, d in facts.clause_lists.items():
        for k, v in d.items():
            obs[f"{key}.{k}"] = frozenset(v)
    return obs


@dataclass(frozen=True)
class Diff:
    """One difference: ``key`` (a comparison, e.g. ``format_banlist``), ``kind`` (``new`` = in the tree, not
    the spec; ``removed`` = in the spec, not the tree; ``missing`` = the list is gone; ``text`` = a clause body
    changed), ``item``."""
    key: str
    kind: str
    item: str

    def __str__(self) -> str:
        if self.kind == "new":
            return f"{self.key}: NEW in the tree, absent from the spec: {self.item!r}"
        if self.kind == "removed":
            return f"{self.key}: in the spec, REMOVED from the tree: {self.item!r}"
        if self.kind == "missing":
            return f"{self.key}: not found in the tree (the clause body changed shape)"
        return (f"{self.key}: its body differs from the pinned engine's — the op / move-resolution rules "
                f"(agents/model/status_rules.py) restate this mechanic; re-verify them against it")


def compare(facts: FormatFacts, spec: Optional[fs.FormatSpec] = None,
            pinned_text: Optional[Dict[str, str]] = None) -> List[Diff]:
    """Every difference between a tree's ``facts`` and the spec (empty = no drift). ``pinned_text``: the
    pinned engine's normalised clause bodies for the TEXT comparison (default: read ``deps/pokemon-showdown``)."""
    spec = spec or fs.active()
    exp, obs = expected(spec), _observed(facts)
    out: List[Diff] = []
    for key, want in exp.items():
        got = obs.get(key)
        if got is None:
            out.append(Diff(key, "missing", ""))
            continue
        assert isinstance(want, frozenset)
        out += [Diff(key, "new", x) for x in sorted(got - want)]
        out += [Diff(key, "removed", x) for x in sorted(want - got)]
    if pinned_text is None:
        pinned_text = read_facts(PINNED_ROOT).clause_text
    out += [Diff(key, "text", "") for key in TEXT_CLAUSES if facts.clause_text.get(key) != pinned_text.get(key)]
    return out


def pinned_gaps(diffs: Sequence[Diff]) -> Tuple[List[Diff], List[Tuple[str, str]]]:
    """Split a PINNED tree's diffs against :data:`format_spec.PINNED_DIFFERENCES`: (unexpected diffs,
    declared gaps the tree no longer shows). Both empty = the pin differs from the ladder exactly as declared."""
    declared = set(fs.PINNED_DIFFERENCES)
    seen = {(d.key, d.item) for d in diffs if d.kind == "removed"}
    unexpected = [d for d in diffs if not (d.kind == "removed" and (d.key, d.item) in declared)]
    return unexpected, sorted(declared - seen)


# ------------------------------------------------------------------------------------------- snapshot
def write_snapshot(showdown: Path, out: Path, commit: str) -> None:
    """Excerpt a Showdown checkout into ``out`` (same layout): the three small files whole, the format entry
    of ``config/formats.ts`` and the cited blocks of ``data/rulesets.ts``; ``MANIFEST.json`` records the
    commit and the full files' sha256."""
    manifest: Dict[str, object] = {"commit": commit, "source": "smogon/pokemon-showdown", "files": {}}
    for rel in SOURCE_FILES:
        full = _read(showdown, rel)
        if rel == FORMATS_TS:
            body = "// EXCERPT: the [Gen 3] OU entry only\n" + format_entry(full) + "\n"
        elif rel == BASE_RULESETS:
            blocks = [b for k in EXCERPT_BLOCKS for b in [top_block(full, k)] if b is not None]
            body = "// EXCERPT: the blocks main.format_drift reads\n" + "\n".join(blocks) + "\n"
        else:
            body = full
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")
        files = manifest["files"]
        assert isinstance(files, dict)
        files[rel] = {"sha256_full": hashlib.sha256(full.encode("utf-8")).hexdigest(),
                      "excerpt": body is not full}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def snapshot_commit(root: Path = SNAPSHOT_DIR) -> str:
    return str(json.loads((root / "MANIFEST.json").read_text())["commit"])


def check(root: Path, spec: Optional[fs.FormatSpec] = None, label: str = "", pinned: bool = False) -> int:
    """Print the comparison of ``root`` with the spec; 0 = clean, 1 = drift. ``pinned``: ``root`` is the
    pinned engine, which may (and must) differ by exactly :data:`format_spec.PINNED_DIFFERENCES`."""
    try:
        diffs = compare(read_facts(root), spec)
    except FormatParseError as exc:
        print(f"[format] ✗ {label or root}: could not parse the format files: {exc}")
        return 1
    stale: List[Tuple[str, str]] = []
    if pinned:
        for key, item in fs.PINNED_DIFFERENCES:
            if any(d.key == key and d.item == item for d in diffs):
                print(f"[format]   declared pin gap: {key} {item!r} — {fs.PINNED_DIFFERENCES[(key, item)]}")
        diffs, stale = pinned_gaps(diffs)
    if diffs or stale:
        print(f"[format] ✗ FORMAT DRIFT vs agents.gen3_data.format_spec ({label or root}):")
        for d in diffs:
            print(f"     {d}")
        for key, item in stale:
            print(f"     {key}: the declared pin gap {item!r} is GONE — delete it from PINNED_DIFFERENCES")
        print("[format]   fix: update agents/gen3_data/format_spec.py (each entry with its source), give any "
              "new rule its ONE story (designs/endstate/design_format_spec.md), refresh the snapshot.")
        return 1
    print(f"[format] ✓ {label or root}: gen3ou's rules, banlist, Standard, Uber tier and clause bodies "
          f"match the spec{' (up to the declared pin gaps)' if pinned else ''}.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="compare a Showdown tree with the spec")
    c.add_argument("--showdown", type=Path, default=None, help="a Showdown checkout (default: the snapshot)")
    c.add_argument("--pinned", action="store_true", help="check deps/pokemon-showdown (lists the pin's gaps)")
    s = sub.add_parser("snapshot", help="re-excerpt a master checkout into the committed snapshot")
    s.add_argument("--showdown", type=Path, required=True)
    s.add_argument("--commit", required=True, help="the checkout's commit sha")
    s.add_argument("--out", type=Path, default=SNAPSHOT_DIR)
    args = ap.parse_args(argv)
    if args.cmd == "snapshot":
        write_snapshot(args.showdown, args.out, args.commit)
        print(f"[format] snapshot of {args.commit} → {args.out}")
        return 0
    if args.pinned:
        return check(PINNED_ROOT, label=f"pinned deps/pokemon-showdown @ {fs.PINNED_COMMIT[:8]}", pinned=True)
    if args.showdown is not None:
        return check(args.showdown)
    return check(SNAPSHOT_DIR, label=f"master snapshot @ {snapshot_commit()[:8]}")


if __name__ == "__main__":
    sys.exit(main())
