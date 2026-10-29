"""(Re)build ``x63_union_v1``: X63's UNION corpus, three registered breadth corpora at their registered tiers, verbatim.

    python -m utils.team_corpora.build_x63_union [--check]

The components, read through the VERIFYING reader (:func:`utils.team_corpora.load_corpus`: data and tier sha256s, the
label guard), in THIS ORDER (:data:`COMPONENTS`): ``x62_recomb_v1@ALL`` (12,000), ``n1_proc_v1@K3`` (3,000, its
registered prefix tier) and ``d3_setmut_v1@ALL`` (2,876): 17,876 teams. Each row keeps its component's team text
byte for byte; ``curator`` names the component, ``component_id`` / ``component_i`` its row there. One tier, ``ALL``.

THE TWO REFUSALS (the build RAISES :class:`UnionError`; it never drops a team silently):

* ``duplicate``: two union teams with the same six mon sets (``gen3_team_overlap_v1``: species + sorted moves, Hidden
  Power typed), whatever their text (block order, nickname, item, spread);
* ``rc72b``: a union team sharing >= :data:`RC72B_REFUSE` (3) mon sets with ONE ``rc72b`` team (the X63 review's m3,
  the identity rc72b's and X62's builds use). ``diagnostic_panels.json`` postdates the play pin, so the pin's startup
  guard cannot see rc72b, and its radius is 4: this builder is the check.

Then the pinned validator (``ladder_corpus_filter.js``) must pass every union team, the engine (``scan_move_probe``,
this corpus's ENGINE PIN, empty) must run every id it carries, and the readers' label guard re-checks the rows: each
failure RAISES (the components passed them at their own builds; a union that does not is a moved input).

REPORTED, never refused (the manifest's ``describe``): the max shared-set count against every panel (``heldout72``,
``heldout72_hpfix``, ``forum72``, ``p72``, ``forum216b``, ``n1p72``, ``rc72b``) as histograms, overall and by component;
the cross-component near-duplicates (>= :data:`NEAR_DUP_REPORT` shared sets with ONE team of another component); the
component share of draws; and the Frustration teams (the legacy-pack MATERIAL class of
``x35_pack_regime_2026-10-27``; the audit itself runs outside the build, its output beside X63's registration).
Deterministic: the same components and panels give the same bytes (``--check``).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.paths import repo_path
from utils.team_corpora import (CORPORA, DATA_NAME, MANIFEST_NAME, SCHEMA, check_label_encodable, encode_rows,
                                 load_corpus, tier_sha256)

NAME = "x63_union_v1"
TIER = "ALL"
#: The components, IN UNION ORDER: (registered corpus, its registered tier).
COMPONENTS = (("x62_recomb_v1", "ALL"), ("n1_proc_v1", "K3"), ("d3_setmut_v1", "ALL"))
#: >= 3 shared mon sets with ONE rc72b team is refused (REVIEW A's revealed-fit radius, rc72b's and X62's).
RC72B_REFUSE = 3
#: Cross-component pairs at >= 5 shared sets are reported (never refused).
NEAR_DUP_REPORT = 5
#: Union teams per validator call (no effect on the output).
BATCH = 4_000
_FRUSTRATION = re.compile(r"^\s*-\s*Frustration\s*$", re.IGNORECASE | re.MULTILINE)
_HAPPY0 = re.compile(r"^\s*Happiness:\s*0\s*$", re.IGNORECASE | re.MULTILINE)


class UnionError(RuntimeError):
    """The union cannot be built as specified (a duplicate six, an rc72b near-copy, a team that no longer validates or
    an id the engine cannot run)."""


# ------------------------------------------------------------------------------------------------ the pure core
def assemble(components: Sequence[Tuple[str, str, Sequence[str], Sequence[str]]],
             rc72b: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """The union rows from ``(name, tier, ids, texts)`` components IN ORDER, refusing (raising :class:`UnionError`)
    a duplicate six across or within components and any team sharing >= :data:`RC72B_REFUSE` sets with ONE ``rc72b``
    team (``(label, text)`` items)."""
    from main.anchors.team_overlap import Reference, parse_items, parse_team

    ref = Reference(parse_items(rc72b))
    rows: List[Dict[str, Any]] = []
    seen: Dict[Any, str] = {}
    dups: List[str] = []
    near: List[str] = []
    for name, tier, ids, texts in components:
        if len(ids) != len(texts):
            raise UnionError(f"component {name}@{tier}: {len(ids)} ids for {len(texts)} texts")
        for ci, (cid, text) in enumerate(zip(ids, texts)):
            team = parse_team(text)
            where = f"{name}:{cid}"
            if text != text.strip():
                raise UnionError(f"{where}: the text is not .strip()ed (the corpus contract)")
            if len(team.mons) != 6:
                raise UnionError(f"{where} parses to {len(team.mons)} mon sets, not 6")
            if team.key in seen:
                dups.append(f"{where} == {seen[team.key]}")
            else:
                seen[team.key] = where
            k, lab = ref.max_shared(team)
            if k >= RC72B_REFUSE:
                near.append(f"{where} shares {k} with rc72b {lab}")
            i = len(rows)
            rows.append({"i": i, "id": f"x63u_{i:05d}", "tier": TIER, "curator": name, "component_tier": tier,
                         "component_id": str(cid), "component_i": ci,
                         "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text})
    if dups or near:
        raise UnionError(f"{len(dups)} duplicate six(es) {dups[:20]}; {len(near)} rc72b near-copies at radius "
                         f"{RC72B_REFUSE} {near[:20]}")
    return rows


def frustration_teams(text: str) -> Tuple[int, int]:
    """``(mons with Frustration, of them without an explicit Happiness: 0 line)`` of one team text."""
    blocks = [b for b in re.split(r"\n[ \t]*\n+", text.strip()) if b.strip()]
    fr = [b for b in blocks if _FRUSTRATION.search(b)]
    return len(fr), sum(1 for b in fr if not _HAPPY0.search(b))


# ------------------------------------------------------------------------------------------------- the build
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def describe(rows: Sequence[Dict[str, Any]], panels: Dict[str, Sequence[Tuple[str, str]]]) -> Dict[str, Any]:
    from main.anchors.team_overlap import Reference, parse_items, parse_team

    teams = [parse_team(r["text"], r["id"]) for r in rows]
    comp = [r["curator"] for r in rows]
    names = [n for n, _ in COMPONENTS]
    hist: Dict[str, Any] = {}
    for p, items in panels.items():
        ref = Reference(parse_items(items))
        ks = [ref.max_shared(t)[0] for t in teams]
        hist[p] = {"all": {str(k): v for k, v in sorted(collections.Counter(ks).items())}, "max": max(ks),
                   "by_component": {n: {str(k): v for k, v in sorted(collections.Counter(
                       k for k, c in zip(ks, comp) if c == n).items())} for n in names}}
    cross: Dict[str, Any] = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            ref = Reference([t for t, c in zip(teams, comp) if c == b])
            ks = [(t.label, *ref.max_shared(t)) for t, c in zip(teams, comp) if c == a]
            cross[f"{a} vs {b}"] = {
                "max_shared_hist": {str(k): v for k, v in sorted(collections.Counter(k for _, k, _ in ks).items())},
                f"at_ge_{NEAR_DUP_REPORT}": [[lab, k, other] for lab, k, other in ks if k >= NEAR_DUP_REPORT]}
    fr = [frustration_teams(r["text"]) for r in rows]
    n = len(rows)
    with_fr = [r["id"] for r, (a, _b) in zip(rows, fr) if a]
    no_h0 = [r["id"] for r, (_a, b) in zip(rows, fr) if b]
    return {
        "n_teams": n,
        "component_share_of_draws": {c: {"n": comp.count(c), "share": round(comp.count(c) / n, 4)} for c in names},
        "max_shared_hist_by_panel": hist,
        "cross_component_near_duplicates": cross,
        "frustration_by_text": {
            "rule": "a mon block with a '- Frustration' move line; 'no_happiness0' = no explicit 'Happiness: 0' line. "
                    "The legacy (pre-d8d674b9) packer played every Frustration at base power 1 "
                    "(x35_pack_regime_2026-10-27); the classified count is that audit's, run on this corpus",
            "teams_with_frustration": len(with_fr), "share_with_frustration": round(len(with_fr) / n, 5),
            "teams_without_happiness0": len(no_h0), "share_without_happiness0": round(len(no_h0) / n, 5),
            "by_component": {c: {"with_frustration": sum(1 for r, (a, _) in zip(rows, fr) if a and r["curator"] == c),
                                 "without_happiness0": sum(1 for r, (_, b) in zip(rows, fr) if b and r["curator"] == c)}
                             for c in names},
            "ids_without_happiness0": no_h0},
    }


def build() -> Tuple[bytes, Dict[str, Any]]:
    from main import heldout_panels as HPN
    from main.anchors.team_overlap import OVERLAP_VERSION
    from utils.ladder_corpus.build import engine_verdicts
    from utils.team_corpora import engine_pin
    from utils.team_corpora.build_w_forum import showdown_rows
    from utils.team_corpora.build_x62_recomb import N1P72_MANIFEST_REL, panel_items

    comps = [load_corpus(n, t) for n, t in COMPONENTS]
    panels = panel_items()
    rows = assemble([(c.name, c.tier, c.ids, c.texts) for c in comps], panels["rc72b"])
    sd: Dict[str, Dict[str, Any]] = {}
    for k in range(0, len(rows), BATCH):
        sd.update(showdown_rows({r["id"]: r["text"] for r in rows[k:k + BATCH]}))
    bad = sorted(f"{r['id']} ({r['curator']}:{r['component_id']}): {sd[r['id']]['verdict']}" for r in rows
                 if sd[r["id"]]["verdict"] != "ok")
    if bad:
        raise UnionError(f"{len(bad)} union team(s) the pinned validator rejects: {bad[:20]}")
    unplayable = engine_pin.unplayable(NAME, engine_verdicts([sd[r["id"]] for r in rows]))
    if any(unplayable.values()):
        raise UnionError(f"the union carries ids the engine cannot run: {unplayable}")
    check_label_encodable([r["id"] for r in rows], [r["text"] for r in rows], f"the {NAME} build")
    data = encode_rows(rows)
    texts = [r["text"] for r in rows]
    reg_path = HPN.registry_path()
    manifest = {
        "schema": SCHEMA, "name": NAME,
        "what": "X63's union corpus: x62_recomb_v1@ALL + n1_proc_v1@K3 + d3_setmut_v1@ALL, verbatim, in that order; "
                "TRAINING DATA, never a prior",
        "built_by": "python -m utils.team_corpora.build_x63_union",
        "components": [{"name": c.name, "tier": c.tier, "n": len(c), "tier_sha256": c.tier_sha256,
                        "data_sha256": c.data_sha256, "manifest_sha256": c.manifest_sha256,
                        "first_union_i": sum(len(x) for x in comps[:k])} for k, c in enumerate(comps)],
        "refusals": {"duplicate": "the same six mon sets as another union team (raises)",
                     "rc72b": f">= {RC72B_REFUSE} shared mon sets with ONE rc72b team (raises; the X63 review's m3)",
                     "matcher": OVERLAP_VERSION},
        "panels": {"histogrammed": {p: {"n": len(v), "content_sha256": _sha("\x00".join(sorted(t for _, t in v)).encode())}
                                    for p, v in panels.items()},
                   "panel_registry": {"path": HPN.REGISTRY_REL, "sha256": _sha(reg_path.read_bytes())},
                   "diagnostic_registry": {"path": HPN.DIAGNOSTIC_REL,
                                           "sha256": _sha(repo_path(*HPN.DIAGNOSTIC_REL.split("/")).read_bytes())},
                   "n1p72_manifest": N1P72_MANIFEST_REL},
        "validator": "TeamValidator('gen3ou') via src/rust_sim/harness/ladder_corpus_filter.js, pinned "
                     "deps/pokemon-showdown: every union team must pass (raises otherwise)",
        "engine": {"oracle": "scan_move_probe", "output_unplayable": unplayable,
                   "rule": "every id the union carries is asked; an unplayable one THROWS (the pin is empty)"},
        "describe": describe(rows, panels),
        "tiers": {TIER: {"n": len(texts), "sha256": tier_sha256(texts), "source_tiers": [TIER]}},
        "default_tier": TIER, "data_file": DATA_NAME, "data_sha256": _sha(data),
    }
    return data, manifest


def _dump(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 unless the committed files reproduce")
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    data, manifest = build()
    out = CORPORA[NAME]
    if a.check:
        same = ((out / DATA_NAME).exists() and (out / DATA_NAME).read_bytes() == data
                and (out / MANIFEST_NAME).read_text() == _dump(manifest))
        print("reproduces the committed corpus" if same else "DOES NOT reproduce the committed corpus")
        return 0 if same else 1
    out.mkdir(parents=True, exist_ok=True)
    (out / DATA_NAME).write_bytes(data)
    (out / MANIFEST_NAME).write_text(_dump(manifest))
    d = manifest["describe"]
    print(json.dumps({k: v["n"] for k, v in manifest["tiers"].items()}),
          json.dumps({p: v["max"] for p, v in d["max_shared_hist_by_panel"].items()}),
          "frustration w/o Happiness 0:", d["frustration_by_text"]["teams_without_happiness0"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
