"""CROSS-STATE IMPORT: a group's cells for some models taken from an EARLIER read's state (``designs/tools/reads.md``
§3; N1-P FINDING 6: look 2's spec re-played every look-1 final, about 60 % of look 2's CPU).

A group's ``import: {state, spec_sha256, group, models}`` (``main.reads.spec.Import``) names the earlier state, the
sha256 of ITS ``spec.json`` (registered before the import is read), the earlier group and the labels imported. Those
labels are never planned or played here. The import is refused (:class:`ReadImportError`, a typed refusal) unless:

* the earlier state's ``spec.json`` hashes to ``spec_sha256``, and it is not this state;
* the earlier group is the same cell: an anchor group of the same opponent whose team sets and unit seeds include
  this group's; an S group on the same frozen R (``refpop_content_sha256``), panel, ``pairs`` and ``batch_pairs``,
  and the same registered ``regime_id`` where both register one;
* each imported label is a model of the earlier group with the same arm, seed and load options, and the bytes the
  earlier state PLAYED (its ``models.json`` sha256) are the bytes this state resolved (its ``models.json``).

Engine-matched (REVIEW F F-M2): the imported units or S rows must have played on THIS read's engine
(:mod:`main.reads.engine`): an import across engines, or of rows whose engine nobody can name beside this read's
own, refuses (:func:`check_engine`). A regime id cannot tell two banked engines apart, so the engines are compared
themselves.

Regime-matched: imported anchor units enter :func:`main.reads.assemble.anchor_group` with this state's own, so a
(group, team set) whose units span two regime ids is INCONCLUSIVE, exactly as within one state; an imported S node
must carry the group's regime id. The record names every import (state, spec sha256, row digest, the earlier
playing commit).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.reads import layout as LY


class ReadImportError(RuntimeError):
    """An import that is not the same cell, the same bytes, the same engine, or the registered earlier read (named)."""


def check_engine(what: str, here: Optional[str], there: Optional[str]) -> None:
    """Refuse an import whose rows played on another engine than this read's own (or on one nobody can name)."""
    from main.reads import engine as EN

    try:
        EN.require_same(what, here, there)
    except EN.EngineError as exc:
        raise ReadImportError(str(exc)) from None


def check_peer_pack(what: str, mine: Sequence[Mapping[str, Any]], theirs: Sequence[Mapping[str, Any]]) -> None:
    """Refuse an import whose counted units, on a team set this state also played, ran another TEAM-PACK regime
    (gen3_peer_showdown_pack_v1, ``EXTERNAL_ANCHORS_SOP.md`` H22): banked forum72 / away / pool units had Kakuna play
    poke-env's pack (forum72's sft431 Gyarados at Frustration base power 1); a unit after the fix plays Showdown's. Foul Play's
    banked units played its own pack's sets (Hidden Power at the all-31 IV type), a unit after the front end plays the
    validator's re-pack (``frontend_changed``; ``gen3_frontend_validated_pack_v1``). The
    regime ids already differ, which alone would only make the group INCONCLUSIVE; an import never crosses them."""
    here: Dict[str, set] = {}
    for r in mine:
        here.setdefault(str(r.get("panel")), set()).add(str(r.get("peer_pack", "legacy")))
    for r in theirs:
        if not r.get("counted") or str(r.get("panel")) not in here:
            continue
        state = str(r.get("peer_pack", "legacy"))
        if here[str(r.get("panel"))] != {state}:
            raise ReadImportError(f"{what}: team set {r.get('panel')} imported unit {r.get('unit')} played the "
                                  f"{state!r} team pack, this read's units {sorted(here[str(r.get('panel'))])}: an "
                                  f"import never crosses a team-pack regime (SOP H22: Kakuna's pack, Foul Play's played team)")


def earlier_state(imp: Any) -> Path:
    from main.reads.supervisors import canonical

    return Path(canonical(imp.state))


def load_earlier(imp: Any, *, this_state: Optional[Path] = None) -> Any:
    """The earlier state's spec, its bytes checked against ``imp.spec_sha256``."""
    from main.reads.spec import SpecError, load as load_spec
    from main.reads.supervisors import canonical

    st = earlier_state(imp)
    if this_state is not None and canonical(this_state) == str(st):
        raise ReadImportError(f"import from {st}: that is this read's own state")
    sf = LY.spec_file(st)
    try:
        got = hashlib.sha256(sf.read_bytes()).hexdigest()
    except OSError as exc:
        raise ReadImportError(f"import from {st}: no spec.json ({exc})") from None
    if got != imp.spec_sha256:
        raise ReadImportError(f"import from {st}: spec.json sha256 {got[:12]} is not the registered "
                              f"{imp.spec_sha256[:12]}")
    try:
        return load_spec(sf, rule_params=False)          # its rule's params were checked under its own day's rules
    except SpecError as exc:
        raise ReadImportError(f"import from {st}: its spec does not load ({exc})") from None


def check(spec: Any, lane: str, g: str, *, this_state: Optional[Path] = None) -> Any:
    """Every precondition of group ``g``'s import that the two SPECS decide (module docstring); returns the earlier
    spec. Pure but for reading the earlier ``spec.json``."""
    grp = (spec.anchor if lane == "anchor" else spec.s)[g]
    imp = grp.imports
    old = load_earlier(imp, this_state=this_state)
    pool = old.anchor if lane == "anchor" else old.s
    if imp.group not in pool:
        raise ReadImportError(f"{lane}.{g}: the earlier read {old.read} has no {lane} group {imp.group!r}")
    og = pool[imp.group]
    probs: List[str] = []
    if lane == "anchor":
        if og.opponent != grp.opponent:
            probs.append(f"opponent {og.opponent} is not {grp.opponent}")
        if not set(grp.teamsets) <= set(og.teamsets):
            probs.append(f"team sets {sorted(set(grp.teamsets) - set(og.teamsets))} were not played there")
        if not set(grp.unit_seeds) <= set(og.unit_seeds):
            probs.append(f"unit seeds {sorted(set(grp.unit_seeds) - set(og.unit_seeds))} were not played there")
    else:
        for k in ("refpop_content_sha256", "panel", "pairs", "batch_pairs"):
            if getattr(og, k) != getattr(grp, k):
                probs.append(f"{k} {getattr(og, k)!r} is not {getattr(grp, k)!r}")
        if og.regime_id and grp.regime_id and og.regime_id != grp.regime_id:
            probs.append(f"registered regime {og.regime_id} is not {grp.regime_id}")
    for lab in imp.models:
        if lab not in og.models:
            probs.append(f"{lab} is not a model of the earlier group")
            continue
        a, b = spec.models[lab], old.models[lab]
        if (a.arm, a.seed, a.load) != (b.arm, b.seed, b.load):
            probs.append(f"{lab}: arm / seed / load {(b.arm, b.seed, b.load)} there, {(a.arm, a.seed, a.load)} here")
    if probs:
        raise ReadImportError(f"{lane}.{g} import from {old.read} ({imp.group}): not the same cell: {probs}")
    return old


def _played_shas(st: Path) -> Dict[str, str]:
    p = LY.models_file(st)
    return {lab: e["sha256"] for lab, e in json.loads(p.read_text()).items()} if p.exists() else {}


def check_shas(imp: Any, shas: Mapping[str, str]) -> Dict[str, str]:
    """``{label: sha256}`` of the imported labels this state has resolved, each equal to what the earlier state
    played; a label the earlier state never resolved, or resolved to other bytes, refuses."""
    played = _played_shas(earlier_state(imp))
    out: Dict[str, str] = {}
    for lab in imp.models:
        if lab not in shas:
            continue                    # not resolved here yet: the group names it unresolved
        if played.get(lab) != shas[lab]:
            raise ReadImportError(f"import {lab}: the earlier state played sha "
                                  f"{str(played.get(lab))[:12]}, this read resolved {shas[lab][:12]}")
        out[lab] = shas[lab]
    return out


def _meta(imp: Any, old: Any, rows_digest: str, n: int) -> Dict[str, Any]:
    from main.reads import repin as RP

    st = earlier_state(imp)
    cf = LY.code_file(st)
    return {"state": str(st), "read": old.read, "spec_sha256": imp.spec_sha256, "group": imp.group,
            "labels": list(imp.models), "rows": n, "rows_sha256": rows_digest,
            "code": json.loads(cf.read_text()) if cf.exists() else None, "repins": RP.load_repins(st)}


def anchor_rows(spec: Any, g: str, shas: Mapping[str, str], *, this_state: Optional[Path] = None,
                own: Sequence[Mapping[str, Any]] = ()) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The imported labels' PLAYED unit rows on this group's team sets and unit seeds, and the import's record.
    ``own``: this state's own rows of the group; when any counted, the imported counted rows must share their ONE
    engine (:func:`check_engine`)."""
    from main.reads import engine as EN

    from main.reads.assemble import anchor_units

    grp = spec.anchor[g]
    imp = grp.imports
    old = check(spec, "anchor", g, this_state=this_state)
    have = check_shas(imp, shas)
    rows: List[Dict[str, Any]] = []
    for lab in sorted(have):
        st = LY.anchor_state(earlier_state(imp), imp.group, lab)
        if not (st / "plan.json").exists():
            continue
        rows += [dict(r, imported_from=str(earlier_state(imp))) for r in anchor_units(st)
                 if r["panel"] in grp.teamsets and int(r["unit_seed"]) in grp.unit_seeds]
    digest = hashlib.sha256(json.dumps(sorted(rows, key=lambda r: r["unit"]), sort_keys=True,
                                       default=str).encode()).hexdigest()
    meta = _meta(imp, old, digest, len(rows))
    meta["regime_ids"] = {ts: sorted({str(r.get("regime_id")) for r in rows if r["panel"] == ts and r["counted"]})
                          for ts in grp.teamsets}
    theirs, why = EN.one(r.get("engine") for r in rows if r["counted"])
    meta["engine"] = theirs
    mine = [r for r in own if r.get("counted")]
    check_peer_pack(f"anchor.{g} import from {old.read} ({imp.group})", mine, rows)
    if mine and any(r["counted"] for r in rows):
        here, why_here = EN.one(r.get("engine") for r in mine)
        if why or why_here:
            raise ReadImportError(f"anchor.{g} import from {old.read}: {why or why_here}")
        check_engine(f"anchor.{g} import from {old.read} ({imp.group})", here, theirs)
    return rows, meta


def s_group(spec: Any, g: str, shas: Mapping[str, str], members: Mapping[str, str], *,
            this_state: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The imported labels' S nodes, read from the earlier state's ledger and family, and the import's record."""
    from main.reads.assemble import s_group as _s_group

    x = spec.s[g]
    imp = x.imports
    old = check(spec, "s", g, this_state=this_state)
    have = check_shas(imp, shas)
    ox = old.s[imp.group]
    grp = _s_group(LY.s_ledger(earlier_state(imp), imp.group), ox.family, {lab: ox.request(lab) for lab in have},
                   have, members, panel=x.panel, want_regime=x.regime_id or ox.regime_id,
                   refpop_content_sha256=x.refpop_content_sha256, min_pairs=x.pairs)
    meta = _meta(imp, old, str(grp.get("ledger_digest")), int(grp.get("rows") or 0))
    meta.update(family=ox.family, ledger_digest=grp.get("ledger_digest"), regime_id=grp.get("regime_id"),
                engine=grp.get("engine"))
    return grp, meta
