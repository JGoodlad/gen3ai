"""ASSEMBLE: the played units and ledger rows -> CELLS, each with its regime id (``designs/tools/reads.md`` §5).

THE REGIME ID IS ALWAYS RECORDED. Five S drivers wrote ``regime_ids: [""]`` because the ledger's ``Cell`` carries no
regime and they read it with ``getattr(c, "regime_id", "")`` (x26c FINDING 4, g1 FINDING 3); the regime was then
checked by hand on the raw rows. Here:

* an S group's regime is the eval-ledger READ's own (``LedgerRead.regime_id``: one family = one regime, a mixed
  family is the reader's typed refusal), recorded with the read's digest; a spec that registers a ``regime_id`` and a
  ledger that holds another, or a panel whose team set is not the rows', makes the group INCONCLUSIVE;
* an anchor unit's regime is :func:`anchor_regime_id` over the cell identity its ``summary.json`` stamps (opponent
  and version, both regimes, transport, server, format, forfeit limit, both team-content digests, the declared team
  cell): every counted unit of a (group, team set) must share ONE, else the group is INCONCLUSIVE, and the group's
  id is :func:`group_regime_id` over its team sets' ids. The playing commit
  (``server_version``) and file paths are recorded per unit, not part of the id (x26c FINDING 1: the play is the
  same across commits; ``code.json`` pins the commit separately).

A cell or node with no regime id is never conclusive.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training import eval_ledger as L

ANCHOR_REGIME_SCHEMA = "gen3_read_anchor_regime_v1"
#: The cell fields of an anchor summary that ARE the regime (everything else is recorded, not identity).
ANCHOR_REGIME_KEYS = ("battle_format", "forfeit_turn_limit", "mirrored_pairs", "opponent", "opponent_kind",
                      "opponent_version", "our_regime", "their_regime", "regime_matched", "our_transport",
                      "server_impl", "our_side", "model_loader", "our_teamset", "teamset", "our_team_count",
                      "their_team_count", "our_team_digest", "their_team_digest", "search_time_ms",
                      "search_parallelism", "showdown_pin")
CELLS_SCHEMA = "gen3_read_cells_v1"
PROTOCOL = "gen3_eval_protocol_v1_h2h"

#: The Metamon weights EVERY anchor read banked before the pins existed (FR-1 FINDING 3, 2026-10-25) played: the one
#: hub commit in the box's cache, ``{agent: weights_sha256}`` (``main.anchors.peer_scripts.weight_pins.weights_digest``
#: of the agent's ``{path: sha256}``, each file hashed against its blob). Those summaries stamp no weights, so their
#: regime ids were computed without them. A NEW summary whose verified weights equal this entry keeps that SAME id (the
#: weights are the ones the id always meant); any other revision or digest enters the id (:func:`anchor_regime_id`),
#: so different weights are a different regime. FROZEN: a repin never edits it.
LEGACY_HF_REVISION = "ac8e88e7da8dc2a2b85176d88f81fd8854b1445b"
LEGACY_WEIGHTS: Dict[str, str] = {
    "SmallRL": "b8849c52b443d7a1607c34e806dd959baa89c254aa0737c8bbdb8160a0f96ce2",
    "SyntheticRLV2": "3f00c9cbabe6e6bc776f3bbecdc9c134537704c7cc45c7f628fdda4d1e045961",
    "Kakuna": "bbfa8f2d9425341f0299c721ea847b7d74170cb72b80440951861efb4643992b",
    "Kakuna28": "7e4c71c28905d31c56cd8bb933c19eaac8641ade81fb3b64eb178181a38cf288",
    "LargeRL": "b26296d4963eb3d642ee7edcd16f069175bbc22083fa9e71a77addb8b7e82098",
    "LargeIL": "2167e509ef6e6123b8dbe8b69fe86a966d9781fc9d4c9c250d55d442a83aa02d",
    "PastaMittens": "b9f2c07e051c9a69d199484f919babc8b160ef7486cac578874fa22aa982e303",
}

#: S's look read: an A/B family's mirrored greedy-vs-greedy rows at the h2h protocol (``main.h2h.cross``'s terms).
S_FAMILY_READ = L.ReaderDecl(
    name="main.reads.assemble.s_family", purposes=frozenset({"ab"}),
    regime=L.RegimeFilter(protocol="gen3_eval_protocol_v1_h2h", play="greedy", opponent_play="greedy", mirrored=True),
    requests="family", selection="include", flags_ok=frozenset(), inference="across_runs",
    decision_kind="ab_verdict")


class AssembleError(RuntimeError):
    """The state cannot be assembled as this read (named)."""


# ------------------------------------------------------------------------------------------------ anchor
def anchor_regime_id(summary: Mapping[str, Any]) -> Optional[str]:
    """The regime id of one anchor unit (16 hex), or ``None`` when its summary stamps no cell."""
    cell = summary.get("cell")
    if not isinstance(cell, dict) or not cell.get("opponent_version"):
        return None
    ident = {k: cell.get(k) for k in ANCHOR_REGIME_KEYS}
    ident.update(_weights_identity(cell))
    ident.update(schema=ANCHOR_REGIME_SCHEMA, declared_team_cell=summary.get("declared_team_cell"),
                 team_digest_schema=summary.get("team_digest_schema"))
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


def _weights_identity(cell: Mapping[str, Any]) -> Dict[str, Any]:
    """The weight-identity inputs of a regime id: ``{"their_weights": {...}}`` / ``{"our_weights": {...}}`` for each
    Metamon side whose verified (revision, sha256) is NOT :data:`LEGACY_WEIGHTS`; ``{}`` for a summary that stamps none
    (a pre-pin read) or stamps the legacy weights, so every banked id is unchanged and a re-read of the same weights
    is comparable to it."""
    out: Dict[str, Any] = {}
    sides = (("their", cell.get("opponent_agent"), cell.get("hf_revision"), cell.get("weights_sha256")),
             ("our", str(cell.get("our_side") or "").partition("metamon:")[2], cell.get("our_hf_revision"),
              cell.get("our_weights_sha256")))
    for prefix, agent, revision, digest in sides:
        if not (revision and digest):
            continue
        if revision == LEGACY_HF_REVISION and LEGACY_WEIGHTS.get(agent) == digest:
            continue
        out[f"{prefix}_weights"] = {"hf_revision": revision, "weights_sha256": digest}
    return out


def group_regime_id(by_teamset: Mapping[str, Optional[str]]) -> str:
    """An anchor GROUP's one id: the digest of its per-team-set regime ids (each team set is its own cell, with its
    own team-content digests, so a group spans one regime per team set)."""
    return hashlib.sha256(json.dumps(dict(by_teamset), sort_keys=True).encode()).hexdigest()[:16]


#: A unit's per-game identity (``games.jsonl``): what a determinism guard compares game for game.
GAME_KEYS = ("half", "index", "our_team", "result", "turns")


def unit_games(kakuna_state: Path, name: str) -> Optional[List[List[Any]]]:
    """``[[half, index, our_team, result, turns], ...]`` of one played unit, in ``(half, index)`` order; ``None`` when
    the unit wrote no ``games.jsonl``."""
    p = kakuna_state / "cells" / name / "games.jsonl"
    try:
        rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return None
    return sorted(([r.get(k) for k in GAME_KEYS] for r in rows), key=lambda g: (str(g[0]), int(g[1])))


def unit_row(u: Any, summary: Mapping[str, Any], games: Optional[List[List[Any]]] = None) -> Dict[str, Any]:
    """One played unit's row (x26c's ``kakuna_units.jsonl`` fields, plus its regime, reasons, model-load options and,
    when given, its per-game identities :data:`GAME_KEYS`)."""
    from main.anchors.results import unit_counts

    uc = unit_counts(summary, games=int(u.games))
    cell = summary.get("cell") or {}
    return {"unit": u.name, "label": u.label, "panel": u.teamset, "unit_seed": int(u.unit_seed),
            "belief_prior_mix": getattr(u, "belief_prior_mix", None), "our_regime": cell.get("our_regime"),
            "games": games,
            "model_sha256": u.model_sha256, "status": summary.get("status"), "n": summary.get("n"),
            "wins": summary.get("wins"), "losses": summary.get("losses"), "ties": summary.get("ties", 0),
            "regime_verified_decisions": summary.get("regime_verified_decisions"),
            "peer_clean": summary.get("peer_clean"), "team_source_asymmetry": summary.get("team_source_asymmetry"),
            "counted": uc.counts, "reasons": list(uc.reasons), "regime_id": anchor_regime_id(summary),
            "server_version": cell.get("server_version"), "model_step": cell.get("model_step")}


def anchor_units(kakuna_state: Path) -> List[Dict[str, Any]]:
    """Every PLAYED unit of one ``main.x35.kakuna`` state (a unit with no summary is unplayed, not a row)."""
    from main.x35 import kakuna as KA

    rows = []
    for u in KA.load_plan(kakuna_state):
        s = KA.read_summary(kakuna_state, u.name)
        if s is not None:
            rows.append(unit_row(u, s, unit_games(kakuna_state, u.name)))
    return rows


def anchor_group(rows: Sequence[Mapping[str, Any]], *, labels: Sequence[str], teamsets: Sequence[str],
                 unit_seeds: Sequence[int], shas: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """One anchor group's cells ``{label: {teamset: {wins, ties, n, units, planned, skipped}}}`` over COUNTED units,
    its ONE regime id and every reason it is not conclusive. ``shas``: the resolved models' sha256 (a unit played
    on other bytes refuses)."""
    why: List[str] = []
    cells: Dict[str, Dict[str, Dict[str, Any]]] = {
        lab: {ts: {"wins": 0, "ties": 0, "n": 0, "units": 0, "planned": len(unit_seeds), "skipped": []}
              for ts in teamsets} for lab in labels}
    regimes: Dict[str, set] = {ts: set() for ts in teamsets}
    seen = set()
    for r in rows:
        key = (r["label"], r["panel"], int(r["unit_seed"]))
        if r["label"] not in cells or r["panel"] not in cells[r["label"]] or int(r["unit_seed"]) not in unit_seeds:
            raise AssembleError(f"unit {r['unit']} is not a unit of this group")
        if key in seen:
            raise AssembleError(f"unit {key} appears twice")
        seen.add(key)
        if shas is not None and r.get("model_sha256") and r["model_sha256"] != shas[r["label"]]:
            raise AssembleError(f"unit {r['unit']} played sha {str(r['model_sha256'])[:12]}, not "
                                f"{shas[r['label']][:12]}")
        c = cells[r["label"]][r["panel"]]
        if not r["counted"]:
            c["skipped"].append(r["unit"])
            continue
        if r.get("regime_id") is None:
            why.append(f"unit {r['unit']}: no regime id (its summary stamps no cell)")
        regimes[r["panel"]].add(r.get("regime_id"))
        c["units"] += 1
        c["wins"] += int(r["wins"])
        c["ties"] += int(r.get("ties", 0))
        c["n"] += int(r["n"])
    by_ts: Dict[str, Optional[str]] = {}
    for ts in teamsets:
        named = sorted(x for x in regimes[ts] if x is not None)
        by_ts[ts] = named[0] if len(named) == 1 else None
        if len(named) > 1:
            why.append(f"{ts}: counted units span {len(named)} regimes {named}")
        elif not named:
            why.append(f"{ts}: no counted unit carries a regime id")
    rid = group_regime_id(by_ts) if all(by_ts.values()) else None
    unplayed = len(labels) * len(teamsets) * len(unit_seeds) - len(seen)
    return {"regime_id": rid, "regime_ids": by_ts, "teamsets": list(teamsets), "unit_seeds": list(unit_seeds),
            "cells": cells, "units": sorted((dict(r) for r in rows), key=lambda r: r["unit"]),
            "unplayed": unplayed, "inconclusive": sorted(set(why))}


# ------------------------------------------------------------------------------------------------ S
def members_of(refpop: Path, *, verify: bool = True, want_content: Optional[str] = None) -> Dict[str, str]:
    """``{label: sha256}`` of a FROZEN R (``main.x35.d1.refpop_members``: every file re-hashed first), its
    ``content_sha256`` checked against ``want_content`` (R moved under a read: the drift pre-read's hazard 1)."""
    from main.x35 import d1 as D

    doc = json.loads(Path(refpop).read_text())
    if want_content is not None and doc.get("content_sha256") != want_content:
        raise AssembleError(f"{refpop}: content_sha256 {str(doc.get('content_sha256'))[:12]} is not the spec's "
                            f"{want_content[:12]}")
    return D.refpop_members(str(refpop), verify=verify)


def s_node(cells: Sequence[Any], sha: str, members: Mapping[str, str], *, min_pairs: int = 500) -> Dict[str, Any]:
    """One node's S against R: ``readout.population_score`` plus the per-member SE, se(S) and its 95 % interval."""
    from main.reads.rules import Z95, cell_se, s_se
    from main.x35 import readout as R

    label_of = {v: k for k, v in members.items()}
    ps = R.population_score(cells, sha, members, min_pairs=min_pairs)
    mine = [c for c in cells if c.player == sha and c.opponent in label_of]
    ses = {label_of[c.opponent]: 100.0 * cell_se(c.pairs) for c in mine if c.pairs and sum(c.pairs)}
    node: Dict[str, Any] = {"sha256": sha, "s_pp": ps.s_pp, "per_member_pp": dict(ps.per_member_pp),
                            "per_member_se_pp": ses, "inconclusive": list(ps.inconclusive),
                            "n_pairs": {label_of[c.opponent]: c.n_pairs for c in mine},
                            "population_score": json.loads(json.dumps(ps.as_dict())), "se_pp": None,
                            "ci95_pp": None}
    if ps.s_pp is not None and len(ses) == len(members):
        se = s_se(list(ses.values()))
        node.update(se_pp=se, ci95_pp=[ps.s_pp - Z95 * se, ps.s_pp + Z95 * se])
    return node


def cell_pairs(root: Path, family: str, request: str, sha: str, members: Mapping[str, str]) -> Dict[str, int]:
    """``{member label: completed pairs}`` of node ``sha``'s cells in one request so far (the S lane's resume)."""
    got = L.read(S_FAMILY_READ, root=root, family=family)
    label_of = {v: k for k, v in members.items()}
    cells = dict(L.looks(got)).get(request, [])
    return {label_of[c.opponent]: c.n_pairs for c in cells if c.player == sha and c.opponent in label_of}


def s_group(root: Path, family: str, requests: Mapping[str, str], shas: Mapping[str, str],
            members: Mapping[str, str], *, panel: Optional[str] = None, want_regime: Optional[str] = None,
            refpop_content_sha256: Optional[str] = None, min_pairs: int = 500) -> Dict[str, Any]:
    """One S group from its eval ledger: ``requests`` = ``{label: request id}`` (a request may hold several
    nodes), ``shas`` = ``{label: sha256}``. The regime and digest are the family READ's own (module docstring)."""
    why: List[str] = []
    try:
        got = L.read(S_FAMILY_READ, root=root, family=family)
    except L.ReaderDeclError as exc:
        raise AssembleError(f"family {family} at {root}: {exc}") from None
    except L.MixedRegimeError as exc:
        return {"family": family, "regime_id": None, "ledger_digest": None, "panel": panel, "team_set": None,
                "rows": 0, "nodes": {lab: {"sha256": sha, "s_pp": None, "se_pp": None,
                                           "inconclusive": [f"mixed regimes: {exc}"]} for lab, sha in shas.items()},
                "inconclusive": [f"family {family}: mixed regimes ({exc})"], "refpop_content_sha256":
                    refpop_content_sha256}
    team_sets = sorted({str(r["regime"].get("team_set")) for r in got.rows})
    want_ts = None
    if panel is not None:
        from main.h2h import panel as PN

        want_ts = PN.panel_team_set_id(panel)
        off = [t for t in team_sets if t != want_ts]
        if off:
            why.append(f"rows on team set(s) {off}, not panel {panel}'s {want_ts}")
    if not got.rows:
        why.append(f"family {family}: no row")
    if got.rows and got.regime_id is None:
        why.append(f"family {family}: the read carries no regime id")
    if want_regime is not None and got.regime_id is not None and got.regime_id != want_regime:
        why.append(f"regime {got.regime_id} is not the registered {want_regime}")
    looks = dict(L.looks(got))
    for req in sorted(set(requests.values())):
        if req not in looks:
            why.append(f"request {req} of family {family}: no row")
    nodes = {lab: s_node(looks.get(requests[lab], []), sha, members, min_pairs=min_pairs) for lab, sha in shas.items()}
    return {"family": family, "regime_id": got.regime_id, "ledger_digest": got.digest, "panel": panel,
            "team_set": want_ts if panel is not None else (team_sets[0] if len(team_sets) == 1 else team_sets),
            "rows": len(got.rows), "nodes": nodes, "inconclusive": why,
            "refpop_content_sha256": refpop_content_sha256}


# ------------------------------------------------------------------------------------------------ the read
def assemble(spec: Any, state: Path) -> Dict[str, Any]:
    """Every cell group of ``spec`` from its state (``main.reads.layout``), the models as resolved."""
    from main.reads import layout as LY
    from main.reads.models import spec_path

    mf = LY.models_file(state)
    models = json.loads(mf.read_text()) if mf.exists() else {}
    out: Dict[str, Any] = {"schema": CELLS_SCHEMA, "read": spec.read, "spec_sha256": spec.sha256,
                           "models": {lab: dict(models[lab], arm=m.arm, seed=m.seed) if lab in models
                                      else {"arm": m.arm, "seed": m.seed, "unresolved": True}
                                      for lab, m in spec.models.items()},
                           "anchor": {}, "s": {}}
    shas = {lab: e["sha256"] for lab, e in models.items()}
    from main.reads import imports as IM
    from main.reads.spec import imported

    for g, a in spec.anchor.items():
        rows: List[Dict[str, Any]] = []
        for lab in a.models:
            st = LY.anchor_state(state, g, lab)
            if lab not in imported(a) and (st / "plan.json").exists():
                rows += anchor_units(st)
        meta = None
        if a.imports is not None:                         # the earlier read's units: sha- and regime-checked
            got, meta = IM.anchor_rows(spec, g, shas, this_state=state)
            rows += got
        grp = anchor_group(rows, labels=a.models, teamsets=a.teamsets, unit_seeds=a.unit_seeds, shas=shas)
        missing = [lab for lab in a.models if lab not in shas]
        if missing:
            grp["inconclusive"] = sorted(set(grp["inconclusive"]) | {f"model(s) {missing} not resolved"})
        if meta is not None:
            grp["imports"] = meta
        out["anchor"][g] = grp
    for g, x in spec.s.items():
        mem = members_of(spec_path(x.refpop), want_content=x.refpop_content_sha256)
        have = {lab: shas[lab] for lab in x.models if lab in shas and lab not in imported(x)}
        grp = None
        if have or x.imports is None:
            grp = s_group(LY.s_ledger(state, g), x.family, {lab: x.request(lab) for lab in have}, have, mem,
                          panel=x.panel, want_regime=x.regime_id, refpop_content_sha256=x.refpop_content_sha256,
                          min_pairs=x.pairs)
        if x.imports is not None:
            ig, meta = IM.s_group(spec, g, shas, mem, this_state=state)
            if grp is None:
                grp = dict(ig)                                # every node imported: the earlier ledger's read
            else:
                grp["nodes"] = dict(grp["nodes"], **ig["nodes"])
                grp["inconclusive"] = list(grp["inconclusive"]) + [f"imported: {w}" for w in ig["inconclusive"]]
                if ig.get("regime_id") != grp.get("regime_id"):
                    grp["inconclusive"].append(f"imported regime {ig.get('regime_id')} is not this ledger's "
                                               f"{grp.get('regime_id')}")
            grp["imports"] = meta
        missing = [lab for lab in x.models if lab not in shas]
        if missing:
            grp["inconclusive"] = list(grp["inconclusive"]) + [f"model(s) {missing} not resolved"]
        out["s"][g] = grp
    return out


def group_reasons(lane: str, g: str, grp: Mapping[str, Any]) -> List[str]:
    """Every reason ONE assembled group is not complete and regime-stamped (empty = it is)."""
    why = [f"{lane}.{g}: {w}" for w in grp.get("inconclusive") or ()]
    if not grp.get("regime_id"):
        why.append(f"{lane}.{g}: no regime id")
    if lane == "anchor" and grp.get("unplayed"):
        why.append(f"anchor.{g}: {grp['unplayed']} unit(s) unplayed")
    if lane == "s":
        for lab, n in (grp.get("nodes") or {}).items():
            if n.get("s_pp") is None or n.get("se_pp") is None or not math.isfinite(float(n["s_pp"])):
                why.append(f"s.{g}.{lab}: no S ({'; '.join(n.get('inconclusive') or ['incomplete'])})")
    return why


def conclusive(cells: Mapping[str, Any], *, reported: Sequence[Tuple[str, str]] = ()) -> List[str]:
    """Every reason the assembled cells are not a complete, regime-stamped read (empty = record-ready). The groups in
    ``reported`` (``(lane, group)``: the rule's REPORTED-only groups, which no verdict reads) are left out: they are
    :func:`reported_incomplete`'s, and never block the record of the decided primary (N1-P FINDING 7)."""
    skip = set(reported)
    why: List[str] = []
    for lane in ("anchor", "s"):
        for g, grp in (cells.get(lane) or {}).items():
            if (lane, g) not in skip:
                why += group_reasons(lane, g, grp)
    return why


def reported_incomplete(cells: Mapping[str, Any], reported: Sequence[Tuple[str, str]]) -> Dict[str, List[str]]:
    """``{"<lane>.<group>": [reasons]}`` of each REPORTED-only group that is not complete (recorded INCOMPLETE)."""
    out: Dict[str, List[str]] = {}
    for lane, g in reported:
        grp = (cells.get(lane) or {}).get(g)
        why = [f"{lane}.{g}: not assembled"] if grp is None else group_reasons(lane, g, grp)
        if why:
            out[f"{lane}.{g}"] = why
    return out
