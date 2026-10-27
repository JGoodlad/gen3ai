"""The anchor guards RE-DERIVED at assemble time (``designs/ops/reviews/anchor_guards_review_2026-10-27.md`` H1).

The forfeit-win guard (``main.anchors.runner.forced_end_verdict``, SOP G6) and the H21 peer-fault check
(``runner.apply_peer_fault``) run INSIDE the playing tree, so a unit a pre-guard pin plays carries neither, and a
registered read keeps playing at its pin. Both are re-derived here from what every pinned unit already writes:

* **forced end**: each game's capture (``capture/<battle_tag>.json``) holds every ``FORCELOSE <slot>`` the front end
  wrote. A FORCELOSE decides the battle against ``<slot>``, so the ONE forced end that is an outcome is OUR side's own
  turn-limit forfeit: the game's row booked a ``loss`` stamped ``hit_forfeit_limit`` at ``turns >= forfeit_turn_limit
  > 0``, our side not a Metamon peer. Every other FORCELOSE is a HIT (the peer's drop that booked our win, a tie, our
  drop before the cap). The capture carries no cause, so our own disconnect AT the cap reads as the cap, and a
  FORCELOSE written after the result was relayed (it decides nothing) reads as a hit: fail closed;
  The front end's ``ws_frontend.log`` is a second witness: every ``disconnected mid-battle`` line it holds is a hit,
  which also covers a unit played with capture off (a peer's ``/forfeit`` is logged only by a pin with the guard);
* **peer fault**: any :data:`main.anchors.session.PEER_FAULT_LINE` in a played half's peer log
  (``<half>/peer_<role>.log``, both peers of a pair cell) is a HIT.

A check whose evidence is missing (capture off, a pruned log) is ``unavailable`` and says why; the unit is then
judged as before. How a HIT acts is the spec's ``rederived_guards`` (:data:`MODES`): ``count`` (a unit with a hit
does not count) or ``report`` (counting is unchanged; ``main.reads.api.decide`` HOLDS the read until the orchestrator
releases it). A spec that does not name it defaults to ``report`` when it was registered before this check existed
(:data:`REGISTERED_BEFORE`: counting never changes under a read in flight), else ``count``.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

SCHEMA = "gen3_read_rederived_guards_v1"
COUNT, REPORT = "count", "report"
MODES = (COUNT, REPORT)
CLEAN, HIT, UNAVAILABLE = "clean", "hit", "unavailable"
RELEASE_SCHEMA = "gen3_read_rederived_release_v1"
RELEASE_NAME = "rederived_release.json"

#: sha256 of every read spec's bytes registered before the re-derivation existed: each committed version of every
#: ``read_spec*.json`` under ``designs/research_state/measurements/`` and every read state's ``spec.json`` under the
#: archive, 2026-10-27 (the in-flight reads N1-P look 1, H-CTX rc72 / am1 / am2, X59-S, D3 looks 1-2, S-U and its
#: gate test, and the belief-prior-mix reads). Such a spec defaults to ``report``: its counting rule is the one it
#: was registered under.
#: FROZEN: a spec registered later names ``rederived_guards`` or defaults to ``count``.
REGISTERED_BEFORE = frozenset({
    "eca5c8f839ae8e37e7dc2d6ea63c0cba71977bb20029dc8ead9f89470c01b2cd",   # belief_prior_mix l025
    "de5065ac2c8e4354e50c0334b8a0ec9c94972e49c48e746bebc22050868ab796",   # belief_prior_mix l100
    "b9b52d6e18457f98dd5247c1e05020fe2c16111784ff0d977b081514ec03c101",   # hctx_rc72
    "93b3fa31c50bc7e9a446452053a4d78c95e6e2d20964c9abf8c8ed533350b244",   # hctx_rc72 am1
    "84901059df89a23793e75b52fd846eb0395c49ebb2696f50274453c1b3f90666",   # hctx_rc72 am2
    "c2b115329763c0e217be30459b7cb8a14174964950f9132304d6559216cd78ad",   # su_headroom
    "d1fde5b550e74e73836c97c140933b5c7edd0f3bf4b88bbdc59ec0f25d96c957",   # su_gate (registered ef05d165c)
    "4c51064fabbd079af350b846d2a5a90ac224a9dfb02a0a9c716ffd07c200a0cb",   # x41 d3 look 1
    "2e98bf748cf7680632e9942e289d9d146b8c7d614e0c09ffed5b1ceb69ceebcc",   # x41 d3 look 2 (first)
    "f67c56d436d4abdbfb56e6afe8213a8907ce2a38e62e1e61a686c1f578e8b55c",   # x41 d3 look 2
    "cdd17ccfa402e61362bccd752870677ed7a32526b663f3662cae571a509e286e",   # x59s full
    "1375f9b12cbb20981da81891f0952001a263b57c3afffeadebf90a65a460c0b6",   # x59s full
    "f3b747f889f587cb8e0d10a41e897707615620fad0fea9c29ccd92fccd5d7c1d",   # x59s full
    "f9af50a9ba5d2bf6393c86d26cadd592c67a7c0dfb1f25f28b39c9a12648dfd5",   # x59s full
    "ea6f8d1bfef143054eb9f87600ed8298ba6eb1eb2244244646d3df5ee50ee32e",   # x59s gate
    "ca4ac84dffa8b4faf4558dd8c4491f0f420a661c39740604514f41f5e2338fbe",   # x59s gate
    "74e2a01bbb96b767996f062271c1e90087b78fcfcffe71c37f9d6803d1f5c59b",   # x59s pilot
    "dc860ae36e590ad699e487e6d58c60dee252a746df1956ee4ccc805f1e849288",   # x59s pilot
    "f21ceb632de72d5967cdfb69d013ce232f4b6b40f1ca5cb5b846987c0fb8a7f0",   # x59s restriction
    "8dd9ab1f1abbebc4ca05d7fef6732247e50b7ae13d39467b427e83cebb402237",   # x61p n1p look 1
    "1266daadeefe4142a2f2f339bbdedaf4e052c3f2ef3fb8a8b0d5550fb8729f03",   # x61p n1p look 1
    "51fc90a50897201e89216eb0dc927f995a99bb3fc4f925e9e21e81e9e58788bb",   # x61p n1p look 2
    "f28654ccf31c3c3e6a0a3c3f9d61b904bf504f59d197b67baa3f17d510c2c620",   # x61p n1p look 2
})

_NEEDLE = b'"FORCELOSE '


class GuardHoldError(RuntimeError):
    """A ``report``-mode read whose counted units the re-derived guards hit: ``decide`` is HELD for the orchestrator."""


def default_mode(spec_sha256: str) -> str:
    """``report`` for a spec registered before the re-derivation (:data:`REGISTERED_BEFORE`), else ``count``."""
    return REPORT if spec_sha256 in REGISTERED_BEFORE else COUNT


def _rows(cell: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        return [json.loads(ln) for ln in (cell / "games.jsonl").read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return None


def forced_ends(cell: Path, summary: Mapping[str, Any], rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """``{status, hits, why}`` of the forced-end re-derivation over the games ``rows`` (``games.jsonl``)."""
    if rows is None:
        return {"status": UNAVAILABLE, "hits": [], "why": f"no readable games.jsonl in {cell}"}
    c = summary.get("cell") or {}
    limit = int(c.get("forfeit_turn_limit") or 0)
    peer_side = str(c.get("our_side") or "").startswith("metamon:")
    hits: List[Dict[str, Any]] = []
    missing: List[str] = []
    for r in rows:
        tag = r.get("battle_tag")
        cap = cell / "capture" / f"{tag}.json"
        try:
            raw = cap.read_bytes() if tag else b""
        except OSError:
            raw = b""
        if not raw:
            missing.append(f"{r.get('half')}/{r.get('index')}")
            continue
        if _NEEDLE not in raw:
            continue
        try:
            cmds = json.loads(raw).get("commands") or []
        except ValueError:
            missing.append(f"{r.get('half')}/{r.get('index')} (unreadable)")
            continue
        for cmd in cmds:
            parts = str(cmd).split()
            if len(parts) < 2 or parts[0] != "FORCELOSE":
                continue
            cap_ok = (not peer_side and limit > 0 and r.get("result") == "loss"
                      and r.get("hit_forfeit_limit") is True and int(r.get("turns") or 0) >= limit)
            if not cap_ok:
                hits.append({"check": "forced_end", "source": "capture", "tag": tag, "half": r.get("half"),
                             "index": r.get("index"), "slot": parts[1], "booked": r.get("result"),
                             "turns": r.get("turns"), "hit_forfeit_limit": r.get("hit_forfeit_limit"),
                             "forfeit_turn_limit": limit})
    hits += _logged_drops(cell, rows, {(h["tag"], h["slot"]) for h in hits})
    if hits:
        return {"status": HIT, "hits": hits, "why": f"{len(missing)} game(s) without a capture" if missing else ""}
    if missing:
        return {"status": UNAVAILABLE, "hits": [], "why": f"no capture for game(s) {missing[:5]} in {cell}"}
    return {"status": CLEAN, "hits": [], "why": ""}


#: The front end's own line for a client that went away mid-battle (``ws_frontend._drop``, the same text at every pin
#: since the front end existed): ``<user> disconnected mid-battle <tag> — forfeiting <slot>``.
_DROP_RE = re.compile(r"(\S+) disconnected mid-battle (\S+) \S+ forfeiting (p[12])")


def _logged_drops(cell: Path, rows: List[Dict[str, Any]], seen: Any) -> List[Dict[str, Any]]:
    """Every mid-battle disconnect the unit's ``ws_frontend.log`` records (a second witness, and the only one for a
    unit played with capture off): a disconnect is never our cap, so each is a hit."""
    try:
        text = (cell / "ws_frontend.log").read_text(errors="replace")
    except OSError:
        return []
    by_tag = {r.get("battle_tag"): r for r in rows}
    out = []
    for m in _DROP_RE.finditer(text):
        user, tag, slot = m.groups()
        if (tag, slot) in seen:
            continue
        seen.add((tag, slot))
        r = by_tag.get(tag) or {}
        out.append({"check": "forced_end", "source": "ws_frontend.log", "tag": tag, "half": r.get("half"),
                    "index": r.get("index"), "slot": slot, "user": user, "cause": "disconnect",
                    "booked": r.get("result"), "turns": r.get("turns")})
    return out


def peer_faults(cell: Path, rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """``{status, hits, why}`` of the H21 re-derivation: every played half's peer log(s) scanned for the fault line."""
    from main.anchors.session import PeerFaultScan

    halves = sorted({str(r.get("half")) for r in rows or () if r.get("half")})
    if not halves:
        return {"status": UNAVAILABLE, "hits": [], "why": f"no played half recorded in {cell}"}
    hits: List[Dict[str, Any]] = []
    missing: List[str] = []
    for half in halves:
        logs = sorted((cell / half).glob("peer_*.log"))
        if not logs:
            missing.append(half)
        for log in logs:
            scan = PeerFaultScan(log)
            if scan.final():
                hits.append({"check": "peer_fault", "half": half, "log": str(log), "lines": scan.count,
                             "first_line": scan.first_line})
    if hits:
        return {"status": HIT, "hits": hits, "why": f"no peer log for half(s) {missing}" if missing else ""}
    if missing:
        return {"status": UNAVAILABLE, "hits": [], "why": f"no peer log for half(s) {missing} in {cell}"}
    return {"status": CLEAN, "hits": [], "why": ""}


def unit_check(cell: Path, summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Both re-derived checks of ONE played unit's cell dir: ``{schema, forced_end, peer_fault, hits, why}``."""
    rows = _rows(cell)
    fe, pf = forced_ends(cell, summary, rows), peer_faults(cell, rows)
    return {"schema": SCHEMA, "forced_end": fe["status"], "peer_fault": pf["status"], "hits": fe["hits"] + pf["hits"],
            "why": [w for w in (fe["why"], pf["why"]) if w]}


def hit_reason(check: Mapping[str, Any]) -> str:
    """The one counting reason of a unit the re-derived guards hit (``count`` mode)."""
    what = sorted({h["check"] for h in check.get("hits") or ()})
    return (f"rederived guards hit ({', '.join(what)}: {len(check.get('hits') or ())} record(s)): a non-battle "
            "termination or a peer long-tail fault re-derived from the unit's captures / peer logs (H1)")


def summarize(cells: Mapping[str, Any]) -> Dict[str, Any]:
    """The read's re-derivation as recorded: ``{schema, mode, units_checked, unavailable, hits, held}``. ``hits`` lists
    each COUNTED unit with a hit (an uncounted unit enters no cell); ``held`` is True in ``report`` mode with hits."""
    mode = str((cells.get("guards") or {}).get("rederived") or "")
    checked, unavailable, hits = 0, [], []
    for g, grp in sorted((cells.get("anchor") or {}).items()):
        for u in grp.get("units") or ():
            chk = u.get("rederived")
            if not chk:
                continue
            checked += 1
            if UNAVAILABLE in (chk.get("forced_end"), chk.get("peer_fault")):
                unavailable.append({"group": g, "unit": u["unit"], "why": list(chk.get("why") or ())})
            counted = u.get("counted") or u.get("rederived_uncounted")
            if chk.get("hits") and counted:
                hits.append({"group": g, "unit": u["unit"], "imported_from": u.get("imported_from"),
                             "hits": list(chk["hits"]), "counted": bool(u.get("counted"))})
    return {"schema": SCHEMA, "mode": mode, "units_checked": checked, "unavailable": unavailable, "hits": hits,
            "held": mode == REPORT and bool(hits)}


def hits_sha256(summary: Mapping[str, Any]) -> str:
    """The digest a release names: a release covers exactly these hits, never a later one."""
    return hashlib.sha256(json.dumps(list(summary.get("hits") or ()), sort_keys=True,
                                     default=str).encode()).hexdigest()


def read_release(state: Path) -> Optional[Dict[str, Any]]:
    p = Path(state) / RELEASE_NAME
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise GuardHoldError(f"{p}: a damaged release is never a release ({exc})") from None
    if not isinstance(doc, dict) or doc.get("schema") != RELEASE_SCHEMA:
        raise GuardHoldError(f"{p}: not a {RELEASE_SCHEMA} record")
    return doc


def check_hold(state: Path, cells: Mapping[str, Any]) -> Dict[str, Any]:
    """The re-derivation of ``cells``, refused (:class:`GuardHoldError`) while a ``report``-mode hit is not released
    by a release naming exactly these hits (:func:`release`). Returns the summary, with ``released`` when one applies."""
    summ = summarize(cells)
    if not summ["held"]:
        return summ
    rel = read_release(state)
    digest = hits_sha256(summ)
    if rel is not None and rel.get("hits_sha256") == digest:
        return dict(summ, released={k: rel.get(k) for k in ("by", "why", "at", "hits_sha256")})
    units = "; ".join(f"{h['group']}/{h['unit']}: " + ", ".join(
        f"{x['check']} {x.get('tag') or x.get('half')}" for x in h["hits"]) for h in summ["hits"])
    raise GuardHoldError(
        f"HELD FOR THE ORCHESTRATOR (FINDING): the re-derived anchor guards hit {len(summ['hits'])} COUNTED unit(s) of "
        f"a read registered before them (rederived_guards = report, so counting is unchanged and decide waits): "
        f"{units}. Read the records (cells.json units[].rederived), then rule: release with `python -m main.reads "
        f"release-guard-hold --state {state} --by WHO --why TEXT` (hits {digest[:12]})"
        + ("" if rel is None else f"; the release on file names other hits ({str(rel.get('hits_sha256'))[:12]})"))


def release(state: Path, cells: Mapping[str, Any], *, by: str, why: str) -> Dict[str, Any]:
    """The orchestrator's release of THESE hits (``<state>/rederived_release.json``); refused when nothing is held."""
    import time

    if not (by and by.strip() and why and why.strip()):
        raise GuardHoldError("a release names who (--by) and why (--why)")
    summ = summarize(cells)
    if not summ["held"]:
        raise GuardHoldError("nothing is held: the re-derived guards hit no counted unit of a report-mode read")
    doc = {"schema": RELEASE_SCHEMA, "by": by, "why": why, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "hits_sha256": hits_sha256(summ), "hits": summ["hits"]}
    p = Path(state) / RELEASE_NAME
    tmp = p.with_suffix(".part")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(p)
    return doc
