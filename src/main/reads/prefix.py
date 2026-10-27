"""G-PREFIX: the paired-prefix tooth of a SEARCHED read (S-U, ``designs/endstate/design_search_u_headroom.md`` §2).

A searched model (``models.<label>.load.search``, ``<preset>@<turn>``) and its RAW TWIN (``main.reads.spec.search_twins``:
the same bytes, no search) play every DEAL of a ``per_game`` anchor group: the same seed's game ``g`` on the same team
set, one ONE-GAME unit each (``main.x35.kakuna.plan_units(per_game=True)``). Before the search's first turn both arms are
the checkpoint greedy against the same greedy peer, on the same battle seed and the same two teams, so each pair's
protocol must be BYTE-IDENTICAL through the start of that turn (``|turn|<from_turn>``, inclusive), on both sides. A
mismatch is a nondeterminism that would corrupt the pairing: the lane is HELD for a root cause (``main.reads.runner``),
and while any pair diverges the group is not conclusive, so no record is written (the read is VOID unless it is
resolved).

What is compared, from each unit's front-end capture (``<cell>/capture/<battle>.json``, written by ``main.anchors
--capture-dir``, which a per-game unit always passes): the battle's ``START`` document (format, both names and packed
teams, the seed, the core-obs side) and, per side, every protocol line the battle relayed, the wall-clock ``|t:|``
lines removed (the one non-deterministic line of the sim), up to and including that side's first ``|turn|<from_turn>``
line. A game that ends before that turn is compared whole. The commands are not compared: two sides' choices of one
update reach the child in either order, and the protocol they produce is the deterministic record.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

PREFIX_SCHEMA = "gen3_read_prefix_v1"


class PrefixError(RuntimeError):
    """A unit's capture cannot be read (none, or more than one battle): never a match."""


def capture_of(cell: Path) -> Dict[str, Any]:
    """The ONE battle capture of a per-game unit's cell dir."""
    caps = sorted((cell / "capture").glob("*.json"))
    if len(caps) != 1:
        raise PrefixError(f"{cell}: {len(caps)} battle captures (a per-game unit plays exactly one)")
    try:
        return dict(json.loads(caps[0].read_text()))
    except (OSError, ValueError) as exc:
        raise PrefixError(f"{caps[0]}: unreadable ({exc})") from None


def _start(cap: Mapping[str, Any]) -> Dict[str, Any]:
    cmds = cap.get("commands") or []
    if not cmds or not str(cmds[0]).startswith("START "):
        raise PrefixError("the capture's first command is not START")
    return dict(json.loads(str(cmds[0])[len("START "):]))


def side_prefix(cap: Mapping[str, Any], slot: str, from_turn: int) -> Tuple[List[str], bool]:
    """``(lines, reached)``: ``slot``'s relayed protocol lines without ``|t:|``, through its first ``|turn|<from_turn>``
    line inclusive; ``reached`` = that line was seen (else the game ended first and every line is returned)."""
    out: List[str] = []
    stop = f"|turn|{int(from_turn)}"
    for s, chunk in cap.get("chunks") or []:
        if s != slot:
            continue
        for ln in str(chunk).split("\n"):
            if ln.startswith("|t:|"):
                continue
            out.append(ln)
            if ln == stop:
                return out, True
    return out, False


def compare(raw: Mapping[str, Any], searched: Mapping[str, Any], from_turn: int) -> Dict[str, Any]:
    """One deal's pair: ``{ok, why, reached, lines}`` (``why`` names the first difference)."""
    a, b = _start(raw), _start(searched)
    if a != b:
        keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        return {"ok": False, "why": f"the START documents differ in {keys}", "reached": None, "lines": {}}
    out: Dict[str, Any] = {"ok": True, "why": None, "reached": {}, "lines": {}}
    for slot in ("p1", "p2"):
        la, ra = side_prefix(raw, slot, from_turn)
        lb, rb = side_prefix(searched, slot, from_turn)
        out["reached"][slot] = bool(ra and rb)
        out["lines"][slot] = len(la)
        if la != lb or ra != rb:
            i = next((k for k, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
            out.update(ok=False, why=(f"{slot}'s protocol differs at line {i} before turn {from_turn}: raw "
                                      f"{(la[i] if i < len(la) else '<end>')[:160]!r} vs searched "
                                      f"{(lb[i] if i < len(lb) else '<end>')[:160]!r}"))
            return out
    return out


def deal_pairs(spec: Any, state: Path, group: str) -> List[Dict[str, Any]]:
    """Every (searched unit, raw twin unit) pair of ``group`` as planned: ``{deal, searched, raw, from_turn, cells}``."""
    from main.anchors.search_presets import parse
    from main.reads import layout as LY
    from main.reads.spec import search_twins
    from main.x35 import kakuna as KA

    a = spec.anchor[group]
    out: List[Dict[str, Any]] = []
    for lab, twin in search_twins(spec.models, a.models, f"anchor.{group}").items():
        st_s, st_r = LY.anchor_state(state, group, lab), LY.anchor_state(state, group, twin)
        if not (st_s / "plan.json").exists() or not (st_r / "plan.json").exists():
            continue
        raw = {(u.teamset, u.unit_seed, u.game): u for u in KA.load_plan(st_r)}
        _p, from_turn = parse(spec.models[lab].load["search"])
        for u in KA.load_plan(st_s):
            r = raw.get((u.teamset, u.unit_seed, u.game))
            if r is None:
                raise PrefixError(f"searched unit {u.name} has no raw twin unit in {twin}'s plan")
            out.append({"deal": f"{u.teamset}:{u.unit_seed}:{u.game}", "label": lab, "twin": twin,
                        "searched": (st_s, u), "raw": (st_r, r), "from_turn": from_turn})
    return out


def check_pair(pair: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The pair's comparison, or None while either unit is not counted yet."""
    from main.x35 import kakuna as KA

    (st_s, us), (st_r, ur) = pair["searched"], pair["raw"]
    if not (KA.is_done(st_s, us) and KA.is_done(st_r, ur)):
        return None
    try:
        res = compare(capture_of(st_r / "cells" / ur.name), capture_of(st_s / "cells" / us.name), pair["from_turn"])
    except PrefixError as exc:
        res = {"ok": False, "why": f"unreadable: {exc}", "reached": None, "lines": {}}
    return dict(res, deal=pair["deal"], searched=us.name, raw=ur.name)


def check_group(spec: Any, state: Path, group: str) -> Dict[str, Any]:
    """``{schema, pairs, checked, pending, mismatches}`` over every planned pair of ``group``."""
    pairs = deal_pairs(spec, state, group)
    checked, mism, pending = 0, [], 0
    for p in pairs:
        r = check_pair(p)
        if r is None:
            pending += 1
            continue
        checked += 1
        if not r["ok"]:
            mism.append({k: r[k] for k in ("deal", "searched", "raw", "why")})
    return {"schema": PREFIX_SCHEMA, "pairs": len(pairs), "checked": checked, "pending": pending, "mismatches": mism}


def pair_of_unit(spec: Any, state: Path, kakuna_state: Path, unit_name: str) -> Optional[Dict[str, Any]]:
    """The planned pair a just-ended unit (searched or raw) belongs to, or None (a group with no search)."""
    for g, a in spec.anchor.items():
        if not a.per_game:
            continue
        for p in deal_pairs(spec, state, g):
            for side in ("searched", "raw"):
                st, u = p[side]
                if Path(st) == Path(kakuna_state) and u.name == unit_name:
                    return dict(p, group=g)
    return None
