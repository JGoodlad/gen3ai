"""INFORMATION PERSPECTIVE — who could know each board fact (`designs/prober/battle_viewer_ux_2026-10-09.md` §4).

Three perspectives, defined once:

* ``public`` (the SPECTATOR): what anyone watching saw — mons that appeared, HP as %, the moves they
  used, the items / abilities a protocol line announced, status, boosts, hazards, weather.
* ``model`` (WHAT THE MODEL SAW, the default): ``public`` + OUR OWN team's private facts — all six of
  our mons from turn 0, our exact HP, our full movesets, items and abilities.
* ``truth`` (``model`` + GROUND TRUTH): also the opponent's hidden facts from the battle's
  reconstruction record — their unrevealed mons, moves, items and abilities — each one MARKED.

Every board fact is a FIELD ``{"v": value, "vis": "public" | "ours" | "hidden"}``: ``public`` = a
protocol line announced it; ``ours`` = known to our agent only; ``hidden`` = unknown to our agent at
that point (ground truth). `shown(vis, perspective)` is the ONE rule a renderer applies — the web
layer's `fv` macro calls nothing else — so two panels cannot disagree about who knows what.

Pure: the fold's board (`turn_events.TurnFold.board`) + both teams' `team_details()` in, dicts out.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

PUBLIC, OURS, HIDDEN = "public", "ours", "hidden"

#: The perspectives, as the URL names them, and which visibilities each shows.
PERSPECTIVES = ("model", "truth", "public")
DEFAULT_PERSPECTIVE = "model"
_SHOWS = {"public": frozenset({PUBLIC}), "model": frozenset({PUBLIC, OURS}),
          "truth": frozenset({PUBLIC, OURS, HIDDEN})}

#: What each perspective is called on the page, and the one line that explains it.
PERSPECTIVE_WORDS = {
    "model": ("What the model saw", "public facts + our own team's private ones (exact HP, full sets) — "
                                    "exactly what the model observed"),
    "truth": ("+ Truth", "also the opponent's hidden facts from the battle record, each marked ◇ — "
                         "never seen by the model at this point"),
    "public": ("Spectator", "only what anyone watching the battle saw"),
}


def shown(vis: Optional[str], perspective: str) -> bool:
    """Whether a field with visibility ``vis`` renders under ``perspective`` (an unknown perspective
    reads as the default)."""
    return (vis or PUBLIC) in _SHOWS.get(perspective, _SHOWS[DEFAULT_PERSPECTIVE])


def normalize(perspective: Optional[str]) -> str:
    """A client's ``view`` value, as one of `PERSPECTIVES` (anything else is the default)."""
    return perspective if perspective in PERSPECTIVES else DEFAULT_PERSPECTIVE


def _f(v, vis: str) -> dict:
    return {"v": v, "vis": vis}


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _species_name(sid: str) -> str:
    try:
        from agents.gen3_data import species as sp
        return str((sp.raw().get(_norm(sid)) or {}).get("name") or sid)
    except Exception:  # noqa: BLE001 — a display name is never worth raising for
        return sid


def _move_name(mid: str) -> str:
    try:
        from agents.gen3_data import moves as mv
        return str((mv.raw().get(_norm(mid)) or {}).get("name") or mid)
    except Exception:  # noqa: BLE001
        return mid


def _item_name(iid: str) -> str:
    try:
        from agents.gen3_data import items
        it = items.get(_norm(iid))
        return it.name if it else iid
    except Exception:  # noqa: BLE001
        return iid


def _ability_name(aid: str) -> str:
    try:
        from agents.gen3_data import abilities
        ab = abilities.get(_norm(aid))
        return ab.name if ab else aid
    except Exception:  # noqa: BLE001
        return aid


def species_abilities(species: str) -> List[str]:
    """The abilities ``species`` is seen with in gen 3 OU (the Smogon ability table — the only
    ability source `data/` carries; display names; empty when unknown). One entry ⇒ anyone who sees
    the species knows its ability."""
    try:
        from agents.gen3_data import priors
        return [_ability_name(a) for a, p in (priors.ability(_norm(species)) or {}).items() if p > 0]
    except Exception:  # noqa: BLE001
        return []


def _move_match(public_name: str, detail_id: str) -> bool:
    """A move the protocol named matches a team-sheet move id: exact, or a bare "Hidden Power" (its
    type is not public) against any typed Hidden Power."""
    a, b = _norm(public_name), _norm(detail_id)
    return a == b or (a == "hiddenpower" and b.startswith("hiddenpower"))


def _moves(revealed: Sequence[str], details: Optional[dict], private_vis: str) -> List[dict]:
    """The revealed moves (public), then the team sheet's other moves at ``private_vis`` (``ours`` for
    our mon, ``hidden`` for theirs). Our own Hidden Power shows its type (we know it)."""
    out = [_f(m, PUBLIC) for m in revealed]
    for mid in (details or {}).get("moves") or []:
        hit = next((i for i, m in enumerate(revealed) if _move_match(m, mid)), None)
        if hit is not None:
            if private_vis == OURS and _norm(revealed[hit]) == "hiddenpower":
                out[hit] = _f(_move_name(mid), PUBLIC)       # our sheet names the type of our own HP
            continue
        out.append(_f(_move_name(mid), private_vis))
    return out


def _mon_view(mon: Optional[dict], details: Optional[dict], side: str) -> dict:
    """One mon's fields. ``mon`` is the fold's view (None = never appeared); ``details`` its
    team-sheet entry (None = no reconstruction record)."""
    private = OURS if side == "we" else HIDDEN
    appeared = mon is not None and mon.get("first_turn") is not None
    sid = _norm((details or {}).get("species") or (mon or {}).get("species"))
    name = (mon or {}).get("species") or _species_name(sid)
    out: dict = {"key": sid, "appeared": appeared, "active": bool((mon or {}).get("active")),
                 "fainted": bool((mon or {}).get("fainted")), "first_turn": (mon or {}).get("first_turn")}
    out["species"] = _f(name, PUBLIC if appeared else private)
    if mon is not None and mon.get("hp_pct") is not None:
        out["hp_pct"] = _f(mon["hp_pct"], PUBLIC if appeared else private)
    else:
        out["hp_pct"] = _f(100.0, private) if details is not None else None
    if side == "we" and mon is not None and mon.get("hp") is not None and mon.get("max_hp"):
        out["hp_exact"] = _f(f"{int(mon['hp'])}/{int(mon['max_hp'])}", OURS)
    else:
        out["hp_exact"] = None
    status = (mon or {}).get("status") or ""
    out["status"] = _f(status, PUBLIC if appeared else private) if status and not out["fainted"] else None
    out["moves"] = _moves(list((mon or {}).get("moves") or []), details, private)
    item_pub = (mon or {}).get("item")
    if item_pub:
        out["item"] = _f(item_pub, PUBLIC)
        out["item_gone"] = bool((mon or {}).get("item_gone"))
    elif (details or {}).get("item"):
        out["item"] = _f(_item_name(details["item"]), private)
        out["item_gone"] = False
    else:
        out["item"], out["item_gone"] = None, False
    ab_pub = (mon or {}).get("ability")
    options = species_abilities(sid)
    if ab_pub:
        out["ability"] = _f(ab_pub, PUBLIC)
    elif appeared and len(options) == 1:
        out["ability"] = dict(_f(options[0], PUBLIC), inferred=True)     # only one is possible
    elif (details or {}).get("ability"):
        out["ability"] = _f(_ability_name(details["ability"]), private)
    else:
        out["ability"] = None
    return out


def _match(details: Optional[Sequence[dict]], species: str, used: set) -> Optional[int]:
    if not details:
        return None
    key = _norm(species)
    for i, d in enumerate(details):
        if i not in used and _norm(d.get("species")) == key:
            return i
    return None


def side_view(side_board: dict, details: Optional[Sequence[dict]], side: str) -> dict:
    """One side of the perspective board. Our mons in TEAM-SHEET order (the order the model knows
    them in); theirs in the order they appeared, then (truth only) the ones never seen."""
    mons = list(side_board.get("mons") or [])
    used: set = set()
    paired = []
    for m in mons:
        j = _match(details, m.get("species"), used)
        if j is not None:
            used.add(j)
        paired.append((m, details[j] if (details and j is not None) else None, j))
    rest = [(None, d, j) for j, d in enumerate(details or []) if j not in used]
    if side == "we" and details:
        order = sorted(paired + rest, key=lambda t: (t[2] is None, t[2] if t[2] is not None else 0))
    else:
        order = paired + rest
    out_mons = [_mon_view(m, d, side) for m, d, _j in order]
    appeared = sum(1 for m in out_mons if m["appeared"])
    size = side_board.get("team_size") or (len(details) if details else None)
    return {"mons": out_mons, "conditions": dict(side_board.get("conditions") or {}),
            "boosts": dict(side_board.get("boosts") or {}), "volatiles": list(side_board.get("volatiles") or []),
            "team_size": size, "unseen": (max(0, size - appeared) if size else None)}


def perspective_board(board: Optional[dict], *, our_details: Optional[Sequence[dict]] = None,
                      opp_details: Optional[Sequence[dict]] = None) -> Optional[dict]:
    """The fold's board as per-field-visibility sides (see the module docstring)."""
    if board is None:
        return None
    return {"we": side_view(board.get("we") or {}, our_details, "we"),
            "opp": side_view(board.get("opp") or {}, opp_details, "opp"),
            "weather": board.get("weather"), "has_truth": bool(opp_details)}


def fields_of(view: dict) -> Iterable[dict]:
    """Every field in a perspective board (for guards: each must carry a ``vis``)."""
    for side in ("we", "opp"):
        for m in (view.get(side) or {}).get("mons") or []:
            for k in ("species", "hp_pct", "hp_exact", "status", "item", "ability"):
                if isinstance(m.get(k), dict):
                    yield m[k]
            yield from m.get("moves") or []


__all__ = ["DEFAULT_PERSPECTIVE", "HIDDEN", "OURS", "PERSPECTIVES", "PERSPECTIVE_WORDS", "PUBLIC",
           "fields_of", "normalize", "perspective_board", "shown", "side_view", "species_abilities"]
