"""`/game`'s SCOUTING NOTES — the model's beliefs about each opponent mon, the way a player tracks the
opponent (`designs/prober/battle_viewer_ux_2026-10-09.md` §7; field map in
`designs/prober/battle_view_v2.md` "Scouting notes").

Pure: the battle capture (`ProbeModel.capture_battle` → `model_capture.capture`, whose belief summaries
are already bounded top-k per opponent slot) + the opponent's TRUE sets from the reconstruction record in,
JSON-ready dicts out. No torch, no IO beyond the `agents.gen3_data` name tables. Every probability is the
model's own, rounded to 3 dp for display; every RULE that turns one into a word (believed, band, changed)
is a named constant here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from main.prober.engine.readout import (_display_species, _norm, hypotheses_view, move_id, move_name,
                                        move_name_from_id)
from main.prober.engine.spread import _SPREAD_COLS, _derived_stat, _true_derived_spread

#: A move is BELIEVED (in the model's set for that mon) at presence ≥ this — the verdict's rule (§7).
BELIEVED_AT = 0.5
#: "Since the last decision" lists a move / item whose probability moved by at least this much.
DELTA_MIN = 0.10
#: The spread BAND thresholds — a DISPLAY rule on where the believed stat sits in the species' range.
BAND_LOW, BAND_HIGH = 1.0 / 3.0, 2.0 / 3.0
#: The range a stat can take at L100: IV 31 everywhere, from 0 EV under a hindering nature to 252 EV
#: under a boosting one (the same integer formula `engine.spread` checks the truth with).
RANGE_LO = (31, 0, 0.9)
RANGE_HI = (31, 252, 1.1)

_ITEMS: Optional[Dict[int, Tuple[str, str]]] = None
_NATURES: Optional[Dict[int, str]] = None


def _item_table() -> Dict[int, Tuple[str, str]]:
    """item num → (id, display name), from the facade's item table — the axis `ItemBelief`'s prior is
    built on (`belief_tables.build_item_prior`). Num 0 is ``nothing`` (no item), as the prior treats it.

    The table SHARES a num between a gen-3 item and its gen-2 namesake (Lum Berry / Miracle Berry at 157,
    Sitrus / Gold Berry, Silk Scarf / Pink Bow, …: eleven nums). The prior is built only from Smogon's ids,
    so a shared num names the id the Smogon item prior uses; otherwise the first id alphabetically."""
    global _ITEMS
    if _ITEMS is None:
        from agents.gen3_data import items, priors, species
        smogon = {iid for sid in species.base_form_ids() for iid in (priors.items(sid) or {})}
        table: Dict[int, Tuple[str, str]] = {}
        for iid, v in sorted(items.raw().items()):
            num = int(v["num"])
            if num not in table or (iid in smogon and table[num][0] not in smogon):
                table[num] = (iid, str(v.get("name") or iid))
        table.setdefault(0, ("nothing", "No item"))
        _ITEMS = table
    return _ITEMS


def _nature_table() -> Dict[int, str]:
    """nature num → display name (the nature head's axis, `belief_tables.build_nature_mult`)."""
    global _NATURES
    if _NATURES is None:
        from agents.gen3_data import natures
        _NATURES = {int(v["num"]): str(name).capitalize() for name, v in natures.raw().items()}
    return _NATURES


def item_name(num: int) -> str:
    return _item_table().get(int(num), ("", f"item #{int(num)}"))[1]


def item_id(num: int) -> str:
    return _item_table().get(int(num), ("", ""))[0]


def _item_name_from_id(iid: str) -> str:
    if not iid or iid == "nothing":
        return "No item"
    from agents.gen3_data import items
    it = items.get(iid)
    return it.name if it is not None else iid


def _hp_collapse(mid: str) -> str:
    """Any Hidden Power (bare or typed) is ONE move for the set comparison — the capture collapses the
    16 typed channels into one presence, and a set holds at most one Hidden Power."""
    return "hiddenpower" if mid.startswith("hiddenpower") else mid


def _r(x: float) -> float:
    return round(float(x), 3)


# ------------------------------------------------------------------------------- per-field pieces

def _moves(cap: Dict[str, Any], i: int, j: int) -> Tuple[Optional[List[dict]], Optional[float], Dict[str, float]]:
    """(display rows, floor, {id: unrounded p}) for opponent slot ``j``; (None, None, {}) when the move
    belief is not built."""
    if "belief_move_nums" not in cap:
        return None, None, {}
    revealed = cap.get("opp_revealed_moves")
    seen = set(revealed[i][j]) if revealed is not None else None
    hp_names = cap.get("hp_type_names") or []
    rows, raw = [], {}
    for c, (num, p) in enumerate(zip(cap["belief_move_nums"][i][j], cap["belief_move_p"][i][j])):
        mid = move_id(int(num)) or f"move{int(num)}"
        hp = None
        if mid == "hiddenpower" and "belief_move_hp_type" in cap and float(p) > 0:
            hp = [{"type": str(hp_names[int(t)]).capitalize() if int(t) < len(hp_names) else f"type #{int(t)}",
                   "p": _r(tp)}
                  for t, tp in zip(cap["belief_move_hp_type"][i][j], cap["belief_move_hp_type_p"][i][j])
                  if float(tp) > 0] or None
        raw[mid] = float(p)
        rows.append({"id": mid, "name": move_name(int(num)), "p": _r(p),
                     "seen": (mid in seen) if seen is not None else None, "hp_type": hp, "_order": c})
    rows.sort(key=lambda r: (-r["p"], r["_order"]))
    for r in rows:
        del r["_order"]
    return rows, float(cap["belief_move_floor"][i][j]), raw


def _item(cap: Dict[str, Any], i: int, j: int) -> Tuple[Optional[List[dict]], Optional[float]]:
    if "belief_item_nums" not in cap:
        return None, None
    rows = [{"id": item_id(int(n)), "name": item_name(int(n)), "p": _r(p)}
            for n, p in zip(cap["belief_item_nums"][i][j], cap["belief_item_p"][i][j])]
    rows.sort(key=lambda r: -r["p"])
    return rows, float(cap["belief_item_floor"][i][j])


def stat_range(species_id: str) -> Optional[Dict[str, List[int]]]:
    """Per stat, the [lo, hi] a mon of this species can have at L100 (`RANGE_LO` / `RANGE_HI`); None when
    the species is not in the dex."""
    from agents.gen3_data import species
    sp = species.get(species_id)
    if sp is None:
        return None
    out = {}
    for s in _SPREAD_COLS:
        base = int(sp.base_stats.get(s, 0))
        out[s] = [_derived_stat(base, *RANGE_LO), _derived_stat(base, *RANGE_HI)]
    return out


def band_of(pos: float) -> str:
    """The DISPLAY band of a position in the range: ``low`` below 1/3, ``high`` above 2/3, else ``mid``."""
    return "low" if pos < BAND_LOW else ("high" if pos > BAND_HIGH else "mid")


def _spread(cap: Dict[str, Any], i: int, j: int, sid: str) -> Optional[dict]:
    if "belief_spread" not in cap:
        return None
    vals = [float(x) for x in cap["belief_spread"][i][j]]
    stats = {s: round(v, 1) for s, v in zip(_SPREAD_COLS, vals)}
    rng = stat_range(sid)
    pos = band = None
    if rng is not None:
        pos, band = {}, {}
        for s, v in zip(_SPREAD_COLS, vals):
            lo, hi = rng[s]
            x = min(1.0, max(0.0, (v - lo) / (hi - lo))) if hi > lo else 0.5
            pos[s], band[s] = _r(x), band_of(x)
    nature = None
    if "belief_nature_nums" in cap:
        nature = sorted(({"name": _nature_table().get(int(n), f"nature #{int(n)}"), "p": _r(p)}
                         for n, p in zip(cap["belief_nature_nums"][i][j], cap["belief_nature_p"][i][j])),
                        key=lambda r: -r["p"])
    return {"stats": stats, "range": rng, "pos": pos, "band": band, "nature": nature}


def _hp_type_head(cap: Dict[str, Any], i: int, j: int) -> Optional[List[dict]]:
    if "belief_hp_type" not in cap:
        return None
    names = cap.get("hp_type_names") or []
    return sorted(({"type": str(names[int(t)]).capitalize() if int(t) < len(names) else f"type #{int(t)}",
                    "p": _r(p)} for t, p in zip(cap["belief_hp_type"][i][j], cap["belief_hp_type_p"][i][j])),
                  key=lambda r: -r["p"])


def _truth(detail: dict) -> dict:
    moves = [str(m) for m in detail.get("moves") or [] if m]
    td = _true_derived_spread(detail)
    item = str(detail.get("item") or "") or "nothing"
    return {"moves": moves, "move_names": [move_name_from_id(m) for m in moves],
            "item": item, "item_name": _item_name_from_id(item),
            "ability": str(detail.get("ability") or "") or None,
            "nature": (str(detail.get("nature") or "").capitalize() or None),
            "evs": dict(detail.get("evs") or {}),
            "stats": ({s: float(v) for s, v in zip(_SPREAD_COLS, td[0])} if td is not None else None)}


def _verdict(truth: dict, moves: Optional[List[dict]], floor: Optional[float], raw: Dict[str, float],
             items: Optional[List[dict]], spread: Optional[dict]) -> dict:
    """moves right / missed / false (a true move is RIGHT at presence ≥ `BELIEVED_AT`), the item check, the
    believed-minus-true Speed. A true move outside the captured top-k has presence ≤ the floor: MISSED
    (``p`` None) when the floor is below `BELIEVED_AT`, else UNDETERMINED (counted, never guessed)."""
    out: dict = {"moves_right": None, "moves_total": None, "moves_undetermined": None, "missed": None,
                 "false": None, "item_right": None, "item_p_true": None, "spe_err": None}
    true_ids = [_hp_collapse(m) for m in truth["moves"]]
    if moves is not None:
        right, und, missed = 0, 0, []
        for m, tid in zip(truth["moves"], true_ids):
            p = raw.get(tid)
            if p is None:
                if floor is not None and floor >= BELIEVED_AT:
                    und += 1
                else:
                    missed.append({"name": move_name_from_id(m), "p": None})
            elif p >= BELIEVED_AT:
                right += 1
            else:
                missed.append({"name": move_name_from_id(m), "p": _r(p)})
        tset = set(true_ids)
        out.update(moves_right=right, moves_total=len(true_ids), moves_undetermined=und, missed=missed,
                   false=[{"name": r["name"], "p": r["p"]} for r in moves
                          if raw.get(r["id"], 0.0) >= BELIEVED_AT and r["id"] not in tset])
    if items:
        out["item_right"] = items[0]["id"] == truth["item"]
        out["item_p_true"] = next((r["p"] for r in items if r["id"] == truth["item"]), None)
    if spread is not None and truth.get("stats"):
        out["spe_err"] = round(float(spread["stats"]["spe"]) - float(truth["stats"]["spe"]), 1)
    return out


def _delta_rows(cur: Optional[List[dict]], cur_floor: Optional[float], prev: Optional[List[dict]],
                prev_floor: Optional[float]) -> List[dict]:
    """The entries whose probability moved by ≥ `DELTA_MIN` between two views of the same mon, |Δ| desc.
    An entry listed on one side only is compared against the other side's FLOOR (an upper bound on its
    unlisted value), marked ``unlisted`` = that side, and kept only when even the bound clears the bar."""
    if not cur or not prev:
        return []
    a = {r["id"]: r for r in prev}
    b = {r["id"]: r for r in cur}
    out = []
    for mid in list(dict.fromkeys(list(b) + list(a))):
        if mid in a and mid in b:
            frm, to, unl = a[mid]["p"], b[mid]["p"], None
        elif mid in b:
            if prev_floor is None:
                continue
            frm, to, unl = _r(prev_floor), b[mid]["p"], "from"
            if to <= frm:
                continue
        else:
            if cur_floor is None:
                continue
            frm, to, unl = a[mid]["p"], _r(cur_floor), "to"
            if frm <= to:
                continue
        d = round(abs(to - frm), 3)
        if d >= DELTA_MIN:
            out.append({"name": (b.get(mid) or a.get(mid))["name"], "from": frm, "to": to, "unlisted": unl,
                        "_d": d})
    out.sort(key=lambda r: -r["_d"])
    for r in out:
        del r["_d"]
    return out


# ------------------------------------------------------------------------------- the view

def scouting_view(cap: Dict[str, Any], i: int, opp_team_details: Optional[Sequence[dict]] = None,
                  prev: Optional[dict] = None) -> dict:
    """Decision ``i``'s scouting notes.

    ``mons`` — one card per opponent OBS slot holding a REVEALED species, the active first, then slot order:

    * ``moves`` — the move posterior's top-k (sigmoid presence, sorted desc; Hidden Power ONE entry with
      its believed type split ``hp_type``; ``seen`` = the obs has revealed it). ``floor`` carries the
      largest presence NOT listed, an upper bound on every unlisted move (and ``item``'s likewise).
    * ``item`` — the item posterior's top-3 (softmax).
    * ``spread`` — the believed L100 stats; per stat its ``range`` [lo, hi] (IV 31; 0 EV × 0.9 nature →
      252 EV × 1.1), ``pos`` = (believed − lo)/(hi − lo) clamped to [0, 1], and ``band`` — a DISPLAY rule:
      ``low`` when pos < 1/3, ``high`` when pos > 2/3, else ``mid``; ``nature`` = the nature head's top-3.
    * ``hp_type_head`` — the HP-type HEAD's own posterior (top-3), before the composition's certain
      narrowing; ``moves``' Hidden Power split is the narrowed one the op consumes.
    * ``truth`` / ``verdict`` — the reconstruction's true set and the check against it (a true move is
      right at presence ≥ 0.5; ``false`` = a believed move not in the set; ``item_right`` = the top item
      is the true one); None without ``opp_team_details``. Matched by exact normalized species id.
    * ``delta`` — vs ``prev`` (the previous decision's view), the SAME species' moves / items whose
      probability moved by ≥ 0.10, |Δ| desc; [] when none or no prev.

    ``unseen`` — the hidden slots: their count, the hypothesis tokens' guesses (`hypotheses_view`, sorted
    by presence; ``on_team`` against the truth), OTHER_species' "at least one" mass, and the true mons not
    yet revealed (None without the truth).

    A head the model does not build gives None for its fields, never zeros."""
    teams = (cap.get("teams") or [{}] * (i + 1))[i]
    opp = list(teams.get("opp") or [""] * 6)
    active = int(cap["opp_active"][i]) if "opp_active" in cap else -1
    details = list(opp_team_details) if opp_team_details is not None else None
    by_species = {_norm(d.get("species", "")): d for d in details or () if d.get("species")}
    prev_mons = {m["id"]: m for m in (prev or {}).get("mons") or []}
    order = sorted((j for j in range(len(opp)) if opp[j]), key=lambda j: (j != active, j))
    mons = []
    for j in order:
        sid = _norm(opp[j])
        moves, mfloor, raw = _moves(cap, i, j)
        items, ifloor = _item(cap, i, j)
        spread = _spread(cap, i, j, sid)
        d = by_species.get(sid) if details is not None else None
        truth = _truth(d) if d is not None else None
        pm = prev_mons.get(sid)
        pf = (pm or {}).get("floor") or {}
        mons.append({
            "slot": j, "species": _display_species(sid), "id": sid, "active": j == active,
            "moves": moves, "item": items, "spread": spread, "hp_type_head": _hp_type_head(cap, i, j),
            "floor": {"moves": _r(mfloor) if mfloor is not None else None,
                      "item": _r(ifloor) if ifloor is not None else None},
            "truth": truth,
            "verdict": _verdict(truth, moves, mfloor, raw, items, spread) if truth is not None else None,
            "delta": ({"moves": _delta_rows(moves, mfloor, pm.get("moves"), pf.get("moves")),
                       "item": _delta_rows(items, ifloor, pm.get("item"), pf.get("item"))}
                      if pm is not None else {"moves": [], "item": []})})
    revealed = {_norm(s) for s in opp if s}
    hv = hypotheses_view(cap, i, [d.get("species", "") for d in details or ()]) if "slot_species" in cap else None
    guesses = None
    if hv is not None:
        guesses = sorted(({"species": s["species"], "id": s["id"],
                           "p": _r(s["presence"]) if s["presence"] is not None else None, "on_team": s["on_team"]}
                          for s in hv["slots"] if not s["revealed"]),
                         key=lambda g: -(g["p"] or 0.0))
    unseen = {"n": sum(1 for s in opp if not s), "guesses": guesses,
              "other_any": _r(hv["other_any"]) if hv is not None else None,
              "truth": ([_display_species(_norm(d.get("species", ""))) for d in details
                         if _norm(d.get("species", "")) not in revealed] if details is not None else None)}
    return {"mons": mons, "unseen": unseen}


#: How many public reveals a card's "changed since the last decision" line quotes as its evidence.
REVEALS_SHOWN = 3


def reveal_notes(turns: Sequence[dict], t_from: int, t_to: int, species: str) -> List[str]:
    """The PUBLIC events that touched ``species`` (theirs) in game turns ``t_from`` ≤ turn < ``t_to`` — what
    happened between two decisions, quoted beside a belief that moved ("after: it used Spikes"). It is the
    evidence a reader checks the update against, not a claim that it CAUSED the update. Oldest first, at
    most `REVEALS_SHOWN`. ``turns`` is `battle_story`'s."""
    key = _norm(species)
    out: List[str] = []
    for t in turns or ():
        if not (t_from <= int(t.get("turn") or 0) < t_to):
            continue
        for e in t.get("events") or ():
            if e.get("side") != "opp" or _norm(e.get("species") or "") != key:
                continue
            k = e.get("kind")
            note = None
            if k == "move" and not e.get("via"):
                note = f"it used {e.get('move')}"
            elif k in ("switch", "drag"):
                note = "it came in"
            elif k == "item" and e.get("item"):
                note = f"its {e['item']} showed"
            elif k == "ability" and e.get("ability"):
                note = f"its {e['ability']} showed"
            elif k in ("heal", "damage") and e.get("source") and e["source"] not in (
                    "Sandstorm", "Hail", "Spikes", "poison", "burn", "recoil", "Leech Seed"):
                note = f"its {e['source']} showed"
            if note and note not in out:
                out.append(note)
    return out[:REVEALS_SHOWN]


__all__ = ["BAND_HIGH", "BAND_LOW", "BELIEVED_AT", "DELTA_MIN", "RANGE_HI", "RANGE_LO", "REVEALS_SHOWN", "band_of",
           "item_id", "item_name", "reveal_notes", "scouting_view", "stat_range"]
