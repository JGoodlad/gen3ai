"""`/game`'s MODEL panels, as plain JSON-ready dicts, from one battle capture
(`ProbeModel.capture_battle` → `model_capture.capture`) plus the battle's summary
(`designs/prober/battle_view_v2.md` §2–§4).

Pure: numpy arrays + dicts in, dicts out. No torch, no IO beyond the `agents.gen3_data` name tables
(the same facade `beliefs.py` reads). Every label is in plain words; the model's own term rides
beside it as `term` so a surface can show both.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------------------- name tables

_SPECIES: Optional[Dict[int, Tuple[str, str]]] = None
_MOVES: Optional[Dict[int, Tuple[str, str]]] = None


def _species_table() -> Dict[int, Tuple[str, str]]:
    """num → (id, display name), BASE forms only (a forme shares its base's num — root CLAUDE.md)."""
    global _SPECIES
    if _SPECIES is None:
        from agents.gen3_data import species as sp
        raw = sp.raw()
        _SPECIES = {int(raw[i]["num"]): (i, str(raw[i].get("name") or i)) for i in sp.base_form_ids()
                    if raw.get(i, {}).get("num")}
    return _SPECIES


def _move_table() -> Dict[int, Tuple[str, str]]:
    """num → (id, display name). Typed Hidden Power is its own num (355–370, `hiddenpowerice` /
    "Hidden Power Ice"); the bare 237 is the untyped presence token."""
    global _MOVES
    if _MOVES is None:
        from agents.gen3_data import moves as mv
        raw = mv.raw()
        _MOVES = {}
        for mid, v in raw.items():
            if not v.get("num"):
                continue
            num = int(v["num"])
            name = str(v.get("name") or mid)
            if mid.startswith("hiddenpower") and mid != "hiddenpower":
                name = "Hidden Power " + mid[len("hiddenpower"):].capitalize()
            if mid == "hiddenpower":
                _MOVES.setdefault(num, (mid, "Hidden Power"))
            else:
                _MOVES[num] = (mid, name)
    return _MOVES


def species_name(num: int) -> str:
    return _species_table().get(int(num), ("", f"species #{int(num)}"))[1]


def species_id(num: int) -> str:
    return _species_table().get(int(num), ("", ""))[0]


def move_name(num: int) -> str:
    return _move_table().get(int(num), ("", f"move #{int(num)}"))[1]


def move_id(num: int) -> str:
    return _move_table().get(int(num), ("", ""))[0]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _display_species(sid: str) -> str:
    if not sid:
        return ""
    try:
        from agents.gen3_data import species as sp
        return str((sp.raw().get(_norm(sid)) or {}).get("name") or sid)
    except Exception:  # noqa: BLE001 — a display name is never worth raising for
        return sid


# ------------------------------------------------------------------------------- the label

def opp_actual_action(inv: dict) -> Optional[Dict[str, str]]:
    """What the OPPONENT actually chose in the window this decision opened: ``{"kind": "move", "id"}``
    or ``{"kind": "switch", "id": <species id>}``; None when they made no CHOICE (a post-faint
    replacement, nothing recorded). A phazing move is a move (the drag is its consequence)."""
    action = str(((inv.get("outcome") or {}).get("opp") or {}).get("action") or "").strip()
    if not action or action in ("none", "unknown"):
        return None
    if action.startswith("switched_to:"):
        return {"kind": "switch", "id": _norm(action.split(":", 1)[1])}
    head = action.split("→")[0].strip()
    if not head or head.endswith("_sent_in"):
        return None
    return {"kind": "move", "id": _norm(head)}


def _move_matches(seat_move_id: str, actual_id: str) -> bool:
    a, b = _norm(seat_move_id), _norm(actual_id)
    if not a or not b:
        return False
    if a == b:
        return True
    # a bare recorded Hidden Power matches any HP seat; a typed one only its own type (the bare 237 seat
    # is a REVEALED Hidden Power whose type is unknown, so it matches any HP the opponent used)
    return a.startswith("hiddenpower") and b.startswith("hiddenpower") and (b == "hiddenpower" or a == "hiddenpower")


# ------------------------------------------------------------------------------- intent

def intent_view(cap: Dict[str, Any], i: int, actual: Optional[Dict[str, str]],
                opp_team_ids: Sequence[str] = ()) -> Optional[dict]:
    """Decision ``i``'s flat opponent pointer: every LIVE candidate as a bar (plain-word label + the
    model term), which one the opponent actually took (the label, mapped by the training rule — a move
    beyond the seats is OTHER_move, a switch-in not among the slots' species is OTHER_species)."""
    if "intent_p" not in cap:
        return None
    k = int(cap["intent_k"])
    p = cap["intent_p"][i]
    live = cap["intent_live"][i]
    cand = cap["intent_cand"][i]
    seat_rev = cap.get("seat_revealed")
    slot_hyp = cap.get("slot_is_hyp")
    slot_pi = cap.get("slot_pi")
    team = {_norm(t) for t in opp_team_ids if t}
    rows: List[dict] = []
    F = int(p.shape[0])
    for c in range(F):
        if c < k:
            mid, mname = move_id(int(cand[c])), move_name(int(cand[c]))
            seen = bool(seat_rev[i][c]) if seat_rev is not None and c < seat_rev.shape[1] else False
            rows.append({"col": c, "kind": "move", "id": mid, "label": f"{mname} ({'seen' if seen else 'guess'})",
                         "term": f"move seat {c + 1}", "seen": seen})
        elif c == k:
            rows.append({"col": c, "kind": "other_move", "id": "", "label": "any other move",
                         "term": "OTHER_move", "seen": False})
        elif c < F - 1:
            j = c - k - 1
            sid, sname = species_id(int(cand[c])), species_name(int(cand[c]))
            hyp = bool(slot_hyp[i][j]) if slot_hyp is not None else False
            pi = float(slot_pi[i][j]) if (slot_pi is not None and hyp) else None
            label = f"switch → {sname}" + (f" (guess, {pi:.0%} present)" if hyp and pi is not None else
                                           (" (guess)" if hyp else " (seen)"))
            rows.append({"col": c, "kind": "switch", "id": sid, "label": label, "term": f"switch to slot {j + 1}",
                         "seen": not hyp, "slot": j, "presence": pi,
                         "on_team": (sid in team) if (team and hyp and sid) else None})
        else:
            rows.append({"col": c, "kind": "other_species", "id": "", "label": "switch → a mon not on our list",
                         "term": "OTHER_species", "seen": False})
        rows[-1]["p"] = float(p[c])
        rows[-1]["live"] = bool(live[c])
    actual_col = None
    if actual is not None:
        if actual["kind"] == "move":
            hits = [r["col"] for r in rows if r["kind"] == "move" and r["live"] and _move_matches(r["id"], actual["id"])]
            actual_col = hits[0] if hits else k
        else:
            hits = [r["col"] for r in rows if r["kind"] == "switch" and r["live"] and _norm(r["id"]) == actual["id"]]
            actual_col = hits[0] if hits else F - 1
        if not rows[actual_col]["live"]:
            actual_col = None
    for r in rows:
        r["actual"] = r["col"] == actual_col
    live_rows = [r for r in rows if r["live"]]
    top = max(live_rows, key=lambda r: r["p"]) if live_rows else None
    if actual is None:
        actual_text = None
    elif actual["kind"] == "move":
        actual_text = f"they used {move_name_from_id(actual['id'])}"
    else:
        actual_text = f"they switched to {_display_species(actual['id'])}"
    return {"candidates": live_rows, "actual_col": actual_col, "actual": actual, "actual_text": actual_text,
            "p_actual": (rows[actual_col]["p"] if actual_col is not None else None),
            "top_col": top["col"] if top else None, "top_hit": (top is not None and actual_col == top["col"])}


def action_display(label: str) -> str:
    """A recorded action label in words: ``switch:skarmory`` → ``switch → Skarmory``,
    ``earthquake`` → ``Earthquake``. A placeholder (empty) stays empty."""
    s = str(label or "")
    if not s:
        return ""
    if s.startswith("switch:"):
        return "switch → " + _display_species(s.split(":", 1)[1])
    return move_name_from_id(s)


def move_name_from_id(mid: str) -> str:
    try:
        from agents.gen3_data import moves as mv
        return str((mv.raw().get(_norm(mid)) or {}).get("name") or mid)
    except Exception:  # noqa: BLE001
        return mid


#: Reliability bins for the battle's top-pick calibration (edges, probability of the top pick).
CALIBRATION_BINS = (0.0, 0.25, 0.5, 0.75, 1.0000001)


def intent_calibration(views: Sequence[Optional[dict]]) -> Optional[dict]:
    """α across the battle, over the decisions WITH a label: mean P(actual), top-1 hit rate, mean log
    loss (nats; a P of 0 is floored at 1e-6 and counted), and the top pick's reliability by bin."""
    rows = [v for v in views if v and v.get("actual_col") is not None]
    if not rows:
        return None
    ps = [max(float(v["p_actual"]), 1e-6) for v in rows]
    bins = []
    for lo, hi in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
        sel = [v for v in rows if v["top_col"] is not None
               and lo <= next(c["p"] for c in v["candidates"] if c["col"] == v["top_col"]) < hi]
        if sel:
            bins.append({"lo": lo, "hi": min(hi, 1.0), "n": len(sel),
                         "mean_p": float(np.mean([next(c["p"] for c in v["candidates"] if c["col"] == v["top_col"])
                                                  for v in sel])),
                         "hit_rate": float(np.mean([1.0 if v["top_hit"] else 0.0 for v in sel]))})
    return {"n": len(rows), "mean_p_actual": float(np.mean(ps)),
            "top1": float(np.mean([1.0 if v["top_hit"] else 0.0 for v in rows])),
            "log_loss": float(np.mean([-math.log(x) for x in ps])),
            "n_floored": int(sum(1 for v in rows if float(v["p_actual"]) < 1e-6)),
            "bins": bins,
            "note": "one battle — descriptive only, not a calibration estimate"}


# ------------------------------------------------------------------------------- hypotheses

def hypotheses_view(cap: Dict[str, Any], i: int, opp_team_ids: Sequence[str] = ()) -> Optional[dict]:
    """Per opponent slot: the REVEALED mon or the hypothesis a hidden slot holds (+ presence π, and
    whether it is on their true team when the reconstruction says), OTHER_species' mass + its likeliest
    members, the active's move group."""
    if "slot_species" not in cap:
        return None
    team = {_norm(t) for t in opp_team_ids if t}
    obs_opp = (cap.get("teams") or [{}] * (i + 1))[i].get("opp", [""] * 6)
    slots = []
    for j in range(6):
        hyp = bool(cap["slot_is_hyp"][i][j])
        if hyp:
            num = int(cap["slot_species"][i][j])
            sid = species_id(num)
            slots.append({"slot": j, "revealed": False, "species": species_name(num), "id": sid,
                          "presence": float(cap["slot_pi"][i][j]) if "slot_pi" in cap else None,
                          "on_team": (sid in team) if (team and sid) else None})
        else:
            sid = _norm(obs_opp[j]) if j < len(obs_opp) else ""
            slots.append({"slot": j, "revealed": True, "species": _display_species(sid) if sid else "—",
                          "id": sid, "presence": 1.0, "on_team": True if sid else None})
    tail = [{"species": species_name(int(n)), "id": species_id(int(n)), "p": float(pp)}
            for n, pp in zip(cap["tail_idx"][i], cap["tail_p"][i]) if pp > 0]
    moves = None
    if "seat_nums" in cap:
        moves = [{"move": move_name(int(n)), "seen": bool(r), "presence": float(pp)}
                 for n, r, pp, lv in zip(cap["seat_nums"][i], cap["seat_revealed"][i], cap["seat_pi"][i],
                                         cap["seat_live"][i]) if lv]
    return {"slots": slots, "other_mass": float(cap["other_mass"][i]), "other_live": bool(cap["other_live"][i]),
            "other_any": float(cap["other_any"][i]), "other_top": tail, "active_moves": moves,
            "other_move_mass": float(cap["move_other_mass"][i]) if "move_other_mass" in cap else None}


def belief_evolution(cap: Dict[str, Any], *, max_species: int = 8) -> Optional[dict]:
    """The presence π of every species that was ever among the top hypotheses, decision by decision
    (revealed ⇒ π = 1). Kept to the ``max_species`` with the largest peak π so the chart stays legible."""
    if "species_idx" not in cap:
        return None
    n = int(cap["species_idx"].shape[0])
    peak: Dict[int, float] = {}
    for i in range(n):
        for num, pi in zip(cap["species_idx"][i][:6], cap["species_pi"][i][:6]):
            if pi > 0:
                peak[int(num)] = max(peak.get(int(num), 0.0), float(pi))
    keep = sorted(peak, key=lambda s: -peak[s])[:max_species]
    series = []
    for i in range(n):
        row = dict(zip((int(x) for x in cap["species_idx"][i]), (float(x) for x in cap["species_pi"][i])))
        for num in keep:
            series.append({"decision": i, "species": species_name(num), "presence": row.get(num)})
    return {"species": [species_name(s) for s in keep], "points": series}


# ------------------------------------------------------------------------------- attention

def token_labels(layout: Dict[str, Any], teams: Dict[str, List[str]], cap: Dict[str, Any], i: int,
                 action_labels: Sequence[str]) -> List[dict]:
    """A plain-word label per trunk seat at decision ``i`` (`designs/prober/battle_view_v2.md` §3)."""
    n = int(layout["n_tokens"])
    if not layout.get("ok"):
        return [{"label": f"token {t}", "group": "other", "term": f"seat {t}"} for t in range(n)]
    out: List[dict] = []
    our = teams.get("our", [""] * 6)
    opp = teams.get("opp", [""] * 6)
    for j in range(6):
        out.append({"label": f"our {_display_species(our[j]) or 'slot ' + str(j + 1)}", "group": "our mons",
                    "term": f"our team token {j + 1}"})
    for j in range(6):
        hyp = "slot_is_hyp" in cap and bool(cap["slot_is_hyp"][i][j])
        name = species_name(int(cap["slot_species"][i][j])) if hyp else (_display_species(opp[j]) if opp[j] else "")
        out.append({"label": f"their {name or 'slot ' + str(j + 1)}" + (" (guess)" if hyp else ""),
                    "group": "their mons", "term": f"their team token {j + 1}" + (" — hypothesis" if hyp else "")})
    base = int(layout["base"])
    board_names = (["board (weather, hazards, clock)"] if base == 13
                   else ["our side (hazards, screens)", "their side (hazards, screens)", "field (weather, clock)"])
    for b in board_names:
        out.append({"label": b, "group": "board", "term": "board token"})
    for m in range(int(layout["n_e3"])):
        lab = action_display(action_labels[6 + m]) if len(action_labels) > 6 + m else ""
        out.append({"label": f"our move: {lab or '—'}", "group": "our moves", "term": f"E3 seat {m + 1} (our move)"})
    K = int(layout["k_e4"])
    for c in range(K):
        if "seat_nums" in cap and c < cap["seat_nums"].shape[1] and bool(cap["seat_live"][i][c]):
            nm = move_name(int(cap["seat_nums"][i][c]))
            tag = "seen" if bool(cap["seat_revealed"][i][c]) else "guess"
            lab = f"their move: {nm} ({tag})"
        else:
            lab = f"their move seat {c + 1}"
        out.append({"label": lab, "group": "their moves", "term": f"E4 seat {c + 1} (their threat move)"})
    for j in range(int(layout["n_tail"])):
        nm = out[6 + j]["label"].replace("their ", "", 1).replace(" (guess)", "")
        out.append({"label": f"their other moves ({nm})", "group": "their moves",
                    "term": f"E5 tail seat {j + 1} (the active's is OTHER_move)"})
    if layout.get("has_other"):
        out.append({"label": "a mon not on our list", "group": "their mons", "term": "OTHER_species"})
    ne = int(layout["n_events"])
    for e in range(ne):
        out.append({"label": f"event −{ne - e}", "group": "history", "term": "event seat (most recent last)"})
    while len(out) < n:
        out.append({"label": f"token {len(out)}", "group": "other", "term": "seat"})
    return out[:n]


def chosen_token(action: Optional[int], layout: Dict[str, Any]) -> Optional[int]:
    """The trunk seat the pointer head scores ``action`` from: a switch to slot i → our token i, move k
    → E3 seat k. Struggle has no token."""
    if action is None or not layout.get("ok"):
        return None
    a = int(action)
    if 0 <= a < 6:
        return a
    if 6 <= a < 10:
        return int(layout["base"]) + (a - 6)
    return None


def top_keys(row: np.ndarray, labels: Sequence[dict], masked: Optional[np.ndarray], k: int = 10) -> List[dict]:
    order = np.argsort(-row.astype(np.float64))
    out = []
    for t in order:
        if masked is not None and bool(masked[t]):
            continue
        out.append({"token": int(t), "label": labels[int(t)]["label"], "group": labels[int(t)]["group"],
                    "weight": float(row[t])})
        if len(out) >= k:
            break
    return out


def attention_summary(cap: Dict[str, Any], i: int, labels: Sequence[dict], action: Optional[int],
                      our_active_slot: Optional[int]) -> Optional[dict]:
    """The DEFAULT attention read: the layer×head average, and the two rows that matter — the chosen
    action's token and our active mon — as ranked keys."""
    if "attention" not in cap:
        return None
    A = cap["attention"][i].astype(np.float32)                 # [L,H,T,T]
    mean = A.mean(axis=(0, 1))
    km = cap.get("key_masked")
    masked = km[i] if km is not None else None
    layout = cap["layout"]
    ct = chosen_token(action, layout)
    rows = {}
    if ct is not None:
        rows["chosen"] = {"token": ct, "label": labels[ct]["label"], "keys": top_keys(mean[ct], labels, masked)}
    if our_active_slot is not None:
        rows["active"] = {"token": int(our_active_slot), "label": labels[int(our_active_slot)]["label"],
                          "keys": top_keys(mean[int(our_active_slot)], labels, masked)}
    recv = mean.sum(axis=0)
    if masked is not None:
        recv = np.where(masked, -1.0, recv)
    return {"rows": rows, "most_attended": top_keys(recv / max(1, mean.shape[0]), labels, masked, k=8),
            "n_layers": int(A.shape[0]), "n_heads": int(A.shape[1])}


def attention_matrix(cap: Dict[str, Any], i: int, layer: Optional[int], head: Optional[int]) -> np.ndarray:
    """[T,T] attention at decision ``i`` — one layer/head, or the average over whichever is None."""
    A = cap["attention"][i].astype(np.float32)
    A = A[[layer]] if layer is not None else A
    A = A[:, [head]] if head is not None else A
    return A.mean(axis=(0, 1))


# ------------------------------------------------------------------------------- operator facts

def operator_view(cap: Dict[str, Any], i: int, action_labels: Sequence[str], legal: Sequence[bool]) -> Optional[dict]:
    """Per OUR move (action 6+k): damage range vs their active (% of max HP), crit damage, P(KO),
    P(lands) for a status move, P(we move first); the move-resolution family's P(resolve) / P(KO first)
    when it is built (``None`` = not built in this model — a surface greys it)."""
    ops = cap.get("op")
    if not ops:
        return None
    d = ops[i]
    out = d.get("outgoing")
    land = d.get("status_landing") or []
    mr = cap.get("mr_move")
    coords = cap.get("mr_move_coords") or []
    idx = {c: n for n, c in enumerate(coords)}
    moves = []
    for k in range(4):
        a = 6 + k
        lab = action_labels[a] if len(action_labels) > a else ""
        if not lab or (len(legal) > a and not legal[a]):
            continue
        m = (out or {}).get("moves", [{}] * 4)[k] if out else {}
        row = {"action": a, "move": action_display(lab), "low": m.get("low"), "high": m.get("high"), "crit": m.get("crit"),
               "p_ko": m.get("pko"),
               "p_land": (land[k]["p_land"] if k < len(land) and land[k].get("known", 0) > 0.5 else None),
               "p_resolve": float(mr[i][k][idx["p_resolve"]]) if (mr is not None and "p_resolve" in idx) else None,
               "p_ko_first": float(mr[i][k][idx["p_ko_first"]]) if (mr is not None and "p_ko_first" in idx) else None}
        moves.append(row)
    inc = d.get("incoming") or []
    return {"moves": moves, "p_we_first": (out or {}).get("p_outspeed") if out else None,
            "move_resolution_built": mr is not None,
            "incoming": [{"slot": j, "phys_high": r["phys"]["high"], "spec_high": r["spec"]["high"],
                          "p_ko": max(r["phys"]["pko"], r["spec"]["pko"]), "p_they_first": 1.0 - r["p_outspeed"]}
                         for j, r in enumerate(inc)]}


__all__ = ["CALIBRATION_BINS", "action_display", "attention_matrix", "attention_summary", "belief_evolution", "chosen_token",
           "hypotheses_view", "intent_calibration", "intent_view", "move_id", "move_name", "operator_view",
           "opp_actual_action", "species_id", "species_name", "token_labels", "top_keys"]
