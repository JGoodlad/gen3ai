"""State snapshots (from BOTH players' own battle objects) and the expectation checker."""
from __future__ import annotations

from typing import Dict, Optional

from poke_env.data.normalize import to_id_str

STAGE = {-6: 2 / 8, -5: 2 / 7, -4: 2 / 6, -3: 2 / 5, -2: 2 / 4, -1: 2 / 3, 0: 1.0,
         1: 3 / 2, 2: 4 / 2, 3: 5 / 2, 4: 6 / 2, 5: 7 / 2, 6: 8 / 2}


def _status(mon) -> Optional[str]:
    st = getattr(mon, "status", None)
    return None if st is None else st.name.lower()


def _eff_speed(mon) -> float:
    spe = float((mon.stats or {}).get("spe") or 0)
    spe *= STAGE[int(mon.boosts.get("spe", 0))]
    if _status(mon) == "par":
        spe *= 0.25
    return spe


def _side(b) -> Dict:
    a = b.active_pokemon
    return {
        "active": to_id_str(a.species) if a else None,
        "hp_abs": a.current_hp if a else None, "hp_max": a.max_hp if a else None,
        "hp": (a.current_hp / a.max_hp) if a and a.max_hp else None,
        "boosts": {k: v for k, v in (a.boosts or {}).items() if v} if a else {},
        "status": _status(a) if a else None,
        "spe_stat": (a.stats or {}).get("spe") if a else None,
        "eff_speed": _eff_speed(a) if a else None,
        "team": {to_id_str(m.species): round(m.current_hp / m.max_hp, 4) if m.max_hp else None
                 for m in b.team.values()},
        "side_conditions": sorted(str(k.name).lower() for k in b.side_conditions),
        "foe_revealed_moves": sorted(b.opponent_active_pokemon.moves.keys()) if b.opponent_active_pokemon else [],
        "foe_revealed_species": sorted(to_id_str(m.species) for m in b.opponent_team.values()),
    }


def snapshot(b1, b2) -> Dict:
    """b1 = p1's own battle, b2 = p2's own battle (exact HP of each side's own mons)."""
    s1, s2 = _side(b1), _side(b2)
    return {"turn": int(b1.turn), "weather": sorted(str(w.name).lower() for w in b1.weather),
            "p1": s1, "p2": s2,
            "p1_first": (s1["eff_speed"] or 0) > (s2["eff_speed"] or 0),
            "speed_tie": (s1["eff_speed"] or 0) == (s2["eff_speed"] or 0)}


def check(snap: Dict, expect: Dict) -> list:
    """Return the list of FAILED predicates (empty = the state is what was intended)."""
    bad = []
    for k, v in expect.items():
        if k in ("p1_active", "p2_active"):
            got = snap[k[:2]]["active"]
            if got != v:
                bad.append(f"{k}: {got} != {v}")
        elif k in ("p1_status", "p2_status"):
            got = snap[k[:2]]["status"]
            if got != v:
                bad.append(f"{k}: {got} != {v}")
        elif k in ("p1_hp_band", "p2_hp_band"):
            got = snap[k[:2]]["hp"]
            if got is None or not (v[0] <= got <= v[1]):
                bad.append(f"{k}: {got} not in {v}")
        elif k in ("p1_boost_spe", "p2_boost_spe"):
            got = int(snap[k[:2]]["boosts"].get("spe", 0))
            if got != v:
                bad.append(f"{k}: {got} != {v}")
        elif k == "p1_first":
            if snap["speed_tie"] or snap["p1_first"] != v:
                bad.append(f"p1_first: {snap['p1_first']} (tie={snap['speed_tie']}) != {v}")
        elif k == "p2_moves_include":
            got = set(snap["p1"]["foe_revealed_moves"])
            if not set(v) <= got:
                bad.append(f"p2 revealed {sorted(got)} lacks {v}")
        elif k == "p2_moves_exclude":
            got = set(snap["p1"]["foe_revealed_moves"])
            if set(v) & got:
                bad.append(f"p2 revealed {sorted(got)} includes {v}")
        elif k == "p1_moves_include":
            got = set(snap["p2"]["foe_revealed_moves"])
            if not set(v) <= got:
                bad.append(f"p1 revealed {sorted(got)} lacks {v}")
        elif k == "p1_moves_exclude":
            got = set(snap["p2"]["foe_revealed_moves"])
            if set(v) & got:
                bad.append(f"p1 revealed {sorted(got)} includes {v}")
        elif k == "p2_side_conditions":
            got = snap["p2"]["side_conditions"]
            if sorted(v) != got:
                bad.append(f"p2 side conditions {got} != {sorted(v)}")
        elif k == "p1_team_hp":
            for sp, band in v.items():
                got = snap["p1"]["team"].get(sp)
                if got is None or not (band[0] <= got <= band[1]):
                    bad.append(f"p1 {sp} hp {got} not in {band}")
        else:
            bad.append(f"unknown predicate {k}")
    return bad
