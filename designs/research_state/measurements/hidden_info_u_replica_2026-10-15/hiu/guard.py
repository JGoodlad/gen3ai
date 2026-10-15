"""The CONSISTENCY GUARD: every world this build plays is checked twice, and a mismatch RAISES ``GuardError``.

1. CONTRACT (what the arm promised, from the packed teams alone):
   - every arm: a seen mon sits in its TRUE slot with its true name and species; its revealed moves are in its set
     (a bare revealed Hidden Power matches any typed one); a revealed item / ability is held; the never-seen slots
     are exactly the slots no seen mon occupies; no base species twice.
   - O-U: every seen slot is the TRUE packed entry verbatim (realized gender); F+U: every never-seen slot is the
     TRUE packed entry verbatim and every seen slot is F world j's packed entry verbatim; O-M: every never-seen slot
     is true, and a seen mon's item, ability, nature, EVs and IVs are true; F+M: a seen mon's moves are true.
2. ENGINE (the world replayed through the pinned core): the opponent's FIRST request lists, slot by slot, the
   name, species, gender, moves (Hidden Power typed), item, ability and the five non-HP stats plus max HP; each
   must equal what the packed entry says (stats recomputed from nature / EVs / IVs).
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Optional, Sequence

from replica import builder as B
from replica import sampling as S


class GuardError(RuntimeError):
    pass


COUNTS: Counter = Counter()


def _fail(msg: str) -> None:
    COUNTS["raised"] += 1
    raise GuardError(msg)


def u_slots(prep: B.Prep) -> List[int]:
    seen = set(prep.seen_slot.values())
    return [i for i in range(6) if i not in seen]


def true_entry(prep: B.Prep, i: int) -> str:
    t = prep.true_opp[i].copy()
    t.gender = prep.opp_real_gender[i]
    return t.pack()


def contract(prep: B.Prep, arm: str, team: str, f_team: Optional[str] = None) -> None:
    sets = S.unpack(team)
    if len(sets) != 6:
        _fail(f"{prep.sid} {arm}: {len(sets)} mons")
    bases = [S.base_of(m.species) for m in sets]
    if len(set(bases)) != 6:
        _fail(f"{prep.sid} {arm}: a base species twice {bases}")
    us = u_slots(prep)
    if len(us) != 6 - len(prep.seen.mons):
        _fail(f"{prep.sid} {arm}: never-seen slots {us} for {len(prep.seen.mons)} seen mons")
    for name, m in prep.seen.mons.items():
        i = prep.seen_slot[name]
        w, t = sets[i], prep.true_opp[i]
        if w.name != t.name or w.species != t.species:
            _fail(f"{prep.sid} {arm}: slot {i} holds {w.name}/{w.species}, the seen mon is {t.name}/{t.species}")
        for mv in m.moves:
            ok = any(S.is_hp(x) for x in w.moves) if mv == "hiddenpower" else mv in w.moves
            if not ok and mv in t.moves:     # a revealed move the TRUE set holds must be in every world
                _fail(f"{prep.sid} {arm}: slot {i} lacks revealed move {mv}: {w.moves}")
        if m.item is not None and S.to_id_safe(w.item) != S.to_id_safe(m.item) and \
                S.to_id_safe(t.item) == S.to_id_safe(m.item):
            _fail(f"{prep.sid} {arm}: slot {i} item {w.item} != revealed {m.item}")
        if m.ability is not None and S.to_id_safe(w.ability) != S.to_id_safe(m.ability) and \
                S.to_id_safe(t.ability) == S.to_id_safe(m.ability):
            _fail(f"{prep.sid} {arm}: slot {i} ability {w.ability} != revealed {m.ability}")
        if arm in ("O-U", "O", "ID") and w.pack() != true_entry(prep, i):
            _fail(f"{prep.sid} {arm}: seen slot {i} is not the true entry: {w.pack()} vs {true_entry(prep, i)}")
        if arm == "O-M" and (w.item, w.ability, w.nature, tuple(w.evs), list(w.ivs)) != \
                (t.item, t.ability, t.nature, tuple(t.evs), list(t.ivs)):
            _fail(f"{prep.sid} O-M: seen slot {i} I/A/S differ from the truth")
        if arm == "F+M" and [x for x in w.moves] != [x for x in t.moves]:
            _fail(f"{prep.sid} F+M: seen slot {i} moves {w.moves} != true {t.moves}")
    for i in us:
        if arm in ("F+U", "O-M", "O", "ID") and sets[i].pack() != true_entry(prep, i):
            _fail(f"{prep.sid} {arm}: never-seen slot {i} is not the true entry")
    if arm == "F+U":
        if f_team is None:
            _fail(f"{prep.sid} F+U: no F world given")
        fs = S.unpack(f_team)
        for i in prep.seen_slot.values():
            if sets[i].pack() != fs[i].pack():
                _fail(f"{prep.sid} F+U: seen slot {i} differs from F's world: {sets[i].pack()} vs {fs[i].pack()}")
    COUNTS[f"contract_ok:{arm}"] += 1


def _expected_stats(m: S.MonSet) -> tuple:
    from replica.screen import realized
    st = tuple(realized(m.species, m.nature, tuple(m.evs), tuple(m.ivs)))
    return ((1,) + st[1:]) if m.species == "shedinja" else st      # Shedinja's HP is always 1


def _engine_move(x: str) -> str:
    """The request's move id with its happiness-derived power stripped (``return102`` -> ``return``)."""
    for base in ("return", "frustration"):
        if x.startswith(base) and x[len(base):].isdigit():
            return base
    return x


def first_request(res: dict, viewer: int) -> dict:
    for side, lines in res["chunks"]:
        if side != viewer:
            continue
        for ln in lines:
            if ln.startswith("|request|"):
                req = json.loads(ln[len("|request|"):])
                if req.get("side", {}).get("pokemon"):
                    return req
    raise GuardError("no opponent request in the replay")


def engine(prep: B.Prep, arm: str, teams: Sequence[str]) -> None:
    """Replay each team (no commands: the battle's start) and compare the opponent's first request, slot by slot."""
    from agents.battle.core_replay import run_core
    recs = [B._recorded(prep.battle, prep.opp, t, [], prep.own_team, prep.world_seed) for t in teams]
    results = run_core(recs, trackers=False, obs=False)
    ov = 1 - prep.v
    for j, (team, res) in enumerate(zip(teams, results)):
        if not res.get("ok"):
            _fail(f"{prep.sid} {arm} world {j}: the engine refused the team: {res.get('error')}")
        pk = first_request(res, ov)["side"]["pokemon"]
        sets = S.unpack(team)
        if len(pk) != len(sets):
            _fail(f"{prep.sid} {arm} world {j}: engine holds {len(pk)} mons, packed {len(sets)}")
        for i, (p, m) in enumerate(zip(pk, sets)):
            name = p["ident"].split(": ", 1)[1]
            det = p["details"].split(", ")
            gender = next((x for x in det[1:] if x in ("M", "F")), "")
            exp = _expected_stats(m)
            got_stats = (int(p["condition"].split("/")[1].split(" ")[0]),) + tuple(
                int(p["stats"][k]) for k in ("atk", "def", "spa", "spd", "spe"))
            bad = []
            if name != m.name:
                bad.append(f"name {name} != {m.name}")
            if S.to_id_safe(det[0]) != m.species and S.to_id_safe(det[0]) != S.to_id_safe(m.species_name):
                bad.append(f"species {det[0]} != {m.species}")
            if m.gender and gender != m.gender:
                bad.append(f"gender {gender} != {m.gender}")
            if [_engine_move(x) for x in p["moves"]] != list(m.moves):
                bad.append(f"moves {p['moves']} != {m.moves}")
            if S.to_id_safe(p.get("item", "")) != S.to_id_safe(m.item):
                bad.append(f"item {p.get('item')} != {m.item}")
            if S.to_id_safe(p.get("baseAbility", "")) != S.to_id_safe(m.ability):
                bad.append(f"ability {p.get('baseAbility')} != {m.ability}")
            if got_stats != exp:
                bad.append(f"stats {got_stats} != {exp}")
            if bad:
                _fail(f"{prep.sid} {arm} world {j} slot {i}: " + "; ".join(bad))
        COUNTS[f"engine_ok:{arm}"] += 1


def check(prep: B.Prep, arm: str, teams: Sequence[str], f_teams: Optional[Sequence[str]] = None) -> Dict[str, int]:
    for j, t in enumerate(teams):
        contract(prep, arm, t, f_teams[j] if f_teams is not None else None)
    engine(prep, arm, teams)
    return {"contract_ok": len(teams), "engine_ok": len(teams)}
