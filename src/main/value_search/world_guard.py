"""THE WORLD CONSISTENCY GUARD: every world the search builder returns is checked SLOT ↔ SPECIES ↔ FACTS against the
ENGINE's own view of the world team, and a mismatch RAISES (:class:`WorldConsistencyError`, never a ``WorldError``:
a world that is not what its record says is a defect, not an exclusion).

Why (GIGO, 2026-10-15): the builder draws each never-seen mon FOR a slot, then ``worlds.place_genders`` permutes the
drawn never-seen mons among the never-seen slots (the turn-0 gender dice). The provenance record ``Worlds.facts``
stayed indexed by the slot each mon was DRAWN for, so ~46 % of the never-seen fact records of the Tier-1 worlds
described another slot's mon, and the never-seen species "match" compared a species with the truth of a slot it did
not occupy. The worlds themselves (whole packed entries move together) were right; the RECORDS were not.

The checks, per world and opponent slot ``i`` (:func:`world_problems`):

* SHAPE — the team, the fact list and the engine's team have one entry per true slot; ``facts[i]["slot"] == i``;
* SLOT ↔ SPECIES — the packed entry's species, ``facts[i]["species"]`` and the species the ENGINE built at slot ``i``
  (its first request to that side, team order) are one; a SEEN mon sits in its TRUE slot; a never-seen slot holds a
  species play never showed;
* ENTRY ↔ ENGINE — the engine's item, ability, moves (Hidden Power typed by the IVs, a packed alias such as ``wisp``
  resolved through the engine's own table, and no move twice once resolved: a POOLSETS world of the Tier-1 study held
  ``wisp`` AND ``willowisp``, Will-O-Wisp twice) and EXACT stats (max HP included, from the species' base stats, the
  EVs, IVs, nature and level) are the packed entry's: the world the playouts run is the world the text names;
* FACTS ↔ ENTRY — every ``match`` flag of the record is recomputed from the entry against the truth OF THAT MON (a seen
  mon: its true set; a never-seen mon: the true set of the same species when it is a true never-seen species, else
  no truth: every flag false): item, ability, spread, moves (Hidden Power one move whatever its type), and the
  never-seen species flags (``match``: a TRUE never-seen species — set-based —, ``slot_match``: the truth's species
  at this slot; ``true_mon``).

The engine view is ONE batched ``core_events`` replay of the worlds with no command (team construction and the first
requests only): cheap beside the screen.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from agents.info_levers.packed import Spread, is_hp, parse_set, set_evs, set_ivs, toid, typed_moves

if TYPE_CHECKING:  # pragma: no cover
    from main.value_search.worlds import Root


class WorldConsistencyError(RuntimeError):
    """A built world whose slots, species, facts and engine view disagree (never an exclusion: a defect)."""


def species_of(entry: str) -> str:
    f = parse_set(entry)
    return toid(f[1] or f[0])


def engine_view(root: "Root", teams: Sequence[str], label: str = "guard") -> List[Optional[List[dict]]]:
    """Per world team: the ENGINE's view of the opponent's team in team order (the ``side.pokemon`` of the first
    request that side is sent), or None when the engine did not build it (a failed replay)."""
    from agents.battle.core_replay import run_core
    from agents.info_levers import revealed as RV

    out = run_core([root.rec(t, f"{label}#{j}", []) for j, t in enumerate(teams)])
    views: List[Optional[List[dict]]] = []
    for r in out:
        reqs = [rq for rq in RV.own_team_request(r, root.oi) if rq.get("side", {}).get("pokemon")] if r["ok"] else []
        views.append(list(reqs[0]["side"]["pokemon"]) if reqs else None)
    return views


def expected_stats(entry: str) -> Optional[Dict[str, int]]:
    """The exact stats (``hp`` = max HP) the engine builds for a packed entry, or None for an unknown species."""
    from agents.gen3_data import species
    from main.value_search.world_screen import NATURE_EFFECT

    f = parse_set(entry)
    sd = species.get(species_of(entry))
    if sd is None:
        return None
    lv = int(f[10]) if f[10].strip().isdigit() else 100
    evs, ivs = set_evs(f), set_ivs(f)
    up, down = NATURE_EFFECT.get(toid(f[5]), (None, None))
    out: Dict[str, int] = {}
    for k, (stat, ev, iv) in enumerate(zip(("hp", "atk", "def", "spa", "spd", "spe"), evs, ivs)):
        core = (2 * int(sd.base_stats[stat]) + iv + ev // 4) * lv // 100
        if stat == "hp":
            out[stat] = 1 if int(sd.base_stats["hp"]) == 1 else core + lv + 10
        else:
            v = core + 5
            out[stat] = v * 11 // 10 if stat == up else (v * 9 // 10 if stat == down else v)
    return out


def _engine_move(m: str) -> str:
    """A request's move id as the packed set names it: the engine appends Return's / Frustration's base power
    (``return102``), which is a happiness fact, not a move."""
    mid = toid(m)
    base = mid.rstrip("0123456789")
    return base if base in ("return", "frustration") else mid


def _resolved(m: str) -> str:
    """A packed move id through the engine's alias table (``wisp`` → ``willowisp``: the SAME table the Rust dex
    resolves a packed-team alias with)."""
    from agents.gen3_data import moves

    return str(moves.move_aliases().get(m, m))


def _canon(m: str) -> str:
    return "hiddenpower" if is_hp(m) else m


def _true_set_of(root: "Root", sid: str) -> Optional[List[str]]:
    """The TRUE packed set of species ``sid`` on the opponent's team, or None when the truth holds no such mon."""
    for f, s in zip(root.sets, root.sids):
        if s == sid:
            return list(f)
    return None


def world_problems(root: "Root", team: str, facts: Sequence[dict], view: Optional[List[dict]]) -> List[str]:
    """Every slot ↔ species ↔ facts ↔ engine disagreement of ONE world (module docstring); empty when consistent."""
    bad: List[str] = []
    ents = team.split("]")
    n = len(root.entries)
    if len(ents) != n or len(facts) != n:
        return [f"shape: {len(ents)} entries, {len(facts)} fact records, {n} true slots"]
    if view is None or len(view) != n:
        return [f"engine: the world team was not built ({'no view' if view is None else len(view)} mons)"]
    true_never = {s for s in root.sids if s not in root.kept}
    for i, (e, fa, pv) in enumerate(zip(ents, facts, view)):
        f = parse_set(e)
        sid = species_of(e)
        eng = toid(str(pv.get("details", "")).split(",")[0])
        if fa.get("slot") != i:
            bad.append(f"slot {i}: its fact record names slot {fa.get('slot')}")
        if fa.get("species") != sid or eng != sid:
            bad.append(f"slot {i}: entry {sid}, fact record {fa.get('species')}, engine {eng}")
            continue
        seen = sid in root.kept
        if bool(fa.get("seen")) != seen:
            bad.append(f"slot {i}: {sid} recorded seen={fa.get('seen')}, play {'showed' if seen else 'never showed'} it")
        if seen and root.sids[i] != sid:
            bad.append(f"slot {i}: the seen {sid} moved from its true slot {root.sids.index(sid)}")
        # ---- the entry is what the engine runs
        if toid(str(pv.get("item", ""))) != toid(f[2]):
            bad.append(f"slot {i} {sid}: item entry {toid(f[2])!r}, engine {pv.get('item')!r}")
        if f[3] and toid(str(pv.get("baseAbility", ""))) != toid(f[3]):
            bad.append(f"slot {i} {sid}: ability entry {toid(f[3])!r}, engine {pv.get('baseAbility')!r}")
        named = [_resolved(m) for m in typed_moves(f)]
        if [_engine_move(m) for m in pv.get("moves", [])] != named:
            bad.append(f"slot {i} {sid}: moves entry {named}, engine {pv.get('moves')}")
        if len(set(named)) != len(named):
            bad.append(f"slot {i} {sid}: a move twice once aliases resolve: {typed_moves(f)}")
        want = expected_stats(e)
        cond = str(pv.get("condition", "")).split(" ")[0]
        got = dict(pv.get("stats", {}), hp=int(cond.split("/")[1]) if "/" in cond else None)
        if want is None or any(got.get(k) != v for k, v in want.items()):
            bad.append(f"slot {i} {sid}: stats entry {want}, engine {got}")
        # ---- the record's flags are this mon's
        truth = list(root.sets[i]) if seen else (_true_set_of(root, sid) if sid in true_never else None)
        if not seen:
            sf = fa.get("species_fact", {})
            if bool(sf.get("match")) != (sid in true_never):
                bad.append(f"slot {i} {sid}: species match={sf.get('match')}, a true never-seen species: "
                           f"{sid in true_never}")
            if "slot_match" in sf and bool(sf["slot_match"]) != (sid == root.sids[i]):
                bad.append(f"slot {i} {sid}: species slot_match={sf['slot_match']}, true here: {root.sids[i]}")
            if bool(fa.get("true_mon")) != (truth is not None):
                bad.append(f"slot {i} {sid}: true_mon={fa.get('true_mon')}, the truth holds it: {truth is not None}")
        recomputed = {
            "item": truth is not None and toid(f[2]) == toid(truth[2]),
            "ability": truth is not None and toid(f[3]) == toid(truth[3]),
            "spread": truth is not None and Spread.of(f).same(Spread.of(truth)),
            "moves": truth is not None and sorted(map(_canon, typed_moves(f))) == sorted(map(_canon, typed_moves(truth))),
        }
        for cls, ok in recomputed.items():
            rec = fa.get(cls)
            if rec is not None and bool(rec.get("match")) != ok:
                bad.append(f"slot {i} {sid}: {cls} match={rec.get('match')}, recomputed from the entry {ok}")
    return bad


def problems(root: "Root", teams: Sequence[str], facts: Sequence[Sequence[dict]]) -> Dict[int, List[str]]:
    """{world index: its problems} over a set of worlds (one engine replay batch); empty when all are consistent."""
    if len(teams) != len(facts):
        return {-1: [f"{len(teams)} teams, {len(facts)} fact lists"]}
    views = engine_view(root, teams)
    out = {}
    for j, (t, fl, v) in enumerate(zip(teams, facts, views)):
        b = world_problems(root, t, fl, v)
        if b:
            out[j] = b
    return out


def check_worlds(root: "Root", teams: Sequence[str], facts: Sequence[Sequence[dict]], key: str = "") -> None:
    """THROWING: raise :class:`WorldConsistencyError` naming the first problems when any world is inconsistent."""
    bad = problems(root, teams, facts)
    if bad:
        first = "; ".join(f"world {j}: {', '.join(p[:3])}" for j, p in list(bad.items())[:3])
        raise WorldConsistencyError(f"{key}: {len(bad)} of {len(teams)} worlds inconsistent ({first})")
