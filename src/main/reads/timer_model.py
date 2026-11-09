"""THE TURN TIMER: Showdown's ladder battle timer as a PURE model, and the bank-aware SAFETY NET of a deployable
searched player (``designs/endstate/design_searched_player.md`` §2-§3; the cost read's DRAFT §7 R2).

THE MODEL (``deps/pokemon-showdown/server/room-battle.ts``, class ``RoomBattleTimer``, a NON-challenge singles battle
in a format with no ``timer`` rule, as ``[Gen 3] OU``; every constant below is pinned to that file's literals by
``timer_model_test``, so a Showdown bump that moves one fails it): the bank starts at :data:`STARTING_S` +
:data:`GRACE_S`; each NEW turn adds :data:`ADD_PER_TURN_S` (``accelerate``: :data:`TICK_S` less past turn
:data:`ACCEL_TURN`, and past turn :data:`ACCEL_TURN_2` nothing on alternate requests, modelled as odd turns), capped
at :data:`STARTING_S`; a mid-turn request (a second decision of ours in one turn) adds min(addPerTurn, TICK) to the
bank, uncapped; one request's own limit is min(bank, :data:`MAX_TURN_S`); time is charged in whole :data:`TICK_S`
ticks (a 4.9-s answer costs 0); a request that reaches its limit FORFEITS (``timeoutAutoChoose`` false).

OUR RESPONSE TIME is the decision's search wall (searched or excluded) + :data:`OVERHEAD_S`, and
:data:`OVERHEAD_S` alone at a decision the search is not asked to make. The approximations are the design's
(FINDING SP-6): the opponent's own mid-turn requests, which also credit us, are unseen (conservative); a cut is
assumed to leave the rest of the game's costs unchanged.

THE POLICIES (:data:`POLICIES`, the prototype ``measurements/searched_cost_2026-11-09/inputs/timer_replay.py``'s,
reproduced on its banked cells): ``as_played`` (no deadline); ``fixed_<D>`` (π's pick at D s); ``bank_r10`` (π's
pick at min(bank, 150) − 10 − OVERHEAD_S: THE BANK-AWARE DEADLINE); ``+bg`` (the cut decision's step runs on, so the
carry stays a function of I, and delays the next search-path decision: the design §3's safety net, which cuts the
PICK and never the build); ``bank_r10_off`` (the battle's search switched off from its first cut).

THE SAFETY NET (:class:`Clock`): the live searched player (:mod:`main.anchors.search_side`, a preset's
``net_reserve_s``) feeds the clock every decision of ours in order; at a search-path decision whose need (the backlog
of an earlier cut step plus this wall) passes the bank-aware deadline the PICK is π's. It is exactly
:func:`simulate` with ``reserve`` and ``background=True``, so a replay of the player's own records reproduces its
cuts. Pure: no clock is read here; the walls are the caller's.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: ``room-battle.ts`` literals (pinned by ``timer_model_test``)
TICK_S = 5
STARTING_S = 150
GRACE_S = 60
MAX_TURN_S = 150
ADD_PER_TURN_S = 10
ACCEL_TURN = 100
ACCEL_TURN_2 = 200
#: our modelled response time beyond the search's own wall (the prototype's; the live net's)
OVERHEAD_S = 1.0
#: the design §3's bank-aware reserve (answer by min(bank, 150) − 10 s)
RESERVE_S = 10.0
SCHEMA = "gen3_timer_replay_v1"
#: the prototype's policy table (``timer_replay.py`` POLICIES): name -> keyword arguments of :func:`simulate`
POLICIES: Dict[str, Dict[str, Any]] = {
    "as_played": {}, **{f"fixed_{d}": {"deadline": d} for d in (150, 60, 30, 20, 15, 10, 5)},
    "bank_r10": {"reserve": RESERVE_S}, "bank_r10_fixed_15": {"reserve": RESERVE_S, "deadline": 15},
    "bank_r10_off": {"reserve": RESERVE_S, "off_after_cut": True}}
#: the policy the deployable player's safety net IS (the design §3: the cut step runs on)
NET_POLICY = "bank_r10+bg"


def add_for_turn(turn: int) -> int:
    """The bank a NEW turn ``turn`` adds (``updateTurn`` with ``accelerate``)."""
    if turn <= ACCEL_TURN:
        return ADD_PER_TURN_S
    if turn <= ACCEL_TURN_2:
        return ADD_PER_TURN_S - TICK_S
    return 0 if turn % 2 else ADD_PER_TURN_S - TICK_S


@dataclass
class Step:
    """One decision through the clock: the bank and limit it met, the deadline, the need, whether it was CUT (π's
    pick) and whether the response reached the limit (a FORFEIT)."""

    bank_s: float
    limit_s: float
    deadline_s: Optional[float]
    need_s: float
    resp_s: float
    cut: bool
    forfeit: bool

    def record(self) -> Dict[str, Any]:
        """The fields a live record keeps (``search_side``'s ``net`` block)."""
        r = lambda x: None if x is None else round(float(x), 3)       # noqa: E731
        return {"bank_s": r(self.bank_s), "limit_s": r(self.limit_s), "deadline_s": r(self.deadline_s),
                "need_s": r(self.need_s), "cut": bool(self.cut), "forfeit": bool(self.forfeit)}


class Clock:
    """One battle's timer and policy (module docstring). ``step(turn, path, wall)`` per decision of ours, in order:
    ``path`` = the search was asked (searched or excluded), ``wall`` = its wall seconds (ignored off the path)."""

    def __init__(self, *, deadline: Optional[float] = None, reserve: Optional[float] = None,
                 background: bool = False, off_after_cut: bool = False, overhead: float = OVERHEAD_S) -> None:
        self.deadline, self.reserve, self.background = deadline, reserve, bool(background)
        self.off_after_cut, self.overhead = bool(off_after_cut), float(overhead)
        self.bank, self.backlog, self.off = float(STARTING_S + GRACE_S), 0.0, False
        self.prev: Optional[int] = None
        self.forfeited = False

    def step(self, turn: int, path: bool, wall: float = 0.0) -> Step:
        turn = int(turn)
        if self.prev is not None:
            self.bank = (min(self.bank + add_for_turn(turn), float(STARTING_S)) if turn > self.prev
                         else self.bank + TICK_S)
        self.prev = turn
        limit = min(self.bank, float(MAX_TURN_S))
        bank = self.bank
        d: Optional[float] = None
        cut = False
        if path and self.off:
            cut, need, resp = True, 0.0, self.overhead
        elif path:
            need = self.backlog + float(wall)
            d = math.inf if self.deadline is None else float(self.deadline)
            if self.reserve is not None:
                d = min(d, max(0.0, limit - float(self.reserve) - self.overhead))
            if need > d:
                cut = True
                resp, self.backlog = d + self.overhead, ((need - d) if self.background else 0.0)
                self.off = self.off_after_cut
            else:
                resp, self.backlog = need + self.overhead, 0.0
        else:
            need, resp = 0.0, self.overhead
            self.backlog = max(0.0, self.backlog - self.overhead)
        forfeit = resp >= limit
        self.forfeited = self.forfeited or forfeit
        if not forfeit:
            self.bank -= TICK_S * math.floor(resp / TICK_S)
        return Step(bank_s=bank, limit_s=limit, deadline_s=None if d is None or math.isinf(d) else d, need_s=need,
                    resp_s=resp, cut=cut, forfeit=forfeit)


def policy_kwargs(name: str) -> Dict[str, Any]:
    """``"bank_r10+bg"`` -> :class:`Clock`'s keyword arguments."""
    base, bg = (name[:-3], True) if name.endswith("+bg") else (name, False)
    if base not in POLICIES:
        raise ValueError(f"timer policy {name!r} is none of {sorted(POLICIES)} (+bg)")
    return dict(POLICIES[base], background=bg)


def simulate(rows: Iterable[Mapping[str, Any]], **kw: Any) -> Dict[str, int]:
    """One game's decisions (each ``{turn, searched, excluded, wall}``, every decision of ours in order) through a
    :class:`Clock` with ``kw``: ``{forfeit, cut, cut_searched, path, battle_cut}``; the game stops at its forfeit."""
    c = Clock(**kw)
    cut = cut_searched = path = 0
    for r in rows:
        on = bool(r["searched"]) or bool(r["excluded"])
        s = c.step(int(r["turn"]), on, float(r.get("wall") or 0.0))
        path += int(on)
        cut += int(s.cut)
        cut_searched += int(s.cut and bool(r["searched"]))
        if s.forfeit:
            return {"forfeit": 1, "cut": cut, "cut_searched": cut_searched, "path": path, "battle_cut": int(cut > 0)}
    return {"forfeit": 0, "cut": cut, "cut_searched": cut_searched, "path": path, "battle_cut": int(cut > 0)}


def rows_of(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """A live decision log's records (``search_decisions.jsonl``, :mod:`main.anchors.search_side`) as clock rows: a
    searched decision is one whose search produced a pick (a safety-net cut included: its step ran), an excluded one
    ``why == "excluded"``, the wall the record's ``wall_s``."""
    return [{"turn": int(r["turn"]), "searched": bool(r.get("searched")), "excluded": r.get("why") == "excluded",
             "wall": float(r.get("wall_s") or 0.0)} for r in records]


def tally(games: Iterable[Tuple[str, Sequence[Mapping[str, Any]]]], policy: str) -> Dict[str, Dict[str, int]]:
    """``{panel and "U": {games, forfeit, cut, cut_searched, path, searched, battle_cut}}`` (exact counts) of
    ``(panel, rows)`` games under ``policy``; ``searched`` counts the rows' searched decisions (the cut share's
    denominator: every decision whose search produced a pick, as played or replayed)."""
    kw = policy_kwargs(policy)
    acc: Dict[str, Counter] = defaultdict(Counter)
    for panel, rows in games:
        s = simulate(rows, **kw)
        n_s = sum(1 for r in rows if r["searched"])
        for P in (panel, "U"):
            acc[P]["games"] += 1
            acc[P]["searched"] += n_s
            for k, v in s.items():
                acc[P][k] += v
    return {P: dict(a) for P, a in sorted(acc.items())}


def shares(t: Mapping[str, int]) -> Dict[str, Optional[float]]:
    """A tally's shares, rounded as the prototype's ``timer`` block (``forfeit_share`` …)."""
    def q(a: int, b: int) -> Optional[float]:
        return round(a / b, 4) if b else None

    return {"forfeit_share": q(t["forfeit"], t["games"]), "cut_share_of_search_path": q(t["cut"], t["path"]),
            "cut_share_of_searched": q(t["cut_searched"], t["searched"]),
            "battles_with_a_cut": q(t["battle_cut"], t["games"])}


def replay(games: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]],
           policies: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The REPORTED replay table: every policy (default: the prototype's, each also ``+bg`` where it applies) per panel
    and pooled U, as counts and shares."""
    if policies is None:
        policies = [n + bg for n, kw in POLICIES.items() for bg in ("", "+bg")
                    if not (bg and (n == "as_played" or kw.get("off_after_cut")))]
    out: Dict[str, Any] = {"schema": SCHEMA, "overhead_s": OVERHEAD_S, "policies": {}}
    for name in policies:
        t = tally(games, name)
        out["policies"][name] = {P: dict(counts=c, **shares(c)) for P, c in t.items()}
    return out
