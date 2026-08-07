"""Unit tests for the E9 step-1 RecencyTracker (roadmap §3.9) — the pure counter logic.

Load-bearing invariants: turn-delta ticking (a multi-decision turn ticks ONCE), the three
reset sources (MOVE → acted+seen, SWITCH → seen, DAMAGE → was_hit, live actives → seen),
the never-tracked default (max staleness 1.0), and the log-saturation obs form."""
import math
from dataclasses import dataclass
from typing import Optional

from agents.battle.battle_event import OURS, OPP, EventKind
from agents.training.episode_tracker import RecencyTracker


@dataclass
class _Ev:
    kind: EventKind
    side: Optional[str]
    actor_species: Optional[str]


def _norm(n: int) -> float:
    return math.log1p(min(n, 10)) / math.log(11.0)


def test_ticks_by_turn_delta_not_per_decision():
    t = RecencyTracker()
    t.update(1, [_Ev(EventKind.MOVE, OURS, "snorlax")], "snorlax", "skarmory")
    assert t.values(OURS, "snorlax") == (0.0, 0.0, _norm(10))
    # Two decisions on the SAME turn (forced switch) — no tick.
    t.update(1, [], "snorlax", "skarmory")
    assert t.values(OURS, "snorlax")[1] == 0.0
    # Three turns pass without snorlax acting.
    t.update(4, [], "blissey", "skarmory")
    assert t.values(OURS, "snorlax") == (_norm(3), _norm(3), _norm(10))


def test_reset_sources_are_channel_exact():
    t = RecencyTracker()
    t.update(1, [_Ev(EventKind.MOVE, OURS, "snorlax")], "snorlax", "skarmory")
    t.update(5, [_Ev(EventKind.DAMAGE, OURS, "snorlax"),
                 _Ev(EventKind.SWITCH, OPP, "blissey")], "snorlax", "blissey")
    seen, acted, hit = t.values(OURS, "snorlax")
    assert seen == 0.0, "active every decision"
    assert acted == _norm(4), "no MOVE since turn 1"
    assert hit == 0.0, "the DAMAGE event resets was_hit"
    assert t.values(OPP, "blissey")[0] == 0.0, "SWITCH (and active) reset seen"
    assert t.values(OPP, "skarmory")[0] == _norm(4), "benched opp mon goes stale"


def test_never_tracked_reads_max_staleness():
    t = RecencyTracker()
    t.update(3, [], "snorlax", None)
    assert t.values(OPP, "tyranitar") == (1.0, 1.0, 1.0)
    assert t.values(OURS, None) == (1.0, 1.0, 1.0)


def test_saturation_caps_at_ten():
    t = RecencyTracker()
    t.update(1, [_Ev(EventKind.MOVE, OURS, "snorlax")], "snorlax", None)
    t.update(40, [], "blissey", None)
    assert t.values(OURS, "snorlax") == (1.0, 1.0, 1.0)
