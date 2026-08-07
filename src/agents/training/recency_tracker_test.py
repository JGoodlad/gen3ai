"""Unit tests for the E9 step-1 RecencyTracker (roadmap §3.9) — the pure counter logic.

TURN-ANCHORED semantics (the recency fuzz's finding): each fact stores its latest EVENT TURN
and the counter is `cur_turn − event_turn − 1` (clamped ≥ 0) — invariant to which decision's
window processed the event (forced-switch turns used to skew the old tick-then-reset form).
An event from the just-completed turn reads 0; the on-field mon reads 0; a never-tracked mon
reads max staleness 1.0."""
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
    turn: int


def _norm(n: int) -> float:
    return math.log1p(min(n, 10)) / math.log(11.0)


def test_turn_anchored_counters_are_processing_invariant():
    """The same turn-1 MOVE reads identically whether its window is processed at the turn-1
    forced-switch decision or the turn-2 decision — the fuzz-caught invariance."""
    a, b = RecencyTracker(), RecencyTracker()
    ev = [_Ev(EventKind.MOVE, OURS, "snorlax", 1)]
    a.update(1, ev, "snorlax", None)          # processed same-turn (forced switch)
    a.update(4, [], "blissey", None)
    b.update(2, ev, "snorlax", None)          # processed next-turn (normal)
    b.update(4, [], "blissey", None)
    # The EVENT-derived channel (acted) is processing-invariant; seen's active-anchor
    # legitimately differs (a and b OBSERVED snorlax on field at different decisions).
    assert a.values(OURS, "snorlax")[1] == b.values(OURS, "snorlax")[1] == _norm(4 - 1)


def test_fresh_event_and_active_read_zero():
    t = RecencyTracker()
    t.update(2, [_Ev(EventKind.MOVE, OURS, "snorlax", 1),
                 _Ev(EventKind.DAMAGE, OPP, "skarmory", 1)], "snorlax", "skarmory")
    assert t.values(OURS, "snorlax") == (0.0, _norm(1), _norm(10))
    assert t.values(OPP, "skarmory")[2] == _norm(1), "hit LAST turn reads 1 (one turn ago)"
    assert t.values(OPP, "skarmory")[0] == 0.0, "on-field mon reads seen 0"


def test_reset_sources_are_channel_exact():
    t = RecencyTracker()
    t.update(2, [_Ev(EventKind.MOVE, OURS, "snorlax", 1)], "snorlax", "skarmory")
    t.update(5, [_Ev(EventKind.DAMAGE, OURS, "snorlax", 4),
                 _Ev(EventKind.SWITCH, OPP, "blissey", 4)], "snorlax", "blissey")
    seen, acted, hit = t.values(OURS, "snorlax")
    assert seen == 0.0, "active every decision"
    assert acted == _norm(5 - 1), "no MOVE since turn 1"
    assert hit == _norm(1), "the turn-4 DAMAGE reads 1 at turn 5"
    assert t.values(OPP, "blissey")[0] == 0.0, "SWITCH (and active) reset seen"
    assert t.values(OPP, "skarmory")[0] == _norm(5 - 2), "benched opp mon goes stale"


def test_never_tracked_reads_max_staleness():
    t = RecencyTracker()
    t.update(3, [], "snorlax", None)
    assert t.values(OPP, "tyranitar") == (1.0, 1.0, 1.0)
    assert t.values(OURS, None) == (1.0, 1.0, 1.0)


def test_saturation_caps_at_ten():
    t = RecencyTracker()
    t.update(1, [_Ev(EventKind.MOVE, OURS, "snorlax", 1)], "snorlax", None)
    t.update(40, [], "blissey", None)
    assert t.values(OURS, "snorlax") == (1.0, 1.0, 1.0)
