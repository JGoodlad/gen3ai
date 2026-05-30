"""Tests for the seq-cursor slicing API (events_since / event_cursor).

This is the granularity the per-decision TurnDelta window needs: "events since I was
last asked to act", which is NOT a protocol |turn|N boundary (a forced switch splits a
turn into two decision windows).
"""

import logging

from agents.battle.battle_event import EventKind
from agents.battle.gen3_battle import Gen3Battle

LOG = logging.getLogger("event-cursor-test")

_PRE = [
    ["", "player", "p1", "p1user", "", ""],
    ["", "player", "p2", "p2user", "", ""],
    ["", "teamsize", "p1", "6"],
    ["", "teamsize", "p2", "6"],
    ["", "gen", "3"],
    ["", "start"],
    ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
    ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"],
    ["", "turn", "1"],
]


def _battle(extra):
    b = Gen3Battle("battle-gen3ou-cur", "p1user", LOG, gen=3)
    for line in _PRE + extra:
        b.parse_message(line)
    return b


def test_cursor_advances_with_recorded_events():
    b = _battle([])
    c0 = b.event_cursor
    assert c0 == len(b.events)  # cursor == count of events so far
    b.parse_message(["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"])
    b.parse_message(["", "-damage", "p2a: Tyra", "50/100"])
    assert b.event_cursor == c0 + 2  # two EVENTs recorded


def test_control_lines_do_not_advance_cursor():
    b = _battle([])
    c0 = b.event_cursor
    b.parse_message(["", "upkeep"])     # CONTROL — no event
    b.parse_message(["", "turn", "2"])  # CONTROL — no event
    assert b.event_cursor == c0


def test_events_since_returns_exactly_the_window():
    b = _battle([
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"],
        ["", "-damage", "p2a: Tyra", "50/100"],
    ])
    cursor = b.event_cursor  # snapshot "now" — as if asked for input here
    # opponent acts in the next window
    b.parse_message(["", "move", "p2a: Tyra", "Rock Slide", "p1a: Zappy"])
    b.parse_message(["", "-supereffective", "p1a: Zappy"])
    b.parse_message(["", "-damage", "p1a: Zappy", "20/100"])
    window = b.events_since(cursor)
    kinds = [e.kind for e in window]
    assert EventKind.MOVE in kinds and EventKind.SUPEREFFECTIVE in kinds
    # nothing from before the cursor leaked in
    assert all(e.seq >= cursor for e in window)
    # the move in the window is the opponent's, not ours from the prior window
    move = next(e for e in window if e.kind is EventKind.MOVE)
    assert move.side == "opp" and move.move_id == "rockslide"


def test_events_since_zero_is_whole_log():
    b = _battle([
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"],
        ["", "-damage", "p2a: Tyra", "50/100"],
    ])
    assert b.events_since(0) == list(b.events)


def test_window_can_span_a_turn_boundary():
    """A decision window is NOT a protocol turn: if we faint, our next input is a
    forced switch and the window since our last move spans the |turn| line."""
    b = _battle([
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"],
        ["", "-damage", "p2a: Tyra", "50/100"],
    ])
    cursor = b.event_cursor
    # opp KOs us, the turn ends, we're asked to switch (next decision)
    b.parse_message(["", "move", "p2a: Tyra", "Rock Slide", "p1a: Zappy"])
    b.parse_message(["", "-supereffective", "p1a: Zappy"])
    b.parse_message(["", "-damage", "p1a: Zappy", "0 fnt"])
    b.parse_message(["", "faint", "p1a: Zappy"])
    b.parse_message(["", "turn", "2"])
    window = b.events_since(cursor)
    turns_in_window = {e.turn for e in window}
    # the window legitimately contains events from turn 1 (the KO sequence)
    assert 1 in turns_in_window
    assert any(e.kind is EventKind.FAINT and e.side == "ours" for e in window)


def test_consecutive_windows_partition_the_log():
    """Successive (cursor -> event_cursor) windows tile the whole log with no gaps
    or overlaps — the property the TurnDelta fold relies on."""
    b = _battle([])
    cursors = [b.event_cursor]
    scripts = [
        [["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"],
         ["", "-damage", "p2a: Tyra", "50/100"]],
        [["", "move", "p2a: Tyra", "Rock Slide", "p1a: Zappy"],
         ["", "-damage", "p1a: Zappy", "60/100"]],
        [["", "move", "p1a: Zappy", "Roar", "p2a: Tyra"],
         ["", "drag", "p2a: Bliss", "Blissey, L100, F", "100/100"]],
    ]
    start = cursors[0]  # events before the first cursor (the lead switch-ins) precede it
    collected = []
    for s in scripts:
        for line in s:
            b.parse_message(line)
        collected.extend(b.events_since(cursors[-1]))
        cursors.append(b.event_cursor)
    # concatenating each window in order reproduces the log from the first cursor on,
    # exactly once (no gaps, no overlaps)
    assert collected == list(b.events)[start:]
