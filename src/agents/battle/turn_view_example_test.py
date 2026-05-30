"""Worked examples: "what happened this turn?" via TurnView.

This file doubles as living documentation for the long-term consumers (TurnDelta,
the reward manager, the replay agent). Each test reads like the question a consumer
asks. If the API stops being able to answer one of these cleanly, a test breaks.

The driving principle (see the Focus Punch example below): we never collapse a
protocol fact into a fixed vocabulary that can silently drop it. `cant_reason` /
`cant_move` are free-form strings straight from the line, so a never-before-seen
reason is preserved verbatim rather than vanishing into a generic "didn't act" bit.
"""

import logging

import pytest

from agents.battle.battle_event import EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.turn_view import TurnView

LOG = logging.getLogger("turnview-example")

_PRE = [
    ["", "player", "p1", "p1user", "", ""],
    ["", "player", "p2", "p2user", "", ""],
    ["", "teamsize", "p1", "6"],
    ["", "teamsize", "p2", "6"],
    ["", "gametype", "singles"],
    ["", "gen", "3"],
    ["", "start"],
    ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
    ["", "switch", "p2a: Ttar", "Tyranitar, L100, M", "100/100"],
    ["", "turn", "1"],
]


def _battle(turn1_lines):
    b = Gen3Battle("battle-gen3ou-ex", "p1user", LOG, gen=3)
    for line in _PRE + turn1_lines + [["", "turn", "2"]]:
        b.parse_message(line)
    return b


# ───────────────────────────────────────────────────────────────────────────
# THE BASIC EXAMPLE: how do I tell what happened on a turn?
# ───────────────────────────────────────────────────────────────────────────
def test_did_we_both_attack_was_there_a_switch_a_faint():
    """The canonical 'read a turn' walkthrough. A consumer builds one TurnView and
    answers every structural question off `.ours` / `.opp` and a few turn-level reads.
    """
    b = _battle([
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Ttar"],
        ["", "-damage", "p2a: Ttar", "40/100"],
        ["", "move", "p2a: Ttar", "Rock Slide", "p1a: Zappy"],
        ["", "-supereffective", "p1a: Zappy"],
        ["", "-damage", "p1a: Zappy", "0 fnt"],
        ["", "faint", "p1a: Zappy"],
    ])
    v = TurnView.for_turn(b, 1)

    # Did BOTH sides attack this turn?
    both_attacked = v.ours.moved and v.opp.moved
    assert both_attacked is True

    # Who moved first?
    assert v.we_moved_first is True
    assert v.move_order == ["ours", "opp"]

    # What did each side do?
    assert v.ours.move_id == "thunderbolt"
    assert v.opp.move_id == "rockslide"
    assert v.opp.effectiveness == 2.0          # Rock Slide was super-effective on us

    # Was there a switch this turn? (no — both used moves)
    any_switch = v.ours.switched or v.opp.switched
    assert any_switch is False

    # Was there a faint? Who?
    assert v.faints() == ["zapdos"]
    assert v.ours.fainted is True
    assert v.opp.fainted is False


def test_switch_then_attack():
    """We switch; they attack into the incoming mon."""
    b = _battle([
        ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
        ["", "move", "p2a: Ttar", "Earthquake", "p1a: Skarm"],
        ["", "-immune", "p1a: Skarm"],
    ])
    v = TurnView.for_turn(b, 1)

    # exactly one side switched, the other attacked
    assert v.ours.switched is True and v.ours.switched_to == "skarmory"
    assert v.ours.moved is False
    assert v.opp.moved is True and v.opp.move_id == "earthquake"
    assert v.opp.effectiveness == 0.0          # Ground vs Flying/Steel Skarmory: immune
    # move order is "unknown" because only one side used a move
    assert v.we_moved_first is None


# ───────────────────────────────────────────────────────────────────────────
# THE MOTIVATING EXAMPLE: a disrupted Focus Punch must stay interpretable.
#
# Showdown emits  |cant|p1a: Blaziken|Focus Punch|Focus Punch  when the user is hit
# before it can punch. The OLD pipeline collapsed this into a contentless
# failed_to_move=1 bit (its 12-entry cant vocabulary had no "focuspunch", and the
# [hit,miss,fail] outcome was suppressed on cant turns). The "why" was lost.
#
# Here the reason and the prevented move survive verbatim — no fixed vocabulary.
# ───────────────────────────────────────────────────────────────────────────
def test_focus_punch_disrupted_keeps_the_reason():
    # Blaziken (our lead) vs Tyranitar — switched in by full species name so the
    # protocol identifier resolves.
    b = Gen3Battle("battle-gen3ou-fp", "p1user", LOG, gen=3)
    for line in [
        ["", "player", "p1", "p1user", "", ""],
        ["", "player", "p2", "p2user", "", ""],
        ["", "teamsize", "p1", "6"],
        ["", "teamsize", "p2", "6"],
        ["", "gen", "3"],
        ["", "start"],
        ["", "switch", "p1a: Blaziken", "Blaziken, L100, M", "100/100"],
        ["", "switch", "p2a: Ttar", "Tyranitar, L100, M", "100/100"],
        ["", "turn", "1"],
        # our Blaziken declares Focus Punch (charging single-turn marker)
        ["", "-singleturn", "p1a: Blaziken", "move: Focus Punch"],
        # opponent hits it first
        ["", "move", "p2a: Ttar", "Rock Slide", "p1a: Blaziken"],
        ["", "-supereffective", "p1a: Blaziken"],
        ["", "-damage", "p1a: Blaziken", "55/100"],
        # ...so Focus Punch can't fire — Showdown's exact disruption line:
        ["", "cant", "p1a: Blaziken", "Focus Punch", "Focus Punch"],
        ["", "turn", "2"],
    ]:
        b.parse_message(line)
    v = TurnView.for_turn(b, 1)

    # We were prevented from acting — and the WHY is preserved, free-form.
    assert v.ours.moved is False
    assert v.ours.failed_to_move is True
    assert v.ours.cant_reason == "Focus Punch"     # the reason, verbatim
    assert v.ours.cant_move == "focuspunch"        # the move that was prevented
    # The opponent's hit is fully readable too.
    assert v.opp.move_id == "rockslide"
    assert v.opp.effectiveness == 2.0
    assert v.damage_on("blaziken", side="ours") < 0.0

    # And nothing about this is collapsed: the raw |cant| line is on the event.
    cant = next(e for e in b.events_for_turn(1) if e.kind is EventKind.CANT)
    assert cant.raw == ("", "cant", "p1a: Blaziken", "Focus Punch", "Focus Punch")


# ───────────────────────────────────────────────────────────────────────────
# THE TWO SURFACES: history vs current board are answered separately.
#   - "what happened this turn?"  -> TurnView   (from the event log)
#   - "what is true right now?"   -> battle.live_view()  (current board only)
# A consumer never mixes them, and neither can drift from the other.
# ───────────────────────────────────────────────────────────────────────────
def test_history_and_current_board_are_separate_sources():
    b = _battle([
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Ttar"],
        ["", "-damage", "p2a: Ttar", "40/100"],
        ["", "-status", "p2a: Ttar", "par"],
        ["", "move", "p2a: Ttar", "Dragon Dance", "p2a: Ttar"],
        ["", "-boost", "p2a: Ttar", "atk", "1"],
        ["", "-boost", "p2a: Ttar", "spe", "1"],
    ])

    # HISTORY: what happened on turn 1 (from the event log)
    v = TurnView.for_turn(b, 1)
    assert v.both_attacked is True
    assert v.ours.move_id == "thunderbolt"
    assert v.opp.status_applied == "par"        # they GOT paralysed this turn
    assert v.opp.boosts == {"atk": 1, "spe": 1}  # net stage change THIS turn

    # CURRENT BOARD: what is true now (from live_view) — no per-turn deltas, just state
    lv = b.live_view()
    opp = lv.opp.active
    assert opp.species == "tyranitar"
    assert opp.hp_fraction == pytest.approx(0.40)
    assert opp.status == "par"                  # currently paralysed
    assert opp.boosts == {"atk": 1, "spe": 1}   # current stat stages (happen to match)

    # The current-board view exposes no history at all — to learn "what move they
    # used", you MUST go to the event log. This is the boundary, enforced by absence.
    assert not hasattr(opp, "last_move")
