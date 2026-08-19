"""Scripted edge-case parse tests for Gen3Battle (design §7.6 engineered edges).

These feed *real* protocol line sequences through Gen3Battle offline (no server) and
assert the event log + TurnView fold for the gnarly cases the old diff-based TurnDelta
needed heuristics for: phaze (roared out), self-KO (Explosion), hazard switch-in death
(Spikes), delegated moves (Sleep Talk), Taunt-induced |cant|, Knock Off item removal,
and a move that |-fail|s. Each is the kind of case that silently corrupted snapshot
diffing; here we prove the event log gets them right.
"""

import logging

from agents.battle.battle_event import EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.turn_view import TurnView

LOG = logging.getLogger("gen3battle-edge-test")

_PREAMBLE = [
    ["", "player", "p1", "p1user", "", ""],
    ["", "player", "p2", "p2user", "", ""],
    ["", "teamsize", "p1", "6"],
    ["", "teamsize", "p2", "6"],
    ["", "gametype", "singles"],
    ["", "gen", "3"],
    ["", "start"],
]


def run(lines):
    b = Gen3Battle("battle-gen3ou-edge", "p1user", LOG, gen=3)
    for line in _PREAMBLE + lines:
        b.parse_message(line)
    b.assert_conservation()  # every scenario must stay balanced
    return b


# --------------------------------------------------------------------------- #
# 1. Phaze — our active is roared out (|drag|).                                 #
# --------------------------------------------------------------------------- #
def test_phaze_roared_out():
    b = run([
        ["", "switch", "p1a: Ttar", "Tyranitar, L100, M", "100/100"],
        ["", "switch", "p2a: Skarm", "Skarmory, L100, F", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p2a: Skarm", "Whirlwind", "p1a: Ttar"],
        ["", "drag", "p1a: Bliss", "Blissey, L100, F", "100/100"],
        ["", "turn", "2"],
    ])
    v = TurnView.for_turn(b, 1)
    # The phazer's move is captured at |move| time (not lost to the drag), and our
    # side is marked switched-by-force.
    assert v.opp.move_id == "whirlwind"
    assert v.ours.drag is True
    assert v.ours.switched is True
    assert v.ours.switched_to == "blissey"
    assert v.ours.moved is False
    drag = next(e for e in b.events_for_turn(1) if e.kind is EventKind.DRAG)
    assert drag.side == "ours"
    assert drag.actor_species == "blissey"          # the mon dragged IN
    # gen3_event_value_schema_v1: the DRAG payload is EMPTY — `prev_active`/`details` were
    # deleted (unread by anything; the outgoing mon is `v.ours.switched_to`'s counterpart in
    # the fold's own snapshot, and the details string is verbatim in `raw`).
    assert drag.value == {}
    assert "Blissey, L100, F" in drag.raw


# --------------------------------------------------------------------------- #
# 2. Self-KO — Explosion connects, then the user faints.                        #
# --------------------------------------------------------------------------- #
def test_explosion_self_ko():
    b = run([
        ["", "switch", "p1a: Forry", "Forretress, L100, M", "100/100"],
        ["", "switch", "p2a: Bliss", "Blissey, L100, F", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p1a: Forry", "Explosion", "p2a: Bliss"],
        ["", "-damage", "p2a: Bliss", "0 fnt"],
        ["", "faint", "p2a: Bliss"],
        ["", "faint", "p1a: Forry"],
        ["", "turn", "2"],
    ])
    v = TurnView.for_turn(b, 1)
    assert v.ours.move_id == "explosion"
    assert v.ours.fainted is True          # the user self-KO'd
    assert v.opp.fainted is True           # the target was KO'd
    # Explosion is a damaging move even though a neutral hit emits no effectiveness:
    # we recover "it connected" from the HP it dealt.
    assert v.ours.damaging_move is not None
    assert v.ours.damaging_move.move_id == "explosion"
    assert v.damage_on("blissey") < 0.0
    assert set(v.faints()) == {"blissey", "forretress"}


# --------------------------------------------------------------------------- #
# 3. Hazard switch-in death — Spikes KO a freshly-switched mon.                 #
# --------------------------------------------------------------------------- #
def test_spikes_switch_in_death():
    b = run([
        ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
        ["", "switch", "p2a: Starmie", "Starmie, L100", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p1a: Skarm", "Spikes", "p2a: Starmie"],
        ["", "-sidestart", "p2: p2user", "Spikes"],
        ["", "move", "p2a: Starmie", "Rapid Spin", "p1a: Skarm"],
        ["", "turn", "2"],
        # opponent sends in a frail mon that dies to Spikes on entry
        ["", "switch", "p2a: Frail", "Diglett, L100, M", "8/100"],
        ["", "-damage", "p2a: Frail", "0 fnt", "[from] Spikes"],
        ["", "faint", "p2a: Frail"],
        ["", "turn", "3"],
    ])
    v = TurnView.for_turn(b, 2)
    assert v.opp.switched is True
    assert v.opp.switched_to == "diglett"
    assert v.opp.fainted is True
    assert v.opp.drag is False             # voluntary switch, not phazed
    assert v.damage_on("diglett") < 0.0
    dmg = next(
        e for e in b.events_for_turn(2)
        if e.kind is EventKind.DAMAGE and e.actor_species == "diglett"
    )
    assert dmg.value.get("reason") == "Spikes"


# --------------------------------------------------------------------------- #
# 4. Delegated move — Sleep Talk calls Earthquake.                              #
# --------------------------------------------------------------------------- #
def test_sleep_talk_delegation():
    b = run([
        ["", "switch", "p1a: Snorlax", "Snorlax, L100, M", "100/100"],
        ["", "switch", "p2a: Ttar", "Tyranitar, L100, M", "100/100"],
        ["", "turn", "1"],
        ["", "-status", "p1a: Snorlax", "slp"],
        ["", "move", "p1a: Snorlax", "Sleep Talk", "p1a: Snorlax"],
        # The bundled gen3 sim emits the called move as a BARE `[from] <MoveName>` (no `move:`
        # prefix) — the REAL wire form. This was previously written as the modern `[from]move:`
        # form the gen3 sim never produces, which hid a from_move-extraction gap in _delegated_from.
        ["", "move", "p1a: Snorlax", "Earthquake", "p2a: Ttar", "[from] Sleep Talk"],
        ["", "-damage", "p2a: Ttar", "60/100"],
        ["", "turn", "2"],
    ])
    v = TurnView.for_turn(b, 1)
    # The EXECUTED move is Earthquake, attributed via Sleep Talk.
    assert v.ours.move_id == "earthquake"
    assert v.ours.called_via == "sleeptalk"
    assert v.ours.outcome == "hit"
    assert v.damage_on("tyranitar") < 0.0
    # both MOVE lines are recorded (the delegator + the called move), each by its
    # own protocol-named id
    move_ids = sorted(
        e.move_id for e in b.events_for_turn(1) if e.kind is EventKind.MOVE
    )
    assert move_ids == ["earthquake", "sleeptalk"]


# --------------------------------------------------------------------------- #
# 5. Taunt → |cant| (the move is prevented, not "failed").                      #
# --------------------------------------------------------------------------- #
def test_taunt_causes_cant():
    b = run([
        ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
        ["", "switch", "p2a: Snorlax", "Snorlax, L100, M", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p1a: Skarm", "Taunt", "p2a: Snorlax"],
        ["", "-start", "p2a: Snorlax", "move: Taunt"],
        ["", "turn", "2"],
        ["", "cant", "p2a: Snorlax", "move: Taunt", "Curse"],
        ["", "move", "p1a: Skarm", "Spikes", "p2a: Snorlax"],
        ["", "-sidestart", "p2: p2user", "Spikes"],
        ["", "turn", "3"],
    ])
    v = TurnView.for_turn(b, 2)
    assert v.opp.cant_reason == "move: Taunt"
    assert v.opp.failed_to_move is True
    assert v.opp.moved is False
    assert v.opp.outcome is None
    cant = next(e for e in b.events_for_turn(2) if e.kind is EventKind.CANT)
    assert cant.value["reason"] == "move: Taunt"
    assert cant.value["move"] == "curse"


# --------------------------------------------------------------------------- #
# 6. Knock Off — item removed, carrying [from]/[of] attribution.                #
# --------------------------------------------------------------------------- #
def test_knock_off_item_removal():
    b = run([
        ["", "switch", "p1a: Ttar", "Tyranitar, L100, M", "100/100"],
        ["", "switch", "p2a: Bliss", "Blissey, L100, F", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p1a: Ttar", "Knock Off", "p2a: Bliss"],
        ["", "-damage", "p2a: Bliss", "92/100"],
        ["", "-enditem", "p2a: Bliss", "Leftovers", "[from] move: Knock Off",
         "[of] p1a: Ttar"],
        ["", "turn", "2"],
    ])
    end = next(e for e in b.events_for_turn(1) if e.kind is EventKind.ENDITEM)
    assert end.side == "opp"
    assert end.actor_species == "blissey"
    assert end.value["item"] == "leftovers"
    assert end.value.get("from") == "move: Knock Off"
    assert end.value.get("of") == "p1a: Ttar"


# --------------------------------------------------------------------------- #
# 7. A move that |-fail|s (Substitute onto an existing sub).                     #
# --------------------------------------------------------------------------- #
def test_move_fails():
    b = run([
        ["", "switch", "p1a: Snorlax", "Snorlax, L100, M", "100/100"],
        ["", "switch", "p2a: Ttar", "Tyranitar, L100, M", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p1a: Snorlax", "Substitute", "p1a: Snorlax"],
        ["", "-fail", "p1a: Snorlax", "move: Substitute"],
        ["", "turn", "2"],
    ])
    v = TurnView.for_turn(b, 1)
    assert v.ours.moved is True
    assert v.ours.failed is True
    assert v.ours.outcome == "fail"
    fail = next(e for e in b.events_for_turn(1) if e.kind is EventKind.FAIL)
    assert fail.side == "ours"
    assert fail.actor_species == "snorlax"
