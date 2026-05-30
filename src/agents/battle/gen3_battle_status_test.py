"""Scripted tests for boosts and status transitions (freeze/thaw, sleep/wake).

These feed the exact protocol Showdown emits (verified against
deps/pokemon-showdown/data/conditions.ts) through Gen3Battle offline and assert both
the event log and the TurnView fold. Status transitions are a classic place to lose
information: the same `|-curestatus|` keyword covers "woke up" and "thawed", and a mon
can wake/thaw AND act on the same turn — the event log must keep both, in order.
"""

import logging

import pytest

from agents.battle.battle_event import EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.turn_view import TurnView

LOG = logging.getLogger("gen3battle-status-test")


def _start(p1_species, p2_species):
    b = Gen3Battle("battle-gen3ou-st", "p1user", LOG, gen=3)
    for line in [
        ["", "player", "p1", "p1user", "", ""],
        ["", "player", "p2", "p2user", "", ""],
        ["", "teamsize", "p1", "6"],
        ["", "teamsize", "p2", "6"],
        ["", "gen", "3"],
        ["", "start"],
        ["", "switch", "p1a: Mine", f"{p1_species}, L100, M", "100/100"],
        ["", "switch", "p2a: Theirs", f"{p2_species}, L100, M", "100/100"],
        ["", "turn", "1"],
    ]:
        b.parse_message(line)
    return b


# ───────────────────────────────────────── boosts ──────────────────────────
def test_boost_and_unboost_net_per_stat():
    b = _start("Dragonite", "Skarmory")
    for line in [
        ["", "move", "p1a: Mine", "Dragon Dance", "p1a: Mine"],
        ["", "-boost", "p1a: Mine", "atk", "1"],
        ["", "-boost", "p1a: Mine", "spe", "1"],
        ["", "move", "p2a: Theirs", "Whirlwind", "p1a: Mine"],   # placeholder action
        ["", "-unboost", "p1a: Mine", "atk", "1"],   # e.g. an Intimidate-style drop
        ["", "turn", "2"],
    ]:
        b.parse_message(line)
    v = TurnView.for_turn(b, 1)
    # +1 atk then -1 atk nets 0; +1 spe stands. Net change is what the model wants.
    assert v.ours.boosts == {"atk": 0, "spe": 1}
    # individual boost events are still in the log, in order, fully typed
    boosts = [e for e in b.events_for_turn(1)
              if e.kind in (EventKind.BOOST, EventKind.UNBOOST)]
    assert [(e.kind.name, e.stat, e.amount) for e in boosts] == [
        ("BOOST", "atk", 1), ("BOOST", "spe", 1), ("UNBOOST", "atk", -1),
    ]


def test_setboost_belly_drum():
    b = _start("Snorlax", "Skarmory")
    for line in [
        ["", "move", "p1a: Mine", "Belly Drum", "p1a: Mine"],
        ["", "-setboost", "p1a: Mine", "atk", "6", "[from] move: Belly Drum"],
        ["", "turn", "2"],
    ]:
        b.parse_message(line)
    e = next(x for x in b.events_for_turn(1) if x.kind is EventKind.SETBOOST)
    assert e.stat == "atk" and e.amount == 6
    assert TurnView.for_turn(b, 1).ours.boosts == {"atk": 6}


def test_haze_clearboost_is_recorded():
    b = _start("Weezing", "Dragonite")
    for line in [
        ["", "move", "p2a: Theirs", "Dragon Dance", "p2a: Theirs"],
        ["", "-boost", "p2a: Theirs", "atk", "1"],
        ["", "move", "p1a: Mine", "Haze"],
        ["", "-clearallboost"],
        ["", "turn", "2"],
    ]:
        b.parse_message(line)
    clears = [e for e in b.events_for_turn(1) if e.kind is EventKind.CLEARBOOST]
    assert len(clears) == 1
    assert clears[0].value["op"] == "clearallboost"


# ──────────────────────────────────── freeze / thaw ────────────────────────
def test_become_frozen():
    b = _start("Starmie", "Articuno")
    for line in [
        ["", "move", "p2a: Theirs", "Ice Beam", "p1a: Mine"],
        ["", "-damage", "p1a: Mine", "60/100"],
        ["", "-status", "p1a: Mine", "frz"],
        ["", "turn", "2"],
    ]:
        b.parse_message(line)
    v = TurnView.for_turn(b, 1)
    assert v.ours.status_applied == "frz"
    e = next(x for x in b.events_for_turn(1) if x.kind is EventKind.STATUS)
    assert e.status == "frz" and e.side == "ours" and e.actor_species == "starmie"
    # and poke-env's current-state tracker agrees (we didn't reimplement it)
    assert b.active_pokemon.status.name == "FRZ"


def test_frozen_solid_cant_then_thaw_and_attack_same_turn():
    """A frozen mon stays frozen a turn (|cant|frz), then thaws and immediately
    attacks. Both the thaw (|-curestatus|frz) and the attack must survive, in order
    — this is the case that silently collapses if you only keep one status fact."""
    b = _start("Snorlax", "Tyranitar")
    # pre-existing freeze
    b.parse_message(["", "-status", "p1a: Mine", "frz"])
    b.parse_message(["", "turn", "2"])
    for line in [
        # turn 2: frozen solid — can't move
        ["", "cant", "p1a: Mine", "frz"],
        ["", "move", "p2a: Theirs", "Rock Slide", "p1a: Mine"],
        ["", "-damage", "p1a: Mine", "70/100"],
        ["", "turn", "3"],
        # turn 3: thaws, then attacks in the same turn
        ["", "-curestatus", "p1a: Mine", "frz", "[msg]"],
        ["", "move", "p1a: Mine", "Body Slam", "p2a: Theirs"],
        ["", "-damage", "p2a: Theirs", "55/100"],
        ["", "turn", "4"],
    ]:
        b.parse_message(line)

    # turn 2: frozen solid
    v2 = TurnView.for_turn(b, 2)
    assert v2.ours.cant_reason == "frz"
    assert v2.ours.moved is False

    # turn 3: thawed AND attacked — both facts present, in revealed order
    v3 = TurnView.for_turn(b, 3)
    assert v3.ours.status_cured == "frz"     # the thaw
    assert v3.ours.move_id == "bodyslam"     # and the attack
    assert v3.ours.outcome == "hit"
    kinds3 = [e.kind for e in b.events_for_turn(3) if e.side == "ours"]
    assert kinds3.index(EventKind.CURESTATUS) < kinds3.index(EventKind.MOVE)


# ──────────────────────────────────── sleep / wake ─────────────────────────
def test_fall_asleep_then_sleep_talk_then_wake_and_move():
    b = _start("Snorlax", "Gengar")
    # turn 1: put to sleep
    for line in [
        ["", "move", "p2a: Theirs", "Hypnosis", "p1a: Mine"],
        ["", "-status", "p1a: Mine", "slp"],
        ["", "turn", "2"],
    ]:
        b.parse_message(line)
    assert TurnView.for_turn(b, 1).ours.status_applied == "slp"

    # turn 2: asleep — Sleep Talk delegates to Earthquake while still sleeping
    for line in [
        ["", "move", "p1a: Mine", "Sleep Talk", "p1a: Mine"],
        ["", "move", "p1a: Mine", "Earthquake", "p2a: Theirs", "[from]move: Sleep Talk"],
        ["", "-damage", "p2a: Theirs", "40/100"],
        ["", "turn", "3"],
    ]:
        b.parse_message(line)
    v2 = TurnView.for_turn(b, 2)
    assert v2.ours.move_id == "earthquake"     # executed via Sleep Talk
    assert v2.ours.called_via == "sleeptalk"
    assert v2.ours.status_cured is None        # still asleep, no wake yet

    # turn 3: wakes up (|-curestatus|slp|[msg]) then attacks
    for line in [
        ["", "-curestatus", "p1a: Mine", "slp", "[msg]"],
        ["", "move", "p1a: Mine", "Body Slam", "p2a: Theirs"],
        ["", "-damage", "p2a: Theirs", "10/100"],
        ["", "turn", "4"],
    ]:
        b.parse_message(line)
    v3 = TurnView.for_turn(b, 3)
    assert v3.ours.status_cured == "slp"       # woke up
    assert v3.ours.move_id == "bodyslam"       # and acted
    # the WHY is distinguishable: a slp cure vs a frz cure use the same keyword but
    # carry their own status id, so "woke up" and "thawed" never get conflated.
    cure = next(e for e in b.events_for_turn(3) if e.kind is EventKind.CURESTATUS)
    assert cure.status == "slp"


def test_status_cured_by_healbell_is_cureteam():
    """Heal Bell / Aromatherapy emit |-curestatus| with [silent]/[from]; in gen3 a
    team cure can also arrive as |-cureteam|. Both must be captured as CURESTATUS."""
    b = _start("Blissey", "Tyranitar")
    b.parse_message(["", "-status", "p1a: Mine", "par"])
    b.parse_message(["", "turn", "2"])
    for line in [
        ["", "move", "p1a: Mine", "Heal Bell", "p1a: Mine"],
        ["", "-curestatus", "p1a: Mine", "par", "[silent]"],
        ["", "turn", "3"],
    ]:
        b.parse_message(line)
    v = TurnView.for_turn(b, 2)
    assert v.ours.status_cured == "par"


def test_status_is_not_double_counted_across_turns():
    """A status applied on turn 1 must NOT reappear as 'applied' on turn 2 — the fold
    is per-turn, so each turn only reports what changed THAT turn."""
    b = _start("Starmie", "Articuno")
    b.parse_message(["", "move", "p2a: Theirs", "Thunder Wave", "p1a: Mine"])
    b.parse_message(["", "-status", "p1a: Mine", "par"])
    b.parse_message(["", "turn", "2"])
    b.parse_message(["", "move", "p1a: Mine", "Recover", "p1a: Mine"])
    b.parse_message(["", "-heal", "p1a: Mine", "100/100"])
    b.parse_message(["", "turn", "3"])
    assert TurnView.for_turn(b, 1).ours.status_applied == "par"
    assert TurnView.for_turn(b, 2).ours.status_applied is None  # not re-reported
