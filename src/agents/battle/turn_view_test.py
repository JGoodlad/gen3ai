"""TurnView fold tests (design §8.1) — pure, on hand-built event logs.

These are the canonical per-turn cases the old diff-based ``TurnDelta`` needed
heuristics for (KO-before-acting, phaze, Sleep Talk delegation, switch-death). With
the event log they become a straight fold, asserted here without a server or a battle.
"""

import itertools

import pytest

from agents.battle.battle_event import OPP, OURS, BattleEvent, EventKind
from agents.battle.turn_view import TurnView

_seq = itertools.count()


def ev(kind, *, side=None, actor=None, target=None, turn=1, **value):
    return BattleEvent(
        seq=next(_seq),
        turn=turn,
        kind=kind,
        side=side,
        actor_species=actor,
        target_species=target,
        value=value,
    )


def test_normal_trade_fold():
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar",
           move_id="thunderbolt", target_status=None),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.45),
        ev(EventKind.MOVE, side=OPP, actor="tyranitar", target="zapdos",
           move_id="rockslide", target_status=None),
        ev(EventKind.SUPEREFFECTIVE, side=OPP, actor="tyranitar", target="zapdos",
           multiplier=2.0),
        ev(EventKind.DAMAGE, side=OURS, actor="zapdos", amount=-0.7),
    ]
    v = TurnView.from_events(events)
    assert v.move_order == [OURS, OPP]
    assert v.we_moved_first is True
    assert v.ours.move_id == "thunderbolt"
    assert v.ours.outcome == "hit"
    assert v.opp.move_id == "rockslide"
    assert v.opp.effectiveness == 2.0
    assert v.opp.damaging_move.effectiveness == 2.0
    assert v.opp.damaging_move.target_species == "zapdos"
    # net HP change on each target
    assert v.damage_on("tyranitar") == pytest.approx(-0.45)
    assert v.damage_on("zapdos") == pytest.approx(-0.7)


def test_switch_means_no_move_and_unknown_order():
    events = [
        ev(EventKind.SWITCH, side=OURS, actor="skarmory", prev_active="zapdos"),
        ev(EventKind.MOVE, side=OPP, actor="tyranitar", target="skarmory",
           move_id="earthquake"),
        ev(EventKind.DAMAGE, side=OURS, actor="skarmory", amount=-0.2),
    ]
    v = TurnView.from_events(events)
    assert v.ours.switched is True
    assert v.ours.switched_to == "skarmory"
    assert v.ours.moved is False
    assert v.ours.move_id is None
    assert v.ours.outcome is None
    assert v.we_moved_first is None  # only one side used a move


def test_cant_sets_reason_and_no_outcome():
    events = [
        ev(EventKind.CANT, side=OURS, actor="snorlax", reason="slp"),
        ev(EventKind.MOVE, side=OPP, actor="gengar", target="snorlax",
           move_id="shadowball"),
        ev(EventKind.DAMAGE, side=OURS, actor="snorlax", amount=-0.3),
    ]
    v = TurnView.from_events(events)
    assert v.ours.cant_reason == "slp"
    assert v.ours.failed_to_move is True
    assert v.ours.moved is False
    assert v.ours.outcome is None


def test_miss_and_fail_outcomes():
    miss = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="aerodactyl", target="skarmory",
           move_id="rockslide"),
        ev(EventKind.MISS, side=OURS, actor="aerodactyl", target="skarmory", op="miss"),
    ])
    assert miss.ours.missed is True
    assert miss.ours.outcome == "miss"

    fail = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="snorlax", target="snorlax",
           move_id="bellydrum"),
        ev(EventKind.FAIL, side=OURS, actor="snorlax", op="fail"),
    ])
    assert fail.ours.failed is True
    assert fail.ours.outcome == "fail"


def test_crit_is_orthogonal_to_hit():
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OPP, actor="tyranitar", target="zapdos",
           move_id="rockslide"),
        ev(EventKind.CRIT, side=OPP, actor="tyranitar", target="zapdos", op="crit"),
        ev(EventKind.DAMAGE, side=OURS, actor="zapdos", amount=-0.9),
    ])
    assert v.opp.crit is True
    assert v.opp.outcome == "hit"


def test_faint_detected_per_side():
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar",
           move_id="thunderbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-1.0),
        ev(EventKind.FAINT, side=OPP, actor="tyranitar"),
    ])
    assert v.opp.fainted is True
    assert v.ours.fainted is False
    assert v.faints() == ["tyranitar"]


def test_ko_before_acting_opp_never_moves():
    # We move first and KO; the opponent faints before it can act -> no opp move,
    # order has only our side, we_moved_first is None (no contest), opp.outcome None.
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="aerodactyl", target="gengar",
           move_id="rockslide"),
        ev(EventKind.DAMAGE, side=OPP, actor="gengar", amount=-1.0),
        ev(EventKind.FAINT, side=OPP, actor="gengar"),
    ])
    assert v.ours.move_id == "rockslide"
    assert v.opp.moved is False
    assert v.opp.outcome is None
    assert v.opp.fainted is True
    assert v.we_moved_first is None


def test_sleep_talk_delegation_picks_called_move():
    # |move|Snorlax|Sleep Talk then |move|Snorlax|Body Slam|[from]move: Sleep Talk
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="snorlax", move_id="sleeptalk"),
        ev(EventKind.MOVE, side=OURS, actor="snorlax", target="gengar",
           move_id="bodyslam", from_move="sleeptalk"),
        ev(EventKind.DAMAGE, side=OPP, actor="gengar", amount=-0.5),
    ])
    assert v.ours.move_id == "bodyslam"
    assert v.ours.called_via == "sleeptalk"
    assert v.ours.outcome == "hit"


def test_phaze_drag_is_a_switch_marked_forced():
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="skarmory", target="tyranitar",
           move_id="whirlwind"),
        ev(EventKind.DRAG, side=OPP, actor="blissey", prev_active="tyranitar"),
    ])
    assert v.opp.switched is True
    assert v.opp.drag is True
    assert v.opp.switched_to == "blissey"
    assert v.ours.move_id == "whirlwind"


def test_boosts_accumulate_net_change():
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="dragonite", move_id="dragondance"),
        ev(EventKind.BOOST, side=OURS, actor="dragonite", stat="atk", amount=1),
        ev(EventKind.BOOST, side=OURS, actor="dragonite", stat="spe", amount=1),
        ev(EventKind.UNBOOST, side=OPP, actor="skarmory", stat="atk", amount=-1),
    ])
    assert v.ours.boosts == {"atk": 1, "spe": 1}
    assert v.opp.boosts == {"atk": -1}


def test_empty_turn_is_safe():
    v = TurnView.from_events([])
    assert v.move_order == []
    assert v.we_moved_first is None
    assert v.ours.moved is False
    assert v.damage_on(None) == 0.0


def test_damage_on_disambiguates_mirror_species():
    # Both sides field a Tyranitar. damage_on without a side sums both; with a side
    # it isolates the one that was hit.
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="tyranitar", target="tyranitar",
           move_id="earthquake"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.4),
        ev(EventKind.MOVE, side=OPP, actor="tyranitar", target="tyranitar",
           move_id="rockslide"),
        ev(EventKind.DAMAGE, side=OURS, actor="tyranitar", amount=-0.3),
    ])
    assert v.damage_on("tyranitar") == pytest.approx(-0.7)           # both
    assert v.damage_on("tyranitar", side=OPP) == pytest.approx(-0.4)  # the one we hit
    assert v.damage_on("tyranitar", side=OURS) == pytest.approx(-0.3)
    # damaging_move resolves to the correct side's target in a mirror
    assert v.ours.damaging_move.effectiveness == 1.0  # dealt damage, no eff disclosed


def test_multi_hit_damage_sums():
    # A multi-hit move (e.g. Rock Blast) emits several -damage lines on one target;
    # damage_on sums them and the move is recognised as damaging.
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="cloyster", target="blissey",
           move_id="rockblast"),
        ev(EventKind.DAMAGE, side=OPP, actor="blissey", amount=-0.12),
        ev(EventKind.DAMAGE, side=OPP, actor="blissey", amount=-0.11),
        ev(EventKind.DAMAGE, side=OPP, actor="blissey", amount=-0.13),
    ])
    assert v.damage_on("blissey", side=OPP) == pytest.approx(-0.36)
    assert v.ours.damaging_move is not None
    assert v.ours.outcome == "hit"
