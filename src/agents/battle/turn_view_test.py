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


# ---------------------------------------------------------------------------
# faint_details() — cause classification (Step 4)
# ---------------------------------------------------------------------------

from agents.battle.turn_view import FAINT_CAUSE_VOCAB, FaintDetail  # noqa: E402


def test_faint_cause_attack():
    """A KO with no [from] clause → 'attack'."""
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OPP, actor="tyranitar", target="zapdos",
           move_id="rockslide"),
        ev(EventKind.DAMAGE, side=OURS, actor="zapdos", amount=-1.0),  # no reason
        ev(EventKind.FAINT, side=OURS, actor="zapdos"),
    ])
    details = v.faint_details()
    assert len(details) == 1
    assert details[0] == FaintDetail(species="zapdos", side=OURS, cause="attack")


def test_faint_cause_hazard_spikes():
    """Spikes damage → 'hazard'."""
    v = TurnView.from_events([
        ev(EventKind.DAMAGE, side=OURS, actor="raichu", amount=-0.25, reason="Spikes"),
        ev(EventKind.FAINT, side=OURS, actor="raichu"),
    ])
    details = v.faint_details()
    assert len(details) == 1
    assert details[0].cause == "hazard"


def test_faint_cause_weather_sandstorm():
    """Sandstorm residual → 'weather'."""
    v = TurnView.from_events([
        ev(EventKind.DAMAGE, side=OPP, actor="blissey", amount=-0.0625, reason="Sandstorm"),
        ev(EventKind.FAINT, side=OPP, actor="blissey"),
    ])
    assert v.faint_details()[0].cause == "weather"


def test_faint_cause_status_burn():
    """Burn residual → 'status'."""
    v = TurnView.from_events([
        ev(EventKind.DAMAGE, side=OPP, actor="machamp", amount=-0.0625, reason="brn"),
        ev(EventKind.FAINT, side=OPP, actor="machamp"),
    ])
    assert v.faint_details()[0].cause == "status"


def test_faint_cause_recoil():
    """Recoil damage → 'recoil'."""
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="dodrio", target="steelix", move_id="doubleedge"),
        ev(EventKind.DAMAGE, side=OURS, actor="dodrio", amount=-0.25, reason="Recoil"),
        ev(EventKind.FAINT, side=OURS, actor="dodrio"),
    ])
    assert v.faint_details()[0].cause == "recoil"


def test_faint_cause_selfko_explosion():
    """Explosion user faint → 'selfko' (overrides the no-[from] attack rule)."""
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="forretress", target="blissey",
           move_id="explosion"),
        ev(EventKind.DAMAGE, side=OPP, actor="blissey", amount=-1.0),
        ev(EventKind.FAINT, side=OURS, actor="forretress"),  # user faint = selfko
        ev(EventKind.FAINT, side=OPP, actor="blissey"),      # target faint = attack
    ])
    details = v.faint_details()
    assert len(details) == 2
    # The user faint on our side is selfko
    our_faint = next(d for d in details if d.side == OURS)
    opp_faint = next(d for d in details if d.side == OPP)
    assert our_faint.cause == "selfko"
    # The target faint (no [from] on their damage) is attack
    assert opp_faint.cause == "attack"


def test_faint_cause_leechseed():
    """Leech Seed residual → 'leechseed'."""
    v = TurnView.from_events([
        ev(EventKind.DAMAGE, side=OURS, actor="celebi", amount=-0.125, reason="Leech Seed"),
        ev(EventKind.FAINT, side=OURS, actor="celebi"),
    ])
    assert v.faint_details()[0].cause == "leechseed"


def test_faint_cause_other_unrecognised():
    """Unknown [from] clause → 'other' (no crash)."""
    v = TurnView.from_events([
        ev(EventKind.DAMAGE, side=OPP, actor="aggron", amount=-0.5, reason="Mystery Mechanic"),
        ev(EventKind.FAINT, side=OPP, actor="aggron"),
    ])
    assert v.faint_details()[0].cause == "other"


def test_faint_double_ko_different_causes():
    """Two faints in one window with different causes both appear in faint_details."""
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="exploudier", target="blissey",
           move_id="explosion"),
        ev(EventKind.DAMAGE, side=OPP, actor="blissey", amount=-1.0),        # attack
        ev(EventKind.FAINT, side=OURS, actor="exploudier"),                  # selfko
        ev(EventKind.FAINT, side=OPP, actor="blissey"),                      # attack
    ])
    details = v.faint_details()
    assert len(details) == 2
    causes = {d.cause for d in details}
    assert causes == {"selfko", "attack"}


def test_no_faints_returns_empty():
    """A normal trade (no faints) → empty faint_details list."""
    v = TurnView.from_events([
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="tbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.4),
    ])
    assert v.faint_details() == []


def test_all_faint_cause_vocab_covered():
    """Every element of FAINT_CAUSE_VOCAB must be a string."""
    assert all(isinstance(c, str) for c in FAINT_CAUSE_VOCAB)
    assert len(FAINT_CAUSE_VOCAB) == 8
