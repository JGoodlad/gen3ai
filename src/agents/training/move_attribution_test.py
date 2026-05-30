"""Unit tests for protocol-accurate per-side move attribution in TurnDelta.

Covers the logic that decides — for OUR side and the OPPONENT, symmetrically —
what a side actually did on a turn, sourced from the Showdown protocol rather
than the (desync-prone) action index:

  * our_move_id / opp_move_id        — the move that FIRED (delegation-aware),
                                        or None when nothing happened.
  * our/opp_move_outcome             — hit / miss / fail / None.
  * our/opp_cant_reason == "fainted" — KO'd before acting (vs |cant| vs switch).
  * effectiveness / damaging_event   — dropped to None when they disagree with
                                        the move that fired (turn-gate lag guard).

These were all found to be subtly wrong at various points; the e2e
move_outcome / transition fuzz tests validate them against the live protocol at
scale, and these unit tests pin the exact decision table.
"""
import numpy as np
import pytest

from poke_env.battle.abstract_battle import DamagingMoveEvent
from agents.gen3_mechanics import BOOST_DIM
from agents.training.battle_context import (
    BattleContext, TurnDelta,
    _derive_move_outcome, _ko_before_acting, _align_effectiveness, _moves_match,
)

SW0, SW5, MOVE0, MOVE3, STRUGGLE = 0, 5, 6, 9, 10


def _event(move_id, *, user="tyranitar", target="salamence", eff=2.0):
    return DamagingMoveEvent(user_species=user, target_species=target,
                             target_status=None, move_id=move_id, effectiveness=eff)


def _ctx(**ov):
    d = dict(
        turn=1, phase="move_selection", mask=np.zeros(11, dtype=np.int8),
        our_slot_map={"tyranitar": 0}, opp_slot_map={"salamence": 0},
        our_hp=np.zeros(6, dtype=np.float32), opp_hp=np.zeros(6, dtype=np.float32),
        our_active="tyranitar", opp_active="salamence",
        our_fainted_count=0, opp_fainted_count=0,
        active_move_ids=[None, None, None, None],
        opp_last_move_id=None, opp_all_last_move_ids={},
        opp_active_revealed_moves=frozenset(),
        our_cant_reason=None, opp_cant_reason=None,
        our_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
        opp_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
        our_last_effectiveness=None, opp_last_effectiveness=None,
        we_moved_first=None, our_team_order=("tyranitar", "skarmory"),
    )
    d.update(ov)
    return BattleContext(**d)


# ===========================================================================
# Pure helpers
# ===========================================================================

class TestDeriveMoveOutcome:
    def test_suppressed_is_none(self):
        assert _derive_move_outcome(True, False, False, suppressed=True) is None

    def test_no_move_used_is_none(self):
        # A switch turn with a STALE missed/failed flag must NOT read as miss/fail.
        assert _derive_move_outcome(False, missed=True, failed=False, suppressed=False) is None
        assert _derive_move_outcome(False, missed=False, failed=True, suppressed=False) is None

    def test_connected_overrides_stale_miss(self):
        # Explosion dealt damage (connected) but a stale miss flag is set -> hit.
        assert _derive_move_outcome(True, missed=True, failed=False,
                                    suppressed=False, connected=True) == "hit"

    def test_miss_fail_hit(self):
        assert _derive_move_outcome(True, missed=True, failed=False, suppressed=False) == "miss"
        assert _derive_move_outcome(True, missed=False, failed=True, suppressed=False) == "fail"
        assert _derive_move_outcome(True, missed=False, failed=False, suppressed=False) == "hit"


class TestKoBeforeActing:
    def _kwargs(self, **ov):
        d = dict(fainted=True, switched_voluntarily=False, move_resolved=False,
                 other_side_moved_first=True, cant_reason=None)
        d.update(ov)
        return d

    def test_fires_on_ko_before_acting(self):
        assert _ko_before_acting(**self._kwargs()) is True

    def test_not_fired_if_not_fainted(self):
        assert _ko_before_acting(**self._kwargs(fainted=False)) is False

    def test_not_fired_if_voluntary_switch(self):
        assert _ko_before_acting(**self._kwargs(switched_voluntarily=True)) is False

    def test_not_fired_if_move_resolved(self):
        # Moved (and landed) THEN fainted, e.g. Explosion -> not "nothing happened".
        assert _ko_before_acting(**self._kwargs(move_resolved=True)) is False

    def test_not_fired_if_we_were_not_KOd_first(self):
        assert _ko_before_acting(**self._kwargs(other_side_moved_first=False)) is False
        assert _ko_before_acting(**self._kwargs(other_side_moved_first=None)) is False

    def test_not_fired_if_cant(self):
        assert _ko_before_acting(**self._kwargs(cant_reason="slp")) is False


class TestAlignEffectiveness:
    def test_keeps_when_move_matches_event(self):
        ev = _event("earthquake")
        eff, event = _align_effectiveness("earthquake", 2.0, ev)
        assert eff == 2.0 and event is ev

    def test_nulls_when_event_disagrees(self):
        ev = _event("firepunch")
        eff, event = _align_effectiveness("bodyslam", 2.0, ev)
        assert eff is None and event is None

    def test_keeps_when_no_event_or_no_move(self):
        assert _align_effectiveness("earthquake", 1.0, None) == (1.0, None)
        assert _align_effectiveness(None, 1.0, _event("eq"))[0] == 1.0

    def test_hidden_power_variants_match(self):
        ev = _event("hiddenpower")
        eff, event = _align_effectiveness("hiddenpowerice", 4.0, ev)
        assert eff == 4.0 and event is ev


def test_moves_match_hidden_power():
    assert _moves_match("hiddenpower", "hiddenpowerfire")
    assert not _moves_match("surf", "icebeam")


# ===========================================================================
# TurnDelta.build — our side
# ===========================================================================

def test_stayed_in_uses_protocol_last_move_not_action():
    # Agent's action index says slot 0 (rockslide), but the protocol last_move
    # says earthquake actually fired -> trust the protocol.
    prev = _ctx(active_move_ids=["rockslide", "earthquake", "crunch", "fireblast"])
    curr = _ctx(our_last_move_id="earthquake")
    d = TurnDelta.build(prev, curr, MOVE0)
    assert d.our_move_id == "earthquake"
    assert d.our_move_outcome == "hit"


def test_delegated_move_is_first_class():
    # Sleep Talk (action) delegated to Surf — last_move stores the CALLED move.
    prev = _ctx(active_move_ids=["sleeptalk", "rest", "surf", "icebeam"])
    curr = _ctx(our_last_move_id="surf")
    d = TurnDelta.build(prev, curr, MOVE0)
    assert d.our_move_id == "surf"


def test_attacked_then_fainted_uses_damaging_event():
    # We moved first, Explosion dealt damage, then we self-fainted. active_pokemon
    # now reads the replacement, so last_move is wrong; the event names the move.
    prev = _ctx(active_move_ids=["meteormash", "earthquake", "explosion", "brickbreak"])
    curr = _ctx(our_active="NONE", our_fainted_count=1, our_last_move_id="brickbreak",
                our_last_damaging_event=_event("explosion"), we_moved_first=True,
                our_move_missed=True)  # stale miss flag — must be overridden by "connected"
    d = TurnDelta.build(prev, curr, MOVE0)
    assert d.our_move_id == "explosion"
    assert d.our_switch_to is None
    assert d.our_move_outcome == "hit"   # dealt damage -> hit, NOT miss


def test_neutral_explosion_no_event_is_hit():
    # A NEUTRAL Explosion emits no effectiveness event, so there is no damaging
    # event to flag "connected" — but Self-Destruct/Explosion always land when
    # used. A stale miss/fail flag must not make this read as miss/fail; SELF_KO
    # moves are treated as connected -> hit.
    prev = _ctx(active_move_ids=["meteormash", "earthquake", "explosion", "brickbreak"])
    curr = _ctx(our_active="NONE", our_fainted_count=1, our_last_move_id="brickbreak",
                our_last_damaging_event=None,          # neutral hit -> no event
                we_moved_first=True, our_move_failed=True)  # stale fail flag
    d = TurnDelta.build(prev, curr, MOVE0 + 2)  # action 2 -> explosion slot
    assert d.our_move_id == "explosion"
    assert d.our_move_outcome == "hit"


def test_opp_neutral_explosion_no_event_is_hit():
    # Symmetric: opp neutral Explosion with a stale fail flag -> hit, not fail.
    prev = _ctx(opp_active="gengar")
    curr = _ctx(opp_active="claydol", opp_fainted_count=1,
                opp_all_last_move_ids={"gengar": "explosion"},
                opp_last_damaging_event=None,          # neutral hit -> no event
                we_moved_first=False, opp_move_failed=True)  # stale fail flag
    d = TurnDelta.build(prev, curr, MOVE0)
    assert d.opp_move_id == "explosion"
    assert d.opp_move_outcome == "hit"


def test_ko_before_acting_is_nothing_happened():
    # Opponent moved first and KO'd us before our move fired.
    prev = _ctx(active_move_ids=["thunderbolt", "icebeam", "rest", "sleeptalk"])
    curr = _ctx(our_active="NONE", our_fainted_count=1, we_moved_first=False,
                our_last_damaging_event=None)
    d = TurnDelta.build(prev, curr, MOVE0)
    assert d.our_move_id is None
    assert d.our_move_outcome is None
    assert d.our_cant_reason == "fainted"
    assert d.our_failed_to_move is True


def test_switch_in_death_is_a_switch_no_move():
    # We switched a mon in; it fainted (e.g. to Spikes). A stale miss flag must
    # NOT make this read as a missed move.
    prev = _ctx(active_move_ids=["a", "b", "c", "d"], our_team_order=("tyranitar", "gengar"))
    curr = _ctx(our_active="NONE", our_fainted_count=1, our_move_missed=True)
    d = TurnDelta.build(prev, curr, SW5 - 4)  # action 1 -> switch to team slot 1 (gengar)
    assert d.our_switch_to == "gengar"
    assert d.our_move_id is None
    assert d.our_move_outcome is None


def test_effectiveness_dropped_when_event_disagrees():
    prev = _ctx(active_move_ids=["surf", "icebeam", "rest", "sleeptalk"])
    curr = _ctx(our_last_move_id="icebeam", our_last_effectiveness=2.0,
                our_last_damaging_event=_event("surf"))  # stale event (different move)
    d = TurnDelta.build(prev, curr, MOVE0 + 1)
    assert d.our_move_id == "icebeam"
    assert d.our_effectiveness is None
    assert d.our_damaging_event is None


def test_cant_turn_outcome_is_none():
    prev = _ctx(active_move_ids=["thunderbolt", "x", "y", "z"])
    curr = _ctx(our_cant_reason="slp", our_last_move_id=None)
    d = TurnDelta.build(prev, curr, MOVE0)
    assert d.our_move_outcome is None
    assert d.our_cant_reason == "slp"


# ===========================================================================
# TurnDelta.build — opponent side (symmetric)
# ===========================================================================

def test_opp_ko_before_acting_is_nothing_happened():
    # WE moved first and KO'd the opp before it acted. The faint-recovery would
    # otherwise pull a stale prior move from opp_all_last_move_ids.
    prev = _ctx(opp_active="salamence")
    curr = _ctx(opp_active="metagross",          # opp's forced replacement
                opp_fainted_count=1,
                opp_all_last_move_ids={"salamence": "dragonclaw"},  # STALE prior move
                we_moved_first=True,
                opp_last_damaging_event=None)
    d = TurnDelta.build(prev, curr, MOVE0)
    assert d.opp_move_id is None
    assert d.opp_move_outcome is None
    assert d.opp_cant_reason == "fainted"


def test_opp_attacked_then_fainted_keeps_move():
    # Opp moved (Explosion) and dealt damage, then fainted. The move DID land.
    prev = _ctx(opp_active="gengar")
    curr = _ctx(opp_active="claydol", opp_fainted_count=1,
                opp_all_last_move_ids={"gengar": "explosion"},
                opp_last_damaging_event=_event("explosion", user="gengar", target="tyranitar"),
                we_moved_first=False)
    d = TurnDelta.build(prev, curr, MOVE0)
    assert d.opp_move_id == "explosion"
    assert d.opp_move_outcome == "hit"


def test_opp_effectiveness_dropped_when_event_disagrees():
    prev = _ctx(opp_active="blissey")
    curr = _ctx(opp_active="blissey", opp_last_move_id="icebeam",
                opp_last_effectiveness=1.0,
                opp_last_damaging_event=_event("seismictoss", user="blissey"))
    d = TurnDelta.build(prev, curr, MOVE0)
    assert d.opp_effectiveness is None
    assert d.opp_damaging_event is None
