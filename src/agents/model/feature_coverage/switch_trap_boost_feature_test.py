"""
Switch, trapping, and stat-boost edge cases — captured AND reach the network.

Covers 6 edge cases for the TurnDelta turn-history block:
  1. Switch vs move  (our/opp switched bit + switch-to species embedded id)
  2. Boost deltas, every stat & sign  (our/opp boost_delta at each BOOST_STATS index)
  3. Forced-switch phase  (phase_is_forced_switch bit)
  4. Move order  (we_moved_first: True / False / None)
  5. Species block (actor/target species embedded ids)
  6. Attempted-switch rejected  (attempted_switch_rejected bit + our_attempted_switch_to species id)

Each edge case has:
  - a CAPTURE test (encode_delta → read_block / inspect the raw slot vector)
  - a NETWORK test (assert_delta_reaches_network — both policy and value heads must move)
"""
import numpy as np
import pytest

from agents.observation.turn_delta_encoder import (
    OFFSET_OUR_SWITCHED,
    OFFSET_OPP_SWITCHED,
    OFFSET_OUR_BOOST_DELTA,
    OFFSET_OPP_BOOST_DELTA,
    OFFSET_PHASE_FORCED_SWITCH,
    OFFSET_ORDER,
    ORDER_DIM,
    OFFSET_OUR_ACTOR_SPECIES,
    OFFSET_OPP_ACTOR_SPECIES,
    OFFSET_OUR_SWITCH_TO_SPEC,
    OFFSET_OPP_SWITCH_TO_SPEC,
    OFFSET_ATTEMPTED_SWITCH_REJECTED,
    OFFSET_OUR_ATTEMPTED_SWITCH_SPEC,
)
from agents.gen3_mechanics import BOOST_STATS
from agents.model.feature_coverage._support import (
    feature_model,
    make_delta,
    anchor_delta,
    encode_delta,
    assert_delta_reaches_network,
    read_block,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _switch_delta_ours(switch_to: str) -> object:
    """We switched; our_move_id is None. Slot is non-empty via our_prev_active + switched bit."""
    return make_delta(
        our_move_id=None,
        our_switch_to=switch_to,
        our_prev_active="snorlax",
        opp_move_id="tackle",
        opp_prev_active="metagross",
        # required non-optional fields
        our_hp_delta=np.zeros(6, dtype=np.float32),
        opp_hp_delta=np.zeros(6, dtype=np.float32),
        we_fainted=False,
        opp_fainted=False,
        our_failed_to_move=False,
        our_cant_reason=None,
        opp_failed_to_move=False,
        opp_cant_reason=None,
        our_boost_delta=np.zeros(7, dtype=np.int8),
        opp_boost_delta=np.zeros(7, dtype=np.int8),
        our_effectiveness=None,
        opp_effectiveness=None,
        we_moved_first=None,
        opp_switch_to=None,
        opp_move_known=True,
    )


def _switch_delta_opp(switch_to: str) -> object:
    """Opp switched; anchor with our move so the slot is non-empty."""
    return make_delta(
        our_move_id="tackle",
        our_switch_to=None,
        our_prev_active="snorlax",
        opp_move_id=None,
        opp_switch_to=switch_to,
        opp_prev_active="blissey",
        our_hp_delta=np.zeros(6, dtype=np.float32),
        opp_hp_delta=np.zeros(6, dtype=np.float32),
        we_fainted=False,
        opp_fainted=False,
        our_failed_to_move=False,
        our_cant_reason=None,
        opp_failed_to_move=False,
        opp_cant_reason=None,
        our_boost_delta=np.zeros(7, dtype=np.int8),
        opp_boost_delta=np.zeros(7, dtype=np.int8),
        our_effectiveness=None,
        opp_effectiveness=None,
        we_moved_first=None,
        opp_move_known=False,
    )


def _boost_delta(our_delta=None, opp_delta=None) -> object:
    """Anchor delta with specified boost vectors."""
    our = np.zeros(7, dtype=np.int8) if our_delta is None else np.array(our_delta, dtype=np.int8)
    opp = np.zeros(7, dtype=np.int8) if opp_delta is None else np.array(opp_delta, dtype=np.int8)
    return anchor_delta(our_boost_delta=our, opp_boost_delta=opp)


# ---------------------------------------------------------------------------
# 1. Switch vs move
# ---------------------------------------------------------------------------

class TestSwitchVsMove:
    def test_our_switched_bit_set_on_switch(self):
        enc = encode_delta(_switch_delta_ours("tyranitar"))
        assert enc[OFFSET_OUR_SWITCHED] == 1.0, "our_switched bit should be 1 on a switch"
        assert enc[OFFSET_OPP_SWITCHED] == 0.0, "opp_switched bit should be 0"

    def test_our_switched_bit_clear_on_move(self):
        enc = encode_delta(anchor_delta(our_move_id="rockslide"))
        assert enc[OFFSET_OUR_SWITCHED] == 0.0

    def test_our_switch_to_species_id_nonzero(self):
        enc = encode_delta(_switch_delta_ours("tyranitar"))
        val = enc[OFFSET_OUR_SWITCH_TO_SPEC]
        assert val != 0.0, f"our_switch_to species id should be non-zero, got {val}"

    def test_opp_switched_bit_set_on_switch(self):
        enc = encode_delta(_switch_delta_opp("salamence"))
        assert enc[OFFSET_OPP_SWITCHED] == 1.0, "opp_switched bit should be 1"
        assert enc[OFFSET_OUR_SWITCHED] == 0.0

    def test_opp_switch_to_species_id_nonzero(self):
        enc = encode_delta(_switch_delta_opp("salamence"))
        val = enc[OFFSET_OPP_SWITCH_TO_SPEC]
        assert val != 0.0, f"opp_switch_to species id should be non-zero, got {val}"

    def test_switch_to_species_a_vs_b_differ(self):
        enc_a = encode_delta(_switch_delta_ours("tyranitar"))
        enc_b = encode_delta(_switch_delta_ours("salamence"))
        assert enc_a[OFFSET_OUR_SWITCH_TO_SPEC] != enc_b[OFFSET_OUR_SWITCH_TO_SPEC], (
            "Different switch-to species should produce different embedded ids"
        )

    def test_switch_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(our_move_id="tackle"),
            _switch_delta_ours("tyranitar"),
            msg="switch vs move — both heads should move",
        )

    def test_switch_to_species_identity_distinguishable(self):
        """Switching to Tyranitar vs Salamence should differ in network output."""
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            _switch_delta_ours("tyranitar"),
            _switch_delta_ours("salamence"),
            msg="switch to species A vs B — network should distinguish",
        )


# ---------------------------------------------------------------------------
# 2. Boost deltas — every stat index, +2 and -1
# ---------------------------------------------------------------------------

class TestBoostDeltas:
    @pytest.mark.parametrize("stat_idx", range(7))
    def test_our_boost_delta_positive(self, stat_idx: int):
        delta = np.zeros(7, dtype=np.int8)
        delta[stat_idx] = 2
        enc = encode_delta(_boost_delta(our_delta=delta))
        val = enc[OFFSET_OUR_BOOST_DELTA + stat_idx]
        assert val == pytest.approx(2.0), (
            f"our_boost_delta[{stat_idx}] ({BOOST_STATS[stat_idx]}) +2 not captured; got {val}"
        )

    @pytest.mark.parametrize("stat_idx", range(7))
    def test_our_boost_delta_negative(self, stat_idx: int):
        delta = np.zeros(7, dtype=np.int8)
        delta[stat_idx] = -1
        enc = encode_delta(_boost_delta(our_delta=delta))
        val = enc[OFFSET_OUR_BOOST_DELTA + stat_idx]
        assert val == pytest.approx(-1.0), (
            f"our_boost_delta[{stat_idx}] ({BOOST_STATS[stat_idx]}) -1 not captured; got {val}"
        )

    @pytest.mark.parametrize("stat_idx", range(7))
    def test_opp_boost_delta_captured(self, stat_idx: int):
        delta = np.zeros(7, dtype=np.int8)
        delta[stat_idx] = 2
        enc = encode_delta(_boost_delta(opp_delta=delta))
        val = enc[OFFSET_OPP_BOOST_DELTA + stat_idx]
        assert val == pytest.approx(2.0), (
            f"opp_boost_delta[{stat_idx}] ({BOOST_STATS[stat_idx]}) not captured; got {val}"
        )

    def test_boost_reaches_network(self):
        """A +2 boost on atk reaches both network heads."""
        model, layout, _ = feature_model()
        no_boost = _boost_delta()
        dd_boost = np.zeros(7, dtype=np.int8)
        dd_boost[0] = 2  # atk
        dd_boost[4] = 2  # spe (Dragon Dance)
        with_boost = _boost_delta(our_delta=dd_boost)
        assert_delta_reaches_network(model, layout, no_boost, with_boost,
                                     msg="+2 atk/spe boost vs no boost")

    def test_boost_sign_distinguishable(self):
        """+2 vs -1 on the same stat must produce different network output."""
        model, layout, _ = feature_model()
        pos = np.zeros(7, dtype=np.int8); pos[0] = 2
        neg = np.zeros(7, dtype=np.int8); neg[0] = -1
        assert_delta_reaches_network(model, layout, _boost_delta(our_delta=pos),
                                     _boost_delta(our_delta=neg),
                                     msg="+2 vs -1 on atk — network must distinguish sign")


# ---------------------------------------------------------------------------
# 3. Forced-switch phase
# ---------------------------------------------------------------------------

class TestForcedSwitchPhase:
    def test_forced_switch_bit_captured(self):
        enc = encode_delta(anchor_delta(phase_is_forced_switch=True))
        assert enc[OFFSET_PHASE_FORCED_SWITCH] == 1.0

    def test_normal_phase_bit_zero(self):
        enc = encode_delta(anchor_delta(phase_is_forced_switch=False))
        assert enc[OFFSET_PHASE_FORCED_SWITCH] == 0.0

    def test_forced_switch_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(phase_is_forced_switch=False),
            anchor_delta(phase_is_forced_switch=True),
            msg="forced-switch phase bit — network must distinguish",
        )


# ---------------------------------------------------------------------------
# 4. Move order
# ---------------------------------------------------------------------------

class TestMoveOrder:
    def test_we_moved_first_encodes_first_bit(self):
        enc = encode_delta(anchor_delta(we_moved_first=True))
        block = read_block(enc, OFFSET_ORDER, ORDER_DIM)
        np.testing.assert_array_equal(block, [1.0, 0.0])

    def test_opp_moved_first_encodes_second_bit(self):
        enc = encode_delta(anchor_delta(we_moved_first=False))
        block = read_block(enc, OFFSET_ORDER, ORDER_DIM)
        np.testing.assert_array_equal(block, [0.0, 1.0])

    def test_none_order_all_zeros(self):
        enc = encode_delta(anchor_delta(we_moved_first=None))
        block = read_block(enc, OFFSET_ORDER, ORDER_DIM)
        np.testing.assert_array_equal(block, [0.0, 0.0])

    def test_order_we_first_vs_opp_first_reach_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(we_moved_first=True),
            anchor_delta(we_moved_first=False),
            msg="we_first vs opp_first — network must distinguish move order",
        )


# ---------------------------------------------------------------------------
# 5. Species block (actor / target)
# ---------------------------------------------------------------------------

class TestSpeciesBlock:
    def test_our_actor_species_nonzero(self):
        enc = encode_delta(anchor_delta(our_prev_active="tyranitar"))
        val = enc[OFFSET_OUR_ACTOR_SPECIES]
        assert val != 0.0, f"our_actor_species should be non-zero, got {val}"

    def test_opp_actor_species_nonzero(self):
        enc = encode_delta(anchor_delta(opp_prev_active="skarmory"))
        val = enc[OFFSET_OPP_ACTOR_SPECIES]
        assert val != 0.0, f"opp_actor_species should be non-zero, got {val}"

    def test_different_our_actor_species_differ(self):
        enc_a = encode_delta(anchor_delta(our_prev_active="tyranitar"))
        enc_b = encode_delta(anchor_delta(our_prev_active="blissey"))
        assert enc_a[OFFSET_OUR_ACTOR_SPECIES] != enc_b[OFFSET_OUR_ACTOR_SPECIES], (
            "Different our_actor species should have different embedded ids"
        )

    def test_different_opp_actor_species_differ(self):
        enc_a = encode_delta(anchor_delta(opp_prev_active="skarmory"))
        enc_b = encode_delta(anchor_delta(opp_prev_active="metagross"))
        assert enc_a[OFFSET_OPP_ACTOR_SPECIES] != enc_b[OFFSET_OPP_ACTOR_SPECIES]

    def test_actor_species_reaches_network(self):
        """Changing our_prev_active (actor species) changes network output."""
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(our_prev_active="tyranitar"),
            anchor_delta(our_prev_active="blissey"),
            msg="different our_actor species — network must distinguish",
        )


# ---------------------------------------------------------------------------
# 6. Attempted-switch rejected (trapped pivot)
# ---------------------------------------------------------------------------

class TestAttemptedSwitchRejected:
    def test_rejected_bit_captured(self):
        enc = encode_delta(anchor_delta(
            attempted_switch_rejected=True,
            attempted_switch_to="salamence",
        ))
        assert enc[OFFSET_ATTEMPTED_SWITCH_REJECTED] == 1.0

    def test_not_rejected_bit_zero(self):
        enc = encode_delta(anchor_delta(
            attempted_switch_rejected=False,
            attempted_switch_to=None,
        ))
        assert enc[OFFSET_ATTEMPTED_SWITCH_REJECTED] == 0.0

    def test_attempted_switch_to_species_nonzero_on_rejection(self):
        enc = encode_delta(anchor_delta(
            attempted_switch_rejected=True,
            attempted_switch_to="salamence",
        ))
        val = enc[OFFSET_OUR_ATTEMPTED_SWITCH_SPEC]
        assert val != 0.0, f"attempted_switch_to species should be non-zero, got {val}"

    def test_attempted_switch_to_species_zero_when_not_rejected(self):
        enc = encode_delta(anchor_delta(
            attempted_switch_rejected=False,
            attempted_switch_to=None,
        ))
        val = enc[OFFSET_OUR_ATTEMPTED_SWITCH_SPEC]
        assert val == 0.0, (
            f"attempted_switch_to species should be 0 when no rejection, got {val}"
        )

    def test_rejected_switch_reaches_network(self):
        """A rejected switch should alter network output vs a normal delta."""
        model, layout, _ = feature_model()
        normal = anchor_delta()
        rejected = anchor_delta(
            attempted_switch_rejected=True,
            attempted_switch_to="salamence",
        )
        assert_delta_reaches_network(
            model, layout, normal, rejected,
            msg="rejected-switch vs normal — network must register trap signal",
        )

    def test_rejected_switch_vs_normal_switch_distinguishable(self):
        """A rejected pivot is distinguishable from a successful switch."""
        model, layout, _ = feature_model()
        # Successful switch: our_switch_to set, rejected=False
        successful = _switch_delta_ours("salamence")
        # Rejected switch: our_switch_to=None (switch never happened), rejected=True
        rejected = anchor_delta(
            our_switch_to=None,
            our_move_id="tackle",  # we ended up using a move (or struggled)
            attempted_switch_rejected=True,
            attempted_switch_to="salamence",  # same intended mon
        )
        assert_delta_reaches_network(
            model, layout, successful, rejected,
            msg="rejected vs successful switch to same species — network must distinguish",
        )
