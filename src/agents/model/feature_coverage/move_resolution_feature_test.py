"""
Move resolution & prevention edge cases — captured in the obs vector AND reaching the network.

Each edge case follows the two-part pattern from the exemplar (failed_protect_feature_test.py):
  1. CAPTURE  — the right offset/onehot/bit is set in the encoded slot vector.
  2. NETWORK  — the diff between a baseline and a variant moves BOTH network heads.

Covered cases (the exemplar already covers "fail" outcome; we do not redo it):
  1. Accuracy miss  (our_move_outcome="miss" / opp_move_outcome="miss")
  2. Critical hit   (our_move_crit=True / opp_move_crit=True)
  3. Cant reasons   (all CANT_REASONS × our_cant_reason / opp_cant_reason)
  4. Type effectiveness  (immune 0.0 / resisted 0.5 / neutral 1.0 / super 2.0)
  5. Attempted move id when it never fired (our_attempted_move_id="protect", our_move_id=None)
  6. opp_move_known=False
"""
import pytest
import numpy as np
import torch

from agents.observation.turn_delta_encoder import (
    OUTCOME_DIM,
    EFF_DIM,
    CANT_DIM,
    OFFSET_OUR_MOVE_OUTCOME,
    OFFSET_OPP_MOVE_OUTCOME,
    OFFSET_OUR_CRIT,
    OFFSET_OPP_CRIT,
    OFFSET_OUR_CANT,
    OFFSET_OPP_CANT,
    OFFSET_OUR_FAILED_TO_MOVE,
    OFFSET_OPP_FAILED_TO_MOVE,
    OFFSET_OUR_EFF,
    OFFSET_OPP_EFF,
    OFFSET_OUR_ATTEMPTED_MOVE,
    OFFSET_OPP_MOVE_KNOWN,
    TURN_DELTA_EMBEDDED_IDS,
    TURN_DELTA_SCALAR_OFFSETS,
    _OUTCOME_TO_IDX,
)
from agents.observation.gen3_effects import (
    CANT_REASONS,
    _CANT_INDEX,
    normalize_cant_reason,
)
from agents.model.feature_coverage._support import (
    feature_model,
    anchor_delta,
    encode_delta,
    obs_with_delta,
    assert_delta_reaches_network,
    assert_reaches_network,
    read_block,
    onehot_idx,
)


# ===========================================================================
# 1. Accuracy miss
# ===========================================================================

class TestAccuracyMiss:
    """our_move_outcome="miss" and opp_move_outcome="miss" are captured and reach the net."""

    def _our_miss_delta(self):
        return anchor_delta(our_move_outcome="miss")

    def _opp_miss_delta(self):
        return anchor_delta(our_move_outcome="hit", opp_move_id="tackle", opp_move_outcome="miss")

    def test_our_miss_encodes_miss_onehot(self):
        enc = encode_delta(self._our_miss_delta())
        block = read_block(enc, OFFSET_OUR_MOVE_OUTCOME, OUTCOME_DIM)
        np.testing.assert_array_equal(block, [0.0, 1.0, 0.0])
        assert int(np.argmax(block)) == _OUTCOME_TO_IDX["miss"]

    def test_opp_miss_encodes_miss_onehot(self):
        enc = encode_delta(self._opp_miss_delta())
        block = read_block(enc, OFFSET_OPP_MOVE_OUTCOME, OUTCOME_DIM)
        np.testing.assert_array_equal(block, [0.0, 1.0, 0.0])
        assert int(np.argmax(block)) == _OUTCOME_TO_IDX["miss"]

    def test_our_miss_vs_hit_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(our_move_outcome="hit"),
            anchor_delta(our_move_outcome="miss"),
            "our miss vs hit",
        )

    def test_opp_miss_vs_hit_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(opp_move_id="tackle", opp_move_outcome="hit"),
            anchor_delta(opp_move_id="tackle", opp_move_outcome="miss"),
            "opp miss vs hit",
        )


# ===========================================================================
# 2. Critical hit
# ===========================================================================

class TestCriticalHit:
    """our_move_crit=True and opp_move_crit=True are captured and reach the net."""

    def test_our_crit_bit_set(self):
        enc = encode_delta(anchor_delta(our_move_crit=True))
        assert enc[OFFSET_OUR_CRIT] > 0.5, "our_move_crit bit not set"

    def test_opp_crit_bit_set(self):
        enc = encode_delta(anchor_delta(opp_move_id="tackle", opp_move_crit=True))
        assert enc[OFFSET_OPP_CRIT] > 0.5, "opp_move_crit bit not set"

    def test_our_no_crit_bit_clear(self):
        enc = encode_delta(anchor_delta(our_move_crit=False))
        assert enc[OFFSET_OUR_CRIT] < 0.5, "our_move_crit bit unexpectedly set"

    def test_our_crit_vs_no_crit_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(our_move_crit=False),
            anchor_delta(our_move_crit=True),
            "our crit vs no-crit",
        )

    def test_opp_crit_vs_no_crit_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(opp_move_id="tackle", opp_move_crit=False),
            anchor_delta(opp_move_id="tackle", opp_move_crit=True),
            "opp crit vs no-crit",
        )

    def test_crit_orthogonal_to_hit_outcome(self):
        """A crit move can still have outcome="hit"; both bits can be set simultaneously."""
        enc = encode_delta(anchor_delta(our_move_crit=True, our_move_outcome="hit"))
        assert enc[OFFSET_OUR_CRIT] > 0.5
        block = read_block(enc, OFFSET_OUR_MOVE_OUTCOME, OUTCOME_DIM)
        np.testing.assert_array_equal(block, [1.0, 0.0, 0.0])


# ===========================================================================
# 3. Cant reasons — all CANT_REASONS parametrized
# ===========================================================================

class TestCantReasons:
    """Every cant reason in CANT_REASONS maps to the correct one-hot index.
    Parametrize over all 12 reasons and assert capture + network reach."""

    @pytest.mark.parametrize("reason", list(CANT_REASONS))
    def test_our_cant_onehot_index(self, reason):
        """our_cant_reason=<r> sets OFFSET_OUR_CANT + _CANT_INDEX[r] to 1.0."""
        delta = anchor_delta(
            our_cant_reason=reason,
            our_failed_to_move=True,
        )
        enc = encode_delta(delta)
        block = read_block(enc, OFFSET_OUR_CANT, CANT_DIM)
        expected_idx = _CANT_INDEX[normalize_cant_reason(reason)]
        assert block[expected_idx] == 1.0, (
            f"our_cant_reason={reason!r}: expected index {expected_idx} to be 1.0, got {block}"
        )
        # All other bits in the cant block must be zero.
        other_bits = [block[i] for i in range(CANT_DIM) if i != expected_idx]
        assert all(b == 0.0 for b in other_bits), (
            f"our_cant_reason={reason!r}: unexpected extra bits set: {block}"
        )
        # The failed_to_move bit must also be set.
        assert enc[OFFSET_OUR_FAILED_TO_MOVE] > 0.5, (
            f"our_failed_to_move bit not set alongside cant reason {reason!r}"
        )

    @pytest.mark.parametrize("reason", list(CANT_REASONS))
    def test_opp_cant_onehot_index(self, reason):
        """opp_cant_reason=<r> sets OFFSET_OPP_CANT + _CANT_INDEX[r] to 1.0."""
        delta = anchor_delta(
            opp_cant_reason=reason,
            opp_failed_to_move=True,
        )
        enc = encode_delta(delta)
        block = read_block(enc, OFFSET_OPP_CANT, CANT_DIM)
        expected_idx = _CANT_INDEX[normalize_cant_reason(reason)]
        assert block[expected_idx] == 1.0, (
            f"opp_cant_reason={reason!r}: expected index {expected_idx} to be 1.0, got {block}"
        )
        other_bits = [block[i] for i in range(CANT_DIM) if i != expected_idx]
        assert all(b == 0.0 for b in other_bits), (
            f"opp_cant_reason={reason!r}: unexpected extra bits set: {block}"
        )
        assert enc[OFFSET_OPP_FAILED_TO_MOVE] > 0.5, (
            f"opp_failed_to_move bit not set alongside cant reason {reason!r}"
        )

    @pytest.mark.parametrize("reason", list(CANT_REASONS))
    def test_our_cant_reason_reaches_network(self, reason):
        """Each distinct our_cant_reason must produce a distinct network output vs no-cant anchor."""
        model, layout, _ = feature_model()
        base = anchor_delta()  # no cant
        variant = anchor_delta(our_cant_reason=reason, our_failed_to_move=True)
        assert_delta_reaches_network(model, layout, base, variant,
                                     f"our cant={reason!r} vs no cant")

    @pytest.mark.parametrize("reason", list(CANT_REASONS))
    def test_opp_cant_reason_reaches_network(self, reason):
        """Each distinct opp_cant_reason must produce a distinct network output vs no-cant anchor."""
        model, layout, _ = feature_model()
        base = anchor_delta()
        variant = anchor_delta(opp_cant_reason=reason, opp_failed_to_move=True)
        assert_delta_reaches_network(model, layout, base, variant,
                                     f"opp cant={reason!r} vs no cant")

    def test_cant_reasons_map_to_distinct_indices(self):
        """Sanity: every cant reason maps to a unique index (the _CANT_INDEX dict has no collisions)."""
        indices = [_CANT_INDEX[normalize_cant_reason(r)] for r in CANT_REASONS]
        assert len(set(indices)) == len(CANT_REASONS), "duplicate cant reason indices detected"


# ===========================================================================
# 4. Type-effectiveness one-hots
# ===========================================================================

class TestTypeEffectiveness:
    """The four effectiveness buckets (immune/resisted/neutral/super) are captured
    at OFFSET_OUR_EFF / OFFSET_OPP_EFF (EFF_DIM=4: [immune, resisted, normal, super-effective]).

    The encoder maps:
      0.0  -> index 0 (immune)
      0.5  -> index 1 (resisted)
      1.0  -> index 2 (normal)
      2.0+ -> index 3 (super-effective)
    """

    # (mult, expected_idx, label)
    EFF_CASES = [
        (0.0, 0, "immune"),
        (0.5, 1, "resisted"),
        (1.0, 2, "neutral"),
        (2.0, 3, "super"),
    ]

    @pytest.mark.parametrize("mult,expected_idx,label", EFF_CASES)
    def test_our_effectiveness_onehot(self, mult, expected_idx, label):
        # Need a damaging move to make effectiveness meaningful; use "tackle".
        delta = anchor_delta(our_move_id="tackle", our_effectiveness=mult)
        enc = encode_delta(delta)
        block = read_block(enc, OFFSET_OUR_EFF, EFF_DIM)
        assert int(np.argmax(block)) == expected_idx, (
            f"our_effectiveness={mult} ({label}): expected index {expected_idx}, got {block}"
        )
        assert block[expected_idx] == 1.0

    @pytest.mark.parametrize("mult,expected_idx,label", EFF_CASES)
    def test_opp_effectiveness_onehot(self, mult, expected_idx, label):
        delta = anchor_delta(opp_move_id="tackle", opp_effectiveness=mult)
        enc = encode_delta(delta)
        block = read_block(enc, OFFSET_OPP_EFF, EFF_DIM)
        assert int(np.argmax(block)) == expected_idx, (
            f"opp_effectiveness={mult} ({label}): expected index {expected_idx}, got {block}"
        )
        assert block[expected_idx] == 1.0

    def test_super_vs_resisted_our_effectiveness_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(our_move_id="tackle", our_effectiveness=0.5),
            anchor_delta(our_move_id="tackle", our_effectiveness=2.0),
            "our super-effective vs resisted",
        )

    def test_super_vs_resisted_opp_effectiveness_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(opp_move_id="tackle", opp_effectiveness=0.5),
            anchor_delta(opp_move_id="tackle", opp_effectiveness=2.0),
            "opp super-effective vs resisted",
        )

    def test_immune_vs_neutral_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(our_move_id="tackle", our_effectiveness=1.0),
            anchor_delta(our_move_id="tackle", our_effectiveness=0.0),
            "our immune vs neutral",
        )


# ===========================================================================
# 5. Attempted move preserved when it never fired
# ===========================================================================

class TestAttemptedMoveId:
    """our_attempted_move_id is preserved even when our_move_id=None (move never fired).

    OFFSET_OUR_ATTEMPTED_MOVE carries a raw move num (embedded by the extractor).
    It is non-zero when an attempted move is set.
    """

    def test_attempted_move_id_non_zero_when_set(self):
        # our_move_id=None (frozen, never fired) but we pressed "protect".
        delta = anchor_delta(our_move_id=None, our_attempted_move_id="protect")
        enc = encode_delta(delta)
        # The attempted move position holds the raw move num as a float — must be non-zero.
        assert enc[OFFSET_OUR_ATTEMPTED_MOVE] != 0.0, (
            "our_attempted_move_id='protect' should set a non-zero raw num at "
            f"OFFSET_OUR_ATTEMPTED_MOVE={OFFSET_OUR_ATTEMPTED_MOVE}, got {enc[OFFSET_OUR_ATTEMPTED_MOVE]}"
        )

    def test_attempted_move_id_zero_when_absent(self):
        delta = anchor_delta(our_attempted_move_id=None)
        enc = encode_delta(delta)
        assert enc[OFFSET_OUR_ATTEMPTED_MOVE] == 0.0, (
            f"our_attempted_move_id=None should give 0 at OFFSET_OUR_ATTEMPTED_MOVE, "
            f"got {enc[OFFSET_OUR_ATTEMPTED_MOVE]}"
        )

    def test_attempted_move_is_embedded_not_scalar(self):
        """OFFSET_OUR_ATTEMPTED_MOVE must be in the embedded-ID manifest, NOT in scalar offsets.
        If it were a pass-through scalar the raw int would go to the model unembedded — wrong."""
        embedded_positions = frozenset(pos for pos, _ in TURN_DELTA_EMBEDDED_IDS)
        assert OFFSET_OUR_ATTEMPTED_MOVE in embedded_positions, (
            f"OFFSET_OUR_ATTEMPTED_MOVE={OFFSET_OUR_ATTEMPTED_MOVE} is NOT in the embedded-ID manifest — "
            "the attempted move id would flow to the model as a raw scalar rather than being embedded"
        )
        assert OFFSET_OUR_ATTEMPTED_MOVE not in set(TURN_DELTA_SCALAR_OFFSETS), (
            f"OFFSET_OUR_ATTEMPTED_MOVE={OFFSET_OUR_ATTEMPTED_MOVE} is wrongly in scalar offsets"
        )

    def test_attempted_move_reaches_network_vs_no_attempted_move(self):
        """Setting our_attempted_move_id='protect' (move never fired) vs nothing must change output."""
        model, layout, _ = feature_model()
        base = anchor_delta(our_move_id=None, our_attempted_move_id=None)
        variant = anchor_delta(our_move_id=None, our_attempted_move_id="protect")
        assert_delta_reaches_network(model, layout, base, variant,
                                     "attempted protect (never fired) vs no attempted move")

    def test_attempted_move_different_moves_give_different_output(self):
        """Two different attempted moves should produce different network outputs (distinct embeddings)."""
        model, layout, _ = feature_model()
        obs_protect = obs_with_delta(layout, anchor_delta(our_move_id=None, our_attempted_move_id="protect"))
        obs_tackle = obs_with_delta(layout, anchor_delta(our_move_id=None, our_attempted_move_id="tackle"))
        assert_reaches_network(model, obs_protect, obs_tackle,
                                "protect vs tackle as attempted (never fired)")


# ===========================================================================
# 6. opp_move_known=False
# ===========================================================================

class TestOppMoveKnown:
    """opp_move_known=False → bit at OFFSET_OPP_MOVE_KNOWN is 0; True → 1.
    The difference must reach the network."""

    def test_opp_move_known_bit_true(self):
        enc = encode_delta(anchor_delta(opp_move_known=True))
        assert enc[OFFSET_OPP_MOVE_KNOWN] > 0.5, "opp_move_known=True should set the bit"

    def test_opp_move_known_bit_false(self):
        enc = encode_delta(anchor_delta(opp_move_known=False))
        assert enc[OFFSET_OPP_MOVE_KNOWN] < 0.5, "opp_move_known=False should leave the bit clear"

    def test_opp_move_known_is_scalar(self):
        """OFFSET_OPP_MOVE_KNOWN must be a pass-through scalar (not an embedded id)."""
        embedded_positions = frozenset(pos for pos, _ in TURN_DELTA_EMBEDDED_IDS)
        assert OFFSET_OPP_MOVE_KNOWN not in embedded_positions, (
            f"OFFSET_OPP_MOVE_KNOWN={OFFSET_OPP_MOVE_KNOWN} wrongly declared as embedded id"
        )
        assert OFFSET_OPP_MOVE_KNOWN in set(TURN_DELTA_SCALAR_OFFSETS), (
            f"OFFSET_OPP_MOVE_KNOWN={OFFSET_OPP_MOVE_KNOWN} missing from scalar offsets"
        )

    def test_opp_move_known_false_vs_true_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(opp_move_known=True),
            anchor_delta(opp_move_known=False),
            "opp_move_known=False vs True",
        )
