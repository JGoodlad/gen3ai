"""
Faint flags & faint-cause multi-hots are captured in the observation encoding
and reach the real Gen3FeaturesExtractor network output.

Pipeline under test:
    TurnDelta (we_fainted / our_faint_causes / …)
        -> TurnDeltaEncoder.encode
            -> OFFSET_WE_FAINTED / OFFSET_OPP_FAINTED (scalars)
            -> OFFSET_OUR_FAINT_CAUSES / OFFSET_OPP_FAINT_CAUSES (multi-hot)
        -> turn-history slot of the observation
        -> Gen3FeaturesExtractor (pass-through scalars → policy/value heads)

protocol -> TurnDelta capture is validated by event_log_fuzz_test.py and
poke_env_gaps fuzz tests; this suite closes the last hop: that the encoded
dims actually move the network.

Edge cases:
  1. Faint flags (we_fainted / opp_fainted)
  2. Every faint cause individually (attack/hazard/weather/status/recoil/selfko/leechseed/other)
  3. Multi-KO same window — two cause bits set
  4. Explosion self-KO semantics (move landed, then user fainted)
  5. KO-before-acting (fainted before moving, no move outcome)
"""
import numpy as np
import pytest

from agents.observation.turn_delta_encoder import (
    OFFSET_WE_FAINTED,
    OFFSET_OPP_FAINTED,
    OFFSET_OUR_FAINT_CAUSES,
    OFFSET_OPP_FAINT_CAUSES,
    OFFSET_OUR_MOVE_OUTCOME,
    OFFSET_OUR_MOVE_BLOCK,
    OFFSET_OUR_CANT,
    FAINT_CAUSE_DIM,
    OUTCOME_DIM,
    CANT_DIM,
)
from agents.battle.turn_view import FAINT_CAUSE_VOCAB

from agents.model.feature_coverage._support import (
    feature_model,
    anchor_delta,
    encode_delta,
    obs_with_delta,
    assert_reaches_network,
    assert_delta_reaches_network,
    read_block,
    multihot_idxs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cause_vec(*causes: str) -> np.ndarray:
    """Build a FAINT_CAUSE_DIM multi-hot with the named causes set."""
    vec = np.zeros(FAINT_CAUSE_DIM, dtype=np.float32)
    for c in causes:
        idx = FAINT_CAUSE_VOCAB.index(c)
        vec[idx] = 1.0
    return vec


def _faint_delta(
    we_fainted=False,
    opp_fainted=False,
    our_causes=None,
    opp_causes=None,
    our_move_id=None,
    our_attempted_move_id=None,
    our_move_outcome=None,
    our_cant_reason=None,
):
    """Anchor delta for faint-related tests.  Overrides faint fields only."""
    kw = dict(
        we_fainted=we_fainted,
        opp_fainted=opp_fainted,
    )
    if our_causes is not None:
        kw["our_faint_causes"] = our_causes
    if opp_causes is not None:
        kw["opp_faint_causes"] = opp_causes
    if our_move_id is not None:
        kw["our_move_id"] = our_move_id
    if our_attempted_move_id is not None:
        kw["our_attempted_move_id"] = our_attempted_move_id
    if our_move_outcome is not None:
        kw["our_move_outcome"] = our_move_outcome
    if our_cant_reason is not None:
        kw["our_cant_reason"] = our_cant_reason
        kw["our_failed_to_move"] = True
    return anchor_delta(**kw)


# ===========================================================================
# 1. Faint flags
# ===========================================================================

class TestFaintFlags:
    def test_we_fainted_encodes_bit(self):
        enc = encode_delta(_faint_delta(we_fainted=True))
        assert enc[OFFSET_WE_FAINTED] > 0.5, "we_fainted bit not set"

    def test_opp_fainted_encodes_bit(self):
        enc = encode_delta(_faint_delta(opp_fainted=True))
        assert enc[OFFSET_OPP_FAINTED] > 0.5, "opp_fainted bit not set"

    def test_no_faint_bits_clear(self):
        enc = encode_delta(_faint_delta())
        assert enc[OFFSET_WE_FAINTED] < 0.5, "we_fainted spuriously set"
        assert enc[OFFSET_OPP_FAINTED] < 0.5, "opp_fainted spuriously set"

    @pytest.mark.xfail(strict=True, reason=(
        "gen3_event_semantics_v1: the obs gap is CLOSED (faint_cause_id / item_transition "
        "columns; both verified to move each head). What remains is the TRANSLATOR: a TurnDelta "
        "is a per-turn AGGREGATE and the event window is a SEQUENCE, so a multi-KO turn or an "
        "ordering fact needs SEVERAL event rows and delta_to_event_cols emits one. See "
        "designs/ai_v9/design_frame_deletion_coverage_gaps.md §4 (delta_to_event_rows). Strict "
        "xfail: flips RED when the translator becomes multi-row."))
    def test_we_fainted_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            _faint_delta(),
            _faint_delta(we_fainted=True),
            "we_fainted flag vs no faint",
        )

    def test_opp_fainted_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            _faint_delta(),
            _faint_delta(opp_fainted=True),
            "opp_fainted flag vs no faint",
        )


# ===========================================================================
# 2. Every faint cause individually
# ===========================================================================

class TestEachFaintCause:
    @pytest.mark.parametrize("cause_idx,cause_name", list(enumerate(FAINT_CAUSE_VOCAB)))
    def test_cause_capture_our_side(self, cause_idx, cause_name):
        """our_faint_causes multi-hot has exactly the right bit set."""
        enc = encode_delta(_faint_delta(
            we_fainted=True,
            our_causes=_cause_vec(cause_name),
        ))
        assert multihot_idxs(enc, OFFSET_OUR_FAINT_CAUSES, FAINT_CAUSE_DIM) == [cause_idx], (
            f"our_faint_causes: expected bit {cause_idx} ({cause_name}), "
            f"got {multihot_idxs(enc, OFFSET_OUR_FAINT_CAUSES, FAINT_CAUSE_DIM)}"
        )

    @pytest.mark.parametrize("cause_idx,cause_name", list(enumerate(FAINT_CAUSE_VOCAB)))
    def test_cause_capture_opp_side(self, cause_idx, cause_name):
        """opp_faint_causes multi-hot has exactly the right bit set."""
        enc = encode_delta(_faint_delta(
            opp_fainted=True,
            opp_causes=_cause_vec(cause_name),
        ))
        assert multihot_idxs(enc, OFFSET_OPP_FAINT_CAUSES, FAINT_CAUSE_DIM) == [cause_idx], (
            f"opp_faint_causes: expected bit {cause_idx} ({cause_name}), "
            f"got {multihot_idxs(enc, OFFSET_OPP_FAINT_CAUSES, FAINT_CAUSE_DIM)}"
        )

    @pytest.mark.parametrize("cause_name", list(FAINT_CAUSE_VOCAB))
    def test_cause_our_reaches_network(self, cause_name):
        """Each our_faint_causes bit moves the network output."""
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            _faint_delta(),
            _faint_delta(we_fainted=True, our_causes=_cause_vec(cause_name)),
            f"our_faint_causes={cause_name} vs empty",
        )

    @pytest.mark.parametrize("cause_name", list(FAINT_CAUSE_VOCAB))
    def test_cause_opp_reaches_network(self, cause_name):
        """Each opp_faint_causes bit moves the network output."""
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            _faint_delta(),
            _faint_delta(opp_fainted=True, opp_causes=_cause_vec(cause_name)),
            f"opp_faint_causes={cause_name} vs empty",
        )

    def test_two_different_causes_produce_different_outputs(self):
        """Attack-cause and weather-cause produce distinguishable network outputs,
        proving cause identity (not just 'fainted') reaches the net."""
        model, layout, _ = feature_model()
        obs_attack = obs_with_delta(layout, _faint_delta(
            we_fainted=True, our_causes=_cause_vec("attack"),
        ))
        obs_weather = obs_with_delta(layout, _faint_delta(
            we_fainted=True, our_causes=_cause_vec("weather"),
        ))
        assert_reaches_network(
            model, obs_attack, obs_weather,
            "our_faint_causes=attack vs weather (cause identity visible)",
        )


# ===========================================================================
# 3. Multi-KO same window — two cause bits
# ===========================================================================

class TestMultiKoWindow:
    def test_two_bits_set_in_encoding(self):
        """attack + hazard both present in the multi-hot."""
        enc = encode_delta(_faint_delta(
            we_fainted=True,
            our_causes=_cause_vec("attack", "hazard"),
        ))
        bits = multihot_idxs(enc, OFFSET_OUR_FAINT_CAUSES, FAINT_CAUSE_DIM)
        expected = sorted([
            FAINT_CAUSE_VOCAB.index("attack"),
            FAINT_CAUSE_VOCAB.index("hazard"),
        ])
        assert bits == expected, f"Expected {expected}, got {bits}"

    def test_multi_ko_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            _faint_delta(),
            _faint_delta(we_fainted=True, our_causes=_cause_vec("attack", "hazard")),
            "multi-KO (attack+hazard) vs empty",
        )

    @pytest.mark.xfail(strict=True, reason=(
        "gen3_event_semantics_v1: the obs gap is CLOSED (faint_cause_id / item_transition "
        "columns; both verified to move each head). What remains is the TRANSLATOR: a TurnDelta "
        "is a per-turn AGGREGATE and the event window is a SEQUENCE, so a multi-KO turn or an "
        "ordering fact needs SEVERAL event rows and delta_to_event_cols emits one. See "
        "designs/ai_v9/design_frame_deletion_coverage_gaps.md §4 (delta_to_event_rows). Strict "
        "xfail: flips RED when the translator becomes multi-row."))
    def test_multi_ko_vs_single_ko_distinguishable(self):
        """Two causes produce different output from one cause — the extra bit is visible."""
        model, layout, _ = feature_model()
        obs_single = obs_with_delta(layout, _faint_delta(
            we_fainted=True, our_causes=_cause_vec("attack"),
        ))
        obs_double = obs_with_delta(layout, _faint_delta(
            we_fainted=True, our_causes=_cause_vec("attack", "hazard"),
        ))
        assert_reaches_network(
            model, obs_single, obs_double,
            "single-KO vs multi-KO (attack+hazard)",
        )


# ===========================================================================
# 4. Explosion self-KO semantics
# ===========================================================================

class TestExplosionSelfKo:
    def _explosion_delta(self):
        return _faint_delta(
            our_move_id="explosion",
            our_attempted_move_id="explosion",
            our_move_outcome="hit",
            we_fainted=True,
            our_causes=_cause_vec("selfko"),
        )

    def test_selfko_bit_set(self):
        enc = encode_delta(self._explosion_delta())
        bits = multihot_idxs(enc, OFFSET_OUR_FAINT_CAUSES, FAINT_CAUSE_DIM)
        selfko_idx = FAINT_CAUSE_VOCAB.index("selfko")
        assert selfko_idx in bits, f"selfko bit not set; got {bits}"

    def test_move_id_present(self):
        """Explosion move id is encoded (not the unknown sentinel 0)."""
        enc = encode_delta(self._explosion_delta())
        assert enc[OFFSET_OUR_MOVE_BLOCK] != 0.0, "explosion move id missing from encoding"

    def test_outcome_is_hit(self):
        """The move hit before the user fainted — outcome one-hot should show 'hit'."""
        enc = encode_delta(self._explosion_delta())
        outcome_block = read_block(enc, OFFSET_OUR_MOVE_OUTCOME, OUTCOME_DIM)
        # hit/miss/fail = indices 0/1/2
        assert outcome_block[0] > 0.5, f"outcome is not 'hit': {outcome_block}"
        assert outcome_block[1] < 0.5
        assert outcome_block[2] < 0.5

    def test_explosion_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(),
            self._explosion_delta(),
            "explosion self-KO vs plain tackle anchor",
        )

    @pytest.mark.xfail(strict=True, reason=(
        "gen3_event_semantics_v1: the obs gap is CLOSED (faint_cause_id / item_transition "
        "columns; both verified to move each head). What remains is the TRANSLATOR: a TurnDelta "
        "is a per-turn AGGREGATE and the event window is a SEQUENCE, so a multi-KO turn or an "
        "ordering fact needs SEVERAL event rows and delta_to_event_cols emits one. See "
        "designs/ai_v9/design_frame_deletion_coverage_gaps.md §4 (delta_to_event_rows). Strict "
        "xfail: flips RED when the translator becomes multi-row."))
    def test_explosion_distinct_from_plain_faint(self):
        """A selfko faint with Explosion is distinguishable from a faint with no move."""
        model, layout, _ = feature_model()
        plain_faint = _faint_delta(we_fainted=True, our_causes=_cause_vec("selfko"))
        obs_plain = obs_with_delta(layout, plain_faint)
        obs_explosion = obs_with_delta(layout, self._explosion_delta())
        assert_reaches_network(
            model, obs_plain, obs_explosion,
            "explosion self-KO vs selfko faint with no move",
        )


# ===========================================================================
# 5. KO-before-acting
# ===========================================================================

class TestKoBeforeActing:
    def _ko_before_acting_delta(self):
        """Our mon fainted before it could act.

        KO-before-acting does NOT emit a |cant| line — the mon is simply dead
        before the action window closes. The TurnDelta carries:
          - we_fainted=True
          - our_failed_to_move=True  (flagged by the env when no move fired)
          - our_cant_reason=None     (no |cant| protocol message)
          - our_move_id=None         (no move was selected/fired)
          - our_move_outcome=None    (no outcome onehot)
        The faint cause multi-hot records how it died (e.g. attack).
        """
        return anchor_delta(
            we_fainted=True,
            our_faint_causes=_cause_vec("attack"),
            our_failed_to_move=True,
            our_cant_reason=None,
            our_move_id=None,
            our_attempted_move_id=None,
            our_move_outcome=None,
        )

    def test_cant_block_all_zero(self):
        """KO-before-acting has no |cant| reason — the cant one-hot is all-zero."""
        enc = encode_delta(self._ko_before_acting_delta())
        cant_block = read_block(enc, OFFSET_OUR_CANT, CANT_DIM)
        assert cant_block.max() < 0.5, (
            f"Expected cant one-hot all-zero for KO-before-acting, got {cant_block}"
        )

    def test_no_move_outcome_set(self):
        """When KO'd before acting, the move outcome block is all-zero (no outcome)."""
        enc = encode_delta(self._ko_before_acting_delta())
        outcome_block = read_block(enc, OFFSET_OUR_MOVE_OUTCOME, OUTCOME_DIM)
        assert outcome_block.max() < 0.5, (
            f"Expected no move outcome for KO-before-acting, got {outcome_block}"
        )

    def test_faint_flag_still_set(self):
        """we_fainted is set even when the mon died before acting."""
        enc = encode_delta(self._ko_before_acting_delta())
        assert enc[OFFSET_WE_FAINTED] > 0.5, "we_fainted not set for KO-before-acting"

    def test_ko_before_acting_reaches_network(self):
        model, layout, _ = feature_model()
        assert_delta_reaches_network(
            model, layout,
            anchor_delta(),
            self._ko_before_acting_delta(),
            "KO-before-acting (no move, no cant, fainted) vs plain anchor",
        )

    @pytest.mark.xfail(strict=True, reason=(
        "gen3_event_semantics_v1: the obs gap is CLOSED (faint_cause_id / item_transition "
        "columns; both verified to move each head). What remains is the TRANSLATOR: a TurnDelta "
        "is a per-turn AGGREGATE and the event window is a SEQUENCE, so a multi-KO turn or an "
        "ordering fact needs SEVERAL event rows and delta_to_event_cols emits one. See "
        "designs/ai_v9/design_frame_deletion_coverage_gaps.md §4 (delta_to_event_rows). Strict "
        "xfail: flips RED when the translator becomes multi-row."))
    def test_ko_before_acting_distinct_from_normal_faint(self):
        """A KO-before-acting (no move, no outcome) is distinguishable from
        a faint where the mon did get a move off (tackle hit, then died)."""
        model, layout, _ = feature_model()
        normal_faint = _faint_delta(
            our_move_id="tackle",
            our_move_outcome="hit",
            we_fainted=True,
            our_causes=_cause_vec("attack"),
        )
        obs_normal = obs_with_delta(layout, normal_faint)
        obs_ko_before = obs_with_delta(layout, self._ko_before_acting_delta())
        assert_reaches_network(
            model, obs_normal, obs_ko_before,
            "KO-before-acting vs faint-after-moving (tackle+hit)",
        )
