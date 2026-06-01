"""
EXEMPLAR — a FAILED Protect is captured AND reaches the network.

This is the template every test in `feature_coverage/` follows (see
`_support.py` for the shared harness). The pipeline for a move outcome is:

    protocol (|-nothing| on a repeat Protect)
        -> TurnView fold (SideTurn.failed -> outcome == "fail")
        -> TurnDelta.our_move_outcome == "fail"
        -> TurnDeltaEncoder.encode  ([hit, miss, fail] one-hot -> [0, 0, 1])
        -> turn-history slot of the observation
        -> Gen3FeaturesExtractor   (pass-through scalar -> policy/value heads)

`move_outcome_fuzz_test.py` validates protocol -> ... -> encoded-vector on real
battles. This closes the last hop: that the encoded fail dims move the network.
"""
import numpy as np
import torch

from agents.observation.turn_delta_encoder import (
    OUTCOME_DIM,
    OFFSET_OUR_MOVE_OUTCOME,
    OFFSET_OPP_MOVE_OUTCOME,
    OFFSET_OUR_MOVE_BLOCK,
    TURN_DELTA_EMBEDDED_IDS,
    TURN_DELTA_SCALAR_OFFSETS,
    _OUTCOME_TO_IDX,
)

from agents.model.feature_coverage._support import (
    feature_model,
    make_delta,
    encode_delta,
    obs_with_delta,
    obs_zero,
    history_slot_offset,
    assert_reaches_network,
    read_block,
)


def _protect_delta(outcome):
    """We pressed Protect; `outcome` is "fail" / "hit" / None. Everything else
    held identical so the only difference between two of these is the outcome."""
    return make_delta(
        our_move_id="protect",
        our_attempted_move_id="protect",
        our_prev_active="blastoise",
        our_move_outcome=outcome,
    )


# ---------------------------------------------------------------------------
# 1. Capture
# ---------------------------------------------------------------------------

def test_failed_protect_encodes_fail_onehot():
    enc = encode_delta(_protect_delta("fail"))
    block = read_block(enc, OFFSET_OUR_MOVE_OUTCOME, OUTCOME_DIM)
    np.testing.assert_array_equal(block, [0.0, 0.0, 1.0])
    assert int(np.argmax(block)) == _OUTCOME_TO_IDX["fail"]

    # A Protect that went up encodes "hit" — same move, different outcome dim.
    hit = read_block(encode_delta(_protect_delta("hit")), OFFSET_OUR_MOVE_OUTCOME, OUTCOME_DIM)
    np.testing.assert_array_equal(hit, [1.0, 0.0, 0.0])

    # The Protect move id is carried (not the unknown sentinel), so the failure
    # is attributable to Protect specifically.
    assert enc[OFFSET_OUR_MOVE_BLOCK] != 0.0


# ---------------------------------------------------------------------------
# 2. Structural — outcome dims are pass-through scalars (not dropped)
# ---------------------------------------------------------------------------

def test_outcome_dims_reach_extractor_as_scalars():
    embedded = frozenset(pos for pos, _ in TURN_DELTA_EMBEDDED_IDS)
    outcome_positions = (
        list(range(OFFSET_OUR_MOVE_OUTCOME, OFFSET_OUR_MOVE_OUTCOME + OUTCOME_DIM))
        + list(range(OFFSET_OPP_MOVE_OUTCOME, OFFSET_OPP_MOVE_OUTCOME + OUTCOME_DIM))
    )
    for pos in outcome_positions:
        assert pos not in embedded, f"outcome dim {pos} wrongly declared as embedded id"
        assert pos in TURN_DELTA_SCALAR_OFFSETS, f"outcome dim {pos} missing from scalar offsets"

    model, _, _ = feature_model()
    scalar_idx = set(model.embeddings._td_scalar_idx.tolist())
    for pos in outcome_positions:
        assert pos in scalar_idx, f"outcome dim {pos} not in extractor scalar gather"


# ---------------------------------------------------------------------------
# 3. Network
# ---------------------------------------------------------------------------

def test_failed_protect_changes_network_output():
    model, layout, _ = feature_model()
    assert_reaches_network(
        model, obs_zero(layout), obs_with_delta(layout, _protect_delta("fail")),
        "failed Protect vs empty history",
    )


def test_failed_vs_successful_protect_distinguishable_by_network():
    """The fail signal SPECIFICALLY reaches the net: two obs identical except
    the outcome one-hot (fail vs hit) must produce different outputs."""
    model, layout, _ = feature_model()
    obs_fail = obs_with_delta(layout, _protect_delta("fail"))
    obs_hit = obs_with_delta(layout, _protect_delta("hit"))

    # Precondition: the two differ ONLY within the outcome one-hot block
    # (fail=[0,0,1] vs hit=[1,0,0] differ at dims 0 and 2; the middle is zero in
    # both — so the diff is a non-empty subset of the block).
    diff = (obs_fail - obs_hit).abs().squeeze(0).numpy()
    nonzero = set(np.where(diff > 0)[0].tolist())
    slot = history_slot_offset(layout)
    block = set(slot + OFFSET_OUR_MOVE_OUTCOME + i for i in range(OUTCOME_DIM))
    assert nonzero and nonzero <= block, "fail vs hit differ outside the outcome block"

    assert_reaches_network(model, obs_fail, obs_hit, "failed vs successful Protect")
