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

    # gen3_frame_deletion_v1: the extractor-side gather assertion moved to
    # `test_no_linear_reads_a_raw_event_id` below — the lag frame's manifest is deleted, the
    # rule it enforced is not. The TurnDelta-side assertions above still stand: the encoder
    # survives and the prober reads archived runs with it.


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

    # Precondition: the two observations differ ONLY in the outcome columns of the ONE event row
    # they write. gen3_frame_deletion_v1 re-homed this from the lag frame's outcome one-hot to the
    # event window's hit/miss/fail columns (6..8); the claim is the same and is what makes the
    # network assertion below attributable — if they differed anywhere else, a moved output would
    # not be evidence about the outcome at all.
    from agents.observation.constants import (
        OFFSET_EVENT_WINDOW, EVENT_WINDOW_N, EVENT_TOKEN_DIM,
    )
    diff = (obs_fail - obs_hit).abs().squeeze(0).numpy()
    nonzero = set(np.where(diff > 0)[0].tolist())
    row_off = OFFSET_EVENT_WINDOW + (EVENT_WINDOW_N - 1) * EVENT_TOKEN_DIM
    block = {row_off + c for c in (6, 7, 8)}
    assert nonzero and nonzero <= block, (
        f"fail vs hit differ outside the event row's outcome columns: "
        f"{sorted(nonzero - block)}")

    assert_reaches_network(model, obs_fail, obs_hit, "failed vs successful Protect")


def test_no_linear_reads_a_raw_event_id():
    """gen3_frame_deletion_v1 successor to the `_td_scalar_idx` gather assertions.

    Those pinned the lag frame's embedded-ID manifest: a raw dex num must reach an EMBEDDING
    table, never a Linear (a known bug class here — a species num fed to a Linear is a magnitude,
    not an identity). The frames are gone, but the H-B event window carries the SAME kind of raw
    ids and states the same rule, so the invariant moves rather than dies: `EventSeats` must route
    every id column through a table and let only the true scalars ride raw."""
    from agents.observation.constants import EVENT_TOKEN_DIM
    model, _, _ = feature_model()
    es = model.history_events
    assert es is not None, "the fixture must build history_events=True"
    ID_COLS = {0: "kind_emb", 1: "species", 3: "species", 4: "move", 15: "status_emb",
                19: "cant_emb", 20: "faint_emb", 21: "itemtr_emb"}
    SCALAR_COLS = {2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17}
    VALID_COL = {18}
    assert set(ID_COLS) | SCALAR_COLS | VALID_COL == set(range(EVENT_TOKEN_DIM)), (
        "every event column must be classified as an embedded id, a raw scalar, or the pad flag — "
        "an unclassified column is one nobody decided the routing for")
    assert es._N_SCALARS == len(SCALAR_COLS), (
        f"EventSeats says {es._N_SCALARS} raw scalars, the column map says {len(SCALAR_COLS)} — "
        "if a column changed routing, one of the two was not updated")
    for attr in ("kind_emb", "status_emb", "cant_emb", "faint_emb", "itemtr_emb"):
        assert hasattr(es, attr), f"EventSeats lost its {attr} table"
