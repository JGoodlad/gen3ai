"""
Status-transition and item-consumption feature coverage.

Two-part test per edge case:
  1. CAPTURE  — the signal encodes to the expected one-hot / bit position.
  2. NETWORK  — the encoded signal moves BOTH policy and value heads of the
                real Gen3FeaturesExtractor.

Status-transition pipeline:
    protocol (|-status|, |-curestatus|)
        -> TurnView fold (SideTurn.status_applied / status_cured)
        -> TurnDelta.our_status_applied / our_status_cured  (poke-env Status enum)
        -> TurnDeltaEncoder._status_onehot  ([BRN, FNT, FRZ, PAR, PSN, SLP, TOX])
        -> OFFSET_OUR_STATUS_APPLIED / OFFSET_OUR_STATUS_CURED
        -> turn-history slot -> Gen3FeaturesExtractor

Item-consumption pipeline:
    protocol (|-enditem|)
        -> TurnView fold (SideTurn.item_lost = item id string)
        -> TurnDelta.our_item_lost (non-None)
        -> encoder: float(our_item_lost is not None) -> OFFSET_OUR_ITEM_USED (1 bit)
        -> turn-history slot -> Gen3FeaturesExtractor

STATUS INDEX MAPPING (from _STATUS_ORDER in turn_delta_encoder.py):
    0 = BRN, 1 = FNT, 2 = FRZ, 3 = PAR, 4 = PSN, 5 = SLP, 6 = TOX
"""
import numpy as np
import pytest

from agents.enums import Status
from agents.observation.turn_delta_encoder import (
    OFFSET_OUR_STATUS_APPLIED,
    OFFSET_OUR_STATUS_CURED,
    OFFSET_OPP_STATUS_APPLIED,
    OFFSET_OPP_STATUS_CURED,
    OFFSET_OUR_ITEM_USED,
    OFFSET_OPP_ITEM_USED,
    STATUS_DIM,
    TURN_DELTA_SCALAR_OFFSETS,
    TURN_DELTA_EMBEDDED_IDS,
    _STATUS_TO_IDX,  # noqa: PLC2701 — test-only access to verify mapping
)
from agents.model.feature_coverage._support import (
    feature_model,
    anchor_delta,
    encode_delta,
    obs_with_delta,
    obs_zero,
    assert_delta_reaches_network,
    assert_reaches_network,
    read_block,
    onehot_idx,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Statuses that actually transition in a Gen3 battle (FNT is not a "status" in
# the transition sense — a mon doesn't get "cured" of fainting — but it IS in
# the _STATUS_ORDER vocab because target_status reads it from move-fire state).
# For applied/cured transitions we test the realistic set.
_TRANSITION_STATUSES = [
    Status.PSN,
    Status.TOX,
    Status.BRN,
    Status.PAR,
    Status.SLP,
    Status.FRZ,
]

# All statuses in the STATUS_DIM vocab (including FNT at index 1) — used to
# verify the index mapping is correct.
_ALL_STATUS_ORDER = [Status.BRN, Status.FNT, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX]


def _status_applied_delta(status: Status, side: str = "our"):
    """Delta with one side's status_applied set."""
    field = f"{side}_status_applied"
    return anchor_delta(**{field: status})


def _status_cured_delta(status: Status, side: str = "our"):
    """Delta with one side's status_cured set."""
    field = f"{side}_status_cured"
    return anchor_delta(**{field: status})


def _item_used_delta(side: str = "our"):
    """Delta with one side's item_lost set (drives the item_used bit)."""
    field = f"{side}_item_lost"
    return anchor_delta(**{field: "sitrusberry"})


# ---------------------------------------------------------------------------
# 0. Structural: status/item dims are pass-through scalars
# ---------------------------------------------------------------------------

def test_status_item_dims_are_scalars():
    """Status-transition and item-used offsets must be in TURN_DELTA_SCALAR_OFFSETS,
    not in the embedded-ID manifest — they are plain floats, not embedding IDs."""
    embedded = frozenset(pos for pos, _ in TURN_DELTA_EMBEDDED_IDS)
    scalar_set = frozenset(TURN_DELTA_SCALAR_OFFSETS)

    status_offsets = (
        list(range(OFFSET_OUR_STATUS_APPLIED, OFFSET_OUR_STATUS_APPLIED + STATUS_DIM))
        + list(range(OFFSET_OUR_STATUS_CURED,  OFFSET_OUR_STATUS_CURED  + STATUS_DIM))
        + list(range(OFFSET_OPP_STATUS_APPLIED, OFFSET_OPP_STATUS_APPLIED + STATUS_DIM))
        + list(range(OFFSET_OPP_STATUS_CURED,   OFFSET_OPP_STATUS_CURED  + STATUS_DIM))
    )
    for pos in status_offsets:
        assert pos not in embedded, f"status dim {pos} wrongly in embedded-id manifest"
        assert pos in scalar_set, f"status dim {pos} missing from scalar offsets"

    for pos in (OFFSET_OUR_ITEM_USED, OFFSET_OPP_ITEM_USED):
        assert pos not in embedded, f"item_used dim {pos} wrongly in embedded-id manifest"
        assert pos in scalar_set, f"item_used dim {pos} missing from scalar offsets"


def test_status_item_dims_in_extractor_scalar_gather():
    """Extractor's _td_scalar_idx must include all status/item-used positions."""
    model, _, _ = feature_model()
    scalar_idx = set(model.embeddings._td_scalar_idx.tolist())

    all_offsets = (
        list(range(OFFSET_OUR_STATUS_APPLIED,  OFFSET_OUR_STATUS_APPLIED  + STATUS_DIM))
        + list(range(OFFSET_OUR_STATUS_CURED,  OFFSET_OUR_STATUS_CURED   + STATUS_DIM))
        + list(range(OFFSET_OPP_STATUS_APPLIED, OFFSET_OPP_STATUS_APPLIED + STATUS_DIM))
        + list(range(OFFSET_OPP_STATUS_CURED,  OFFSET_OPP_STATUS_CURED   + STATUS_DIM))
        + [OFFSET_OUR_ITEM_USED, OFFSET_OPP_ITEM_USED]
    )
    for pos in all_offsets:
        assert pos in scalar_idx, f"dim {pos} not in extractor scalar gather"


# ---------------------------------------------------------------------------
# 1a. STATUS INDEX MAPPING — verify _STATUS_TO_IDX matches expected order
# ---------------------------------------------------------------------------

def test_status_index_mapping():
    """Verify the exact index mapping for the status one-hot vocab.

    _STATUS_ORDER = (BRN, FNT, FRZ, PAR, PSN, SLP, TOX) → indices 0–6.
    This test pins the contract so the capture tests can assert specific indices.
    """
    expected = {
        Status.BRN: 0,
        Status.FNT: 1,
        Status.FRZ: 2,
        Status.PAR: 3,
        Status.PSN: 4,
        Status.SLP: 5,
        Status.TOX: 6,
    }
    for status, idx in expected.items():
        assert _STATUS_TO_IDX[status] == idx, (
            f"{status.name}: expected index {idx}, got {_STATUS_TO_IDX[status]}"
        )
    assert STATUS_DIM == 7


# ---------------------------------------------------------------------------
# 1b. CAPTURE — status APPLIED one-hot, every transitional status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", _TRANSITION_STATUSES)
def test_our_status_applied_capture(status: Status):
    """our_status_applied=X encodes a one-hot at OFFSET_OUR_STATUS_APPLIED."""
    enc = encode_delta(_status_applied_delta(status))
    idx = onehot_idx(enc, OFFSET_OUR_STATUS_APPLIED, STATUS_DIM)
    expected_idx = _STATUS_TO_IDX[status]
    assert idx == expected_idx, (
        f"our_status_applied={status.name}: expected bit {expected_idx}, got {idx}; "
        f"block={read_block(enc, OFFSET_OUR_STATUS_APPLIED, STATUS_DIM)}"
    )
    # No bleed into cured or opp blocks
    assert onehot_idx(enc, OFFSET_OUR_STATUS_CURED, STATUS_DIM) is None
    assert onehot_idx(enc, OFFSET_OPP_STATUS_APPLIED, STATUS_DIM) is None


@pytest.mark.parametrize("status", _TRANSITION_STATUSES)
def test_opp_status_applied_capture(status: Status):
    """opp_status_applied=X encodes a one-hot at OFFSET_OPP_STATUS_APPLIED."""
    enc = encode_delta(_status_applied_delta(status, side="opp"))
    idx = onehot_idx(enc, OFFSET_OPP_STATUS_APPLIED, STATUS_DIM)
    expected_idx = _STATUS_TO_IDX[status]
    assert idx == expected_idx, (
        f"opp_status_applied={status.name}: expected bit {expected_idx}, got {idx}"
    )
    assert onehot_idx(enc, OFFSET_OPP_STATUS_CURED, STATUS_DIM) is None
    assert onehot_idx(enc, OFFSET_OUR_STATUS_APPLIED, STATUS_DIM) is None


# ---------------------------------------------------------------------------
# 1c. CAPTURE — status CURED one-hot, every transitional status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", _TRANSITION_STATUSES)
def test_our_status_cured_capture(status: Status):
    """our_status_cured=X encodes a one-hot at OFFSET_OUR_STATUS_CURED."""
    enc = encode_delta(_status_cured_delta(status))
    idx = onehot_idx(enc, OFFSET_OUR_STATUS_CURED, STATUS_DIM)
    expected_idx = _STATUS_TO_IDX[status]
    assert idx == expected_idx, (
        f"our_status_cured={status.name}: expected bit {expected_idx}, got {idx}"
    )
    assert onehot_idx(enc, OFFSET_OUR_STATUS_APPLIED, STATUS_DIM) is None


@pytest.mark.parametrize("status", _TRANSITION_STATUSES)
def test_opp_status_cured_capture(status: Status):
    """opp_status_cured=X encodes a one-hot at OFFSET_OPP_STATUS_CURED."""
    enc = encode_delta(_status_cured_delta(status, side="opp"))
    idx = onehot_idx(enc, OFFSET_OPP_STATUS_CURED, STATUS_DIM)
    expected_idx = _STATUS_TO_IDX[status]
    assert idx == expected_idx, (
        f"opp_status_cured={status.name}: expected bit {expected_idx}, got {idx}"
    )


# ---------------------------------------------------------------------------
# 2a. CAPTURE — item-used bit
# ---------------------------------------------------------------------------

def test_our_item_used_capture():
    """our_item_lost != None → OFFSET_OUR_ITEM_USED bit is 1.0."""
    enc = encode_delta(_item_used_delta("our"))
    assert enc[OFFSET_OUR_ITEM_USED] > 0.5, (
        f"our_item_used bit not set; value={enc[OFFSET_OUR_ITEM_USED]}"
    )
    assert enc[OFFSET_OPP_ITEM_USED] < 0.5, "opp_item_used leaked into our delta"


def test_opp_item_used_capture():
    """opp_item_lost != None → OFFSET_OPP_ITEM_USED bit is 1.0."""
    enc = encode_delta(_item_used_delta("opp"))
    assert enc[OFFSET_OPP_ITEM_USED] > 0.5, (
        f"opp_item_used bit not set; value={enc[OFFSET_OPP_ITEM_USED]}"
    )
    assert enc[OFFSET_OUR_ITEM_USED] < 0.5, "our_item_used leaked into opp delta"


def test_item_not_set_when_none():
    """With no item_lost, both bits are 0."""
    enc = encode_delta(anchor_delta())
    assert enc[OFFSET_OUR_ITEM_USED] < 0.5
    assert enc[OFFSET_OPP_ITEM_USED] < 0.5


# ---------------------------------------------------------------------------
# 2b. CAPTURE — applied vs cured use DIFFERENT offsets (transition direction)
# ---------------------------------------------------------------------------

def test_applied_and_cured_use_different_offsets():
    """Status APPLIED and CURED each have their own STATUS_DIM block — they must
    NOT share the same offset range."""
    applied_range = set(range(OFFSET_OUR_STATUS_APPLIED, OFFSET_OUR_STATUS_APPLIED + STATUS_DIM))
    cured_range = set(range(OFFSET_OUR_STATUS_CURED, OFFSET_OUR_STATUS_CURED + STATUS_DIM))
    assert applied_range.isdisjoint(cured_range), (
        "OUR applied and cured status blocks overlap — they would be indistinguishable"
    )
    opp_applied_range = set(range(OFFSET_OPP_STATUS_APPLIED, OFFSET_OPP_STATUS_APPLIED + STATUS_DIM))
    opp_cured_range = set(range(OFFSET_OPP_STATUS_CURED, OFFSET_OPP_STATUS_CURED + STATUS_DIM))
    assert opp_applied_range.isdisjoint(opp_cured_range)


# ---------------------------------------------------------------------------
# 3. NETWORK — status transitions reach both heads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", _TRANSITION_STATUSES)
def test_our_status_applied_reaches_network(status: Status):
    model, layout, _ = feature_model()
    assert_delta_reaches_network(
        model, layout,
        base_delta=anchor_delta(),
        variant_delta=anchor_delta(our_status_applied=status),
        msg=f"our_status_applied={status.name} vs no status",
    )


@pytest.mark.parametrize("status", _TRANSITION_STATUSES)
def test_opp_status_applied_reaches_network(status: Status):
    model, layout, _ = feature_model()
    assert_delta_reaches_network(
        model, layout,
        base_delta=anchor_delta(),
        variant_delta=anchor_delta(opp_status_applied=status),
        msg=f"opp_status_applied={status.name} vs no status",
    )


@pytest.mark.parametrize("status", _TRANSITION_STATUSES)
def test_our_status_cured_reaches_network(status: Status):
    model, layout, _ = feature_model()
    assert_delta_reaches_network(
        model, layout,
        base_delta=anchor_delta(),
        variant_delta=anchor_delta(our_status_cured=status),
        msg=f"our_status_cured={status.name} vs no status",
    )


@pytest.mark.parametrize("status", _TRANSITION_STATUSES)
def test_opp_status_cured_reaches_network(status: Status):
    model, layout, _ = feature_model()
    assert_delta_reaches_network(
        model, layout,
        base_delta=anchor_delta(),
        variant_delta=anchor_delta(opp_status_cured=status),
        msg=f"opp_status_cured={status.name} vs no status",
    )


# ---------------------------------------------------------------------------
# 4. NETWORK — item-used bit reaches both heads
# ---------------------------------------------------------------------------

def test_our_item_used_reaches_network():
    model, layout, _ = feature_model()
    assert_delta_reaches_network(
        model, layout,
        base_delta=anchor_delta(),
        variant_delta=_item_used_delta("our"),
        msg="our_item_lost=sitrusberry vs no item",
    )


def test_opp_item_used_reaches_network():
    model, layout, _ = feature_model()
    assert_delta_reaches_network(
        model, layout,
        base_delta=anchor_delta(),
        variant_delta=_item_used_delta("opp"),
        msg="opp_item_lost=sitrusberry vs no item",
    )


# ---------------------------------------------------------------------------
# 5. NETWORK — CURED is distinguishable from APPLIED (transition direction)
#    A delta with status_applied=PAR must produce different output than one
#    with status_cured=PAR — proves direction is visible to the network.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", [Status.PAR, Status.SLP, Status.BRN])
def test_status_applied_vs_cured_distinguishable(status: Status):
    """applied=X and cured=X set different offset ranges → the network must
    produce different outputs (transition direction is visible)."""
    model, layout, _ = feature_model()
    applied_delta = anchor_delta(our_status_applied=status)
    cured_delta = anchor_delta(our_status_cured=status)
    assert_delta_reaches_network(
        model, layout,
        base_delta=applied_delta,
        variant_delta=cured_delta,
        msg=f"status_applied={status.name} vs status_cured={status.name}",
    )


# ---------------------------------------------------------------------------
# 6. NETWORK — two different statuses give different outputs (status identity)
# ---------------------------------------------------------------------------

def test_different_applied_statuses_distinguishable():
    """PAR-applied and SLP-applied must produce different network outputs —
    the specific status is visible, not just "a status was applied"."""
    model, layout, _ = feature_model()
    assert_delta_reaches_network(
        model, layout,
        base_delta=anchor_delta(our_status_applied=Status.PAR),
        variant_delta=anchor_delta(our_status_applied=Status.SLP),
        msg="our_status_applied=PAR vs SLP",
    )


def test_different_cured_statuses_distinguishable():
    model, layout, _ = feature_model()
    assert_delta_reaches_network(
        model, layout,
        base_delta=anchor_delta(our_status_cured=Status.BRN),
        variant_delta=anchor_delta(our_status_cured=Status.TOX),
        msg="our_status_cured=BRN vs TOX",
    )
