import pytest
import numpy as np
from unittest.mock import MagicMock
from .items import ItemsEncoder
from .constants import ITEM_ID_DIM, ITEM_KNOWN_DIM

ITEM_MAPPING = {"leftovers": {"num": 100}, "sitrusberry": {"num": 101}, "choiceband": {"num": 102}}

def make_encoder():
    return ItemsEncoder(ITEM_MAPPING)

def make_mon(item, consumed_item=None):
    mon = MagicMock()
    mon.item = item
    mon.consumed_item = consumed_item
    return mon

ID = 0
KNOWN = ITEM_ID_DIM           # 1
CONSUMED = ITEM_ID_DIM + ITEM_KNOWN_DIM  # 2


# ── Basic state tests ──────────────────────────────────────────────────────────

def test_held_item_known():
    enc = make_encoder()
    vec = enc.encode(make_mon("Leftovers"), None)
    assert vec[ID] == 100.0
    assert vec[KNOWN] == 1.0
    assert vec[CONSUMED] == 0.0

def test_no_item():
    enc = make_encoder()
    vec = enc.encode(make_mon(None), None)
    assert vec[ID] == 0.0
    assert vec[KNOWN] == 0.0
    assert vec[CONSUMED] == 0.0

def test_unknown_item_placeholder():
    enc = make_encoder()
    vec = enc.encode(make_mon("unknown_item"), None)
    assert vec[ID] == 0.0
    assert vec[KNOWN] == 0.0
    assert vec[CONSUMED] == 0.0

def test_consumed_known_item():
    enc = make_encoder()
    # item=None (consumed), consumed_item set to what was there
    vec = enc.encode(make_mon(None, consumed_item="sitrusberry"), None)
    assert vec[ID] == 101.0
    assert vec[KNOWN] == 1.0
    assert vec[CONSUMED] == 1.0

def test_consumed_unmappable_item_is_clean_unknown():
    # Item consumed but its id isn't in our mapping → encode clean all-zeros (unknown), NOT the old
    # phantom "[id=0, known=1, consumed=1]" mislabel (a "consumed the NONE item" GIGO). Gating the
    # known/consumed bits on a successful map is what removes it.
    enc = make_encoder()
    vec = enc.encode(make_mon(None, consumed_item="mysteryberry"), None)
    assert vec[ID] == 0.0
    assert vec[KNOWN] == 0.0
    assert vec[CONSUMED] == 0.0

def test_consumed_name_form_normalizes_via_to_id_str():
    # consumed_item can arrive name-form (apostrophes/hyphens: King's Rock, Never-Melt Ice); to_id_str
    # canonicalises to the id-form mapping key, so it is captured — the old manual space/underscore
    # strip left the apostrophe in ("king's rock" ∉ mapping) and silently mislabeled it.
    enc = ItemsEncoder({"kingsrock": {"num": 109}})
    vec = enc.encode(make_mon(None, consumed_item="King's Rock"), None)
    assert vec[ID] == 109.0
    assert vec[KNOWN] == 1.0
    assert vec[CONSUMED] == 1.0

def test_mon_none():
    enc = make_encoder()
    vec = enc.encode(None, None)
    assert all(v == 0.0 for v in vec)
    assert len(vec) == 3

def test_dimension():
    enc = make_encoder()
    assert enc.dimension == 3

def test_held_item_overrides_consumed():
    # After Recycle: mon.item is set back, consumed_item should be cleared upstream,
    # but even if not, a held item should show as held (not consumed).
    enc = make_encoder()
    vec = enc.encode(make_mon("Leftovers", consumed_item="sitrusberry"), None)
    assert vec[KNOWN] == 1.0
    assert vec[CONSUMED] == 0.0
    assert vec[ID] == 100.0  # held item ID, not consumed item ID


# ── describe_vector ────────────────────────────────────────────────────────────

def test_describe_held():
    enc = ItemsEncoder(ITEM_MAPPING, reverse_mapping={100: "leftovers"})
    vec = enc.encode(make_mon("Leftovers"), None)
    assert enc.describe_vector(vec) == "LEFTOVERS"

def test_describe_consumed():
    enc = ItemsEncoder(ITEM_MAPPING, reverse_mapping={101: "sitrusberry"})
    vec = enc.encode(make_mon(None, consumed_item="sitrusberry"), None)
    assert enc.describe_vector(vec) == "SITRUSBERRY(CONSUMED)"

def test_describe_unknown():
    enc = make_encoder()
    vec = enc.encode(make_mon("unknown_item"), None)
    assert enc.describe_vector(vec) == "ITM-UNKN"


# ── Fuzz: state-transition sequences ──────────────────────────────────────────

class FuzzMon:
    """Minimal Pokemon-like object that tracks item state transitions."""
    def __init__(self):
        self.item = "unknown_item"   # starts as poke-env placeholder
        self.consumed_item = None

    def reveal(self, item_name: str):
        """Showdown sends -item: we now know what the mon holds."""
        self.item = item_name
        self.consumed_item = None

    def consume(self, item_name: str):
        """Showdown sends -enditem: item was used/knocked off."""
        self.consumed_item = item_name
        self.item = None

    def restore(self, item_name: str):
        """Showdown sends -item after Recycle or Trick: item is back."""
        self.item = item_name
        self.consumed_item = None


@pytest.fixture
def enc():
    return ItemsEncoder(ITEM_MAPPING, reverse_mapping={100: "leftovers", 101: "sitrusberry", 102: "choiceband"})


def assert_unknown(vec):
    assert vec[KNOWN] == 0.0 and vec[CONSUMED] == 0.0, f"expected unknown, got {vec}"

def assert_held(vec, item_id):
    assert vec[ID] == item_id, f"expected id={item_id}, got {vec[ID]}"
    assert vec[KNOWN] == 1.0 and vec[CONSUMED] == 0.0, f"expected held, got {vec}"

def assert_consumed(vec, item_id):
    assert vec[ID] == item_id, f"expected id={item_id}, got {vec[ID]}"
    assert vec[KNOWN] == 1.0 and vec[CONSUMED] == 1.0, f"expected consumed, got {vec}"


def test_fuzz_unrevealed_stays_unknown(enc):
    mon = FuzzMon()
    assert_unknown(enc.encode(mon, None))


def test_fuzz_reveal_then_held(enc):
    mon = FuzzMon()
    mon.reveal("leftovers")
    assert_held(enc.encode(mon, None), 100.0)


def test_fuzz_reveal_then_consume(enc):
    mon = FuzzMon()
    mon.reveal("sitrusberry")
    assert_held(enc.encode(mon, None), 101.0)
    mon.consume("sitrusberry")
    assert_consumed(enc.encode(mon, None), 101.0)


def test_fuzz_consume_without_prior_reveal(enc):
    # Berry activates on the first turn we see the opponent — item was never
    # explicitly revealed before the -enditem message.
    mon = FuzzMon()
    mon.consume("sitrusberry")
    assert_consumed(enc.encode(mon, None), 101.0)


def test_fuzz_consume_then_restore_via_recycle(enc):
    mon = FuzzMon()
    mon.reveal("sitrusberry")
    mon.consume("sitrusberry")
    assert_consumed(enc.encode(mon, None), 101.0)
    mon.restore("sitrusberry")
    assert_held(enc.encode(mon, None), 101.0)


def test_fuzz_trick_gives_new_item(enc):
    # Trick: we gave our Choice Band to the opponent, they now hold it.
    mon = FuzzMon()
    mon.reveal("choiceband")
    mon.consume("choiceband")   # their original item leaves
    assert_consumed(enc.encode(mon, None), 102.0)
    mon.restore("choiceband")   # new item arrives via Trick
    assert_held(enc.encode(mon, None), 102.0)


def test_fuzz_no_item_throughout(enc):
    # Opponent was never holding anything (e.g. in-game NPC).
    mon = FuzzMon()
    mon.item = None
    assert_unknown(enc.encode(mon, None))


def test_fuzz_repeated_consume_idempotent(enc):
    # Shouldn't happen in a real battle, but encoding should be stable.
    mon = FuzzMon()
    mon.reveal("leftovers")
    mon.consume("leftovers")
    first = enc.encode(mon, None).copy()
    # Encoding again without state change must be identical.
    second = enc.encode(mon, None)
    np.testing.assert_array_equal(first, second)


def test_fuzz_item_id_zero_for_unrecognized_consumed(enc):
    mon = FuzzMon()
    mon.consume("oddberry_not_in_mapping")
    vec = enc.encode(mon, None)
    # Unmappable consumed item → clean all-zeros (unknown), not a phantom "known NONE consumed".
    assert vec[ID] == 0.0
    assert vec[KNOWN] == 0.0
    assert vec[CONSUMED] == 0.0


@pytest.mark.parametrize("item_name,expected_id", [
    ("leftovers", 100.0),
    ("sitrusberry", 101.0),
    ("choiceband", 102.0),
])
def test_fuzz_all_known_items_roundtrip(enc, item_name, expected_id):
    mon = FuzzMon()
    mon.reveal(item_name)
    assert_held(enc.encode(mon, None), expected_id)
    mon.consume(item_name)
    assert_consumed(enc.encode(mon, None), expected_id)
