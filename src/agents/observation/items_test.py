import pytest
from .items import ItemsEncoder

ITEM_MAPPING = {"leftovers": {"num": 100}, "sitrusberry": {"num": 101}, "choiceband": {"num": 102}}

def make_encoder():
    return ItemsEncoder(ITEM_MAPPING)

# ── Basic state tests ──────────────────────────────────────────────────────────

def test_dimension():
    enc = make_encoder()
    assert enc.dimension == 3

# ── describe_vector ────────────────────────────────────────────────────────────

# ── Fuzz: state-transition sequences ──────────────────────────────────────────

@pytest.fixture
def enc():
    return ItemsEncoder(ITEM_MAPPING, reverse_mapping={100: "leftovers", 101: "sitrusberry", 102: "choiceband"})
