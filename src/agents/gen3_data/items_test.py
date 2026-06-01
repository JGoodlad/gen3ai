import pytest

from agents.gen3_data import items
from agents.gen3_data.items import ItemData


def test_known_item_fields():
    lo = items.item_data("leftovers")
    assert isinstance(lo, ItemData)
    assert lo.id == "leftovers"
    assert lo.name == "Leftovers"
    # `num` is the true item-dex number (NOT Showdown's spritenum, which is 242).
    assert lo.num == 234


def test_non_item_dex_entries_dropped():
    # Berserk Gene is a removed Gen-2 item (item-dex num 0) — not a real gen-3 item, so it is
    # excluded. Real gen-3 items are present.
    assert items.get("berserkgene") is None
    assert items.get("leftovers") is not None


def test_get_unknown_returns_none():
    assert items.get("notanitem") is None
    assert items.get(None) is None


def test_item_data_raises_on_unknown():
    with pytest.raises(KeyError):
        items.item_data("notanitem")


def test_embedding_id_collisions_are_cross_gen_aliases():
    # The item-dex `num` is intentionally shared across cross-gen aliases of the SAME item
    # (Sitrus Berry / GoldBerry, Lum Berry / MiracleBerry are gen3/gen2 names for one item). So
    # the obs embedding-id space is NOT injective — and that's correct: aliases ARE the same item
    # (the gen3 name is the one that appears in gen3ou teams). Pin a couple so a change that
    # breaks this aliasing is caught.
    assert items.item_data("sitrusberry").num == items.item_data("goldberry").num
    assert items.item_data("lumberry").num == items.item_data("miracleberry").num
    # Distinct, unrelated items still differ.
    assert items.item_data("leftovers").num != items.item_data("choiceband").num
