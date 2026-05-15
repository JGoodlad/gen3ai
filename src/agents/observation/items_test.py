import pytest
import numpy as np
from unittest.mock import MagicMock
from .items import ItemsEncoder

def test_items_encoder_revealed():
    mapping = {"leftovers": {"num": 100}, "choiceband": {"num": 101}}
    encoder = ItemsEncoder(mapping)

    mon = MagicMock()
    mon.item = "Leftovers"
    vec = encoder.encode(mon, None)
    assert vec[0] == 100.0
    assert vec[1] == 1.0  # known flag at ITEM_ID_DIM offset (now 1)

def test_items_encoder_hidden():
    mapping = {"leftovers": {"num": 100}}
    encoder = ItemsEncoder(mapping)

    mon = MagicMock()
    mon.item = None
    vec = encoder.encode(mon, None)
    assert vec[0] == 0.0
    assert vec[1] == 0.0

def test_items_encoder_unknown_placeholder():
    mapping = {"leftovers": {"num": 100}}
    encoder = ItemsEncoder(mapping)

    mon = MagicMock()
    mon.item = "unknown_item"
    vec = encoder.encode(mon, None)
    assert vec[0] == 0.0
    assert vec[1] == 0.0
