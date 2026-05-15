import pytest
import numpy as np
from unittest.mock import MagicMock
from .abilities import AbilitiesEncoder

def test_abilities_encoder_revealed():
    mapping = {"intimidate": {"num": 1}, "levitate": {"num": 2}}
    encoder = AbilitiesEncoder(mapping)
    
    mon = MagicMock()
    mon.ability = "Intimidate"
    vec = encoder.encode(mon, None)
    assert vec[0] == 1.0
    assert vec[1] == 1.0  # known flag at ABILITY_SLOT_DIM offset (now 1)

def test_abilities_encoder_hidden():
    mapping = {"intimidate": {"num": 1}}
    encoder = AbilitiesEncoder(mapping)
    
    mon = MagicMock()
    mon.ability = None
    vec = encoder.encode(mon, None)
    assert vec[0] == 0.0
    assert vec[1] == 0.0

def test_abilities_encoder_unknown_placeholder():
    mapping = {"intimidate": {"num": 1}}
    encoder = AbilitiesEncoder(mapping)
    
    mon = MagicMock()
    mon.ability = "unknown_ability"
    vec = encoder.encode(mon, None)
    assert vec[0] == 0.0
    assert vec[1] == 0.0
