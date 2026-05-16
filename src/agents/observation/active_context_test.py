import pytest
import numpy as np
from unittest.mock import MagicMock
from .active_context import ActiveContextEncoder
from poke_env.battle.effect import Effect

def test_active_context_boosts():
    encoder = ActiveContextEncoder()
    mon = MagicMock()
    # Atk +2, Def -1
    mon.boosts = {"atk": 2, "def": -1}
    mon.volatiles = {}
    
    vec = encoder.encode(mon, None)
    # [pos, neg] for each stat
    assert vec[0] == pytest.approx(2/6) # Atk pos
    assert vec[1] == 0.0             # Atk neg
    assert vec[2] == 0.0             # Def pos
    assert vec[3] == pytest.approx(1/6) # Def neg

def test_active_context_volatiles():
    encoder = ActiveContextEncoder()
    mon = MagicMock()
    mon.boosts = {}
    mon.volatiles = {Effect.SUBSTITUTE: 1, Effect.TAUNT: 1}

    vec = encoder.encode(mon, None)
    # Volatiles start at index 14
    # SUB is index 1, TAUNT is index 2
    assert vec[14 + 1] == 1.0
    assert vec[14 + 2] == 1.0
    assert vec[14 + 0] == 0.0 # CONFUSION


def test_active_context_destiny_bond():
    encoder = ActiveContextEncoder()
    mon = MagicMock()
    mon.boosts = {}
    mon.volatiles = {Effect.DESTINY_BOND: 1}

    vec = encoder.encode(mon, None)
    # DBOND is index 8 in the volatile block (offset 14)
    assert vec[14 + 8] == 1.0
    # No other volatiles should fire
    for i in range(8):
        assert vec[14 + i] == 0.0


def test_active_context_destiny_bond_absent():
    encoder = ActiveContextEncoder()
    mon = MagicMock()
    mon.boosts = {}
    mon.volatiles = {Effect.SUBSTITUTE: 1}

    vec = encoder.encode(mon, None)
    assert vec[14 + 8] == 0.0  # DBOND bit is clear


def test_active_context_describe_vector_destiny_bond():
    encoder = ActiveContextEncoder()
    vec = np.zeros(encoder.dimension, dtype=np.float32)
    vec[14 + 8] = 1.0  # DBOND bit

    desc = encoder.describe_vector(vec)
    assert "DBOND" in desc["volatiles"]
