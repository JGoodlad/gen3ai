import pytest
import numpy as np
from unittest.mock import MagicMock
from .global_env import GlobalEnvEncoder
from poke_env.battle.weather import Weather
from poke_env.battle.side_condition import SideCondition

def test_global_env_weather():
    encoder = GlobalEnvEncoder()
    battle = MagicMock()
    battle.weather = {Weather.SUNNYDAY: 1}
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}
    battle.turn = 1
    
    vec = encoder.encode(battle)
    # Weather.SUNNYDAY is index 1
    assert vec[1] == 1.0
    assert vec[0] == 0.0 # NONE is index 0

def test_global_env_hazards():
    encoder = GlobalEnvEncoder()
    battle = MagicMock()
    battle.weather = {}
    battle.side_conditions = {SideCondition.SPIKES: 2}
    battle.opponent_side_conditions = {SideCondition.SPIKES: 3}
    battle.turn = 1
    
    vec = encoder.encode(battle)
    # MAX_SPIKES is 3
    assert vec[6] == pytest.approx(2/3)
    assert vec[7] == 1.0

def test_global_env_clock():
    encoder = GlobalEnvEncoder()
    battle = MagicMock()
    battle.weather = {}
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}
    battle.turn = 10

    vec = encoder.encode(battle)
    import math
    from .constants import MAX_TURNS
    expected = math.log(1 + 10) / math.log(1 + MAX_TURNS)
    assert vec[8] == pytest.approx(expected)

def test_global_env_screens():
    encoder = GlobalEnvEncoder()
    battle = MagicMock()
    battle.weather = {}
    battle.turn = 1
    battle.side_conditions = {SideCondition.REFLECT: 1}
    battle.opponent_side_conditions = {SideCondition.LIGHT_SCREEN: 1}

    vec = encoder.encode(battle)
    assert vec[9] == 1.0   # our reflect
    assert vec[10] == 0.0  # our light screen
    assert vec[11] == 0.0  # opp reflect
    assert vec[12] == 1.0  # opp light screen

def test_global_env_screens_none():
    encoder = GlobalEnvEncoder()
    battle = MagicMock()
    battle.weather = {}
    battle.turn = 1
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}

    vec = encoder.encode(battle)
    assert vec[9] == 0.0
    assert vec[10] == 0.0
    assert vec[11] == 0.0
    assert vec[12] == 0.0
