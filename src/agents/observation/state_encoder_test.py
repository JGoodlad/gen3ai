import pytest
import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle
from unittest.mock import MagicMock
from .state_encoder import Gen3ObservationEncoder, load_mappings

def test_encoder_dimension():
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    assert encoder.dimension == 1021

def test_encoder_output_shape():
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    # Mock battle
    battle = MagicMock(spec=AbstractBattle)
    battle.team = {}
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None
    battle.weather = None
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}
    battle.turn = 0

    obs = encoder.encode(battle)
    assert obs.shape == (1021,)
