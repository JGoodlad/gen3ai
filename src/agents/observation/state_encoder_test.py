import pytest
import numpy as np
from agents.observation import Gen3ObservationEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from unittest.mock import MagicMock

def test_encoder_dimension():
    encoder = Gen3ObservationEncoder()
    assert encoder.dimension == 1684

def test_encoder_output_shape():
    encoder = Gen3ObservationEncoder()
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
    assert obs.shape == (1684,)
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32

if __name__ == "__main__":
    pytest.main([__file__])
