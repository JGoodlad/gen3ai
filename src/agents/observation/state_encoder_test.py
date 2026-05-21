import pytest
import numpy as np
import torch
import gymnasium as gym
from poke_env.battle.abstract_battle import AbstractBattle
from unittest.mock import MagicMock
from .state_encoder import Gen3ObservationEncoder, load_mappings

EXPECTED_BASE_DIM = 1103  # 6*62 teams(×2) + 46 active_ctx + 13 global + 300 reactive
EXPECTED_OBS_DIM = 1309  # base + 11-dim prev_mask + 5 * 39-dim TurnDelta history


def test_encoder_dimension():
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    assert encoder.base_dimension == EXPECTED_BASE_DIM
    assert encoder.dimension == EXPECTED_OBS_DIM


def test_encoder_output_shape():
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
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
    assert obs.shape == (EXPECTED_BASE_DIM,)  # encode() returns base dims; prev_mask appended by embed_battle


def test_encoder_and_features_extractor_are_compatible():
    """
    Verifies the encoder dimension and features extractor architecture agree.
    Catches obs-space/architecture mismatches before they surface at checkpoint load.
    """
    from agents.model.features_extractor import Gen3FeaturesExtractor

    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    layout = encoder.get_layout()

    obs_space = gym.spaces.Dict({
        "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(encoder.dimension,), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.int8),
    })

    model = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings)
    model.eval()

    dummy_obs = {
        "observation": torch.zeros(1, encoder.dimension),
        "action_mask": torch.ones(1, 11, dtype=torch.int8),
    }
    with torch.no_grad():
        out = model(dummy_obs)

    assert out.shape == (1, model.features_dim)
