import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from agents.inference.player import Gen3Player
from agents.observation.state_encoder import load_mappings, Gen3ObservationEncoder


class _ConcretePlayer(Gen3Player):
    def choose_move(self, battle):
        pass


def _make_player():
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    player = _ConcretePlayer.__new__(_ConcretePlayer)
    player.observation_encoder = encoder
    player.mappings = mappings
    player._stall_logger = MagicMock()
    player._last_battle_tag = None
    return player, encoder


def _make_battle(encoder):
    """Return a mock battle that satisfies embed_battle's requirements."""
    battle = MagicMock()
    battle.wait = False
    battle.battle_tag = "battle-gen3ou-test"
    battle.team = {}
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None
    battle.weather = None
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}
    battle.turn = 1
    return battle


def test_embed_battle_output_dim_matches_encoder():
    """
    Regression test: Gen3Player.embed_battle() must produce an observation whose
    length equals encoder.dimension (base + prev_mask + TurnDelta).
    Previously, TurnDelta was not appended, causing a shape mismatch at inference time.
    """
    player, encoder = _make_player()
    battle = _make_battle(encoder)

    # Stub get_observation to return a known base-dim obs + all-ones mask
    base_obs = np.zeros(encoder.base_dimension, dtype=np.float32)
    mask = np.ones(11, dtype=np.int8)
    player.observation_encoder.get_observation = MagicMock(return_value={
        "observation": base_obs,
        "action_mask": mask,
    })

    result = player.embed_battle(battle)
    assert result["observation"].shape == (encoder.dimension,), (
        f"embed_battle() produced {result['observation'].shape[0]} dims, "
        f"expected {encoder.dimension}. "
        f"Likely missing prev_mask or TurnDelta block in player.embed_battle()."
    )
    assert result["observation"].dtype == np.float32
