import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from poke_env.player.battle_order import ForfeitBattleOrder
from agents.inference.player import Gen3Player
from agents.observation.state_encoder import load_mappings, Gen3ObservationEncoder
from agents.training.stall import StallConfig, StallLogger


class _ConcretePlayer(Gen3Player):
    def choose_move(self, battle):
        pass

    def _battle_finished_callback(self, battle):
        # Provide the poke-env no-op that Gen3Player._battle_finished_callback calls via super()
        Gen3Player._battle_finished_callback(self, battle)


def _make_player(threshold=250):
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)
    player = _ConcretePlayer.__new__(_ConcretePlayer)
    player.observation_encoder = encoder
    player.mappings = mappings
    player._stall_config = StallConfig(threshold=threshold)
    player._stall_loggers = {}
    return player, encoder


def _stall_battle(tag, turn):
    battle = MagicMock()
    battle.battle_tag = tag
    battle.turn = turn
    return battle


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


# ---------------------------------------------------------------------------
# _handle_stall — basic behaviour
# ---------------------------------------------------------------------------

def test_no_forfeit_below_threshold():
    player, _ = _make_player(threshold=10)
    assert player._handle_stall(_stall_battle("A", turn=5), "STALL") is None


def test_forfeit_at_threshold():
    player, _ = _make_player(threshold=10)
    result = player._handle_stall(_stall_battle("A", turn=10), "STALL")
    assert isinstance(result, ForfeitBattleOrder)


def test_forfeit_above_threshold():
    player, _ = _make_player(threshold=10)
    result = player._handle_stall(_stall_battle("A", turn=999), "STALL")
    assert isinstance(result, ForfeitBattleOrder)


# ---------------------------------------------------------------------------
# _handle_stall — concurrent-battle isolation
# ---------------------------------------------------------------------------

def test_separate_stall_logger_per_battle():
    player, _ = _make_player(threshold=10)
    player._handle_stall(_stall_battle("A", turn=1), "STALL")
    player._handle_stall(_stall_battle("B", turn=1), "STALL")

    assert "A" in player._stall_loggers
    assert "B" in player._stall_loggers
    assert player._stall_loggers["A"] is not player._stall_loggers["B"]


def test_battle_a_stall_does_not_silence_battle_b():
    """A reaching threshold must not prevent B from getting its own independent logger."""
    player, _ = _make_player(threshold=5)

    # A stalls (ForfeitBattleOrder returned)
    result_a = player._handle_stall(_stall_battle("A", turn=10), "STALL")
    assert isinstance(result_a, ForfeitBattleOrder)

    # B also stalls: must get its own logger, not share A's
    result_b = player._handle_stall(_stall_battle("B", turn=10), "STALL")
    assert isinstance(result_b, ForfeitBattleOrder)

    assert "A" in player._stall_loggers
    assert "B" in player._stall_loggers
    assert player._stall_loggers["A"] is not player._stall_loggers["B"]


def test_battle_b_under_threshold_does_not_forfeit_battle_a():
    """Evaluating B (healthy) between A's checks must not affect A's outcome."""
    player, _ = _make_player(threshold=10)

    player._handle_stall(_stall_battle("A", turn=1), "STALL")   # A: fine
    player._handle_stall(_stall_battle("B", turn=1), "STALL")   # B: fine
    player._handle_stall(_stall_battle("A", turn=3), "STALL")   # A: still fine
    result_b = player._handle_stall(_stall_battle("B", turn=1), "STALL")  # B: still fine
    result_a = player._handle_stall(_stall_battle("A", turn=20), "STALL") # A: stalled

    assert result_b is None
    assert isinstance(result_a, ForfeitBattleOrder)


def test_interleaved_battles_independent_stall_decisions():
    """A stalling must not forfeit B; B healthy must not un-forfeit A."""
    player, _ = _make_player(threshold=10)

    # Interleave: A at turn 15 (stalled), B at turn 2 (healthy)
    result_a = player._handle_stall(_stall_battle("A", turn=15), "STALL")
    result_b = player._handle_stall(_stall_battle("B", turn=2), "STALL")

    assert isinstance(result_a, ForfeitBattleOrder)
    assert result_b is None


# ---------------------------------------------------------------------------
# _battle_finished_callback — stall logger cleanup
# ---------------------------------------------------------------------------

def test_battle_finished_removes_its_stall_logger():
    player, _ = _make_player(threshold=10)
    player._handle_stall(_stall_battle("A", turn=1), "STALL")
    assert "A" in player._stall_loggers

    player._battle_finished_callback(_stall_battle("A", turn=1))
    assert "A" not in player._stall_loggers


def test_battle_finished_does_not_remove_other_battles_logger():
    """Finishing A must leave B's stall logger intact."""
    player, _ = _make_player(threshold=10)
    player._handle_stall(_stall_battle("A", turn=1), "STALL")
    player._handle_stall(_stall_battle("B", turn=1), "STALL")

    player._battle_finished_callback(_stall_battle("A", turn=1))

    assert "A" not in player._stall_loggers
    assert "B" in player._stall_loggers


def test_stall_state_not_reset_when_different_battle_fires():
    """Old bug: switching battle tags reset the stall logger. Must not happen now."""
    player, _ = _make_player(threshold=10)

    # A makes progress toward threshold
    player._handle_stall(_stall_battle("A", turn=8), "STALL")
    # B fires (simulating interleaved concurrent battle)
    player._handle_stall(_stall_battle("B", turn=1), "STALL")
    # A's logger must still exist with its own state — not reset by B's call
    assert "A" in player._stall_loggers
    assert player._stall_loggers["A"] is not player._stall_loggers["B"]
