"""The per-GAME seed rule (``rust_eval.seeds``) — pure, schedule-free, and the Python twin of the core's
per-episode bot stream seeds (M5 Lane H)."""
from __future__ import annotations

import random

import pytest

from agents.training.rust_eval import seeds as SD


class _Builder:
    """A ``Gen3Teambuilder``-shaped stub: one draw from its ``_rng`` per yield."""

    def __init__(self, teams):
        self.teams = list(teams)
        self._rng = random.Random(0)
        self._block_episodes = 1

    def yield_team(self):
        return self._rng.choice(self.teams)


def test_a_game_key_names_its_draws_whatever_ran_before():
    b = _Builder([f"t{i}" for i in range(50)])
    k = SD.game_key(7, "heuristic", 3)
    first = SD.draw_team(b, k, SD.TRAINEE)
    for g in range(20):                          # any other draws in between
        SD.draw_team(b, SD.game_key(7, "staller", g), SD.OPPONENT)
    assert SD.draw_team(b, k, SD.TRAINEE) == first
    assert SD.battle_seed(k) == SD.battle_seed(SD.game_key(7, "heuristic", 3))
    assert all(0 <= w < 65536 for w in SD.battle_seed(k)) and any(SD.battle_seed(k))


def test_distinct_games_and_sides_draw_distinct_streams():
    keys = [SD.game_key(1, o, g) for o in ("random", "sentinel_0") for g in range(40)]
    seeds = {tuple(SD.battle_seed(k)) for k in keys}
    assert len(seeds) == len(keys)
    k = keys[0]
    assert SD.derived_seed(k, SD.TRAINEE) != SD.derived_seed(k, SD.OPPONENT)
    assert SD.sample_seed(k) != SD.sample_seed(keys[1])


def test_a_stateful_builder_is_refused():
    b = _Builder(["a", "b"])
    b._block_episodes = 3
    with pytest.raises(ValueError, match="block-episodes"):
        SD.draw_team(b, SD.game_key(0, "x", 0), SD.TRAINEE)


def test_the_bot_stream_seeds_are_the_cores_rule():
    """``opponents::episode_stream_seed`` pins these values (``a_per_episode_bot_route_reseeds_…``)."""
    from agents.training.rust_env_opponents import episode_bot_stream_seed, pack_seed_words

    assert pack_seed_words([1, 2, 3, 4]) == 1 | (2 << 16) | (3 << 32) | (4 << 48)
    assert episode_bot_stream_seed(5, [1, 2, 3, 4], 0) == 14338025463950524205
    assert episode_bot_stream_seed(9, [65535, 0, 0, 0], 1) == 12882590778465766730
    assert SD.bot_route_seed("random") == SD.BOT_ROUTE_SEED   # the roster's first bot
