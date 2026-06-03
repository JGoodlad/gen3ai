"""Unit tests for MaskableAgentWrapper's live per-episode opponent selection.

Tests `_select_episode_opponent` / `set_self_play_target` / `opponent_default_stats` directly
with stub env + pool + players (no Showdown server, no full reset)."""

from unittest.mock import MagicMock

import pytest

from agents.training.wrappers import MaskableAgentWrapper


def _stub_env():
    env = MagicMock()
    env.agent1.username = "a1"
    env.observation_spaces = {"a1": MagicMock()}
    env.action_spaces = {"a1": MagicMock()}
    return env


def _make_wrapper(*, fraction=0.0, pool=None, pool_player=None, n_heuristics=2, rng_seed=0):
    heuristics = [MagicMock(name=f"heur{i}") for i in range(n_heuristics)]
    w = MaskableAgentWrapper(
        _stub_env(), heuristic_opponents=heuristics, pool=pool, pool_player=pool_player,
        self_play_fraction=fraction, rng_seed=rng_seed,
    )
    return w, heuristics


def _stub_pool(empty=False, model="MODEL_X"):
    pool = MagicMock()
    pool.is_empty.return_value = empty
    pool.sample.return_value = MagicMock(name="entry")
    pool.load_model.return_value = model
    return pool


# ── back-compat + construction ────────────────────────────────────────────────

def test_legacy_single_opponent_form():
    opp = MagicMock(name="opp")
    w = MaskableAgentWrapper(_stub_env(), opp)
    assert w._heuristic_opponents == [opp]
    w._select_episode_opponent()
    assert w.opponent is opp          # no pool → always the single opponent


def test_requires_an_opponent_or_roster():
    with pytest.raises(ValueError):
        MaskableAgentWrapper(_stub_env())


# ── fraction-driven selection ───────────────────────────────────────────────

def test_fraction_zero_always_heuristic():
    pool = _stub_pool()
    w, heuristics = _make_wrapper(fraction=0.0, pool=pool, pool_player=MagicMock())
    for _ in range(50):
        w._select_episode_opponent()
        assert w.opponent in heuristics
    pool.sample.assert_not_called()   # never even consults the pool


def test_fraction_one_uses_pool_and_swaps_model():
    pool = _stub_pool(model="SAMPLED_MODEL")
    pp = MagicMock(name="pool_player")
    w, _ = _make_wrapper(fraction=1.0, pool=pool, pool_player=pp)
    w._select_episode_opponent()
    assert w.opponent is pp
    assert pp.model == "SAMPLED_MODEL"   # swapped in the sampled snapshot's model


def test_fraction_one_but_empty_pool_falls_back_to_heuristic():
    pool = _stub_pool(empty=True)
    w, heuristics = _make_wrapper(fraction=1.0, pool=pool, pool_player=MagicMock())
    w._select_episode_opponent()
    assert w.opponent in heuristics
    pool._scan.assert_called()          # tried a re-scan to discover a (not-yet-written) seed


def test_no_pool_player_always_heuristic():
    w, heuristics = _make_wrapper(fraction=1.0, pool=_stub_pool(), pool_player=None)
    w._select_episode_opponent()
    assert w.opponent in heuristics


# ── live update + generation re-scan ─────────────────────────────────────────

def test_set_self_play_target_updates_fraction():
    w, heuristics = _make_wrapper(fraction=0.0, pool=_stub_pool(), pool_player=MagicMock())
    w.set_self_play_target(1.0, generation=1)
    assert w._self_play_fraction == 1.0
    w._select_episode_opponent()
    assert w.opponent is w._pool_player   # now uses the pool


def test_generation_bump_triggers_pool_rescan():
    pool = _stub_pool()
    w, _ = _make_wrapper(fraction=1.0, pool=pool, pool_player=MagicMock())
    w._select_episode_opponent()          # first pool use → initial scan (gen -1 → 0)
    first = pool._scan.call_count
    w._select_episode_opponent()          # same generation → no re-scan
    assert pool._scan.call_count == first
    w.set_self_play_target(1.0, generation=5)
    w._select_episode_opponent()          # new generation → re-scan
    assert pool._scan.call_count == first + 1


def test_pool_model_loaded_once_per_generation():
    """FPS-regression guard: the ~27MB snapshot is (re)loaded once per generation, NOT every
    episode. Per-episode load_model thrashed the workers (blocked in reset() on a 27MB
    deserialize → CPU ~40%, FPS ~1400→~500). Within a generation the opponent must be reused."""
    pool = _stub_pool(model="M")
    pp = MagicMock(name="pool_player")
    w, _ = _make_wrapper(fraction=1.0, pool=pool, pool_player=pp)
    for _ in range(50):
        w._select_episode_opponent()
        assert w.opponent is pp           # always the pool, but...
    pool.load_model.assert_called_once()  # ...one load across 50 same-generation episodes
    # A new generation re-samples + loads exactly once more (not per episode).
    w.set_self_play_target(1.0, generation=7)
    for _ in range(50):
        w._select_episode_opponent()
    assert pool.load_model.call_count == 2


# ── telemetry reads the persistent pool player ───────────────────────────────

def test_opponent_default_stats_reads_pool_player():
    pp = MagicMock(_n_decisions=100, _n_defaults=5, _n_redecides=2)
    w, _ = _make_wrapper(fraction=1.0, pool=_stub_pool(), pool_player=pp)
    # Even with a heuristic currently selected, stats come from the persistent pool player.
    w.opponent = w._heuristic_opponents[0]
    assert w.opponent_default_stats() == (100, 5, 2)


def test_opponent_default_stats_zero_without_pool_player():
    w, _ = _make_wrapper(fraction=0.0, pool=None, pool_player=None)
    assert w.opponent_default_stats() == (0, 0, 0)
